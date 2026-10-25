"""Synthesizer baselines for the DM ViT comparison (Tay et al., ICML 2021).

Reference: "Synthesizer: Rethinking Self-Attention in Transformer Models",
Y. Tay, D. Bahri, D. Metzler, D.-C. Juan, Z. Zhao, C. Zheng, ICML 2021,
PMLR v139 -- https://proceedings.mlr.press/v139/tay21a.html

The paper replaces the query-key dot product with a synthesized alignment
matrix S(X) and computes  Y = softmax(S(X)) G(X)  where G(X) is a value
projection.  In the original definitions (paper Sec. 3.1, Table 1) *every*
variant applies the softmax to the synthesized matrix:

    random           : S = R, random-initialized per-head N x N, trainable
                       (paper Eq. 4; the "Fixed" ablation is random_fixed)
    dense            : S = F1 sigma(F2(X_i)), input-conditioned: a per-head
                       2-layer ReLU feed-forward mapping d -> d -> N, where
                       d = dim_head (the per-head token width, not the full
                       model width), i.e. each token predicts its row of the
                       alignment (Eq. 1-3, Table 1)
    factorized_random: S = R1 R2^T with R1, R2 in R^{N x k}, k << N (Eq. 7)
    mix              : S = alpha1 (R1 R2^T) + alpha2 F(X), additive mixture of
                       factorized-random and dense (Eq. 6, paper's example)
    static           : S = R, trainable, NO softmax -- an ablation kept for
                       completeness (it isolates the softmax/constraint step);
                       it is NOT one of the published Synthesizer variants.
    random_fixed     : R random and frozen (paper "Fixed").

Relationship to DM attention (verified against the paper):
    The closest sibling of DM's W_S is the *Random (trainable)* variant --
    a global, sample-independent alignment matrix.  Three differences:
      (1) constraint mechanism: Synthesizer hard-reparameterizes with softmax
          (always on the simplex); DM (published) keeps the raw matrix and
          pushes it toward the simplex softly with L_reg, while the legacy
          ScoreViT variant (which produced the published numbers) used the
          same softmax reparameterization -- hence it is closest of all;
      (2) initialization: 1/N uniform (= average pooling) vs random;
      (3) budget: DM is a single shared N x N with a full-width value
          projection and no output projection, Synthesizer allocates N^2 *per
          head* plus output projection.  The paper notes multiple heads are
          "imperative" for its synthesized variants, so DM's single matrix is
          itself a point to ablate.

Faithfulness notes
------------------
* The value projection and the output projection are kept, as in the paper
  and in vanilla self-attention; only Q/K (the dot product) are replaced.
* Softmax is applied over the last (key) dimension, exactly as Eq. 3/4/6/7.
* The backbone (patch embedding, cls token, block layout, classification
  head) is identical to ``DualMappingViT`` so that a comparison of two
  models isolates the attention module.

This module depends on PyTorch only (no einops / timm).
"""

import math

import torch
from torch import nn

VARIANTS = ("random", "random_fixed", "dense", "factorized_random", "mix", "static")


def _pair(t):
    return t if isinstance(t, tuple) else (t, t)


def _patchify(img, patch):
    """(B, C, H, W) -> (B, N, patch * patch * C); H, W must be divisible."""
    b, c, hh, ww = img.shape
    x = img.reshape(b, c, hh // patch, patch, ww // patch, patch)
    x = x.permute(0, 2, 4, 3, 5, 1)
    return x.reshape(b, (hh // patch) * (ww // patch), patch * patch * c)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SynthesizerAttention(nn.Module):
    """Synthetic attention: Y = softmax(S(X)) V (see module docstring).

    Args:
        dim       : token width.
        num_tokens: sequence length N (num_patches + 1 with a cls token).
        heads     : number of attention heads (research config: 16).
        dim_head  : per-head width (research config: 64).
        variant   : one of VARIANTS.
        k         : factorization rank for factorized_random / mix (paper: 8).
        dropout   : dropout on the attention output.
    """

    def __init__(self, dim, num_tokens, heads=16, dim_head=64, variant="random",
                 k=8, dropout=0.):
        super().__init__()
        assert variant in VARIANTS, variant
        inner = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.variant = variant
        n = num_tokens
        scale = n ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_v = nn.Linear(dim, inner, bias=False)   # G(X), paper's value fn
        self.to_out = nn.Sequential(
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )

        if variant in ("random", "random_fixed", "static"):
            r = torch.randn(heads, n, n) * scale
            if variant == "random_fixed":
                self.register_buffer("R", r)
            else:  # 'random' trainable; 'static' = same but no softmax
                self.R = nn.Parameter(r)
        if variant in ("factorized_random", "mix"):
            self.R1 = nn.Parameter(torch.randn(heads, n, k) * k ** -0.5)
            self.R2 = nn.Parameter(torch.randn(heads, n, k) * k ** -0.5)
        if variant in ("dense", "mix"):
            # per-head F: d -> d -> N with d = dim_head (paper Eqs. 1-2 and
            # Table 1's |theta| = d^2 + dN).  The per-head weights are stored
            # as stacked tensors so the two GEMMs are batched over heads
            # instead of launching 2 * heads small Linear calls per block.
            # When the token width is not already heads * dim_head (the
            # segmentation config: 768 vs 16 * 64), a bias-free projection
            # supplies the per-head input; otherwise tokens are split directly.
            self.head_proj = (nn.Linear(dim, heads * dim_head, bias=False)
                              if dim != heads * dim_head else nn.Identity())
            self.w1 = nn.Parameter(torch.empty(heads, dim_head, dim_head))
            self.b1 = nn.Parameter(torch.empty(heads, dim_head))
            self.w2 = nn.Parameter(torch.empty(heads, n, dim_head))
            self.b2 = nn.Parameter(torch.empty(heads, n))
            for w, b in ((self.w1, self.b1), (self.w2, self.b2)):
                nn.init.kaiming_uniform_(w, a=math.sqrt(5))
                bound = 1 / math.sqrt(w.shape[-1])
                nn.init.uniform_(b, -bound, bound)
        if variant == "mix":
            # per-head learnable mixing weights (paper Eq. 6)
            self.alpha1 = nn.Parameter(torch.zeros(heads))
            self.alpha2 = nn.Parameter(torch.zeros(heads))

    def _dense_logits(self, x):
        """Per-head F applied to the head slices: (B, heads, N, N) logits."""
        x = self.head_proj(x)
        b, n, _ = x.shape
        xh = x.reshape(b, n, self.heads, self.dim_head).permute(0, 2, 1, 3)
        h = torch.matmul(xh, self.w1) + self.b1.view(1, self.heads, 1, self.dim_head)
        h = torch.relu(h)
        return torch.matmul(h, self.w2.transpose(1, 2)) + self.b2.view(1, self.heads, 1, n)

    def _alignment(self, x):
        """S(X) for x of shape (B, N, dim); returns (B, heads, N, N) or (heads, N, N)."""
        if self.variant == "static":               # ablation: no softmax
            return self.R
        if self.variant == "random":
            return torch.softmax(self.R, dim=-1)
        if self.variant == "random_fixed":
            return torch.softmax(self.R, dim=-1)
        if self.variant == "factorized_random":
            s = torch.matmul(self.R1, self.R2.transpose(-1, -2))
            return torch.softmax(s, dim=-1)
        if self.variant == "dense":
            return torch.softmax(self._dense_logits(x), dim=-1)
        if self.variant == "mix":
            fac = torch.matmul(self.R1, self.R2.transpose(-1, -2))
            fx = self._dense_logits(x)
            a1 = self.alpha1.view(1, self.heads, 1, 1)
            a2 = self.alpha2.view(1, self.heads, 1, 1)
            return torch.softmax(a1 * fac + a2 * fx, dim=-1)
        raise AssertionError(self.variant)

    def forward(self, x):
        x = self.norm(x)
        a = self._alignment(x)                                    # (h|B, h, N, N)
        v = self.to_v(x)                                          # (B, N, inner)
        b, n, _ = v.shape
        v = v.reshape(b, n, self.heads, self.dim_head).permute(0, 2, 1, 3)  # (B, h, N, d)
        y = torch.matmul(a, v)                                    # (B, h, N, d)
        y = y.permute(0, 2, 1, 3).reshape(b, n, self.heads * self.dim_head)
        return self.to_out(y)


class SynthesizerTransformer(nn.Module):
    """Stack of pre-norm blocks; each block = SynthesizerAttention + FF."""

    def __init__(self, dim, num_tokens, depth, mlp_dim, heads=16, dim_head=64,
                 variant="random", k=8, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                SynthesizerAttention(dim, num_tokens, heads=heads, dim_head=dim_head,
                                     variant=variant, k=k, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class SynthesizerViT(nn.Module):
    """Synthesizer (ICML 2021) with the DM ViT backbone for controlled comparison.

    Args:
        image_size : square input resolution.
        patch_size : square patch size.
        num_classes: output classes.
        dim        : token embedding width (research config: 1024).
        depth      : number of transformer blocks.
        mlp_dim    : feed-forward hidden width.
        heads      : number of synthetic-attention heads (research config: 16).
        dim_head   : per-head width (research config: 64).
        variant    : one of VARIANTS ('random' is the paper-faithful default).
        k          : factorization rank (paper: 8).
        pool       : 'cls' or 'mean' token pooling.
        channels   : input image channels.
    """

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, mlp_dim,
                 heads=16, dim_head=64, variant="random", k=8, pool='cls', channels=3,
                 dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = _pair(image_size)
        patch_height, patch_width = _pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'
        assert patch_height == patch_width, 'only square patches are supported'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls or mean'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.patch_height = patch_height
        self.pool = pool
        self.variant = variant
        self.num_tokens = num_patches + 1

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = SynthesizerTransformer(dim, self.num_tokens, depth, mlp_dim,
                                                  heads=heads, dim_head=dim_head,
                                                  variant=variant, k=k, dropout=dropout)
        # bare Linear (same as DualMappingViT and the ViT baseline)
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(_patchify(img, self.patch_height))
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :n + 1]
        x = self.dropout(x)

        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        return self.mlp_head(x)


if __name__ == '__main__':
    for variant in VARIANTS:
        model = SynthesizerViT(image_size=224, patch_size=16, num_classes=3,
                               dim=64, depth=2, mlp_dim=128, variant=variant)
        y = model(torch.randn(2, 3, 224, 224))
        n_params = sum(p.numel() for p in model.parameters())
        print(f'{variant:>17}: forward ok, {tuple(y.shape)}, '
              f'{n_params / 1e6:.2f}M params')
