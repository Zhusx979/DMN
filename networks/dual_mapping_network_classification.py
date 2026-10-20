"""Dual Mapping (DM) Vision Transformer -- image classification.

Paper-strict implementation of the classification model from
"Efficient Vision Transformer for Medical Image Analysis"
(International Journal of Neural Systems, 2024).

Core idea (Method, "Dual Mapping"):
    Vanilla self-attention builds the N x N similarity matrix from a scaled
    dot-product, C = softmax(Q K^T / sqrt(d_k)), using three projection
    matrices (Q, K, V).  DM attention replaces that matrix by a *learnable*
    parameter matrix W_S of the same shape and keeps a single value
    projection, so the attention output is computed directly as

        X' = W_S X W_D                                              (Eq. 8)

    where
        W_S in R^{N x N}     -- learnable similarity (token-mixing) matrix,
                                N = sequence length = num_patches + 1
                                (cls token), with row-stochastic / box
                                constraints (Eq. 7):
                                    sum_j W_{S,ij} = 1,
                                    W_{S,ij} >= 0,
                                    W_{S,ij} <= 1;
        W_D in R^{d_m x d_v} -- value projection, learned without constraints
                                (d_v = d_m here so the residual connection is
                                shape-preserving).

    The abstract names the two matrices C and D; in the Method section they
    are denoted W_S (similarity) and W_D (value projection): C == W_S,
    D == W_D.  They replace the three Q/K/V matrices of vanilla
    self-attention -- the query and key projections are replaced by the
    learnable W_S, so the QK^T dot-product is omitted and the number of
    attention parameter matrices is reduced from three to two.

    The constraints are enforced *softly* through a differentiable
    regularizer added to the task loss during training (Eqs. 9-10):

        L_sum   = (1/N)   * sum_i (sum_j W_{S,ij} - 1)^2             (Eq. 9)
        L_lower = (1/N^2) * sum_ij max(-W_{S,ij}, 0)^2               (Eq. 9)
        L_upper = (1/N^2) * sum_ij max( W_{S,ij} - 1, 0)^2           (Eq. 9)
        L_reg   = L_sum + L_lower + L_upper                         (Eq. 10)

    Total training objective:  L = L_task + L_reg  (model.regularization_loss()).
    W_S is initialized uniformly at 1/N, so the constraints (Eq. 7) hold at
    initialization up to floating-point rounding and L_reg ~ 0 there; the
    forward pass itself applies no softmax or normalization (the constraint
    lives in the loss, not in the architecture).

Usage
-----
    from networks.dual_mapping_network_classification import DualMappingViT
    model = DualMappingViT(image_size=224, patch_size=16, num_classes=4,
                           dim=1024, depth=8, mlp_dim=2048)
    logits = model(images)                      # (B, num_classes)
    loss = ce_loss(logits, labels) + model.regularization_loss()

Note
----
The numbers reported in the published paper were produced with the legacy
research implementation in `legacy/networks/score_vit.py`, which uses the
same N x N learnable token-mixing slot but enforces the probability
constraints *hard* through a softmax re-parameterization of the weights in
the forward pass (and adds an output projection).  This file follows the
Method section of the published paper exactly: learnable unconstrained W_S
with the soft regularizer of Eqs. 9-10, no softmax, no output projection.
The two variants are not checkpoint-compatible.  (The pre-publication draft
instead used a d_m x d_m W_S with X' = X W_S W_D, normalized by d_m.)

This module depends on PyTorch only (no einops / timm).
"""

import torch
from torch import nn


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


class DMAttention(nn.Module):
    """Dual Mapping attention: X' = W_S X W_D  (paper Eqs. 7-8).

    A single (W_S, W_D) pair is used per block -- one "head" -- exactly as in
    the paper's formulation: the Q/K projections of vanilla self-attention
    are replaced by the learnable N x N matrix W_S and V by W_D, reducing
    the three attention parameter matrices to two.  The constraints on W_S
    (Eq. 7) are soft (see regularization_loss); the forward pass performs no
    softmax or normalization.

    Args:
        dim       : token width d_m.
        num_tokens: sequence length N (= num_patches + 1 with a cls token).
        dim_out   : output width d_v (defaults to d_m).
        dropout   : dropout on the attention output (paper: 0).
    """

    def __init__(self, dim, num_tokens, dim_out=None, dropout=0.):
        super().__init__()
        dim_out = dim if dim_out is None else dim_out
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(dim)
        # W_S in R^{N x N}: learnable similarity matrix.  Initialized
        # uniformly at 1/N -> Eq. 7 holds at init (up to fp rounding),
        # L_reg ~ 0.
        self.W_S = nn.Parameter(torch.full((num_tokens, num_tokens), 1.0 / num_tokens))
        # W_D in R^{d_m x d_v}: value projection, weight matrix only (no bias).
        self.W_D = nn.Linear(dim, dim_out, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Eq. 8: X' = W_S X W_D -- token mixing by W_S (left), feature
        # projection by W_D (right).
        x = self.norm(x)
        return self.dropout(torch.matmul(self.W_S, self.W_D(x)))

    def regularization_loss(self):
        """L_reg = L_sum + L_lower + L_upper  (Eqs. 9-10); differentiable.

        The .mean() reductions over the N rows / N^2 entries implement the
        1/N and 1/N^2 normalizations of the paper's equations exactly.
        """
        w = self.W_S
        l_sum = (w.sum(dim=1) - 1.0).pow(2).mean()   # (1/N)   sum_i (row sum - 1)^2
        l_lower = torch.relu(-w).pow(2).mean()       # (1/N^2) sum_ij max(-w, 0)^2
        l_upper = torch.relu(w - 1.0).pow(2).mean()  # (1/N^2) sum_ij max(w - 1, 0)^2
        return l_sum + l_lower + l_upper


class DMTransformer(nn.Module):
    """Stack of pre-norm blocks; each block = DMAttention + FeedForward."""

    def __init__(self, dim, num_tokens, depth, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                DMAttention(dim, num_tokens, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

    def regularization_loss(self):
        total = None
        for attn, _ in self.layers:
            layer_reg = attn.regularization_loss()
            total = layer_reg if total is None else total + layer_reg
        return total


class DualMappingViT(nn.Module):
    """DM ViT for classification (paper: ViT encoder + linear classification head).

    Args:
        image_size : square input resolution (paper: 224).
        patch_size : square patch size (paper: 16).
        num_classes: output classes (paper: retinal OCT 4, pneumonia 2,
                     breast cancer 3).
        dim        : token embedding width d_m (paper settings: 1024).
        depth      : number of DM transformer blocks (paper: 8).
        mlp_dim    : feed-forward hidden width (paper settings: 2048).
        pool       : 'cls' or 'mean' token pooling.
        channels   : input image channels (papers datasets: 3; DeepBC
                     pipeline: 4).
    """

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, mlp_dim,
                 pool='cls', channels=3, dropout=0., emb_dropout=0.):
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
        self.num_tokens = num_patches + 1  # patches + cls token = N in Eqs. 7-10

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = DMTransformer(dim, self.num_tokens, depth, mlp_dim, dropout)
        # bare Linear, identical to the ViT baseline's head: the mapping module
        # is the only architectural variable between the two models
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

    def regularization_loss(self):
        """L_reg summed over all DM attention blocks (Eqs. 9-10)."""
        return self.transformer.regularization_loss()


if __name__ == '__main__':
    model = DualMappingViT(image_size=224, patch_size=16, num_classes=3,
                           dim=64, depth=2, mlp_dim=128)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    loss = nn.CrossEntropyLoss()(y, torch.tensor([0, 2])) + model.regularization_loss()
    loss.backward()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'forward ok: {tuple(y.shape)}, loss {loss.item():.4f}, params {n_params/1e6:.2f}M')
    print(f'W_S shape per block: {tuple(model.transformer.layers[0][0].W_S.shape)} (N = num_patches + 1)')
