"""Dual Mapping (DM) Vision Transformer -- medical image segmentation.

Paper-strict implementation of the segmentation model from
"Efficient Vision Transformer for Medical Image Analysis"
(International Journal of Neural Systems, 2024).

Architecture (paper, "Segmentation" experiment of the research line):
    * ViT encoder: patch embedding + class token + positional embedding + a
      stack of `depth` DM transformer blocks.  Each block uses Dual Mapping
      attention        X' = W_S X W_D                            (Eq. 8)
      with W_S in R^{N x N} (N = num_patches + 1 tokens) replacing the Q/K
      projections, W_D in R^{d_m x d_v} as the value projection, and the
      constraints of Eq. 7 enforced softly via L_reg (Eqs. 9-10).  See
      dual_mapping_network_classification.py for the full description of the
      equations; the implementation is shared in spirit but kept here as a
      single self-contained file (torch only).
    * Lightweight convolutional decoder: three transposed-convolution stages
      (kernel 4, stride 2, padding 1) with ReLU, a 1x1 convolution segmentation
      head, and bilinear upsampling back to the input resolution.

Training loss: task loss (Dice + cross entropy for segmentation) plus the
constraint regularizer L_reg of the DM attention matrices, obtained via
`model.regularization_loss()`.

Note: the decoder design follows the research implementation that produced the
published results (legacy/networks/score_VIT_seg.py); the paper only states
"a lightweight convolutional decoder".  The attention itself is paper-strict:
learnable unconstrained W_S (N x N) with the soft regularizer of Eqs. 9-10,
no softmax, no output projection -- see the classification module's note for
the difference with the legacy implementation.

This module depends on PyTorch only (no einops / timm).
"""

import torch
import torch.nn.functional as F
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

    W_S in R^{N x N} (N = num_patches + 1) is the learnable token-mixing
    matrix replacing the Q/K projections; W_D in R^{d_m x d_v} is the value
    projection.  The Eq. 7 constraints are soft (see regularization_loss);
    the forward pass performs no softmax or normalization.

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
        # W_S in R^{N x N}: initialized uniformly at 1/N -> Eq. 7 holds at
        # init (up to fp rounding), L_reg ~ 0.
        self.W_S = nn.Parameter(torch.full((num_tokens, num_tokens), 1.0 / num_tokens))
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
    """Stack of pre-norm DM blocks (attention + feed-forward, residual)."""

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


class DualMappingViTSeg(nn.Module):
    """DM ViT encoder + lightweight convolutional decoder for segmentation.

    Args:
        image_size : training input resolution (paper: random crop 256).
        patch_size : square patch size (paper: 16).
        num_classes: segmentation classes (e.g. BRISC 2, CardiacUDA 5).
        dim        : token embedding width d_m (paper settings: 768).
        depth      : number of DM transformer blocks (paper: 8).
        mlp_dim    : feed-forward hidden width.
        pool       : unused for segmentation output, kept for API parity.
        channels   : input image channels.
    """

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, mlp_dim,
                 pool='cls', channels=3, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = _pair(image_size)
        patch_height, patch_width = _pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'
        assert patch_height == patch_width, 'only square patches are supported'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.patch_height = patch_height
        self.feat_h = image_height // patch_height
        self.feat_w = image_width // patch_width
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

        # Lightweight convolutional decoder (as in the research implementation).
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(dim // 4, dim // 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.segmentation_head = nn.Conv2d(dim // 8, num_classes, kernel_size=1)
        self.final_upsample = nn.Upsample(scale_factor=patch_height, mode='bilinear',
                                          align_corners=False)

    def forward(self, img):
        x = self.to_patch_embedding(_patchify(img, self.patch_height))
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :n + 1]
        x = self.dropout(x)

        x = self.transformer(x)
        x = x[:, 1:, :]  # drop the class token
        x = x.transpose(1, 2).reshape(b, -1, self.feat_h, self.feat_w)

        x = self.decoder(x)
        logits = self.segmentation_head(x)
        # decoder conv-transposes restore 16x16 -> 128x128; a single conditional
        # bilinear resample brings logits back to input resolution -- exactly the
        # legacy/ViTSeg path (final_upsample stays an unused layer there too).
        if logits.shape[2:] != img.shape[2:]:
            logits = F.interpolate(logits, size=img.shape[2:], mode='bilinear',
                                   align_corners=False)
        return logits

    def regularization_loss(self):
        """L_reg summed over all DM attention blocks (Eqs. 9-10)."""
        return self.transformer.regularization_loss()


if __name__ == '__main__':
    model = DualMappingViTSeg(image_size=256, patch_size=16, num_classes=2,
                              dim=64, depth=2, mlp_dim=128, channels=3)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    loss = F.cross_entropy(y, torch.randint(0, 2, (2, 256, 256))) + model.regularization_loss()
    loss.backward()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'forward ok: {tuple(y.shape)}, loss {loss.item():.4f}, params {n_params/1e6:.2f}M')
    print(f'W_S shape per block: {tuple(model.transformer.layers[0][0].W_S.shape)} (N = num_patches + 1)')
