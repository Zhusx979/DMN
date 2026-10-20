"""Vanilla ViT baseline (einops-free reference implementation).

Architecture-identical re-implementation of ``networks/vit.py`` -- the
research baseline used for every comparison in the paper: patch embedding +
cls token + positional embedding + pre-norm transformer blocks (multi-head
self-attention with an explicit Q/K/V projection, softmax attention) + linear
classification head.

This file exists so the vanilla ViT can be built, profiled
(``utils/efficiency.py``) and smoke-tested on machines without einops.
Parameter counts and FLOPs are identical to ``networks/vit.py``; only the
patch-embedding path (einops ``Rearrange``) is replaced by the equivalent
reshape/permute, and ``repeat`` by ``expand``.

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


class Attention(nn.Module):
    """Standard multi-head self-attention (as in networks/vit.py)."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (t.reshape(t.shape[0], t.shape[1], self.heads, -1).permute(0, 2, 1, 3)
                   for t in qkv)
        attn = self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale)
        out = torch.matmul(self.dropout(attn), v)
        out = out.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads,
                 mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        image_height, image_width = _pair(image_size)
        patch_height, patch_width = _pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls or mean'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.patch_height = patch_height
        self.pool = pool

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()
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
        return self.mlp_head(self.to_latent(x))


if __name__ == '__main__':
    model = ViT(image_size=224, patch_size=16, num_classes=3, dim=64, depth=2,
                heads=16, mlp_dim=128)
    y = model(torch.randn(2, 3, 224, 224))
    n_params = sum(p.numel() for p in model.parameters())
    print(f'forward ok: {tuple(y.shape)}, params {n_params / 1e6:.2f}M')
