"""Synthesizer (Tay et al., ICML 2021) -- segmentation baseline.

Segmentation counterpart of ``networks/synthesizer_vit.py``: the encoder
layout (patch embedding, cls token, positional embedding, pre-norm blocks,
final LayerNorm) and the convolutional decoder / segmentation head are
identical to ``networks/dual_mapping_network_segmentation.py`` and
``networks/vit_seg.py``; only the alignment module differs (a synthesized
matrix instead of the Q.K^T dot product / W_S).  See synthesizer_vit.py for
the paper-faithful definitions of the variants (all paper variants apply the
softmax inside the forward pass, paper Eq. 1-7).

There is no parameter-level regularizer (unlike DM models): the alignment
is normalized by softmax, so ``regularization_loss`` does not exist and the
training objective is the plain segmentation loss.

Usage
-----
    from networks.synthesizer_vit_seg import SynthesizerViTSeg
    model = SynthesizerViTSeg(image_size=256, patch_size=16, num_classes=2,
                              dim=768, depth=8, mlp_dim=2048, channels=3,
                              variant="random")
    logits = model(images)                      # (B, num_classes, H, W)
"""

import torch
import torch.nn.functional as F
from torch import nn

from networks.synthesizer_vit import SynthesizerTransformer, _pair, _patchify


class SynthesizerViTSeg(nn.Module):
    """Synthesizer encoder + lightweight conv decoder for segmentation.

    Args:
        image_size  : square input resolution.
        patch_size  : square patch size.
        num_classes : output segmentation classes.
        dim         : token embedding width.
        depth       : number of transformer blocks.
        mlp_dim     : feed-forward hidden width.
        heads       : number of synthetic-attention heads (research config: 16).
        dim_head    : per-head width (research config: 64).
        variant     : one of VARIANTS in networks.synthesizer_vit ('random' is
                      the paper-faithful default).
        k           : factorization rank for factorized_random / mix (paper: 8).
        channels    : input image channels.
        dropout     : dropout on attention output and feed-forward.
        emb_dropout : dropout on the token embedding.
    """

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, mlp_dim,
                 heads=16, dim_head=64, variant="random", k=8, channels=3,
                 dropout=0., emb_dropout=0.):
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
        self.num_tokens = num_patches + 1  # patches + cls token

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

        # Lightweight convolutional decoder (identical to ViTSeg / DMN seg).
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
        # conv-transposes restore the feature map to 16x16; a single conditional
        # bilinear resample brings logits back to input resolution -- exactly
        # the ViTSeg / DMN-seg path (final_upsample stays an unused layer there
        # too).
        if logits.shape[2:] != img.shape[2:]:
            logits = F.interpolate(logits, size=img.shape[2:], mode='bilinear',
                                   align_corners=False)
        return logits


if __name__ == '__main__':
    model = SynthesizerViTSeg(image_size=256, patch_size=16, num_classes=2,
                              dim=64, depth=2, mlp_dim=128, channels=3,
                              variant="random")
    y = model(torch.randn(2, 3, 256, 256))
    n_params = sum(p.numel() for p in model.parameters())
    print(f'forward ok: {tuple(y.shape)}, params {n_params / 1e6:.2f}M')
