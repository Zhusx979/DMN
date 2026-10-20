"""Numeric verification of the Dual Mapping attention against the published paper.

Reference: "Efficient Vision Transformer for Medical Image Analysis",
International Journal of Neural Systems (2024), Method Eqs. (7)-(10):

    W_S in R^{N x N} (N = sequence length = num_patches + 1) with
        sum_j W_{S,ij} = 1,  W_{S,ij} >= 0,  W_{S,ij} <= 1            (Eq. 7)
    X' = W_S X W_D                                                    (Eq. 8)
    L_sum   = (1/N)   * sum_i (sum_j W_ij - 1)^2                      (Eq. 9)
    L_lower = (1/N^2) * sum_ij max(-W_ij, 0)^2                        (Eq. 9)
    L_upper = (1/N^2) * sum_ij max( W_ij - 1, 0)^2                    (Eq. 9)
    L_reg   = L_sum + L_lower + L_upper                               (Eq. 10)

Checks (runnable without pytest: `python tests/test_dual_mapping.py`):
  1. W_S is (N, N) and at initialization (uniform 1/N) the constraints of
     Eq. (7) hold exactly, so L_reg (Eqs. 9-10) is 0.
  2. L_reg matches a direct implementation of Eqs. (9-10) on a perturbed W_S.
  3. L_reg is differentiable: gradients reach W_S.
  4. Forward equals W_S @ X @ W_D explicitly (Eq. 8), with W_S mixing tokens.
  5. DM attention uses 2 parameter matrices where vanilla attention uses 3
     (Q/K/V), i.e. the parameter/compute saving claimed by the paper.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from networks.dual_mapping_network_classification import DMAttention, DualMappingViT  # noqa: E402


def test_init_constraints_and_zero_reg():
    n_tokens = 10
    attn = DMAttention(dim=64, num_tokens=n_tokens)
    w = attn.W_S
    assert w.shape == (n_tokens, n_tokens), f"W_S must be N x N, got {tuple(w.shape)}"
    assert torch.allclose(w.sum(dim=1), torch.ones(n_tokens)), "rows must sum to 1 at init"
    assert w.min().item() >= 0.0 and w.max().item() <= 1.0, "entries must be in [0, 1]"
    # row sums are 1 up to floating-point summation error, so L_reg ~ 0 (not
    # exactly 0.0: sum of N copies of 1/N carries ~1e-16 rounding)
    assert attn.regularization_loss().item() < 1e-12, "L_reg must be ~0 at init"
    print("ok: W_S is N x N, constraints (Eq. 7) hold at init and L_reg ~ 0")


def test_reg_matches_equations():
    n_tokens = 8
    attn = DMAttention(dim=16, num_tokens=n_tokens)
    with torch.no_grad():
        # perturb: introduce row-sum errors and out-of-range entries
        attn.W_S[0, :] *= 1.5      # row sum 1.5
        attn.W_S[1, :] *= 0.5      # row sum 0.5
        attn.W_S[2, 0] = -0.3      # below 0
        attn.W_S[3, 0] = 2.0       # above 1
    w = attn.W_S
    n = w.shape[0]                 # sequence length N of Eqs. (9)-(10)
    l_sum = (1.0 / n) * ((w.sum(dim=1) - 1.0) ** 2).sum()
    l_lower = (1.0 / n ** 2) * (torch.clamp(-w, min=0.0) ** 2).sum()
    l_upper = (1.0 / n ** 2) * (torch.clamp(w - 1.0, min=0.0) ** 2).sum()
    expected = l_sum + l_lower + l_upper
    assert torch.allclose(attn.regularization_loss(), expected, atol=1e-7)
    print(f"ok: L_reg matches Eqs. (9)-(10) exactly ({expected.item():.6f}, N={n})")


def test_reg_gradient_flows():
    attn = DMAttention(dim=16, num_tokens=8)
    with torch.no_grad():
        attn.W_S[0, :] *= 1.5
    attn.regularization_loss().backward()
    assert attn.W_S.grad is not None and attn.W_S.grad.abs().sum() > 0
    print("ok: L_reg is differentiable; gradient reaches W_S")


def test_forward_equals_equation():
    torch.manual_seed(0)
    n_tokens = 5
    attn = DMAttention(dim=16, num_tokens=n_tokens, dim_out=16)
    assert attn.W_S.shape == (n_tokens, n_tokens)  # N x N token-mixing matrix
    x = torch.randn(2, n_tokens, 16)
    expected = torch.matmul(attn.W_S, attn.W_D(attn.norm(x)))
    got = attn(x)
    assert torch.allclose(got, expected, atol=1e-6), "forward must be W_S @ X @ W_D"
    print("ok: forward implements Eq. (8): X' = W_S X W_D (W_S mixes tokens, N x N)")


def test_parameter_saving():
    """DM attention has 2 weight matrices per block; vanilla attention has 3."""
    dim, depth = 64, 2
    image_size, patch_size = 32, 16
    n_tokens = (image_size // patch_size) ** 2 + 1  # num_patches + cls token
    dm = DualMappingViT(image_size=image_size, patch_size=patch_size, num_classes=3,
                        dim=dim, depth=depth, mlp_dim=128)
    assert dm.num_tokens == n_tokens
    assert tuple(dm.transformer.layers[0][0].W_S.shape) == (n_tokens, n_tokens)

    dm_attn_params = sum(p.numel() for n, p in dm.named_parameters()
                         if n.endswith("W_S") or n.endswith("W_D.weight"))
    # DM attention per block: W_S (N x N) + W_D (d_m x d_v), no biases
    assert dm_attn_params == depth * (n_tokens * n_tokens + dim * dim), dm_attn_params

    # structurally "two matrices instead of three (Q, K, V)"
    n_matrices = sum(1 for n, _ in dm.named_parameters()
                     if n.endswith("W_S") or n.endswith("W_D.weight"))
    assert n_matrices == 2 * depth

    try:
        # Cross-check against the repo's vanilla ViT (needs einops installed).
        from networks.vit import ViT
        vit = ViT(image_size=image_size, patch_size=patch_size, num_classes=3,
                  dim=dim, depth=depth, heads=4, mlp_dim=128, dim_head=dim // 4)
        vanilla = sum(p.numel() for n, p in vit.named_parameters() if "to_qkv" in n)
    except ImportError:
        # einops not installed: analytical Q/K/V count, 3 matrices without bias
        vanilla = depth * 3 * dim * dim
        print("    (note: einops not installed - using analytical vanilla count)")

    assert dm_attn_params < vanilla
    print(f"ok: attention projection params DM={dm_attn_params} vs vanilla={vanilla} "
          f"(2 matrices vs 3)")


if __name__ == "__main__":
    test_init_constraints_and_zero_reg()
    test_reg_matches_equations()
    test_reg_gradient_flows()
    test_forward_equals_equation()
    test_parameter_saving()
    print("\nAll Dual Mapping checks passed.")
