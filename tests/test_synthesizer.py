"""Structural and numeric checks of the Synthesizer baselines (ICML 2021).

Checks (runnable without pytest: `python tests/test_synthesizer.py`):
  1. random: forward equals the explicit Y = softmax(R) V computation and
     softmax(R) is row-stochastic (paper Eq. 4).
  2. dense: S = F1 sigma(F2(X_i)) is input-conditioned (changes with the
     input) and forward equals softmax(F(X)) V (paper Eqs. 1-3).
  3. factorized_random / mix: forward equals their explicit Eq. 7 / Eq. 6
     forms with softmax applied.
  4. static (ablation): forward equals R V with no softmax (row sums != 1).
  5. parameter counts: random/dense/static add N^2 per head, factorized
     adds 2Nk, mix adds 2Nk + (d^2 + dN) + 2 per head, with d = dim_head.
  6. backbone parity with DualMappingViT: patch embedding / cls / pos /
     blocks-FF / classification head carry identical parameter names and
     shapes, so a comparison between the two models isolates the attention.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from networks.dual_mapping_network_classification import DualMappingViT  # noqa: E402
from networks.synthesizer_vit import SynthesizerAttention, SynthesizerViT  # noqa: E402


def _merge(y, heads, dim_head):
    b, h, n, d = y.shape
    return y.permute(0, 2, 1, 3).reshape(b, n, heads * dim_head)


def test_random_forward_and_row_stochastic():
    torch.manual_seed(0)
    n_tokens, heads, dim_head = 5, 4, 8
    attn = SynthesizerAttention(dim=16, num_tokens=n_tokens, heads=heads,
                                dim_head=dim_head, variant="random")
    x = torch.randn(2, n_tokens, 16)
    v = attn.to_v(attn.norm(x)).reshape(2, n_tokens, heads, dim_head).permute(0, 2, 1, 3)
    a = torch.softmax(attn.R, dim=-1)
    expected = attn.to_out(_merge(torch.matmul(a, v), heads, dim_head))
    assert torch.allclose(attn(x), expected, atol=1e-6)
    row_sums = a.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    print("ok: random forward == softmax(R) V, rows sum to 1 (Eq. 4)")

    # 'Fixed' ablation: same math, buffer instead of parameter
    fixed = SynthesizerAttention(dim=16, num_tokens=n_tokens, heads=heads,
                                 dim_head=dim_head, variant="random_fixed")
    assert not any(p.requires_grad for p in fixed.parameters() if p is not None) or \
        all(n not in ("R",) for n, p in fixed.named_parameters())
    assert sum(b.numel() for n, b in fixed.named_buffers() if n.endswith("R")) == heads * n_tokens * n_tokens
    print("ok: random_fixed keeps R as a non-learnable buffer")


def test_dense_is_input_conditioned():
    torch.manual_seed(0)
    n_tokens, heads, dim_head = 5, 3, 8
    dim = heads * dim_head
    attn = SynthesizerAttention(dim=dim, num_tokens=n_tokens, heads=heads,
                                dim_head=dim_head, variant="dense")
    x1 = torch.randn(2, n_tokens, dim)
    x2 = torch.randn(2, n_tokens, dim)
    x1n, x2n = attn.norm(x1), attn.norm(x2)

    def align(x):
        return torch.softmax(attn._dense_logits(x), dim=-1)

    a1, a2 = align(x1n), align(x2n)
    assert not torch.allclose(a1, a2, atol=1e-6), "dense S(X) must depend on the input"
    v = attn.to_v(x1n).reshape(2, n_tokens, heads, dim_head).permute(0, 2, 1, 3)
    expected = attn.to_out(_merge(torch.matmul(a1, v), heads, dim_head))
    assert torch.allclose(attn(x1), expected, atol=1e-6)
    print("ok: dense S = F1 ReLU(F2(X_i)) is input-conditioned (Eqs. 1-3)")


def test_dense_uses_per_head_width():
    """F_{h,l}: R^d -> R^N must use d = dim_head (paper Table 1)."""
    n_tokens, heads, dim_head = 5, 4, 8
    dim = 16  # deliberately != heads * dim_head -> projection supplies heads
    attn = SynthesizerAttention(dim=dim, num_tokens=n_tokens, heads=heads,
                                dim_head=dim_head, variant="dense")
    assert attn.w1.shape == (heads, dim_head, dim_head)
    assert attn.b1.shape == (heads, dim_head)
    assert attn.w2.shape == (heads, n_tokens, dim_head)
    assert attn.b2.shape == (heads, n_tokens)
    assert isinstance(attn.head_proj, torch.nn.Linear)
    assert attn.head_proj.in_features == dim
    assert attn.head_proj.out_features == heads * dim_head

    dim = heads * dim_head
    attn = SynthesizerAttention(dim=dim, num_tokens=n_tokens, heads=heads,
                                dim_head=dim_head, variant="dense")
    assert isinstance(attn.head_proj, torch.nn.Identity)
    assert attn.w1.shape == (heads, dim_head, dim_head)
    print("ok: dense F uses per-head width d = dim_head (Table 1)")


def test_factorized_and_mix_forward():
    torch.manual_seed(0)
    n_tokens, heads, dim_head, k = 5, 3, 8, 2
    dim = heads * dim_head
    x = torch.randn(2, n_tokens, dim)

    fr = SynthesizerAttention(dim=dim, num_tokens=n_tokens, heads=heads,
                              dim_head=dim_head, variant="factorized_random", k=k)
    v = fr.to_v(fr.norm(x)).reshape(2, n_tokens, heads, dim_head).permute(0, 2, 1, 3)
    a = torch.softmax(torch.matmul(fr.R1, fr.R2.transpose(-1, -2)), dim=-1)
    assert torch.allclose(fr(x), fr.to_out(_merge(torch.matmul(a, v), heads, dim_head)), atol=1e-6)

    mx = SynthesizerAttention(dim=dim, num_tokens=n_tokens, heads=heads,
                              dim_head=dim_head, variant="mix", k=k)
    xn = mx.norm(x)
    fac = torch.matmul(mx.R1, mx.R2.transpose(-1, -2))
    fx = mx._dense_logits(xn)
    a = torch.softmax(mx.alpha1.view(1, heads, 1, 1) * fac
                      + mx.alpha2.view(1, heads, 1, 1) * fx, dim=-1)
    v = mx.to_v(xn).reshape(2, n_tokens, heads, dim_head).permute(0, 2, 1, 3)
    assert torch.allclose(mx(x), mx.to_out(_merge(torch.matmul(a, v), heads, dim_head)), atol=1e-6)
    print("ok: factorized_random (Eq. 7) and mix (Eq. 6) forwards match explicit forms")


def test_static_ablation_has_no_softmax():
    torch.manual_seed(0)
    n_tokens, heads, dim_head = 5, 3, 8
    st = SynthesizerAttention(dim=16, num_tokens=n_tokens, heads=heads,
                              dim_head=dim_head, variant="static")
    x = torch.randn(2, n_tokens, 16)
    v = st.to_v(st.norm(x)).reshape(2, n_tokens, heads, dim_head).permute(0, 2, 1, 3)
    assert torch.allclose(st(x), st.to_out(_merge(torch.matmul(st.R, v), heads, dim_head)), atol=1e-6)
    row_sums = st.R.sum(dim=-1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
    print("ok: static ablation is Y = R V with no softmax (row sums != 1)")


def test_parameter_counts():
    n_tokens, heads, depth, dim_head, k = 5, 4, 2, 8, 2
    dim = heads * dim_head  # no head projection needed

    def build(variant):
        return SynthesizerViT(image_size=32, patch_size=16, num_classes=3,
                              dim=dim, depth=depth, mlp_dim=32, heads=heads,
                              dim_head=dim_head, variant=variant, k=k)

    random_ = build("random")
    n_R = sum(p.numel() for n, p in random_.named_parameters() if n.endswith("R"))
    assert n_R == depth * heads * n_tokens * n_tokens, n_R

    dense = build("dense")
    n_dense = sum(p.numel() for n, p in dense.named_parameters()
                  if any(k in n for k in (".w1", ".b1", ".w2", ".b2")))
    # per head: Linear(dim_head, dim_head) + bias, then Linear(dim_head, N) + bias
    per_head = dim_head * dim_head + dim_head + dim_head * n_tokens + n_tokens
    assert n_dense == depth * heads * per_head, n_dense
    assert not any("head_proj" in n for n, _ in dense.named_parameters())

    fr = build("factorized_random")
    n_fr = sum(p.numel() for n, p in fr.named_parameters() if n.endswith(("R1", "R2")))
    assert n_fr == depth * heads * 2 * n_tokens * k, n_fr

    mx = build("mix")
    n_mx = sum(p.numel() for n, p in mx.named_parameters()
               if n.endswith(("R1", "R2")) or "alpha" in n
               or any(k in n for k in (".w1", ".b1", ".w2", ".b2")))
    per_head_mix = 2 * n_tokens * k + per_head + 2
    assert n_mx == depth * heads * per_head_mix, n_mx
    print(f"ok: parameter counts R={n_R} dense={n_dense} factorized={n_fr} mix={n_mx}")


def test_backbone_parity_with_dm_vit():
    kw = dict(image_size=32, patch_size=16, num_classes=3, dim=16, depth=2, mlp_dim=32)
    dm = DualMappingViT(**kw)
    syn = SynthesizerViT(**kw, heads=4, dim_head=4)

    def backbone_keys(model):
        keys = {}
        for n, p in model.named_parameters():
            if n.endswith(("W_S", "W_D.weight", "R", "R1", "R2")) \
                    or "alpha" in n or "to_v" in n or "to_out" in n \
                    or any(k in n for k in (".w1", ".b1", ".w2", ".b2")):
                continue
            keys[n] = tuple(p.shape)
        return keys

    assert backbone_keys(dm) == backbone_keys(syn)
    print("ok: patch embedding / cls / pos / blocks-FP / head are parameter-identical")


if __name__ == "__main__":
    test_random_forward_and_row_stochastic()
    test_dense_is_input_conditioned()
    test_dense_uses_per_head_width()
    test_factorized_and_mix_forward()
    test_static_ablation_has_no_softmax()
    test_parameter_counts()
    test_backbone_parity_with_dm_vit()
    print("\nAll Synthesizer checks passed.")
