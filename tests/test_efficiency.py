"""Checks of the efficiency profiler (utils/efficiency.py).

Runnable without pytest: `python tests/test_efficiency.py`.
  1. parameter counting matches the module sum.
  2. FLOPs are positive and training FLOPs > forward FLOPs.
  3. CPU latency/throughput are measured (GPU memory reported as None).
  4. the markdown table renders with the five paper columns.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from networks.dual_mapping_network_classification import DualMappingViT  # noqa: E402
from utils import efficiency  # noqa: E402


def test_params_and_flops():
    model = DualMappingViT(image_size=32, patch_size=16, num_classes=3,
                           dim=32, depth=1, mlp_dim=64)
    params = efficiency.count_parameters(model)
    manual = sum(p.numel() for p in model.parameters())
    assert params["total"] == manual == params["trainable"]
    flops_fwd = efficiency.count_flops(model, (3, 32, 32), device="cpu", mode="forward")
    flops_train = efficiency.count_flops(model, (3, 32, 32), device="cpu", mode="train")
    assert flops_fwd > 0 and flops_train > flops_fwd
    # module is left untouched
    assert all(p.grad is None for p in model.parameters())
    print(f"ok: params={manual}, FLOPs fwd={flops_fwd:.0f} < train={flops_train:.0f}")


def test_latency_and_table():
    model = DualMappingViT(image_size=32, patch_size=16, num_classes=3,
                           dim=32, depth=1, mlp_dim=64)
    res = efficiency.measure_inference(model, (3, 32, 32), device="cpu",
                                       batch_size=2, warmup=1, iters=3)
    assert res["latency_ms_per_image"] > 0
    assert res["throughput_img_per_s"] > 0
    assert res["peak_gpu_mem_gb"] is None  # CPU -> no memory number
    table = efficiency.format_table({"m": {
        "params_total": 1234, "flops_forward": 5e8, "flops_train": 1.5e9,
        "latency_ms_per_image": 1.23, "throughput_img_per_s": 100.0,
        "peak_gpu_mem_infer_gb": None, "peak_gpu_mem_train_step_gb": None}})
    assert "Params (M)" in table and "Latency (ms/img)" in table and "-" in table
    print("ok: CPU latency/throughput measured; table renders")


if __name__ == "__main__":
    test_params_and_flops()
    test_latency_and_table()
    print("\nAll efficiency checks passed.")
