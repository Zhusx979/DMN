#!/usr/bin/env python
# coding: utf-8
"""Efficiency benchmark: the five quantities of the paper's results list.

Profiles the paper models and their comparison baselines at the paper-scale
configuration and prints (and saves) a markdown table with:

    * total parameters
    * FLOPs (per image: forward and forward + backward)
    * GPU memory (peak for inference and for one full training step)
    * inference latency (ms per image)
    * throughput (images per second)

Usage (from the repository root):

    python benchmarks/benchmark_efficiency.py
    python benchmarks/benchmark_efficiency.py --models DualMappingViT ViT \
        SynthesizerViT_dense SynthesizerViT_random SynthesizerViT_mix
    python benchmarks/benchmark_efficiency.py --models DualMappingViTSeg \
        --image_size 256 --dim 768 --num_classes 2
    python benchmarks/benchmark_efficiency.py --device cpu --models DualMappingViT

Available models:
    DualMappingViT        published Method (Eqs. 7-10)
    ViT                   vanilla baseline (torch-only reference of networks/vit.py,
                          identical parameter count and FLOPs)
    SynthesizerViT[_random|_random_fixed|_dense|_factorized_random|_mix|_static]
                          Synthesizer (ICML 2021) baselines; variants follow the
                          paper's definitions (every variant softmaxes S(X));
                          '_static' is an ablation with no softmax
    DualMappingViTSeg     segmentation model (use --image_size 256 --dim 768)

The markdown table is written to --out (default benchmarks/efficiency_report.md)
and printed to stdout.  All timings are CUDA-event based on GPU; on CPU the
GPU-memory columns stay "-".  Deliberately NOT available here: models that
need external packages of the training environment (einops/timm/monai).
"""

import argparse
import datetime
import sys
from collections import OrderedDict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from networks.dual_mapping_network_classification import DualMappingViT  # noqa: E402
from networks.dual_mapping_network_segmentation import DualMappingViTSeg  # noqa: E402
from networks.synthesizer_vit import SynthesizerViT  # noqa: E402
from networks.vit_torch import ViT  # noqa: E402
from utils import efficiency  # noqa: E402


def build_model(name, cfg):
    """Build one model by registry name; return (model, extra_note)."""
    if name == "DualMappingViT":
        model = DualMappingViT(image_size=cfg.image_size, patch_size=cfg.patch_size,
                               num_classes=cfg.num_classes, dim=cfg.dim,
                               depth=cfg.depth, mlp_dim=cfg.mlp_dim,
                               channels=cfg.channels)
        return model, ""
    if name == "ViT":
        model = ViT(image_size=cfg.image_size, patch_size=cfg.patch_size,
                    num_classes=cfg.num_classes, dim=cfg.dim, depth=cfg.depth,
                    heads=cfg.heads, mlp_dim=cfg.mlp_dim, channels=cfg.channels)
        return model, ""
    if name.startswith("SynthesizerViT"):
        variant = name.split("_", 1)[1] if "_" in name else cfg.syn_variant
        model = SynthesizerViT(image_size=cfg.image_size, patch_size=cfg.patch_size,
                               num_classes=cfg.num_classes, dim=cfg.dim,
                               depth=cfg.depth, mlp_dim=cfg.mlp_dim,
                               heads=cfg.heads, variant=variant,
                               channels=cfg.channels)
        return model, ""
    if name == "DualMappingViTSeg":
        model = DualMappingViTSeg(image_size=cfg.image_size, patch_size=cfg.patch_size,
                                  num_classes=cfg.num_classes, dim=cfg.dim,
                                  depth=cfg.depth, mlp_dim=cfg.mlp_dim,
                                  channels=cfg.channels)
        return model, ""
    raise SystemExit(f"unknown model '{name}' (see --help for the registry)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+",
                        default=["DualMappingViT", "ViT", "SynthesizerViT_random"])
    parser.add_argument("--syn_variant", default="random",
                        choices=["random", "random_fixed", "dense",
                                 "factorized_random", "mix", "static"],
                        help="variant used when listing plain 'SynthesizerViT'")
    parser.add_argument("--device", default="cuda", help="cuda | cuda:0 | cuda:1 | cpu")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--mlp_dim", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--latency_batch", type=int, default=1)
    parser.add_argument("--throughput_batch", type=int, default=32)
    parser.add_argument("--train_batch", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / "efficiency_report.md"))
    cfg = parser.parse_args()

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    device = torch.device(cfg.device)
    input_shape = (cfg.channels, cfg.image_size, cfg.image_size)

    results, errors = OrderedDict(), []
    for name in cfg.models:
        try:
            model, _ = build_model(name, cfg)
            model.to(device)
            model.eval()
            res = efficiency.benchmark_model(
                model, input_shape, device=device,
                latency_batch=cfg.latency_batch,
                throughput_batch=cfg.throughput_batch,
                train_batch=cfg.train_batch,
                warmup=cfg.warmup, iters=cfg.iters)
            results[name] = res
            print(f"[done] {name}: {res['params_total'] / 1e6:.2f}M params, "
                  f"{res['flops_forward'] / 1e9:.2f}G FLOPs/img")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:  # keep profiling the remaining models
            errors.append((name, repr(exc)))
            print(f"[skip] {name}: {exc}")

    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" \
        else "CPU"
    header = "\n".join([
        "# Efficiency benchmark (paper results list)",
        "",
        f"- Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- Device: {device_name}",
        f"- Torch: {torch.__version__}",
        f"- Input: {input_shape}, patch {cfg.patch_size}, dim {cfg.dim}, depth {cfg.depth}",
        f"- Latency batch {cfg.latency_batch} / throughput batch {cfg.throughput_batch} "
        f"/ train batch {cfg.train_batch}; warmup {cfg.warmup}, iters {cfg.iters}",
        "- \"ViT\" = torch-only reference of networks/vit.py (identical params/FLOPs); "
        "GPU mem infer measured at the latency batch, train mem = fwd+bwd+AdamW step",
        "",
    ])
    table = efficiency.format_table(results)
    text = header + table + "\n"
    if errors:
        text += "\nSkipped: " + "; ".join(f"{n} ({e})" for n, e in errors) + "\n"
    print("\n" + header + table)
    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.out).write_text(text)
    print(f"\nSaved: {cfg.out}")


if __name__ == "__main__":
    main()
