#!/usr/bin/env python
# coding: utf-8
"""Print the draft-paper segmentation tables (5 & 6) from run summaries.

Reads test metrics from segmentation/runs/<run>/summary.json and the
efficiency fingerprint (trainable params / FLOPs / inference memory /
latency-throughput) from each run's train.log, then prints the two
comparison tables in the draft's layout extended with efficiency columns:

    Table 5  BRISC        (Patch size = 8 / 16 default / 32, Depth = 4 / 8 / 16)
    Table 6  CardiacUDA   (Hospital G; same configuration grid)

The default entries appear twice (identical run) exactly as in the draft:
"Patch size = 16 (default)" and "Depth = 8 (default)" are both p16/d8.

Efficiency columns are measured at run start by utils/efficiency.py:
    Params(M)  trainable parameter count
    FLOPs(G)   forward FLOPs per image @ 256x256 (the config image_size)
    Mem(GB)    inference memory, batch 1
    Lat/Thr    latency ms/image and throughput img/s at batch 32
Latency/throughput can vary with machine load at measurement time; re-run
on an idle GPU for paper-grade numbers.

Usage (from the repository root):
    python scripts/report_seg_tables.py
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "segmentation" / "runs"

# (display label, patch, depth) in the draft's row order
ROWS = [
    ("Patch size = 8", 8, 8),
    ("Patch size = 16 (default)", 16, 8),
    ("Patch size = 32", 32, 8),
    ("Depth = 4", 16, 4),
    ("Depth = 8 (default)", 16, 8),
    ("Depth = 16", 16, 16),
]
METRICS = [("mDice", "mDice"), ("mIoU", "mIoU"),
           ("mSensitivity", "Sensitivity"), ("mPrecision", "Precision")]

WIDTH = 118


def run_name(dataset, site, model, patch, depth):
    base = dataset + ("_{}".format(site) if site else "")
    tag = "DMViT" if "Dual" in model else "ViT"
    return "{}_{}_p{}_d{}".format(base, tag, patch, depth)


def parse_efficiency(run_dir):
    """Extract (params_M, flops_G, infer_mem_GB, latency_ms, throughput) from train.log."""
    text = (run_dir / "train.log").read_text(errors="ignore")
    pat = {
        "params": r"total parameters [\d.]+M \(trainable ([\d.]+)M\)",
        "flops": r"FLOPs ([\d.]+)G fwd",
        "mem": r"GPU memory ([\d.]+)GB infer",
        "lat": r"inference latency ([\d.]+)ms/image, throughput ([\d.]+)img/s",
    }
    out = {}
    for key, pattern in pat.items():
        m = re.search(pattern, text)
        if not m:
            return None
        out[key] = [float(g) for g in m.groups()]
    return (out["params"][0], out["flops"][0], out["mem"][0],
            out["lat"][0], out["lat"][1])


def load_row(dataset, site, model, patch, depth):
    """Return (accuracy dict, efficiency tuple) for one run, or None."""
    rd = RUNS_ROOT / run_name(dataset, site, model, patch, depth)
    if not (rd / "summary.json").exists():
        return None
    test = json.loads((rd / "summary.json").read_text())["test"]
    acc = {key: test[key] for key, _ in METRICS}
    return acc, parse_efficiency(rd)


def print_table(title, dataset, site, default_note):
    print()
    print(title)
    print("=" * WIDTH)
    print("{:<27} {:<8} {:>8} {:>8} {:>12} {:>10} {:>9} {:>8} {:>8} {:>13}".format(
        "Configuration", "Model", "mDice", "mIoU", "Sensitivity", "Precision",
        "Params(M)", "FLOPs(G)", "Mem(GB)", "Lat(ms)/Thr"))
    print("-" * WIDTH)
    missing = []
    for label, patch, depth in ROWS:
        rows = {}
        for model in ("ViTSeg", "DualMappingViTSeg"):
            got = load_row(dataset, site, model, patch, depth)
            if got is None:
                missing.append("{}/{}".format(label, model))
                continue
            rows["DM ViT" if "Dual" in model else "ViT"] = got
        for i, tag in enumerate(("ViT", "DM ViT")):
            if tag not in rows:
                continue
            (acc, (params, flops, mem, lat, thr)) = rows[tag]
            print("{:<27} {:<8} {:>8.4f} {:>8.4f} {:>12.4f} {:>10.4f} "
                  "{:>8.2f}M {:>7.2f}G {:>8.2f} {:>6.2f}/{:<.0f}".format(
                      label if i == 0 else "", tag,
                      *(acc[key] for key, _ in METRICS),
                      params, flops, mem, lat, thr))
        if label != ROWS[-1][0]:
            print("-" * WIDTH)
    print("=" * WIDTH)
    print(default_note)
    if missing:
        print("missing runs: {}".format(", ".join(missing)))


def main():
    print_table(
        "Table 5: Comparison of mDice, mIoU, Sensitivity, and Precision between "
        "ViT and DM ViT\nwith different patch sizes and network depths "
        "(default: patch size 16, depth 8) on the BRISC dataset.\n"
        "Efficiency columns: trainable params / forward FLOPs @256x256 / "
        "inference memory (batch 1) / latency-throughput (batch 32).",
        "BRISC", None,
        "source: segmentation/runs/BRISC_{ViT,DMViT}_p*_d*/{summary.json,train.log} "
        "(official test split)")
    print_table(
        "Table 6: Comparison of mDice, mIoU, Sensitivity, and Precision between "
        "ViT and DM ViT\nwith different patch sizes and network depths "
        "(default: patch size 16, depth 8) on the CardiacUDA dataset.\n"
        "Hospital G\n"
        "Efficiency columns: trainable params / forward FLOPs @256x256 / "
        "inference memory (batch 1) / latency-throughput (batch 32).",
        "CardiacUDA", "G",
        "source: segmentation/runs/CardiacUDA_G_{ViT,DMViT}_p*_d*/{summary.json,train.log} "
        "(official split)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
