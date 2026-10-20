#!/usr/bin/env python
# coding: utf-8
"""Compare segmentation runs against the draft-paper tables 5 / 6 numbers.

Reads every finished run under ``segmentation/runs/*/summary.json`` and prints
(and saves, markdown) a side-by-side table:

    configuration | model | paper mDice/mIoU/Sens/Prec | ours ... | delta

Paper numbers are the pre-publication draft (sn-article.pdf):
  * Table 5  BRISC       (2 classes)
  * Table 6  CardiacUDA  (hospital G and R, 5 classes)

Usage:
    python segmentation/compare_with_paper.py
    python segmentation/compare_with_paper.py --out segmentation/results/comparison.md
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "segmentation" / "runs"

# (mDice, mIoU, Sensitivity, Precision) -- draft tables 5 and 6
PAPER = {
    ("BRISC", None): {
        (8, 8):   {"ViT": (0.7328, 0.6182, 0.7651, 0.7569), "DualMappingViTSeg": (0.7442, 0.6251, 0.7712, 0.7751)},
        (16, 8):  {"ViT": (0.7246, 0.6093, 0.7592, 0.7502), "DualMappingViTSeg": (0.7307, 0.6133, 0.7631, 0.7694)},
        (32, 8):  {"ViT": (0.7054, 0.5903, 0.7405, 0.7312), "DualMappingViTSeg": (0.7129, 0.5957, 0.7456, 0.7501)},
        (16, 4):  {"ViT": (0.7112, 0.5950, 0.7454, 0.7356), "DualMappingViTSeg": (0.7184, 0.6012, 0.7573, 0.7558)},
        (16, 16): {"ViT": (0.7373, 0.6159, 0.7624, 0.7550), "DualMappingViTSeg": (0.7381, 0.6262, 0.7680, 0.7751)},
    },
    ("CardiacUDA", "G"): {
        (8, 8):   {"ViT": (0.7649, 0.6468, 0.7384, 0.8349), "DualMappingViTSeg": (0.7711, 0.6546, 0.7448, 0.8412)},
        (16, 8):  {"ViT": (0.7615, 0.6421, 0.7350, 0.8319), "DualMappingViTSeg": (0.7677, 0.6501, 0.7412, 0.8380)},
        (32, 8):  {"ViT": (0.7567, 0.6371, 0.7303, 0.8276), "DualMappingViTSeg": (0.7679, 0.6481, 0.7395, 0.8397)},
        (16, 4):  {"ViT": (0.7463, 0.6258, 0.7181, 0.8189), "DualMappingViTSeg": (0.7526, 0.6336, 0.7242, 0.8248)},
        (16, 16): {"ViT": (0.7674, 0.6498, 0.7421, 0.8376), "DualMappingViTSeg": (0.7812, 0.6638, 0.7546, 0.8510)},
    },
    ("CardiacUDA", "R"): {
        (8, 8):   {"ViT": (0.7798, 0.6530, 0.7521, 0.8452), "DualMappingViTSeg": (0.7859, 0.6689, 0.7590, 0.8550)},
        (16, 8):  {"ViT": (0.7765, 0.6483, 0.7490, 0.8424), "DualMappingViTSeg": (0.7827, 0.6641, 0.7562, 0.8520)},
        (32, 8):  {"ViT": (0.7718, 0.6431, 0.7446, 0.8383), "DualMappingViTSeg": (0.7782, 0.6591, 0.7518, 0.8481)},
        (16, 4):  {"ViT": (0.7615, 0.6320, 0.7335, 0.8295), "DualMappingViTSeg": (0.7677, 0.6478, 0.7406, 0.8392)},
        (16, 16): {"ViT": (0.7801, 0.6526, 0.7534, 0.8450), "DualMappingViTSeg": (0.7891, 0.6704, 0.7612, 0.8557)},
    },
}

CONFIG_ORDER = [(16, 8), (8, 8), (32, 8), (16, 4), (16, 16)]
MODEL_SHORT = {"DualMappingViTSeg": "DM ViT", "ViTSeg": "ViT"}
MODEL_RUN_SHORT = {"DualMappingViTSeg": "DMViT", "ViTSeg": "ViT"}
MODEL_PAPER_KEY = {"DualMappingViTSeg": "DualMappingViTSeg", "ViTSeg": "ViT"}


def run_name(dataset, site, model, patch, depth):
    return "{}{}_{}_p{}_d{}".format(dataset, "_{}".format(site) if site else "",
                                    MODEL_RUN_SHORT[model], patch, depth)


def load_ours(dataset, site, model, patch, depth):
    p = RUNS_ROOT / run_name(dataset, site, model, patch, depth) / "summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    t = s.get("test", {})
    return (t.get("mDice"), t.get("mIoU"), t.get("mSensitivity"), t.get("mPrecision"))


def fmt(vals, dec=4):
    return " / ".join("-" if v is None else f"{v:.{dec}f}" for v in vals)


def delta(ours, paper):
    out = []
    for o, p in zip(ours, paper):
        out.append("-" if o is None else f"{o - p:+.4f}")
    return " / ".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(REPO_ROOT / "segmentation" / "results" /
                                             "comparison_vs_paper.md"))
    args = parser.parse_args()

    lines = ["# Segmentation results vs draft-paper tables 5 / 6",
             "",
             "Metrics: mDice / mIoU / Sensitivity / Precision (foreground classes, test split).",
             "Ours: `segmentation/runs/<run>/summary.json`; paper: sn-article.pdf tables 5-6.",
             "",
             "**Caveat (hospital R)**: the public Kaggle release cannot reproduce the",
             "draft's R split 545/365/260 (`Site_R_73` ships images without labels).",
             "R runs use `UDAR_train`/`UDAR_val` (338/75 frames carved from `Site_R_126`)",
             "and the complete `Site_R_52` test split (260). See `docs/PAPER_MAPPING.md`.",
             ""]
    for (dataset, site), table in PAPER.items():
        title = dataset + (f" (hospital {site})" if site else "")
        lines += [f"## {title}", "",
                  "| Config (p/d) | Model | Paper | Ours | Delta |",
                  "|---|---|---|---|---|"]
        for patch, depth in CONFIG_ORDER:
            for model in ("DualMappingViTSeg", "ViTSeg"):
                paper = table.get((patch, depth), {}).get(MODEL_PAPER_KEY[model])
                ours = load_ours(dataset, site, model, patch, depth)
                tag = " **(default)**" if (patch, depth) == (16, 8) else ""
                lines.append("| {} | {} | {} | {} | {} |".format(
                    f"p{patch} d{depth}{tag}", MODEL_SHORT[model],
                    fmt(paper) if paper else "-",
                    fmt(ours) if ours else "PENDING",
                    delta(ours, paper) if (ours and paper) else "-"))
        lines.append("")
    text = "\n".join(lines)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
