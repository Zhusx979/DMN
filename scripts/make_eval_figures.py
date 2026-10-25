#!/usr/bin/env python
# coding: utf-8
"""Test-set analysis figures for ViT vs DMN (TBX11K or BC, same checkpoints).

Two publication-style figures sharing the Grad-CAM figure's sans-serif style
(Arial / Helvetica with a metric-compatible fallback) and one colour language:

1. ``confusion_<tag>`` -- two square confusion matrices (ViT, DMN).  Cell
   colour is the row-normalized fraction and each cell prints the raw count;
   the most frequent confusion is boxed.

2. ``roc_pr_<tag>`` -- one-vs-rest ROC and precision-recall curves.  Colour
   encodes the class, line style the model (solid DMN, dashed ViT); one
   shared legend replaces per-curve entries.  Curves are PCHIP-smoothed for
   readability (metric values in ``metrics_<tag>.json`` are computed from the
   raw curves), and each panel zooms its informative corner in an inset.

Both canvases follow the golden ratio.  ``--dataset bc`` uses the same style
with the BC class names, the fixed BC test list and the 4-channel input.

Usage
-----
    python scripts/make_eval_figures.py --dataset tbx11k
    python scripts/make_eval_figures.py --dataset bc --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import ConnectionPatch, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score, confusion_matrix, precision_recall_curve,
    precision_recall_fscore_support, roc_curve, auc)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import make_gradcam_figure as gc  # noqa: E402

DATASET_KEY = "tbx11k"
CLASS_NAMES = []
CLASS_LABELS = []
N_CLASSES = 0
# Okabe-Ito palette (colour-blind safe); one colour per class
CLASS_COLORS = ["#0072B2", "#D55E00", "#009E73"]
MODEL_STYLES = {"DMN": dict(linestyle="-", linewidth=1.45),
                "ViT": dict(linestyle=(0, (3.6, 1.6)), linewidth=1.35)}
GOLDEN = 1.618

# sequential teal -> indigo ramp for the row-normalized confusion fractions
CONF_CMAP = LinearSegmentedColormap.from_list(
    "teal_indigo",
    ["#FBFFFE", "#D3EDE8", "#8FD0C4", "#47A99B", "#24707E", "#16324F"])


def configure(dataset_key):
    global DATASET_KEY, CLASS_NAMES, CLASS_LABELS, N_CLASSES
    cfg = gc.DATASETS[dataset_key]
    DATASET_KEY = dataset_key
    CLASS_NAMES = list(cfg["class_names"])
    CLASS_LABELS = list(cfg["class_labels"])
    N_CLASSES = len(CLASS_NAMES)
    if N_CLASSES > len(CLASS_COLORS):
        raise SystemExit("no colour assigned for {} classes".format(N_CLASSES))


def predict_proba(model, dataset, device, batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    ys, ps = [], []
    with torch.no_grad():
        for imgs, labels, _ in loader:
            probs = torch.softmax(model(imgs.to(device)), dim=1).cpu().numpy()
            ys.append(labels.numpy())
            ps.append(probs)
    return np.concatenate(ys), np.vstack(ps)


def analyse(y_true, probs):
    y_pred = probs.argmax(1)
    cm = confusion_matrix(y_true, y_pred, labels=range(N_CLASSES))
    # sklearn returns (precision, recall, f1, support) in this order
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(N_CLASSES), zero_division=0)
    accuracy = float((y_pred == y_true).mean())

    roc, pr = {}, {}
    for c in range(N_CLASSES):
        binary = (y_true == c).astype(int)
        fpr, tpr, _ = roc_curve(binary, probs[:, c])
        roc[c] = {"fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr))}
        prec, rec, _ = precision_recall_curve(binary, probs[:, c])
        pr[c] = {"precision": prec, "recall": rec,
                 "ap": float(average_precision_score(binary, probs[:, c]))}

    off = cm.copy()
    np.fill_diagonal(off, -1)
    a, b = np.unravel_index(np.argmax(off), off.shape)
    return {
        "accuracy": accuracy,
        "confusion": cm,
        "recall": recall, "precision": precision, "f1": f1,
        "support": support,
        "roc": roc, "pr": pr,
        "macro_auc": float(np.mean([roc[c]["auc"] for c in range(N_CLASSES)])),
        "macro_ap": float(np.mean([pr[c]["ap"] for c in range(N_CLASSES)])),
        "most_confused": {"true": CLASS_NAMES[a], "predicted": CLASS_NAMES[b],
                          "count": int(off[a, b]),
                          "row_total": int(cm[a].sum())},
    }


def _check_canvas(fig, artists):
    """Raise if any text artist is clipped by the canvas."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist, what in artists:
        bb = artist.get_window_extent(renderer=renderer)
        if bb.x0 < -0.5 or bb.x1 > fig.bbox.width + 0.5 or \
                bb.y0 < -0.5 or bb.y1 > fig.bbox.height + 0.5:
            raise SystemExit("clipped {}: {}".format(what, bb))


def _panel_header(fig, fig_w, fig_h, x_in, y_in, letter, title):
    """Bold panel letter plus a short panel name, above the axes."""
    fig.text(x_in / fig_w, y_in / fig_h, letter, ha="left", va="baseline",
             fontsize=12.5, fontweight="bold", color="black")
    fig.text((x_in + 0.21) / fig_w, y_in / fig_h, title, ha="left",
             va="baseline", fontsize=11.5, color="#222222")


def draw_confusion(results, out_png, out_pdf):
    """Two square confusion matrices with counts in the cells."""
    gc.setup_style()
    M = 2.60            # matrix side, inches (square subplots)
    GAP_BLOCKS = 0.90
    LEFT, RIGHT = 0.95, 0.18
    width = LEFT + 2 * M + GAP_BLOCKS + RIGHT
    height = width / GOLDEN
    assert abs(width / height - GOLDEN) < 1e-6

    fig = plt.figure(figsize=(width, height), facecolor="white")
    fig_w, fig_h = fig.get_size_inches()
    norm = Normalize(0.0, 1.0)

    top_pad = 0.55
    matrix_top = height - top_pad
    matrix_bottom = matrix_top - M

    for bi, (model_name, res) in enumerate(results):
        x0 = LEFT + bi * (M + GAP_BLOCKS)
        cm = res["confusion"]
        row_tot = np.maximum(cm.sum(axis=1), 1)
        frac = cm / row_tot[:, None]

        ax = fig.add_axes([x0 / fig_w, matrix_bottom / fig_h,
                           M / fig_w, M / fig_h])
        ax.imshow(frac, cmap=CONF_CMAP, norm=norm, interpolation="nearest",
                  origin="upper", extent=(-0.5, N_CLASSES - 0.5,
                                          N_CLASSES - 0.5, -0.5),
                  aspect="equal", zorder=1)
        ax.set_xlim(-0.5, N_CLASSES - 0.5)
        ax.set_ylim(N_CLASSES - 0.5, -0.5)
        for x in np.arange(-0.5, N_CLASSES, 1.0):
            ax.axvline(x, color="white", linewidth=1.4, zorder=2)
            ax.axhline(x, color="white", linewidth=1.4, zorder=2)
        ax.add_patch(Rectangle((-0.5, -0.5), N_CLASSES, N_CLASSES, fill=False,
                               edgecolor="#DDDDDD", linewidth=0.6, zorder=3,
                               clip_on=False))

        for r in range(N_CLASSES):
            for c in range(N_CLASSES):
                count = int(cm[r, c])
                color = ("#9C9C9C" if count == 0 else
                         "white" if frac[r, c] > 0.55 else "#1A1A1A")
                ax.text(c, r, str(count), ha="center", va="center",
                        fontsize=10.5, color=color, zorder=5)

        ax.set_xticks(range(N_CLASSES))
        ax.set_xticklabels(CLASS_LABELS, fontsize=10.5)
        ax.set_yticks(range(N_CLASSES))
        ax.set_yticklabels(CLASS_LABELS, fontsize=10.5)
        ax.tick_params(length=0, pad=3)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # box the most frequent confusion
        mc = res["most_confused"]
        r = CLASS_NAMES.index(mc["true"])
        c = CLASS_NAMES.index(mc["predicted"])
        ax.add_patch(Rectangle((c - 0.45, r - 0.45), 0.90, 0.90, fill=False,
                               edgecolor="#C0392B", linewidth=1.1,
                               linestyle=(0, (3, 2)), zorder=4))

        _panel_header(fig, fig_w, fig_h, x0 - 0.20, matrix_top + 0.14,
                      "ab"[bi], model_name)

    # one axis label per direction (the two panels share the same axes)
    fig.text(0.22 / fig_w, (matrix_top + matrix_bottom) / 2.0 / fig_h,
             "True class", ha="center", va="center", rotation=90,
             fontsize=11.0)
    fig.text((LEFT + 2 * M + GAP_BLOCKS) / 2.0 / fig_w, 0.90 / fig_h,
             "Predicted class", ha="center", va="bottom", fontsize=11.0)

    # shared colourbar under both matrices --------------------------------
    cbar_w, cbar_h, cbar_y = 2.60, 0.07, 0.50
    cax = fig.add_axes([(LEFT + 2 * M + GAP_BLOCKS - cbar_w) / 2.0 / fig_w,
                        cbar_y / fig_h, cbar_w / fig_w, cbar_h / fig_h])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=CONF_CMAP), cax=cax,
                        orientation="horizontal")
    cbar.set_ticks([])
    cbar.outline.set_linewidth(0.5)
    fig.text(0.5 * (LEFT + 2 * M + GAP_BLOCKS) / fig_w,
             (cbar_y - 0.07) / fig_h, "Row-normalized fraction",
             ha="center", va="top", fontsize=10.0, color="#555555")

    artists = [(t, "text %r" % t.get_text()) for t in fig.texts]
    artists += [(cbar.outline, "colorbar outline")]
    _check_canvas(fig, artists)

    fig.savefig(out_png, dpi=600, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)
    print("wrote {} and {} ({:.2f} x {:.2f} in, aspect {:.3f})".format(
        out_png, out_pdf, width, height, width / height))


def _envelope(x, y):
    """Strictly increasing x with the max y at each x (removes the staircase)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    ux, inv = np.unique(x, return_inverse=True)
    uy = np.full(ux.shape, -np.inf)
    np.maximum.at(uy, inv, y)
    return ux, uy


def smooth_curve(x, y, metric, n=400):
    """PCHIP interpolation of the upper envelope of a ROC or PR curve."""
    ux, uy = _envelope(x, y)
    if ux.size < 2:
        return ux, uy

    if metric == "roc":
        # keep the origin and the (near) vertical rise at FPR = 0
        pts_x, pts_y = [0.0], [0.0]
        if ux[0] <= 1e-9:
            pts_x.append(1e-4)
            pts_y.append(uy[0])
            ux, uy = ux[1:], uy[1:]
        px = np.r_[pts_x, ux]
        py = np.r_[pts_y, uy]
        if px[-1] < 1.0 - 1e-9:
            px = np.r_[px, 1.0]
            py = np.r_[py, 1.0]
    else:
        px, py = [0.0], [1.0]
        keep = ux > 1e-9
        px = np.r_[px, ux[keep]]
        py = np.r_[py, uy[keep]]
        if px[-1] < 1.0 - 1e-9:
            px = np.r_[px, 1.0]
            py = np.r_[py, py[-1]]

    xs = np.linspace(px[0], px[-1], n)
    ys = np.clip(PchipInterpolator(px, py)(xs), 0.0, 1.0)
    return xs, ys


def _nice_ticks(lo, hi=1.0, max_ticks=5):
    """Tick values inside [lo, hi] with the largest step that keeps <= max."""
    for step in (0.1, 0.2, 0.25, 0.5):
        start = np.ceil(lo / step) * step
        ticks = np.arange(start, hi + 1e-9, step)
        if len(ticks) <= max_ticks:
            return [round(float(v), 2) for v in ticks]
    return [round(float(lo), 3), 1.0]


def _auto_zoom(curves, metric):
    """Zoom window fitted to the actual curves (used for non-TBX datasets).

    The TBX figures use fixed windows tuned for their high accuracy; BC is a
    harder three-class problem, so the inset follows the curves instead of
    assuming TPR/precision above 0.75.
    """
    if metric == "roc":
        vals = [y for xs, ys, _, _ in curves
                for x, y in zip(xs, ys) if 0.02 <= x <= 0.20]
        x0, x1 = 0.0, 0.20
        xticks = [0.0, 0.05, 0.10, 0.15, 0.20]
        inset, link = (0.54, 0.14, 0.42, 0.30), (1, 0)
    else:
        vals = [y for xs, ys, _, _ in curves
                for x, y in zip(xs, ys) if x >= 0.70]
        x0, x1 = 0.70, 1.0
        xticks = [0.7, 0.8, 0.9, 1.0]
        inset, link = (0.14, 0.14, 0.40, 0.28), (0, 1)
    y0 = float(np.percentile(vals, 25)) - 0.05 if vals else 0.0
    y0 = float(np.floor(min(max(y0, 0.0), 0.90) * 10) / 10)
    return dict(box=(x0, y0, x1, 1.0), inset=inset, xlim=(x0, x1),
                ylim=(y0, 1.0), xticks=xticks,
                yticks=_nice_ticks(y0), link=link)


def draw_curves(results, out_png, out_pdf):
    """One-vs-rest ROC and PR curves; colour = class, style = model."""
    gc.setup_style()
    panel_w = panel_h = 3.05
    left, gap, right = 0.70, 0.50, 0.16
    width = left + 2 * panel_w + gap + right
    height = width / GOLDEN
    assert abs(width / height - GOLDEN) < 1e-6

    fig = plt.figure(figsize=(width, height), facecolor="white")
    fig_w, fig_h = fig.get_size_inches()
    top_pad = 0.50
    panel_top = height - top_pad

    # TBX uses fixed windows tuned for its high accuracy, so the delivered
    # figures stay byte-for-byte reproducible.
    tbx_zoom = {
        "roc": dict(box=(0.0, 0.60, 0.15, 1.0),
                    inset=(0.54, 0.14, 0.42, 0.30),
                    xlim=(0.0, 0.15), ylim=(0.55, 1.0),
                    xticks=[0.0, 0.05, 0.10, 0.15],
                    yticks=[0.6, 0.7, 0.8, 0.9, 1.0],
                    link=(1, 0)),
        "pr": dict(box=(0.70, 0.78, 1.0, 1.0),
                   inset=(0.14, 0.14, 0.40, 0.28),
                   xlim=(0.70, 1.0), ylim=(0.75, 1.0),
                   xticks=[0.7, 0.8, 0.9, 1.0],
                   yticks=[0.8, 0.9, 1.0],
                   link=(0, 1)),
    }
    fmt = FuncFormatter(lambda v, _: "{:g}".format(round(v, 2)))

    axes = []
    for pi, metric in enumerate(("roc", "pr")):
        x0 = left + pi * (panel_w + gap)
        ax = fig.add_axes([x0 / fig_w, (panel_top - panel_h) / fig_h,
                           panel_w / fig_w, panel_h / fig_h])
        axes.append(ax)

        curves = []
        for c in range(N_CLASSES):
            for model_name, res in results:
                data = res[metric][c]
                if metric == "roc":
                    xs, ys = smooth_curve(data["fpr"], data["tpr"], "roc")
                else:
                    xs, ys = smooth_curve(data["recall"], data["precision"],
                                          "pr")
                curves.append((xs, ys, c, model_name))
                ax.plot(xs, ys, color=CLASS_COLORS[c],
                        **MODEL_STYLES[model_name], alpha=0.95,
                        solid_capstyle="round")

        if metric == "roc":
            ax.plot([0, 1], [0, 1], color="0.62", linestyle=":",
                    linewidth=0.9)
            ax.set_xlabel("False positive rate", fontsize=11.0)
            ax.set_ylabel("True positive rate", fontsize=11.0)
        else:
            ax.axhline(1.0 / N_CLASSES, color="0.62", linestyle=":",
                       linewidth=0.9)
            ax.set_xlabel("Recall", fontsize=11.0)
            ax.set_ylabel("Precision", fontsize=11.0)

        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.tick_params(labelsize=10.5, length=2.5, width=0.6, pad=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)

        # zoom box on the main axes, inset in the empty lower corner
        z = tbx_zoom[metric] if DATASET_KEY == "tbx11k" else _auto_zoom(
            curves, metric)
        bx0, by0, bx1, by1 = z["box"]
        ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False,
                               edgecolor="#444444", linewidth=1.0,
                               linestyle=(0, (3, 2)), zorder=6))
        ia = ax.inset_axes(list(z["inset"]), zorder=7)
        ia.set_facecolor("white")
        for xs, ys, c, model_name in curves:
            style = dict(MODEL_STYLES[model_name])
            style["linewidth"] = 1.0
            ia.plot(xs, ys, color=CLASS_COLORS[c], alpha=0.95, **style)
        ia.set_xlim(*z["xlim"])
        ia.set_ylim(*z["ylim"])
        ia.set_xticks(z["xticks"])
        ia.set_yticks(z["yticks"])
        ia.xaxis.set_major_formatter(fmt)
        ia.yaxis.set_major_formatter(fmt)
        ia.tick_params(labelsize=9.5, length=2.0, width=0.5, pad=2)
        for side in ("top", "right"):
            ia.spines[side].set_visible(False)
        ia.spines["left"].set_linewidth(0.5)
        ia.spines["bottom"].set_linewidth(0.5)

        lx, ly = z["link"]
        fig.add_artist(ConnectionPatch(
            xyA=(bx0 if lx == 0 else bx1, by0), coordsA=ax.transData,
            xyB=(1 if lx == 0 else 0, 1), coordsB=ia.transAxes,
            color="#777777", linewidth=0.9, linestyle=(0, (2.5, 2)),
            zorder=6, clip_on=False))

        _panel_header(fig, fig_w, fig_h, x0 - 0.18, panel_top + 0.14,
                      "ab"[pi], ("ROC", "Precision\u2013Recall")[pi])

    # one shared legend: colours = classes, black line styles = models
    handles = [Line2D([], [], color=CLASS_COLORS[c], linewidth=1.6,
                      label=CLASS_LABELS[c]) for c in range(N_CLASSES)]
    handles += [
        Line2D([], [], color="0.20", linewidth=1.45, linestyle="-",
               label="DMN"),
        Line2D([], [], color="0.20", linewidth=1.35,
               linestyle=(0, (3.6, 1.6)), label="ViT"),
    ]
    legend = fig.legend(handles=handles, loc="lower center",
                        bbox_to_anchor=(0.5, 0.05), ncol=N_CLASSES + 2,
                        frameon=False,
                        handlelength=1.9, columnspacing=1.3,
                        handletextpad=0.5, fontsize=10.5)

    artists = [(t, "text %r" % t.get_text()) for t in fig.texts]
    artists += [(legend, "legend")]
    for ax in axes:
        artists += [(ax.xaxis.label, "xlabel"), (ax.yaxis.label, "ylabel")]
    _check_canvas(fig, artists)

    fig.savefig(out_png, dpi=600, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)
    print("wrote {} and {} ({:.2f} x {:.2f} in, aspect {:.3f})".format(
        out_png, out_pdf, width, height, width / height))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(gc.DATASETS),
                        default="tbx11k", help="tbx11k (p32/d8) or bc (p16/d8)")
    parser.add_argument("--patch", type=int, default=None,
                        help="patch size; default: dataset-specific "
                             "(tbx11k 32, bc 16)")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=911,
                        help="TBX split seed (ignored for bc: fixed test.txt)")
    parser.add_argument("--vit-ckpt", default=None)
    parser.add_argument("--dmn-ckpt", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    configure(args.dataset)
    cfg = gc.DATASETS[args.dataset]
    patch = args.patch if args.patch is not None else cfg["default_patch"]
    exp = gc.EXPERIMENTS[args.dataset]
    suffix = "AdamW_depth_{}_patch_{}".format(args.depth, patch)
    ckpts = {
        "ViT": Path(args.vit_ckpt or REPO_ROOT / "ckpts" / args.dataset /
                    "ViT" / suffix / "ckpt_best.pth"),
        "DMN": Path(args.dmn_ckpt or REPO_ROOT / "ckpts" / args.dataset /
                    "DualMappingViT" / suffix / "ckpt_best.pth"),
    }
    out_dir = Path(args.out_dir or REPO_ROOT / "figures" /
                   "{}_eval".format(args.dataset))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or "p{}_d{}".format(patch, args.depth)
    device = torch.device(args.device)

    vit, dmn = gc.build_models(exp, patch, args.depth)
    gc.load_checkpoint(vit, ckpts["ViT"]).to(device)
    gc.load_checkpoint(dmn, ckpts["DMN"]).to(device)
    dataset = gc.test_dataset(args.dataset, exp, args.seed)

    results = []
    summary = {}
    for model_name, model in (("ViT", vit), ("DMN", dmn)):
        y_true, probs = predict_proba(model, dataset, device)
        res = analyse(y_true, probs)
        results.append((model_name, res))
        summary[model_name] = {
            "accuracy": res["accuracy"],
            "confusion": res["confusion"].tolist(),
            "per_class": {
                CLASS_NAMES[c]: {
                    "recall": float(res["recall"][c]),
                    "precision": float(res["precision"][c]),
                    "f1": float(res["f1"][c]),
                    "support": int(res["support"][c]),
                    "auc": res["roc"][c]["auc"],
                    "ap": res["pr"][c]["ap"],
                } for c in range(N_CLASSES)
            },
            "macro_auc": res["macro_auc"],
            "macro_ap": res["macro_ap"],
            "most_confused": res["most_confused"],
        }
        print("{}: acc {:.4f} | most confused {} -> {} ({}/{})".format(
            model_name, res["accuracy"], res["most_confused"]["true"],
            res["most_confused"]["predicted"], res["most_confused"]["count"],
            res["most_confused"]["row_total"]))

    draw_confusion(results, out_dir / "confusion_{}.png".format(tag),
                   out_dir / "confusion_{}.pdf".format(tag))
    draw_curves(results, out_dir / "roc_pr_{}.png".format(tag),
                out_dir / "roc_pr_{}.pdf".format(tag))

    meta = {
        "dataset": args.dataset,
        "config": {"patch_size": patch, "depth": args.depth,
                   "split_seed": args.seed,
                   "split": ("fixed label_classification/test.txt"
                             if args.dataset == "bc"
                             else "seed-split TBX11K test set")},
        "checkpoints": {k: str(v) for k, v in ckpts.items()},
        "metrics": summary,
        "notes": "probabilities = softmax over the {} logits; ROC/PR are "
                 "one-vs-rest; AP = average precision (step-wise PR AUC); "
                 "figure curves are PCHIP interpolations of the raw curves."
                 .format(N_CLASSES),
    }
    with open(out_dir / "metrics_{}.json".format(tag), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({m: {"acc": round(s["accuracy"], 4),
                          "macro_auc": round(s["macro_auc"], 4),
                          "macro_ap": round(s["macro_ap"], 4)}
                      for m, s in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
