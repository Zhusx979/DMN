#!/usr/bin/env python
# coding: utf-8
"""Grad-CAM comparison figure (TBX11K or BC): original / ViT / DMN per class.

Builds the manuscript-style 3-row panel (rows: original, ViT Grad-CAM,
DMN Grad-CAM; column groups: the classes x two examples) from the trained
checkpoints, and writes PNG + PDF.  Both datasets use the same style; the
only differences are the class names, the input pipeline (BC is 4-channel:
RGB + lesion-box mask, fixed test list) and the checkpoint directory.

Grad-CAM (Selvaraju et al.): channel weights = spatial mean of d logit_c/dA,
cam = ReLU(weighted sum), 98th-percentile normalized and bicubically resized
to the input size.  Hook = the last attention block's LayerNorm output; that
is the deepest token representation whose patch tokens still influence the
class token.  (The final post-LayerNorm patch tokens are a dead end under
cls pooling -- only the cls row reaches the head -- so their gradient is
exactly zero.)

Examples: test images that are in both models' test splits, classified
correctly by BOTH, and whose Grad-CAM is non-degenerate for both (>=5% of
tokens positive); the target is the ground-truth class for both heatmaps.
The heatmap is blended with an activation-proportional alpha so the
radiograph stays legible away from the hot regions.

Usage
-----
    python scripts/make_gradcam_figure.py --dataset tbx11k   # p32/d8 pair
    python scripts/make_gradcam_figure.py --dataset tbx11k --skip-per-class 2
    python scripts/make_gradcam_figure.py --dataset bc       # p16/d8 pair
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from configs.experiments import EXPERIMENTS  # noqa: E402
from datasets.classification.tbx11k import TBX11KDataset  # noqa: E402
from datasets.transforms import image_transforms as tr  # noqa: E402
from networks.vit import ViT  # noqa: E402
from networks.dual_mapping_network_classification import DualMappingViT  # noqa: E402

DATASETS = {
    "tbx11k": {
        "class_names": ["Healthy", "Sick", "Tuberculosis"],
        "class_labels": ["Healthy", "Sick", "TB"],   # short labels in figures
        "default_patch": 32,   # only configuration with both checkpoints
        "figure_stem": "tbx11k_gradcam",
    },
    "bc": {
        "class_names": ["Low risk", "Medium risk", "High risk"],
        "class_labels": ["Low", "Medium", "High"],
        "default_patch": 16,   # paper default (patch 16, depth 8)
        "figure_stem": "bc_gradcam",
    },
}
ROW_LABELS = ["Original", "ViT", "DMN"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MIN_POS = 0.05          # keep examples whose Grad-CAM is non-degenerate
ALPHA_MAX = 0.85        # peak alpha of the activation-proportional overlay


def setup_style():
    # Shared sans-serif stack.  Arial / Helvetica when the platform has them;
    # Liberation Sans is the metric-compatible Arial substitute on Linux, so
    # the fallback keeps the same text widths.  DejaVu Sans is the final
    # cross-platform fallback.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
    except ValueError:
        try:
            font_manager.findfont("Helvetica", fallback_to_default=False)
        except ValueError:
            plt.rcParams["font.sans-serif"].insert(2, "Liberation Sans")
    plt.rcParams.update({
        "font.size": 11.0,
        "axes.labelsize": 11.5,
        "axes.titlesize": 12.0,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    })
    # TrueType embedding keeps text editable in the PDF
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def build_models(exp, patch, depth):
    vit = ViT(image_size=exp["image_size"], patch_size=patch,
              channels=exp["in_channels"], num_classes=exp["num_classes"],
              dim=exp["dim"], depth=depth, heads=16, mlp_dim=exp["mlp_dim"],
              dropout=exp["dropout"], emb_dropout=exp["emb_dropout"])
    dmn = DualMappingViT(image_size=exp["image_size"], patch_size=patch,
                         channels=exp["in_channels"],
                         num_classes=exp["num_classes"], dim=exp["dim"],
                         depth=depth, mlp_dim=exp["mlp_dim"],
                         dropout=exp["dropout"], emb_dropout=exp["emb_dropout"])
    return vit, dmn


def load_checkpoint(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


class _FigureDataset(Dataset):
    """Uniform [image, label, path] view over the TBX11K and BC loaders."""

    def __init__(self, base, kind, seed=0):
        self.base = base
        self.kind = kind
        self.seed = int(seed)
        if kind == "bc":
            self.samples = [(str(p), int(l))
                            for p, l in zip(base.data_files, base.labels)]
        else:
            self.samples = [(str(p), int(l)) for p, l in base.samples]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        if self.kind == "bc":
            # CenterCropMask falls back to a random crop when a box is
            # missing/out of bounds.  Seed per index so ViT and DMN (and
            # repeated runs) see exactly the same test crop.
            random.seed(self.seed + idx)
            np.random.seed(self.seed + idx)
            sample = self.base[idx]
            image = sample["image"]
            if isinstance(image, np.ndarray):
                # BC's ToTensor returns a CHW numpy array; the training
                # DataLoader collates it to a tensor, so do the same here.
                image = torch.from_numpy(np.ascontiguousarray(image))
            return (image, int(sample["label"]), self.samples[idx][0])
        image, label, _ = self.base[idx]
        return image, int(label), self.samples[idx][0]


def test_dataset(dataset_key, exp, seed):
    if dataset_key == "bc":
        from datasets.classification.breast_cancer_mask import BCData
        from datasets.transforms import breast_cancer_mask_transforms as btr
        transform = transforms.Compose([
            btr.CenterCropMask(),   # crop around the lesion box, adds mask0
            btr.Resize(size=(exp["image_size"], exp["image_size"])),
            btr.Normalize(),        # RGB /255 + mask channel -> 4xHxW
            btr.ToTensor(),
        ])
        base = BCData(data_root=exp["data_root"],
                      data_txt=exp["label_dir"] + "/test.txt",
                      transform=transform)
    else:
        transform = transforms.Compose([
            tr.FixedResize([exp["image_size"], exp["image_size"]]),
            tr.Normalize(),
            tr.ToTensor(),
        ])
        base = TBX11KDataset(exp["data_root"], transform=transform,
                             mode="test", seed=seed)
    return _FigureDataset(base, dataset_key, seed=seed)


def predict(model, dataset, device, batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    out = {}
    with torch.no_grad():
        for imgs, labels, names in loader:
            logits = model(imgs.to(device))
            preds = logits.argmax(1).cpu()
            for name, label, pred in zip(names, labels, preds):
                out[name] = (int(label), int(pred))
    acc = np.mean([int(l == p) for l, p in out.values()])
    return out, acc


def grad_cam(model, tensor, target):
    """Grad-CAM over patch tokens of the last block's attention norm output.

    Returns (cam at input resolution in [0, 1], fraction of positive tokens).
    """
    acts = {}

    def hook(_module, _inp, out):
        out.retain_grad()
        acts["out"] = out

    handle = model.transformer.layers[-1][0].norm.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        logits[0, target].backward()
        tokens = acts["out"][0, 1:, :]          # patch tokens only
        grads = acts["out"].grad[0, 1:, :]
        raw = (tokens * grads.mean(dim=0)).sum(dim=-1)
        positive = float((raw > 0).float().mean())
        cam = raw.relu()
        side = int(round(cam.numel() ** 0.5))
        cam = cam.reshape(side, side)
        values = cam[cam > 0]
        scale = torch.quantile(values, 0.98) if values.numel() else cam.max()
        cam = torch.clamp(cam / (scale + 1e-12), 0.0, 1.0)
        cam = F.interpolate(cam[None, None],
                            size=(tensor.shape[-2], tensor.shape[-1]),
                            mode="bicubic", align_corners=False)
        return torch.clamp(cam[0, 0], 0.0, 1.0).detach().cpu().numpy(), positive
    finally:
        handle.remove()


def display_image(tensor, dataset_key):
    """Undo the dataset normalization -> RGB in [0, 1]."""
    arr = tensor.cpu().numpy().transpose(1, 2, 0)
    if dataset_key == "bc":
        # BC: RGB in [0, 1] plus one lesion-mask channel; show RGB only.
        return np.clip(arr[..., :3], 0.0, 1.0)
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(arr, 0, 255).astype(np.uint8) / 255.0


def select_examples(model_vit, model_dmn, dataset, pred_vit, pred_dmn, device,
                    dataset_key, class_names, n_per_class=2, cap=25,
                    min_pos=MIN_POS, skip_per_class=0):
    """Correct for both models, shared test split, non-degenerate Grad-CAMs."""
    index = {p: i for i, (p, _) in enumerate(dataset.samples)}
    common = set(pred_vit) & set(pred_dmn)
    chosen = []
    for cls in range(len(class_names)):
        cands = sorted(n for n in common
                       if pred_vit[n][0] == cls
                       and pred_vit[n][0] == pred_vit[n][1]
                       and pred_dmn[n][0] == pred_dmn[n][1])
        got = []
        usable = 0
        for name in cands[:cap]:
            tensor, label, _ = dataset[index[name]]
            x = tensor.unsqueeze(0).to(device)
            cam_v, pos_v = grad_cam(model_vit, x, int(label))
            cam_d, pos_d = grad_cam(model_dmn, x, int(label))
            if pos_v < min_pos or pos_d < min_pos:
                continue
            usable += 1
            if usable <= skip_per_class:
                print("  skip {}: {} (usable #{})".format(
                    class_names[cls], Path(name).name, usable))
                continue
            got.append({"name": name, "label": int(label),
                        "original": display_image(tensor, dataset_key),
                        "cam": [cam_v, cam_d],
                        "positive": [pos_v, pos_d]})
            print("  {}: {} (Grad-CAM positive: ViT {:.0%} / DMN {:.0%})".format(
                class_names[cls], Path(name).name, pos_v, pos_d))
            if len(got) == n_per_class:
                break
        if len(got) < n_per_class:
            raise SystemExit("class {}: only {} usable examples within the "
                             "first {} candidates".format(
                                 class_names[cls], len(got), cap))
        chosen.extend(got)
    return chosen


def build_figure(records, class_labels, out_png, out_pdf):
    setup_style()
    n_ex = len(records) // len(class_labels)   # examples per class
    S = 0.94           # panel side, inches
    gi = 0.045         # gap inside a class group
    gI = 1.9 * gi      # gap between class groups (1.5-2x the inner gap)
    g_row = 0.06
    left = 0.40        # rotated row-label margin
    right = 0.48       # colorbar + rotated label margin
    top = 0.38         # class-title band
    bottom = 0.12

    group_w = n_ex * S + (n_ex - 1) * gi
    content_w = len(class_labels) * group_w + (len(class_labels) - 1) * gI
    content_h = 3 * S + 2 * g_row
    width = left + content_w + right
    height = top + content_h + bottom

    fig = plt.figure(figsize=(width, height), facecolor="white")
    fig_w, fig_h = fig.get_size_inches()

    def add_img_axes(x0, y0):
        return fig.add_axes([x0 / fig_w, y0 / fig_h, S / fig_w, S / fig_h])

    def group_x(g):
        return left + g * (group_w + gI)

    # thin light separators between class groups
    for g in range(len(class_labels) - 1):
        xm = (group_x(g) + group_w + gI / 2.0) / fig_w
        fig.add_artist(Line2D([xm, xm],
                              [bottom / fig_h, (bottom + content_h) / fig_h],
                              color="#d4d4d4", linewidth=0.6, zorder=0))

    for row in range(len(ROW_LABELS)):
        for idx, rec in enumerate(records):
            g, k = idx // n_ex, idx % n_ex
            y_bottom = height - top - (row + 1) * S - row * g_row
            ax = add_img_axes(group_x(g) + k * (S + gi), y_bottom)
            ax.imshow(rec["original"], interpolation="bilinear")
            if row > 0:
                cam = rec["cam"][row - 1]
                ax.imshow(cam, cmap="jet", vmin=0.0, vmax=1.0,
                          alpha=np.clip(ALPHA_MAX * cam, 0.0, 1.0),
                          interpolation="bilinear")
            ax.set_axis_off()

    # row labels, rotated 90 deg counter-clockwise, close to the images
    for row, label in enumerate(ROW_LABELS):
        yc = height - top - (row + 0.5) * S - row * g_row
        fig.text((left - 0.16) / fig_w, yc / fig_h, label, ha="center",
                 va="center", rotation=90, fontsize=12.0, color="black")

    # class titles centred over the two sub-columns of each group
    for g, name in enumerate(class_labels):
        xc = (group_x(g) + group_w / 2.0) / fig_w
        fig.text(xc, (height - 0.165) / fig_h, name, ha="center", va="center",
                 fontsize=13.0, fontweight="bold", color="black")

    # shared colorbar aligned with the two Grad-CAM rows
    cax = fig.add_axes([
        (left + content_w + 0.15) / fig_w,
        bottom / fig_h,
        0.05 / fig_w,
        (2 * S + g_row) / fig_h,
    ])
    cbar = fig.colorbar(ScalarMappable(norm=Normalize(0.0, 1.0), cmap="jet"),
                        cax=cax)
    cbar.set_ticks([])
    cbar.outline.set_linewidth(0.5)
    cbar.set_label("Grad-CAM", rotation=90, fontsize=11.0, labelpad=3)

    # self-check: no text may be clipped by the canvas
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist, what in ([(t, "text %r" % t.get_text()) for t in fig.texts]
                         + [(cbar.ax.yaxis.label, "colorbar label")]):
        bb = artist.get_window_extent(renderer=renderer)
        if bb.x0 < -0.5 or bb.x1 > fig.bbox.width + 0.5 or \
                bb.y0 < -0.5 or bb.y1 > fig.bbox.height + 0.5:
            raise SystemExit("clipped {}".format(what))
    fig.savefig(out_png, dpi=600, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)
    print("wrote {} and {}".format(out_png, out_pdf))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS),
                        default="tbx11k", help="tbx11k (p32/d8) or bc (p16/d8)")
    parser.add_argument("--patch", type=int, default=None,
                        help="patch size; default: dataset-specific "
                             "(tbx11k 32, bc 16)")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--seed-vit", type=int, default=911,
                        help="TBX split seed (ignored for bc: fixed test.txt)")
    parser.add_argument("--seed-dmn", type=int, default=911,
                        help="TBX split seed (ignored for bc: fixed test.txt)")
    parser.add_argument("--vit-ckpt", default=None)
    parser.add_argument("--dmn-ckpt", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-per-class", type=int, default=0,
                        help="skip the first N usable examples per class "
                             "(to show a different batch)")
    parser.add_argument("--candidate-cap", type=int, default=40,
                        help="max candidates examined per class")
    parser.add_argument("--min-pos", type=float, default=MIN_POS,
                        help="minimum fraction of positive CAM tokens for "
                             "both models (default 0.05)")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    patch = args.patch if args.patch is not None else cfg["default_patch"]
    exp = EXPERIMENTS[args.dataset]
    suffix = "AdamW_depth_{}_patch_{}".format(args.depth, patch)
    vit_ckpt = Path(args.vit_ckpt or REPO_ROOT / "ckpts" / args.dataset /
                    "ViT" / suffix / "ckpt_best.pth")
    dmn_ckpt = Path(args.dmn_ckpt or REPO_ROOT / "ckpts" / args.dataset /
                    "DualMappingViT" / suffix / "ckpt_best.pth")
    out_dir = Path(args.out_dir or REPO_ROOT / "figures" /
                   "{}_gradcam".format(args.dataset))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or "p{}_d{}".format(patch, args.depth)
    device = torch.device(args.device)

    print("checkpoints:")
    print("  ViT:", vit_ckpt)
    print("  DMN:", dmn_ckpt)
    vit, dmn = build_models(exp, patch, args.depth)
    load_checkpoint(vit, vit_ckpt).to(device)
    load_checkpoint(dmn, dmn_ckpt).to(device)

    ds_vit = test_dataset(args.dataset, exp, args.seed_vit)
    ds_dmn = (ds_vit if args.dataset == "bc"
              else test_dataset(args.dataset, exp, args.seed_dmn))
    pred_vit, acc_vit = predict(vit, ds_vit, device)
    pred_dmn, acc_dmn = predict(dmn, ds_dmn, device)
    print("test accuracy: ViT {:.4f} ({} imgs), DMN {:.4f} ({} imgs)".format(
        acc_vit, len(ds_vit), acc_dmn, len(ds_dmn)))

    records = select_examples(vit, dmn, ds_vit, pred_vit, pred_dmn, device,
                              args.dataset, cfg["class_names"],
                              cap=args.candidate_cap,
                              min_pos=args.min_pos,
                              skip_per_class=args.skip_per_class)
    print("selected:", [Path(r["name"]).name for r in records])

    out_png = out_dir / "{}_{}.png".format(cfg["figure_stem"], tag)
    out_pdf = out_dir / "{}_{}.pdf".format(cfg["figure_stem"], tag)
    build_figure(records, cfg["class_labels"], out_png, out_pdf)

    meta = {
        "dataset": args.dataset,
        "config": {"patch_size": patch, "depth": args.depth,
                   "split_seed_vit": args.seed_vit,
                   "split_seed_dmn": args.seed_dmn,
                   "split": ("fixed label_classification/test.txt"
                             if args.dataset == "bc"
                             else "seed-split TBX11K test set")},
        "checkpoints": {"vit": str(vit_ckpt), "dmn": str(dmn_ckpt)},
        "test_acc": {"ViT": acc_vit, "DMN": acc_dmn},
        "examples": [{"file": Path(r["name"]).name,
                      "class": cfg["class_names"][r["label"]],
                      "gradcam_positive": {"ViT": r["positive"][0],
                                           "DMN": r["positive"][1]}}
                     for r in records],
        "gradcam": "Selvaraju et al.; hook = last attention block LayerNorm "
                   "output (patch tokens), target = ground-truth class, "
                   "98th-percentile normalization",
        "overlay": "jet, alpha = {:.2f} * cam".format(ALPHA_MAX),
        "selection": "correct for both models, shared test split, Grad-CAM "
                     "positive fraction >= {:.0%} for both{}".format(
                         args.min_pos,
                         "" if args.skip_per_class == 0 else
                         "; first {} usable per class skipped".format(
                             args.skip_per_class)),
    }
    with open(out_dir / "selection_{}.json".format(tag), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta["examples"], indent=2))


if __name__ == "__main__":
    main()
