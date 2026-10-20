#!/usr/bin/env python
# coding: utf-8
"""Segmentation training with interruption-safe resume (draft-paper protocol).

Model: DM ViT segmentation network = 8-layer ViT-style encoder + lightweight
convolutional decoder (networks/dual_mapping_network_segmentation.py);
baselines ViTSeg / Unet are available for comparison.

Protocol (pre-publication draft, Sec. 4.2 -- tables 5/6):
    * Adam, lr 1e-3, weight decay 1e-4, momentum 0.9 (== Adam beta1),
      batch size 16, 600 epochs;
    * augmentation: resize 384 -> random crop 256 + horizontal/vertical flips;
    * inference: resize 384 -> center crop 256;
    * loss: Dice + cross entropy (monai DiceCELoss); DM models additionally
      back-propagate the constraint regularizer L_reg (Eqs. 9-10);
    * model selection on the validation split (highest mIoU), metrics
      reported on the test split (foreground classes only, as in the paper).

Everything is organized per experiment under ``segmentation/runs/<run_name>/``::

    <run_name>/config.json      resolved configuration (dataset/model/patch/depth/...)
    <run_name>/train.log        loguru log of every epoch
    <run_name>/metrics.csv      one row per epoch (losses, val metrics, best flag)
    <run_name>/ckpt_last.pth    latest epoch: model+optimizer+RNG (atomic save)
    <run_name>/ckpt_best.pth    best validation mIoU so far
    <run_name>/test_metrics.json / summary.json   final test evaluation
    <run_name>/overlays/        qualitative test overlays

Interruption safety: re-running the exact same command resumes automatically
from ``ckpt_last.pth`` (nothing is lost except the current epoch); SIGTERM /
SIGINT finish and save the current epoch before exiting.  Epoch shuffling is
seeded with ``seed + epoch``, so a resumed run sees the same sample order as
an uninterrupted run.  Use ``--fresh`` to deliberately restart a run.

Run from the repository root:
    python segmentation/train_segmentation.py --dataset BRISC --model DualMappingViTSeg
    python segmentation/train_segmentation.py --dataset CardiacUDA --site G --model ViTSeg
"""

import argparse
import csv
import datetime
import json
import os
import random
import signal
import socket
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from configs.experiments import SEGMENTATION_BRISC, SEGMENTATION_CARDIAC_UDA  # noqa: E402
from networks.dual_mapping_network_segmentation import DualMappingViTSeg  # noqa: F401,E402
from datasets.segmentation import brisc, cardiac_uda  # noqa: E402
from segmentation.model_factory import MODELS, build_model  # noqa: E402
from networks.synthesizer_vit import VARIANTS as SYN_VARIANTS  # noqa: E402
from utils.segmentation import (medical_multi_class_metrics, print_metrics,  # noqa: E402
                                set_seed)
from utils import efficiency  # noqa: E402

from loguru import logger  # noqa: E402
from monai.losses import DiceCELoss  # noqa: E402

EXPERIMENTS = {"BRISC": SEGMENTATION_BRISC, "CardiacUDA": SEGMENTATION_CARDIAC_UDA}
MODEL_SHORT = {"DualMappingViTSeg": "DMViT", "ViTSeg": "ViT", "Unet": "UNet"}
SYN_TAG = {"random": "SynRandom", "random_fixed": "SynRandomFixed",
           "dense": "SynDense", "factorized_random": "SynFactorized",
           "mix": "SynMix", "static": "SynStatic"}
RUNS_ROOT = REPO_ROOT / "segmentation" / "runs"

_STOP = False


def _handle_stop(signum, frame):
    global _STOP
    _STOP = True
    print(f"\n[signal {signum}] finishing the current epoch, then saving and exiting...")


def build_loaders(dataset, site, exp, batch_size, num_workers):
    if dataset == "BRISC":
        return brisc.build_dataloaders(root=exp["data_root"], batch_size=batch_size,
                                       num_workers=num_workers, seed=exp["seed"])
    if dataset == "CardiacUDA":
        return cardiac_uda.build_dataloaders(site=site, exp=exp, batch_size=batch_size,
                                             num_workers=num_workers, seed=exp["seed"])
    raise ValueError("Unknown dataset: {}".format(dataset))


def seed_epoch(train_loader, seed, epoch):
    """Deterministic per-epoch RNG (== resume reproduces the same shuffling)."""
    s = seed + epoch
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    gen = getattr(train_loader, "generator", None)
    if gen is not None:
        gen.manual_seed(s)


def train_one_epoch(model, loader, criterion, optimizer, device, exp):
    model.train()
    losses, reg_losses, metrics = [], [], []
    from tqdm import tqdm
    for input, label in tqdm(loader, desc="Training"):
        input = input.to(device).float()
        label = label.to(device).long().unsqueeze(1)
        output = model(input)
        seg_loss = criterion(output, label)

        if hasattr(model, "regularization_loss"):  # DM constraint regularizer
            reg_loss = model.regularization_loss()
        else:
            reg_loss = torch.zeros((), device=device)
        total_loss = seg_loss + exp["reg_weight"] * reg_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        losses.append(seg_loss.item())
        reg_losses.append(float(reg_loss.detach()))
        preds = output.argmax(dim=1)
        metrics.append(medical_multi_class_metrics(
            label, preds, num_classes=exp["num_classes"], include_background=False))
    return np.mean(losses), np.mean(reg_losses), metrics


def evaluate(model, loader, criterion, device, exp):
    model.eval()
    losses, metrics = [], []
    with torch.no_grad():
        for input, label in loader:
            input = input.to(device).float()
            label = label.to(device).long().unsqueeze(1)
            output = model(input)
            losses.append(criterion(output, label).item())
            preds = output.argmax(dim=1)
            metrics.append(medical_multi_class_metrics(
                label, preds, num_classes=exp["num_classes"], include_background=False))
    return np.mean(losses), metrics


def mean_metrics(metrics):
    """Average per-image metric dicts; per-class lists are averaged element-wise."""
    out = {}
    for k, v in metrics[0].items():
        if isinstance(v, (int, float)):
            out[k] = float(np.mean([m[k] for m in metrics]))
        else:  # per-class lists (Dice_per_class, IoU_per_class, ...)
            out[k] = np.mean(np.array([m[k] for m in metrics], dtype=np.float64), axis=0).tolist()
    return out


def save_multiclass_overlay(image_tensor, pred, gt, path, imagenet_stats=(0.485, 0.456, 0.406),
                            imagenet_std=(0.229, 0.224, 0.225)):
    """gray | GT colored | prediction colored, for one test frame."""
    import cv2
    from utils.segmentation import COLORS

    img = image_tensor.detach().cpu().float()
    if img.dim() == 3:
        if img.shape[0] == 1:
            img = img.squeeze(0)
        elif img.shape[0] == 3:
            mean = torch.tensor(imagenet_stats).view(3, 1, 1)
            std = torch.tensor(imagenet_std).view(3, 1, 1)
            img = (img * std + mean).mean(dim=0)
    img = img.clamp(0, 1).numpy()
    gray = (img * 255).astype(np.uint8)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def colorize(label):
        label = np.asarray(label).astype(np.uint8)
        out = gray.copy()
        for cls in range(1, min(len(COLORS), int(label.max()) + 1)):
            out[label == cls] = COLORS[cls]
        return out

    sep = np.full((gray.shape[0], 10, 3), 255, dtype=np.uint8)
    combined = np.hstack([gray, sep, colorize(gt), sep, colorize(pred)])
    cv2.imwrite(str(path), combined)


def atomic_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_last(ckpt_path, epoch, model, optimizer, best_miou, best_epoch, exp):
    atomic_save({
        "epoch": epoch,                     # last fully completed epoch
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_miou": best_miou,
        "best_epoch": best_epoch,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "num_classes": exp["num_classes"],
    }, ckpt_path)


def append_csv(csv_path, row, header=None):
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def acquire_lock(run_dir):
    lock = run_dir / "run.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().split()[0])
            os.kill(pid, 0)  # alive?
            raise SystemExit(
                f"another training process (pid {pid}) is already running in {run_dir}")
        except (ProcessLookupError, ValueError):
            pass  # stale lock
    lock.write_text(f"{os.getpid()} {socket.gethostname()} "
                    f"{datetime.datetime.now().isoformat()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="BRISC", choices=list(EXPERIMENTS))
    parser.add_argument("--model", default="DualMappingViTSeg", choices=MODELS)
    parser.add_argument("--variant", default="random", choices=list(SYN_VARIANTS),
                        help="synthesizer variant (SynthesizerViTSeg only)")
    parser.add_argument("--site", default=None, choices=["G", "R"],
                        help="hospital site for CardiacUDA (separate runs per site)")
    parser.add_argument("--patch_size", type=int, default=None, help="override patch size (8|16|32)")
    parser.add_argument("--depth", type=int, default=None, help="override encoder depth (4|8|16)")
    parser.add_argument("--epochs", type=int, default=None, help="override paper epochs (600)")
    parser.add_argument("--batch_size", type=int, default=None, help="override paper batch size (16)")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--run_name", default=None,
                        help="run directory name (default: <dataset>[_<site>]_<model>_p<ps>_d<depth>)")
    parser.add_argument("--fresh", action="store_true", help="archive an existing run and restart")
    parser.add_argument("--no_test", action="store_true", help="skip the final test evaluation")
    parser.add_argument("--overlays", type=int, default=8,
                        help="number of qualitative test overlays to save (0 = none)")
    args = parser.parse_args()

    if args.dataset == "CardiacUDA" and args.site is None:
        raise SystemExit("--site {G,R} is required for CardiacUDA (one run per hospital)")

    exp = dict(EXPERIMENTS[args.dataset])
    if args.patch_size is not None:
        exp["patch_size"] = args.patch_size
    if args.depth is not None:
        exp["depth"] = args.depth
    if args.model == "SynthesizerViTSeg":
        exp["syn_variant"] = args.variant
    epochs = args.epochs if args.epochs is not None else exp["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else exp["batch_size"]

    model_short = MODEL_SHORT.get(args.model, args.model)
    if args.model == "SynthesizerViTSeg":
        # one run directory per synthesizer variant (same backbone otherwise)
        model_short = SYN_TAG[args.variant]
    run_name = args.run_name or "{}{}_{}_p{}_d{}".format(
        args.dataset, "_{}".format(args.site) if args.site else "",
        model_short, exp["patch_size"], exp["depth"])
    run_dir = RUNS_ROOT / run_name
    if args.fresh and run_dir.exists():
        archive = run_dir.with_name(run_name + "__bak_" +
                                    datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        run_dir.rename(archive)
        print(f"[fresh] existing run archived to {archive}")
    run_dir.mkdir(parents=True, exist_ok=True)
    acquire_lock(run_dir)

    logger.remove()  # drop loguru's default stderr handler (avoid double lines)
    logger.add(str(run_dir / "train.log"), level="INFO")
    logger.add(sys.stdout, level="INFO")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, "| CUDA_VISIBLE_DEVICES =", args.gpu)

    config = {k: v for k, v in exp.items() if not callable(v)}
    config.update({"run_name": run_name, "epochs": epochs, "batch_size": batch_size,
                   "dataset": args.dataset, "site": args.site, "model": args.model,
                   "gpu": args.gpu, "started": datetime.datetime.now().isoformat()})
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    set_seed(exp["seed"])
    model = build_model(args.model, exp).to(device)
    net_name = type(model).__name__

    criterion = DiceCELoss(to_onehot_y=True, softmax=True,
                           include_background=True, weight=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=exp["lr"],
                                 betas=exp["betas"], weight_decay=exp["weight_decay"])

    train_loader, val_loader, test_loader = build_loaders(
        args.dataset, args.site, exp, batch_size, args.num_workers)
    logger.info(f"dataset {args.dataset}{('/' + args.site) if args.site else ''} | "
                f"model {net_name} | patch {exp['patch_size']} | depth {exp['depth']} | "
                f"epochs {epochs} | batch {batch_size} | lr {exp['lr']} | wd {exp['weight_decay']}")
    logger.info(f"train {len(train_loader.dataset)} | val {len(val_loader.dataset)} | "
                f"test {len(test_loader.dataset)} | run dir {run_dir}")

    ckpt_last = run_dir / "ckpt_last.pth"
    ckpt_best = run_dir / "ckpt_best.pth"
    csv_path = run_dir / "metrics.csv"

    start_epoch, best_miou, best_epoch = 0, -1.0, -1
    if ckpt_last.exists():
        # map_location="cpu": RNG state tensors must stay on CPU for set_rng_state
        state = torch.load(ckpt_last, map_location="cpu", weights_only=False)
        assert state["num_classes"] == exp["num_classes"], \
            f"checkpoint num_classes {state['num_classes']} != {exp['num_classes']}"
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best_miou = state["best_miou"]
        best_epoch = state["best_epoch"]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        logger.info(f"[resume] continuing from epoch {start_epoch} "
                    f"(best val mIoU {best_miou:.4f} at epoch {best_epoch})")
    else:
        # efficiency fingerprint only on fresh runs (it costs ~30 s)
        try:
            efficiency.log_efficiency_summary(
                model, (exp["in_channels"], exp["image_size"], exp["image_size"]),
                device, logger, name=net_name)
        except Exception as exc:
            logger.warning(f"efficiency profiling skipped: {exc}")

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    for epoch in range(start_epoch, epochs):
        seed_epoch(train_loader, exp["seed"], epoch)
        print(f"\nEpoch: {epoch}")
        print("-----------Training------------")
        t0 = datetime.datetime.now()
        train_loss, train_reg, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, exp)
        t_train = (datetime.datetime.now() - t0).total_seconds()
        print(f"Training Loss: {train_loss:.4f} (L_reg {train_reg:.6f}), "
              f"Train time: {t_train:.1f} s")
        print_metrics(train_metrics)

        print("------------Validation------------")
        t0 = datetime.datetime.now()
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, exp)
        t_val = (datetime.datetime.now() - t0).total_seconds()
        print(f"Validation Loss: {val_loss:.4f}, Validation time: {t_val:.1f} s")
        print_metrics(val_metrics)

        mm = mean_metrics(val_metrics)
        improved = mm["mIoU"] > best_miou
        if improved:
            best_miou, best_epoch = mm["mIoU"], epoch
            atomic_save({"epoch": epoch, "model": model.state_dict(),
                         "val_metrics": mm, "num_classes": exp["num_classes"]},
                        ckpt_best)
        save_last(ckpt_last, epoch, model, optimizer, best_miou, best_epoch, exp)

        row = {"epoch": epoch, "train_loss": round(train_loss, 6),
               "train_reg": round(train_reg, 6), "val_loss": round(val_loss, 6),
               "val_mDice": round(mm["mDice"], 6), "val_mIoU": round(mm["mIoU"], 6),
               "val_mSensitivity": round(mm["mSensitivity"], 6),
               "val_mPrecision": round(mm["mPrecision"], 6),
               "best": int(improved), "train_s": round(t_train, 1), "val_s": round(t_val, 1)}
        append_csv(csv_path, row)
        logger.info(f"epoch {epoch} | train loss {train_loss:.4f} | val loss {val_loss:.4f} "
                    f"| val mDice {mm['mDice']:.4f} | val mIoU {mm['mIoU']:.4f} "
                    f"(best {best_miou:.4f} @ {best_epoch}) | {t_train:.0f}+{t_val:.0f}s")

        if _STOP:
            logger.info("[stop] interrupted -- checkpoint saved; rerun the same "
                        "command to resume")
            return

    # ---------------- final test evaluation (paper protocol) ----------------
    if args.no_test:
        return
    if best_epoch < 0:
        logger.warning("no validation improvement recorded -- skipping test")
        return
    print(f"\n=========== Test (best epoch {best_epoch}, val mIoU {best_miou:.4f}) ===========")
    state = torch.load(ckpt_best, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device, exp)
    tm = mean_metrics(test_metrics)
    print_metrics(test_metrics)
    logger.info("test | " + " | ".join(
        f"{k} {v:.4f}" for k, v in tm.items() if isinstance(v, (int, float))))

    (run_dir / "test_metrics.json").write_text(json.dumps(
        {"loss": test_loss, "metrics": tm, "best_epoch": best_epoch,
         "best_val_miou": best_miou, "num_test_images": len(test_loader.dataset)},
        indent=2))
    (run_dir / "summary.json").write_text(json.dumps({
        "run_name": run_name, "dataset": args.dataset, "site": args.site,
        "model": args.model, "patch_size": exp["patch_size"], "depth": exp["depth"],
        "epochs": epochs, "best_epoch": best_epoch, "best_val_miou": best_miou,
        "test": tm, "finished": datetime.datetime.now().isoformat(),
    }, indent=2))

    if args.overlays > 0:
        overlay_dir = run_dir / "overlays"
        overlay_dir.mkdir(exist_ok=True)
        imagenet = args.dataset == "BRISC"  # brisc normalizes with ImageNet stats
        model.eval()
        with torch.no_grad():
            for i, (input, label) in enumerate(test_loader):
                if i >= args.overlays:
                    break
                pred = model(input.to(device).float()).argmax(dim=1)
                save_multiclass_overlay(input[0], pred[0].cpu().numpy(),
                                        label[0].numpy(),
                                        overlay_dir / f"compare_{i:04d}.png",
                                        imagenet_stats=(0.485, 0.456, 0.406) if imagenet else (0.0, 0.0, 0.0),
                                        imagenet_std=(0.229, 0.224, 0.225) if imagenet else (1.0, 1.0, 1.0))
        logger.info(f"test overlays saved to {overlay_dir}")

    print(json.dumps(tm, indent=2))


if __name__ == "__main__":
    main()
