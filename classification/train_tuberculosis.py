#!/usr/bin/env python
# coding: utf-8
"""Pre-publication experiment: TBX11K tuberculosis classification.

NOTE: TBX11K is NOT part of the published version ("Efficient Vision
Transformer for Medical Image Analysis", Int. J. Neural Systems 2024) -- the
published classification datasets are OCT / pneumonia / breast cancer.  This
script is kept for reproducibility of the earlier draft, where DM ViT reached
0.9500 test accuracy vs 0.9250 for the vanilla ViT baseline under the
configuration below (input 224, patch 16, depth 8, AdamW lr 1e-4, 600 epochs
with a 30-epoch linear warmup followed by cosine decay).

The full experiment configuration lives in configs/experiments.py
(TBX11K_CLASSIFICATION); the training objective is
    L = CrossEntropy(logits, labels) + L_reg
where L_reg is the Dual Mapping constraint regularizer of Eqs. (9)-(10),
obtained from model.regularization_loss() (differentiable; only DM models).

Run from the repository root:
    python classification/train_tuberculosis.py --network DualMappingViT
    python classification/train_tuberculosis.py --network ViT          # baseline
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from configs.experiments import TBX11K_CLASSIFICATION as EXP  # noqa: E402
from datasets.classification.tbx11k import TBX11KDataset  # noqa: E402
from datasets.transforms import image_transforms as tr  # noqa: E402
from networks.dual_mapping_network_classification import DualMappingViT  # noqa: E402
from networks.vit import ViT  # noqa: E402
from networks.vit_timm import ViT_timm  # noqa: E402
from networks.synthesizer_vit import SynthesizerViT  # noqa: E402
from utils import efficiency  # noqa: E402
from utils.metrics import Confusion  # noqa: E402
from utils.training import (WarmupCosineSchedule, acquire_run_lock,  # noqa: E402
                            install_graceful_stop, stop_requested,
                            save_resume_state, save_best, load_resume_state)

try:
    import aim
except ImportError:  # metrics tracking is optional
    aim = None


class NullRun:
    """No-op stand-in for aim.Run when aim is not installed."""

    def __init__(self, *args, **kwargs):
        pass

    def __setitem__(self, key, value):
        pass

    def track(self, *args, **kwargs):
        pass


# ==================== 训练函数 ====================
def train(model, train_loader, criterion, optimizer, device, epoch, run, reg_weight=1.0):
    losses, reg_losses = [], []
    model.train()
    total, correct = 0, 0
    with tqdm(total=len(train_loader)) as pbar:
        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)

            # Dual Mapping constraint regularizer (paper Eqs. 9-10), added to
            # the task loss.  Non-DM baselines (ViT, ...) have no W_S matrices.
            if hasattr(model, "regularization_loss"):
                reg_loss = model.regularization_loss()
            else:
                reg_loss = torch.zeros((), device=device)
            total_loss = loss + reg_weight * reg_loss

            total_loss.backward()
            optimizer.step()

            losses.append(loss.item())
            reg_losses.append(float(reg_loss.detach()))
            predicted = torch.argmax(logits.detach(), dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            pbar.update(1)
            pbar.set_description(
                f"Batch {batch_idx+1}/{len(train_loader)}, Train loss: {np.mean(losses):.4f}, reg: {np.mean(reg_losses):.5f}")

    loss = np.mean(losses)
    acc = correct / total
    run.track(loss, step=epoch, name="training_loss")
    run.track(acc, step=epoch, name="training_acc")
    run.track(np.mean(reg_losses), step=epoch, name="training_reg_loss")
    return loss, acc


# ==================== 验证/测试函数 ====================
def validate(model, data_loader, device, criterion, epoch, num_classes, mode, run):
    model.eval()
    total, correct = 0, 0
    epoch_loss = []
    epoch_logits, y_logits = [], []
    with torch.no_grad():
        for images, labels, _ in data_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            epoch_loss.append(loss.item())

            predicted = torch.argmax(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            epoch_logits.extend(predicted.cpu().numpy())
            y_logits.extend(labels.cpu().numpy())

    confusion_mat, _, recall, precision, f1 = statistic(epoch_logits, y_logits, num_classes)
    logger.info(f"Confusion matrix:\n{confusion_mat}\nRecall: {recall}\nPrecision: {precision}\nF1: {f1}")

    acc = correct / total
    loss = np.mean(epoch_loss)
    run.track(loss, step=epoch, name=f"{mode}_loss")
    run.track(acc, step=epoch, name=f"{mode}_acc")
    return loss, acc


def statistic(pred_logits, y_logits, label_size):
    confusion_mat = np.zeros((label_size, label_size))
    for i, pred in enumerate(pred_logits):
        confusion_mat[y_logits[i], pred] += 1
    c = Confusion(confusion_mat)
    recall = c.recall()
    precision = c.precision()
    f1 = c.f1()
    return confusion_mat, None, recall, precision, f1


def _init_fn(worker_id):
    np.random.seed(config.seed)
    random.seed(config.seed)


# ==================== 配置（论文设置，见 configs/experiments.py） ====================
class Config:
    def __init__(self):
        self.train_batch = EXP["batch_size"]      # 16    (paper)
        self.test_batch = EXP["batch_size"]
        self.nepoch = EXP["epochs"]               # 600   (paper)
        self.dataset = EXP["name"]                # tbx11k

        self.size = EXP["image_size"]             # 224   (paper)
        self.num_classes = EXP["num_classes"]     # 3     health / sick / tb
        self.depth = EXP["depth"]                 # 8     (paper)
        self.patch_size = EXP["patch_size"]       # 16    (paper)
        self.dim = EXP["dim"]                     # 1024
        self.mlp_dim = EXP["mlp_dim"]             # 2048

        self.network = "DualMappingViT"           # DualMappingViT | ViT | vit_timm
        self.optimizer = EXP["optimizer"]         # AdamW (paper)
        self.lr = EXP["lr"]                       # 1e-4  (paper)
        self.weight_decay = EXP["weight_decay"]   # 0.01  (paper)
        self.warmup_epochs = EXP["warmup_epochs"] # 30    (paper)
        self.reg_weight = EXP["reg_weight"]       # 1.0   weight of L_reg

        # deterministic run tag (no timestamp): auto-resume needs re-running
        # the same command to land in the same checkpoint folder
        self.suffix = f"{self.optimizer}_depth_{self.depth}_patch_{self.patch_size}"
        self.gpu = "0"
        self.num_workers = 8
        self.seed = EXP["seed"]                   # 911


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--network", default="DualMappingViT",
                        choices=["DualMappingViT", "ViT", "SynthesizerViT", "vit_timm"],
                        help="DM ViT (paper model) or baselines")
    parser.add_argument("--syn_variant", default="random",
                        choices=["random", "random_fixed", "dense", "factorized_random", "mix", "static"],
                        help="Synthesizer attention variant (for --network SynthesizerViT)")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES (default: 0)")
    parser.add_argument("--epochs", type=int, default=None, help="override paper epochs (600)")
    parser.add_argument("--patch_size", type=int, default=None,
                        help="override paper patch size (16); draft Table 3 uses 8 / 32")
    parser.add_argument("--depth", type=int, default=None,
                        help="override paper depth (8); draft Table 3 uses 4 / 16")
    parser.add_argument("--suffix", default=None, help="run-name suffix (default: optimizer_depth_patch; deterministic so re-runs auto-resume)")
    return parser.parse_args()


config = Config()
args = parse_args()
config.network = args.network
config.syn_variant = args.syn_variant
config.gpu = args.gpu
# configs store the optimizer fully lower-case ("adamw"); the branch below
# and the run suffix expect the canonical "AdamW" spelling
config.optimizer = {"adam": "Adam", "adamw": "AdamW",
                    "adadelta": "Adadelta"}.get(config.optimizer.lower(), config.optimizer)
if args.epochs is not None:
    config.nepoch = args.epochs
if args.patch_size is not None:
    config.patch_size = args.patch_size
if args.depth is not None:
    config.depth = args.depth
# rebuild the default suffix after any ablation override so draft-Table-3
# runs land in their own folders instead of colliding with the default run
# (an explicit --suffix always wins)
if args.suffix is not None:
    config.suffix = args.suffix
else:
    config.suffix = "{}_depth_{}_patch_{}".format(config.optimizer, config.depth, config.patch_size)
    if config.network == "SynthesizerViT":
        # the synthesizer variants share the backbone but not the mapping:
        # keep them in separate checkpoint folders
        config.suffix += "_syn-{}".format(config.syn_variant)

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

# 设置随机种子
np.random.seed(config.seed)
random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

# ==================== 数据加载 ====================
train_transform = transforms.Compose([
    tr.FixedResize([config.size, config.size]),
    tr.RandomRotate(15),
    tr.Normalize(),
    tr.ToTensor(),
])

test_transform = transforms.Compose([
    tr.FixedResize([config.size, config.size]),
    tr.Normalize(),
    tr.ToTensor(),
])

tbx11k_root = EXP["data_root"]  # /ssd/data/medical/TBX11K/imgs (configs/dataset_paths.py)
trainset = TBX11KDataset(tbx11k_root, transform=train_transform, mode='train', seed=config.seed)
valset = TBX11KDataset(tbx11k_root, transform=test_transform, mode='val', seed=config.seed)
testset = TBX11KDataset(tbx11k_root, transform=test_transform, mode='test', seed=config.seed)

train_loader = DataLoader(trainset, batch_size=config.train_batch, shuffle=True,
                          num_workers=config.num_workers, worker_init_fn=_init_fn)
val_loader = DataLoader(valset, batch_size=config.test_batch, shuffle=False,
                        num_workers=config.num_workers, worker_init_fn=_init_fn)
test_loader = DataLoader(testset, batch_size=config.test_batch, shuffle=False,
                         num_workers=config.num_workers, worker_init_fn=_init_fn)

logger.info(f"Number of training images: {len(trainset)}, val: {len(valset)}, test: {len(testset)}")

# ==================== 模型 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if config.network == "DualMappingViT":
    model = DualMappingViT(image_size=config.size,
                           patch_size=config.patch_size,
                           channels=EXP["in_channels"],
                           num_classes=config.num_classes,
                           dim=config.dim,
                           depth=config.depth,
                           mlp_dim=config.mlp_dim,
                           dropout=EXP["dropout"],
                           emb_dropout=EXP["emb_dropout"])
    model.to(device)

elif config.network == "ViT":
    model = ViT(image_size=config.size,
                patch_size=config.patch_size,
                channels=EXP["in_channels"],
                num_classes=config.num_classes,
                dim=config.dim,
                depth=config.depth,
                heads=16,
                mlp_dim=config.mlp_dim,
                dropout=EXP["dropout"],
                emb_dropout=EXP["emb_dropout"])
    model.to(device)

elif config.network == "SynthesizerViT":
    model = SynthesizerViT(image_size=config.size,
                           patch_size=config.patch_size,
                           channels=EXP["in_channels"],
                           num_classes=config.num_classes,
                           dim=config.dim,
                           depth=config.depth,
                           heads=16,
                           mlp_dim=config.mlp_dim,
                           variant=config.syn_variant,
                           dropout=EXP["dropout"],
                           emb_dropout=EXP["emb_dropout"])
    model.to(device)

elif config.network == "vit_timm":
    model = ViT_timm(hidden_dim=config.dim, num_classes=config.num_classes)
    model.to(device)

else:
    raise Exception("Unknown network: {}".format(config.network))

# Efficiency fingerprint of the run: total parameters, FLOPs, GPU memory,
# inference latency and throughput (the paper's results list).
try:
    efficiency.log_efficiency_summary(
        model, (EXP["in_channels"], config.size, config.size), device, logger,
        name=config.network)
except Exception as exc:  # profiling must never break a training run
    logger.warning(f"efficiency profiling skipped: {exc}")

criterion = nn.CrossEntropyLoss()

if config.optimizer == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  betas=EXP["betas"],
                                  eps=1e-08, weight_decay=config.weight_decay)
    lr_scheduler = WarmupCosineSchedule(optimizer, warmup_steps=config.warmup_epochs,
                                        t_total=config.nepoch)
else:
    raise ValueError("Unsupported optimizer")

# ==================== 日志和 Aim 实验跟踪 ====================
log_dir = REPO_ROOT / "logs" / "tbx11k" / config.network
log_dir.mkdir(parents=True, exist_ok=True)
ckpt_folder = REPO_ROOT / "ckpts" / "tbx11k" / config.network / config.suffix
ckpt_folder.mkdir(parents=True, exist_ok=True)
logger.add(str(log_dir / f"{config.suffix}.log"))

exp_name = f"{config.dataset}_{config.network}_{config.suffix}"
if aim is None:
    logger.warning("aim is not installed - experiment tracking disabled (pip install aim)")
    run = NullRun()
else:
    run = aim.Run(experiment=exp_name)
run["hparams"] = config.__dict__

# ==================== 训练循环 ====================
# ---- interruption-safe checkpointing: single-writer lock, graceful
# SIGINT/SIGTERM stop, and auto-resume from ckpt_last.pth ----
acquire_run_lock(ckpt_folder)
install_graceful_stop()
best_val_acc, best_val_epoch, test_acc = -1, -1, 0.0
start_epoch, best = load_resume_state(ckpt_folder, model, optimizer, lr_scheduler,
                                      config.num_classes)
if start_epoch > 0:
    best_val_acc = best.get("best_val_acc", -1)
    best_val_epoch = best.get("best_val_epoch", -1)
    test_acc = best.get("test_acc", 0.0)
    logger.info("[resume] continuing from epoch {} (best val acc {:.6f} at epoch {})".format(
        start_epoch, best_val_acc, best_val_epoch))

epoch = start_epoch - 1   # defined even when the resume leaves the loop empty
for epoch in range(start_epoch, config.nepoch):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch, run,
                                  reg_weight=config.reg_weight)
    logger.info(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}")

    val_loss, val_acc = validate(model, val_loader, device, criterion, epoch,
                                 config.num_classes, "Validation", run)
    logger.info(f"Epoch {epoch}: Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        best_val_epoch = epoch
        logger.info(f"Epoch {epoch}: Validation Acc increased to {val_acc:.4f}")

        test_loss, test_acc = validate(model, test_loader, device, criterion, epoch,
                                       config.num_classes, "Test", run)
        logger.info(f"Epoch {epoch}: Test Loss {test_loss:.4f}, Test Acc {test_acc:.4f}")

        save_best(ckpt_folder, model)
        logger.info(f"Best model saved to {ckpt_folder / 'ckpt_best.pth'}")
    else:
        logger.info(f"Epoch {epoch}: Val acc did not increase. Best val acc {best_val_acc:.4f} at epoch {best_val_epoch}, corresponding test acc {test_acc:.4f}")

    if lr_scheduler is not None:
        lr_scheduler.step()
        current_lr = lr_scheduler.get_last_lr()[0]
        logger.info(f"Epoch {epoch}: Learning rate {current_lr:.7f}")

    # full resume state (model+optimizer+scheduler+RNG), rewritten atomically
    # every epoch -- bounded: exactly ckpt_last.pth + ckpt_best.pth on disk
    save_resume_state(ckpt_folder, model, optimizer, lr_scheduler, epoch,
                      {"best_val_acc": best_val_acc, "best_val_epoch": best_val_epoch,
                       "test_acc": test_acc}, config.num_classes)
    if stop_requested():
        logger.info(f"Stop requested: exiting cleanly after epoch {epoch}")
        break

# ==================== final summary (draft Table 2/3 result + queue marker) ==========
if epoch == config.nepoch - 1:
    summary = {
        "dataset": config.dataset, "network": config.network, "suffix": config.suffix,
        "patch_size": config.patch_size, "depth": config.depth, "epochs": config.nepoch,
        "syn_variant": config.syn_variant if config.network == "SynthesizerViT" else None,
        "best_val_acc": best_val_acc, "best_val_epoch": best_val_epoch,
        "test_acc": test_acc, "finished": datetime.now().isoformat(),
    }
    with open(ckpt_folder / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Run summary written: {summary}")
else:
    logger.info(f"Stopped early at epoch {epoch}/{config.nepoch} -- no summary written; re-run to resume")
