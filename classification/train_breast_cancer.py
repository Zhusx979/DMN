#!/usr/bin/env python
# coding: utf-8
"""Published-paper experiment: breast-cancer ultrasound classification (Table 2).

DM ViT reaches 0.7776 test accuracy vs 0.7729 for the vanilla ViT baseline
under the paper configuration (input 224, patch 16, depth 8, AdamW lr 1e-4,
600 epochs with a 30-epoch linear warmup followed by cosine decay).

The full experiment configuration lives in configs/experiments.py
(DEEPBC_CLASSIFICATION); the training objective is
    L = CrossEntropy(logits, labels) + L_reg
where L_reg is the Dual Mapping constraint regularizer of Eqs. (9)-(10),
obtained from model.regularization_loss() (differentiable; only DM models).

Note: this dataset pipeline follows the research code and feeds 4-channel
(mask-cropped) images to the model -- see transforms/bc_mask_transforms.py.

Run from the repository root:
    python classification/train_breast_cancer.py --network DualMappingViT
    python classification/train_breast_cancer.py --network ViT               # baseline
"""

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

from configs.experiments import DEEPBC_CLASSIFICATION as EXP  # noqa: E402
from datasets.classification.breast_cancer_mask import BCData  # noqa: E402
from datasets.transforms import breast_cancer_mask_transforms as tr  # noqa: E402
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
    from prefetch_generator import BackgroundGenerator
except ImportError:  # prefetching is an optional speed-up
    def BackgroundGenerator(loader, max_prefetch=None):
        return loader

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


def train(model, train_loader, criterion, optimizer, device, epoch, reg_weight=1.0):
    losses, reg_losses = [], []
    model.train()
    total, correct = 0, 0
    with tqdm(len(train_loader)) as pbar:
        for batch_idx, sample in enumerate(BackgroundGenerator(train_loader, max_prefetch=1024)):
            images, labels = sample['image'], sample['label']
            images = images.to(device)
            labels = labels.to(device)

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
            predicted = torch.argmax(logits.detach(), axis=1).detach().cpu()

            correct += torch.sum(predicted == labels.detach().cpu())
            total += labels.size(0)

            pbar.update(1)
            pbar.set_description("Batch %d/%d, Train loss: %.4f, reg: %.5f" % (
                batch_idx + 1, len(train_loader), np.mean(losses), np.mean(reg_losses)))

    loss = np.mean(losses)
    acc = float(correct) / total

    run.track(loss, step=epoch, name="training_loss")
    run.track(acc, step=epoch, name="training_acc")
    run.track(np.mean(reg_losses), step=epoch, name="training_reg_loss")

    return loss, acc

def validate(model, test_loader, device, criterion, epoch, num_classes=3, mode="Validation"):
    model.eval()
    total, correct = 0, 0
    epoch_loss, epoch_logits, y_logits = [], [], []
    with torch.no_grad():
        for sample in tqdm(BackgroundGenerator(test_loader, max_prefetch=1024)):
            images, labels = sample['image'], sample['label']
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = criterion(logits, labels)
            epoch_loss.append(loss.item())

            predicted = torch.argmax(logits.detach(), axis=1).detach().cpu()
            correct += torch.sum(predicted == labels.detach().cpu())
            total += labels.size(0)

            epoch_logits = epoch_logits + [idx for idx in predicted]
            y_logits = y_logits + [idx for idx in labels]

    confusion_mat, confusion_normed, recall, precision, f1 = statistic(epoch_logits, y_logits, num_classes)
    logger.info("Confusion matrix: \n{}\nConfusion normed: \n{}\nRecall: {}\nPrecision: {}\nF1:{}".format(str(confusion_mat), str(confusion_normed), str(recall), str(precision), str(f1)))

    acc = float(correct) / float(total)
    loss = np.mean(epoch_loss)

    run.track(loss, step=epoch, name="{}_loss".format(mode))
    run.track(acc, step=epoch, name="{}_acc".format(mode))

    return loss, acc

def statistic(pred_logits, y_logits, label_size=2):
    '''get recall, precision and f1
    '''
    confusion_mat = np.zeros((label_size, label_size))
    for i, sample in enumerate(pred_logits):
        confusion_mat[y_logits[i], pred_logits[i]] += 1
    c = Confusion(confusion_mat)
    acc = c.accuracy()
    recall_list = c.recall()
    precision_list = c.precision()
    f1_list = c.f1()
    mat_normed = confusion_mat / np.sum(confusion_mat, axis=1, keepdims=True)
    return np.around(confusion_mat, 4), np.around(mat_normed, 4), np.around(recall_list, 4), np.around(precision_list, 4), np.around(f1_list, 4)

def _init__fn(worker_id):
    np.random.seed(config.seed)
    random.seed(config.seed)

# In[4]:


class Config(object):
    def __init__(self):

        self.train_batch = EXP["batch_size"]      # 16    (paper)
        self.test_batch = EXP["batch_size"]
        self.nepoch = EXP["epochs"]               # 600   (paper)
        self.dataset = EXP["name"]                # bc

        self.training_txt = EXP["label_dir"] + "/training.txt"
        self.val_txt = EXP["label_dir"] + "/val.txt"
        self.test_txt = EXP["label_dir"] + "/test.txt"

        self.data_root = EXP["data_root"]         # /ssd/.../DeepBC/developmentSet/data

        self.size = EXP["image_size"]             # 224   (paper)
        self.num_classes = EXP["num_classes"]     # 3

        self.depth = EXP["depth"]                 # 8     (paper)
        self.patch_size = EXP["patch_size"]       # 16    (paper)
        self.dim = EXP["dim"]
        self.mlp_dim = EXP["mlp_dim"]

        self.network = 'DualMappingViT'           # DualMappingViT | ViT | vit_timm
        self.optimizer = EXP["optimizer"]         # AdamW (paper)
        self.lr = EXP["lr"]                       # 1e-4  (paper)
        self.weight_decay = EXP["weight_decay"]   # 0.01  (paper)
        self.warmup_epochs = EXP["warmup_epochs"] # 30    (paper)
        self.reg_weight = EXP["reg_weight"]       # 1.0   weight of L_reg

        # deterministic run tag (no timestamp): auto-resume needs re-running
        # the same command to land in the same checkpoint folder
        self.suffix = "{}_depth_{}_patch_{}".format(self.optimizer, self.depth, self.patch_size)
        self.note = "test norm"

        self.gpu = "0"
        self.num_workers = 8

        self.seed = EXP["seed"]                   # 911


def parse_args():
    import argparse
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
config.gpu = args.gpu
# configs store the optimizer fully lower-case ("adamw"); the branches below
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

if config.seed is None:
    config.seed = random.randint(1, 10000)

np.random.seed(config.seed)
random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(str(config.seed))

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu


# In[5]:

train_transform = transforms.Compose([
    tr.RandomCropMask(),
    tr.Resize(size=(config.size, config.size)),
    tr.HorizontalFlip(),
    tr.Normalize(),
    tr.ToTensor(),
])

test_transform = transforms.Compose([
    tr.CenterCropMask(),
    tr.Resize(size=(config.size, config.size)),
    tr.Normalize(),
    tr.ToTensor(),
])

train_dataset = BCData(data_root=config.data_root, data_txt=config.training_txt, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=config.train_batch, shuffle=True, num_workers=config.num_workers, drop_last=True, worker_init_fn=_init__fn)

val_dataset = BCData(data_root=config.data_root, data_txt=config.val_txt, transform=test_transform)
val_loader = DataLoader(val_dataset, batch_size=config.test_batch, shuffle=False, num_workers=config.num_workers, drop_last=False, worker_init_fn=_init__fn)

test_dataset = BCData(data_root=config.data_root, data_txt=config.test_txt, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=config.test_batch, shuffle=False, num_workers=config.num_workers, drop_last=False, worker_init_fn=_init__fn)

# In[6]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if config.network == "ViT":
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

elif config.network == "DualMappingViT":
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
    raise Exception("Unkown network: {}".format(config.network))

# Efficiency fingerprint of the run: total parameters, FLOPs, GPU memory,
# inference latency and throughput (the paper's results list).
try:
    efficiency.log_efficiency_summary(
        model, (EXP["in_channels"], config.size, config.size), device, logger,
        name=config.network)
except Exception as exc:  # profiling must never break a training run
    logger.warning(f"efficiency profiling skipped: {exc}")

criterion = nn.CrossEntropyLoss()

if config.optimizer == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    lr_scheduler = None

elif config.optimizer == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=EXP["betas"],
                                  eps=1e-08, weight_decay=config.weight_decay)
    lr_scheduler = WarmupCosineSchedule(optimizer, warmup_steps=config.warmup_epochs, t_total=config.nepoch)
else:
    raise ("Unknown optimizer {}".format(config.optimizer))


# In[ ]:


# 训练
log_dir = REPO_ROOT / "logs" / "bc" / config.network
log_dir.mkdir(parents=True, exist_ok=True)
ckpt_folder = REPO_ROOT / "ckpts" / "bc" / config.network / config.suffix
ckpt_folder.mkdir(parents=True, exist_ok=True)
logger.add(str(log_dir / "{}.log".format(config.suffix)))

logger.info(config.__dict__)

exp_name = "{}_{}_{}".format(config.dataset, config.network, config.suffix)
if aim is None:
    logger.warning("aim is not installed - experiment tracking disabled (pip install aim)")
    run = NullRun()
else:
    run = aim.Run(experiment=exp_name)
run['hparams'] = config.__dict__

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
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch, reg_weight=config.reg_weight)
    logger.info("Epoch: {}, Training Loss: {:.4f}, Training acc: {:.4f}".format(epoch, train_loss, train_acc))

    val_loss, val_acc = validate(model, val_loader, device, criterion, epoch=epoch, num_classes=config.num_classes, mode="Validation")
    logger.info("Epoch: {}, Val Loss: {:.4f}, Val acc: {:.4f}".format(epoch, val_loss, val_acc))

    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        best_val_epoch = epoch
        logger.info("Epoch: %d, Validation Acc increased to %.4f" % (epoch, val_acc))

        test_loss, test_acc = validate(model, test_loader, device, criterion, epoch=epoch, num_classes=config.num_classes, mode="Test")
        logger.info("Epoch: %d, Test Loss is %.4f, Test Acc is %.4f" % (epoch, test_loss, test_acc))

        save_best(ckpt_folder, model)
        logger.info("Best model saved in file: %s" % (ckpt_folder / "ckpt_best.pth"))
    else:
        logger.info("Epoch: %d, Val's acc doesn't increase. Best Val's acc is %.6f in epoch %d. Corresponding Test's is %.6f." % (epoch, best_val_acc, best_val_epoch, test_acc))

    if lr_scheduler is not None:
        lr_scheduler.step()
        lr = lr_scheduler.get_last_lr()[0]
        logger.info("Epoch: %d, Learning rate: %.7f" % (epoch, lr))

    # full resume state (model+optimizer+scheduler+RNG), rewritten atomically
    # every epoch -- bounded: exactly ckpt_last.pth + ckpt_best.pth on disk
    save_resume_state(ckpt_folder, model, optimizer, lr_scheduler, epoch,
                      {"best_val_acc": best_val_acc, "best_val_epoch": best_val_epoch,
                       "test_acc": test_acc}, config.num_classes)
    if stop_requested():
        logger.info("Stop requested: exiting cleanly after epoch %d" % epoch)
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
    logger.info("Run summary written: {}".format(summary))
else:
    logger.info("Stopped early at epoch %d/%d -- no summary written; re-run to resume"
                % (epoch, config.nepoch))
