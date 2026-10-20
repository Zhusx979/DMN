# coding: utf-8
"""Classification entry point for the OCT / pneumonia datasets and extras.

Dataset coverage (--dataset):
    * oct       retinal diseases OCT, 4 classes -- published paper Table 2
                (split 4000 / 1000 / 1000, verified against the paper)
    * pneumonia pneumonia X-ray, 2 classes -- published paper Table 2
                (split 4632 / 600 / 624, verified against the paper)
    * tbx11k    tuberculosis chest X-ray, 3 classes (pre-publication experiment)
    * glaucoma  non-paper extra dataset

Networks (--network): DualMappingViT (published Method, Eqs. 7-10; default),
ViT (baseline), ScoreViT (legacy research variant that produced the archived
numbers), CCT / CCTScore.  Training objective:
    L = CrossEntropy(logits, labels) + reg_weight * model.regularization_loss()
for DM models.  To follow the paper protocol use --epochs 600 (AdamW 1e-4 is
the default optimizer/lr of this script).

Run from the repository root:
    python classification/train_auxiliary_datasets.py --dataset oct --network DualMappingViT
    python classification/train_auxiliary_datasets.py --dataset pneumonia --network DualMappingViT
"""

# In[ ]:

import os, pdb, cv2, random, datetime, math, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datasets.classification.tbx11k import TBX11KDataset
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
from loguru import logger
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.base_cam import BaseCAM
from utils.metrics import Confusion
from configs import dataset_paths
from configs.experiments import EXPERIMENTS
from utils.training import (get_settings, WarmupCosineSchedule, acquire_run_lock,
                            install_graceful_stop, stop_requested,
                            save_resume_state, save_best, load_resume_state)
from utils import efficiency
from torch.optim.lr_scheduler import _LRScheduler

from datasets.classification.image_list_dataset import Image_dataset
from datasets.transforms import image_transforms as tr
from datasets.classification import auxiliary_datasets as tasks

from networks.vit import ViT
from networks.dual_mapping_network_classification import DualMappingViT
from networks.synthesizer_vit import SynthesizerViT
from legacy.networks.score_vit import ScoreViT  # legacy research model (matches archived checkpoints)
from networks.cct import CCT
from networks.cct_score import CCTScore

import timm

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


def train(model, data_loader, device, criterion, optimizer, epoch, run, network):
    model.train()
    correct, total = 0, 0
    epoch_ce_loss, epoch_score_loss = [], []
    time_start = datetime.datetime.now()
    
    with tqdm(len(data_loader)) as pbar:
        for batch_idx, (inputs, targets, img_names) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            model.train()

            outputs = model(inputs)
            # Dual Mapping models: differentiable L_reg (paper Eqs. 9-10),
            # back-propagated jointly with the task loss.
            # Legacy ScoreViT: cal_attn_loss() is a monitoring value (the
            # constraint there lives in the forward softmax hook).
            if hasattr(model, "regularization_loss"):
                attn_loss = model.regularization_loss()
            elif hasattr(model, "cal_attn_loss"):
                attn_loss = model.cal_attn_loss()
            else:
                attn_loss = torch.tensor(0.0, device=device)

            ce_loss = criterion(outputs, targets)
            total_loss = ce_loss + config.reg_weight * attn_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_ce_loss.append(ce_loss.item())
            epoch_score_loss.append(attn_loss.item())
            
            predicted = torch.argmax(outputs.detach(), axis=1).detach().cpu()
            
            correct += torch.sum(predicted == targets.detach().cpu())
            total += targets.size(0)
            pbar.update(1)
            pbar.set_description("Epoch: %d, Batch %d/%d, Train ce loss: %.6f, Train score loss: %.6f, Train acc: %.4f"%(epoch, 
                                                                                               batch_idx+1, len(data_loader), 
                                                                                               np.mean(epoch_ce_loss), np.mean(epoch_score_loss), float(correct)/total))
    time_end = datetime.datetime.now()
    acc = float(correct) / total
    
    ce_avg_loss = np.mean(epoch_ce_loss)
    score_avg_loss = np.mean(epoch_score_loss)
    
    run.track(ce_avg_loss, step=epoch, name="training_loss")
    run.track(score_avg_loss, step=epoch, name="training_score_loss")
    run.track(acc, step=epoch, name="training_acc")
    
    time_spend = (time_end-time_start).seconds
    return acc, ce_avg_loss, score_avg_loss, time_spend


def test(model, data_loader, device, num_classes, criterion, epoch, run, mode = "Validation"):
    model.eval()
    correct, total = 0, 0
    epoch_ce_loss, epoch_logits, y_logits, outputs_softmax = [], [], [], []
    time_start = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (inputs, targets, img_names) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            epoch_ce_loss.append(loss.item()) # 仅记录cross entropy loss
            outputs_softmax = outputs_softmax + [softmax[1] for softmax in F.softmax(outputs, dim=1).detach().cpu().numpy()]
            
            predicted = torch.argmax(outputs.detach(), axis=1).detach().cpu()
            correct += torch.sum(predicted == targets.detach().cpu())
            total += targets.size(0)
            
            epoch_logits = epoch_logits + [idx for idx in predicted]
            y_logits = y_logits + [idx for idx in targets.detach().cpu().numpy()]
    
    confusion_mat, recall, precision, f1 = statistic(epoch_logits, y_logits, num_classes)
    logger.info("%s Confusion matrix: \n %s \n Recall: %s \n Precision: %s \n F1: %s"%(mode, str(confusion_mat),str(recall),str(precision), str(f1)))
    
    time_end = datetime.datetime.now()
    acc = float(correct) / total
    loss = np.mean(epoch_ce_loss)
    
    run.track(loss, step=epoch, name="{}_loss".format(mode))
    run.track(acc, step=epoch, name="{}_acc".format(mode))
    
    time_spend = (time_end-time_start).seconds
    return acc, loss, time_spend


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
    return np.around(confusion_mat, 4), np.around(recall_list, 4), np.around(precision_list,4), np.around(f1_list,4) 

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def _init__fn(worker_id):
    np.random.seed(config.seed)
    random.seed(config.seed)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--network", type=str, default="DualMappingViT") # DualMappingViT | ViT | SynthesizerViT | ScoreViT (legacy) | CCT | CCTScore
    parser.add_argument("--syn_variant", type=str, default="random") # random | random_fixed | dense | factorized_random | mix | static (SynthesizerViT)
    parser.add_argument("--reg_weight", type=float, default=1.0, help="weight of L_reg / attn loss")
    
    parser.add_argument("--patch_size", type=int, default=16) # 8 | 16 | 32 default: 16
    parser.add_argument("--depth", type=int, default=8) # 4 | 8 | 16 default: 8
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--dataset", type=str, default="tbx11k") # pneumonia | oct | tbx11k
    parser.add_argument("--suffix", type=str, default=None,
                        help="run folder tag; the deterministic default lets re-running the same command auto-resume")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=None,
                        help="defaults to the dataset's paper value from configs/experiments.py")
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=911)
    config = parser.parse_args()

    if config.num_classes is None:  # e.g. oct->4, pneumonia->2, tbx11k->3
        config.num_classes = EXPERIMENTS[config.dataset]["num_classes"]
    
    # no timestamp in the default suffix: auto-resume requires the run folder
    # name to be reproducible (a date rolls into an empty folder at midnight)
    base_suffix = "AdamW_1e-4" if config.suffix is None else config.suffix
    config.suffix = "{}_patch_{}_depth_{}".format(base_suffix, config.patch_size, config.depth)
    
    log_path, checkpoint_path = get_settings(config.dataset, config.network, config.suffix)
    logger.add(log_path, level="INFO")

    if config.seed is None:
        config.seed = random.randint(1, 10000)
    
    exp_name = "{}_{}_{}_{}".format(config.dataset, config.network, config.suffix, config.seed)
    if aim is None:
        logger.warning("aim is not installed - experiment tracking disabled (pip install aim)")
        run = NullRun()
    else:
        run = aim.Run(experiment=exp_name)
    run['hparams'] = config.__dict__
    
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(str(config.seed))

#     os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    device = torch.device("cuda:{}".format(config.gpu))
#     device = torch.device("cuda:0")

    torch.backends.cudnn.deterministic=True # this item has impact on the deterministic results when the optimizers are adaptive ones
    torch.cuda.set_device(device) # set the default device in order to allocate bernoulli variable on GPU
    
    if config.network == "ViT":
        net = ViT(image_size = config.size,
                    patch_size = config.patch_size,
                    channels=config.in_channels,
                    num_classes = config.num_classes,
                    dim = 1024,
                    depth = config.depth,
                    heads = 16,
                    mlp_dim = 2048,
                    dropout = 0,
                    emb_dropout = 0)
        net.to(device)
    elif config.network == "DualMappingViT":
        net = DualMappingViT(image_size = config.size,
                patch_size = config.patch_size,
                channels=config.in_channels,
                num_classes = config.num_classes,
                dim = 1024,
                depth = config.depth,
                mlp_dim = 2048,
                dropout = 0,
                emb_dropout = 0)
        net.to(device)
    elif config.network == "SynthesizerViT":
        net = SynthesizerViT(image_size = config.size,
                patch_size = config.patch_size,
                channels=config.in_channels,
                num_classes = config.num_classes,
                dim = 1024,
                depth = config.depth,
                heads = 16,
                mlp_dim = 2048,
                variant = config.syn_variant,
                dropout = 0,
                emb_dropout = 0)
        net.to(device)
    elif config.network == "ScoreViT":
        net = ScoreViT(image_size = config.size,
                patch_size = config.patch_size,
                channels=config.in_channels,
                num_classes = config.num_classes,
                dim = 1024,
                depth = config.depth,
                heads = 16,
                mlp_dim = 2048,
                dropout = 0,
                emb_dropout = 0)
        net.to(device)
    
    elif config.network == 'CCT':
        net = CCT(
            img_size = (224, 224),
            embedding_dim = 384,
            n_input_channels = config.in_channels,
            n_conv_layers = 2,
            kernel_size = 7,
            stride = 2,
            padding = 3,
            pooling_kernel_size = 3,
            pooling_stride = 2,
            pooling_padding = 1,
            num_layers = 14,
            num_heads = 16,
            mlp_ratio = 3.,
            num_classes = config.num_classes)
        net.to(device)
    elif config.network == 'CCTScore':
        net = CCTScore(
            img_size = (224, 224),
            embedding_dim = 384,
            n_input_channels = config.in_channels,
            n_conv_layers = 2,
            kernel_size = 7,
            stride = 2,
            padding = 3,
            pooling_kernel_size = 3,
            pooling_stride = 2,
            pooling_padding = 1,
            num_layers = 14,
            num_heads = 16,
            mlp_ratio = 3.,
            num_classes = config.num_classes)
        net.to(device)
        
        
    else:
        raise Exception("Unknown network: {}".format(config.network))

    net.to(device)

    # Efficiency fingerprint of the run: total parameters, FLOPs, GPU memory,
    # inference latency and throughput (the paper's results list).
    try:
        efficiency.log_efficiency_summary(
            net, (config.in_channels, config.size, config.size), device, logger,
            name=config.network)
    except Exception as exc:  # profiling must never break a training run
        logger.warning(f"efficiency profiling skipped: {exc}")

    criterion = nn.CrossEntropyLoss()

    if config.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        lr_scheduler = None
        use_scheduler = False
    elif config.optimizer == "Adadelta":
        optimizer = optim.Adadelta(net.parameters(), lr = 1.0)
        lr_scheduler = None
        use_scheduler = False
    elif config.optimizer == "AdamW":
        use_scheduler = True
        optimizer = optim.AdamW(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        lr_scheduler = WarmupCosineSchedule(optimizer, warmup_steps=30, t_total=config.epochs)
    else:
        raise("Unknown optimizer {}".format(config.optimizer))

    train_tr = transforms.Compose([
                tr.FixedResize([224, 224]), # h, w
                tr.RandomRotate(15),
                tr.Normalize(),
                tr.ToTensor()
            ])

    test_tr = transforms.Compose([
                tr.FixedResize([224, 224]),
                tr.Normalize(),
                tr.ToTensor()
            ])
    
    if config.dataset == "pneumonia":
        train_dataset = tasks.zk_pneumonia_train # tasks.zk_pneumonia_train | tasks.zk_oct_train | task.glaucoma_train
        val_dataset = tasks.zk_pneumonia_val # tasks.zk_pneumonia_val | tasks.zk_oct_val | task.glaucoma_val
        test_dataset = tasks.zk_pneumonia_test # tasks.zk_pneumonia_test | tasks.zk_oct_test | task.glaucoma_test
        trainset = Image_dataset(train_dataset, train_tr)
        valset = Image_dataset(val_dataset, test_tr)
        testset = Image_dataset(test_dataset, test_tr)
    elif config.dataset == "oct":
        train_dataset = tasks.zk_oct_train # tasks.zk_pneumonia_train | tasks.zk_oct_train | task.glaucoma_train
        val_dataset = tasks.zk_oct_val # tasks.zk_pneumonia_val | tasks.zk_oct_val | task.glaucoma_val
        test_dataset = tasks.zk_oct_test # tasks.zk_pneumonia_test | tasks.zk_oct_test | task.glaucoma_test
        trainset = Image_dataset(train_dataset, train_tr)
        valset = Image_dataset(val_dataset, test_tr)
        testset = Image_dataset(test_dataset, test_tr)
    elif config.dataset == "glaucoma":
        train_dataset = tasks.glaucoma_train
        val_dataset = tasks.glaucoma_val
        test_dataset = tasks.glaucoma_test
        trainset = Image_dataset(train_dataset, train_tr)
        valset = Image_dataset(val_dataset, test_tr)
        testset = Image_dataset(test_dataset, test_tr)
    elif config.dataset == "tbx11k":
        tbx11k_root = dataset_paths.TBX11K_ROOT  # /ssd/data/medical/TBX11K/imgs
        config.in_channels = 3
        trainset = TBX11KDataset(tbx11k_root, transform=train_tr, mode='train', seed=config.seed)
        valset = TBX11KDataset(tbx11k_root, transform=test_tr, mode='val', seed=config.seed)
        testset = TBX11KDataset(tbx11k_root, transform=test_tr, mode='test', seed=config.seed)
    else:
        raise Exception("unknown dataset: {}".format(config.dataset))
    
    
    trainloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, worker_init_fn=_init__fn)
    valloader = DataLoader(valset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, worker_init_fn=_init__fn)
    testloader = DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, worker_init_fn=_init__fn)
    
    logger.info("Number of training images: {}, val images: {}, test images: {}".format(len(trainset), len(valset), len(testset)))
    
    # ---- interruption-safe checkpointing: single-writer lock, graceful
    # SIGINT/SIGTERM stop, and auto-resume from ckpt_last.pth ----
    os.makedirs(checkpoint_path, exist_ok=True)
    acquire_run_lock(checkpoint_path)
    install_graceful_stop()
    best_val_acc, best_epoch, correspond_test_acc = -1, -1, -1
    start_epoch, best = load_resume_state(checkpoint_path, net, optimizer,
                                          lr_scheduler, config.num_classes)
    if start_epoch > 0:
        best_val_acc = best.get("best_val_acc", -1)
        best_epoch = best.get("best_epoch", -1)
        correspond_test_acc = best.get("correspond_test_acc", -1)
        logger.info("[resume] continuing from epoch {} (best val acc {:.6f} at epoch {})".format(
            start_epoch, best_val_acc, best_epoch))

    for epoch in range(start_epoch, config.epochs):
        train_acc, train_ce_loss, train_score_loss, train_time = train(net, trainloader, device, criterion, optimizer, epoch, run, config.network)
        logger.info("Epoch: %d, Train Time: %s(s), Train CE Loss: %.4f, Train Score Loss: %.4f, Train Acc: %.4f"%(epoch, train_time, train_ce_loss, train_score_loss, train_acc))

        val_acc, val_loss, val_time = test(net, valloader, device, config.num_classes, criterion, epoch, run, mode = "Validation")
        logger.info("Epoch: %d, Val Time: %s(s), Val Loss: %.4f, Val Acc: %.4f"%(epoch, val_time, val_loss, val_acc))

        if val_acc > best_val_acc:
            logger.info("Epoch: %d, Val acc increased to %.6f."%(epoch, val_acc))
            best_val_acc = val_acc
            best_epoch = epoch
            test_acc, test_loss, test_time = test(net, testloader, device, config.num_classes, criterion, epoch, run, mode = "Test")
            correspond_test_acc = test_acc
            logger.info("Epoch: %d, Test acc is %.6f."%(epoch, test_acc))
            save_best(checkpoint_path, net)
            logger.info("Best model saved in file: {}".format(
                os.path.join(checkpoint_path, "ckpt_best.pth")))
        else:
            logger.info("Epoch: %d, Val acc doesn't increase. Best Val acc is %.6f in epoch %d. Corresponding Test Acc is %.6f."%(epoch, best_val_acc, best_epoch, correspond_test_acc))

        if use_scheduler:
            lr_scheduler.step()
            logger.info("Epoch: %d, Learning rate: %.6f"%(epoch, lr_scheduler.get_last_lr()[0]))

        # full resume state (model+optimizer+scheduler+RNG), rewritten
        # atomically every epoch -- bounded: exactly ckpt_last.pth +
        # ckpt_best.pth on disk instead of one epoch{N}.pth per improvement
        save_resume_state(checkpoint_path, net, optimizer, lr_scheduler, epoch,
                          {"best_val_acc": best_val_acc, "best_epoch": best_epoch,
                           "correspond_test_acc": correspond_test_acc},
                          config.num_classes)
        if stop_requested():
            logger.info("Stop requested: exiting cleanly after epoch {}".format(epoch))
            break