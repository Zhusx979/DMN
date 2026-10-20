"""Losses, metrics and helpers for segmentation training/evaluation."""

import math
import os.path
import random

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import LambdaLR


def make_one_hot(input, num_classes):
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result



class DiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'

        dice = BinaryDiceLoss(**self.kwargs)

        total_loss = 0
        predict = F.softmax(predict, dim=1)

        target_one_hot = F.one_hot(target, num_classes=predict.shape[1]).float()

        for i in range(target.shape[1]):  # iterate over each class
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target_one_hot[:, i])  # calculate Dice loss for the current class

                if self.weight is not None:
                    # Ensure weight array matches the number of classes
                    assert self.weight.shape[0] == target.shape[1], \
                        f'Expect weight shape [{target.shape[1]}], but got [{self.weight.shape[0]}]'
                    dice_loss *= self.weight[i]  # apply class-specific weight

                total_loss += dice_loss  # accumulate loss

        # Average loss over the number of classes (excluding the ignored class)
        return total_loss / (target.shape[1] - 1 if self.ignore_index is not None else target.shape[1])


class BinaryDiceLoss(nn.Module):
    """Binary Dice Loss for each class"""

    def __init__(self, smooth=1e-5):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        # Flatten the prediction and target tensors
        predict = predict.view(-1)
        target = target.view(-1)

        intersection = torch.sum(predict * target)
        dice = (2. * intersection + self.smooth) / (torch.sum(predict) + torch.sum(target) + self.smooth)

        return 1 - dice  # Dice loss is 1 - Dice coefficient
class WarmupCosineSchedule(LambdaLR):

    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")
def print_metrics(metrics):
    final_resuls = {
        'Pixel_Accuracy': [],
        'mDice': [],
        'mIoU': [],
        'mSensitivity': [],
        'mPrecision': [],
        'mSpecificity': [],
        'Dice_per_class':[],
        'IoU_per_class': [],
    }
    for metric in metrics:
        for k, v in final_resuls.items():
            final_resuls[k].append(metric[k])

    for k, v in final_resuls.items():
        if type(v[0]) != type(list()):
            print(f"{k}: {np.mean(v):.4f}")

        else:
            results = [np.mean(x) for x in zip(*v)]
            print(f"{k}:")
            for result in results:
                print(f"{result:.4f}")


COLORS = np.array([
    [0,   0,   0],     # 0: 背景 - 黑色或透明，但这里保持黑
    [30, 144, 255],    # 1: 蓝色 (e.g. 左心室)
    [0, 255,   0],     # 2: 绿色
    [255,  0,   0],    # 3: 红色
    [0, 255, 255]      # 4: 青色
], dtype=np.uint8)


import os
import numpy as np
import cv2
import torch
def save_vis_overlay(input_img, pred_label, gt_label, save_dir, idx, mask_color=(0, 0, 255)):
    os.makedirs(save_dir, exist_ok=True)

    img = input_img.detach().cpu()
    if img.dim() == 4 and img.size(0) == 1:
        img = img.squeeze(0)          # (3, H, W)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean            
    img = img.clamp(0, 1).numpy()     # (3, H, W)
    img = np.transpose(img, (1, 2, 0))  # (H, W, 3)
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def to_binary_mask(mask):
        mask = mask.detach().cpu()
        if mask.dim() == 4:
            mask = mask.squeeze(0).squeeze(0)
        elif mask.dim() == 3:
            mask = mask.squeeze(0)
        elif mask.dim() == 2:
            pass
        mask = (mask > 0.5).byte().numpy()
        return mask

    gt_mask = to_binary_mask(gt_label)
    pred_mask = to_binary_mask(pred_label)

    def overlay(mask, base_bgr, color):
        overlay = base_bgr.copy()
        overlay[mask == 1] = color
        return overlay

    gt_overlay = overlay(gt_mask, gray_bgr, mask_color)
    pred_overlay = overlay(pred_mask, gray_bgr, mask_color)

    h, w, _ = gray_bgr.shape
    sep = np.full((h, 10, 3), 255, dtype=np.uint8)   
    combined = np.hstack([gray_bgr, sep, gt_overlay, sep, pred_overlay])
    save_path = os.path.join(save_dir, f"compare_{idx:04d}.png")
    cv2.imwrite(save_path, combined, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"Saved: {save_path}")

@torch.no_grad()
def medical_multi_class_metrics(
    gt: torch.Tensor,
    pred: torch.Tensor,
    num_classes: int = 5,
    include_background: bool = False,
    eps: float = 1e-5,
) -> dict:

    device = gt.device
    gt = gt.view(-1).long()
    pred = pred.view(-1).long()

    start_cls = 0 if include_background else 1
    num_avg_cls = num_classes if include_background else num_classes - 1

    dices = torch.zeros(num_classes, device=device)
    ious = torch.zeros(num_classes, device=device)
    sensitivities = torch.zeros(num_classes, device=device)
    specificities = torch.zeros(num_classes, device=device)
    precisions = torch.zeros(num_classes, device=device)
    vol_gts = torch.zeros(num_classes, device=device)
    vol_preds = torch.zeros(num_classes, device=device)

    for cls in range(num_classes):
        gt_cls = (gt == cls)
        pred_cls = (pred == cls)

        inter = (gt_cls & pred_cls).sum().float()
        union = (gt_cls | pred_cls).sum().float()
        gt_vol = gt_cls.sum().float()
        pred_vol = pred_cls.sum().float()

        dices[cls]       = (2. * inter + eps) / (gt_vol + pred_vol + eps)
        ious[cls]        = (inter + eps) / (union + eps)
        sensitivities[cls] = inter / (gt_vol + eps)
        precisions[cls]   = inter / (pred_vol + eps)
        tn = (gt.numel() - gt_vol - (pred_vol - inter))
        specificities[cls] = tn / (tn + (pred_vol - inter) + eps)

        vol_gts[cls] = gt_vol
        vol_preds[cls] = pred_vol

    mean_dice = dices[start_cls:].mean().item() if not include_background else dices.mean().item()
    mean_iou  = ious[start_cls:].mean().item() if not include_background else ious.mean().item()

    pixel_acc = (pred == gt).float().mean().item()


    results = {
        'Pixel_Accuracy': pixel_acc,
        'mDice': mean_dice,
        'mIoU': mean_iou,
        'mSensitivity': sensitivities[start_cls:].mean().item() if not include_background else sensitivities.mean().item(),
        'mPrecision': precisions[start_cls:].mean().item() if not include_background else precisions.mean().item(),
        'mSpecificity': specificities[start_cls:].mean().item() if not include_background else specificities.mean().item(),
        'Dice_per_class': dices.cpu().tolist(),
        'IoU_per_class': ious.cpu().tolist(),
    }

    return results