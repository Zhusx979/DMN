#!/usr/bin/env python
# coding: utf-8
"""Evaluate a trained segmentation checkpoint (research-line inference protocol).

Loads segmentation/checkpoint/<dataset>/<model>/best_model.pth by default and
runs the official test split (BRISC: 430 images; inference = resize 384 ->
center crop 256, center-crop transform built in segmentation/datasets/).
Every 10th test image is saved as an overlay for qualitative inspection.

Run from the repository root:
    python segmentation/test_segmentation.py --dataset BRISC --model DualMappingViTSeg
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from configs.experiments import SEGMENTATION_BRISC, SEGMENTATION_CARDIAC_UDA  # noqa: E402
from datasets.segmentation import brisc, cardiac_uda  # noqa: E402
from segmentation.model_factory import MODELS, build_model  # noqa: E402
from utils.segmentation import (medical_multi_class_metrics, print_metrics,  # noqa: E402
                                      save_vis_overlay, set_seed)

EXPERIMENTS = {"BRISC": SEGMENTATION_BRISC, "CardiacUDA": SEGMENTATION_CARDIAC_UDA}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="BRISC", choices=list(EXPERIMENTS))
    parser.add_argument("--model", default="DualMappingViTSeg", choices=MODELS)
    parser.add_argument("--checkpoint", default=None,
                        help="path to checkpoint (default: best_model.pth of the run)")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()

    exp = dict(EXPERIMENTS[args.dataset])
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(exp["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.model, exp).to(device)
    net_name = type(model).__name__

    ckpt_path = Path(args.checkpoint) if args.checkpoint else \
        REPO_ROOT / "segmentation" / "checkpoint" / args.dataset / net_name / "best_model.pth"
    model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    model.eval()
    logger.info(f"loaded checkpoint {ckpt_path}")

    if args.dataset == "BRISC":
        _, _, test_loader = brisc.build_dataloaders(root=exp["data_root"], batch_size=1,
                                                    num_workers=args.num_workers,
                                                    seed=exp["seed"])
    else:
        _, _, test_loader = cardiac_uda.build_dataloaders(root=exp["data_root"], batch_size=1,
                                                         num_workers=args.num_workers,
                                                         seed=exp["seed"], test_batch_size=1)

    visualization_path = REPO_ROOT / "segmentation" / "visualization" / args.dataset / net_name
    visualization_path.mkdir(parents=True, exist_ok=True)

    metrics = []
    test_start = datetime.datetime.now()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            input, label = data
            input = input.to(device).float()
            label = label.to(device).long().unsqueeze(1)
            output = model(input)
            preds = output.argmax(dim=1)

            if i % 10 == 0:
                save_vis_overlay(input, preds, label, str(visualization_path), i,
                                 mask_color=(0, 0, 255))

            metrics.append(medical_multi_class_metrics(
                label, preds,
                num_classes=exp["num_classes"],
                include_background=False,
            ))
    test_end = datetime.datetime.now()
    print(f"Test time {(test_end - test_start).total_seconds():.2f} seconds")
    print_metrics(metrics)
    logger.info(f"test on {args.dataset}/{net_name}: "
                f"mIoU {np.mean([m['mIoU'] for m in metrics]):.4f}")


if __name__ == "__main__":
    main()
