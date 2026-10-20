"""BRISC brain-MRI segmentation dataset (paper segmentation experiment).

Expected layout under configs.dataset_paths.BRISC_ROOT:

    <root>/train/images/*.jpg   <root>/train/masks/*.png
    <root>/test/images/*.jpg    <root>/test/masks/*.png

Paper protocol:
    * train: all 3933 training volumes.  Augmentation: resize 384x384 ->
      random crop 256x256 + horizontal/vertical flips (paper).
    * the official 860-image test split is divided into 430 validation /
      430 test images (seed 42), matching the research code.
    * inference: resize 384 -> center crop 256 (paper).

Images are loaded lazily and transformed on every access, so the random
training augmentation differs on each epoch.
"""

import os
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from configs import dataset_paths

IMAGE_SIZE = 256                                  # paper: random crop 256
RESIZE_SIZE = 384                                 # paper: resize before crop
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform():
    return A.Compose([
        A.Resize(RESIZE_SIZE, RESIZE_SIZE),
        A.RandomCrop(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})


def test_transform():
    return A.Compose([
        A.Resize(RESIZE_SIZE, RESIZE_SIZE),
        A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})


class BRISCDataset(Dataset):
    """Binary (background / lesion) segmentation samples, paired by file stem."""

    def __init__(self, fold_path, transform=None):
        self.transform = transform
        self.image_path = os.path.join(fold_path, 'images')
        self.mask_path = os.path.join(fold_path, 'masks')

        images = sorted(f for f in os.listdir(self.image_path) if f.lower().endswith('.jpg'))
        masks = sorted(f for f in os.listdir(self.mask_path) if f.lower().endswith('.png'))
        assert len(images) == len(masks), \
            f"image/mask count mismatch in {fold_path}: {len(images)} vs {len(masks)}"

        mask_by_stem = {os.path.splitext(m)[0]: m for m in masks}
        self.pairs = []
        for img in images:
            stem = os.path.splitext(img)[0]
            if stem in mask_by_stem:
                self.pairs.append((os.path.join(self.image_path, img),
                                   os.path.join(self.mask_path, mask_by_stem[stem])))
        assert len(self.pairs) == len(images), \
            f"unpaired image files in {fold_path} ({len(images) - len(self.pairs)} missing masks)"

        self.fold_path = fold_path

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        image = np.array(Image.open(image_path).convert('RGB')).astype(np.uint8)
        mask = np.array(Image.open(mask_path)).astype(np.uint8)
        mask[mask > 0] = 1
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']        # torch.Tensor (C, H, W)
            mask = transformed['mask']          # torch.Tensor (H, W)
        return image, mask


def build_dataloaders(root=None, batch_size=16, num_workers=8, seed=42,
                      test_batch_size=1):
    """Build (train, validation, test) loaders under the paper protocol.

    The 860-image official test split is halved into 430 validation / 430 test
    with a fixed seed (as in the research code).  Test keeps batch size 1 so
    metrics can be accumulated per image.
    """
    root = Path(root) if root is not None else Path(dataset_paths.BRISC_ROOT)

    train_set = BRISCDataset(str(root / 'train'), train_transform())
    full_test_set = BRISCDataset(str(root / 'test'), test_transform())

    val_size = len(full_test_set) // 2
    test_size = len(full_test_set) - val_size
    val_set, test_set = torch.utils.data.random_split(
        full_test_set, [val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=test_batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = build_dataloaders(num_workers=0)
    print(f"BRISC lengths: train {len(train_loader.dataset)}, "
          f"val {len(val_loader.dataset)}, test {len(test_loader.dataset)}")
    image, mask = train_loader.dataset[0]
    print(f"sample shapes: image {tuple(image.shape)}, mask {tuple(mask.shape)}")
