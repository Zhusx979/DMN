"""CardiacUDA (CardiacUDC) heart-ultrasound segmentation dataset (draft paper).

Two layouts are supported:

1. **Prepared npz mode (paper protocol, preferred)** -- the official nii
   volumes downloaded from Kaggle (``xiaoweixumedicalai/cardiacudc-dataset``)
   are converted once by ``scripts/prepare_cardiacudc.py`` into per-patient
   npz files (research-code frame filter + per-frame max normalization), and
   this loader reads them::

       <prepared_root>/<Site_...>/<patient>.npz   (images float32, masks uint8)

   Splits follow the draft-paper site partitions (configs/experiments.py
   ``site_splits``; G 485/145/105 = Site_G_100/Site_G_29/Site_G_20,
   R 545/365/260 = Site_R_126/Site_R_73/Site_R_52 -- verify with the frame
   counts printed by the prepare script).

2. **jpg frame-slice export (fallback)** -- ``<root>/image`` + ``<root>/mask``
   jpg pairs, random 8:1:1 split with a fixed seed; only usable for smoke
   tests since the site labels of that export are not wired into this loader.

Paper protocol: resize 384x384 -> random crop 256x256 + vertical/horizontal
flips for training; resize 384 -> center crop 256 for inference (see below).
"""

import os
from collections import OrderedDict
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from configs import dataset_paths

IMAGE_SIZE = 256          # paper: random crop 256
RESIZE_SIZE = 384         # paper: resize before crop


def train_transform():
    return A.Compose([
        A.Resize(RESIZE_SIZE, RESIZE_SIZE),
        A.RandomCrop(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})


def test_transform():
    return A.Compose([
        A.Resize(RESIZE_SIZE, RESIZE_SIZE),
        A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})


class CardiacPreparedDataset(Dataset):
    """Frames from prepared per-patient npz files (see module docstring).

    npz files are opened lazily with a small LRU cache (training shuffles
    across patients, so caching the last few volumes matters).
    """

    def __init__(self, split_dir, transform=None, cache_size=16):
        self.transform = transform
        self.split_dir = Path(split_dir)
        self.files = sorted(self.split_dir.glob("*.npz"))
        assert self.files, "no prepared npz files under {}".format(split_dir)

        self.index = []                       # (file_idx, frame_idx)
        for fi, path in enumerate(self.files):
            with np.load(path) as z:
                n = int(z["images"].shape[0])
            self.index.extend([(fi, j) for j in range(n)])
        assert self.index, "prepared split {} has no frames".format(split_dir)
        self._cache = OrderedDict()
        self._cache_size = cache_size

    def _patient(self, fi):
        if fi in self._cache:
            self._cache.move_to_end(fi)
            return self._cache[fi]
        with np.load(self.files[fi]) as z:
            data = (z["images"], z["masks"])
        self._cache[fi] = data
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return data

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, j = self.index[idx]
        images, masks = self._patient(fi)
        image = images[j]
        mask = masks[j]
        # grayscale frame -> (H, W, 3): the model patch embedding expects the
        # paper's 3 input channels, and albumentations/ToTensorV2 would
        # otherwise keep the tensor 2-D and lose the channel dim in batching
        image = np.stack([image] * 3, axis=-1)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        return image, mask.long()


class CardiacSliceDatasetJpg(Dataset):
    """jpg frame-slice export; frames paired by identical file stem (fallback)."""

    def __init__(self, root, transform=None):
        self.transform = transform
        self.root = Path(root)
        image_dir = self.root / 'image'
        mask_dir = self.root / 'mask'

        masks = {Path(f).stem: f for f in os.listdir(mask_dir) if f.lower().endswith('.jpg')}
        images = sorted(f for f in os.listdir(image_dir) if f.lower().endswith('.jpg'))
        self.pairs = []
        for img in images:
            stem = Path(img).stem
            if stem in masks:
                self.pairs.append((image_dir / img, mask_dir / masks[stem]))
        assert self.pairs, "no image/mask pairs found under {}".format(root)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        image = np.array(Image.open(image_path).convert('RGB')).astype(np.uint8)
        mask = np.array(Image.open(mask_path)).astype(np.uint8)
        mask[mask > 0] = 1
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        return image, mask.long()


def _loader(dataset, batch_size, shuffle, num_workers, seed=None):
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    def _worker_init(worker_id):
        s = torch.initial_seed() % (2 ** 32)
        np.random.seed(s)
        import random
        random.seed(s)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, generator=generator,
                      worker_init_fn=_worker_init, drop_last=False)


def build_dataloaders(site, exp, prepared_root=None, batch_size=16,
                      num_workers=8, seed=42, epoch=0):
    """Build (train, validation, test) loaders for one hospital site.

    ``exp`` provides ``site_splits`` (site -> {train, val, test} folder names)
    and ``data_root``; ``prepared_root`` defaults to
    configs.dataset_paths.CARDIACUDC_PREPARED_ROOT.  The train shuffle
    generator is seeded with ``seed + epoch`` so that a resumed run
    reproduces the same shuffling per epoch.
    """
    prepared_root = Path(prepared_root or dataset_paths.CARDIACUDC_PREPARED_ROOT)
    if site not in exp["site_splits"]:
        raise ValueError("unknown site {!r} (choices: {})".format(
            site, list(exp["site_splits"])))
    split = exp["site_splits"][site]
    dirs = {k: prepared_root / v for k, v in split.items()}
    missing = [str(d) for d in dirs.values() if not d.is_dir()]
    if missing:
        raise FileNotFoundError(
            "prepared CardiacUDA splits missing: {}.  Run:\n"
            "  python scripts/prepare_cardiacudc.py --root {} --out {}".format(
                missing, exp["data_root"], prepared_root))

    train_set = CardiacPreparedDataset(dirs["train"], train_transform())
    val_set = CardiacPreparedDataset(dirs["val"], test_transform())
    test_set = CardiacPreparedDataset(dirs["test"], test_transform())

    train_loader = _loader(train_set, batch_size, True, num_workers,
                           seed=seed + epoch)
    val_loader = _loader(val_set, 1, False, num_workers)
    test_loader = _loader(test_set, 1, False, num_workers)
    return train_loader, val_loader, test_loader


def build_jpg_dataloaders(root=None, batch_size=16, num_workers=4, seed=42,
                          test_batch_size=1):
    """Fallback jpg-export loaders with a random 8:1:1 split (smoke tests only)."""
    root = Path(root) if root is not None else Path(dataset_paths.CARDIACUDC_ROOT)
    assert (root / 'image').is_dir() and (root / 'mask').is_dir(), \
        "expected jpg export layout <root>/{{image,mask}} under {}".format(root)

    train_full = CardiacSliceDatasetJpg(root, train_transform())
    eval_full = CardiacSliceDatasetJpg(root, test_transform())
    n_total = len(eval_full)
    n_val = int(round(n_total * 0.1))
    n_test = int(round(n_total * 0.1))
    n_train = n_total - n_val - n_test
    sizes = [n_train, n_val, n_test]
    train_set, _, _ = torch.utils.data.random_split(
        train_full, sizes, generator=torch.Generator().manual_seed(seed))
    _, val_set, test_set = torch.utils.data.random_split(
        eval_full, sizes, generator=torch.Generator().manual_seed(seed))

    train_loader = _loader(train_set, batch_size, True, num_workers, seed=seed)
    val_loader = _loader(val_set, test_batch_size, False, num_workers)
    test_loader = _loader(test_set, test_batch_size, False, num_workers)
    return train_loader, val_loader, test_loader
