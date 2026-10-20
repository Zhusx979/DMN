#!/usr/bin/env python
# coding: utf-8
"""Build the public-release R-domain splits from the prepared npz files.

The Kaggle release of cardiacUDC_dataset is incomplete relative to the draft
paper's Table 4 (R: 545/365/260 frames from Site_R_126 / Site_R_73 / Site_R_52):

  * ``Site_R_73`` ships images but NO label files (0/70), so it cannot serve
    as the validation split;
  * ``Site_R_126`` ships 85 of 126 videos and ``patient-73-4`` has a truncated
    (uncorruptable) label inside the official split archive;
  * ``Site_R_52`` (260 frames, test) is complete and matches the draft.

Workaround used here: carve a video-level hold-out out of Site_R_126 --
``--val-patients`` patients (default 15, deterministic ``seed`` selection over
sorted npz names) become ``prepared/UDAR_val``, the rest become
``prepared/UDAR_train`` (symlinks, so ``--re-extract`` stays consistent);
test stays ``Site_R_52``.  The draft counts 545/365 are unreproducible with
the public files; the deviation is recorded in configs/experiments.py and
docs/PAPER_MAPPING.md.

Usage:
    python scripts/make_r_holdout.py [--prepared data/cardiacudc/prepared]
                                     [--val-patients 15] [--seed 42] [--dry-run]
"""

import argparse
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", default="data/cardiacudc/prepared")
    parser.add_argument("--train-src", default="Site_R_126",
                        help="prepared site folder to carve train/val from")
    parser.add_argument("--val-patients", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prepared = Path(args.prepared) if Path(args.prepared).is_absolute() else REPO_ROOT / args.prepared
    src = prepared / args.train_src
    files = sorted(p.name for p in src.glob("*.npz"))
    assert len(files) > args.val_patients, "not enough npz files in {}".format(src)

    rng = random.Random(args.seed)
    val = sorted(rng.sample(files, args.val_patients))
    train = [f for f in files if f not in set(val)]

    val_dir, train_dir = prepared / "UDAR_val", prepared / "UDAR_train"
    print("train patients: {} -> {}\nval patients:   {} -> {}".format(
        len(train), train_dir, len(val), val_dir))
    for p in val:
        print("  [val]", p)

    def frames_of(path):
        import numpy as np
        with np.load(path) as z:
            return int(z["images"].shape[0])

    n_f_train = sum(frames_of(src / f) for f in train)
    n_f_val = sum(frames_of(src / f) for f in val)
    print("frames: train {} | val {} (draft: 545/365 -- public release lacks "
          "Site_R_73 labels)".format(n_f_train, n_f_val))
    if args.dry_run:
        return

    for d in (train_dir, val_dir):
        if d.exists():
            for link in d.glob("*.npz"):
                link.unlink()
        else:
            d.mkdir(parents=True)
    for name, names in ((train_dir, train), (val_dir, val)):
        for f in names:
            link = name / f
            target = (src / f).resolve()
            if not link.exists() and not link.is_symlink():
                link.symlink_to(target)
    print("symlink trees written: {} ({}), {} ({})".format(
        train_dir, len(train), val_dir, len(val)))


if __name__ == "__main__":
    main()
