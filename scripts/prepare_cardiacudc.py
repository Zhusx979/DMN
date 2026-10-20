#!/usr/bin/env python
# coding: utf-8
"""Convert the official CardiacUDC nii volumes into per-patient npz slices.

The official dataset (Kaggle ``xiaoweixumedicalai/cardiacudc-dataset``) ships
one nii volume per patient/video inside per-site folders::

    <root>/Site_<G|R>_<k>/<patient-id>_image.nii.gz
    <root>/Site_<G|R>_<k>/<patient-id>_label.nii.gz

This script reproduces the research-code frame extraction exactly
(segmentation/Datasets/cardiacudc_dataset.py of the original repo):

  * per frame: skip if the mask is constant or ``mask.sum() < 2000``;
  * image is normalized per frame by its own maximum: ``x / (x.max() + 1e-8)``.

Output::

    <out>/<Site_...>/<patient-id>.npz   (images float32 T'HW, masks uint8 T'HW)
    <out>/manifest.json                 (frame counts per site + per file)

The per-site frame counts are printed so they can be checked against the
draft-paper split table (G: 485/145/105, R: 545/365/260).

Usage:
    python scripts/prepare_cardiacudc.py --root data/cardiacudc/cardiacUDC_dataset \
        --out data/cardiacudc/prepared
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MIN_FOREGROUND = 2000  # research code threshold (segmentation/Datasets/cardiacudc_dataset.py)


def extract_patient(image_path, label_path, max_label=None):
    """Return (images, masks, slice_ids) for one patient volume.

    ``max_label`` optionally drops frames containing labels above the model's
    class range (the released ``label_all_frame`` folder has a few videos with
    annotation values 5/6 outside the 5-class label set).
    """
    import nibabel as nib

    image = nib.load(str(image_path)).get_fdata()
    mask = nib.load(str(label_path)).get_fdata()
    assert image.shape[2] == mask.shape[2], \
        f"slice-count mismatch {image_path.name} vs {label_path.name}"

    images, masks, ids, dropped = [], [], [], 0
    for i in range(mask.shape[2]):
        m = mask[:, :, i]
        if m.min() == m.max() or m.sum() < MIN_FOREGROUND:
            continue
        if max_label is not None and m.max() > max_label:
            dropped += 1
            continue
        x = image[:, :, i]
        x = x / (x.max() + 1e-8)
        images.append(x.astype(np.float32))
        masks.append(m.astype(np.uint8))
        ids.append(i)
    return images, masks, ids, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/cardiacudc/cardiacUDC_dataset",
                        help="extracted official dataset root (with Site_* folders)")
    parser.add_argument("--out", default="data/cardiacudc/prepared",
                        help="output root for per-patient npz files")
    parser.add_argument("--sites", nargs="*", default=None,
                        help="restrict to given folder names (default: all Site_*)")
    parser.add_argument("--max_label", type=int, default=None,
                        help="drop frames whose mask contains a label > max_label "
                             "(use 4 for the 5-class label set)")
    args = parser.parse_args()

    root = (REPO_ROOT / args.root) if not Path(args.root).is_absolute() else Path(args.root)
    out = (REPO_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    assert root.is_dir(), f"dataset root not found: {root}"

    sites = args.sites or sorted(p.name for p in root.iterdir()
                                 if p.is_dir() and p.name.startswith("Site_"))
    assert sites, f"no Site_* folders under {root}"

    manifest = {"root": str(root), "out": str(out), "min_foreground": MIN_FOREGROUND,
                "max_label": args.max_label, "sites": {}, "skipped": {}}
    for site in sites:
        site_dir = root / site
        out_dir = out / site
        out_dir.mkdir(parents=True, exist_ok=True)
        image_files = sorted(f for f in site_dir.iterdir()
                             if "image" in f.name and f.name.endswith((".nii", ".nii.gz")))
        site_frames, site_files = 0, 0
        for img_path in image_files:
            patient = img_path.name.split("_image")[0]
            lbl_path = None
            for cand in (img_path.with_name(img_path.name.replace("image", "label")),
                         img_path.with_name(img_path.name.replace("_image", "_label"))):
                if cand.exists():
                    lbl_path = cand
                    break
            if lbl_path is None:
                print(f"  [skip] no label for {img_path.name}")
                manifest["skipped"][f"{site}/{patient}"] = "no label file"
                continue
            npz_path = out_dir / (patient + ".npz")
            if npz_path.exists():
                with np.load(npz_path) as z:
                    n = int(z["images"].shape[0])
                site_frames += n
                site_files += 1
                continue
            try:
                images, masks, ids, dropped = extract_patient(img_path, lbl_path,
                                                              max_label=args.max_label)
            except Exception as e:
                print(f"  [skip] {img_path.name}: unreadable ({type(e).__name__}: {e})")
                manifest["skipped"][f"{site}/{patient}"] = f"{type(e).__name__}"
                continue
            if dropped:
                print(f"  [drop] {img_path.name}: {dropped} frames with labels > "
                      f"{args.max_label}")
            if not images:
                continue
            np.savez_compressed(npz_path,
                                images=np.stack(images),
                                masks=np.stack(masks),
                                slice_ids=np.asarray(ids, dtype=np.int32))
            site_frames += len(images)
            site_files += 1
            print(f"  [{site}] {npz_path.name}: {len(images)} frames "
                  f"({img_path.name})")
        manifest["sites"][site] = {"files": site_files, "frames": site_frames}
        print(f"[{site}] {site_files} patients, {site_frames} valid frames")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nSummary (valid frames per site, research-code filter >= "
          f"{MIN_FOREGROUND}):")
    for site, info in manifest["sites"].items():
        print(f"  {site}: {info['frames']} frames / {info['files']} patients")
    print(f"\nmanifest written to {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
