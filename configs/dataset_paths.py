"""Central dataset path registry for the paper experiments.

Every path below was verified on the shared server (2026-10-16).  Run

    python configs/dataset_paths.py

from the repository root to print the current existence status of all
datasets used by the paper experiments.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Published paper -- classification (Table 1: OCT / pneumonia / breast cancer)
# ---------------------------------------------------------------------------
# Retinal diseases in OCT, 4 classes (normal / DME / CNV / drusen), open
# source (Kermany et al. 2018).  Published split 4000 / 1000 / 1000, matched
# exactly by the verified lists under data_lists/zk_oct (see
# datasets/classification/auxiliary_datasets.py).
ZK_OCT_ROOT = "/ssd/data/medical/zk_OCT"

# Pneumonia chest X-ray, 2 classes (normal / pneumonia), open source
# (Kermany et al. 2018).  Published split 4632 / 600 / 624, matched exactly
# by the verified lists under data_lists/zk_pneumonia.
ZK_PNEUMONIA_ROOT = "/ssd/data/medical/zk_pneumonia"

# DeepBC breast-cancer ultrasound (3 BIRADS-based classes; Dazhou Central
# Hospital -- the paper's private breast-cancer dataset).  Label lists below,
# images under DEEPBC_DATA_DIR.  Published split 16008 / 1800 / 4452.
DEEPBC_ROOT = str(REPO_ROOT / "data" / "DeepBC")
DEEPBC_TRAIN_TXT = f"{DEEPBC_ROOT}/label_classification/training.txt"
DEEPBC_VAL_TXT = f"{DEEPBC_ROOT}/label_classification/val.txt"
DEEPBC_TEST_TXT = f"{DEEPBC_ROOT}/label_classification/test.txt"
DEEPBC_DATA_DIR = f"{DEEPBC_ROOT}/data"

# TBX11K tuberculosis chest X-ray, 3 classes.  NOT part of the published
# version; kept for the pre-publication experiments.  datasets/classification/
# tbx11k.py builds the pre-publication split 1920 / 240 / 240 (seed 911).
TBX11K_ROOT = os.environ.get(
    "TBX11K_ROOT", str(REPO_ROOT / "data" / "TBX11K" / "imgs"))

# ---------------------------------------------------------------------------
# Research-line segmentation experiments (not in the published version)
# ---------------------------------------------------------------------------
# BRISC brain MRI, 2 classes.  In this repository the dataset lives under
# data/BRISC/{train,test}; the official test split holds 860 images and the
# research line carves it into 430 validation / 430 test (seed 42, see
# datasets/segmentation/brisc.py).  Official train count is 3933.
BRISC_ROOT = str(REPO_ROOT / "data" / "BRISC")

# CardiacUDA heart ultrasound, 5 classes (4 foreground structures + background;
# draft-paper Table 4).  The dataset was introduced by GraphEcho (Yang et al.,
# ICCV 2023, arXiv:2309.11145) and is publicly available as
# "cardiacUDC_dataset":
#     https://www.kaggle.com/datasets/xiaoweixumedicalai/cardiacudc-dataset
#     (4.2 GB, CC BY-NC 4.0; code: github.com/xmed-lab/GraphEcho)
# The official nii volumes were downloaded on 2026-10-16 (Kaggle bearer
# token) into data/cardiacudc/ and are converted once by
# scripts/prepare_cardiacudc.py into data/cardiacudc/prepared (per-patient
# npz, research-code frame filter mask.sum() >= 2000 + per-frame max
# normalization).  Draft site splits (485/145/105 for G, 545/365/260 for R)
# map onto the Site_* folders -- see CARDIAC_UDA site_splits in
# configs/experiments.py.
# Besides the nii copy there is also a jpg frame export on this machine at
# /data1/lixiang/US/015_heart_3d22d_seg whose file-name id equals the idx
# column of its csv; that export is only used as a smoke-test fallback.
CARDIACUDC_ROOT = "/data1/lixiang/US/015_heart_3d22d_seg"
CARDIACUDC_NII_ROOT = str(REPO_ROOT / "data" / "cardiacudc" / "cardiacUDC_dataset")
CARDIACUDC_PREPARED_ROOT = str(REPO_ROOT / "data" / "cardiacudc" / "prepared")
CARDIACUDC_SPLIT_CSV = f"{CARDIACUDC_ROOT}/cardiacudc-dataset.csv"

# ---------------------------------------------------------------------------
# Auxiliary / non-paper datasets
# ---------------------------------------------------------------------------
# Glaucoma classification lists (non-paper extra) are defined inside
# datasets/classification/auxiliary_datasets.py, which points at /ssd paths.

DATASETS = {
    "zk_OCT (paper: retinal diseases, 4-class)": ZK_OCT_ROOT,
    "zk_pneumonia (paper: pneumonia, 2-class)": ZK_PNEUMONIA_ROOT,
    "DeepBC (paper: breast cancer, 3-class)": DEEPBC_ROOT,
    "TBX11K (pre-publication, 3-class)": TBX11K_ROOT,
    "BRISC (segmentation, 2-class)": BRISC_ROOT,
    "CardiacUDA jpg export (segmentation)": CARDIACUDC_ROOT,
    "CardiacUDA official split csv": CARDIACUDC_SPLIT_CSV,
    "CardiacUDA official nii (segmentation)": CARDIACUDC_NII_ROOT,
    "CardiacUDA prepared npz (segmentation)": CARDIACUDC_PREPARED_ROOT,
}


def report():
    """Print [OK]/[MISSING] for every dataset path; returns the status dict."""
    statuses = {}
    for name, path in DATASETS.items():
        ok = bool(path) and Path(path).exists()
        statuses[name] = ok
        print(f"[{'OK' if ok else 'MISSING'}] {name}: {path}")
    return statuses


if __name__ == "__main__":
    report()
