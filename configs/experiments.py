"""Paper experiment configurations.

Every hyper-parameter below is taken from the published paper "Efficient
Vision Transformer for Medical Image Analysis" (International Journal of
Neural Systems, 2024); the comment next to each entry states where it comes
from.  Training scripts import these dicts so the paper setup is defined in
exactly one place.

Classification (published version, Sections 4.1-4.3, Tables 1-3):
    input 224x224, default patch 16x16, default network depth 8, AdamW with
    initial lr 1e-4, 600 epochs; the learning rate is linearly warmed up
    during the first 30 epochs, followed by cosine decay for the remaining
    epochs.  Batch size is not stated in the paper; 16 (research
    implementation value) is used here.
    Reported final test accuracy (Table 2):
        Retinal diseases (OCT, 4-class)     ViT 0.8400 -> DM ViT 0.8580
        Pneumonia (X-ray, 2-class)          ViT 0.8349 -> DM ViT 0.8669
        Breast cancer (ultrasound, 3-class) ViT 0.7729 -> DM ViT 0.7776
    Table 3 ablations (patch 8/32, depth 4/16) are documented in
    docs/PAPER_MAPPING.md.

Segmentation (research-line extension; NOT part of the published version --
the published paper lists segmentation/detection as future work.  The
pre-publication draft version used these two datasets):
    ViT encoder (8 transformer layers) + lightweight conv/transpose-conv
    decoder + conv segmentation head; Adam lr 1e-3, weight decay 1e-4,
    momentum 0.9, 600 epochs, batch size 16; augmentation: resize 384x384 ->
    random crop 256x256 + horizontal/vertical flips; inference: center crop.
"""

from configs import dataset_paths as dp

# ---------------------------------------------------------------------------
# Classification (published paper, Section 4.1)
# ---------------------------------------------------------------------------

# Retinal diseases in OCT (published Table 1 row 1, Table 2 row 1): four
# classes -- normal / DME / CNV / drusen; open source (Kermany et al.,
# Cell 172(5) 2018).  The verified example split under data_lists/zk_oct
# matches 4000 / 1000 / 1000 exactly.
RETINAL_OCT_CLASSIFICATION = dict(
    name="oct",
    task="classification",
    description="Retinal diseases OCT, 4 classes (NORMAL / DRUSEN / DME / CNV)",
    data_root=dp.ZK_OCT_ROOT,
    data_lists="data_lists/zk_oct",
    loader="datasets/classification/image_list_dataset.py via "
           "datasets/classification/auxiliary_datasets.py (zk_oct_*)",
    num_classes=4,
    in_channels=3,
    image_size=224,          # paper
    patch_size=16,           # paper (default)
    dim=1024,                # paper (token embedding d_m)
    depth=8,                 # paper (default)
    mlp_dim=2048,            # paper
    dropout=0.0,
    emb_dropout=0.0,
    batch_size=16,           # not stated in the paper; research implementation value
    epochs=600,              # paper
    optimizer="adamw",       # paper
    lr=1e-4,                 # paper
    betas=(0.9, 0.999),
    weight_decay=0.01,       # paper (AdamW)
    warmup_epochs=30,        # paper (linear warmup, then cosine decay)
    seed=911,
    reg_weight=1.0,          # weight of L_reg (Eqs. 9-10), added to task loss
    split=(4000, 1000, 1000),  # paper Table 1
    paper_acc={"ViT": 0.8400, "DualMappingViT": 0.8580},  # paper Table 2
)

# Pneumonia in chest X-ray (published Table 1 row 2, Table 2 row 2): two
# classes -- normal / pneumonia; open source (Kermany et al., ref [44]).
# The verified example split under data_lists/zk_pneumonia matches
# 4632 / 600 / 624 exactly.
PNEUMONIA_CLASSIFICATION = dict(
    name="pneumonia",
    task="classification",
    description="Pneumonia chest X-ray, 2 classes (NORMAL / PNEUMONIA)",
    data_root=dp.ZK_PNEUMONIA_ROOT,
    data_lists="data_lists/zk_pneumonia",
    loader="datasets/classification/image_list_dataset.py via "
           "datasets/classification/auxiliary_datasets.py (zk_pneumonia_*)",
    num_classes=2,
    in_channels=3,
    image_size=224,          # paper
    patch_size=16,           # paper (default)
    dim=1024,                # paper
    depth=8,                 # paper (default)
    mlp_dim=2048,            # paper
    dropout=0.0,
    emb_dropout=0.0,
    batch_size=16,           # not stated in the paper; research implementation value
    epochs=600,              # paper
    optimizer="adamw",       # paper
    lr=1e-4,                 # paper
    betas=(0.9, 0.999),
    weight_decay=0.01,       # paper (AdamW)
    warmup_epochs=30,        # paper
    seed=911,
    reg_weight=1.0,
    split=(4632, 600, 624),  # paper Table 1
    paper_acc={"ViT": 0.8349, "DualMappingViT": 0.8669},  # paper Table 2
)

# Breast cancer in ultrasound (published Table 1 row 3, Table 2 row 3): three
# BIRADS-risk classes (low / medium / high); private, collected at Dazhou
# Central Hospital.  Local copy = DeepBC.
DEEPBC_CLASSIFICATION = dict(
    name="bc",
    task="classification",
    description="Breast-cancer ultrasound, 3 classes (BIRADS risk groups; "
                "Dazhou Central Hospital -- published Table 1 row 3)",
    data_root=dp.DEEPBC_DATA_DIR,
    label_dir=dp.DEEPBC_ROOT + "/label_classification",
    num_classes=3,
    in_channels=4,           # BC mask-crop pipeline returns 4 channels
    image_size=224,          # paper
    patch_size=16,           # paper (default)
    dim=1024,
    depth=8,
    mlp_dim=2048,
    dropout=0.0,
    emb_dropout=0.0,
    batch_size=16,
    epochs=600,
    optimizer="adamw",
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    warmup_epochs=30,
    seed=911,
    reg_weight=1.0,
    split=(16008, 1800, 4452),  # paper Table 1
    paper_acc={"ViT": 0.7729, "DualMappingViT": 0.7776},  # paper Table 2
    draft_acc={"ViT": 0.7729, "DualMappingViT": 0.7850},  # draft Table 2 (DM differs from published 0.7776)
    draft_table3={  # draft Table 3: patch/depth ablation test accuracies (one dim varied from defaults 16/8)
        ("patch", 8):  {"ViT": 0.7596, "DualMappingViT": 0.7715},
        ("patch", 32): {"ViT": 0.6716, "DualMappingViT": 0.6821},
        ("depth", 4):  {"ViT": 0.7023, "DualMappingViT": 0.7133},
        ("depth", 16): {"ViT": 0.7634, "DualMappingViT": 0.7754},
    },
)

# TBX11K tuberculosis chest X-ray, 3 classes (health / sick / tb).
# NOT part of the published version -- kept for reproducibility of the
# pre-publication experiments (accuracy below is from that draft).
TBX11K_CLASSIFICATION = dict(
    name="tbx11k",
    task="classification",
    description="Tuberculosis chest X-ray, 3 classes (pre-publication experiment)",
    data_root=dp.TBX11K_ROOT,
    num_classes=3,
    in_channels=3,
    image_size=224,
    patch_size=16,
    dim=1024,
    depth=8,
    mlp_dim=2048,
    dropout=0.0,
    emb_dropout=0.0,
    batch_size=16,
    epochs=600,
    optimizer="adamw",
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    warmup_epochs=30,
    seed=911,
    reg_weight=1.0,
    split=(1920, 240, 240),  # pre-publication split (draft Table 1)
    draft_acc={"ViT": 0.9250, "DualMappingViT": 0.9500},  # pre-publication draft (Table 2)
    draft_table3={  # draft Table 3: patch/depth ablation test accuracies (one dim varied from defaults 16/8)
        ("patch", 8):  {"ViT": 0.9397, "DualMappingViT": 0.9545},
        ("patch", 32): {"ViT": 0.9037, "DualMappingViT": 0.9351},
        ("depth", 4):  {"ViT": 0.9005, "DualMappingViT": 0.9233},
        ("depth", 16): {"ViT": 0.9336, "DualMappingViT": 0.9583},
    },
)

# ---------------------------------------------------------------------------
# Segmentation (research-line extension; not in the published version)
# ---------------------------------------------------------------------------

SEGMENTATION_BRISC = dict(
    name="BRISC",
    task="segmentation",
    description="Brain MRI stroke lesion, 2 classes (research-line extension)",
    data_root=dp.BRISC_ROOT,
    num_classes=2,
    in_channels=3,
    image_size=256,          # random crop 256 from resized 384
    resize=384,              # augmentation input size
    patch_size=16,
    dim=768,                 # research configuration for the seg encoder
    depth=8,                 # 8 transformer layers
    mlp_dim=2048,
    dropout=0.0,
    emb_dropout=0.0,
    batch_size=16,
    epochs=600,
    optimizer="adam",
    lr=1e-3,
    betas=(0.9, 0.999),      # momentum 0.9 == Adam beta1
    weight_decay=1e-4,
    seed=42,
    reg_weight=1.0,
    split=(3933, 430, 430),  # val/test carved from the official 860 test split
)

SEGMENTATION_CARDIAC_UDA = dict(
    name="CardiacUDA",
    task="segmentation",
    description="Cardiac ultrasound (CardiacUDC / GraphEcho ICCV 2023), "
                "hospital G/R domains, 5 classes (4 structures + background; "
                "draft-paper Table 4)",
    data_root=dp.CARDIACUDC_NII_ROOT,
    num_classes=5,           # LV / RV / LA / RA + background (draft Table 4)
    in_channels=3,
    image_size=256,
    resize=384,
    patch_size=16,
    dim=768,
    depth=8,
    mlp_dim=2048,
    dropout=0.0,
    emb_dropout=0.0,
    batch_size=16,
    epochs=600,
    optimizer="adam",
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
    seed=42,
    reg_weight=1.0,
    # Draft-paper split counts (labelled frames): G 485/145/105, R 545/365/260.
    # G maps onto the official Kaggle release exactly: Site_G_100/29/20 hold
    # 97/29/21 videos x 5 annotated frames = 485/145/105 (verified 2026-10-16).
    # R does NOT reproduce on the public release: Site_R_73 (draft val, 365
    # frames) ships 70 images but zero label files; Site_R_126 (draft train,
    # 545 frames) ships 85/126 videos of which 83 are usable (patient-73-4's
    # label is truncated in the archive, patient-60-4's mask is all-zero);
    # only Site_R_52 (test) is complete and matches the draft's 260 frames.
    # Workaround (scripts/make_r_holdout.py, seed 42): carve a 15-patient
    # video-level hold-out from Site_R_126 into UDAR_train/UDAR_val symlink
    # folders; test stays Site_R_52.  A separately prepared all-frame set
    # (prepared/label_all_frame, 8 patients / 428 frames) is available as an
    # alternative validation set but is not used by this split.  Deviation
    # recorded in docs/PAPER_MAPPING.md.
    site_splits=dict(
        G=dict(train="Site_G_100", val="Site_G_29", test="Site_G_20"),
        R=dict(train="UDAR_train", val="UDAR_val", test="Site_R_52"),
    ),
    draft_split_cardiac_uda_g=(485, 145, 105),
    draft_split_cardiac_uda_r=(545, 365, 260),
)

EXPERIMENTS = {
    "oct": RETINAL_OCT_CLASSIFICATION,
    "pneumonia": PNEUMONIA_CLASSIFICATION,
    "bc": DEEPBC_CLASSIFICATION,
    "tbx11k": TBX11K_CLASSIFICATION,
    "BRISC": SEGMENTATION_BRISC,
    "CardiacUDA": SEGMENTATION_CARDIAC_UDA,
}


def get(name):
    """Return a copy of the configuration dict for an experiment name."""
    return dict(EXPERIMENTS[name])
