"""Segmentation model registry shared by train/test scripts (no heavy deps)."""

from configs.experiments import SEGMENTATION_BRISC, SEGMENTATION_CARDIAC_UDA
from networks.dual_mapping_network_segmentation import DualMappingViTSeg
from networks.synthesizer_vit_seg import SynthesizerViTSeg
from networks.unet import Unet
from networks.vit_seg import ViTSeg

EXPERIMENTS = {"BRISC": SEGMENTATION_BRISC, "CardiacUDA": SEGMENTATION_CARDIAC_UDA}
MODELS = ["DualMappingViTSeg", "ViTSeg", "SynthesizerViTSeg", "Unet"]


def build_model(name, exp):
    """Instantiate a segmentation model with the paper configuration.

    The synthesizer variant/k are read from ``exp`` (set by
    train_segmentation.py --variant); they only affect SynthesizerViTSeg.
    """
    if name == "DualMappingViTSeg":
        return DualMappingViTSeg(image_size=exp["image_size"],
                                 patch_size=exp["patch_size"],
                                 num_classes=exp["num_classes"],
                                 dim=exp["dim"],
                                 depth=exp["depth"],
                                 mlp_dim=exp["mlp_dim"],
                                 channels=exp["in_channels"])
    if name == "ViTSeg":
        return ViTSeg(image_size=exp["image_size"],
                      patch_size=exp["patch_size"],
                      num_classes=exp["num_classes"],
                      dim=exp["dim"],
                      depth=exp["depth"],
                      heads=16,
                      mlp_dim=exp["mlp_dim"],
                      channels=exp["in_channels"])
    if name == "SynthesizerViTSeg":
        return SynthesizerViTSeg(image_size=exp["image_size"],
                                 patch_size=exp["patch_size"],
                                 num_classes=exp["num_classes"],
                                 dim=exp["dim"],
                                 depth=exp["depth"],
                                 mlp_dim=exp["mlp_dim"],
                                 heads=16,
                                 variant=exp.get("syn_variant", "random"),
                                 k=exp.get("syn_k", 8),
                                 channels=exp["in_channels"])
    if name == "Unet":
        return Unet()
    raise ValueError("Unknown model: {}".format(name))
