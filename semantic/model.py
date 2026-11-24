"""The pretrained LR-ASPP model and its offline inference checkpoint."""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.segmentation import (
    LRASPP_MobileNet_V3_Large_Weights,
    lraspp_mobilenet_v3_large,
)

from .data import CLASS_NAMES, INPUT_SIZE, NUM_CLASSES, normalized_image

ARCHITECTURE = "lraspp_mobilenet_v3_large"
PRETRAINED = LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1


def build_pretrained():
    model = lraspp_mobilenet_v3_large(weights=PRETRAINED)
    for name in ("low_classifier", "high_classifier"):
        classifier = getattr(model.classifier, name)
        setattr(model.classifier, name, nn.Conv2d(classifier.in_channels, NUM_CLASSES, 1))
    return model


def choose_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check PyTorch and the WSL NVIDIA driver, or use CPU.")
    if requested == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    return torch.device(requested)


def load_checkpoint(path, device):
    if not path.is_file():
        raise FileNotFoundError(f"Trained CamVid checkpoint not found at {path}. Run training first.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("architecture") != ARCHITECTURE
            or checkpoint.get("classes") != CLASS_NAMES
            or tuple(checkpoint.get("input_size", ())) != INPUT_SIZE):
        raise ValueError(f"Incompatible CamVid checkpoint: {path}")
    model = lraspp_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=NUM_CLASSES
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


@torch.inference_mode()
def predict_mask(model, rgb, device):
    inputs = torch.from_numpy(normalized_image(rgb)).unsqueeze(0).to(device)
    logits = model(inputs)["out"]
    # Upsample logits to the original frame before choosing the winning class.
    logits = F.interpolate(logits, size=(rgb.height, rgb.width), mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)[0].byte().cpu().numpy()
