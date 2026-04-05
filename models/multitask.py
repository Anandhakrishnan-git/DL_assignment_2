"""Unified multi-task model
"""

import torch
import torch.nn as nn

from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet
from .vgg11 import VGG11Encoder


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def _load_weights(path: str):
    ckpt = torch.load(path, map_location="cpu")
    state = _extract_state_dict(ckpt)
    return _strip_module_prefix(state)


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "unet.pth"):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super().__init__()

        classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        localizer = VGG11Localizer(in_channels=in_channels)
        unet = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        classifier_state = _load_weights(classifier_path)
        localizer_state = _load_weights(localizer_path)
        unet_state = _load_weights(unet_path)

        try:
            classifier.load_state_dict(classifier_state, strict=True)
        except RuntimeError:
            classifier.load_state_dict(classifier_state, strict=False)

        try:
            localizer.load_state_dict(localizer_state, strict=True)
        except RuntimeError:
            localizer.load_state_dict(localizer_state, strict=False)

        try:
            unet.load_state_dict(unet_state, strict=True)
        except RuntimeError:
            unet.load_state_dict(unet_state, strict=False)

        # Shared backbone initialized from trained classifier encoder.
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.encoder.load_state_dict(classifier.encoder.state_dict())

        # Attach shared encoder to all heads.
        classifier.encoder = self.encoder
        localizer.encoder = self.encoder
        unet.encoder = self.encoder

        self.classifier = classifier
        self.localizer = localizer
        self.unet = unet

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        return {
            "classification": self.classifier(x),
            "localization": self.localizer(x),
            "segmentation": self.unet(x),
        }
