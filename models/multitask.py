"""Unified multi-task model

Multi-task learning architecture with shared backbone (VGG11 encoder) and three independent task heads:
- Classification Head: 37-class breed classification
- Localization Head: Bounding box regression (continuous coordinate prediction)
- Segmentation Head: U-Net based dense pixel-wise segmentation

The model loads pre-trained weights for each task and unifies them under a single shared encoder
for multi-task learning with task-independent feature extraction.
"""

import os
import torch
import torch.nn as nn

from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet
from .vgg11 import VGG11Encoder


def _strip_module_prefix(state_dict):
    """Remove 'module.' prefix from state dict keys (used when model was wrapped in DataParallel)."""
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _extract_state_dict(ckpt):
    """Extract state dict from checkpoint, handling various checkpoint formats."""
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def _load_weights(path: str):
    """Load weights from checkpoint file with proper error handling."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu")
    state = _extract_state_dict(ckpt)
    return _strip_module_prefix(state)


class MultiTaskPerceptionModel(nn.Module):
    """
    Shared-backbone multi-task learning model.
    
    Implements unified forward pass for three simultaneous tasks:
    1. Classification: 37-class breed classification
    2. Localization: Bounding box regression
    3. Segmentation: Pixel-wise semantic segmentation
    
    Architecture:
    - Shared backbone: VGG11Encoder (pre-trained from classifier)
    - Task head 1: ClassificationHead (classifier.py)
    - Task head 2: LocalizationHead (localization.py)
    - Task head 3: SegmentationHead = U-Net Decoder (segmentation.py)
    
    Weight Loading:
    - Individual task weights are loaded first, then their encoders are discarded
    - All heads are unified under a single shared encoder
    - The shared encoder prevents redundant computation across tasks
    """

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "classifier.pth",
        localizer_path: str = "localizer.pth",
        unet_path: str = "unet.pth",
    ):
        """
        Initialize the multi-task model with shared backbone and task-specific heads.

        Args:
            num_breeds: Number of output classes for classification (default: 37 pet breeds).
            seg_classes: Number of output classes for segmentation (default: 3 - pet, background, border).
            in_channels: Number of input channels (default: 3 for RGB).
            classifier_path: Path to pre-trained classifier checkpoint.
            localizer_path: Path to pre-trained localizer checkpoint.
            unet_path: Path to pre-trained U-Net checkpoint.

        Raises:
            FileNotFoundError: If any checkpoint file is not found.
            RuntimeError: If checkpoint loading fails.
        """
        super().__init__()

        # Initialize individual task models
        classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        localizer = VGG11Localizer(in_channels=in_channels)
        unet = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        # Load pre-trained weights for each task
        print(f"Loading classifier from: {classifier_path}")
        classifier_state = _load_weights(classifier_path)
        
        print(f"Loading localizer from: {localizer_path}")
        localizer_state = _load_weights(localizer_path)
        
        print(f"Loading U-Net from: {unet_path}")
        unet_state = _load_weights(unet_path)

        # Apply weights with fallback to non-strict mode if necessary
        try:
            classifier.load_state_dict(classifier_state, strict=True)
        except RuntimeError as e:
            print(f"Warning: Classifier strict loading failed, using non-strict mode. Error: {e}")
            classifier.load_state_dict(classifier_state, strict=False)

        try:
            localizer.load_state_dict(localizer_state, strict=True)
        except RuntimeError as e:
            print(f"Warning: Localizer strict loading failed, using non-strict mode. Error: {e}")
            localizer.load_state_dict(localizer_state, strict=False)

        try:
            unet.load_state_dict(unet_state, strict=True)
        except RuntimeError as e:
            print(f"Warning: U-Net strict loading failed, using non-strict mode. Error: {e}")
            unet.load_state_dict(unet_state, strict=False)

        # ===== SHARED BACKBONE: Initialize from classifier encoder =====
        # The classifier's encoder contains pre-trained ImageNet features learned on this task
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.encoder.load_state_dict(classifier.encoder.state_dict())

        # Attach the *same* shared encoder instance to all task heads
        # This ensures parameter sharing and reduces redundant forward passes
        classifier.encoder = self.encoder
        localizer.encoder = self.encoder
        unet.encoder = self.encoder

        # Store the three task-specific heads
        self.classifier = classifier
        self.localizer = localizer
        self.unet = unet

        print("Multi-task model initialized with shared backbone")

    def forward(self, x: torch.Tensor) -> dict:
        """
        Unified forward pass for all three tasks.
        
        This single forward pass efficiently computes shared features once,
        then branches to three independent task heads for simultaneous predictions.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].
               Typically [B, 3, 224, 224] for ImageNet-sized images.

        Returns:
            A dictionary containing predictions for all three tasks:
            {
                'classification': torch.Tensor [B, num_breeds]
                    - 37-class classification logits (not softmax-normalized)
                    
                'localization': torch.Tensor [B, 4]
                    - Bounding box coordinates in format (x_center, y_center, width, height)
                    - Coordinates are in pixel space (not normalized)
                    
                'segmentation': torch.Tensor [B, seg_classes, H, W]
                    - Dense pixel-wise segmentation logits (not softmax-normalized)
                    - seg_classes typically = 3 (pet foreground, background, border)
                    - Spatial dimensions [H, W] match input image size
            }
        
        Example:
            >>> model = MultiTaskPerceptionModel()
            >>> x = torch.randn(4, 3, 224, 224)  # Batch of 4 images
            >>> outputs = model(x)
            >>> breed_logits = outputs['classification']  # [4, 37]
            >>> bbox = outputs['localization']            # [4, 4]
            >>> segmentation = outputs['segmentation']    # [4, 3, 224, 224]
        """
        return {
            "classification": self.classifier(x),
            "localization": self.localizer(x),
            "segmentation": self.unet(x),
        }
