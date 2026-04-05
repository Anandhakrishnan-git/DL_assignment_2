"""Localization modules
"""

import torch
import torch.nn as nn

from .layers import CustomDropout
from .vgg11 import VGG11Encoder

class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.gap = nn.AdaptiveAvgPool2d((7,7))  # Global average pooling to reduce spatial dimensions
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 4),
            nn.Sigmoid()  # Output normalized bbox coordinates in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
        """
        feats = self.encoder(x)
        feats = self.gap(feats)
        feats= self.regressor(feats)
        x_center, y_center, w, h = feats.unbind(dim=1)
        
        # Convert normalized coordinates to pixel coordinates
        _,_, H, W = x.shape
        x_center = x_center * W
        y_center = y_center * H
        w = w * W
        h = h * H
        bboxes = torch.stack([x_center, y_center, w, h], dim=1)

        return bboxes

if __name__ == "__main__":
    model = VGG11Localizer(in_channels=3, dropout_p=0.5)
    input_tensor = torch.randn(1, 3, 224, 224)  # Example input
    output = model(input_tensor)
    print("Output shape:", output.shape)
    print("Predicted bounding box:", output)
