"""Segmentation model
"""

import torch
import torch.nn as nn

from .layers import CustomDropout
from .vgg11 import VGG11Encoder

class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Decoder: upsample + concat skip + convs
        self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        bottleneck, feats = self.encoder(x, return_features=True)

        d1 = self.up1(bottleneck)
        f5 = feats["block5"]
        d1 = self.dec1(torch.cat([d1, f5], dim=1))

        d2 = self.up2(d1)
        f4 = feats["block4"]
        d2 = self.dec2(torch.cat([d2, f4], dim=1))

        d3 = self.up3(d2)
        f3 = feats["block3"]
        d3 = self.dec3(torch.cat([d3, f3], dim=1))

        d4 = self.up4(d3)
        f2 = feats["block2"]
        d4 = self.dec4(torch.cat([d4, f2], dim=1))

        d5 = self.up5(d4)
        f1 = feats["block1"]
        d5 = self.dec5(torch.cat([d5, f1], dim=1))

        return self.head(d5)

if __name__ == "__main__":
    model = VGG11UNet(num_classes=3, in_channels=3, dropout_p=0.5)
    input_tensor = torch.randn(1, 3, 224, 224)  # Example input
    output = model(input_tensor)
    print("Output shape:", output.shape)
