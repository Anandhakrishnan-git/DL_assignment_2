"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer with configurable masking mode."""

    def __init__(
        self,
        p: float = 0.5,
        mode: str = "element",
        scale: bool = True,
    ):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability.
            mode: Masking mode. One of {"element", "channel", "spatial"}.
                - element: elementwise dropout (default).
                - channel: drop entire channels per sample.
                - spatial: drop spatial locations across all channels.
            scale: If True, scale by 1 / (1 - p) to preserve expectation.
        """
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError("Dropout probability p must be in [0.0, 1.0).")
        if mode not in {"element", "channel", "spatial"}:
            raise ValueError("mode must be one of: element, channel, spatial.")
        self.p = p
        self.mode = mode
        self.scale = scale

    def _mask_shape(self, x: torch.Tensor) -> torch.Size:
        if self.mode == "element":
            return x.shape
        if self.mode == "channel":
            if x.dim() < 2:
                return x.shape
            return (x.size(0), x.size(1), *([1] * (x.dim() - 2)))
        if x.dim() < 3:
            raise ValueError("spatial mode expects input with dim >= 3.")
        return (x.size(0), 1, *x.shape[2:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        if (not self.training) or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask_shape = self._mask_shape(x)
        mask = (torch.rand(mask_shape, device=x.device) < keep_prob).to(x.dtype)
        if self.scale:
            return x * mask / keep_prob
        return x * mask


class ChannelDropout(CustomDropout):
    """Channel-wise dropout (drops entire channels per sample)."""

    def __init__(self, p: float = 0.5, scale: bool = True):
        super().__init__(p=p, mode="channel", scale=scale)


class SpatialDropout(CustomDropout):
    """Spatial dropout (drops spatial locations across all channels)."""

    def __init__(self, p: float = 0.5, scale: bool = True):
        super().__init__(p=p, mode="spatial", scale=scale)


class GaussianDropout(nn.Module):
    """Multiplicative Gaussian dropout."""

    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Dropout probability (controls noise variance).
        """
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError("Dropout probability p must be in [0.0, 1.0).")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p == 0.0:
            return x
        std = (self.p / (1.0 - self.p)) ** 0.5
        noise = torch.randn_like(x) * std + 1.0
        return x * noise

if __name__ == "__main__":
    # Example usage
    dropout = CustomDropout(p=0.3, mode="element")
    dropout.train()  # Set to training mode
    input_tensor = torch.randn(2, 3, 4, 4)  # Example input
    output_tensor = dropout(input_tensor)
    print("Input Tensor:\n", input_tensor)
    print("Output Tensor after CustomDropout:\n", output_tensor)
