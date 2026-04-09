"""Reusable custom layers."""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """
    Custom Dropout layer implementing element-wise dropout.
    
    Uses inverted dropout scaling (scales by 1/(1-p) during training so that
    no scaling is needed during inference).
    
    NOT using torch.nn.Dropout or torch.nn.functional.dropout.
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability (probability of dropping a neuron).
               Must be in [0.0, 1.0).
        """
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError("Dropout probability p must be in [0.0, 1.0).")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor for shape [B, C, H, W].

        Returns:
            Output tensor.
        """
        if (not self.training) or self.p == 0.0:
            return x
        
        # Probability of keeping an element
        keep_prob = 1.0 - self.p
        
        # Create binary mask: same shape as input
        # Bernoulli distribution: 1 with prob=keep_prob, 0 with prob=p
        mask = (torch.rand_like(x) < keep_prob).to(x.dtype)
        
        # Apply mask and scale by 1/keep_prob (inverted dropout)
        return x * mask / keep_prob
