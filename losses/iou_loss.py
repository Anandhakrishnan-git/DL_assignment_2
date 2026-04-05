"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of {'none', 'mean', 'sum'}")

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
        pred_boxes = pred_boxes.float()
        target_boxes = target_boxes.float()

        px, py, pw, ph = pred_boxes.unbind(dim=-1)
        tx, ty, tw, th = target_boxes.unbind(dim=-1)

        pw = torch.abs(pw)
        ph = torch.abs(ph)
        tw = torch.abs(tw)
        th = torch.abs(th)

        px1 = px - pw / 2.0
        py1 = py - ph / 2.0
        px2 = px + pw / 2.0
        py2 = py + ph / 2.0

        tx1 = tx - tw / 2.0
        ty1 = ty - th / 2.0
        tx2 = tx + tw / 2.0
        ty2 = ty + th / 2.0

        ix1 = torch.max(px1, tx1)
        iy1 = torch.max(py1, ty1)
        ix2 = torch.min(px2, tx2)
        iy2 = torch.min(py2, ty2)

        inter_w = torch.clamp(ix2 - ix1, min=0.0)
        inter_h = torch.clamp(iy2 - iy1, min=0.0)
        inter_area = inter_w * inter_h

        p_area = torch.clamp(px2 - px1, min=0.0) * torch.clamp(py2 - py1, min=0.0)
        t_area = torch.clamp(tx2 - tx1, min=0.0) * torch.clamp(ty2 - ty1, min=0.0)

        union = p_area + t_area - inter_area
        iou = inter_area / (union + self.eps)
        iou = torch.clamp(iou, min=0.0, max=1.0)
        loss = 1.0 - iou

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


if __name__ == "__main__":
    # Example usage
    pred_boxes = torch.tensor([[50, 50, 20, 20], [30, 30, 10, 10]], dtype=torch.float32)
    target_boxes = torch.tensor([[48, 48, 22, 22], [32, 32, 8, 8]], dtype=torch.float32)

    iou_loss = IoULoss()
    loss = iou_loss(pred_boxes, target_boxes)
    print(f"IoU Loss: {loss.item():.4f}")