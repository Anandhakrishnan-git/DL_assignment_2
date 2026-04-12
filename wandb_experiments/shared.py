"""Shared utilities for W&B experiment scripts."""

from __future__ import annotations

import inspect
import os
import random
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def add_repo_root_to_path() -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AlbumentationsTransform:
    def __init__(
        self,
        train: bool,
        size: int = IMAGE_SIZE,
        mean: Tuple[float, float, float] = IMAGENET_MEAN,
        std: Tuple[float, float, float] = IMAGENET_STD,
    ):
        if train:
            rrc = _build_random_resized_crop(
                size=size,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.33),
                p=1.0,
            )
            self.transform = A.Compose(
                [
                    rrc,
                    A.HorizontalFlip(p=0.5),
                    A.Affine(
                        translate_percent=(-0.05, 0.05),
                        scale=(0.95, 1.05),
                        rotate=(-10, 10),
                        p=0.3,
                    ),
                    A.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.05,
                        p=0.3,
                    ),
                    A.CoarseDropout(
                        num_holes_range=(1, 8),
                        hole_height_range=(0.04, 0.1),
                        hole_width_range=(0.04, 0.1),
                        fill=0,
                        p=0.2,
                    ),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(height=size, width=size),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ]
            )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image, dtype=np.uint8)
        return self.transform(image=arr)["image"]


def _build_random_resized_crop(
    size: int,
    scale: Tuple[float, float],
    ratio: Tuple[float, float],
    p: float,
):
    sig = inspect.signature(A.RandomResizedCrop)
    params = sig.parameters
    if "height" in params and "width" in params:
        return A.RandomResizedCrop(
            height=size,
            width=size,
            scale=scale,
            ratio=ratio,
            p=p,
        )
    if "size" in params:
        return A.RandomResizedCrop(
            size=(size, size),
            scale=scale,
            ratio=ratio,
            p=p,
        )
    return A.RandomResizedCrop(size, size, scale=scale, ratio=ratio, p=p)


class SegmentationTargetTransform:
    def __init__(self, size: int = IMAGE_SIZE):
        self.size = size

    def __call__(self, targets):
        out = dict(targets)
        if "segmentation" in out:
            mask = out["segmentation"]
            if not isinstance(mask, np.ndarray):
                mask = np.asarray(mask, dtype=np.int64)
            mask_img = Image.fromarray(mask.astype(np.uint8))
            mask_img = mask_img.resize((self.size, self.size), resample=Image.NEAREST)
            out["segmentation"] = torch.tensor(
                np.array(mask_img, dtype=np.int64, copy=True),
                dtype=torch.long,
            )
        return out


class LocalizationTargetTransform:
    def __init__(self, size: int = IMAGE_SIZE):
        self.size = float(size)

    def __call__(self, targets):
        out = dict(targets)
        if "localization" in out:
            bbox = np.asarray(out["localization"], dtype=np.float32).reshape(4)
            if float(np.max(np.abs(bbox))) <= 1.5:
                bbox = bbox * self.size
            out["localization"] = torch.tensor(bbox, dtype=torch.float32)
        return out


def build_dataloaders(
    task: str,
    root: str,
    batch_size: int,
    num_workers: int,
    use_augmentation: bool = True,
    train_subset: Optional[int] = None,
    val_subset: Optional[int] = None,
):
    add_repo_root_to_path()
    from data.pets_dataset import OxfordIIITPetDataset

    if task == "classification":
        train_transform = AlbumentationsTransform(train=use_augmentation)
        val_transform = AlbumentationsTransform(train=False)
        target_transform = None
        tasks = ("category",)
        pin_memory = torch.cuda.is_available()
    elif task == "segmentation":
        train_transform = AlbumentationsTransform(train=False)
        val_transform = AlbumentationsTransform(train=False)
        target_transform = SegmentationTargetTransform(size=IMAGE_SIZE)
        tasks = ("segmentation",)
        pin_memory = False
    elif task == "localization":
        train_transform = AlbumentationsTransform(train=False)
        val_transform = AlbumentationsTransform(train=False)
        target_transform = LocalizationTargetTransform(size=IMAGE_SIZE)
        tasks = ("localization",)
        pin_memory = torch.cuda.is_available()
    else:
        raise ValueError(f"Unknown task: {task}")

    train_set = OxfordIIITPetDataset(
        root=root,
        split="train",
        tasks=tasks,
        transform=train_transform,
        target_transform=target_transform,
    )
    val_set = OxfordIIITPetDataset(
        root=root,
        split="val",
        tasks=tasks,
        transform=val_transform,
        target_transform=target_transform,
    )

    if train_subset is not None and train_subset > 0 and train_subset < len(train_set):
        train_set = Subset(train_set, list(range(int(train_subset))))
    if val_subset is not None and val_subset > 0 and val_subset < len(val_set):
        val_set = Subset(val_set, list(range(int(val_subset))))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def to_numpy_image(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized CHW tensor to uint8 HWC for visualization."""
    if img_tensor.ndim != 3 or img_tensor.shape[0] != 3:
        raise ValueError("Expected image tensor with shape [3, H, W].")
    img = img_tensor.detach().cpu().float()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = img * std + mean
    img = torch.clamp(img, 0.0, 1.0)
    img = (img * 255.0).byte().permute(1, 2, 0).numpy()
    return img


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Colorize trimap mask (0/1/2) for logging."""
    if mask.ndim != 2:
        raise ValueError("Expected mask with shape [H, W].")
    colors = np.array(
        [
            [0, 0, 0],       # background
            [0, 255, 0],     # pet
            [255, 165, 0],   # border
        ],
        dtype=np.uint8,
    )
    mask = np.clip(mask, 0, colors.shape[0] - 1).astype(np.int64)
    return colors[mask]


def xywh_to_xyxy(box_xywh: Sequence[float]) -> Tuple[float, float, float, float]:
    x_c, y_c, w, h = [float(v) for v in box_xywh]
    x1 = x_c - w / 2.0
    y1 = y_c - h / 2.0
    x2 = x_c + w / 2.0
    y2 = y_c + h / 2.0
    return x1, y1, x2, y2


def clamp_box_xyxy(
    box_xyxy: Sequence[float],
    w: int,
    h: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    x1 = int(max(0, min(w - 1, round(x1))))
    y1 = int(max(0, min(h - 1, round(y1))))
    x2 = int(max(0, min(w - 1, round(x2))))
    y2 = int(max(0, min(h - 1, round(y2))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def draw_boxes_xywh(
    image_rgb: np.ndarray,
    gt_xywh: Optional[Sequence[float]] = None,
    pred_xywh: Optional[Sequence[float]] = None,
    gt_color: Tuple[int, int, int] = (0, 255, 0),
    pred_color: Tuple[int, int, int] = (255, 0, 0),
    width: int = 3,
) -> Image.Image:
    pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil)
    w, h = pil.size
    if gt_xywh is not None:
        x1, y1, x2, y2 = clamp_box_xyxy(xywh_to_xyxy(gt_xywh), w=w, h=h)
        draw.rectangle([x1, y1, x2, y2], outline=gt_color, width=width)
    if pred_xywh is not None:
        x1, y1, x2, y2 = clamp_box_xyxy(xywh_to_xyxy(pred_xywh), w=w, h=h)
        draw.rectangle([x1, y1, x2, y2], outline=pred_color, width=width)
    return pil


def compute_iou_xywh(
    pred_xywh: torch.Tensor,
    target_xywh: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred_xywh = pred_xywh.float()
    target_xywh = target_xywh.float()

    px, py, pw, ph = pred_xywh.unbind(dim=-1)
    tx, ty, tw, th = target_xywh.unbind(dim=-1)

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
    return torch.clamp(inter_area / (union + eps), 0.0, 1.0)


def dice_score_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_index: int = 1,
    eps: float = 1e-6,
) -> float:
    """Dice for a single class (default: foreground=1) over a batch."""
    preds = torch.argmax(logits, dim=1)
    preds_fg = (preds == class_index).to(torch.float32)
    targets_fg = (targets == class_index).to(torch.float32)
    inter = torch.sum(preds_fg * targets_fg).item()
    denom = torch.sum(preds_fg).item() + torch.sum(targets_fg).item()
    return float((2.0 * inter + eps) / (denom + eps))


def feature_map_grid(
    act: torch.Tensor,
    num_maps: int = 16,
    grid_cols: int = 4,
) -> Image.Image:
    """Convert [1, C, H, W] feature maps to a tiled grayscale grid."""
    if act.ndim != 4 or act.shape[0] != 1:
        raise ValueError("Expected activation tensor with shape [1, C, H, W].")
    c = int(act.shape[1])
    num_maps = int(min(num_maps, c))
    grid_cols = int(max(1, grid_cols))
    grid_rows = int((num_maps + grid_cols - 1) // grid_cols)

    maps = act[0, :num_maps].detach().cpu().float().numpy()
    tiles: List[Image.Image] = []
    for m in maps:
        vmin = float(m.min())
        vmax = float(m.max())
        if vmax - vmin < 1e-8:
            norm = np.zeros_like(m, dtype=np.uint8)
        else:
            norm = (255.0 * (m - vmin) / (vmax - vmin)).astype(np.uint8)
        tiles.append(Image.fromarray(norm, mode="L"))

    tile_w, tile_h = tiles[0].size
    grid = Image.new("L", (grid_cols * tile_w, grid_rows * tile_h))
    for idx, tile in enumerate(tiles):
        r = idx // grid_cols
        c0 = idx % grid_cols
        grid.paste(tile, (c0 * tile_w, r * tile_h))
    return grid


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        if all(isinstance(key, str) for key in checkpoint.keys()):
            return checkpoint
    return checkpoint


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def load_model_weights(model: nn.Module, path: str, strict: bool = True) -> None:
    ckpt = torch.load(path, map_location="cpu")
    state = _strip_module_prefix(_extract_state_dict(ckpt))
    model.load_state_dict(state, strict=strict)


def load_encoder_from_classifier_checkpoint(
    encoder_sequential: nn.Module,
    classifier_ckpt_path: str,
) -> None:
    ckpt = torch.load(classifier_ckpt_path, map_location="cpu")
    state = _strip_module_prefix(_extract_state_dict(ckpt))
    encoder_state = {
        k.replace("encoder.", ""): v
        for k, v in state.items()
        if isinstance(k, str) and k.startswith("encoder.")
    }
    encoder_sequential.load_state_dict(encoder_state, strict=False)


def freeze_vgg11_encoder_blocks(
    encoder: nn.Module,
    trainable_blocks: Sequence[int],
) -> None:
    """Freeze all VGG11 conv blocks except those in trainable_blocks (1..5)."""
    if not hasattr(encoder, "encoder"):
        raise ValueError("Expected VGG11Encoder-like module with .encoder Sequential.")

    # Block definitions correspond to models/vgg11.py layout.
    blocks = {
        1: list(range(0, 3)),        # conv1/bn/relu
        2: list(range(4, 7)),        # conv2/bn/relu (pool is idx 3)
        3: list(range(8, 14)),       # conv3a..relu3b (pool is idx 7)
        4: list(range(15, 21)),      # conv4a..relu4b (pool is idx 14)
        5: list(range(22, 28)),      # conv5a..relu5b (pool is idx 21)
    }

    trainable_blocks = set(int(b) for b in trainable_blocks)

    for block_id, module_indices in blocks.items():
        trainable = block_id in trainable_blocks
        for idx in module_indices:
            module = encoder.encoder[idx]
            for p in module.parameters(recurse=True):
                p.requires_grad = trainable

    # Keep BN in frozen blocks in eval mode (prevents running-stat updates).
    for module in encoder.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()


def enable_custom_dropout_only(model: nn.Module) -> None:
    """Enable CustomDropout modules during inference (MC Dropout style)."""
    add_repo_root_to_path()
    from models.layers import CustomDropout

    model.eval()
    for module in model.modules():
        if isinstance(module, CustomDropout):
            module.train()


