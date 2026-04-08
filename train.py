"""Training entrypoint for Oxford-IIIT Pet classification."""

import argparse
import inspect
import os
import time
from typing import Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data.pets_dataset import OxfordIIITPetDataset
from losses import IoULoss
from models import VGG11Classifier, VGG11UNet, VGG11Localizer

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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

                    #A.RandomGamma(gamma_limit=(80, 120), p=0.2),
                    #A.GaussianBlur(blur_limit=(3, 5), p=0.1),
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
    """Handle Albumentations API changes between versions."""
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


def build_dataloaders(
    root: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    overfit_subset: int = 0,
    use_augmentation: bool = True,
):
    base_set = OxfordIIITPetDataset(
        root=root,
        split="trainval",
        tasks=("category",),
        transform=None,
    )
    val_size = int(len(base_set) * val_ratio)
    train_size = len(base_set) - val_size
    generator = torch.Generator().manual_seed(42)
    train_split, val_split = random_split(base_set, [train_size, val_size], generator=generator)

    train_indices = list(train_split.indices)
    val_indices = list(val_split.indices)

    if overfit_subset > 0:
        subset_size = min(overfit_subset, len(train_indices))
        subset_indices = train_indices[:subset_size]
        train_indices = subset_indices
        val_indices = subset_indices

    train_transform = AlbumentationsTransform(train=use_augmentation)
    val_transform = AlbumentationsTransform(train=False)

    train_set = Subset(
        OxfordIIITPetDataset(
            root=root,
            split="trainval",
            tasks=("category",),
            transform=train_transform,
        ),
        train_indices,
    )
    val_set = Subset(
        OxfordIIITPetDataset(
            root=root,
            split="trainval",
            tasks=("category",),
            transform=val_transform,
        ),
        val_indices,
    )

    pin_memory = torch.cuda.is_available()
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


class SegmentationTargetTransform:
    """Convert segmentation masks to resized LongTensor targets."""

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
            # Force an owning, contiguous CPU array before tensor conversion.
            mask_arr = np.array(mask_img, dtype=np.int64, copy=True)
            out["segmentation"] = torch.tensor(mask_arr, dtype=torch.long)
        return out


class LocalizationTargetTransform:
    """Convert bbox targets to float tensors in IMAGE_SIZE pixel coordinates."""

    def __init__(self, size: int = IMAGE_SIZE):
        self.size = float(size)

    def __call__(self, targets):
        out = dict(targets)
        if "localization" in out:
            bbox = np.asarray(out["localization"], dtype=np.float32).reshape(4)

            # Dataset stores normalized xywh in [0, 1]; convert to pixel space.
            # This keeps training targets aligned with model outputs in pixels.
            if float(np.max(np.abs(bbox))) <= 1.5:
                bbox = bbox * self.size

            out["localization"] = torch.tensor(bbox, dtype=torch.float32)
        return out


def build_segmentation_dataloaders(
    root: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    overfit_subset: int = 0,
):
    """Build segmentation dataloaders with aligned image/mask resizing."""
    base_set = OxfordIIITPetDataset(
        root=root,
        split="trainval",
        tasks=("segmentation",),
        transform=None,
        target_transform=None,
    )
    val_size = int(len(base_set) * val_ratio)
    train_size = len(base_set) - val_size
    generator = torch.Generator().manual_seed(42)
    train_split, val_split = random_split(base_set, [train_size, val_size], generator=generator)

    train_indices = list(train_split.indices)
    val_indices = list(val_split.indices)

    if overfit_subset > 0:
        subset_size = min(overfit_subset, len(train_indices))
        subset_indices = train_indices[:subset_size]
        train_indices = subset_indices
        val_indices = subset_indices

    image_transform = AlbumentationsTransform(train=False)
    target_transform = SegmentationTargetTransform(size=IMAGE_SIZE)

    train_set = Subset(
        OxfordIIITPetDataset(
            root=root,
            split="trainval",
            tasks=("segmentation",),
            transform=image_transform,
            target_transform=target_transform,
        ),
        train_indices,
    )
    val_set = Subset(
        OxfordIIITPetDataset(
            root=root,
            split="trainval",
            tasks=("segmentation",),
            transform=image_transform,
            target_transform=target_transform,
        ),
        val_indices,
    )

    # Keep pin_memory off for segmentation on Windows/CUDA to avoid
    # intermittent "cudaErrorAlreadyMapped" from DataLoader pin threads.
    pin_memory = False
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


def build_localization_dataloaders(
    root: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    overfit_subset: int = 0,
):
    """Build localization dataloaders with aligned image/bbox resizing."""
    base_set = OxfordIIITPetDataset(
        root=root,
        split="trainval",
        tasks=("localization",),
        transform=None,
        target_transform=None,
    )
    val_size = int(len(base_set) * val_ratio)
    train_size = len(base_set) - val_size
    generator = torch.Generator().manual_seed(42)
    train_split, val_split = random_split(base_set, [train_size, val_size], generator=generator)

    train_indices = list(train_split.indices)
    val_indices = list(val_split.indices)

    if overfit_subset > 0:
        subset_size = min(overfit_subset, len(train_indices))
        subset_indices = train_indices[:subset_size]
        train_indices = subset_indices
        val_indices = subset_indices

    image_transform = AlbumentationsTransform(train=False)
    target_transform = LocalizationTargetTransform(size=IMAGE_SIZE)

    train_set = Subset(
        OxfordIIITPetDataset(
            root=root,
            split="trainval",
            tasks=("localization",),
            transform=image_transform,
            target_transform=target_transform,
        ),
        train_indices,
    )
    val_set = Subset(
        OxfordIIITPetDataset(
            root=root,
            split="trainval",
            tasks=("localization",),
            transform=image_transform,
            target_transform=target_transform,
        ),
        val_indices,
    )

    pin_memory = torch.cuda.is_available()
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


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> int:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).sum().item()


def _init_classifier_weights(module: nn.Module) -> None:
    """Initialize VGG-style modules for ReLU activations."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def initialize_model_weights(model: VGG11Classifier) -> None:
    """Apply custom initialization and soften the logits layer scale."""
    model.apply(_init_classifier_weights)
    final_layer = model.classifier[-1]
    if isinstance(final_layer, nn.Linear):
        nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
        if final_layer.bias is not None:
            nn.init.zeros_(final_layer.bias)


def _grad_norm_l2(model: nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        param_norm = param.grad.detach().norm(2).item()
        total += param_norm * param_norm
    return total**0.5


def _unnormalize_image(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
):
    """Convert a normalized CHW tensor to a displayable HWC numpy image."""
    img = tensor.detach().cpu().float().permute(1, 2, 0).numpy()
    img = img * np.array(std, dtype=np.float32) + np.array(mean, dtype=np.float32)
    return np.clip(img, 0.0, 1.0)


def visualize_augmentations(
    root: str,
    num_images: int = 8,
    seed: int = 42,
    save_path: str = "augmented_samples.png",
    show: bool = False,
):
    """Visualize a few augmented samples from the training pipeline."""
    rng = np.random.default_rng(seed)
    transform = AlbumentationsTransform(train=True)
    dataset = OxfordIIITPetDataset(
        root=root,
        split="trainval",
        tasks=("category",),
        transform=transform,
    )

    num_images = max(1, min(num_images, len(dataset)))
    indices = rng.choice(len(dataset), size=num_images, replace=False)

    cols = min(4, num_images)
    rows = int(np.ceil(num_images / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = axes.reshape(rows, 1)

    for idx, sample_idx in enumerate(indices):
        image, label = dataset[int(sample_idx)]
        img = _unnormalize_image(image)
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.imshow(img)
        ax.set_title(f"label: {label}")
        ax.axis("off")

    # Hide any unused subplots.
    for idx in range(num_images, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved augmentation preview to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def visualize_augmentation_variants(
    root: str,
    num_variants: int = 8,
    image_index: int = -1,
    seed: int = 42,
    save_path: str = "augmented_variants.png",
    show: bool = False,
):
    """Visualize multiple augmented variants of the same image."""
    rng = np.random.default_rng(seed)
    base_dataset = OxfordIIITPetDataset(
        root=root,
        split="trainval",
        tasks=("category",),
        transform=None,
    )
    if len(base_dataset) == 0:
        print("Dataset is empty; skipping augmentation variant visualization.")
        return

    if image_index < 0 or image_index >= len(base_dataset):
        image_index = int(rng.integers(len(base_dataset)))

    base_image, label = base_dataset[image_index]
    base_transform = AlbumentationsTransform(train=False)
    aug_transform = AlbumentationsTransform(train=True)

    variants = [base_transform(base_image)]
    for _ in range(max(1, num_variants)):
        variants.append(aug_transform(base_image))

    total = len(variants)
    cols = min(4, total)
    rows = int(np.ceil(total / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = axes.reshape(rows, 1)

    for idx, tensor in enumerate(variants):
        img = _unnormalize_image(tensor)
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        title = "base" if idx == 0 else f"aug {idx}"
        ax.imshow(img)
        ax.set_title(f"{title} | label: {label}")
        ax.axis("off")

    for idx in range(total, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved augmentation variants to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def _rand_bbox(size, lam: float):
    """Generate CutMix bounding box."""
    width = size[3]
    height = size[2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_rat)
    cut_h = int(height * cut_rat)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    bbx1 = int(np.clip(cx - cut_w // 2, 0, width))
    bby1 = int(np.clip(cy - cut_h // 2, 0, height))
    bbx2 = int(np.clip(cx + cut_w // 2, 0, width))
    bby2 = int(np.clip(cy + cut_h // 2, 0, height))
    return bbx1, bby1, bbx2, bby2


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    log_interval: int,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    mix_prob: float = 1.0,
    debug_stats: bool = False,
    debug_batches: int = 0,
):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0
    start_time = time.perf_counter()

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, dtype=torch.long, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        use_mix = (
            (mixup_alpha > 0.0 or cutmix_alpha > 0.0)
            and np.random.rand() < mix_prob
        )
        if use_mix and (mixup_alpha > 0.0 or cutmix_alpha > 0.0):
            use_cutmix = False
            if mixup_alpha > 0.0 and cutmix_alpha > 0.0:
                use_cutmix = np.random.rand() < 0.5
            elif cutmix_alpha > 0.0:
                use_cutmix = True

            if use_cutmix:
                lam = np.random.beta(cutmix_alpha, cutmix_alpha)
                rand_index = torch.randperm(images.size(0), device=images.device)
                bbx1, bby1, bbx2, bby2 = _rand_bbox(images.size(), lam)
                images[:, :, bby1:bby2, bbx1:bbx2] = images[rand_index, :, bby1:bby2, bbx1:bbx2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
                labels_a, labels_b = labels, labels[rand_index]
            else:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                rand_index = torch.randperm(images.size(0), device=images.device)
                images = lam * images + (1 - lam) * images[rand_index]
                labels_a, labels_b = labels, labels[rand_index]
            logits = model(images)
            loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
        else:
            logits = model(images)
            loss = criterion(logits, labels)
        loss.backward()

        if debug_stats and batch_idx <= debug_batches:
            grad_norm = _grad_norm_l2(model)
            logit_std = logits.detach().std().item()
            feat_std = float("nan")
            if hasattr(model, "encoder") and hasattr(model, "avgpool"):
                with torch.no_grad():
                    feats = model.avgpool(model.encoder(images))
                    feat_std = feats.std().item()
            print(
                f"  debug batch {batch_idx} "
                f"- grad_norm: {grad_norm:.6f} "
                f"logits_std: {logit_std:.6f} "
                f"feats_std: {feat_std:.6f}"
            )
        optimizer.step()


        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_correct += _accuracy(logits, labels)
        total += batch_size

        if log_interval > 0 and batch_idx % log_interval == 0:
            elapsed = time.perf_counter() - start_time
            samples_per_sec = total / max(1e-6, elapsed)
            avg_loss = running_loss / max(1, total)
            avg_acc = running_correct / max(1, total)
            print(
                f"  batch {batch_idx}/{len(loader)} "
                f"- loss: {avg_loss:.4f} acc: {avg_acc:.4f} "
                f"({samples_per_sec:.1f} samples/s)"
            )

    
    avg_loss = running_loss / max(1, total)
    avg_acc = running_correct / max(1, total)

    return avg_loss, avg_acc


def evaluate(model, loader, criterion, device: torch.device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, dtype=torch.long, non_blocking=device.type == "cuda")

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_correct += _accuracy(logits, labels)
            total += batch_size

    avg_loss = running_loss / max(1, total)
    avg_acc = running_correct / max(1, total)
    return avg_loss, avg_acc


def _update_segmentation_confusion(
    confusion: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> None:
    preds = preds.detach().view(-1).to(dtype=torch.int64)
    targets = targets.detach().view(-1).to(dtype=torch.int64)
    valid = (targets >= 0) & (targets < num_classes)
    if valid.any():
        encoded = (num_classes * targets[valid] + preds[valid]).cpu()
        bincount = torch.bincount(encoded, minlength=num_classes * num_classes)
        confusion += bincount.view(num_classes, num_classes)


def _mean_iou_from_confusion(confusion: torch.Tensor, eps: float = 1e-6) -> float:
    conf = confusion.to(dtype=torch.float32)
    true_pos = torch.diag(conf)
    false_pos = conf.sum(dim=0) - true_pos
    false_neg = conf.sum(dim=1) - true_pos
    denom = true_pos + false_pos + false_neg
    valid = denom > 0
    if not valid.any():
        return 0.0
    iou = true_pos[valid] / (denom[valid] + eps)
    return float(iou.mean().item())


def train_one_epoch_segmentation(
    model: VGG11UNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_interval: int,
    num_classes: int,
):
    model.train()
    model.encoder.eval()  # keep frozen encoder BN/dropout behavior fixed
    running_loss = 0.0
    total_samples = 0
    pixel_correct = 0
    total_pixels = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    start_time = time.perf_counter()

    for batch_idx, (images, masks) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, dtype=torch.long, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        pixel_correct += (preds == masks).sum().item()
        total_pixels += masks.numel()
        _update_segmentation_confusion(confusion, preds, masks, num_classes)

        if log_interval > 0 and batch_idx % log_interval == 0:
            elapsed = time.perf_counter() - start_time
            samples_per_sec = total_samples / max(1e-6, elapsed)
            avg_loss = running_loss / max(1, total_samples)
            pixel_acc = pixel_correct / max(1, total_pixels)
            mean_iou = _mean_iou_from_confusion(confusion)
            print(
                f"  batch {batch_idx}/{len(loader)} "
                f"- loss: {avg_loss:.4f} pix_acc: {pixel_acc:.4f} "
                f"miou: {mean_iou:.4f} "
                f"({samples_per_sec:.1f} samples/s)"
            )

    avg_loss = running_loss / max(1, total_samples)
    pixel_acc = pixel_correct / max(1, total_pixels)
    mean_iou = _mean_iou_from_confusion(confusion)
    return avg_loss, pixel_acc, mean_iou


def evaluate_segmentation(
    model: VGG11UNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    pixel_correct = 0
    total_pixels = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            masks = masks.to(device, dtype=torch.long, non_blocking=device.type == "cuda")

            logits = model(images)
            loss = criterion(logits, masks)
            preds = torch.argmax(logits, dim=1)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            pixel_correct += (preds == masks).sum().item()
            total_pixels += masks.numel()
            _update_segmentation_confusion(confusion, preds, masks, num_classes)

    avg_loss = running_loss / max(1, total_samples)
    pixel_acc = pixel_correct / max(1, total_pixels)
    mean_iou = _mean_iou_from_confusion(confusion)
    return avg_loss, pixel_acc, mean_iou


class LocalizationRegressionLoss(nn.Module):
    """Combined localization loss: MSE + IoU."""

    def __init__(
        self,
        mse_weight: float = 1.0,
        iou_weight: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.iou_weight = float(iou_weight)
        self.mse = nn.MSELoss(reduction=reduction)
        self.iou = IoULoss(reduction=reduction)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.mse_weight * self.mse(preds, targets) + self.iou_weight * self.iou(
            preds, targets
        )


def train_one_epoch_localization(
    model,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    log_interval: int,
):
    model.train()
    running_loss = 0.0
    total_samples = 0
    start_time = time.perf_counter()

    for batch_idx, (images, bboxes) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        bboxes = bboxes.to(device, dtype=torch.float32, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        preds = model(images)
        loss = criterion(preds, bboxes)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        if log_interval > 0 and batch_idx % log_interval == 0:
            elapsed = time.perf_counter() - start_time
            samples_per_sec = total_samples / max(1e-6, elapsed)
            avg_loss = running_loss / max(1, total_samples)
            print(
                f"  batch {batch_idx}/{len(loader)} "
                f"- loss: {avg_loss:.4f} "
                f"({samples_per_sec:.1f} samples/s)"
            )

    avg_loss = running_loss / max(1, total_samples)
    return avg_loss


def evaluate_localization(
    model,
    loader,
    criterion,
    device: torch.device,
):
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, bboxes in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            bboxes = bboxes.to(device, dtype=torch.float32, non_blocking=device.type == "cuda")

            preds = model(images)
            loss = criterion(preds, bboxes)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = running_loss / max(1, total_samples)
    return avg_loss


def _extract_state_dict(checkpoint) -> dict:
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        if "model_state_dict" in checkpoint and isinstance(
            checkpoint["model_state_dict"], dict
        ):
            return checkpoint["model_state_dict"]
        if all(isinstance(key, str) for key in checkpoint.keys()):
            return checkpoint
    raise ValueError("Unsupported checkpoint format; expected a state_dict-like mapping.")


def _strip_module_prefix(state_dict: dict) -> dict:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def load_and_freeze_encoder_from_classifier(
    model: VGG11UNet,
    classifier_path: str,
) -> None:
    if not os.path.isfile(classifier_path):
        raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_path}")

    checkpoint = torch.load(classifier_path, map_location="cpu")
    state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))
    encoder_state = {
        key: value for key, value in state_dict.items() if key.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError(
            "No encoder.* keys found in classifier checkpoint; cannot initialize UNet encoder."
        )

    incompatible = model.load_state_dict(encoder_state, strict=False)
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.encoder.eval()

    print(
        f"Loaded {len(encoder_state)} encoder tensors from {classifier_path} "
        f"and froze encoder parameters."
    )
    if incompatible.unexpected_keys:
        print(f"Unexpected keys while loading encoder: {incompatible.unexpected_keys}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        choices=("classification", "segmentation", "localization"),
        default="classification",
        help="Task to train.",
    )
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=("adam", "adamw"),
        default="adamw",
        help="Optimizer type.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        choices=("none", "cosine", "plateau"),
        default="cosine",
        help="Learning rate scheduler.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum LR for cosine schedule.",
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=3,
        help="Plateau scheduler patience (epochs).",
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=0.5,
        help="Plateau scheduler decay factor.",
    )
    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=0.0,
        help="MixUp alpha (0 disables).",
    )
    parser.add_argument(
        "--cutmix_alpha",
        type=float,
        default=0.0,
        help="CutMix alpha (0 disables).",
    )
    parser.add_argument(
        "--mix_prob",
        type=float,
        default=1.0,
        help="Probability to apply MixUp/CutMix per batch.",
    )
    parser.add_argument(
        "--dropout_mode",
        type=str,
        choices=("element", "channel", "spatial"),
        default="channel",
        help="CustomDropout mode for the classifier head.",
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--save_path", type=str, default="classifier.pth")
    parser.add_argument(
        "--classifier_path",
        type=str,
        default="classifier.pth",
        help="Classifier checkpoint for encoder initialization in segmentation mode.",
    )
    parser.add_argument(
        "--seg_num_classes",
        type=int,
        default=3,
        help="Number of segmentation classes.",
    )
    parser.add_argument(
        "--loc_mse_weight",
        type=float,
        default=1.0,
        help="Weight for localization MSE term.",
    )
    parser.add_argument(
        "--loc_iou_weight",
        type=float,
        default=1.0,
        help="Weight for localization IoU term.",
    )
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument(
        "--disable_aug",
        action="store_true",
        help="Disable training data augmentation.",
    )
    parser.add_argument(
        "--overfit_subset",
        type=int,
        default=0,
        help="If >0, train/val on a tiny subset to sanity-check overfitting.",
    )
    parser.add_argument(
        "--debug_stats",
        action="store_true",
        help="Print grad/logit/feature stats for the first few batches.",
    )
    parser.add_argument(
        "--debug_batches",
        type=int,
        default=3,
        help="Number of initial batches to print debug stats for.",
    )
    parser.add_argument(
        "--aug_vis",
        type=int,
        default=0,
        help="Number of augmented samples to visualize (0 disables).",
    )
    parser.add_argument(
        "--aug_vis_path",
        type=str,
        default="augmented_samples.png",
        help="Path to save the augmentation visualization grid.",
    )
    parser.add_argument(
        "--aug_vis_show",
        action="store_true",
        help="Display the augmentation grid interactively.",
    )
    parser.add_argument(
        "--aug_vis_only",
        action="store_true",
        help="Only generate augmentation visualization and exit.",
    )
    parser.add_argument(
        "--aug_vis_variants",
        type=int,
        default=0,
        help="Number of augmented variants of a single image to visualize (0 disables).",
    )
    parser.add_argument(
        "--aug_vis_variants_path",
        type=str,
        default="augmented_variants.png",
        help="Path to save the single-image augmentation grid.",
    )
    parser.add_argument(
        "--aug_vis_index",
        type=int,
        default=-1,
        help="Image index for variant visualization (-1 picks random).",
    )
    return parser.parse_args()


def run_classification_training(args, device: torch.device):
    ran_visuals = False
    if args.aug_vis > 0:
        visualize_augmentations(
            root=args.data_root,
            num_images=args.aug_vis,
            save_path=args.aug_vis_path,
            show=args.aug_vis_show,
        )
        ran_visuals = True

    if args.aug_vis_variants > 0:
        visualize_augmentation_variants(
            root=args.data_root,
            num_variants=args.aug_vis_variants,
            image_index=args.aug_vis_index,
            save_path=args.aug_vis_variants_path,
            show=args.aug_vis_show,
        )
        ran_visuals = True

    if args.aug_vis_only and ran_visuals:
        return

    train_loader, val_loader = build_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        overfit_subset=args.overfit_subset,
        use_augmentation=not args.disable_aug,
    )
    print(
        f"Train size: {len(train_loader.dataset)} | "
        f"Val size: {len(val_loader.dataset)} | "
        f"Batches per epoch: {len(train_loader)}"
    )

    model = VGG11Classifier(
        num_classes=37,
        dropout_p=0.5,
        dropout_mode=args.dropout_mode,
    )
    initialize_model_weights(model)
    model = model.to(device)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
        )

    criterion = nn.CrossEntropyLoss()
    best_acc = -1.0
    print(f"Using device: {device}")
    print("Starting classification training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.log_interval,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            mix_prob=args.mix_prob,
            debug_stats=args.debug_stats,
            debug_batches=args.debug_batches,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"- train: {train_loss:.4f} acc: {train_acc:.4f} "
            f"- val: {val_loss:.4f} acc: {val_acc:.4f} "
            f"- lr: {current_lr:.2e}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            if args.save_path:
                save_dir = os.path.dirname(args.save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), args.save_path)

    if args.save_path:
        print(f"Best checkpoint saved to {args.save_path} (val acc: {best_acc:.4f})")


def run_segmentation_training(args, device: torch.device):
    if args.save_path == "classifier.pth":
        args.save_path = "unet.pth"

    train_loader, val_loader = build_segmentation_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        overfit_subset=args.overfit_subset,
    )
    print(
        f"Train size: {len(train_loader.dataset)} | "
        f"Val size: {len(val_loader.dataset)} | "
        f"Batches per epoch: {len(train_loader)}"
    )

    model = VGG11UNet(num_classes=args.seg_num_classes)
    model.apply(_init_classifier_weights)
    load_and_freeze_encoder_from_classifier(model, args.classifier_path)
    model = model.to(device)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found for segmentation training.")

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
        )

    criterion = nn.CrossEntropyLoss()
    best_miou = -1.0
    print(f"Using device: {device}")
    print("Starting segmentation training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_pix_acc, train_miou = train_one_epoch_segmentation(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            log_interval=args.log_interval,
            num_classes=args.seg_num_classes,
        )
        val_loss, val_pix_acc, val_miou = evaluate_segmentation(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=args.seg_num_classes,
        )

        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"- train: {train_loss:.4f} pix_acc: {train_pix_acc:.4f} miou: {train_miou:.4f} "
            f"- val: {val_loss:.4f} pix_acc: {val_pix_acc:.4f} miou: {val_miou:.4f} "
            f"- lr: {current_lr:.2e}"
        )

        if val_miou > best_miou:
            best_miou = val_miou
            if args.save_path:
                save_dir = os.path.dirname(args.save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), args.save_path)

    if args.save_path:
        print(f"Best checkpoint saved to {args.save_path} (val mIoU: {best_miou:.4f})")


def run_localization_training(args, device: torch.device):
    if args.save_path == "classifier.pth":
        args.save_path = "localizer.pth"

    train_loader, val_loader = build_localization_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        overfit_subset=args.overfit_subset,
    )
    print(
        f"Train size: {len(train_loader.dataset)} | "
        f"Val size: {len(val_loader.dataset)} | "
        f"Batches per epoch: {len(train_loader)}"
    )

    model = VGG11Localizer()
    model.apply(_init_classifier_weights)
    model = model.to(device)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
        )

    criterion = LocalizationRegressionLoss(
        mse_weight=args.loc_mse_weight,
        iou_weight=args.loc_iou_weight,
    )
    best_loss = float('inf')
    print(f"Using device: {device}")
    print("Starting localization training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch_localization(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            log_interval=args.log_interval,
        )
        val_loss = evaluate_localization(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"- train: {train_loss:.4f} "
            f"- val: {val_loss:.4f} "
            f"- lr: {current_lr:.2e}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            if args.save_path:
                save_dir = os.path.dirname(args.save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), args.save_path)

    if args.save_path:
        print(f"Best checkpoint saved to {args.save_path} (val loss: {best_loss:.4f})")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.task == "segmentation":
        run_segmentation_training(args, device)
        return
    elif args.task == "localization":
        run_localization_training(args, device)
        return

    run_classification_training(args, device)


if __name__ == "__main__":
    main()
