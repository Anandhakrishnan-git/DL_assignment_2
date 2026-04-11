"""Training entrypoint for Oxford-IIIT Pet classification."""

import argparse
import inspect
import os
import time
from typing import Tuple

import numpy as np
from PIL import Image
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data.pets_dataset import OxfordIIITPetDataset
from losses import IoULoss
from models import VGG11Classifier, VGG11UNet, VGG11Localizer


# ============================================================================
# CONFIGURATION
# ============================================================================

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================================
# AUGMENTATION: Image transformation with train/eval modes
# ============================================================================

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


# ============================================================================
# ADDITIONAL TRANSFORMS: Target-specific transformations
# ============================================================================

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


# ============================================================================
# DATA LOADING
# ============================================================================

def build_dataloaders(
    task: str,
    root: str,
    batch_size: int,
    num_workers: int,
    use_augmentation: bool = True,
):
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
        pin_memory = False  # Avoid issues on Windows/CUDA
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


# ============================================================================
# METRICS & UTILITIES
# ============================================================================

def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> int:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).sum().item()


# ============================================================================
# CUTMIX/MIXUP: Data augmentation utilities
# ============================================================================

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


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_one_epoch(
    task: str,
    model,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    log_interval: int,
    num_classes: int = 0,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    mix_prob: float = 1.0,
):
    model.train()
    running_loss = 0.0
    total = 0
    start_time = time.perf_counter()
    running_correct = 0
    running_pixel_correct = 0
    total_pixels = 0
    confusion = None

    if task == "segmentation":
        confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        if hasattr(model, "encoder"):
            model.encoder.eval()

    for batch_idx, batch in enumerate(loader, start=1):
        images = batch[0].to(device, non_blocking=device.type == "cuda")
        targets = batch[1]

        if task == "classification":
            targets = targets.to(device, dtype=torch.long, non_blocking=device.type == "cuda")
        elif task == "segmentation":
            targets = targets.to(device, dtype=torch.long, non_blocking=device.type == "cuda")
        elif task == "localization":
            targets = targets.to(device, dtype=torch.float32, non_blocking=device.type == "cuda")
        else:
            raise ValueError(f"Unknown task: {task}")

        optimizer.zero_grad(set_to_none=True)

        if task == "classification" and (mixup_alpha > 0.0 or cutmix_alpha > 0.0):
            use_mix = np.random.rand() < mix_prob
        else:
            use_mix = False

        if task == "classification" and use_mix:
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
                labels_a, labels_b = targets, targets[rand_index]
            else:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                rand_index = torch.randperm(images.size(0), device=images.device)
                images = lam * images + (1 - lam) * images[rand_index]
                labels_a, labels_b = targets, targets[rand_index]

            logits = model(images)
            loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
        else:
            logits = model(images)
            loss = criterion(logits, targets)


        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

        if task == "classification":
            running_correct += _accuracy(logits, targets)
        elif task == "segmentation":
            preds = torch.argmax(logits, dim=1)
            running_pixel_correct += (preds == targets).sum().item()
            total_pixels += targets.numel()
            _update_segmentation_confusion(confusion, preds, targets, num_classes)

        if log_interval > 0 and batch_idx % log_interval == 0:
            elapsed = time.perf_counter() - start_time
            samples_per_sec = total / max(1e-6, elapsed)
            avg_loss = running_loss / max(1, total)
            if task == "classification":
                avg_metric = running_correct / max(1, total)
                metric_name = "acc"
            elif task == "segmentation":
                avg_metric = running_pixel_correct / max(1, total_pixels)
                metric_name = "pix_acc"
            else:
                avg_metric = None
                metric_name = ""

            message = f"  batch {batch_idx}/{len(loader)} - loss: {avg_loss:.4f}"
            if avg_metric is not None:
                message += f" {metric_name}: {avg_metric:.4f}"
            message += f" ({samples_per_sec:.1f} samples/s)"
            print(message)

    avg_loss = running_loss / max(1, total)

    if task == "classification":
        avg_acc = running_correct / max(1, total)
        return avg_loss, avg_acc
    elif task == "segmentation":
        pixel_acc = running_pixel_correct / max(1, total_pixels)
        mean_iou = _mean_iou_from_confusion(confusion)
        return avg_loss, pixel_acc, mean_iou
    elif task == "localization":
        return avg_loss


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


# ============================================================================
# LOCALIZATION LOSS
# ============================================================================

class LocalizationRegressionLoss(nn.Module):
    """Combined localization loss: MSE + IoU weighted sum."""

    def __init__(self, mse_weight: float = 1.0, iou_weight: float = 1.0):
        """
        Initialize the LocalizationRegressionLoss.

        Args:
            mse_weight: Weight for MSE loss component.
            iou_weight: Weight for IoU loss component.
        """
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.iou_weight = float(iou_weight)
        self.mse = nn.MSELoss()
        self.iou = IoULoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining MSE and IoU losses.

        Args:
            preds: Predicted bounding boxes [B, 4] in pixel coordinates (xywh).
            targets: Target bounding boxes [B, 4] in pixel coordinates (xywh).

        Returns:
            Combined loss value.
        """
        mse_loss = self.mse(preds, targets)
        iou_loss = self.iou(preds, targets)
        return self.mse_weight * mse_loss + self.iou_weight * iou_loss


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


def compute_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute IoU between predicted and target bounding boxes.
    
    Args:
        pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
        target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format.
        eps: Small value to avoid division by zero.
    
    Returns:
        IoU values [B] for each sample in the batch.
    """
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
    iou = inter_area / (union + eps)
    iou = torch.clamp(iou, min=0.0, max=1.0)
    return iou


def evaluate_localization(
    model,
    loader,
    criterion,
    device: torch.device,
):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    iou_values = []

    with torch.no_grad():
        for images, bboxes in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            bboxes = bboxes.to(device, dtype=torch.float32, non_blocking=device.type == "cuda")

            preds = model(images)
            loss = criterion(preds, bboxes)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Compute IoU for each sample in batch
            batch_iou = compute_iou(preds, bboxes)
            iou_values.extend(batch_iou.cpu().numpy())

    avg_loss = running_loss / max(1, total_samples)
    avg_iou = float(np.mean(iou_values)) if iou_values else 0.0
    return avg_loss, avg_iou


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

    import os
    import torch

    if not os.path.isfile(classifier_path):
        raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_path}")

    # ---- Load checkpoint ----
    checkpoint = torch.load(classifier_path, map_location="cpu")
    state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))

    # ---- Extract encoder weights ----
    encoder_state = {
        key.replace("encoder.", ""): value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }

    if not encoder_state:
        raise ValueError(
            "No encoder.* keys found in classifier checkpoint; cannot initialize UNet encoder."
        )

    # ---- CORRECT: load into inner Sequential ----
    incompatible = model.encoder.encoder.load_state_dict(encoder_state, strict=False)

    # ---- Freeze encoder ----
    for param in model.encoder.parameters():
        param.requires_grad = False

    # ---- Keep BN stable ----
    model.encoder.eval()

    # ---- Logging ----
    print(
        f"Loaded {len(encoder_state)} encoder tensors from {classifier_path} "
        f"and froze encoder parameters."
    )

    if incompatible.missing_keys:
        print(f"Missing keys while loading encoder: {incompatible.missing_keys}")

    if incompatible.unexpected_keys:
        print(f"Unexpected keys while loading encoder: {incompatible.unexpected_keys}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",type=str,choices=("classification", "segmentation", "localization"),default="classification",help="Task to train.",)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer",type=str,choices=("adam", "adamw"),default="adamw",help="Optimizer type.",)
    parser.add_argument("--lr_scheduler",type=str,choices=("none", "cosine", "plateau"),default="cosine",help="Learning rate scheduler.",)
    parser.add_argument("--min_lr",type=float,default=1e-6,help="Minimum LR for cosine schedule.",)
    parser.add_argument("--lr_patience",type=int,default=3,help="Plateau scheduler patience (epochs).",)
    parser.add_argument("--lr_factor",type=float,default=0.5,help="Plateau scheduler decay factor.",)
    parser.add_argument("--mixup_alpha",type=float,default=0.0,help="MixUp alpha (0 disables).",)
    parser.add_argument("--cutmix_alpha",type=float,default=0.0,help="CutMix alpha (0 disables).",)
    parser.add_argument("--mix_prob",type=float,default=1.0,help="Probability to apply MixUp/CutMix per batch.",)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_path", type=str, default="checkpoints/classifier.pth")
    parser.add_argument("--classifier_path",type=str, default="checkpoints/classifier.pth", help="Classifier checkpoint for encoder initialization in segmentation mode.",)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--disable_aug",action="store_true",help="Disable training data augmentation.",)
    parser.add_argument("--iou_w", type=float, default=0.5, help="Weight for IoU loss vs MSE loss in localization training.",)
    return parser.parse_args()


def run_classification_training(args, device: torch.device):
    train_loader, val_loader = build_dataloaders(
        task="classification",
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_augmentation=not args.disable_aug,
    )
    print(
        f"Train size: {len(train_loader.dataset)} | "
        f"Val size: {len(val_loader.dataset)} | "
        f"Batches per epoch: {len(train_loader)}"
    )

    model = VGG11Classifier(dropout_p=0.5)
    model = model.to(device)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,)

    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr,)
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience,)

    criterion = nn.CrossEntropyLoss()
    best_acc = -1.0
    print(f"Using device: {device}")
    print("Starting classification training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            task="classification",
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            log_interval=args.log_interval,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            mix_prob=args.mix_prob,
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
    if args.save_path == "checkpoints/classifier.pth":
        args.save_path = "checkpoints/unet.pth"

    train_loader, val_loader = build_dataloaders(
        task="segmentation",
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(
        f"Train size: {len(train_loader.dataset)} | "
        f"Val size: {len(val_loader.dataset)} | "
        f"Batches per epoch: {len(train_loader)}"
    )

    model = VGG11UNet(num_classes=3)
    load_and_freeze_encoder_from_classifier(model, args.classifier_path)
    model = model.to(device)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found for segmentation training.")

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay,)
    else:
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay,)

    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr,)
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience,)

    criterion = nn.CrossEntropyLoss()
    best_miou = -1.0
    print(f"Using device: {device}")
    print("Starting segmentation training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_pix_acc, train_miou = train_one_epoch(
            task="segmentation",
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            log_interval=args.log_interval,
            num_classes=3,
        )
        val_loss, val_pix_acc, val_miou = evaluate_segmentation(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=3,
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
    if args.save_path == "checkpoints/classifier.pth":
        args.save_path = "checkpoints/localizer.pth"

    train_loader, val_loader = build_dataloaders(
        task="localization",
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(
        f"Train size: {len(train_loader.dataset)} | "
        f"Val size: {len(val_loader.dataset)} | "
        f"Batches per epoch: {len(train_loader)}"
    )

    model = VGG11Localizer()
    load_and_freeze_encoder_from_classifier(model, args.classifier_path)
    model = model.to(device)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,)

    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr,)
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=args.lr_factor,patience=args.lr_patience,)

    criterion = LocalizationRegressionLoss(
        mse_weight=1.0 - args.iou_w,
        iou_weight=args.iou_w,
    )
    best_iou = -1.0
    print(f"Using device: {device}")
    print("Starting localization training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            task="localization",
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            log_interval=args.log_interval,
        )
        val_loss, val_iou = evaluate_localization(
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
            f"- val: {val_loss:.4f} iou: {val_iou:.4f} "
            f"- lr: {current_lr:.2e}"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            if args.save_path:
                save_dir = os.path.dirname(args.save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), args.save_path)

    if args.save_path:
        print(f"Best checkpoint saved to {args.save_path} (val iou: {best_iou:.4f})")


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
