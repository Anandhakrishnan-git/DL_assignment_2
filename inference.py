"""Inference and evaluation for Oxford-IIIT Pet classification/segmentation."""

import argparse
import os
from typing import Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data.pets_dataset import OxfordIIITPetDataset
from models import VGG11Classifier, VGG11UNet, VGG11Localizer
from losses.iou_loss import IoULoss


# ============================================================================
# CONFIGURATION
# ============================================================================

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================================
# AUGMENTATION: Image transformation for inference
# ============================================================================

class AlbumentationsTransform:
    def __init__(
        self,
        size: int = IMAGE_SIZE,
        mean: Tuple[float, float, float] = IMAGENET_MEAN,
        std: Tuple[float, float, float] = IMAGENET_STD,
    ):
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
            mask_img = Image.fromarray(mask.astype(np.uint8), mode="L")
            mask_img = mask_img.resize((self.size, self.size), resample=Image.NEAREST)
            mask_arr = np.array(mask_img, dtype=np.int64, copy=True)
            out["segmentation"] = torch.tensor(mask_arr, dtype=torch.long)
        return out


class LocalizationTargetTransform:
    """Convert localization targets to pixel-space float tensors."""

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


# ============================================================================
# DATA LOADING
# ============================================================================

def build_test_loader(task: str, root: str, batch_size: int, num_workers: int):
    test_transform = AlbumentationsTransform()
    if task == "classification":
        tasks = ("category",)
        target_transform = None
        pin_memory = torch.cuda.is_available()
    elif task == "segmentation":
        tasks = ("segmentation",)
        target_transform = SegmentationTargetTransform(size=IMAGE_SIZE)
        pin_memory = False
    elif task == "localization":
        tasks = ("localization",)
        target_transform = LocalizationTargetTransform(size=IMAGE_SIZE)
        pin_memory = torch.cuda.is_available()
    else:
        raise ValueError(f"Unknown task: {task}")

    test_set = OxfordIIITPetDataset(
        root=root,
        split="test",
        tasks=tasks,
        transform=test_transform,
        target_transform=target_transform,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return test_loader


# ============================================================================
# METRICS & UTILITIES
# ============================================================================

def _update_confusion_matrix(
    conf: np.ndarray, targets: np.ndarray, preds: np.ndarray, num_classes: int
):
    indices = num_classes * targets + preds
    conf += np.bincount(indices, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def _compute_classification_metrics(conf: np.ndarray):
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(axis=0).astype(np.float64) - tp
    fn = conf.sum(axis=1).astype(np.float64) - tp
    support = conf.sum(axis=1).astype(np.float64)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    class_acc = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "class_acc": class_acc,
        "support": support,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }
    return metrics


def evaluate_classification(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    num_classes: int,
    topk: int = 5,
):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_topk = 0
    total = 0
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, dtype=torch.long, non_blocking=device.type == "cuda")

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == labels).sum().item()

            if topk > 1:
                k = min(topk, logits.size(1))
                topk_preds = torch.topk(logits, k, dim=1).indices
                running_topk += (
                    topk_preds == labels.view(-1, 1)
                ).any(dim=1).sum().item()

            _update_confusion_matrix(
                conf,
                labels.detach().cpu().numpy(),
                preds.detach().cpu().numpy(),
                num_classes,
            )

    avg_loss = running_loss / max(1, total)
    acc = running_correct / max(1, total)
    topk_acc = None
    if topk > 1:
        topk_acc = running_topk / max(1, total)

    metrics = _compute_classification_metrics(conf)
    return avg_loss, acc, topk_acc, metrics


def _update_segmentation_confusion(
    conf: np.ndarray,
    targets: np.ndarray,
    preds: np.ndarray,
    num_classes: int,
):
    targets_flat = targets.reshape(-1)
    preds_flat = preds.reshape(-1)
    valid = (targets_flat >= 0) & (targets_flat < num_classes)
    if not np.any(valid):
        return
    indices = num_classes * targets_flat[valid] + preds_flat[valid]
    conf += np.bincount(indices, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def _compute_segmentation_metrics(conf: np.ndarray):
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(axis=0).astype(np.float64) - tp
    fn = conf.sum(axis=1).astype(np.float64) - tp
    support = conf.sum(axis=1).astype(np.float64)

    denom = tp + fp + fn
    iou = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
    valid = denom > 0
    mean_iou = float(iou[valid].mean()) if np.any(valid) else 0.0
    pixel_acc = float(tp.sum() / max(1.0, conf.sum()))

    metrics = {
        "iou": iou,
        "support": support,
        "pixel_acc": pixel_acc,
        "mean_iou": mean_iou,
    }
    return metrics


def _colorize_segmentation_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    if num_classes == 3:
        return OxfordIIITPetDataset.colorize_segmentation_mask(mask)

    palette = np.stack(
        [
            (37 * np.arange(num_classes)) % 255,
            (67 * np.arange(num_classes)) % 255,
            (97 * np.arange(num_classes)) % 255,
        ],
        axis=1,
    ).astype(np.uint8)
    mask_clipped = np.clip(mask, 0, num_classes - 1).astype(np.int64)
    return palette[mask_clipped]


# ============================================================================
# SEGMENTATION EVALUATION
# ============================================================================

def evaluate_segmentation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    num_classes: int,
):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

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

            _update_segmentation_confusion(
                conf,
                masks.detach().cpu().numpy(),
                preds.detach().cpu().numpy(),
                num_classes,
            )

    avg_loss = running_loss / max(1, total_samples)
    metrics = _compute_segmentation_metrics(conf)
    return avg_loss, metrics["pixel_acc"], metrics


# ============================================================================
# CHECKPOINT UTILITIES
# ============================================================================

def _extract_state_dict(checkpoint):
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


def _strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))
    model.load_state_dict(state_dict)


# ============================================================================
# ARGUMENT PARSING & INFERENCE RUNNERS
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        choices=("classification", "segmentation", "localization"),
        default="classification",
        help="Inference task.",
    )
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--classifier_path", type=str, default="classifier.pth")
    parser.add_argument("--unet_path", type=str, default="unet.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=37)
    parser.add_argument("--seg_num_classes", type=int, default=3)
    parser.add_argument(
        "--dropout_mode",
        type=str,
        choices=("element", "channel", "spatial"),
        default="channel",
        help="Must match the checkpoint's classifier head.",
    )
    parser.add_argument(
        "--dropout_p",
        type=float,
        default=0.6,
        help="Dropout probability used in the classifier head.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Compute top-k accuracy (set 1 to disable).",
    )
    parser.add_argument(
        "--print_per_class",
        action="store_true",
        help="Print per-class precision/recall/F1/accuracy for classification.",
    )
    parser.add_argument(
        "--print_seg_per_class",
        action="store_true",
        help="Print per-class IoU/support for segmentation.",
    )
    parser.add_argument("--localizer_path", type=str, default="localizer.pth")
    return parser.parse_args()


def run_classification_inference(args, device: torch.device):
    test_loader = build_test_loader(
        task="classification",
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(
        f"Test size: {len(test_loader.dataset)} | "
        f"Batches: {len(test_loader)}"
    )

    model = VGG11Classifier(
        num_classes=args.num_classes,
        dropout_p=args.dropout_p,
        dropout_mode=args.dropout_mode,
    ).to(device)
    load_checkpoint(model, args.classifier_path, device)

    criterion = nn.CrossEntropyLoss()

    print(f"Using device: {device}")
    avg_loss, acc, topk_acc, metrics = evaluate_classification(
        model=model,
        loader=test_loader,
        device=device,
        criterion=criterion,
        num_classes=args.num_classes,
        topk=args.topk,
    )

    print(f"Test loss: {avg_loss:.4f}")
    print(f"Test accuracy: {acc:.4f}")
    if topk_acc is not None:
        print(f"Top-{min(args.topk, args.num_classes)} accuracy: {topk_acc:.4f}")
    print(
        "Macro precision/recall/F1: "
        f"{metrics['macro_precision']:.4f} / "
        f"{metrics['macro_recall']:.4f} / "
        f"{metrics['macro_f1']:.4f}"
    )

    if args.print_per_class:
        print("Per-class metrics (index: support, precision, recall, f1, acc):")
        for idx in range(args.num_classes):
            support = int(metrics["support"][idx])
            precision = metrics["precision"][idx]
            recall = metrics["recall"][idx]
            f1 = metrics["f1"][idx]
            class_acc = metrics["class_acc"][idx]
            print(
                f"{idx:02d}: support={support} "
                f"prec={precision:.4f} rec={recall:.4f} "
                f"f1={f1:.4f} acc={class_acc:.4f}"
            )


def run_segmentation_inference(args, device: torch.device):
    model = VGG11UNet(num_classes=args.seg_num_classes).to(device)
    load_checkpoint(model, args.unet_path, device)
    criterion = nn.CrossEntropyLoss()

    test_loader = build_test_loader(
        task="segmentation",
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(
        f"Test size: {len(test_loader.dataset)} | "
        f"Batches: {len(test_loader)}"
    )

    print(f"Using device: {device}")
    avg_loss, pixel_acc, metrics = evaluate_segmentation(
        model=model,
        loader=test_loader,
        device=device,
        criterion=criterion,
        num_classes=args.seg_num_classes,
    )

    print(f"Segmentation test loss: {avg_loss:.4f}")
    print(f"Segmentation pixel accuracy: {pixel_acc:.4f}")
    print(f"Segmentation mean IoU: {metrics['mean_iou']:.4f}")

    if args.print_seg_per_class:
        print("Per-class metrics (index: support, iou):")
        for idx in range(args.seg_num_classes):
            support = int(metrics["support"][idx])
            iou = metrics["iou"][idx]
            print(f"{idx:02d}: support={support} iou={iou:.4f}")



def run_localization_inference(args, device: torch.device):
    model = VGG11Localizer().to(device)
    load_checkpoint(model, args.localizer_path, device)
    criterion = IoULoss()

    test_loader = build_test_loader(
        task="localization",
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(
        f"Test size: {len(test_loader.dataset)} | "
        f"Batches: {len(test_loader)}"
    )

    print(f"Using device: {device}")
    avg_loss = evaluate_localization(
        model=model,
        loader=test_loader,
        device=device,
        criterion=criterion,
    )

    print(f"Localization test loss: {avg_loss:.4f}")


# ============================================================================
# LOCALIZATION EVALUATION
# ============================================================================

def evaluate_localization(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
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


# ============================================================================
# MAIN INFERENCE ENTRY POINT
# ============================================================================

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.task == "segmentation":
        run_segmentation_inference(args, device)
    elif args.task == "classification":
        run_classification_inference(args, device)
    elif args.task == "localization":
        run_localization_inference(args, device)
    else:
        raise ValueError(f"Unsupported task: {args.task}")


if __name__ == "__main__":
    main()
