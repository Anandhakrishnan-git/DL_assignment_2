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
from models import VGG11Classifier, VGG11UNet

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def build_classification_test_loader(root: str, batch_size: int, num_workers: int):
    test_transform = AlbumentationsTransform()
    test_set = OxfordIIITPetDataset(
        root=root,
        split="test",
        tasks=("category",),
        transform=test_transform,
    )
    pin_memory = torch.cuda.is_available()
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return test_loader


def build_segmentation_test_loader(root: str, batch_size: int, num_workers: int):
    image_transform = AlbumentationsTransform()
    target_transform = SegmentationTargetTransform(size=IMAGE_SIZE)
    test_set = OxfordIIITPetDataset(
        root=root,
        split="test",
        tasks=("segmentation",),
        transform=image_transform,
        target_transform=target_transform,
    )
    # Keep pin memory off to avoid rare Windows/CUDA pin-thread crashes.
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return test_loader


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


def _unnormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().float().permute(1, 2, 0).numpy()
    image = image * np.array(IMAGENET_STD, dtype=np.float32) + np.array(
        IMAGENET_MEAN, dtype=np.float32
    )
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


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


def visualize_segmentation_prediction(
    model: nn.Module,
    root: str,
    device: torch.device,
    num_classes: int,
    sample_index: int,
    out_dir: str,
    prefix: str,
    alpha: float,
):
    image_transform = AlbumentationsTransform()
    target_transform = SegmentationTargetTransform(size=IMAGE_SIZE)
    dataset = OxfordIIITPetDataset(
        root=root,
        split="test",
        tasks=("segmentation",),
        transform=image_transform,
        target_transform=target_transform,
    )

    if len(dataset) == 0:
        raise RuntimeError("Empty test dataset; cannot visualize segmentation output.")

    if sample_index < 0:
        sample_index = int(np.random.randint(0, len(dataset)))
    sample_index = int(sample_index % len(dataset))

    image_tensor, gt_mask = dataset[sample_index]
    image_batch = image_tensor.unsqueeze(0).to(device, non_blocking=device.type == "cuda")

    model.eval()
    with torch.no_grad():
        logits = model(image_batch)
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.int64)

    gt_mask_np = gt_mask.detach().cpu().numpy().astype(np.int64)
    image_np = _unnormalize_image(image_tensor)
    pred_color = _colorize_segmentation_mask(pred_mask, num_classes)
    gt_color = _colorize_segmentation_mask(gt_mask_np, num_classes)

    alpha = float(np.clip(alpha, 0.0, 1.0))
    pred_overlay = ((1.0 - alpha) * image_np + alpha * pred_color).astype(np.uint8)
    gt_overlay = ((1.0 - alpha) * image_np + alpha * gt_color).astype(np.uint8)

    panel = np.concatenate(
        [image_np, gt_color, pred_color, gt_overlay, pred_overlay],
        axis=1,
    )

    os.makedirs(out_dir, exist_ok=True)
    base = f"{prefix}_idx_{sample_index:04d}"
    image_path = os.path.join(out_dir, f"{base}_image.png")
    gt_path = os.path.join(out_dir, f"{base}_gt_mask.png")
    pred_path = os.path.join(out_dir, f"{base}_pred_mask.png")
    gt_overlay_path = os.path.join(out_dir, f"{base}_gt_overlay.png")
    pred_overlay_path = os.path.join(out_dir, f"{base}_pred_overlay.png")
    panel_path = os.path.join(out_dir, f"{base}_panel.png")

    Image.fromarray(image_np).save(image_path)
    Image.fromarray(gt_color).save(gt_path)
    Image.fromarray(pred_color).save(pred_path)
    Image.fromarray(gt_overlay).save(gt_overlay_path)
    Image.fromarray(pred_overlay).save(pred_overlay_path)
    Image.fromarray(panel).save(panel_path)

    print("Saved segmentation visualization:")
    print(f"  sample index: {sample_index}")
    print(f"  image: {image_path}")
    print(f"  gt mask: {gt_path}")
    print(f"  pred mask: {pred_path}")
    print(f"  gt overlay: {gt_overlay_path}")
    print(f"  pred overlay: {pred_overlay_path}")
    print(f"  panel: {panel_path}")


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        choices=("classification", "segmentation"),
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
    parser.add_argument(
        "--save_seg_viz",
        action="store_true",
        help="Save segmentation prediction visualization for one test sample.",
    )
    parser.add_argument(
        "--seg_viz_index",
        type=int,
        default=0,
        help="Test sample index for segmentation visualization (-1 selects random).",
    )
    parser.add_argument(
        "--seg_viz_dir",
        type=str,
        default="segmentation_viz",
        help="Directory where segmentation visualization files are saved.",
    )
    parser.add_argument(
        "--seg_viz_prefix",
        type=str,
        default="segmentation_test",
        help="Filename prefix for saved segmentation visualization files.",
    )
    parser.add_argument(
        "--seg_viz_alpha",
        type=float,
        default=0.35,
        help="Overlay alpha for segmentation visualization in [0, 1].",
    )
    parser.add_argument(
        "--seg_viz_only",
        action="store_true",
        help="Only save segmentation visualization and skip full test evaluation.",
    )
    return parser.parse_args()


def run_classification_inference(args, device: torch.device):
    test_loader = build_classification_test_loader(
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

    if args.seg_viz_only:
        visualize_segmentation_prediction(
            model=model,
            root=args.data_root,
            device=device,
            num_classes=args.seg_num_classes,
            sample_index=args.seg_viz_index,
            out_dir=args.seg_viz_dir,
            prefix=args.seg_viz_prefix,
            alpha=args.seg_viz_alpha,
        )
        return

    test_loader = build_segmentation_test_loader(
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

    if args.save_seg_viz:
        visualize_segmentation_prediction(
            model=model,
            root=args.data_root,
            device=device,
            num_classes=args.seg_num_classes,
            sample_index=args.seg_viz_index,
            out_dir=args.seg_viz_dir,
            prefix=args.seg_viz_prefix,
            alpha=args.seg_viz_alpha,
        )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.task == "segmentation":
        run_segmentation_inference(args, device)
        return

    run_classification_inference(args, device)


if __name__ == "__main__":
    main()
