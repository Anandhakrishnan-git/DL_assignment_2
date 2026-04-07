"""Inference and evaluation for Oxford-IIIT Pet classification."""

import argparse
from typing import Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data.pets_dataset import OxfordIIITPetDataset
from models import VGG11Classifier

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


def build_test_loader(root: str, batch_size: int, num_workers: int):
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


def evaluate(
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--classifier_path", type=str, default="classifier.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=37)
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
        help="Print per-class precision/recall/F1/accuracy.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = build_test_loader(
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

    state = torch.load(args.classifier_path, map_location=device)
    model.load_state_dict(state)

    criterion = nn.CrossEntropyLoss()

    print(f"Using device: {device}")
    avg_loss, acc, topk_acc, metrics = evaluate(
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


if __name__ == "__main__":
    main()
