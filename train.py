"""Training entrypoint for Oxford-IIIT Pet classification."""

import argparse
import os
import time
from typing import Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

from data.pets_dataset import OxfordIIITPetDataset
from models import VGG11Classifier

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageTransform:
    def __init__(
        self,
        size: int = IMAGE_SIZE,
        mean: Tuple[float, float, float] = IMAGENET_MEAN,
        std: Tuple[float, float, float] = IMAGENET_STD,
    ):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.size, self.size), resample=Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)


def build_dataloaders(
    root: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    overfit_subset: int = 0,
):
    transform = ImageTransform()

    full_set = OxfordIIITPetDataset(
        root=root,
        split="trainval",
        tasks=("category",),
        transform=transform,
    )
    val_size = int(len(full_set) * val_ratio)
    train_size = len(full_set) - val_size
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(full_set, [train_size, val_size], generator=generator)

    if overfit_subset > 0:
        subset_size = min(overfit_subset, len(train_set))
        subset_indices = list(range(subset_size))
        base_train = train_set
        train_set = Subset(base_train, subset_indices)
        val_set = Subset(base_train, subset_indices)

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


def _grad_norm_l2(model: nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        param_norm = param.grad.detach().norm(2).item()
        total += param_norm * param_norm
    return total**0.5


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    log_interval: int,
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
        #print(f"Logits std: {logits.std().item()}")
        #print(f"Grad norm: {torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()}")


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--save_path", type=str, default="classifier.pth")
    parser.add_argument("--log_interval", type=int, default=50)
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
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(
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

    model = VGG11Classifier(num_classes=37, dropout_p= 0.2).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    print(f"Using device: {device}")
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.log_interval,
            debug_stats=args.debug_stats,
            debug_batches=args.debug_batches,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"- train: {train_loss:.4f} acc: {train_acc:.4f} "
            f"- val: {val_loss:.4f} acc: {val_acc:.4f}"
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


if __name__ == "__main__":
    main()
