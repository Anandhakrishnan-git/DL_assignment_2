"""2.3 Transfer Learning Showdown (Segmentation).

Strategies:
  - strict  : freeze all VGG11 encoder blocks (feature extractor)
  - partial : freeze early blocks, fine-tune last N blocks + decoder
  - full    : fine-tune entire network end-to-end

Logs:
  - train/val loss, pixel accuracy, Dice (foreground=class 1), mIoU
  - time per epoch
  - optional sample visualizations (Task 2.6 compatible)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wandb_experiments.shared import (
    add_repo_root_to_path,
    build_dataloaders,
    load_encoder_from_classifier_checkpoint,
    mask_to_color,
    seed_everything,
    to_numpy_image,
    freeze_vgg11_encoder_blocks,
)
from wandb_experiments.train_utils import eval_segmentation, train_segmentation_one_epoch


def _concat_triplet(img: np.ndarray, gt_rgb: np.ndarray, pred_rgb: np.ndarray) -> Image.Image:
    a = Image.fromarray(img)
    b = Image.fromarray(gt_rgb)
    c = Image.fromarray(pred_rgb)
    w, h = a.size
    out = Image.new("RGB", (w * 3, h))
    out.paste(a, (0, 0))
    out.paste(b, (w, 0))
    out.paste(c, (w * 2, 0))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, choices=("adam", "adamw"), default="adamw")
    p.add_argument("--lr_scheduler", type=str, choices=("none", "cosine", "plateau"), default="cosine")
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--lr_patience", type=int, default=3)
    p.add_argument("--lr_factor", type=float, default=0.5)

    p.add_argument("--strategy", type=str, choices=("strict", "partial", "full"), default="strict")
    p.add_argument("--unfreeze_last_blocks", type=int, default=2, help="For partial: unfreeze last N blocks (1..5).")
    p.add_argument("--classifier_ckpt", type=str, default="checkpoints/classifier.pth")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_subset", type=int, default=0)
    p.add_argument("--val_subset", type=int, default=0)

    p.add_argument("--log_samples", action="store_true")
    p.add_argument("--num_samples", type=int, default=5)

    # W&B
    p.add_argument("--project", type=str, default="dl-assignment-2")
    p.add_argument("--entity", type=str, default=None)
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--tags", type=str, default="")
    p.add_argument("--mode", type=str, choices=("online", "offline", "disabled"), default="online")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    add_repo_root_to_path()
    seed_everything(args.seed)

    from models import VGG11UNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(
        task="segmentation",
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_augmentation=False,
        train_subset=args.train_subset or None,
        val_subset=args.val_subset or None,
    )

    model = VGG11UNet(num_classes=3, use_batchnorm=True).to(device)
    load_encoder_from_classifier_checkpoint(model.encoder.encoder, args.classifier_ckpt)

    # Freeze policy
    post_train_prepare = None
    if args.strategy == "strict":
        for p in model.encoder.parameters():
            p.requires_grad = False

        def post_train_prepare(_model):
            _model.encoder.eval()

    elif args.strategy == "partial":
        n = int(max(1, min(5, args.unfreeze_last_blocks)))
        trainable = list(range(6 - n, 6))  # e.g. n=2 -> [4, 5]
        freeze_vgg11_encoder_blocks(model.encoder, trainable_blocks=trainable)

        def post_train_prepare(_model):
            # model.train() flips all BN to train; force frozen BN back to eval.
            for m in _model.encoder.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) and not any(
                    p.requires_grad for p in m.parameters()
                ):
                    m.eval()

    else:  # full
        for p in model.encoder.parameters():
            p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found (check freeze strategy).")

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience
        )

    criterion = nn.CrossEntropyLoss()

    run_name = args.name
    if not run_name:
        run_name = f"2.3-seg-{args.strategy}-lr{args.lr:g}-unf{args.unfreeze_last_blocks}"

    wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        tags=[t for t in args.tags.split(",") if t],
        mode=args.mode,
        config={
            "task": "segmentation",
            "strategy": args.strategy,
            "unfreeze_last_blocks": args.unfreeze_last_blocks,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": args.optimizer,
            "lr_scheduler": args.lr_scheduler,
            "seed": args.seed,
            "train_subset": args.train_subset,
            "val_subset": args.val_subset,
            "classifier_ckpt": args.classifier_ckpt,
            "trainable_params": int(sum(p.numel() for p in trainable_params)),
        },
    )

    # Fixed sample batch for visualization
    fixed_samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
    if args.log_samples:
        for images, masks in val_loader:
            for i in range(int(images.size(0))):
                fixed_samples.append((images[i : i + 1], masks[i : i + 1]))
                if len(fixed_samples) >= args.num_samples:
                    break
            if len(fixed_samples) >= args.num_samples:
                break

    best_val_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.perf_counter()

        train_stats = train_segmentation_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_classes=3,
            dice_class_index=1,
            post_train_prepare=post_train_prepare,
        )
        val_stats = eval_segmentation(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=3,
            dice_class_index=1,
        )

        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(val_stats["loss"])
            else:
                scheduler.step()

        lr_now = float(optimizer.param_groups[0]["lr"])
        epoch_sec = time.perf_counter() - epoch_t0

        payload = {
            "train/loss": train_stats["loss"],
            "train/pix_acc": train_stats["pix_acc"],
            "train/dice": train_stats["dice"],
            "train/miou": train_stats["miou"],
            "val/loss": val_stats["loss"],
            "val/pix_acc": val_stats["pix_acc"],
            "val/dice": val_stats["dice"],
            "val/miou": val_stats["miou"],
            "lr": lr_now,
            "time/epoch_sec": epoch_sec,
        }

        if args.log_samples and fixed_samples:
            model.eval()
            vis_images = []
            with torch.no_grad():
                for idx, (img_t, gt_t) in enumerate(fixed_samples):
                    img = img_t.to(device)
                    gt = gt_t.to(device)
                    logits = model(img)
                    pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.int64)
                    gt_np = gt[0].detach().cpu().numpy().astype(np.int64)
                    img_np = to_numpy_image(img_t[0])
                    triplet = _concat_triplet(img_np, mask_to_color(gt_np), mask_to_color(pred))
                    vis_images.append(wandb.Image(triplet, caption=f"sample_{idx} (orig | gt | pred)"))
            payload["val/samples"] = vis_images

        wandb.log(payload, step=epoch)

        if val_stats["dice"] > best_val_dice:
            best_val_dice = val_stats["dice"]
            wandb.summary["best_val_dice"] = best_val_dice

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train loss {train_stats['loss']:.4f} dice {train_stats['dice']:.4f} pix {train_stats['pix_acc']:.4f} | "
            f"val loss {val_stats['loss']:.4f} dice {val_stats['dice']:.4f} pix {val_stats['pix_acc']:.4f} | "
            f"lr {lr_now:.2e}"
        )

    wandb.finish()


if __name__ == "__main__":
    main()
