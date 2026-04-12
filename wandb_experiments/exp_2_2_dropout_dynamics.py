"""2.2 Internal Dynamics: No Dropout vs p=0.2 vs p=0.5.

Run this script three times with:
  --dropout_p 0.0
  --dropout_p 0.2
  --dropout_p 0.5

W&B will overlay the curves across runs in the same project.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wandb_experiments.shared import add_repo_root_to_path, build_dataloaders, seed_everything
from wandb_experiments.train_utils import eval_classification, train_classification_one_epoch


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
    p.add_argument("--dropout_p", type=float, default=0.5)
    p.add_argument("--batchnorm", type=str, choices=("on", "off"), default="on")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_subset", type=int, default=0)
    p.add_argument("--val_subset", type=int, default=0)

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

    from models import VGG11Classifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bn = args.batchnorm == "on"

    train_loader, val_loader = build_dataloaders(
        task="classification",
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_augmentation=True,
        train_subset=args.train_subset or None,
        val_subset=args.val_subset or None,
    )

    model = VGG11Classifier(dropout_p=args.dropout_p, use_batchnorm=use_bn).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience
        )

    run_name = args.name
    if not run_name:
        run_name = f"2.2-dropout-p{args.dropout_p:g}-bn-{args.batchnorm}-lr{args.lr:g}"

    wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        tags=[t for t in args.tags.split(",") if t],
        mode=args.mode,
        config={
            "task": "classification",
            "dropout_p": args.dropout_p,
            "batchnorm": args.batchnorm,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": args.optimizer,
            "lr_scheduler": args.lr_scheduler,
            "seed": args.seed,
            "train_subset": args.train_subset,
            "val_subset": args.val_subset,
        },
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_stats = train_classification_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        val_stats = eval_classification(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(val_stats["loss"])
            else:
                scheduler.step()

        lr_now = float(optimizer.param_groups[0]["lr"])
        epoch_sec = time.perf_counter() - t0

        wandb.log(
            {
                "train/loss": train_stats["loss"],
                "train/acc": train_stats["acc"],
                "val/loss": val_stats["loss"],
                "val/acc": val_stats["acc"],
                "lr": lr_now,
                "time/epoch_sec": epoch_sec,
            },
            step=epoch,
        )

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train loss {train_stats['loss']:.4f} acc {train_stats['acc']:.4f} | "
            f"val loss {val_stats['loss']:.4f} acc {val_stats['acc']:.4f} | "
            f"lr {lr_now:.2e}"
        )

    wandb.finish()


if __name__ == "__main__":
    main()
