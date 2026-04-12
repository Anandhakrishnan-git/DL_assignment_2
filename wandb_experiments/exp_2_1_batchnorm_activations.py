"""2.1 BatchNorm: activation distributions + convergence.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wandb_experiments.shared import (
    add_repo_root_to_path,
    build_dataloaders,
    seed_everything,
    to_numpy_image,
)
from wandb_experiments.train_utils import eval_classification, train_classification_one_epoch


def _capture_activation(model: nn.Module, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    captured = {}

    def hook(_module, _inputs, output):
        captured["act"] = output.detach()

    handle = layer.register_forward_hook(hook)
    try:
        model.eval()
        with torch.no_grad():
            _ = model(x)
    finally:
        handle.remove()
    return captured["act"]


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
    p.add_argument("--train_subset", type=int, default=0, help="Use first N train samples (0=all).")
    p.add_argument("--val_subset", type=int, default=0, help="Use first N val samples (0=all).")

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
        run_name = f"2.1-bn-{args.batchnorm}-lr{args.lr:g}-do{args.dropout_p:g}"

    wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        tags=[t for t in args.tags.split(",") if t],
        mode=args.mode,
        config={
            "task": "classification",
            "batchnorm": args.batchnorm,
            "dropout_p": args.dropout_p,
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

    # Fixed input for activation histograms: first validation image
    fixed_images, _ = next(iter(val_loader))
    fixed_x = fixed_images[:1].to(device)
    wandb.log({"fixed_input": wandb.Image(to_numpy_image(fixed_images[0]))}, step=0)

    # 3rd conv layer activation (conv3a -> bn -> relu): ReLU index 10 in models/vgg11.py
    act_layer = model.encoder.encoder[10]

    best_val_acc = -1.0
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

        act = _capture_activation(model, act_layer, fixed_x)
        act_np = act.detach().cpu().float().flatten().numpy()

        wandb.log(
            {
                "train/loss": train_stats["loss"],
                "train/acc": train_stats["acc"],
                "val/loss": val_stats["loss"],
                "val/acc": val_stats["acc"],
                "lr": lr_now,
                "time/epoch_sec": epoch_sec,
                "activations/conv3_relu": wandb.Histogram(act_np),
                "activations/conv3_relu_mean": float(np.mean(act_np)),
                "activations/conv3_relu_std": float(np.std(act_np)),
            },
            step=epoch,
        )

        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            wandb.summary["best_val_acc"] = best_val_acc

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train loss {train_stats['loss']:.4f} acc {train_stats['acc']:.4f} | "
            f"val loss {val_stats['loss']:.4f} acc {val_stats['acc']:.4f} | "
            f"lr {lr_now:.2e}"
        )

    wandb.finish()


if __name__ == "__main__":
    main()
