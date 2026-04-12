"""2.1 BatchNorm: LR range test (max stable LR proxy).

Runs a short "LR finder" style sweep where LR increases exponentially each step.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn as nn

import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wandb_experiments.shared import add_repo_root_to_path, build_dataloaders, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--dropout_p", type=float, default=0.5)
    p.add_argument("--batchnorm", type=str, choices=("on", "off"), default="on")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_subset", type=int, default=1024)

    p.add_argument("--start_lr", type=float, default=1e-6)
    p.add_argument("--end_lr", type=float, default=1.0)
    p.add_argument("--num_iters", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, choices=("adam", "adamw"), default="adamw")
    p.add_argument("--diverge_factor", type=float, default=4.0)

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

    train_loader, _ = build_dataloaders(
        task="classification",
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_augmentation=True,
        train_subset=args.train_subset or None,
        val_subset=0,
    )

    model = VGG11Classifier(dropout_p=args.dropout_p, use_batchnorm=use_bn).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.start_lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.start_lr,
            weight_decay=args.weight_decay,
        )

    run_name = args.name
    if not run_name:
        run_name = f"2.1-lr-range-bn-{args.batchnorm}-do{args.dropout_p:g}"

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
            "batch_size": args.batch_size,
            "start_lr": args.start_lr,
            "end_lr": args.end_lr,
            "num_iters": args.num_iters,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "train_subset": args.train_subset,
        },
    )

    lr_mult = (args.end_lr / args.start_lr) ** (1.0 / max(1, args.num_iters - 1))
    best_loss = float("inf")
    best_lr = args.start_lr
    last_stable_lr = args.start_lr

    model.train()
    it = 0
    data_iter = iter(train_loader)
    while it < args.num_iters:
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, labels = next(data_iter)

        lr = args.start_lr * (lr_mult**it)
        optimizer.param_groups[0]["lr"] = lr

        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, dtype=torch.long, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            break

        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        if loss_val < best_loss:
            best_loss = loss_val
            best_lr = lr
        if loss_val <= args.diverge_factor * best_loss:
            last_stable_lr = lr

        wandb.log({"lr": lr, "loss": loss_val, "best_loss": best_loss}, step=it)

        if loss_val > args.diverge_factor * best_loss:
            break

        it += 1

    wandb.summary["best_loss"] = best_loss
    wandb.summary["best_lr_at_min_loss"] = best_lr
    wandb.summary["last_stable_lr"] = last_stable_lr
    wandb.finish()


if __name__ == "__main__":
    main()
