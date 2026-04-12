"""2.6 Segmentation evaluation: Dice vs Pixel Accuracy + 5 sample images.
"""

from __future__ import annotations

import argparse
import os
import sys
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
    load_model_weights,
    mask_to_color,
    seed_everything,
    to_numpy_image,
)
from wandb_experiments.train_utils import eval_segmentation


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
    p.add_argument("--ckpt", type=str, default="checkpoints/unet.pth")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
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

    from models import VGG11UNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = build_dataloaders(
        task="segmentation",
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_augmentation=False,
        train_subset=0,
        val_subset=args.val_subset or None,
    )

    model = VGG11UNet(num_classes=3, use_batchnorm=True).to(device)
    load_model_weights(model, args.ckpt, strict=False)
    criterion = nn.CrossEntropyLoss()

    stats = eval_segmentation(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        num_classes=3,
        dice_class_index=1,
    )

    samples: List[wandb.Image] = []
    model.eval()
    with torch.no_grad():
        collected = 0
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            for i in range(int(images.size(0))):
                img_np = to_numpy_image(images[i].cpu())
                gt_np = masks[i].detach().cpu().numpy().astype(np.int64)
                pr_np = preds[i].detach().cpu().numpy().astype(np.int64)
                triplet = _concat_triplet(img_np, mask_to_color(gt_np), mask_to_color(pr_np))
                samples.append(wandb.Image(triplet, caption="orig | gt | pred"))
                collected += 1
                if collected >= args.num_samples:
                    break
            if collected >= args.num_samples:
                break

    run_name = args.name or "2.6-seg-eval"
    wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        tags=[t for t in args.tags.split(",") if t],
        mode=args.mode,
        config={
            "task": "segmentation",
            "ckpt": args.ckpt,
            "batch_size": args.batch_size,
            "num_samples": args.num_samples,
            "seed": args.seed,
            "val_subset": args.val_subset,
        },
    )

    wandb.log(
        {
            "val/loss": stats["loss"],
            "val/pix_acc": stats["pix_acc"],
            "val/dice": stats["dice"],
            "val/miou": stats["miou"],
            "val/samples": samples,
        },
        step=0,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
