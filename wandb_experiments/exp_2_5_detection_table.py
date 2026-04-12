"""2.5 Object detection-style table: confidence & IoU for bbox predictions.

Logs a W&B table with >=10 test images:
  - Green: ground-truth bbox
  - Red  : predicted bbox
  - Columns include IoU and a simple confidence score.

Confidence here is MC-Dropout confidence:
  confidence = 1 / (1 + mean_std_px)
where mean_std_px is the mean std-dev of bbox coords (xywh) across stochastic passes.
This can produce high-confidence but low-IoU failures (confidently wrong).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch

import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wandb_experiments.shared import (
    AlbumentationsTransform,
    LocalizationTargetTransform,
    add_repo_root_to_path,
    compute_iou_xywh,
    draw_boxes_xywh,
    enable_custom_dropout_only,
    load_model_weights,
    seed_everything,
    to_numpy_image,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--ckpt", type=str, default="checkpoints/localizer.pth")
    p.add_argument("--split", type=str, choices=("test", "val", "train"), default="test")
    p.add_argument("--num_images", type=int, default=10)
    p.add_argument("--mc_samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batchnorm", type=str, choices=("on", "off"), default="on")
    p.add_argument("--dropout_p", type=float, default=0.5)

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

    from data.pets_dataset import OxfordIIITPetDataset
    from models import VGG11Localizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bn = args.batchnorm == "on"

    ds = OxfordIIITPetDataset(
        root=args.data_root,
        split=args.split,
        tasks=("localization",),
        transform=AlbumentationsTransform(train=False),
        target_transform=LocalizationTargetTransform(),
    )
    if len(ds) == 0:
        raise RuntimeError(f"No samples found for split={args.split} under {args.data_root}.")

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=min(args.num_images, len(ds)), replace=False).tolist()

    model = VGG11Localizer(dropout_p=args.dropout_p, use_batchnorm=use_bn).to(device)
    load_model_weights(model, args.ckpt, strict=False)

    run_name = args.name or f"2.5-bbox-table-{args.split}-mc{args.mc_samples}"
    wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        tags=[t for t in args.tags.split(",") if t],
        mode=args.mode,
        config={
            "task": "localization",
            "ckpt": args.ckpt,
            "split": args.split,
            "num_images": args.num_images,
            "mc_samples": args.mc_samples,
            "seed": args.seed,
            "batchnorm": args.batchnorm,
            "dropout_p": args.dropout_p,
        },
    )

    table = wandb.Table(
        columns=[
            "idx",
            "overlay",
            "iou",
            "confidence",
            "std_px",
            "gt_xywh",
            "pred_xywh",
        ]
    )

    all_ious: List[float] = []
    all_confs: List[float] = []
    all_overlays: List[Image.Image] = []

    enable_custom_dropout_only(model)
    for idx in indices:
        img_t, gt_xywh = ds[idx]
        x = img_t.unsqueeze(0).to(device)
        gt = gt_xywh.to(device).unsqueeze(0)

        preds = []
        with torch.no_grad():
            for _ in range(int(max(1, args.mc_samples))):
                preds.append(model(x).detach())
        preds_t = torch.stack(preds, dim=0)  # [S, 1, 4]
        pred_mean = preds_t.mean(dim=0)[0]
        pred_std = preds_t.std(dim=0)[0]

        std_px = float(pred_std.mean().item())
        conf = float(1.0 / (1.0 + std_px))
        iou = float(compute_iou_xywh(pred_mean.unsqueeze(0), gt).item())

        img_np = to_numpy_image(img_t)
        overlay = draw_boxes_xywh(img_np, gt_xywh=gt_xywh.tolist(), pred_xywh=pred_mean.cpu().tolist())

        table.add_data(
            int(idx),
            wandb.Image(overlay),
            iou,
            conf,
            std_px,
            [float(v) for v in gt_xywh.tolist()],
            [float(v) for v in pred_mean.cpu().tolist()],
        )

        all_ious.append(iou)
        all_confs.append(conf)
        all_overlays.append(overlay)

    wandb.log({"detections": table}, step=0)

    # Failure case: highest-confidence sample with IoU below threshold.
    order = np.argsort(-np.array(all_confs))
    failure_idx = None
    for k in order:
        if all_ious[int(k)] < 0.3:
            failure_idx = int(k)
            break
    if failure_idx is None:
        failure_idx = int(np.argmin(np.array(all_ious)))

    wandb.summary["failure_iou"] = float(all_ious[failure_idx])
    wandb.summary["failure_confidence"] = float(all_confs[failure_idx])
    wandb.summary["failure_dataset_idx"] = int(indices[failure_idx])
    wandb.log({"failure_case": wandb.Image(all_overlays[failure_idx])}, step=0)

    wandb.finish()


if __name__ == "__main__":
    main()
