"""2.4 Feature maps from first vs last conv layer (classifier).

Loads a trained classifier checkpoint and logs feature-map grids from:
  - first conv layer (after ReLU)
  - last conv layer before pooling (after ReLU)
"""

from __future__ import annotations

import argparse
import os
import sys

from PIL import Image
import torch

import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wandb_experiments.shared import (
    AlbumentationsTransform,
    add_repo_root_to_path,
    feature_map_grid,
    load_model_weights,
    seed_everything,
    to_numpy_image,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/classifier.pth")
    p.add_argument("--image_path", type=str, required=True)
    p.add_argument("--num_maps", type=int, default=16)
    p.add_argument("--grid_cols", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

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

    model = VGG11Classifier(dropout_p=0.0, use_batchnorm=True).to(device)
    load_model_weights(model, args.ckpt, strict=False)
    model.eval()

    pil = Image.open(args.image_path).convert("RGB")
    transform = AlbumentationsTransform(train=False)
    x = transform(pil).unsqueeze(0).to(device)

    captured = {}

    def hook_first(_m, _i, o):
        captured["first"] = o.detach()

    def hook_last(_m, _i, o):
        captured["last"] = o.detach()

    h1 = model.encoder.encoder[2].register_forward_hook(hook_first)
    h2 = model.encoder.encoder[27].register_forward_hook(hook_last)
    try:
        with torch.no_grad():
            _ = model(x)
    finally:
        h1.remove()
        h2.remove()

    first_grid = feature_map_grid(captured["first"], num_maps=args.num_maps, grid_cols=args.grid_cols)
    last_grid = feature_map_grid(captured["last"], num_maps=args.num_maps, grid_cols=args.grid_cols)

    run_name = args.name or "2.4-feature-maps"
    wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        tags=[t for t in args.tags.split(",") if t],
        mode=args.mode,
        config={
            "ckpt": args.ckpt,
            "image_path": args.image_path,
            "num_maps": args.num_maps,
            "grid_cols": args.grid_cols,
            "seed": args.seed,
        },
    )

    wandb.log(
        {
            "input": wandb.Image(pil, caption="input (original)"),
            "first_conv_maps": wandb.Image(first_grid, caption="first conv (after ReLU)"),
            "last_conv_maps": wandb.Image(last_grid, caption="last conv before pooling (after ReLU)"),
        },
        step=0,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
