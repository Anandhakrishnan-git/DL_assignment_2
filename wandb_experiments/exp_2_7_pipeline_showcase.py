"""2.7 Final pipeline showcase on novel images (not from dataset).

For each input image:
  1) Predict bbox (localizer)
  2) Crop using predicted bbox
  3) Classify crop (classifier) + predict trimap (U-Net)
  4) Log a 3-panel visualization to W&B
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.request
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch

import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wandb_experiments.shared import (
    AlbumentationsTransform,
    add_repo_root_to_path,
    clamp_box_xyxy,
    draw_boxes_xywh,
    load_model_weights,
    mask_to_color,
    seed_everything,
    xywh_to_xyxy,
)


def _load_pil(path_or_url: str) -> Image.Image:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        with urllib.request.urlopen(path_or_url) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")


def _overlay_mask(image_rgb: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    img = image_rgb.astype(np.float32)
    m = mask_rgb.astype(np.float32)
    out = img * (1.0 - alpha) + m * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _concat_three(a: Image.Image, b: Image.Image, c: Image.Image) -> Image.Image:
    w, h = a.size
    out = Image.new("RGB", (w * 3, h))
    out.paste(a, (0, 0))
    out.paste(b, (w, 0))
    out.paste(c, (w * 2, 0))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=str, nargs="+", required=True, help="3+ local paths or URLs")
    p.add_argument("--classifier_ckpt", type=str, default="checkpoints/classifier.pth")
    p.add_argument("--localizer_ckpt", type=str, default="checkpoints/localizer.pth")
    p.add_argument("--unet_ckpt", type=str, default="checkpoints/unet.pth")
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

    from models import VGG11Classifier, VGG11Localizer, VGG11UNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = AlbumentationsTransform(train=False)

    classifier = VGG11Classifier(dropout_p=0.0, use_batchnorm=True).to(device)
    localizer = VGG11Localizer(dropout_p=0.0, use_batchnorm=True).to(device)
    unet = VGG11UNet(num_classes=3, use_batchnorm=True).to(device)

    load_model_weights(classifier, args.classifier_ckpt, strict=False)
    load_model_weights(localizer, args.localizer_ckpt, strict=False)
    load_model_weights(unet, args.unet_ckpt, strict=False)

    classifier.eval()
    localizer.eval()
    unet.eval()

    run_name = args.name or "2.7-pipeline-showcase"
    wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        tags=[t for t in args.tags.split(",") if t],
        mode=args.mode,
        config={
            "classifier_ckpt": args.classifier_ckpt,
            "localizer_ckpt": args.localizer_ckpt,
            "unet_ckpt": args.unet_ckpt,
            "images": args.images,
            "seed": args.seed,
        },
    )

    logged = []
    for idx, src in enumerate(args.images):
        pil = _load_pil(src)
        pil_resized = pil.resize((224, 224), resample=Image.BILINEAR)
        x = transform(pil_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            bbox_xywh = localizer(x)[0].detach().cpu()

        # Crop on resized image using predicted bbox (xywh in px).
        x1, y1, x2, y2 = clamp_box_xyxy(xywh_to_xyxy(bbox_xywh.tolist()), w=224, h=224)
        if (x2 - x1) < 2 or (y2 - y1) < 2:
            x1, y1, x2, y2 = 0, 0, 223, 223
        crop = pil_resized.crop((x1, y1, x2, y2))
        crop_224 = crop.resize((224, 224), resample=Image.BILINEAR)
        x_crop = transform(crop_224).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = classifier(x_crop)[0]
            probs = torch.softmax(logits, dim=0)
            top_prob, top_idx = torch.max(probs, dim=0)

            seg_logits = unet(x_crop)
            seg_pred = torch.argmax(seg_logits, dim=1)[0].detach().cpu().numpy().astype(np.int64)

        seg_rgb = mask_to_color(seg_pred)
        crop_rgb = np.asarray(crop_224, dtype=np.uint8)
        seg_overlay = Image.fromarray(_overlay_mask(crop_rgb, seg_rgb, alpha=0.45))

        # Panel: original+bbox | crop | crop+seg
        orig_np = np.asarray(pil_resized, dtype=np.uint8)
        orig_bbox = draw_boxes_xywh(orig_np, gt_xywh=None, pred_xywh=bbox_xywh.tolist())
        panel = _concat_three(orig_bbox, crop_224, seg_overlay)

        caption = f"top_class={int(top_idx)} prob={float(top_prob):.3f} bbox_xywh={[round(float(v), 1) for v in bbox_xywh.tolist()]}"
        logged.append(wandb.Image(panel, caption=caption))

        print(f"[{idx}] {src} -> class {int(top_idx)} prob {float(top_prob):.3f} bbox {bbox_xywh.tolist()}")

    wandb.log({"pipeline_outputs": logged}, step=0)
    wandb.finish()


if __name__ == "__main__":
    main()
