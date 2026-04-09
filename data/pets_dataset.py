"""Dataset loader for Oxford-IIIT Pet."""

import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


class OxfordIIITPetDataset(Dataset):
    """
    Oxford-IIIT Pet multi-task dataset loader.

    Supports:
    - classification (breed label)
    - segmentation (trimap mask)
    - localization (bounding box)
    """

    def __init__(
        self,
        root,
        split="train",              # "train", "val", or "test"
        tasks=("category",),          # ("category", "segmentation", "localization")
        transform=None,
        target_transform=None
    ):
        self.root = root
        self.split = split
        self.tasks = tuple(tasks) if isinstance(tasks, (list, tuple)) else (tasks,)
        self.transform = transform
        self.target_transform = target_transform

        if self.split not in ("train", "val", "test"):
            raise ValueError("split must be 'train', 'val', or 'test'")

        # Allow "classification" as an alias for "category"
        self._class_key = "classification" if "classification" in self.tasks and "category" not in self.tasks else "category"

        # Allow passing a parent folder that contains "oxford-iiit-pet"
        candidate = os.path.join(self.root, "oxford-iiit-pet")
        if not os.path.isdir(os.path.join(self.root, "images")) and os.path.isdir(candidate):
            self.root = candidate

        # Paths
        self.images_dir = os.path.join(self.root, "images")
        self.masks_dir = os.path.join(self.root, "annotations", "trimaps")
        self.xml_dir = os.path.join(self.root, "annotations", "xmls")

        # Load all samples
        self.all_samples = self._load_all_annotations()

        # Filter by split
        self.samples = [s for s in self.all_samples if s["split"] == self.split]

        # Filter samples based on required tasks
        if "segmentation" in self.tasks:
            self.samples = [s for s in self.samples if os.path.isfile(s["mask_path"])]
        if "localization" in self.tasks:
            self.samples = [s for s in self.samples if os.path.isfile(s["xml_path"])]

    @staticmethod
    def colorize_segmentation_mask(mask: np.ndarray) -> np.ndarray:
        """Map class ids (0,1,2) to RGB colors for quick visualization."""
        palette = np.array(
            [
                [0, 0, 0],        # class 0: background
                [0, 255, 0],      # class 1: pet
                [255, 0, 0],      # class 2: boundary
            ],
            dtype=np.uint8,
        )
        mask_clipped = np.clip(mask, 0, len(palette) - 1).astype(np.int64)
        return palette[mask_clipped]

    def get_segmentation_visualization(self, idx: int):
        """Return raw image, raw mask, colorized mask, and overlay for sample `idx`."""
        sample = self.samples[idx]
        image = np.array(Image.open(sample["img_path"]).convert("RGB"), dtype=np.uint8)
        mask = np.array(Image.open(sample["mask_path"]), dtype=np.int64) - 1
        color_mask = self.colorize_segmentation_mask(mask)
        overlay = (0.65 * image + 0.35 * color_mask).astype(np.uint8)
        return image, mask, color_mask, overlay

    def get_localization_visualization(self, idx: int, box_width: int = 3):
        """Return raw image, bbox (xyxy), normalized bbox (xywh), and bbox overlay."""
        sample = self.samples[idx]
        image_pil = Image.open(sample["img_path"]).convert("RGB")

        xmin, ymin, xmax, ymax, width, height = self._parse_bbox_xyxy_from_xml(
            sample["xml_path"]
        )
        overlay_pil = image_pil.copy()
        drawer = ImageDraw.Draw(overlay_pil)

        # Draw bbox rectangle + center point for easier inspection.
        drawer.rectangle(
            [(xmin, ymin), (xmax, ymax)],
            outline=(255, 0, 0),
            width=max(1, int(box_width)),
        )
        cx = int(round((xmin + xmax) * 0.5))
        cy = int(round((ymin + ymax) * 0.5))
        r = max(2, int(box_width))
        drawer.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=(255, 0, 0))

        image = np.array(image_pil, dtype=np.uint8)
        overlay = np.array(overlay_pil, dtype=np.uint8)

        bbox_xyxy = (xmin, ymin, xmax, ymax)
        bbox_xywh_norm = (
            ((xmin + xmax) * 0.5) / width,
            ((ymin + ymax) * 0.5) / height,
            (xmax - xmin) / width,
            (ymax - ymin) / height,
        )
        return image, bbox_xyxy, bbox_xywh_norm, overlay

    def save_localization_visualizations(
        self,
        out_dir: str = "bbox_samples",
        num_samples: int = 8,
        indices=None,
        box_width: int = 3,
    ):
        """Save bbox overlays for selected dataset samples."""
        os.makedirs(out_dir, exist_ok=True)

        if indices is None:
            total = min(max(1, num_samples), len(self))
            step = max(1, len(self) // total)
            indices = [i * step for i in range(total)]

        saved = []
        for idx in indices:
            _, bbox_xyxy, bbox_xywh_norm, overlay = self.get_localization_visualization(
                int(idx),
                box_width=box_width,
            )
            out_path = os.path.join(out_dir, f"{self.split}_bbox_{int(idx):04d}.png")
            Image.fromarray(overlay).save(out_path)
            saved.append((int(idx), out_path, bbox_xyxy, bbox_xywh_norm))
        return saved

    def _load_all_annotations(self):
        samples = []

        # Load test samples
        test_ann_path = os.path.join(self.root, "annotations", "test.txt")
        with open(test_ann_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            img_id = parts[0]
            label = int(parts[1]) - 1  # convert to 0-based
            img_path = os.path.join(self.images_dir, img_id + ".jpg")
            mask_path = os.path.join(self.masks_dir, img_id + ".png")
            xml_path = os.path.join(self.xml_dir, img_id + ".xml")
            samples.append({
                "img_path": img_path,
                "mask_path": mask_path,
                "xml_path": xml_path,
                "label": label,
                "split": "test"
            })

        # Load trainval samples and split into train/val
        trainval_ann_path = os.path.join(self.root, "annotations", "trainval.txt")
        trainval_samples = []
        with open(trainval_ann_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            img_id = parts[0]
            label = int(parts[1]) - 1  # convert to 0-based
            img_path = os.path.join(self.images_dir, img_id + ".jpg")
            mask_path = os.path.join(self.masks_dir, img_id + ".png")
            xml_path = os.path.join(self.xml_dir, img_id + ".xml")
            trainval_samples.append({
                "img_path": img_path,
                "mask_path": mask_path,
                "xml_path": xml_path,
                "label": label,
                "split": "trainval"  # temporary
            })

        # Split trainval into train and val (80-20)
        import random
        random.seed(42)  # for reproducibility
        random.shuffle(trainval_samples)
        split_idx = int(0.8 * len(trainval_samples))
        for s in trainval_samples[:split_idx]:
            s["split"] = "train"
        for s in trainval_samples[split_idx:]:
            s["split"] = "val"

        samples.extend(trainval_samples)
        return samples

    def _parse_bbox_xyxy_from_xml(self, xml_path: str) -> tuple:
        """Parse bounding box from XML as (xmin, ymin, xmax, ymax, width, height)."""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get image size
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        # Get bounding box
        obj = root.find("object")
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        return xmin, ymin, xmax, ymax, width, height

    def _parse_bbox_from_xml(self, xml_path: str) -> tuple:
        """Parse bbox from XML as normalized (x_center, y_center, width, height)."""
        xmin, ymin, xmax, ymax, width, height = self._parse_bbox_xyxy_from_xml(xml_path)

        # Convert to center, width, height normalized.
        x_center = (xmin + xmax) / 2 / width
        y_center = (ymin + ymax) / 2 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        return x_center, y_center, w, h

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample["img_path"]).convert("RGB")

        targets = {}

        # Classification target
        if "category" in self.tasks or "classification" in self.tasks:
            targets[self._class_key] = sample["label"]

        # Segmentation target
        if "segmentation" in self.tasks:
            mask = Image.open(sample["mask_path"])
            mask = np.array(mask, dtype=np.int64)
            # Convert trimap (1,2,3) -> (0,1,2)
            mask = mask - 1
            targets["segmentation"] = mask

        # Localization target
        if "localization" in self.tasks:
            bbox = self._parse_bbox_from_xml(sample["xml_path"])
            targets["localization"] = bbox

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            targets = self.target_transform(targets)

        # Return format
        if len(targets) == 1:
            return image, list(targets.values())[0]

        return image, targets


if __name__ == "__main__":
    dataset = OxfordIIITPetDataset(
        root="data",
        split="train",
        tasks=("category", "segmentation", "localization"),
    )
    print(f"Dataset size: {len(dataset)}")
    img, target = dataset[0]
    print(f"Image size: {img.size}, Target keys: {list(target.keys())}")
    print(f"Segmentation mask unique labels: {np.unique(target['segmentation'])}")

    image, mask, color_mask, overlay = dataset.get_segmentation_visualization(10)
    Image.fromarray(image).save("seg_sample_image.png")
    Image.fromarray(color_mask).save("seg_sample_mask_color.png")
    Image.fromarray(overlay).save("seg_sample_overlay.png")
    print("Saved visualization files:")
    print("  seg_sample_image.png")
    print("  seg_sample_mask_color.png")
    print("  seg_sample_overlay.png")

    saved = dataset.save_localization_visualizations(
        out_dir="bbox_samples",
        num_samples=6,
        box_width=3,
    )
    print("Saved localization bbox visualizations:")
    for idx, out_path, bbox_xyxy, bbox_xywh_norm in saved:
        print(
            f"  idx={idx} path={out_path} "
            f"xyxy={bbox_xyxy} "
            f"xywh_norm=({bbox_xywh_norm[0]:.4f}, {bbox_xywh_norm[1]:.4f}, "
            f"{bbox_xywh_norm[2]:.4f}, {bbox_xywh_norm[3]:.4f})"
        )

    
