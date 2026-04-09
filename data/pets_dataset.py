"""Dataset loader for Oxford-IIIT Pet."""

import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
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

