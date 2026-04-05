"""Dataset loader for Oxford-IIIT Pet.
"""

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class OxfordIIITPetDataset(Dataset):
    """
    Oxford-IIIT Pet multi-task dataset loader.

    Supports:
    - classification (breed label)
    - segmentation (trimap mask)
    """

    def __init__(
        self,
        root,
        split="trainval",              # "trainval" or "test"
        tasks=("category",),          # ("category", "segmentation")
        transform=None,
        target_transform=None
    ):
        self.root = root
        self.split = split
        self.tasks = tuple(tasks) if isinstance(tasks, (list, tuple)) else (tasks,)
        self.transform = transform
        self.target_transform = target_transform

        if self.split not in ("trainval", "test"):
            raise ValueError("split must be 'trainval' or 'test'")

        # Allow "classification" as an alias for "category"
        self._class_key = "classification" if "classification" in self.tasks and "category" not in self.tasks else "category"

        # Allow passing a parent folder that contains "oxford-iiit-pet"
        candidate = os.path.join(self.root, "oxford-iiit-pet")
        if not os.path.isdir(os.path.join(self.root, "images")) and os.path.isdir(candidate):
            self.root = candidate

        # Paths
        self.images_dir = os.path.join(self.root, "images")
        self.masks_dir = os.path.join(self.root, "annotations", "trimaps")

        # Annotation file
        split_file = "trainval.txt" if self.split == "trainval" else "test.txt"
        self.ann_path = os.path.join(self.root, "annotations", split_file)

        # Load metadata
        self.samples = self._load_annotations()

    def _load_annotations(self):
        samples = []

        with open(self.ann_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            img_id = parts[0]
            label = int(parts[1]) - 1   # convert to 0-based

            img_path = os.path.join(self.images_dir, img_id + ".jpg")
            mask_path = os.path.join(self.masks_dir, img_id + ".png")

            samples.append({
                "img_path": img_path,
                "mask_path": mask_path,
                "label": label
            })

        return samples

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
    dataset = OxfordIIITPetDataset(root="data", split="test", tasks=("category", "segmentation"))
    print(f"Dataset size: {len(dataset)}")
    img, target = dataset[0]
    print(f"Image size: {img.size}, Target keys: {list(target.keys())}")
