"""DeepfakeBench-compatible image manifest dataset.

Each JSONL row is: {"path": "...", "label": 0|1, "type_label": -1|0..C-1,
"video_id": "optional identifier"}.  `type_label=-1` is required for real images.
"""
import json
import random
from pathlib import Path

from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_transform(image_size: int, train: bool):
    ops = []
    if train:
        ops.extend([
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        ])
    else:
        ops.extend([transforms.Resize(int(image_size * 1.14)), transforms.CenterCrop(image_size)])
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transforms.Compose(ops)


class DeepfakeBenchManifest(Dataset):
    def __init__(self, manifest_path: str, image_size: int = 224, train: bool = False):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        with self.manifest_path.open() as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        if not self.rows:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")
        self.transform = build_transform(image_size, train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        try:
            image = Image.open(row["path"]).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Unreadable image in manifest: {row['path']}") from exc
        return {
            "pixel_values": self.transform(image),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "type_label": torch.tensor(int(row.get("type_label", -1)), dtype=torch.long),
            "video_id": str(row.get("video_id", Path(row["path"]).parent.name)),
            "path": row["path"],
        }
