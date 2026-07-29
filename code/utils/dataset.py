# geo_dataset.py

import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, normalize


class ImageGeolocationDataset(Dataset):
    def __init__(self, root_dir, label_vocab_path):
        
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"
        self.labels = pd.read_csv(self.root_dir / "labels.csv")
        
        self.categorical_cols = ["country", "land_cover", "climate", "soil",]
        self.continuous_cols = ["latitude", "longitude", "dist_sea",]

        # Load the editable categorical label vocabulary.
        with open(label_vocab_path, "r") as f:
            self.label_vocab = json.load(f)

        # Build maps from the JSON list order.
        self.category_maps = {}

        for col in self.categorical_cols:
            self.category_maps[col] = {
                value: idx
                for idx, value in enumerate(self.label_vocab[col])
            }

        # Normalize categorical columns.
        for col in self.categorical_cols:
            self.labels[col] = self.labels[col].fillna("__missing__").astype(str).map(str.strip)

        # Skip rows where any categorical value is not in the JSON list.
        keep_mask = pd.Series(True, index=self.labels.index)

        for col in self.categorical_cols:
            keep_mask &= self.labels[col].isin(self.label_vocab[col])

        skipped = len(self.labels) - keep_mask.sum()
        self.labels = self.labels[keep_mask].reset_index(drop=True)

        print(
            f"{self.root_dir}: kept {len(self.labels)} rows, skipped {skipped} rows "
            f"because of disallowed categorical labels."
        )

        # Encode categorical columns as integer class IDs using the JSON order.
        for col in self.categorical_cols:
            self.labels[col] = self.labels[col].map(self.category_maps[col]).astype("int64")
        
        # Make continuous columns numeric floats.
        for col in self.continuous_cols:
            self.labels[col] = pd.to_numeric(self.labels[col], errors="coerce").astype("float32")
            
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]

        image = self._load_image(row["id"])

        sample = {
            "image": image,
            "latitude": torch.tensor(row["latitude"], dtype=torch.float32),
            "longitude": torch.tensor(row["longitude"], dtype=torch.float32),
            "country": torch.tensor(row["country"], dtype=torch.long),
            "land_cover": torch.tensor(row["land_cover"], dtype=torch.long),
            "climate": torch.tensor(row["climate"], dtype=torch.long),
            "soil": torch.tensor(row["soil"], dtype=torch.long),
            "dist_sea": torch.tensor(row["dist_sea"], dtype=torch.float32),
        }

        return sample

    def _load_image(self, image_id):
        for ext in [".jpg", ".jpeg", ".png"]:
            path = self.image_dir / f"{image_id}{ext}"
            if path.exists():
                image = Image.open(path).convert("RGB")
                image = to_tensor(image)
                return image

        raise FileNotFoundError(f"No image found for id: {image_id}")


def _center_crop_or_reflect_pad(image, target_h=512, target_w=910):
    _, h, w = image.shape

    # Center crop if too tall.
    if h > target_h:
        top = (h - target_h) // 2
        image = image[:, top:top + target_h, :]

    # Center crop if too wide.
    if w > target_w:
        left = (w - target_w) // 2
        image = image[:, :, left:left + target_w]

    _, h, w = image.shape

    # Center pad with reflected pixels if too short or narrow.
    pad_h = target_h - h
    pad_w = target_w - w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if (pad_top >= h or pad_bottom >= h or pad_left >= w or pad_right >= w):
        pad_mode = "replicate"
    else:
        pad_mode = "reflect"

    return F.pad(
        image,
        pad=(pad_left, pad_right, pad_top, pad_bottom),
        mode=pad_mode,
    )

def collate_fn(samples):
    """
    Crops or reflect-pads every image to 3 x 512 x 910.
    Returns a dictionary of batched tensors.
    """
    images = [
        normalize(
            _center_crop_or_reflect_pad(sample["image"]),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        for sample in samples
    ]

    batch = {
        "images": torch.stack(images, dim=0),
    }

    label_keys = [
        "latitude",
        "longitude",
        "country",
        "land_cover",
        "climate",
        "soil",
        "dist_sea",
    ]

    for key in label_keys:
        batch[key] = torch.stack([sample[key] for sample in samples], dim=0)

    return batch


def make_dataloader(
    root_dir,
    label_vocab_path,
    batch_size=32,
    shuffle=True,
    num_workers=4,
):
    dataset = ImageGeolocationDataset(root_dir, label_vocab_path)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return dataloader