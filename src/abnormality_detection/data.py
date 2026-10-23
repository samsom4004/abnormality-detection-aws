"""Manifest handling, leakage-safe splitting, class weights, datasets and a synthetic image generator.

Manifest CSV columns: ``filename`` (relative to the image root), ``label`` (one of ``labels.CLASSES``) and,
strongly recommended, ``video_id`` (which capsule video the frame came from).

Why ``video_id`` matters: consecutive video frames are nearly identical. Splitting individual *frames* at
random puts near-duplicates in both train and test and inflates accuracy. We split by video instead.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from .labels import CLASSES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_manifest(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"filename", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    unknown = set(df["label"]) - set(CLASSES)
    if unknown:
        raise ValueError(f"unknown labels in manifest: {sorted(unknown)}")
    return df


def grouped_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42, group_col: str = "video_id"):
    """Return (train_df, test_df). Splits by video when ``group_col`` exists, otherwise falls back to a
    stratified frame split and emits a warning (results will be optimistic)."""
    if group_col in df.columns and df[group_col].nunique() > 1:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(gss.split(df, groups=df[group_col]))
        return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    warnings.warn(f"no usable '{group_col}' column: splitting frames at random; expect optimistic metrics", stacklevel=2)
    strat = df["label"] if df["label"].value_counts().min() >= 2 else None
    tr, te = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def class_weights(labels, num_classes: int = len(CLASSES)) -> torch.Tensor:
    """Inverse-frequency weights: total / (num_classes * count). Absent classes get weight 0."""
    idx = np.asarray([CLASSES.index(x) if isinstance(x, str) else int(x) for x in labels])
    counts = np.bincount(idx, minlength=num_classes).astype(float)
    w = np.zeros(num_classes)
    nz = counts > 0
    w[nz] = counts.sum() / (num_classes * counts[nz])
    return torch.tensor(w, dtype=torch.float32)


def train_transform(size: int = 224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ColorJitter(0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(size: int = 224):
    """No augmentation. (The original script reused the augmenting generator for testing.)"""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: str, transform):
        self.files = [os.path.join(root, f) for f in df["filename"]]
        self.y = np.array([CLASSES.index(x) for x in df["label"]], dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.files[i]) as im:
            x = self.transform(im.convert("RGB"))
        return x, int(self.y[i])


def make_synthetic_dataset(root: str, n_per_class: int = 8, size: int = 64, n_videos: int = 4, seed: int = 0) -> pd.DataFrame:
    """Write tiny class-coded images plus a manifest. For tests, demos and CI only - NOT medical data."""
    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)
    rows = []
    for ci, cls in enumerate(CLASSES):
        hue = np.array([(37 * ci) % 255, (91 * ci + 40) % 255, (53 * ci + 120) % 255], dtype=float)
        for k in range(n_per_class):
            img = np.clip(hue[None, None, :] + rng.normal(0, 18, (size, size, 3)), 0, 255).astype(np.uint8)
            r0 = (ci * 3) % (size - 12)
            img[r0:r0 + 12, (k * 5) % (size - 12):(k * 5) % (size - 12) + 12, :] = 255 - hue.astype(np.uint8)
            name = f"{ci:02d}_{k:03d}.png"
            Image.fromarray(img).save(os.path.join(root, name))
            rows.append({"filename": name, "label": cls, "video_id": f"video_{(k + ci) % n_videos}"})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(root, "manifest.csv"), index=False)
    return df
