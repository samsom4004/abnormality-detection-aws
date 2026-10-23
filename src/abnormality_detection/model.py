"""Model definitions and the saved bundle format (weights + metadata)."""

from __future__ import annotations

import json
import os

import torch
from torch import nn
from torchvision import models

from .labels import CLASSES, NORMAL_LIKE

ARCHS = ("vgg16", "resnet50", "densenet121")
META_FILE, WEIGHTS_FILE = "meta.json", "model.pt"


def build_model(arch: str = "vgg16", num_classes: int = len(CLASSES), pretrained: bool = True,
                freeze_backbone: bool = True, dropout: float = 0.3) -> nn.Module:
    """``vgg16`` reproduces the thesis model (frozen convolutional base, flatten, dense softmax head).
    ``resnet50`` and ``densenet121`` are lighter stand-ins for the ResNet-152 / DenseNet-161 comparison."""
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {ARCHS}")
    weights = "IMAGENET1K_V1" if pretrained else None
    if arch == "vgg16":
        m = models.vgg16(weights=weights)
        backbone = m.features
        m.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(512 * 7 * 7, num_classes))
    elif arch == "resnet50":
        m = models.resnet50(weights=weights)
        backbone = nn.Sequential(*list(m.children())[:-1])
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
    else:
        m = models.densenet121(weights=weights)
        backbone = m.features
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier.in_features, num_classes))
    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
    return m


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def save_bundle(model: nn.Module, out_dir: str, *, arch: str, image_size: int, temperature: float = 1.0,
                dropout: float = 0.3, metrics: dict | None = None, version: str = "0.1.0") -> str:
    """Write ``model.pt`` (state dict) and ``meta.json``. State dicts avoid pickling arbitrary code."""
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, WEIGHTS_FILE))
    meta = {"arch": arch, "image_size": image_size, "classes": CLASSES, "normal_like": NORMAL_LIKE,
            "temperature": float(temperature), "dropout": dropout, "version": version, "metrics": metrics or {}}
    with open(os.path.join(out_dir, META_FILE), "w") as f:
        json.dump(meta, f, indent=1)
    return out_dir


def load_bundle(model_dir: str):
    with open(os.path.join(model_dir, META_FILE)) as f:
        meta = json.load(f)
    model = build_model(meta["arch"], len(meta["classes"]), pretrained=False, freeze_backbone=False, dropout=meta.get("dropout", 0.3))
    state = torch.load(os.path.join(model_dir, WEIGHTS_FILE), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, meta
