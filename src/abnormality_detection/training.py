"""Training and evaluation loops (CPU or GPU)."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .calibration import fit_temperature, softmax
from .data import ManifestDataset, class_weights, eval_transform, train_transform
from .evaluation import compute_metrics
from .labels import CLASSES
from .model import build_model, count_parameters


@dataclass
class TrainConfig:
    arch: str = "vgg16"
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-4
    image_size: int = 224
    freeze_backbone: bool = True
    dropout: float = 0.3
    pretrained: bool = True
    use_class_weights: bool = True
    patience: int = 3
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    zs, ys = [], []
    for x, y in loader:
        zs.append(model(x.to(device)).float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(zs), np.concatenate(ys)


def fit(train_df, val_df, root: str, cfg: TrainConfig):
    """Train, keep the best epoch by validation macro-F1, calibrate on the validation set.

    Returns ``(model, report)`` where report holds history, validation metrics and temperature."""
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    model = build_model(cfg.arch, pretrained=cfg.pretrained, freeze_backbone=cfg.freeze_backbone, dropout=cfg.dropout).to(device)

    tr = ManifestDataset(train_df, root, train_transform(cfg.image_size))
    va = ManifestDataset(val_df, root, eval_transform(cfg.image_size))
    tl = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    vl = DataLoader(va, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    w = class_weights(train_df["label"]).to(device) if cfg.use_class_weights else None
    loss_fn = nn.CrossEntropyLoss(weight=w)  # the weights are actually applied here
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)

    best, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(cfg.epochs):
        model.train()
        total, n = 0.0, 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(y)
            n += len(y)
        z, y_val = predict_logits(model, vl, device)
        m = compute_metrics(y_val, softmax(z))
        history.append({"epoch": epoch + 1, "train_loss": total / max(n, 1), "val_macro_f1": m["macro_f1"], "val_accuracy": m["accuracy"]})
        if m["macro_f1"] > best:
            best, best_state, bad = m["macro_f1"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    z, y_val = predict_logits(model, vl, device)
    temperature = fit_temperature(z, y_val)
    report = {"config": asdict(cfg), "history": history, "temperature": temperature, "parameters": count_parameters(model),
              "val_metrics": compute_metrics(y_val, softmax(z, temperature))}
    return model.cpu(), report


def evaluate(model: nn.Module, df, root: str, image_size: int, temperature: float = 1.0, batch_size: int = 32, device: str = "auto") -> dict:
    ds = ManifestDataset(df, root, eval_transform(image_size))
    dev = pick_device(device)
    z, y = predict_logits(model.to(dev), DataLoader(ds, batch_size=batch_size), dev)
    return compute_metrics(y, softmax(z, temperature), CLASSES)
