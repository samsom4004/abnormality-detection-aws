"""Inference: preprocessing, calibrated probabilities, anomaly score and a human-review flag."""

from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image

from .calibration import softmax
from .data import eval_transform
from .evaluation import abnormal_scores
from .labels import DISCLAIMER, LOW_QUALITY
from .model import load_bundle

REVIEW_CONFIDENCE = 0.60       # below this top-class probability: send to a human
UNCERTAIN_BAND = (0.35, 0.65)  # abnormal probability inside this band: send to a human


class Predictor:
    def __init__(self, model, meta: dict):
        self.model, self.meta = model.eval(), meta
        self.classes = meta["classes"]
        self.tf = eval_transform(meta["image_size"])
        self.temperature = float(meta.get("temperature", 1.0))

    @classmethod
    def load(cls, model_dir: str) -> "Predictor":
        return cls(*load_bundle(model_dir))

    @staticmethod
    def open_image(data: bytes) -> Image.Image:
        try:
            im = Image.open(io.BytesIO(data))
            im.load()
            return im.convert("RGB")
        except Exception as exc:
            raise ValueError("could not read the image (use JPEG or PNG)") from exc

    @torch.no_grad()
    def predict(self, images: list[Image.Image], top_k: int = 3) -> list[dict]:
        x = torch.stack([self.tf(im.convert("RGB")) for im in images])
        probs = softmax(self.model(x).float().numpy(), self.temperature)
        abn = abnormal_scores(probs, self.classes, self.meta["normal_like"])
        out = []
        for p, a in zip(probs, abn):
            order = np.argsort(-p)[:top_k]
            top = [{"label": self.classes[i], "probability": round(float(p[i]), 4)} for i in order]
            entropy = float(-(p * np.log(np.clip(p, 1e-12, 1))).sum() / np.log(len(p)))
            review = (top[0]["probability"] < REVIEW_CONFIDENCE or UNCERTAIN_BAND[0] <= a <= UNCERTAIN_BAND[1]
                      or top[0]["label"] in LOW_QUALITY)
            out.append({"top": top, "abnormal_probability": round(float(a), 4), "uncertainty": round(entropy, 4),
                        "needs_review": bool(review), "model_version": self.meta.get("version"), "disclaimer": DISCLAIMER})
        return out
