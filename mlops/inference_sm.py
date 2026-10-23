"""SageMaker inference handlers (PyTorch container). Accepts ``image/jpeg`` or ``image/png``."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from abnormality_detection.predictor import Predictor  # noqa: E402

MAX_BYTES = 8 * 1024 * 1024


def model_fn(model_dir):
    return Predictor.load(model_dir)


def input_fn(request_body, content_type="image/jpeg"):
    if content_type not in ("image/jpeg", "image/png", "application/x-image"):
        raise ValueError(f"unsupported content type: {content_type}")
    if len(request_body) > MAX_BYTES:
        raise ValueError("image too large (8 MB max)")
    return Predictor.open_image(request_body)


def predict_fn(image, predictor):
    return predictor.predict([image])


def output_fn(prediction, accept="application/json"):
    return json.dumps({"predictions": prediction}), "application/json"
