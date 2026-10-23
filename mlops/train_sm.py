"""SageMaker training entry point (PyTorch script mode).

Channel ``training`` must contain ``manifest.csv`` (+ ``video_id`` column recommended) and the images.
Writes the model bundle to ``$SM_MODEL_DIR`` and ``metrics.json`` to ``$SM_OUTPUT_DATA_DIR``. Metrics are also
printed as ``test_macro_f1=...`` so SageMaker Pipelines can read them through metric definitions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from abnormality_detection.data import grouped_split, make_synthetic_dataset, read_manifest  # noqa: E402
from abnormality_detection.model import save_bundle  # noqa: E402
from abnormality_detection.training import TrainConfig, evaluate, fit  # noqa: E402


def str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="vgg16")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrained", type=str2bool, default=True)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--synthetic", type=str2bool, default=False, help="smoke test on synthetic images")
    p.add_argument("--training", default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--output-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    root = a.training
    if a.synthetic:
        root = os.path.join(a.output_dir, "synthetic")
        make_synthetic_dataset(root, n_per_class=8, size=64)
    df = read_manifest(os.path.join(root, "manifest.csv"))
    train_df, test_df = grouped_split(df, 0.2, a.seed)
    train_df, val_df = grouped_split(train_df, 0.15, a.seed)
    cfg = TrainConfig(arch=a.arch, epochs=a.epochs, batch_size=a.batch_size, image_size=a.image_size, lr=a.lr,
                      pretrained=a.pretrained, seed=a.seed, num_workers=a.num_workers)
    model, report = fit(train_df, val_df, root, cfg)
    test = evaluate(model, test_df, root, cfg.image_size, report["temperature"])
    save_bundle(model, a.model_dir, arch=cfg.arch, image_size=cfg.image_size, temperature=report["temperature"], dropout=cfg.dropout,
                metrics={k: test[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "mcc", "ece")})
    os.makedirs(a.output_dir, exist_ok=True)
    with open(os.path.join(a.output_dir, "metrics.json"), "w") as f:
        json.dump({**test, "validation": report["val_metrics"], "history": report["history"]}, f, indent=1)
    abn = test["abnormal_vs_normal"]
    print(f"test_macro_f1={test['macro_f1']:.4f}", flush=True)
    print(f"test_balanced_accuracy={test['balanced_accuracy']:.4f}", flush=True)
    print(f"test_abnormal_recall={abn.get('recall', 0.0):.4f}", flush=True)


if __name__ == "__main__":
    main()
