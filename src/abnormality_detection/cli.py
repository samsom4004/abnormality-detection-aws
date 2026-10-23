"""Command line interface."""

from __future__ import annotations

import argparse
import json
import os

from .bedrock_report import model_card_draft, triage_note
from .data import grouped_split, make_synthetic_dataset, read_manifest
from .model import save_bundle
from .predictor import Predictor
from .training import TrainConfig, evaluate, fit


def cmd_synthetic(a):
    df = make_synthetic_dataset(a.out, n_per_class=a.per_class, size=a.size)
    print(f"wrote {len(df)} synthetic images + manifest.csv to {a.out} (NOT medical data)")


def cmd_train(a):
    df = read_manifest(os.path.join(a.data, "manifest.csv") if os.path.isdir(a.data) else a.data)
    root = a.data if os.path.isdir(a.data) else os.path.dirname(a.data)
    train_df, test_df = grouped_split(df, test_size=0.2, seed=a.seed)
    train_df, val_df = grouped_split(train_df, test_size=0.15, seed=a.seed)
    cfg = TrainConfig(arch=a.arch, epochs=a.epochs, batch_size=a.batch_size, image_size=a.image_size, lr=a.lr,
                      pretrained=not a.no_pretrained, seed=a.seed)
    model, report = fit(train_df, val_df, root, cfg)
    test = evaluate(model, test_df, root, cfg.image_size, report["temperature"])
    metrics = {**test, "validation": report["val_metrics"], "history": report["history"]}
    save_bundle(model, a.out, arch=cfg.arch, image_size=cfg.image_size, temperature=report["temperature"], dropout=cfg.dropout,
                metrics={k: test[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "mcc", "ece")})
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print(f"test macro-F1 {test['macro_f1']:.3f} | balanced acc {test['balanced_accuracy']:.3f} | saved to {a.out}")


def cmd_predict(a):
    p = Predictor.load(a.model)
    with open(a.image, "rb") as f:
        res = p.predict([p.open_image(f.read())])[0]
    print(json.dumps(res, indent=1))
    if a.note:
        print(triage_note(res, dry_run=a.dry_run))


def cmd_export(a):
    from .onnx_export import export_onnx

    print("wrote", export_onnx(Predictor.load(a.model), a.out))


def cmd_card(a):
    with open(os.path.join(a.model, "metrics.json")) as f:
        print(model_card_draft(json.load(f), dry_run=a.dry_run))


def main(argv=None):
    p = argparse.ArgumentParser(prog="abnormality-detection")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synthetic", help="write a tiny synthetic image dataset (demo/CI only)")
    s.add_argument("--out", required=True)
    s.add_argument("--per-class", type=int, default=8)
    s.add_argument("--size", type=int, default=64)
    s.set_defaults(fn=cmd_synthetic)
    t = sub.add_parser("train", help="train, calibrate, evaluate and save a model bundle")
    t.add_argument("--data", required=True, help="folder with manifest.csv and images")
    t.add_argument("--out", default="artifacts/model")
    t.add_argument("--arch", default="vgg16", choices=["vgg16", "resnet50", "densenet121"])
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--image-size", type=int, default=224)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--no-pretrained", action="store_true", help="random init (no ImageNet download)")
    t.set_defaults(fn=cmd_train)
    q = sub.add_parser("predict", help="classify one image")
    q.add_argument("--model", required=True)
    q.add_argument("--image", required=True)
    q.add_argument("--note", action="store_true", help="also print a Bedrock triage note")
    q.add_argument("--dry-run", action="store_true")
    q.set_defaults(fn=cmd_predict)
    e = sub.add_parser("export-onnx")
    e.add_argument("--model", required=True)
    e.add_argument("--out", default="artifacts/model.onnx")
    e.set_defaults(fn=cmd_export)
    c = sub.add_parser("model-card", help="draft a model card from metrics.json with Bedrock")
    c.add_argument("--model", required=True)
    c.add_argument("--dry-run", action="store_true")
    c.set_defaults(fn=cmd_card)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
