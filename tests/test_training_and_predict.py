import numpy as np
from PIL import Image

from abnormality_detection.predictor import Predictor


def test_training_runs_and_reports_everything(bundle):
    _, metrics, report = bundle
    assert 0 <= metrics["macro_f1"] <= 1 and len(report["history"]) >= 1 and report["temperature"] > 0
    assert report["parameters"]["trainable"] < report["parameters"]["total"]
    assert set(metrics["abnormal_vs_normal"]) >= {"threshold", "n_abnormal"}


def test_synthetic_task_is_learnable_with_a_few_epochs(dataset, tmp_path):
    """Sanity check that the loop actually learns: accuracy beats chance (1/14) clearly."""
    from abnormality_detection.data import grouped_split
    from abnormality_detection.training import TrainConfig, fit

    root, df = dataset
    tr, te = grouped_split(df, 0.25, seed=3)
    cfg = TrainConfig(epochs=6, batch_size=16, image_size=64, pretrained=False, lr=1e-3, seed=3, patience=10)
    model, rep = fit(tr, tr, root, cfg)  # train and validate on the same frames: tests the loop, not generalisation
    assert rep["val_metrics"]["accuracy"] > 0.3


def test_predictor_output_contract(bundle, dataset):
    model_dir, _, _ = bundle
    root, df = dataset
    p = Predictor.load(model_dir)
    imgs = [Image.open(f"{root}/{f}") for f in df["filename"].head(3)]
    out = p.predict(imgs)
    assert len(out) == 3
    for o in out:
        assert abs(sum(t["probability"] for t in o["top"])) <= 1.0 + 1e-6 and len(o["top"]) == 3
        assert 0 <= o["abnormal_probability"] <= 1 and 0 <= o["uncertainty"] <= 1
        assert isinstance(o["needs_review"], bool) and "Not a medical device" in o["disclaimer"]


def test_predictor_rejects_non_images(bundle):
    import pytest

    p = Predictor.load(bundle[0])
    with pytest.raises(ValueError):
        p.open_image(b"this is not an image")


def test_low_confidence_triggers_review(bundle):
    p = Predictor.load(bundle[0])
    noise = Image.fromarray((np.random.default_rng(0).random((64, 64, 3)) * 255).astype("uint8"))
    out = p.predict([noise])[0]
    assert out["needs_review"] in (True, False)  # contract: always present
    p.temperature = 50.0                          # a very flat distribution must always be reviewed
    assert p.predict([noise])[0]["needs_review"] is True
