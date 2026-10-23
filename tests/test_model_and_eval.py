import numpy as np
import pytest
import torch

from abnormality_detection.calibration import fit_temperature, softmax
from abnormality_detection.evaluation import abnormal_scores, compute_metrics, expected_calibration_error
from abnormality_detection.labels import CLASSES
from abnormality_detection.model import build_model, count_parameters, load_bundle, save_bundle


def test_vgg16_matches_thesis_architecture():
    m = build_model("vgg16", pretrained=False, freeze_backbone=True)
    c = count_parameters(m)
    assert c["trainable"] == 512 * 7 * 7 * 14 + 14 and 14_000_000 < c["total"] < 16_000_000  # ~15M params, like the 63 MB .h5; only the dense head trains
    assert m(torch.zeros(2, 3, 64, 64)).shape == (2, 14)


@pytest.mark.parametrize("arch", ["resnet50", "densenet121"])
def test_other_architectures_build(arch):
    m = build_model(arch, pretrained=False)
    assert m.eval()(torch.zeros(1, 3, 64, 64)).shape == (1, 14)


def test_unknown_arch_rejected():
    with pytest.raises(ValueError):
        build_model("alexnet")


def test_bundle_roundtrip(tmp_path):
    m = build_model("vgg16", pretrained=False)
    save_bundle(m, str(tmp_path), arch="vgg16", image_size=64, temperature=1.7)
    m2, meta = load_bundle(str(tmp_path))
    assert meta["temperature"] == 1.7 and meta["classes"] == CLASSES
    x = torch.randn(1, 3, 64, 64)
    assert torch.allclose(m.eval()(x), m2(x), atol=1e-5)


def test_metrics_on_perfect_and_imperfect_predictions():
    y = np.array([0, 1, 8, 9, 13, 8])
    perfect = np.eye(14)[y]
    m = compute_metrics(y, perfect)
    assert m["accuracy"] == 1 and m["macro_f1"] > 0 and m["abnormal_vs_normal"]["recall"] == 1 and m["abnormal_vs_normal"]["auroc"] == 1
    wrong = np.eye(14)[(y + 1) % 14]
    assert compute_metrics(y, wrong)["accuracy"] == 0
    assert len(m["confusion_matrix"]) == 14


def test_abnormal_score_is_mass_on_non_normal_classes():
    p = np.zeros((2, 14))
    p[0, CLASSES.index("Polyp")] = 1
    p[1, CLASSES.index("Pylorus")] = 1
    assert abnormal_scores(p).tolist() == [1.0, 0.0]


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 14, 600)
    logits = rng.normal(0, 1, (600, 14))
    logits[np.arange(600), y] += 1.5       # informative but modest
    logits = logits * 6                    # grossly overconfident
    t = fit_temperature(logits, y)
    assert t > 2
    assert expected_calibration_error(softmax(logits, t), y) < expected_calibration_error(softmax(logits), y)
