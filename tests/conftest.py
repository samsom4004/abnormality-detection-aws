import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"mlops_{name}", ROOT / "mlops" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    from abnormality_detection.data import make_synthetic_dataset

    root = tmp_path_factory.mktemp("synthetic")
    df = make_synthetic_dataset(str(root), n_per_class=8, size=64)
    return str(root), df


@pytest.fixture(scope="session")
def bundle(dataset, tmp_path_factory):
    """A tiny trained model (VGG16, random init, 64px, 2 epochs) saved as a bundle."""
    from abnormality_detection.data import grouped_split
    from abnormality_detection.model import save_bundle
    from abnormality_detection.training import TrainConfig, evaluate, fit

    root, df = dataset
    train_df, test_df = grouped_split(df, 0.25, seed=0)
    train_df, val_df = grouped_split(train_df, 0.2, seed=0)
    cfg = TrainConfig(arch="vgg16", epochs=2, batch_size=16, image_size=64, pretrained=False, seed=0, patience=5)
    model, report = fit(train_df, val_df, root, cfg)
    metrics = evaluate(model, test_df, root, 64, report["temperature"])
    out = tmp_path_factory.mktemp("bundle")
    save_bundle(model, str(out), arch="vgg16", image_size=64, temperature=report["temperature"], dropout=cfg.dropout,
                metrics={"macro_f1": metrics["macro_f1"]})
    return str(out), metrics, report
