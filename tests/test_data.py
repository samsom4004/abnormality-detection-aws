import pytest
import torch

from abnormality_detection.data import (
    ManifestDataset,
    class_weights,
    eval_transform,
    grouped_split,
    read_manifest,
    train_transform,
)
from abnormality_detection.labels import CLASSES, NORMAL_LIKE, abnormal_indices


def test_labels_are_consistent():
    assert len(CLASSES) == 14 and set(NORMAL_LIKE) <= set(CLASSES)
    assert len(abnormal_indices()) == 14 - len(NORMAL_LIKE)


def test_synthetic_dataset_and_manifest(dataset):
    root, df = dataset
    assert len(df) == 14 * 8 and set(df["label"]) == set(CLASSES)
    again = read_manifest(f"{root}/manifest.csv")
    assert list(again.columns) == ["filename", "label", "video_id"]


def test_manifest_validation(tmp_path):
    (tmp_path / "bad.csv").write_text("filename,label\na.png,Alien\n")
    with pytest.raises(ValueError, match="unknown labels"):
        read_manifest(str(tmp_path / "bad.csv"))
    (tmp_path / "bad2.csv").write_text("filename\na.png\n")
    with pytest.raises(ValueError, match="missing columns"):
        read_manifest(str(tmp_path / "bad2.csv"))


def test_grouped_split_never_shares_a_video(dataset):
    _, df = dataset
    tr, te = grouped_split(df, 0.25, seed=1)
    assert set(tr["video_id"]).isdisjoint(set(te["video_id"]))
    assert len(tr) + len(te) == len(df)


def test_split_without_video_ids_warns(dataset):
    _, df = dataset
    with pytest.warns(UserWarning, match="optimistic"):
        grouped_split(df.drop(columns=["video_id"]), 0.25)


def test_class_weights_upweight_rare_classes():
    labels = ["Polyp"] * 90 + ["ulcer"] * 10
    w = class_weights(labels)
    assert w[CLASSES.index("ulcer")] > w[CLASSES.index("Polyp")]
    assert w[CLASSES.index("Pylorus")] == 0  # absent class


def test_transforms_and_dataset(dataset):
    root, df = dataset
    ds = ManifestDataset(df.head(5), root, eval_transform(64))
    x, y = ds[0]
    assert x.shape == (3, 64, 64) and isinstance(y, int)
    # evaluation must be deterministic (no augmentation); training augmentation is random
    assert torch.equal(ds[0][0], ds[0][0])
    from PIL import Image

    im = Image.open(f"{root}/{df.iloc[0]['filename']}").convert("RGB")
    t = train_transform(64)
    assert not all(torch.equal(t(im), t(im)) for _ in range(5))


def test_dataset_labels_match_manifest(dataset):
    root, df = dataset
    ds = ManifestDataset(df, root, eval_transform(64))
    assert [CLASSES[i] for i in ds.y] == list(df["label"])
