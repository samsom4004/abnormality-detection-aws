import json

from abnormality_detection import cli


def test_cli_end_to_end(tmp_path, capsys):
    data, out = tmp_path / "data", tmp_path / "model"
    cli.main(["synthetic", "--out", str(data), "--per-class", "6"])
    cli.main(["train", "--data", str(data), "--out", str(out), "--epochs", "1", "--image-size", "64", "--no-pretrained", "--batch-size", "16"])
    assert (out / "metrics.json").exists() and "test macro-F1" in capsys.readouterr().out
    img = next(data.glob("*.png"))
    cli.main(["predict", "--model", str(out), "--image", str(img), "--note", "--dry-run"])
    text = capsys.readouterr().out
    assert "abnormal_probability" in text and "Not a medical device" in text
    cli.main(["model-card", "--model", str(out), "--dry-run"])
    assert "Model card draft" in capsys.readouterr().out
    assert json.loads((out / "meta.json").read_text())["image_size"] == 64
