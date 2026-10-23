import base64
import io
import json

import pytest
from conftest import load_script
from fastapi.testclient import TestClient
from PIL import Image


def _png(size=64):
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (120, 30, 30)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture()
def client(bundle, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", bundle[0])
    from abnormality_detection import serve

    serve._state["predictor"] = None
    return TestClient(serve.app)


def test_ping_and_raw_image(client):
    assert client.get("/ping").json() == {"status": "ok"}
    r = client.post("/invocations", content=_png(), headers={"content-type": "image/png"})
    assert r.status_code == 200 and r.json()["predictions"][0]["top"]


def test_json_base64(client):
    r = client.post("/invocations", json={"image_base64": base64.b64encode(_png()).decode()})
    assert r.status_code == 200


@pytest.mark.parametrize("kw,code", [
    (dict(content=b"junk", headers={"content-type": "image/png"}), 400),
    (dict(content=b"x", headers={"content-type": "text/plain"}), 415),
    (dict(json={"nope": 1}), 400),
    (dict(content=b"0" * (8 * 1024 * 1024 + 1), headers={"content-type": "image/png"}), 413),
])
def test_bad_requests(client, kw, code):
    assert client.post("/invocations", **kw).status_code == code


def test_ping_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    from abnormality_detection import serve

    serve._state["predictor"] = None
    assert TestClient(serve.app).get("/ping").status_code == 503


def test_sagemaker_inference_handlers(bundle):
    inf = load_script("inference_sm")
    p = inf.model_fn(bundle[0])
    body, ctype = inf.output_fn(inf.predict_fn(inf.input_fn(_png(), "image/png"), p))
    assert ctype == "application/json" and json.loads(body)["predictions"][0]["top"]
    with pytest.raises(ValueError):
        inf.input_fn(b"x", "text/csv")


def test_sagemaker_training_script_smoke(tmp_path):
    train = load_script("train_sm")
    train.main(["--synthetic", "true", "--arch", "vgg16", "--epochs", "1", "--image-size", "64", "--pretrained", "false",
                "--batch-size", "16", "--num-workers", "0", "--model-dir", str(tmp_path / "m"), "--output-dir", str(tmp_path / "o")])
    assert (tmp_path / "m" / "model.pt").exists() and (tmp_path / "m" / "meta.json").exists()
    assert "macro_f1" in json.loads((tmp_path / "o" / "metrics.json").read_text())


@pytest.fixture(scope="module")
def bundle224(tmp_path_factory):
    from abnormality_detection.model import build_model, save_bundle

    d = tmp_path_factory.mktemp("b224")
    save_bundle(build_model("vgg16", pretrained=False), str(d), arch="vgg16", image_size=224)
    return str(d)


def test_onnx_export_is_valid_and_matches_pytorch(bundle224, tmp_path):
    onnx = pytest.importorskip("onnx")
    import numpy as np
    import torch

    from abnormality_detection.onnx_export import export_onnx
    from abnormality_detection.predictor import Predictor

    p = Predictor.load(bundle224)
    path = export_onnx(p, str(tmp_path / "m.onnx"))
    onnx.checker.check_model(onnx.load(path))
    ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
    x = torch.randn(2, 3, 224, 224)
    got = ort.InferenceSession(path).run(None, {"image": x.numpy()})[0]
    assert np.allclose(got, p.model(x).detach().numpy(), atol=1e-3)


def test_onnx_export_rejects_unsupported_vgg_size(bundle, tmp_path):
    from abnormality_detection.onnx_export import export_onnx
    from abnormality_detection.predictor import Predictor

    with pytest.raises(ValueError, match="224"):
        export_onnx(Predictor.load(bundle[0]), str(tmp_path / "m.onnx"))
