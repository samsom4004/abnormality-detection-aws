"""HTTP server following the SageMaker container contract (``/ping`` and ``/invocations`` on port 8080).

``POST /invocations`` accepts a raw image (``image/jpeg`` or ``image/png``) or JSON ``{"image_base64": "..."}``.
Images are processed in memory and never written to disk or logged.
"""

from __future__ import annotations

import base64
import binascii
import os

from fastapi import FastAPI, HTTPException, Request

from .predictor import Predictor

MAX_BYTES = 8 * 1024 * 1024
app = FastAPI(title="abnormality-detection", version="0.1.0")
_state: dict = {"predictor": None}


def _predictor() -> Predictor:
    if _state["predictor"] is None:
        d = os.environ.get("MODEL_DIR", "/opt/ml/model")
        if not os.path.exists(os.path.join(d, "meta.json")):
            raise HTTPException(status_code=503, detail="model not loaded")
        _state["predictor"] = Predictor.load(d)
    return _state["predictor"]


@app.get("/ping")
def ping():
    _predictor()
    return {"status": "ok"}


@app.post("/invocations")
async def invocations(request: Request):
    ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    body = await request.body()
    if len(body) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="image too large (8 MB max)")
    if ctype in ("image/jpeg", "image/png", "application/octet-stream"):
        data = body
    elif ctype == "application/json":
        try:
            import json

            data = base64.b64decode(json.loads(body)["image_base64"], validate=True)
        except (KeyError, ValueError, binascii.Error, TypeError) as exc:
            raise HTTPException(status_code=400, detail="send {'image_base64': '<base64 image>'}") from exc
    else:
        raise HTTPException(status_code=415, detail="use image/jpeg, image/png or application/json")
    p = _predictor()
    try:
        return {"predictions": p.predict([p.open_image(data)])}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
