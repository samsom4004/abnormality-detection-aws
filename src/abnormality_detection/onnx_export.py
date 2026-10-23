"""Export to ONNX (portable runtime, SageMaker Neo / edge). Replaces the original TFLite conversion."""

from __future__ import annotations

import inspect

import torch

from .predictor import Predictor


def export_onnx(predictor: Predictor, path: str, opset: int = 17) -> str:
    size = predictor.meta["image_size"]
    if predictor.meta["arch"] == "vgg16" and size % 224 != 0:
        raise ValueError("ONNX export of VGG16 needs image_size 224 (or a multiple): the adaptive pooling layer "
                         "to 7x7 is only exportable when the feature map is a multiple of 7")
    dummy = torch.zeros(1, 3, size, size)
    kwargs = dict(input_names=["image"], output_names=["logits"],
                  dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=opset)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False  # newer torch defaults to the dynamo exporter, which needs extra packages
    torch.onnx.export(predictor.model, dummy, path, **kwargs)
    try:
        import onnx

        onnx.checker.check_model(onnx.load(path))
    except ImportError:
        pass
    return path
