"""Amazon Bedrock helpers: a cautious triage note and a model-card draft.

Hard rules: the model never sees images or patient identifiers, only *model outputs* (label, probability,
flags) or *aggregate validation metrics*, and every note carries the research-prototype disclaimer. The
language model cannot change the classification or approve anything.
"""

from __future__ import annotations

import json
import os

from .labels import DISCLAIMER

ALLOWED_PREDICTION_KEYS = {"top", "abnormal_probability", "uncertainty", "needs_review", "model_version", "disclaimer"}
ALLOWED_TOP_KEYS = {"label", "probability"}
FORBIDDEN_IN_METRICS = {"image", "images", "image_base64", "filename", "filenames", "patient", "patient_id", "video_id", "name", "dob"}

TRIAGE_SYSTEM = (
    "You write short, cautious notes for a clinician about an automated image-analysis result. You are not "
    "diagnosing. Use hedged language ('the model suggests'). State the top finding and its probability, the "
    "abnormal probability, and whether human review is required. Do not add medical facts that are not in "
    "the input. Keep it under 90 words and end with the disclaimer exactly as given."
)
CARD_SYSTEM = (
    "You draft sections of a model card from aggregate validation metrics. Use only the numbers provided, say "
    "plainly where performance is weak (for example classes with low recall), and list limitations. Do not "
    "claim clinical validity. Keep it under 300 words."
)


def check_prediction(pred: dict) -> None:
    extra = set(pred) - ALLOWED_PREDICTION_KEYS
    if extra:
        raise ValueError(f"unexpected keys in prediction (possible identifiers): {sorted(extra)}")
    for t in pred.get("top", []):
        if set(t) - ALLOWED_TOP_KEYS:
            raise ValueError("unexpected keys in prediction['top']")


def check_metrics(metrics: dict) -> None:
    def walk(x):
        if isinstance(x, dict):
            bad = FORBIDDEN_IN_METRICS & {str(k).lower() for k in x}
            if bad:
                raise ValueError(f"metrics contain forbidden keys: {sorted(bad)}")
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(metrics)
    if len(json.dumps(metrics)) > 20000:
        raise ValueError("metrics payload too large")


def template_triage(pred: dict) -> str:
    top = pred["top"][0]
    review = "Human review is required." if pred.get("needs_review") else "Routine human review still applies."
    return (f"The model suggests '{top['label']}' (probability {top['probability']:.0%}); abnormal probability "
            f"{pred['abnormal_probability']:.0%}. {review} {DISCLAIMER}")


def _converse(system: str, prompt: str, model_id: str | None, region: str | None, max_tokens: int = 500) -> str:
    import boto3

    client = boto3.client("bedrock-runtime", region_name=region or os.environ.get("AWS_REGION", "eu-west-2"))
    kwargs = dict(modelId=model_id or os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"),
                  system=[{"text": system}], messages=[{"role": "user", "content": [{"text": prompt}]}],
                  inferenceConfig={"maxTokens": max_tokens, "temperature": 0.2})
    gid = os.environ.get("BEDROCK_GUARDRAIL_ID")
    if gid:
        kwargs["guardrailConfig"] = {"guardrailIdentifier": gid, "guardrailVersion": os.environ.get("BEDROCK_GUARDRAIL_VERSION", "DRAFT")}
    return client.converse(**kwargs)["output"]["message"]["content"][0]["text"]


def triage_note(pred: dict, dry_run: bool = False, model_id: str | None = None, region: str | None = None) -> str:
    check_prediction(pred)
    if dry_run:
        return template_triage(pred)
    text = _converse(TRIAGE_SYSTEM, "Result (JSON):\n" + json.dumps(pred), model_id, region, 250)
    return text if DISCLAIMER in text else text.rstrip() + "\n" + DISCLAIMER


def model_card_draft(metrics: dict, dry_run: bool = False, model_id: str | None = None, region: str | None = None) -> str:
    check_metrics(metrics)
    slim = {k: metrics[k] for k in ("n", "accuracy", "balanced_accuracy", "macro_f1", "mcc", "ece", "abnormal_vs_normal", "per_class") if k in metrics}
    if dry_run:
        a = slim.get("abnormal_vs_normal", {})
        return (f"Model card draft (template). Macro-F1 {slim.get('macro_f1', float('nan')):.2f}, balanced accuracy "
                f"{slim.get('balanced_accuracy', float('nan')):.2f}, abnormal recall {a.get('recall', float('nan')):.2f}. "
                f"Limitations: single public dataset, frame-level labels, class imbalance. {DISCLAIMER}")
    return _converse(CARD_SYSTEM, "Validation metrics (JSON):\n" + json.dumps(slim), model_id, region, 700)
