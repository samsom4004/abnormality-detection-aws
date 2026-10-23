import pytest

from abnormality_detection import bedrock_report as br

PRED = {"top": [{"label": "Polyp", "probability": 0.81}], "abnormal_probability": 0.9, "uncertainty": 0.2,
        "needs_review": True, "model_version": "0.1.0", "disclaimer": br.DISCLAIMER}


def test_dry_run_note_has_disclaimer_and_numbers():
    t = br.triage_note(PRED, dry_run=True)
    assert "Polyp" in t and "81%" in t and "Not a medical device" in t and "Human review is required" in t


def test_identifiers_are_refused():
    with pytest.raises(ValueError):
        br.triage_note({**PRED, "patient_id": "123"}, dry_run=True)
    with pytest.raises(ValueError):
        br.triage_note({**PRED, "top": [{"label": "x", "probability": 1, "filename": "a.png"}]}, dry_run=True)
    with pytest.raises(ValueError):
        br.model_card_draft({"per_class": {}, "patient": "x"}, dry_run=True)


def test_converse_call_and_disclaimer_is_enforced(monkeypatch):
    calls = {}

    class Stub:
        def converse(self, **kw):
            calls.update(kw)
            return {"output": {"message": {"content": [{"text": "The model suggests a polyp."}]}}}

    import boto3

    monkeypatch.setattr(boto3, "client", lambda *a, **k: Stub())
    monkeypatch.setenv("BEDROCK_GUARDRAIL_ID", "gr-1")
    note = br.triage_note(PRED, model_id="m-1")
    assert note.endswith(br.DISCLAIMER)              # added even if the model forgot it
    assert calls["guardrailConfig"]["guardrailIdentifier"] == "gr-1" and calls["inferenceConfig"]["temperature"] <= 0.3
    sent = calls["messages"][0]["content"][0]["text"].lower()
    assert "image_base64" not in sent and "filename" not in sent and len(sent) < 1500


def test_model_card_dry_run(bundle):

    m = {"n": 10, "macro_f1": 0.5, "balanced_accuracy": 0.6, "abnormal_vs_normal": {"recall": 0.9}}
    assert "Macro-F1 0.50" in br.model_card_draft(m, dry_run=True)
