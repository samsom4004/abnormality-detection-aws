# Machine-learning strategy

## 1. Problem framing

Wireless capsule endoscopy produces tens of thousands of frames per patient; clinicians must find the few
that show bleeding, ulcers, polyps and so on. The goal is **decision support**: rank and flag frames so a
clinician looks at the right ones sooner, never to diagnose automatically.

The cost of errors is asymmetric: a missed abnormal frame (false negative) matters more than an extra frame
to review. So the primary operating metric is **abnormal-vs-normal recall at an acceptable review burden**,
with per-class recall tracked for the dangerous classes (blood, ulcer, polyp, angiectasia).

## 2. Metrics and targets (initial; agree with clinicians)

| Metric | Why | Starting target |
| --- | --- | --- |
| Abnormal recall (sensitivity) | missed findings are the main risk | >= 0.85 |
| Specificity / review rate | clinician time | report the trade-off curve |
| Macro-F1, balanced accuracy, MCC | accuracy hides rare-class failure | macro-F1 >= 0.60 to register |
| Per-class recall | no class silently ignored | report all 14 |
| Expected calibration error | probabilities must mean something | < 0.10 |
| Latency p95 | workflow fit | < 2 s per image on CPU |

The 83% accuracy in the thesis is **not** comparable to these until it is re-measured with video-level splits.

## 3. Data strategy

* Public Kvasir-Capsule dataset (47,238 labelled frames, 14 classes, from 117 videos), verify the licence.
* **Split by video**; keep a never-touched final test set; record dataset and manifest hashes per run.
* Expect strong class imbalance: use class-weighted loss, report macro metrics, consider focal loss or
  re-sampling only after the baseline is honest.
* External validation on a different centre or device is required before any claim of generalisation.

## 4. Modelling roadmap

| Step | Change | Success check |
| --- | --- | --- |
| 0 | Re-run the thesis baseline with leakage fixed | honest baseline numbers |
| 1 | Unfreeze the last VGG block; try ResNet-50 / DenseNet-121 | macro-F1 up, abnormal recall maintained |
| 2 | Better augmentation, focal loss | rare-class recall up |
| 3 | Temperature scaling (done) and threshold selection by cost | calibrated, chosen operating point |
| 4 | One-class model on normal mucosa as a second opinion | compare AUROC |
| 5 | Explainability (Grad-CAM) for clinician review | reviewed by clinicians |

## 5. MLOps maturity

| Stage | Status |
| --- | --- |
| Tested package + CLI | done (37 tests, CPU, synthetic data) |
| Reproducible training in SageMaker | written, not yet run on AWS |
| Quality gate + manual approval | implemented in the pipeline |
| Monitoring | Model Monitor and drift plan documented; to implement |
| Model card + risk file | draft in MODEL_CARD.md |
| Clinical validation | not started |

## 6. Where Bedrock helps and where it must not

* **Helps:** cautious triage notes for the reviewing clinician, model-card drafts from aggregate metrics.
* **Must not:** see images, identifiers or filenames; change or override a classification; approve a model.
* Controls: input allow-list, low temperature, Bedrock Guardrail (optional), disclaimer always appended,
  human sign-off.

## 7. Risks

| Risk | Mitigation |
| --- | --- |
| Automation bias (clinician over-trusts the model) | `needs_review`, disclaimers, show probabilities, train users |
| Dataset shift (new capsule hardware, preparation, populations) | external validation, drift monitoring |
| Label noise | clinician re-labelling of a sample, report uncertainty |
| Regulatory | treat as a medical device in design: MHRA, clinical safety case (DCB0129 / DCB0160 for NHS use) |
| Privacy | no identifiers in logs or prompts; private bucket; region eu-west-2 |
| Cost | serverless endpoint, budget alert, delete script |
