# Model card (draft)

> Research prototype. Not a medical device. Complete and review before sharing outside the project.

* **Model:** VGG-16 convolutional base (ImageNet weights, frozen) + dropout + dense softmax over 14 classes.
  Options: ResNet-50, DenseNet-121.
* **Intended use:** research and teaching on automated triage of capsule-endoscopy frames.
* **Out of scope:** diagnosis, treatment decisions, any use without clinician review.
* **Data:** Kvasir-Capsule style images (14 labelled findings). Verify licence and consent terms.
* **Splits:** by video, with a held-out test set. Metrics must be re-measured on the real data.
* **Outputs:** top-3 classes with calibrated probabilities, abnormal probability, uncertainty, `needs_review`.
* **Metrics to report:** macro-F1, balanced accuracy, MCC, per-class recall and precision, confusion matrix,
  abnormal-vs-normal sensitivity/specificity/AUROC, expected calibration error.
* **Known limitations:** class imbalance; frame-level labels; one dataset and device family; no external
  validation; poor-view frames; normal-like grouping needs clinical review.
* **Ethical considerations:** automation bias, unequal performance across patient groups (not measured),
  privacy.
* **Maintenance:** re-evaluate on new data; monitor drift; version every model with its manifest hash.

_The `model-card` CLI command can draft the metrics sections with Amazon Bedrock from `metrics.json`._
