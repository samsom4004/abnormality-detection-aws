# From the thesis to this repository

Original: *A Deep Learning Approach for Accurate Classification of Abnormalities in Intestinal Tracts*
(University of Southampton MSc). Archived design code: VGG-16 training in Keras, class balancing, TFLite
conversion and a real-time test notebook.

| Thesis | Here |
| --- | --- |
| VGG-16, ImageNet base frozen, Flatten + Dense(14, softmax), Adam 1e-4, 10 epochs | `model.build_model("vgg16")`, `TrainConfig` defaults |
| 14 classes | `labels.CLASSES` (same order as the label encoder) |
| Keras `ImageDataGenerator` augmentation | torchvision transforms |
| Accuracy about 83% | macro-F1, balanced accuracy, MCC, per-class, abnormal-vs-normal, calibration |
| TFLite model and real-time notebook | ONNX export, FastAPI server |
| Colab + Google Drive | S3 + SageMaker Pipeline |

## Issues found in the archived design script, and the fix

| Issue (as read in the script) | Why it matters | Fix here |
| --- | --- | --- |
| Random train/test split of **frames** | consecutive frames are near-duplicates, so test accuracy is inflated | `grouped_split` by `video_id` |
| The test generator reuses the **augmenting** `ImageDataGenerator` | test images are randomly rotated, sheared and flipped, so results are noisy and not a clean test | separate `eval_transform` with no augmentation |
| `class_weight` is passed to `flow_from_dataframe`, which does not take it | the imbalance handling appears not to be applied | weights passed to the loss and covered by a test |
| "Weighted" and "regular" metrics computed from the same predictions; both loss variables read the same history key | the report looks like a comparison but is not | one honest set of metrics |
| Only accuracy reported as the headline | rare, clinically important classes can fail silently | macro-F1, per-class recall, abnormal recall |
| No calibration or uncertainty | probabilities are overconfident | temperature scaling + `needs_review` |

These come from reading the archived script (`intestine_abnormilities_image_balancing.py`). Your final
submitted code may differ; check it before relying on this list.

## What this repository does **not** claim

* It has not been re-trained on the real dataset here, so it makes **no accuracy claim**. Expect the honest
  numbers to be lower than 83%.
* It is not clinically validated and is not a medical device.
