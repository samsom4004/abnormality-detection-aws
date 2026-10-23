# Data governance

* **Dataset.** The thesis used the public Kvasir-Capsule dataset (Simula, 2021; 47,238 labelled frames from 117
  capsule videos, 14 classes). No images are included in this repository. Download them from the official
  page and **check the current licence and citation requirements** (it has been published under a Creative
  Commons licence).
* **Medical context.** Even de-identified medical images deserve care. Do not mix them with any identifiable
  patient data. If you ever use hospital data you need ethics approval, a data-sharing agreement, and a Data
  Protection Impact Assessment (UK GDPR).
* **Region and access.** Default `eu-west-2` (London). Private S3 bucket with public access blocked, SSE
  encryption, least-privilege IAM role.
* **Artefacts.** `.gitignore` excludes images, CSVs, weights and archives. Model weights trained on a dataset
  inherit that dataset's licence.
* **Logging.** The API processes images in memory and never writes or logs them.
* **Language models.** Bedrock receives only labels, probabilities and aggregate metrics. Forbidden keys
  (filenames, ids, base64 images) are rejected before any call.
* **Regulation.** Software that informs diagnosis is likely a medical device in the UK. This repository is a
  research prototype and says so in every output.
