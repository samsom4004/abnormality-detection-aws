# Deploying on AWS

> Written against the AWS and SageMaker Python SDK v2 documentation but **not yet run on a live account**.
> Use a sandbox account and keep the budget alert on for the first run.

## Prerequisites

AWS CLI configured, Terraform >= 1.5, Python 3.10+, SageMaker quota for `ml.g4dn.xlarge` training (request an
increase if it is 0), `pip install -e ".[aws]"`.

## 1. Infrastructure

```bash
cd infra/terraform
terraform init && terraform apply -var-file=example.tfvars    # set budget_email first
terraform output
```

## 2. Data

Download Kvasir-Capsule, create `manifest.csv` (`filename,label,video_id`; labels must match `labels.py`), then:

```bash
aws s3 sync ./kvasir-capsule s3://$BUCKET/kvasir-capsule/ --sse AES256
```

## 3. Train and register

```bash
make package
python mlops/pipeline.py --role-arn "$ROLE" --bucket "$BUCKET" --data-uri s3://$BUCKET/kvasir-capsule/ --start
```

Approve the model in the Model Registry only after reviewing metrics with a clinician.

## 4. Deploy and test

```bash
python scripts/deploy_endpoint.py --role-arn "$ROLE" --model-data s3://.../model.tar.gz     # serverless CPU
python scripts/invoke_endpoint.py path/to/test_image.png
python scripts/delete_endpoint.py
```

The VGG-16 bundle is about 60 MB. The first serverless call has a cold start (a few seconds). If latency
matters, use `--realtime ml.g4dn.xlarge` (always on; delete it afterwards).

## 5. Bedrock

```bash
export AWS_REGION=eu-west-2 BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
abnormality-detection predict --model artifacts/model --image img.png --note
abnormality-detection model-card --model artifacts/model
```

Enable model access in the Bedrock console first. Optionally set `BEDROCK_GUARDRAIL_ID`.

## 6. Monitoring

CloudWatch alarms on 5xx rate and p95 latency; Model Monitor baseline on the validation set (image statistics
and prediction-distribution drift); weekly review of the share of `needs_review` cases.

## Notes

* The training container installs `scikit-learn`, `pandas` and `pillow` from `requirements.txt` at start-up,
  so the job needs internet access (or a custom image if you run in an isolated VPC).
* ImageNet weights are downloaded by torchvision at start-up; pre-bake them into the source dir for isolated VPCs.
* ONNX export of VGG-16 requires `image_size` 224 (or a multiple).

## Rough cost (verify on the AWS pricing pages)

| Item | Typical |
| --- | --- |
| Training, `ml.g4dn.xlarge`, under an hour | a few USD at most |
| Serverless endpoint | pay per use, scales to zero |
| Real-time GPU endpoint | the main risk if forgotten: delete it |
| S3, Bedrock | small |
