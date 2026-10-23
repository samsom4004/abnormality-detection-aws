"""Deploy a model.tar.gz as a SageMaker endpoint. Serverless by default (scales to zero, CPU).

    python scripts/deploy_endpoint.py --role-arn ... --model-data s3://bucket/.../model.tar.gz
    python scripts/deploy_endpoint.py ... --realtime ml.g4dn.xlarge      # always-on GPU (costs money until deleted)
"""

import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role-arn", required=True)
    p.add_argument("--model-data", required=True)
    p.add_argument("--endpoint-name", default="abnormality-detection")
    p.add_argument("--memory-mb", type=int, default=6144)
    p.add_argument("--max-concurrency", type=int, default=3)
    p.add_argument("--realtime", metavar="INSTANCE_TYPE")
    a = p.parse_args()

    from sagemaker.pytorch import PyTorchModel
    from sagemaker.serverless import ServerlessInferenceConfig

    model = PyTorchModel(model_data=a.model_data, role=a.role_arn, framework_version="2.1", py_version="py310",
                         entry_point="mlops/inference_sm.py", source_dir=os.path.join(ROOT, "build", "source"))
    if a.realtime:
        model.deploy(endpoint_name=a.endpoint_name, initial_instance_count=1, instance_type=a.realtime)
    else:
        model.deploy(endpoint_name=a.endpoint_name,
                     serverless_inference_config=ServerlessInferenceConfig(memory_size_in_mb=a.memory_mb, max_concurrency=a.max_concurrency))
    print("endpoint ready:", a.endpoint_name)


if __name__ == "__main__":
    main()
