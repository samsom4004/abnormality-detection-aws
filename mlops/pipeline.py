"""Amazon SageMaker Pipeline: train (GPU) -> quality gate -> register model for approval.

Requires ``pip install -e ".[aws]"`` (SageMaker Python SDK v2) and AWS credentials; run ``make package`` first.

    python mlops/pipeline.py --role-arn <role> --bucket <bucket> --data-uri s3://<bucket>/kvasir-capsule/ --start

``--data-uri`` must be an S3 prefix containing ``manifest.csv`` and the images. Download the public
Kvasir-Capsule dataset yourself (see docs/DATA_GOVERNANCE.md); no data is included here.
"""

from __future__ import annotations

import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(ROOT, "build", "source")


def build_pipeline(role: str, bucket: str, data_uri: str, name: str = "abnormality-detection",
                   package_group: str = "abnormality-detection-models"):
    from sagemaker.inputs import TrainingInput
    from sagemaker.pytorch import PyTorch, PyTorchModel
    from sagemaker.workflow.condition_step import ConditionStep
    from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
    from sagemaker.workflow.model_step import ModelStep
    from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
    from sagemaker.workflow.pipeline import Pipeline
    from sagemaker.workflow.pipeline_context import PipelineSession
    from sagemaker.workflow.steps import TrainingStep

    if not os.path.isdir(SOURCE_DIR):
        raise SystemExit("run `make package` first (creates build/source)")
    session = PipelineSession(default_bucket=bucket)
    arch = ParameterString("Arch", default_value="vgg16")
    epochs = ParameterInteger("Epochs", default_value=10)
    instance = ParameterString("TrainingInstance", default_value="ml.g4dn.xlarge")
    min_f1 = ParameterFloat("MinMacroF1", default_value=0.60)
    min_recall = ParameterFloat("MinAbnormalRecall", default_value=0.85)

    est = PyTorch(
        entry_point="mlops/train_sm.py", source_dir=SOURCE_DIR, role=role, framework_version="2.1", py_version="py310",
        instance_type=instance, instance_count=1, sagemaker_session=session, output_path=f"s3://{bucket}/{name}/models",
        hyperparameters={"arch": arch, "epochs": epochs},
        metric_definitions=[
            {"Name": "test:macro_f1", "Regex": "test_macro_f1=([0-9\\.]+)"},
            {"Name": "test:abnormal_recall", "Regex": "test_abnormal_recall=([0-9\\.]+)"},
        ],
    )
    step_train = TrainingStep(name="TrainClassifier",
                              step_args=est.fit(inputs={"training": TrainingInput(data_uri, content_type="application/x-image")}))

    model = PyTorchModel(model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts, role=role, framework_version="2.1",
                         py_version="py310", entry_point="mlops/inference_sm.py", source_dir=SOURCE_DIR, sagemaker_session=session)
    register_args = model.register(content_types=["image/jpeg", "image/png"], response_types=["application/json"],
                                   inference_instances=["ml.m5.xlarge"], transform_instances=["ml.m5.xlarge"],
                                   model_package_group_name=package_group, approval_status="PendingManualApproval")
    step_register = ModelStep(name="RegisterModel", step_args=register_args)

    gate = ConditionStep(
        name="QualityGate",
        conditions=[
            ConditionGreaterThanOrEqualTo(left=step_train.properties.FinalMetricDataList["test:macro_f1"].Value, right=min_f1),
            ConditionGreaterThanOrEqualTo(left=step_train.properties.FinalMetricDataList["test:abnormal_recall"].Value, right=min_recall),
        ],
        if_steps=[step_register], else_steps=[],
    )
    return Pipeline(name=name, parameters=[arch, epochs, instance, min_f1, min_recall], steps=[step_train, gate], sagemaker_session=session)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role-arn", required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--data-uri", required=True)
    p.add_argument("--start", action="store_true")
    a = p.parse_args()
    pipe = build_pipeline(a.role_arn, a.bucket, a.data_uri)
    pipe.upsert(role_arn=a.role_arn)
    print("pipeline upserted:", pipe.name)
    if a.start:
        print("execution started:", pipe.start().arn)


if __name__ == "__main__":
    main()
