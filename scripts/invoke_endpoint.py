"""Send one image to a deployed endpoint and print the result. Use test images you are allowed to use."""

import argparse
import json

import boto3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--endpoint-name", default="abnormality-detection")
    p.add_argument("--region", default="eu-west-2")
    a = p.parse_args()
    with open(a.image, "rb") as f:
        body = f.read()
    ctype = "image/png" if a.image.lower().endswith(".png") else "image/jpeg"
    rt = boto3.client("sagemaker-runtime", region_name=a.region)
    r = rt.invoke_endpoint(EndpointName=a.endpoint_name, ContentType=ctype, Body=body)
    print(json.dumps(json.loads(r["Body"].read()), indent=1))


if __name__ == "__main__":
    main()
