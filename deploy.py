# deploy.py

import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker import Session

s3_model_path = "s3://models-sagemaker-iris/model/model.tar.gz"  # Replace with your S3 path
role = "arn:aws:iam::396608790362:role/service-role/AmazonSageMaker-ExecutionRole-20250621T181246"  # Replace this

sagemaker_session = Session()

model = SKLearnModel(
    model_data=s3_model_path,
    role=role,
    entry_point="inference/inference.py",
    framework_version="0.23-1",
    sagemaker_session=sagemaker_session
)

predictor = model.deploy(instance_type="ml.t2.medium", initial_instance_count=1)

print("Model deployed at endpoint:", predictor.endpoint_name)
