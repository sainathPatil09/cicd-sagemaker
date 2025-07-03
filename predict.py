import boto3

# Set your endpoint name
endpoint_name = "sagemaker-scikit-learn-2025-07-02-10-43-05-651"
region = "ap-south-1"

# Input as plain CSV string
csv_input = "5.1,3.5,1.4,0.2"

# Initialize runtime client
runtime = boto3.client("sagemaker-runtime", region_name=region)

# Invoke the endpoint
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="text/csv",
    Body=csv_input
)

# Read and print prediction
print("🧠 Prediction:", response["Body"].read().decode())
