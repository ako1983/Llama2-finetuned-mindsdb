import os
import json
import time
import boto3
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Configuration ===
S3_BUCKET = "sagemaker-XXXXXXXXX"
S3_KEY = "synthetic_qa.jsonl"
OUTPUT_FILE = "clean_synthetic_qa.jsonl"
INPUT_FILE = "chunked_docs.json"


# === Upload to S3 with Metadata ===
def upload_to_s3():
    s3_client = boto3.client('s3')

    metadata = {
        'project': 'mindsdb-finetune',
        'dataset': 'synthetic-qa',
        'created_by': 'ako'
    }

    try:
        s3_client.upload_file(
            OUTPUT_FILE,
            S3_BUCKET,
            S3_KEY,
            ExtraArgs={'Metadata': metadata}
        )

        s3_console_url = (
            f"https://s3.console.aws.amazon.com/s3/object/{S3_BUCKET}?region=us-east-1&prefix={S3_KEY}"
        )
        print(f"✅ Uploaded to s3://{S3_BUCKET}/{S3_KEY}")
        print(f"🔗 View in S3 Console: {s3_console_url}")

    except Exception as e:
        print(f"❌ Failed to upload to S3: {e}")

# === Optional - Detect SageMaker Environment ===
def detect_environment():
    if "SAGEMAKER_PROJECT_ARN" in os.environ:
        print("✅ Running inside SageMaker Studio")
        return True
    else:
        print("✅ Running locally")
        return False

# === Run this directly in Jupyter or Python CLI ===
if __name__ == "__main__":
    detect_environment()
    process_chunks_parallel()
