from pathlib import Path
from dotenv import load_dotenv
import boto3
import os

env_path = Path(__file__).resolve().parent.parent / ".env"
print(f"Loading .env from: {env_path}, exists: {env_path.exists()}")
load_dotenv(dotenv_path=env_path)

bucket = os.getenv("S3_BUCKET")
print(f"S3_BUCKET = {bucket!r}")

s3 = boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION"))
s3.put_object(Bucket=bucket, Key="raw-data/_smoke_test.txt", Body=b"hello from astro-ib-probe")
resp = s3.list_objects_v2(Bucket=bucket, Prefix="raw-data/")
print("Objects under raw-data/:", [obj["Key"] for obj in resp.get("Contents", [])])