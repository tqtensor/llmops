import argparse
import os
import tempfile
from pathlib import Path

from google.cloud import storage
from huggingface_hub import snapshot_download


def download_and_upload_model(bucket_name: str, model_id: str):
    """
    Downloads the specified model to a temporary local directory and uploads it
    to GCS bucket.

    Args:
        bucket_name: Name of the GCS bucket
        model_id: HuggingFace model ID (e.g. "google/flan-t5-base")
    """
    # Create a temp directory for download
    model_name = model_id.split("/")[-1] if "/" in model_id else model_id
    local_dir = os.path.join(tempfile.gettempdir(), model_name)
    os.makedirs(local_dir, exist_ok=True)
    print(f"Using local directory for download: {local_dir}")

    # Use a local temp directory for caching
    cache_dir = os.path.join(tempfile.gettempdir(), "hf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Using cache directory: {cache_dir}")

    try:
        # Download the model
        print(f"Downloading model {model_id} from HuggingFace Hub...")
        model_path = snapshot_download(
            repo_id=model_id,
            ignore_patterns=["*.md"],
            local_dir=local_dir,
            cache_dir=cache_dir,
        )
        print(f"Successfully downloaded model to {model_path}")

        # Upload the model to GCS
        gcs_path = f"hf/{model_id}"
        print(f"Uploading model to gs://{bucket_name}/{gcs_path}/...")

        client = storage.Client()
        bucket = client.get_bucket(bucket_name)

        # Upload all files in the model directory
        for local_file in Path(model_path).glob("**/*"):
            if local_file.is_file():
                # Get relative path from the model_path
                relative_path = local_file.relative_to(model_path)
                # Create destination blob path
                dest_path = f"{gcs_path}/{relative_path}"

                blob = bucket.blob(dest_path)
                blob.upload_from_filename(str(local_file))
                print(f"Uploaded {local_file} to gs://{bucket_name}/{dest_path}")

        print(f"Model successfully uploaded to gs://{bucket_name}/{gcs_path}/")
        return model_path

    except Exception as error:
        print(f"Error: {error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download model and upload to GCS bucket"
    )
    parser.add_argument("bucket_name", type=str, help="GCS bucket name")
    parser.add_argument(
        "model_id", type=str, help="HuggingFace model ID (e.g., 'google/flan-t5-base')"
    )
    args = parser.parse_args()

    download_and_upload_model(args.bucket_name, args.model_id)
