import os
from typing import Optional, Tuple
from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError
from dotenv import load_dotenv
from pathlib import Path

# Load .env file from project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")


class StorageServiceError(Exception):
    """Custom exception for storage service errors"""
    pass


def get_storage_client() -> storage.Client:
    """
    Get Google Cloud Storage client.
    
    Returns:
        storage.Client instance
        
    Raises:
        StorageServiceError: If client cannot be created
    """
    try:
        if GOOGLE_APPLICATION_CREDENTIALS:
            client = storage.Client.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)
        else:
            # Use default credentials (e.g., from environment or gcloud auth)
            client = storage.Client()
        return client
    except Exception as e:
        raise StorageServiceError(f"Failed to create GCS client: {str(e)}")


def upload_image_to_gcs(
    image_bytes: bytes,
    place_id: str,
    photo_name: str,
    content_type: str = "image/jpeg"
) -> Tuple[str, str]:
    """
    Upload image to Google Cloud Storage.
    
    Args:
        image_bytes: Image data as bytes
        place_id: Google Places place_id (used for organizing in bucket)
        photo_name: Google Places photo name/path (used as filename identifier)
        content_type: MIME type of the image (default: image/jpeg)
        
    Returns:
        Tuple of (public_url, bucket_path)
        
    Raises:
        StorageServiceError: If upload fails
    """
    if not GCS_BUCKET_NAME:
        raise StorageServiceError("GCS_BUCKET_NAME environment variable is not set")
    
    try:
        client = get_storage_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        
        # Create a safe filename from photo_name
        # photo_name is like "places/ChIJ.../photos/Aap_uEA..."
        # Extract just the photo ID part for the filename
        photo_id = photo_name.split("/")[-1] if "/" in photo_name else photo_name
        
        # Organize by place_id: images/{place_id}/{photo_id}.jpg
        bucket_path = f"images/{place_id}/{photo_id}.jpg"
        
        # Upload the image
        blob = bucket.blob(bucket_path)
        blob.upload_from_string(image_bytes, content_type=content_type)
        
        # For buckets with uniform bucket-level access, we can't use make_public()
        # Instead, construct the public URL directly (bucket must be configured for public access)
        try:
            blob.make_public()
            public_url = blob.public_url
        except Exception:
            # Bucket has uniform access - construct URL directly
            public_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{bucket_path}"
        
        return public_url, bucket_path
        
    except GoogleCloudError as e:
        raise StorageServiceError(f"GCS upload error: {str(e)}")
    except Exception as e:
        raise StorageServiceError(f"Failed to upload image to GCS: {str(e)}")


def get_public_url(bucket_path: str) -> str:
    """
    Get public URL for an image given its bucket path.
    
    Args:
        bucket_path: Path to the image in the bucket
        
    Returns:
        Public URL string
    """
    if not GCS_BUCKET_NAME:
        raise StorageServiceError("GCS_BUCKET_NAME environment variable is not set")
    
    # For buckets with uniform bucket-level access, just construct the URL directly
    return f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{bucket_path}"


def delete_image_from_gcs(bucket_path: str) -> None:
    """
    Delete an image from Google Cloud Storage.
    
    Args:
        bucket_path: Path to the image in the bucket
        
    Raises:
        StorageServiceError: If deletion fails
    """
    if not GCS_BUCKET_NAME:
        raise StorageServiceError("GCS_BUCKET_NAME environment variable is not set")
    
    try:
        client = get_storage_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(bucket_path)
        blob.delete()
    except GoogleCloudError as e:
        if "No such object" in str(e):
            # Image doesn't exist, that's okay
            return
        raise StorageServiceError(f"GCS delete error: {str(e)}")
    except Exception as e:
        raise StorageServiceError(f"Failed to delete image from GCS: {str(e)}")

