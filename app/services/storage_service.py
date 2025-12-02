import os
import hashlib
import base64
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


def compute_content_hash(image_bytes: bytes) -> str:
    """
    Compute MD5 hash of image content for duplicate detection.
    Returns base64-encoded hash (same format as GCS md5Hash).
    """
    md5 = hashlib.md5(image_bytes)
    return base64.b64encode(md5.digest()).decode('utf-8')


def find_existing_image_by_content(
    bucket: storage.Bucket,
    place_id: str,
    content_hash: str,
    content_size: int
) -> Optional[str]:
    """
    Check if an image with the same content already exists in GCS.
    
    Args:
        bucket: GCS bucket object
        place_id: Google Places place_id to search within
        content_hash: Base64-encoded MD5 hash of the image content
        content_size: Size of the image in bytes
        
    Returns:
        bucket_path of existing image if found, None otherwise
    """
    prefix = f"images/{place_id}/"
    
    try:
        for blob in bucket.list_blobs(prefix=prefix):
            # Compare both hash and size for accurate matching
            if blob.md5_hash == content_hash and blob.size == content_size:
                return blob.name
    except Exception:
        # If listing fails, proceed with upload
        pass
    
    return None


def upload_image_to_gcs(
    image_bytes: bytes,
    place_id: str,
    photo_name: str,
    content_type: str = "image/jpeg"
) -> Tuple[str, str]:
    """
    Upload image to Google Cloud Storage with content-hash duplicate prevention.
    
    If an image with identical content already exists for this place_id,
    returns the existing image's URL instead of uploading a duplicate.
    
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
        
        # Compute content hash for duplicate detection
        content_hash = compute_content_hash(image_bytes)
        content_size = len(image_bytes)
        
        # Check if identical image already exists
        existing_path = find_existing_image_by_content(
            bucket, place_id, content_hash, content_size
        )
        
        if existing_path:
            # Duplicate found - return existing image's URL
            public_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{existing_path}"
            print(f"[GCS] ♻️  Duplicate detected, reusing existing: {existing_path[-40:]}...")
            return public_url, existing_path
        
        # No duplicate - proceed with upload
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
        
        print(f"[GCS] ✅ Uploaded new image: {bucket_path[-40:]}...")
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


def stream_upload_to_gcs(
    image_url: str,
    place_id: str,
    photo_name: str,
    content_type: str = "image/jpeg",
    timeout: int = 30
) -> Tuple[str, str]:
    """
    Download image from URL and upload to GCS in a single operation.
    
    This is more efficient than separate download/upload as it:
    - Uses a single function call (better for parallelization)
    - Still buffers in memory (GCS requires content-length)
    
    Args:
        image_url: URL to download image from
        place_id: Google Places place_id (used for organizing in bucket)
        photo_name: Unique identifier for the photo
        content_type: MIME type of the image (default: image/jpeg)
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (public_url, bucket_path)
        
    Raises:
        StorageServiceError: If download or upload fails
    """
    import requests
    
    if not GCS_BUCKET_NAME:
        raise StorageServiceError("GCS_BUCKET_NAME environment variable is not set")
    
    try:
        # Download image
        response = requests.get(image_url, timeout=timeout)
        response.raise_for_status()
        image_bytes = response.content
        
        # Use existing upload function (handles deduplication)
        return upload_image_to_gcs(image_bytes, place_id, photo_name, content_type)
        
    except requests.RequestException as e:
        raise StorageServiceError(f"Failed to download image: {str(e)}")
    except Exception as e:
        raise StorageServiceError(f"Failed to stream upload: {str(e)}")

