#!/usr/bin/env python3
"""
Attempt to set up Google Cloud Storage using the Google Places API key.

Note: GCS typically requires service account credentials, but we'll try using
the API key via REST API for bucket creation and management.
"""

import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")  # We'll need to extract this or ask user


def get_project_id_from_api_key():
    """
    Try to get the project ID from the API key by making a test API call.
    This is a workaround - we can't directly get project ID from API key.
    """
    # Unfortunately, API keys don't directly reveal project ID
    # We'll need to ask the user or try to infer it
    return None


def create_bucket_via_rest_api(api_key: str, bucket_name: str, project_id: str, location: str = "us-central1"):
    """
    Attempt to create a GCS bucket using REST API with API key.
    
    Note: This typically requires OAuth2 or service account, but let's try.
    """
    url = f"https://storage.googleapis.com/storage/v1/b"
    
    params = {
        "project": project_id,
        "key": api_key
    }
    
    payload = {
        "name": bucket_name,
        "location": location,
        "storageClass": "STANDARD"
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, params=params, headers=headers)
        
        if response.status_code == 200:
            print(f"✅ Successfully created bucket: {bucket_name}")
            return True
        elif response.status_code == 409:
            print(f"ℹ️  Bucket {bucket_name} already exists")
            return True
        else:
            print(f"❌ Failed to create bucket: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error creating bucket: {str(e)}")
        return False


def setup_gcs_with_api_key():
    """
    Main setup function - attempts to use API key for GCS setup.
    """
    print("=" * 60)
    print("Google Cloud Storage Setup with API Key")
    print("=" * 60)
    print()
    
    if not GOOGLE_PLACES_API_KEY:
        print("❌ GOOGLE_PLACES_API_KEY not found in .env file")
        return False
    
    print(f"✓ Found API key: {GOOGLE_PLACES_API_KEY[:10]}...")
    
    # Ask for project ID and bucket name
    print("\nTo set up GCS, I need some information:")
    print("(Note: API keys typically don't have permission to create buckets)")
    print()
    
    project_id = input("Enter your Google Cloud Project ID (or press Enter to skip): ").strip()
    if not project_id:
        print("\n⚠️  Cannot proceed without Project ID")
        print("   API keys don't have permission to create buckets.")
        print("   Please follow the manual setup in GCS_SETUP.md")
        return False
    
    bucket_name = input("Enter desired bucket name (e.g., restaurant-recs-images): ").strip()
    if not bucket_name:
        print("❌ Bucket name is required")
        return False
    
    # Validate bucket name
    if not (3 <= len(bucket_name) <= 63):
        print("❌ Bucket name must be 3-63 characters")
        return False
    
    location = input("Enter bucket location (default: us-central1): ").strip() or "us-central1"
    
    print(f"\nAttempting to create bucket '{bucket_name}' in project '{project_id}'...")
    print("(This may fail if API key doesn't have sufficient permissions)")
    print()
    
    success = create_bucket_via_rest_api(GOOGLE_PLACES_API_KEY, bucket_name, project_id, location)
    
    if success:
        print("\n✅ Bucket setup complete!")
        print(f"\nAdd to your .env file:")
        print(f"GCS_BUCKET_NAME={bucket_name}")
        print(f"GOOGLE_CLOUD_PROJECT_ID={project_id}")
        print("\n⚠️  Note: You may still need service account credentials for file uploads.")
        print("   The API key might work for some operations but not all.")
        return True
    else:
        print("\n❌ Automatic setup failed.")
        print("   This is expected - API keys typically don't have permission to create buckets.")
        print("   Please follow the manual setup in GCS_SETUP.md")
        return False


if __name__ == "__main__":
    try:
        success = setup_gcs_with_api_key()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

