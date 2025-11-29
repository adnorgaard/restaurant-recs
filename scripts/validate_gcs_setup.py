#!/usr/bin/env python3
"""
Validation script for Google Cloud Storage setup.

This script checks:
1. Environment variables are set
2. GCS credentials are valid
3. Bucket exists and is accessible
4. Service account has proper permissions
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load .env file
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)


def check_env_variables():
    """Check if required environment variables are set."""
    print("Checking environment variables...")
    
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    issues = []
    
    if not bucket_name:
        issues.append("❌ GCS_BUCKET_NAME is not set in .env file")
    else:
        print(f"  ✓ GCS_BUCKET_NAME: {bucket_name}")
    
    if not creds_path:
        issues.append("❌ GOOGLE_APPLICATION_CREDENTIALS is not set in .env file")
    else:
        print(f"  ✓ GOOGLE_APPLICATION_CREDENTIALS: {creds_path}")
        
        # Check if file exists
        if not os.path.exists(creds_path):
            issues.append(f"❌ Credentials file not found: {creds_path}")
        else:
            print(f"  ✓ Credentials file exists")
            
            # Check if it's a valid JSON
            try:
                import json
                with open(creds_path, 'r') as f:
                    json.load(f)
                print(f"  ✓ Credentials file is valid JSON")
            except json.JSONDecodeError:
                issues.append(f"❌ Credentials file is not valid JSON: {creds_path}")
    
    return issues, bucket_name, creds_path


def check_gcs_connection(bucket_name):
    """Check if we can connect to GCS and access the bucket."""
    print("\nChecking GCS connection...")
    
    issues = []
    
    try:
        from app.services.storage_service import get_storage_client, StorageServiceError
        
        # Test client creation
        try:
            client = get_storage_client()
            print("  ✓ GCS client created successfully")
        except Exception as e:
            issues.append(f"❌ Failed to create GCS client: {str(e)}")
            return issues
        
        # Test bucket access
        try:
            from google.cloud import storage
            bucket = client.bucket(bucket_name)
            
            # Try to get bucket metadata
            bucket.reload()
            print(f"  ✓ Bucket '{bucket_name}' exists and is accessible")
            print(f"    Location: {bucket.location}")
            print(f"    Storage class: {bucket.storage_class}")
            
        except Exception as e:
            if "not found" in str(e).lower():
                issues.append(f"❌ Bucket '{bucket_name}' not found. Check the bucket name.")
            elif "permission" in str(e).lower() or "denied" in str(e).lower():
                issues.append(f"❌ Permission denied accessing bucket '{bucket_name}'. Check service account permissions.")
            else:
                issues.append(f"❌ Error accessing bucket: {str(e)}")
            return issues
        
        # Test write permission (create a test blob, then delete it)
        try:
            test_blob_name = "__test_write_permission__"
            test_blob = bucket.blob(test_blob_name)
            test_blob.upload_from_string(b"test", content_type="text/plain")
            test_blob.delete()
            print("  ✓ Write permission verified")
        except Exception as e:
            issues.append(f"❌ Write permission test failed: {str(e)}")
            print(f"    Warning: Cannot write to bucket. Check service account has 'Storage Admin' or 'Storage Object Admin' role.")
        
        # Test public access (optional check)
        try:
            # This is just informational
            bucket_iam = bucket.get_iam_policy()
            all_users = None
            for binding in bucket_iam.bindings:
                if 'allUsers' in binding.get('members', []):
                    all_users = binding
                    break
            
            if all_users:
                print(f"  ✓ Bucket has public access configured")
            else:
                print(f"  ⚠ Bucket is not publicly accessible (this is OK - objects will be made public individually)")
        except Exception as e:
            print(f"  ⚠ Could not check public access settings: {str(e)}")
        
    except ImportError as e:
        issues.append(f"❌ Failed to import required modules: {str(e)}")
        issues.append("   Make sure you've installed: pip install google-cloud-storage")
    except Exception as e:
        issues.append(f"❌ Unexpected error: {str(e)}")
    
    return issues


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("Google Cloud Storage Setup Validation")
    print("=" * 60)
    print()
    
    # Check environment variables
    env_issues, bucket_name, creds_path = check_env_variables()
    
    if env_issues:
        print("\n❌ Environment variable issues found:")
        for issue in env_issues:
            print(f"  {issue}")
        print("\nPlease fix these issues and run the script again.")
        return 1
    
    # Check GCS connection
    if bucket_name:
        gcs_issues = check_gcs_connection(bucket_name)
        
        if gcs_issues:
            print("\n❌ GCS connection issues found:")
            for issue in gcs_issues:
                print(f"  {issue}")
            print("\nPlease refer to GCS_SETUP.md for troubleshooting.")
            return 1
    
    print("\n" + "=" * 60)
    print("✅ All checks passed! Your GCS setup is configured correctly.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

