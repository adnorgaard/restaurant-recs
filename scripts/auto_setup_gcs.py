#!/usr/bin/env python3
"""
Automated GCS setup script that tries multiple approaches:
1. Use API key if it has sufficient permissions
2. Use gcloud CLI if available
3. Provide instructions for manual setup
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")


def check_gcloud_cli():
    """Check if gcloud CLI is installed and configured."""
    try:
        result = subprocess.run(
            ["gcloud", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✓ gcloud CLI is installed")
            
            # Check if authenticated
            auth_result = subprocess.run(
                ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if auth_result.returncode == 0 and auth_result.stdout.strip():
                print(f"✓ Authenticated as: {auth_result.stdout.strip()}")
                return True
            else:
                print("⚠️  Not authenticated with gcloud. Run: gcloud auth login")
                return False
        return False
    except FileNotFoundError:
        print("ℹ️  gcloud CLI not found")
        return False
    except Exception as e:
        print(f"⚠️  Error checking gcloud: {str(e)}")
        return False


def get_current_project():
    """Get current gcloud project."""
    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            project_id = result.stdout.strip()
            if project_id:
                return project_id
        return None
    except Exception:
        return None


def create_bucket_with_gcloud(bucket_name: str, location: str = "us-central1"):
    """Create bucket using gcloud CLI."""
    try:
        print(f"\nCreating bucket '{bucket_name}' with gcloud...")
        
        cmd = [
            "gcloud", "storage", "buckets", "create",
            f"gs://{bucket_name}",
            f"--location={location}",
            "--uniform-bucket-level-access"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"✅ Successfully created bucket: {bucket_name}")
            return True
        elif "already exists" in result.stderr.lower():
            print(f"ℹ️  Bucket {bucket_name} already exists")
            return True
        else:
            print(f"❌ Failed to create bucket: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def create_service_account_with_gcloud(service_account_name: str, project_id: str):
    """Create service account using gcloud CLI."""
    try:
        print(f"\nCreating service account '{service_account_name}'...")
        
        # Create service account
        cmd = [
            "gcloud", "iam", "service-accounts", "create",
            service_account_name,
            f"--project={project_id}",
            "--display-name=Restaurant Recs Storage Service Account"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0 and "already exists" not in result.stderr.lower():
            print(f"⚠️  Could not create service account: {result.stderr}")
            return None
        
        # Grant Storage Admin role
        email = f"{service_account_name}@{project_id}.iam.gserviceaccount.com"
        cmd = [
            "gcloud", "projects", "add-iam-policy-binding", project_id,
            f"--member=serviceAccount:{email}",
            "--role=roles/storage.admin"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"⚠️  Could not grant permissions: {result.stderr}")
        
        # Create and download key
        key_file = project_root / f"{service_account_name}-key.json"
        cmd = [
            "gcloud", "iam", "service-accounts", "keys", "create",
            str(key_file),
            f"--iam-account={email}",
            f"--project={project_id}"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and key_file.exists():
            print(f"✅ Created service account key: {key_file}")
            return str(key_file)
        else:
            print(f"⚠️  Could not create key file: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error creating service account: {str(e)}")
        return None


def auto_setup():
    """Main auto-setup function."""
    print("=" * 60)
    print("Automated Google Cloud Storage Setup")
    print("=" * 60)
    print()
    
    # Check for API key
    if not GOOGLE_PLACES_API_KEY:
        print("❌ GOOGLE_PLACES_API_KEY not found in .env")
        print("   Please add it to your .env file first")
        return False
    
    print(f"✓ Found API key: {GOOGLE_PLACES_API_KEY[:10]}...")
    
    # Try gcloud CLI approach (most reliable)
    if check_gcloud_cli():
        project_id = get_current_project()
        
        if not project_id:
            project_id = input("\nEnter your Google Cloud Project ID: ").strip()
            if project_id:
                subprocess.run(["gcloud", "config", "set", "project", project_id], 
                             capture_output=True)
        
        if project_id:
            print(f"✓ Using project: {project_id}")
            
            bucket_name = input("\nEnter bucket name (e.g., restaurant-recs-images): ").strip()
            if not bucket_name:
                print("❌ Bucket name required")
                return False
            
            location = input("Enter location (default: us-central1): ").strip() or "us-central1"
            
            # Create bucket
            if create_bucket_with_gcloud(bucket_name, location):
                # Try to create service account
                service_account_name = f"restaurant-recs-storage"
                key_file = create_service_account_with_gcloud(service_account_name, project_id)
                
                # Update .env file
                env_file = project_root / ".env"
                env_content = env_file.read_text() if env_file.exists() else ""
                
                updates = []
                if f"GCS_BUCKET_NAME=" not in env_content:
                    updates.append(f"GCS_BUCKET_NAME={bucket_name}")
                
                if key_file and f"GOOGLE_APPLICATION_CREDENTIALS=" not in env_content:
                    updates.append(f"GOOGLE_APPLICATION_CREDENTIALS={key_file}")
                
                if updates:
                    print(f"\n📝 Add these to your .env file:")
                    for update in updates:
                        print(f"   {update}")
                    
                    append = input("\nAppend to .env file automatically? (y/n): ").strip().lower()
                    if append == 'y':
                        with open(env_file, 'a') as f:
                            f.write("\n" + "\n".join(updates) + "\n")
                        print("✅ Updated .env file")
                
                print("\n✅ Setup complete!")
                return True
    
    # Fallback: manual instructions
    print("\n" + "=" * 60)
    print("Automatic setup not available")
    print("=" * 60)
    print("\nPlease follow the manual setup instructions in GCS_SETUP.md")
    print("\nOr install gcloud CLI and run this script again:")
    print("  https://cloud.google.com/sdk/docs/install")
    return False


if __name__ == "__main__":
    try:
        success = auto_setup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

