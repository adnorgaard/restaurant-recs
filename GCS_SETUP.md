# Google Cloud Storage Setup Guide

This guide will walk you through setting up Google Cloud Storage for the restaurant image caching system.

## Quick Start (Automated Setup)

If you have `gcloud` CLI installed, try the automated setup:

```bash
python scripts/auto_setup_gcs.py
```

This will attempt to:
- Create the GCS bucket
- Create a service account with proper permissions
- Generate and download the service account key
- Update your `.env` file automatically

**Note**: The Google Places API key alone typically doesn't have permission to create buckets. The automated script uses `gcloud` CLI which requires you to be authenticated with your Google account.

## Prerequisites

- A Google Cloud account
- A Google Cloud project (or create a new one)
- (Optional) `gcloud` CLI installed for automated setup

## Manual Step-by-Step Setup

### 1. Create a Google Cloud Project (if you don't have one)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click on the project dropdown at the top
3. Click "New Project"
4. Enter a project name (e.g., "restaurant-recs")
5. Click "Create"

### 2. Enable Google Cloud Storage API

1. In the Google Cloud Console, go to "APIs & Services" > "Library"
2. Search for "Cloud Storage API"
3. Click on it and click "Enable"

### 3. Create a Storage Bucket

1. Go to "Cloud Storage" > "Buckets" in the left sidebar
2. Click "Create Bucket"
3. Configure the bucket:
   - **Name**: Choose a unique name (e.g., `restaurant-recs-images` or `your-project-name-images`)
     - Note: Bucket names must be globally unique across all GCS buckets
     - Use lowercase letters, numbers, and hyphens only
   - **Location type**: Choose "Region" (cheaper) or "Multi-region" (more resilient)
   - **Region**: Choose a region close to you (e.g., `us-central1`, `us-east1`, `europe-west1`)
   - **Storage class**: "Standard" (default)
   - **Access control**: "Uniform" (recommended for simplicity)
   - **Public access**: You can leave this as default - we'll make individual objects public
4. Click "Create"

### 4. Create a Service Account

1. Go to "IAM & Admin" > "Service Accounts"
2. Click "Create Service Account"
3. Enter details:
   - **Service account name**: `restaurant-recs-storage` (or any name you prefer)
   - **Service account ID**: Will auto-populate
   - **Description**: "Service account for restaurant image storage"
4. Click "Create and Continue"
5. Grant role: Select "Storage Admin" (or "Storage Object Admin" for more restricted access)
6. Click "Continue"
7. Skip optional step and click "Done"

### 5. Create and Download Service Account Key

1. Find the service account you just created in the list
2. Click on it to open details
3. Go to the "Keys" tab
4. Click "Add Key" > "Create new key"
5. Choose "JSON" format
6. Click "Create"
7. The JSON key file will download automatically
8. **IMPORTANT**: Save this file securely! You'll need it for the `GOOGLE_APPLICATION_CREDENTIALS` environment variable

### 6. Make the Bucket Public (Optional but Recommended)

If you want images to be publicly accessible (which is typical for restaurant images):

1. Go to "Cloud Storage" > "Buckets"
2. Click on your bucket name
3. Go to the "Permissions" tab
4. Click "Grant Access"
5. Add principal: `allUsers`
6. Select role: "Storage Object Viewer"
7. Click "Save"
8. You'll see a warning - click "Allow public access"

**Note**: Alternatively, you can make objects public individually when uploading (which our code does automatically).

### 7. Configure Your .env File

Add these variables to your `.env` file:

```bash
GCS_BUCKET_NAME=your-bucket-name-here
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/your-service-account-key.json
```

**Important Notes:**
- Use the exact bucket name you created (e.g., `restaurant-recs-images`)
- Use the absolute path to your JSON key file (e.g., `/Users/adamnorgaard/restaurant-recs/gcs-key.json`)
- On Windows, use forward slashes or double backslashes: `C:/path/to/key.json` or `C:\\path\\to\\key.json`

## Verification

Run the validation script to verify your setup:

```bash
python scripts/validate_gcs_setup.py
```

Or use the Python script directly:

```python
python -c "from app.services.storage_service import get_storage_client; client = get_storage_client(); print('✓ GCS connection successful!')"
```

## Troubleshooting

### "Bucket not found" error
- Verify the bucket name in `.env` matches exactly (case-sensitive)
- Make sure you're using the correct Google Cloud project

### "Permission denied" error
- Verify the service account has "Storage Admin" or "Storage Object Admin" role
- Check that the JSON key file path is correct and the file exists

### "Invalid credentials" error
- Verify the JSON key file is valid (not corrupted)
- Make sure `GOOGLE_APPLICATION_CREDENTIALS` points to the correct file path
- Try regenerating the service account key

### Images not publicly accessible
- Make sure you've granted "Storage Object Viewer" to `allUsers` on the bucket
- Or verify that `blob.make_public()` is being called (which our code does)

## Cost Considerations

Google Cloud Storage pricing (as of 2024):
- **Storage**: ~$0.020 per GB/month (Standard storage)
- **Operations**: 
  - Class A (uploads): $0.05 per 10,000 operations
  - Class B (downloads): $0.004 per 10,000 operations
- **Network egress**: First 1 GB/month free, then ~$0.12 per GB

For a typical restaurant image (800px width, JPEG): ~100-500 KB
- 1,000 restaurants × 10 images = ~1-5 GB storage
- Monthly cost: ~$0.02-0.10 for storage + minimal operation costs

**Free Tier**: Google Cloud offers $300 free credit for new accounts, which should cover this use case for a long time.

