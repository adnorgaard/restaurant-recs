# Quick GCS Setup Guide

## ✅ Step 1: Authenticate with Google Cloud

Run this command in your terminal (it will open a browser):

```bash
gcloud auth login
```

Follow the prompts to sign in with your Google account.

## ✅ Step 2: Set Your Project

If you know your Google Cloud Project ID:

```bash
gcloud config set project YOUR_PROJECT_ID
```

If you don't know it, list your projects:

```bash
gcloud projects list
```

## ✅ Step 3: Run Automated Setup

Once authenticated, run:

```bash
python scripts/auto_setup_gcs.py
```

This will:
- Create a GCS bucket (you'll be prompted for the name)
- Create a service account with Storage Admin permissions
- Generate and download the service account key file
- Automatically update your `.env` file

## Manual Alternative

If the automated script doesn't work, you can run these commands manually:

```bash
# Create bucket (replace BUCKET_NAME with your desired name)
gcloud storage buckets create gs://BUCKET_NAME --location=us-central1

# Create service account
gcloud iam service-accounts create restaurant-recs-storage \
    --display-name="Restaurant Recs Storage Service Account"

# Grant Storage Admin role (replace PROJECT_ID and SERVICE_ACCOUNT_EMAIL)
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:SERVICE_ACCOUNT_EMAIL" \
    --role="roles/storage.admin"

# Create and download key
gcloud iam service-accounts keys create restaurant-recs-storage-key.json \
    --iam-account=SERVICE_ACCOUNT_EMAIL
```

Then add to your `.env`:
```
GCS_BUCKET_NAME=your-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/restaurant-recs-storage-key.json
```

