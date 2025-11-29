#!/usr/bin/env python3
"""
Utility script to detect and optionally consolidate duplicate images in the GCS bucket.

What it does:
- Scans the configured GCS bucket (prefix: "images/").
- Groups objects with the same content hash (md5 + size) as duplicates.
- For each duplicate group:
  - Picks a canonical blob to keep (prefers ones referenced in the DB).
  - Optionally updates DB rows to point at the canonical object.
  - Optionally deletes non‑canonical duplicate blobs from the bucket.

By default this script runs in DRY‑RUN mode and prints what it *would* change.
Pass --apply to actually update the database and delete duplicate objects.
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv


def load_env():
    """Load .env from project root."""
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    # Ensure project root is on sys.path so we can import app.*
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def get_gcs_bucket():
    """Return (client, bucket, bucket_name) using app.services.storage_service."""
    from app.services.storage_service import get_storage_client, GCS_BUCKET_NAME, StorageServiceError

    if not GCS_BUCKET_NAME:
        raise StorageServiceError("GCS_BUCKET_NAME is not set in environment/.env")

    client = get_storage_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    return client, bucket, GCS_BUCKET_NAME


def find_duplicate_blobs(bucket, prefix: str = "images/"):
    """
    Scan bucket and group blobs that have identical content.

    We use (md5_hash, size) as the identity key – good enough for dedupe.
    Returns:
        List of lists, where each inner list is a group of duplicate blobs.
    """
    from google.cloud.storage import Blob  # type: ignore

    print(f"Scanning bucket '{bucket.name}' for blobs with prefix '{prefix}'...")
    groups = defaultdict(list)
    total = 0

    for blob in bucket.list_blobs(prefix=prefix):
        if not isinstance(blob, Blob):
            continue
        total += 1
        # md5_hash is base64-encoded; combine with size to reduce false positives
        content_hash = blob.md5_hash or blob.crc32c
        if not content_hash:
            # If we can't get a hash, skip from dedupe logic
            continue
        key = (content_hash, blob.size)
        groups[key].append(blob)

    duplicate_groups = [blobs for blobs in groups.values() if len(blobs) > 1]

    print(f"Scanned {total} blobs; found {len(duplicate_groups)} duplicate groups.")
    return duplicate_groups


def consolidate_duplicates(duplicate_groups, apply_changes: bool = False):
    """
    For each duplicate group:
      - Pick canonical blob (preferring ones referenced in DB).
      - Update DB references to non‑canonical blobs to point to canonical.
      - Delete non‑canonical blobs from GCS (if apply_changes is True).
    """
    from app.database import SessionLocal
    from app.models.database import RestaurantImage

    session = SessionLocal()
    total_deleted = 0
    total_db_rows_updated = 0

    try:
        for idx, group in enumerate(duplicate_groups, start=1):
            # Sort to make output deterministic
            group_sorted = sorted(group, key=lambda b: b.name)

            # Count DB references for each blob
            db_counts = {}
            for blob in group_sorted:
                count = (
                    session.query(RestaurantImage)
                    .filter(RestaurantImage.gcs_bucket_path == blob.name)
                    .count()
                )
                db_counts[blob.name] = count

            # Choose canonical: any blob with DB refs; otherwise first by name
            canonical_blob = None
            for blob in group_sorted:
                if db_counts.get(blob.name, 0) > 0:
                    canonical_blob = blob
                    break
            if canonical_blob is None:
                canonical_blob = group_sorted[0]

            print("\n" + "=" * 80)
            print(f"Duplicate group #{idx}:")
            print(f"  Canonical: {canonical_blob.name} (db_refs={db_counts.get(canonical_blob.name, 0)})")
            for blob in group_sorted:
                marker = "*" if blob.name == canonical_blob.name else " "
                print(
                    f"  {marker} {blob.name}  "
                    f"(size={blob.size}, db_refs={db_counts.get(blob.name, 0)})"
                )

            if not apply_changes:
                # Dry-run: just show what would happen
                continue

            # Ensure canonical is public and compute canonical URL
            canonical_blob.make_public()
            canonical_url = canonical_blob.public_url

            # Update DB references & delete non‑canonical blobs
            for blob in group_sorted:
                if blob.name == canonical_blob.name:
                    continue

                # Update any DB rows that reference this blob
                if db_counts.get(blob.name, 0) > 0:
                    images = (
                        session.query(RestaurantImage)
                        .filter(RestaurantImage.gcs_bucket_path == blob.name)
                        .all()
                    )
                    for img in images:
                        img.gcs_bucket_path = canonical_blob.name
                        img.gcs_url = canonical_url
                        total_db_rows_updated += 1

                # Delete the duplicate object from GCS
                blob.delete()
                total_deleted += 1

            session.commit()

    finally:
        session.close()

    print("\n" + "=" * 80)
    if apply_changes:
        print(
            f"Consolidation complete. Deleted {total_deleted} duplicate blobs; "
            f"updated {total_db_rows_updated} DB rows."
        )
    else:
        print("Dry-run complete. No changes were made.")


def main():
    parser = argparse.ArgumentParser(
        description="Find and consolidate duplicate images in the GCS bucket."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes: update DB references and delete duplicate blobs. "
        "Without this flag, the script only prints what it would do.",
    )
    parser.add_argument(
        "--prefix",
        default="images/",
        help="GCS object prefix to scan (default: 'images/').",
    )
    args = parser.parse_args()

    load_env()

    try:
        _, bucket, bucket_name = get_gcs_bucket()
    except Exception as e:
        print(f"Error getting GCS bucket: {e}")
        return 1

    print(f"Using bucket: {bucket_name}")
    duplicate_groups = find_duplicate_blobs(bucket, prefix=args.prefix)

    if not duplicate_groups:
        print("No duplicate objects found. Nothing to do.")
        return 0

    consolidate_duplicates(duplicate_groups, apply_changes=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


