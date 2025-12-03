from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.models.database import Restaurant, RestaurantImage, PromptVersion
from app.services.storage_service import (
    upload_image_to_gcs,
    delete_image_from_gcs,
    StorageServiceError,
    get_storage_client,
    GCS_BUCKET_NAME,
)
from app.config.ai_versions import (
    QUALITY_PEOPLE_THRESHOLD,
    IMAGE_QUALITY_THRESHOLD,
)


def image_passes_quality_thresholds(
    image: "RestaurantImage",
    people_threshold: float = QUALITY_PEOPLE_THRESHOLD,
    image_quality_threshold: float = IMAGE_QUALITY_THRESHOLD,
) -> bool:
    """
    Check if an image passes quality thresholds.
    
    An image passes if:
    1. It has been quality-scored (has scores)
    2. All scores are above their respective thresholds
    
    Args:
        image: RestaurantImage to check
        people_threshold: Minimum people score (higher = fewer people)
        image_quality_threshold: Minimum image quality score (higher = clearer)
        
    Returns:
        True if image passes all quality thresholds, False otherwise
    """
    # If not scored yet, fail quality check (require scoring first)
    if image.people_confidence_score is None:
        return False
    if image.image_quality_score is None:
        return False
    
    # Check all thresholds
    return (
        image.people_confidence_score > people_threshold and
        image.image_quality_score > image_quality_threshold
    )


class DatabaseServiceError(Exception):
    """Custom exception for database service errors"""
    pass


def get_or_create_restaurant(db: Session, place_id: str, name: str) -> Restaurant:
    """
    Get restaurant by place_id or create if it doesn't exist.
    
    Args:
        db: Database session
        place_id: Google Places place_id
        name: Restaurant name
        
    Returns:
        Restaurant model instance
        
    Raises:
        DatabaseServiceError: If operation fails
    """
    try:
        restaurant = db.query(Restaurant).filter(Restaurant.place_id == place_id).first()
        
        if not restaurant:
            restaurant = Restaurant(place_id=place_id, name=name)
            db.add(restaurant)
            db.commit()
            db.refresh(restaurant)
        else:
            # Update name if it has changed
            if restaurant.name != name:
                restaurant.name = name
                db.commit()
                db.refresh(restaurant)
        
        return restaurant
    except IntegrityError as e:
        db.rollback()
        raise DatabaseServiceError(f"Failed to create/get restaurant: {str(e)}")
    except Exception as e:
        db.rollback()
        raise DatabaseServiceError(f"Database error: {str(e)}")


def get_cached_images(
    db: Session,
    place_id: str,
    max_images: Optional[int] = None
) -> List[RestaurantImage]:
    """
    Retrieve cached images for a restaurant.
    Prioritizes images that have categories and AI tags (fully cached).
    
    Args:
        db: Database session
        place_id: Google Places place_id
        max_images: Maximum number of images to return (None for all)
        
    Returns:
        List of RestaurantImage instances, with fully-cached images first
    """
    from sqlalchemy import case, and_
    
    restaurant = db.query(Restaurant).filter(Restaurant.place_id == place_id).first()
    
    if not restaurant:
        return []
    
    # Prioritize images that have both category and AI tags (fully cached)
    # Then images with just category, then uncategorized images
    # This ensures we use fully-cached images first for fast responses
    priority = case(
        (and_(RestaurantImage.category.isnot(None), RestaurantImage.ai_tags.isnot(None)), 0),  # Both: highest priority
        (RestaurantImage.category.isnot(None), 1),  # Just category: medium priority
        else_=2  # Neither: lowest priority
    )
    
    query = db.query(RestaurantImage).filter(
        RestaurantImage.restaurant_id == restaurant.id
    ).order_by(priority, RestaurantImage.created_at.desc())
    
    if max_images:
        query = query.limit(max_images)
    
    return query.all()


def cache_images(
    db: Session,
    place_id: str,
    restaurant_name: str,
    images_with_metadata: List[Tuple[bytes, str, str]]
) -> List[RestaurantImage]:
    """
    Store images and metadata in database and upload to GCS.
    
    Uses content-hash duplicate prevention:
    - GCS layer: If identical image content exists, reuses existing file
    - DB layer: If GCS path already has a record, reuses existing record
    
    Args:
        db: Database session
        place_id: Google Places place_id
        restaurant_name: Restaurant name
        images_with_metadata: List of tuples (image_bytes, photo_name, category)
        
    Returns:
        List of created RestaurantImage instances
        
    Raises:
        DatabaseServiceError: If operation fails
    """
    try:
        # Get or create restaurant
        restaurant = get_or_create_restaurant(db, place_id, restaurant_name)
        
        cached_images = []
        
        for image_bytes, photo_name, category in images_with_metadata:
            # Check if image already exists by photo_name
            existing_image = db.query(RestaurantImage).filter(
                RestaurantImage.restaurant_id == restaurant.id,
                RestaurantImage.photo_name == photo_name
            ).first()
            
            if existing_image:
                # Don't overwrite existing categories - preserve what's already cached
                # Only set category if it was previously None/empty
                if not existing_image.category and category is not None:
                    existing_image.category = category
                    db.commit()
                    db.refresh(existing_image)
                cached_images.append(existing_image)
                continue
            
            # Upload to GCS (with content-hash duplicate detection)
            try:
                public_url, bucket_path = upload_image_to_gcs(
                    image_bytes,
                    place_id,
                    photo_name
                )
            except StorageServiceError as e:
                # Log error but continue with other images
                import traceback
                print(f"[ERROR] Failed to upload image {photo_name} to GCS: {str(e)}")
                print(f"[ERROR] Traceback: {traceback.format_exc()}")
                continue
            except Exception as e:
                import traceback
                print(f"[ERROR] Unexpected error uploading to GCS: {str(e)}")
                print(f"[ERROR] Traceback: {traceback.format_exc()}")
                continue
            
            # Check if a DB record already exists for this GCS path
            # (This happens when GCS detected a content duplicate)
            existing_by_path = db.query(RestaurantImage).filter(
                RestaurantImage.restaurant_id == restaurant.id,
                RestaurantImage.gcs_bucket_path == bucket_path
            ).first()
            
            if existing_by_path:
                # GCS path already has a record - this is a content duplicate
                # Don't overwrite existing categories - preserve what's already cached
                if not existing_by_path.category and category is not None:
                    existing_by_path.category = category
                    db.commit()
                    db.refresh(existing_by_path)
                print(f"[DB] ♻️  Reusing existing DB record for duplicate content")
                cached_images.append(existing_by_path)
                continue
            
            # Create new database record
            new_image = RestaurantImage(
                restaurant_id=restaurant.id,
                photo_name=photo_name,
                gcs_url=public_url,
                gcs_bucket_path=bucket_path,
                category=category
            )
            
            db.add(new_image)
            cached_images.append(new_image)
        
        db.commit()
        print(f"[DEBUG] Committed {len(cached_images)} images to database")
        
        # Refresh all images to get IDs
        for image in cached_images:
            db.refresh(image)
        
        print(f"[DEBUG] Successfully cached {len(cached_images)} images for {place_id}")
        return cached_images
        
    except Exception as e:
        db.rollback()
        import traceback
        print(f"[ERROR] Failed to cache images: {str(e)}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        raise DatabaseServiceError(f"Failed to cache images: {str(e)}")


def delete_restaurant_images(
    db: Session,
    place_id: str,
    delete_from_gcs: bool = True
) -> Dict[str, int]:
    """
    Delete all images for a restaurant from database and optionally from GCS.
    
    This is used for completely refreshing a restaurant's images.
    
    Args:
        db: Database session
        place_id: Google Places place_id
        delete_from_gcs: If True, also delete images from GCS bucket
        
    Returns:
        Dict with counts: {"deleted_db": N, "deleted_gcs": N, "gcs_errors": N}
        
    Raises:
        DatabaseServiceError: If operation fails
    """
    try:
        restaurant = db.query(Restaurant).filter(Restaurant.place_id == place_id).first()
        
        if not restaurant:
            return {"deleted_db": 0, "deleted_gcs": 0, "gcs_errors": 0}
        
        # Get all images for this restaurant
        images = db.query(RestaurantImage).filter(
            RestaurantImage.restaurant_id == restaurant.id
        ).all()
        
        deleted_db = 0
        deleted_gcs = 0
        gcs_errors = 0
        
        for image in images:
            # Delete from GCS if requested
            if delete_from_gcs and image.gcs_bucket_path:
                try:
                    delete_image_from_gcs(image.gcs_bucket_path)
                    deleted_gcs += 1
                except StorageServiceError as e:
                    print(f"[WARNING] Failed to delete from GCS: {image.gcs_bucket_path}: {e}")
                    gcs_errors += 1
                except Exception as e:
                    print(f"[WARNING] Unexpected error deleting from GCS: {e}")
                    gcs_errors += 1
            
            # Delete from database
            db.delete(image)
            deleted_db += 1
        
        # Also clear the restaurant's description and embedding since they're based on images
        restaurant.description = None
        restaurant.description_version = None
        restaurant.description_updated_at = None
        restaurant.embedding = None
        restaurant.embedding_version = None
        restaurant.embedding_updated_at = None
        
        db.commit()
        
        print(f"[DB] 🗑️ Deleted {deleted_db} images for {place_id} (GCS: {deleted_gcs}, errors: {gcs_errors})")
        
        return {
            "deleted_db": deleted_db,
            "deleted_gcs": deleted_gcs,
            "gcs_errors": gcs_errors
        }
        
    except Exception as e:
        db.rollback()
        import traceback
        print(f"[ERROR] Failed to delete restaurant images: {str(e)}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        raise DatabaseServiceError(f"Failed to delete restaurant images: {str(e)}")


def backfill_images_from_gcs(
    db: Session,
    place_id: str,
    restaurant_name: Optional[str] = None,
) -> Dict[str, int]:
    """
    Backfill RestaurantImage rows for existing images already stored in GCS.

    This is useful when images were uploaded to the bucket (e.g., via scripts)
    but no corresponding database rows were created.

    Args:
        db: Database session
        place_id: Google Places place_id (used in GCS path: images/{place_id}/...)
        restaurant_name: Optional restaurant name to use when creating Restaurant

    Returns:
        Dict with counts of created and skipped images.
    """
    if not GCS_BUCKET_NAME:
        raise DatabaseServiceError("GCS_BUCKET_NAME environment variable is not set")

    try:
        # Ensure we have a restaurant row
        if not restaurant_name:
            # Fallback name if caller doesn't provide one
            restaurant_name = place_id
        restaurant = get_or_create_restaurant(db, place_id, restaurant_name)

        client = get_storage_client()
        bucket = client.bucket(GCS_BUCKET_NAME)

        prefix = f"images/{place_id}/"
        blobs = list(bucket.list_blobs(prefix=prefix))

        created = 0
        skipped = 0

        for blob in blobs:
            # Skip non-file placeholders
            if not blob.name or blob.name.endswith("/"):
                continue

            # Check if we already have a DB row for this bucket path
            existing = (
                db.query(RestaurantImage)
                .filter(
                    RestaurantImage.restaurant_id == restaurant.id,
                    RestaurantImage.gcs_bucket_path == blob.name,
                )
                .first()
            )
            if existing:
                skipped += 1
                continue

            # Get URL for the object.
            # If the bucket uses uniform bucket-level access, make_public() will fail,
            # so we rely on the standard media URL format instead of legacy ACLs.
            try:
                blob.make_public()
                public_url = blob.public_url
            except Exception:
                public_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{blob.name}"

            # Derive a stable photo_name from the blob path
            filename = blob.name.split("/")[-1]
            photo_name = filename.rsplit(".", 1)[0] if "." in filename else filename

            new_image = RestaurantImage(
                restaurant_id=restaurant.id,
                photo_name=photo_name,
                gcs_url=public_url,
                gcs_bucket_path=blob.name,
                category=None,
            )
            db.add(new_image)
            created += 1

        if created:
            db.commit()

        return {"created": created, "skipped": skipped}
    except Exception as e:
        db.rollback()
        import traceback

        print(f"[ERROR] Failed to backfill images from GCS: {str(e)}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        raise DatabaseServiceError(f"Failed to backfill images from GCS: {str(e)}")


def update_image_tags(
    db: Session,
    image_id: int,
    category: Optional[str] = None,
    ai_tags: Optional[List[str]] = None,
    category_version: Optional[str] = None,
    tags_version: Optional[str] = None,
    is_displayed: Optional[bool] = None
) -> RestaurantImage:
    """
    Update category, AI tags, and/or display status for a specific image.
    
    Args:
        db: Database session
        image_id: ID of the image to update
        category: Optional new category value
        ai_tags: Optional new AI tags list
        category_version: Version of the categorization logic used
        tags_version: Version of the tagging logic used
        is_displayed: Optional flag to mark if image should be shown on website
        
    Returns:
        Updated RestaurantImage instance
        
    Raises:
        DatabaseServiceError: If image not found or update fails
    """
    try:
        image = db.query(RestaurantImage).filter(RestaurantImage.id == image_id).first()
        
        if not image:
            raise DatabaseServiceError(f"Image with id {image_id} not found")
        
        now = datetime.now(timezone.utc)
        
        # Update category if provided
        if category is not None:
            image.category = category
            if category_version:
                image.category_version = category_version
                image.category_updated_at = now
        
        # Update AI tags if provided
        if ai_tags is not None:
            image.ai_tags = ai_tags
            if tags_version:
                image.tags_version = tags_version
                image.tags_updated_at = now
        
        # Update display status if provided
        if is_displayed is not None:
            image.is_displayed = is_displayed
        
        db.commit()
        db.refresh(image)
        
        return image
        
    except DatabaseServiceError:
        raise
    except Exception as e:
        db.rollback()
        raise DatabaseServiceError(f"Failed to update image tags: {str(e)}")


def get_image_by_id(db: Session, image_id: int) -> Optional[RestaurantImage]:
    """
    Retrieve a single image by ID.
    
    Args:
        db: Database session
        image_id: ID of the image
        
    Returns:
        RestaurantImage instance or None if not found
    """
    return db.query(RestaurantImage).filter(RestaurantImage.id == image_id).first()


def get_displayed_images(
    db: Session,
    place_id: str,
) -> List[RestaurantImage]:
    """
    Get only images that are marked for display on the website.
    
    Args:
        db: Database session
        place_id: Google Places place_id
        
    Returns:
        List of RestaurantImage instances where is_displayed=True
    """
    restaurant = db.query(Restaurant).filter(Restaurant.place_id == place_id).first()
    
    if not restaurant:
        return []
    
    return db.query(RestaurantImage).filter(
        RestaurantImage.restaurant_id == restaurant.id,
        RestaurantImage.is_displayed == True
    ).order_by(RestaurantImage.created_at.desc()).all()


def mark_images_as_displayed(
    db: Session,
    place_id: str,
    image_ids: List[int],
    reset_others: bool = True
) -> int:
    """
    Mark specific images as displayed (for showing on website).
    Optionally reset all other images for this restaurant to not displayed.
    
    Args:
        db: Database session
        place_id: Google Places place_id
        image_ids: List of image IDs to mark as displayed
        reset_others: If True, set is_displayed=False for all other images of this restaurant
        
    Returns:
        Number of images marked as displayed
        
    Raises:
        DatabaseServiceError: If operation fails
    """
    try:
        restaurant = db.query(Restaurant).filter(Restaurant.place_id == place_id).first()
        
        if not restaurant:
            raise DatabaseServiceError(f"Restaurant with place_id {place_id} not found")
        
        # Reset all images for this restaurant if requested
        if reset_others:
            db.query(RestaurantImage).filter(
                RestaurantImage.restaurant_id == restaurant.id
            ).update({"is_displayed": False})
        
        # Mark specified images as displayed
        updated = db.query(RestaurantImage).filter(
            RestaurantImage.id.in_(image_ids),
            RestaurantImage.restaurant_id == restaurant.id
        ).update({"is_displayed": True}, synchronize_session=False)
        
        db.commit()
        
        print(f"[DB] ✅ Marked {updated} images as displayed for {place_id}")
        return updated
        
    except DatabaseServiceError:
        raise
    except Exception as e:
        db.rollback()
        raise DatabaseServiceError(f"Failed to mark images as displayed: {str(e)}")


def store_ai_tags_for_images(
    db: Session,
    place_id: str,
    image_ai_tags: List[Tuple[str, List[str]]],
    version: Optional[str] = None
) -> None:
    """
    Store AI-generated tags for images.
    
    Args:
        db: Database session
        place_id: Google Places place_id
        image_ai_tags: List of tuples (photo_name, ai_tags_list)
        version: Version of the tagging logic used
        
    Raises:
        DatabaseServiceError: If operation fails
    """
    try:
        restaurant = db.query(Restaurant).filter(Restaurant.place_id == place_id).first()
        
        if not restaurant:
            raise DatabaseServiceError(f"Restaurant with place_id {place_id} not found")
        
        # Create a mapping of photo_name to ai_tags
        tags_map = {photo_name: tags for photo_name, tags in image_ai_tags}
        
        # Update images
        images = db.query(RestaurantImage).filter(
            RestaurantImage.restaurant_id == restaurant.id,
            RestaurantImage.photo_name.in_(tags_map.keys())
        ).all()
        
        now = datetime.now(timezone.utc)
        
        for image in images:
            if image.photo_name in tags_map:
                image.ai_tags = tags_map[image.photo_name]
                if version:
                    image.tags_version = version
                    image.tags_updated_at = now
        
        db.commit()
        
    except Exception as e:
        db.rollback()
        raise DatabaseServiceError(f"Failed to store AI tags: {str(e)}")


def get_cached_categories_and_tags(
    db: Session,
    place_id: str,
    photo_names: List[str]
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Get cached categories and AI tags for images by photo_name.
    Only returns valid, non-empty categories and tags.
    
    Args:
        db: Database session
        place_id: Google Places place_id
        photo_names: List of photo_names to look up
        
    Returns:
        Tuple of (categories_dict, ai_tags_dict) where:
        - categories_dict: {photo_name: category} (only valid categories)
        - ai_tags_dict: {photo_name: [tags]} (only non-empty tag lists)
    """
    restaurant = db.query(Restaurant).filter(Restaurant.place_id == place_id).first()
    
    if not restaurant:
        return {}, {}
    
    images = db.query(RestaurantImage).filter(
        RestaurantImage.restaurant_id == restaurant.id,
        RestaurantImage.photo_name.in_(photo_names)
    ).all()
    
    # Valid categories (drink is a valid category now)
    valid_categories = {"interior", "exterior", "food", "drink", "menu", "bar", "other"}
    
    categories = {}
    ai_tags = {}
    
    for image in images:
        # Only include valid, non-empty categories
        if image.category and image.category.strip() and image.category.lower() in valid_categories:
            categories[image.photo_name] = image.category.lower()
        # Only include non-empty tag lists
        if image.ai_tags and len(image.ai_tags) > 0:
            ai_tags[image.photo_name] = image.ai_tags
    
    return categories, ai_tags


def get_cached_restaurant_analysis(
    db: Session,
    place_id: str,
    photo_names: List[str]
) -> Optional[Dict[str, any]]:
    """
    Get cached AI analysis (tags and description) for a restaurant.
    Returns the cached AI-generated description if available, otherwise None.
    
    Args:
        db: Database session
        place_id: Google Places place_id
        photo_names: List of photo_names for selected images
        
    Returns:
        Dictionary with 'tags' and 'description', or None if not fully cached
    """
    restaurant = db.query(Restaurant).filter(Restaurant.place_id == place_id).first()
    
    if not restaurant:
        return None
    
    images = db.query(RestaurantImage).filter(
        RestaurantImage.restaurant_id == restaurant.id,
        RestaurantImage.photo_name.in_(photo_names)
    ).all()
    
    # Check if all images have AI tags
    if len(images) != len(photo_names):
        return None
    
    all_tags = []
    for image in images:
        if not image.ai_tags:
            return None
        all_tags.extend(image.ai_tags)
    
    # Get unique tags
    unique_tags = list(set(all_tags))
    
    if not unique_tags:
        return None
    
    # Use the cached AI-generated description from the restaurant
    # If no cached description, return None to trigger a fresh AI analysis
    if restaurant.description:
        return {
            "tags": unique_tags,
            "description": restaurant.description
        }
    
    # No cached description - return None to trigger AI analysis
    print(f"[CACHE] ❌ No cached description for {place_id}, need AI analysis")
    return None


def store_restaurant_description(
    db: Session,
    place_id: str,
    description: str,
    version: Optional[str] = None
) -> None:
    """
    Store the AI-generated description for a restaurant.
    
    Args:
        db: Database session
        place_id: Google Places place_id
        description: AI-generated description to store
        version: Version of the description generation logic used
    """
    try:
        restaurant = db.query(Restaurant).filter(Restaurant.place_id == place_id).first()
        
        if restaurant:
            restaurant.description = description
            if version:
                restaurant.description_version = version
                restaurant.description_updated_at = datetime.now(timezone.utc)
            db.commit()
            print(f"[CACHE] ✅ Stored AI description for {place_id} (version: {version})")
        else:
            print(f"[CACHE] ❌ Cannot store description - restaurant {place_id} not found")
    except Exception as e:
        db.rollback()
        print(f"[ERROR] Failed to store description: {str(e)}")


def find_restaurant_by_text(
    db: Session,
    query_text: str,
    min_similarity: float = 0.75
) -> Optional[str]:
    """
    Find a restaurant in the database by text query using semantic search.
    This allows us to avoid Places API calls when the restaurant is already cached.
    
    Args:
        db: Database session
        query_text: Search query (restaurant name, address, etc.)
        min_similarity: Minimum similarity score to consider a match
        
    Returns:
        place_id if found, None otherwise
    """
    try:
        # Normalize query: lowercase, strip
        normalized_query = query_text.strip().lower()
        
        # Strategy 1: Try exact match on full query
        restaurant = db.query(Restaurant).filter(
            Restaurant.name.ilike(f"%{normalized_query}%")
        ).first()
        
        if restaurant:
            print(f"[DB] ✅ Found restaurant by exact match: {restaurant.name}")
            return restaurant.place_id
        
        # Strategy 2: If query contains comma (e.g., "Restaurant, City"), try just the name part
        if ',' in normalized_query:
            name_part = normalized_query.split(',')[0].strip()
            restaurant = db.query(Restaurant).filter(
                Restaurant.name.ilike(f"%{name_part}%")
            ).first()
            
            if restaurant:
                print(f"[DB] ✅ Found restaurant by name part: {restaurant.name}")
                return restaurant.place_id
        
        # If no exact match, try semantic search using embeddings
        from app.services.recommendation_service import search_restaurants_by_text
        results = search_restaurants_by_text(
            db,
            query_text,
            limit=1,
            min_similarity=min_similarity
        )
        
        if results and len(results) > 0:
            restaurant, score = results[0]
            if score >= min_similarity:
                return restaurant.place_id
        
        return None
    except Exception as e:
        # Log error but don't fail - fall back to API
        print(f"[WARNING] Database search failed: {str(e)}")
        return None


def get_complete_cached_restaurant_data(
    db: Session,
    place_id: str,
    max_images: int = 50,
    max_selected: int = 20,
    quota: Optional[Dict[str, int]] = None,
    max_bar: int = 2,
    update_displayed: bool = True
) -> Optional[Dict]:
    """
    Get complete cached restaurant data including name, images, categories, and AI analysis.
    This allows for instant responses when all data is fully cached.
    
    Returns None if ANY required data is missing, triggering the normal flow to fill the cache.
    
    When data is successfully returned, marks selected images as is_displayed=True
    so the website can filter to only show those images.
    
    Rules:
    - Bar counts as interior (max 2 bar images allowed by default)
    - Menu, "other", and "skipped" images are excluded entirely
    - Drink is a valid category
    
    Args:
        db: Database session
        place_id: Google Places place_id
        max_images: Maximum number of images to retrieve from cache
        max_selected: Maximum number of images to select for display
        quota: Dict mapping category to desired count.
               Default: {"food": 10, "interior": 10}
        max_bar: Maximum bar images (counts toward interior quota)
        update_displayed: If True, update is_displayed flag for selected images
        
    Returns:
        Dictionary with 'restaurant_name', 'images', 'image_urls', 'photo_names', 
        'categories', 'analysis', or None if not all data is available
    """
    # Default quota: 2 exterior, 8 interior, 8 food, 2 drink = 20 total
    if quota is None:
        quota = {"food": 10, "interior": 10}
    
    total_quota = sum(quota.values())
    
    # Excluded categories - never select these
    excluded_categories = {"menu", "other", "skipped"}
    
    restaurant = db.query(Restaurant).filter(Restaurant.place_id == place_id).first()
    
    if not restaurant:
        print(f"[CACHE] ❌ No restaurant found for place_id: {place_id}")
        return None
    
    # First check if we have already-displayed images (fast path for subsequent requests)
    displayed_images = db.query(RestaurantImage).filter(
        RestaurantImage.restaurant_id == restaurant.id,
        RestaurantImage.is_displayed == True
    ).all()
    
    # If we have displayed images with AI tags, use them directly
    # BUT only if they all pass quality thresholds (to avoid showing bad images)
    if displayed_images and len(displayed_images) >= 5:
        # Filter to only images that pass quality thresholds
        quality_passing = [img for img in displayed_images if image_passes_quality_thresholds(img)]
        
        # If some displayed images don't pass quality, we need to re-select
        if len(quality_passing) < len(displayed_images):
            print(f"[CACHE] ⚠️ {len(displayed_images) - len(quality_passing)} displayed images fail quality thresholds, re-selecting...")
            displayed_images = None  # Force re-selection below
        else:
            images_with_ai_tags = [img for img in displayed_images if img.ai_tags and len(img.ai_tags) > 0]
            if len(images_with_ai_tags) == len(displayed_images):
                photo_names_with_tags = [img.photo_name for img in images_with_ai_tags]
                analysis = get_cached_restaurant_analysis(db, place_id, photo_names_with_tags)
                if analysis:
                    print(f"[CACHE] ✅ Using pre-selected displayed images for {place_id}")
                    seen_urls = set()
                    image_metadata = []
                    for img in displayed_images:
                        if img.gcs_url not in seen_urls:
                            image_metadata.append({
                                'id': img.id,
                                'photo_name': img.photo_name,
                                'gcs_url': img.gcs_url,
                                'category': img.category.lower() if img.category else 'other',
                                'ai_tags': img.ai_tags,
                                'is_displayed': img.is_displayed
                            })
                            seen_urls.add(img.gcs_url)
                    
                    return {
                        'restaurant_name': restaurant.name,
                        'images': image_metadata,
                        'image_urls': [img['gcs_url'] for img in image_metadata],
                        'photo_names': [img['photo_name'] for img in image_metadata],
                        'categories': {img['photo_name']: img['category'] for img in image_metadata},
                        'analysis': analysis
                    }
    
    # Get all cached images
    cached_images = get_cached_images(db, place_id, max_images)
    
    if not cached_images:
        print(f"[CACHE] ❌ No cached images found")
        return None
    
    # Valid categories for selection (excluding menu, other, skipped)
    valid_categories = {"interior", "exterior", "food", "drink", "bar"}
    
    # Filter to only valid categories AND passing quality thresholds
    images_with_valid_categories = [
        img for img in cached_images 
        if img.category and img.category.strip().lower() in valid_categories
    ]
    
    # Apply quality filtering - only select images that pass quality thresholds
    quality_passing_images = [
        img for img in images_with_valid_categories
        if image_passes_quality_thresholds(img)
    ]
    
    # Log quality filtering stats
    quality_failed = len(images_with_valid_categories) - len(quality_passing_images)
    if quality_failed > 0:
        print(f"[CACHE] 🔍 Quality filter: {quality_failed} images filtered out, {len(quality_passing_images)} passed")
    
    # Use quality-passing images for selection
    images_with_valid_categories = quality_passing_images
    
    # Only require minimum images, not the full quota (restaurant may have fewer photos)
    min_valid_images = 5
    if len(images_with_valid_categories) < min_valid_images:
        print(f"[CACHE] ❌ Not enough quality images: {len(images_with_valid_categories)} < {min_valid_images}")
        return None
    
    # Group images by category for quota-based selection
    by_category = {}
    for img in images_with_valid_categories:
        category = img.category.lower()
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(img)
    
    print(f"[CACHE] Categories available (after quality filter): {[(k, len(v)) for k, v in by_category.items()]}")
    
    # Select images based on quota (2 exterior, 8 food, 8 interior, 2 drink)
    # Bar counts as interior (max 2 bar allowed)
    selected_images = []
    selected_indices = set()  # Prevent duplicates
    remaining_slots = 0
    bar_count = 0
    
    # First pass: fill quotas from specified categories
    for category, count in quota.items():
        category_lower = category.lower()
        available = by_category.get(category_lower, [])
        added = 0
        
        for img in available:
            if img.id not in selected_indices and added < count:
                selected_images.append(img)
                selected_indices.add(img.id)
                added += 1
        
        remaining_slots += (count - added)
        
        if added < count:
            print(f"[CACHE] {category} only had {added}/{count} images")
    
    # Second pass: fill remaining interior slots with bar (bar counts as interior)
    interior_shortfall = quota.get("interior", 0) - len([
        img for img in selected_images if img.category.lower() == "interior"
    ])
    
    if interior_shortfall > 0 and bar_count < max_bar:
        for img in by_category.get("bar", []):
            if img.id not in selected_indices and bar_count < max_bar and interior_shortfall > 0:
                selected_images.append(img)
                selected_indices.add(img.id)
                bar_count += 1
                remaining_slots -= 1
                interior_shortfall -= 1
                print(f"[CACHE] Using bar image as interior ({bar_count}/{max_bar})")
    
    # Third pass: fill remaining slots from valid fallback categories (no menu/other)
    if remaining_slots > 0:
        fallback_priority = ["food", "interior", "exterior", "drink"]
        
        for category in fallback_priority:
            if remaining_slots <= 0:
                break
            for img in by_category.get(category, []):
                if img.id not in selected_indices and remaining_slots > 0:
                    selected_images.append(img)
                    selected_indices.add(img.id)
                    remaining_slots -= 1
        
        # Can also use bar if still have remaining slots and haven't used max bar yet
        if remaining_slots > 0 and bar_count < max_bar:
            for img in by_category.get("bar", []):
                if img.id not in selected_indices and remaining_slots > 0 and bar_count < max_bar:
                    selected_images.append(img)
                    selected_indices.add(img.id)
                    bar_count += 1
                    remaining_slots -= 1
    
    # We might not have 20 images if the restaurant doesn't have that many valid photos
    # That's okay - work with what we have as long as we have at least some
    min_images_required = 5  # Minimum to provide useful analysis
    
    if len(selected_images) < min_images_required:
        print(f"[CACHE] ❌ Not enough valid images: {len(selected_images)} < {min_images_required}")
        return None
    
    if len(selected_images) < total_quota:
        print(f"[CACHE] ⚠️ Only have {len(selected_images)}/{total_quota} valid images (restaurant may have fewer photos)")
    
    # Check that ALL selected images have AI tags (not just 20)
    images_with_ai_tags = [img for img in selected_images if img.ai_tags and len(img.ai_tags) > 0]
    
    if len(images_with_ai_tags) < len(selected_images):
        print(f"[CACHE] ❌ Not all images have AI tags: {len(images_with_ai_tags)}/{len(selected_images)}")
        return None
    
    # Use all selected images (they all have categories at this point)
    final_images = selected_images
    selected_photo_names = [img.photo_name for img in final_images]
    
    # Get cached analysis - need all photo_names that have AI tags
    photo_names_with_tags = [img.photo_name for img in images_with_ai_tags]
    analysis = get_cached_restaurant_analysis(db, place_id, photo_names_with_tags)
    if not analysis:
        print(f"[CACHE] ❌ Failed to construct analysis from cached AI tags")
        return None
    
    # Mark selected images as displayed (for future fast lookups)
    if update_displayed:
        selected_ids = [img.id for img in final_images]
        mark_images_as_displayed(db, place_id, selected_ids, reset_others=True)
    
    print(f"[CACHE] ✅ Complete cached data available for {place_id} ({restaurant.name})")
    
    # Build image metadata - ensure no duplicates
    seen_urls = set()
    image_metadata = []
    for img in final_images:
        if img.gcs_url not in seen_urls:
            image_metadata.append({
                'id': img.id,
                'photo_name': img.photo_name,
                'gcs_url': img.gcs_url,
                'category': img.category.lower(),
                'ai_tags': img.ai_tags,
                'is_displayed': True  # These are the displayed images
            })
            seen_urls.add(img.gcs_url)
    
    return {
        'restaurant_name': restaurant.name,
        'images': image_metadata,
        'image_urls': [img['gcs_url'] for img in image_metadata],
        'photo_names': [img['photo_name'] for img in image_metadata],
        'categories': {img['photo_name']: img['category'] for img in image_metadata},
        'analysis': analysis
    }


# =============================================================================
# Version Tracking Functions
# =============================================================================

def get_stale_restaurants(
    db: Session,
    component: str,
    current_version: str
) -> List[Restaurant]:
    """
    Get restaurants where the specified component is stale (version < current).
    
    Args:
        db: Database session
        component: "description" or "embedding"
        current_version: The current active version to compare against
        
    Returns:
        List of Restaurant instances that need refresh
        
    Raises:
        DatabaseServiceError: If invalid component specified
    """
    if component == "description":
        # Stale if: no version set, or version doesn't match current
        return db.query(Restaurant).filter(
            (Restaurant.description_version == None) | 
            (Restaurant.description_version != current_version)
        ).all()
    elif component == "embedding":
        return db.query(Restaurant).filter(
            (Restaurant.embedding_version == None) | 
            (Restaurant.embedding_version != current_version)
        ).all()
    else:
        raise DatabaseServiceError(f"Invalid component: {component}. Must be 'description' or 'embedding'")


def get_stale_images(
    db: Session,
    component: str,
    current_version: str
) -> List[RestaurantImage]:
    """
    Get restaurant images where the specified component is stale (version < current).
    
    Args:
        db: Database session
        component: "category", "tags", or "quality"
        current_version: The current active version to compare against
        
    Returns:
        List of RestaurantImage instances that need refresh
        
    Raises:
        DatabaseServiceError: If invalid component specified
    """
    if component == "category":
        return db.query(RestaurantImage).filter(
            (RestaurantImage.category_version == None) |   # noqa: E711
            (RestaurantImage.category_version != current_version)
        ).all()
    elif component == "tags":
        return db.query(RestaurantImage).filter(
            (RestaurantImage.tags_version == None) |   # noqa: E711
            (RestaurantImage.tags_version != current_version)
        ).all()
    elif component == "quality":
        return db.query(RestaurantImage).filter(
            (RestaurantImage.quality_version == None) |   # noqa: E711
            (RestaurantImage.quality_version != current_version)
        ).all()
    else:
        raise DatabaseServiceError(f"Invalid component: {component}. Must be 'category', 'tags', or 'quality'")


def get_restaurants_missing_data(
    db: Session,
    component: str
) -> List[Restaurant]:
    """
    Get restaurants that are completely missing data for a component.
    
    Args:
        db: Database session
        component: "description" or "embedding"
        
    Returns:
        List of Restaurant instances missing the specified data
    """
    if component == "description":
        return db.query(Restaurant).filter(Restaurant.description == None).all()
    elif component == "embedding":
        return db.query(Restaurant).filter(Restaurant.embedding == None).all()
    else:
        raise DatabaseServiceError(f"Invalid component: {component}. Must be 'description' or 'embedding'")


def get_images_missing_data(
    db: Session,
    component: str
) -> List[RestaurantImage]:
    """
    Get restaurant images that are completely missing data for a component.
    
    Args:
        db: Database session
        component: "category", "tags", or "quality"
        
    Returns:
        List of RestaurantImage instances missing the specified data
    """
    if component == "category":
        return db.query(RestaurantImage).filter(RestaurantImage.category == None).all()  # noqa: E711
    elif component == "tags":
        return db.query(RestaurantImage).filter(RestaurantImage.ai_tags == None).all()  # noqa: E711
    elif component == "quality":
        return db.query(RestaurantImage).filter(RestaurantImage.quality_version == None).all()  # noqa: E711
    else:
        raise DatabaseServiceError(f"Invalid component: {component}. Must be 'category', 'tags', or 'quality'")


def get_version_stats(db: Session) -> Dict:
    """
    Get statistics about version coverage across all data.
    
    Returns:
        Dict with counts of versioned/unversioned data for each component
    """
    stats = {}
    
    # Restaurant description stats
    total_restaurants = db.query(Restaurant).count()
    with_description = db.query(Restaurant).filter(Restaurant.description != None).count()
    with_description_version = db.query(Restaurant).filter(Restaurant.description_version != None).count()
    stats["description"] = {
        "total": total_restaurants,
        "with_data": with_description,
        "with_version": with_description_version,
        "missing_data": total_restaurants - with_description,
        "unversioned": with_description - with_description_version,
    }
    
    # Restaurant embedding stats
    with_embedding = db.query(Restaurant).filter(Restaurant.embedding != None).count()
    with_embedding_version = db.query(Restaurant).filter(Restaurant.embedding_version != None).count()
    stats["embedding"] = {
        "total": total_restaurants,
        "with_data": with_embedding,
        "with_version": with_embedding_version,
        "missing_data": total_restaurants - with_embedding,
        "unversioned": with_embedding - with_embedding_version,
    }
    
    # Image category stats
    total_images = db.query(RestaurantImage).count()
    with_category = db.query(RestaurantImage).filter(RestaurantImage.category != None).count()
    with_category_version = db.query(RestaurantImage).filter(RestaurantImage.category_version != None).count()
    stats["category"] = {
        "total": total_images,
        "with_data": with_category,
        "with_version": with_category_version,
        "missing_data": total_images - with_category,
        "unversioned": with_category - with_category_version,
    }
    
    # Image tags stats
    with_tags = db.query(RestaurantImage).filter(RestaurantImage.ai_tags != None).count()
    with_tags_version = db.query(RestaurantImage).filter(RestaurantImage.tags_version != None).count()
    stats["tags"] = {
        "total": total_images,
        "with_data": with_tags,
        "with_version": with_tags_version,
        "missing_data": total_images - with_tags,
        "unversioned": with_tags - with_tags_version,
    }
    
    return stats


# =============================================================================
# Prompt Version CRUD Functions
# =============================================================================

def create_prompt_version(
    db: Session,
    component: str,
    version: str,
    prompt_text: Optional[str] = None,
    model: Optional[str] = None,
    notes: Optional[str] = None,
    set_active: bool = True
) -> PromptVersion:
    """
    Create a new prompt version record.
    
    Args:
        db: Database session
        component: One of "category", "tags", "description", "embedding"
        version: Version string, e.g., "v1.1"
        prompt_text: The full prompt text (optional for embeddings)
        model: The model used, e.g., "gpt-4o"
        notes: Optional changelog notes
        set_active: If True, deactivates other versions for this component
        
    Returns:
        Created PromptVersion instance
        
    Raises:
        DatabaseServiceError: If creation fails
    """
    valid_components = {"category", "tags", "description", "embedding"}
    if component not in valid_components:
        raise DatabaseServiceError(f"Invalid component: {component}. Must be one of {valid_components}")
    
    try:
        # Deactivate existing versions for this component if setting this one active
        if set_active:
            db.query(PromptVersion).filter(
                PromptVersion.component == component,
                PromptVersion.is_active == True
            ).update({"is_active": False})
        
        # Create new version
        prompt_version = PromptVersion(
            component=component,
            version=version,
            prompt_text=prompt_text,
            model=model,
            notes=notes,
            is_active=set_active
        )
        db.add(prompt_version)
        db.commit()
        db.refresh(prompt_version)
        
        return prompt_version
        
    except IntegrityError as e:
        db.rollback()
        raise DatabaseServiceError(f"Version {version} already exists for component {component}")
    except Exception as e:
        db.rollback()
        raise DatabaseServiceError(f"Failed to create prompt version: {str(e)}")


def get_prompt_version(
    db: Session,
    component: str,
    version: Optional[str] = None
) -> Optional[PromptVersion]:
    """
    Get a prompt version record.
    
    Args:
        db: Database session
        component: One of "category", "tags", "description", "embedding"
        version: Specific version to get, or None for active version
        
    Returns:
        PromptVersion instance or None if not found
    """
    query = db.query(PromptVersion).filter(PromptVersion.component == component)
    
    if version:
        return query.filter(PromptVersion.version == version).first()
    else:
        return query.filter(PromptVersion.is_active == True).first()


def get_all_prompt_versions(
    db: Session,
    component: Optional[str] = None
) -> List[PromptVersion]:
    """
    Get all prompt version records.
    
    Args:
        db: Database session
        component: Optional filter by component
        
    Returns:
        List of PromptVersion instances
    """
    query = db.query(PromptVersion).order_by(
        PromptVersion.component,
        PromptVersion.created_at.desc()
    )
    
    if component:
        query = query.filter(PromptVersion.component == component)
    
    return query.all()


def set_active_prompt_version(
    db: Session,
    component: str,
    version: str
) -> PromptVersion:
    """
    Set a specific version as the active version for a component.
    
    Args:
        db: Database session
        component: One of "category", "tags", "description", "embedding"
        version: Version to activate
        
    Returns:
        Activated PromptVersion instance
        
    Raises:
        DatabaseServiceError: If version not found
    """
    try:
        # Deactivate all versions for this component
        db.query(PromptVersion).filter(
            PromptVersion.component == component
        ).update({"is_active": False})
        
        # Activate the specified version
        prompt_version = db.query(PromptVersion).filter(
            PromptVersion.component == component,
            PromptVersion.version == version
        ).first()
        
        if not prompt_version:
            raise DatabaseServiceError(f"Version {version} not found for component {component}")
        
        prompt_version.is_active = True
        db.commit()
        db.refresh(prompt_version)
        
        return prompt_version
        
    except DatabaseServiceError:
        raise
    except Exception as e:
        db.rollback()
        raise DatabaseServiceError(f"Failed to set active version: {str(e)}")

