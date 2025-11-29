from typing import List, Optional, Tuple, Dict
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.models.database import Restaurant, RestaurantImage
from app.services.storage_service import (
    upload_image_to_gcs,
    StorageServiceError,
    get_storage_client,
    GCS_BUCKET_NAME,
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
            # Check if image already exists
            existing_image = db.query(RestaurantImage).filter(
                RestaurantImage.restaurant_id == restaurant.id,
                RestaurantImage.photo_name == photo_name
            ).first()
            
            if existing_image:
                # Update category if it's different
                if existing_image.category != category:
                    existing_image.category = category
                    db.commit()
                    db.refresh(existing_image)
                cached_images.append(existing_image)
                continue
            
            # Upload to GCS
            try:
                public_url, bucket_path = upload_image_to_gcs(
                    image_bytes,
                    place_id,
                    photo_name
                )
                print(f"[DEBUG] Successfully uploaded {photo_name} to GCS: {public_url[:50]}...")
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
            
            # Create database record
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
    ai_tags: Optional[List[str]] = None
) -> RestaurantImage:
    """
    Update category and/or AI tags for a specific image.
    
    Args:
        db: Database session
        image_id: ID of the image to update
        category: Optional new category value
        ai_tags: Optional new AI tags list
        
    Returns:
        Updated RestaurantImage instance
        
    Raises:
        DatabaseServiceError: If image not found or update fails
    """
    try:
        image = db.query(RestaurantImage).filter(RestaurantImage.id == image_id).first()
        
        if not image:
            raise DatabaseServiceError(f"Image with id {image_id} not found")
        
        # Update category if provided
        if category is not None:
            image.category = category
        
        # Update AI tags if provided
        if ai_tags is not None:
            image.ai_tags = ai_tags
        
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


def store_ai_tags_for_images(
    db: Session,
    place_id: str,
    image_ai_tags: List[Tuple[str, List[str]]]
) -> None:
    """
    Store AI-generated tags for images.
    
    Args:
        db: Database session
        place_id: Google Places place_id
        image_ai_tags: List of tuples (photo_name, ai_tags_list)
        
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
        
        for image in images:
            if image.photo_name in tags_map:
                image.ai_tags = tags_map[image.photo_name]
        
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
    
    # Valid categories
    valid_categories = {"interior", "exterior", "food", "menu", "bar", "other"}
    
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
    description: str
) -> None:
    """
    Store the AI-generated description for a restaurant.
    
    Args:
        db: Database session
        place_id: Google Places place_id
        description: AI-generated description to store
    """
    try:
        restaurant = db.query(Restaurant).filter(Restaurant.place_id == place_id).first()
        
        if restaurant:
            restaurant.description = description
            db.commit()
            print(f"[CACHE] ✅ Stored AI description for {place_id}")
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
    max_images: int = 10,
    max_selected: int = 5
) -> Optional[Dict]:
    """
    Get complete cached restaurant data including name, images, categories, and AI analysis.
    This allows for instant responses when all data is fully cached.
    
    Returns None if ANY required data is missing, triggering the normal flow to fill the cache.
    
    Args:
        db: Database session
        place_id: Google Places place_id
        max_images: Maximum number of images to retrieve
        max_selected: Maximum number of images to select for analysis
        
    Returns:
        Dictionary with 'restaurant_name', 'images', 'image_urls', 'photo_names', 
        'categories', 'analysis', or None if not all data is available
    """
    restaurant = db.query(Restaurant).filter(Restaurant.place_id == place_id).first()
    
    if not restaurant:
        print(f"[CACHE] ❌ No restaurant found for place_id: {place_id}")
        return None
    
    # Get cached images
    cached_images = get_cached_images(db, place_id, max_images)
    
    if not cached_images or len(cached_images) < max_selected:
        print(f"[CACHE] ❌ Not enough cached images: {len(cached_images) if cached_images else 0} < {max_selected}")
        return None
    
    # Valid categories for filtering
    valid_categories = {"interior", "exterior", "food", "menu", "bar", "other"}
    
    # Check that we have enough images with VALID categories
    images_with_valid_categories = [
        img for img in cached_images 
        if img.category and img.category.strip().lower() in valid_categories
    ]
    
    if len(images_with_valid_categories) < max_selected:
        print(f"[CACHE] ❌ Not enough images with valid categories: {len(images_with_valid_categories)} < {max_selected}")
        return None
    
    # Group images by category for diverse selection
    by_category = {}
    for img in images_with_valid_categories:
        category = img.category.lower()
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(img)
    
    # Select diverse images (deterministic selection for consistent results)
    category_priority = ["interior", "food", "exterior", "bar", "menu", "other"]
    selected_images = []
    selected_indices = set()  # Prevent duplicates
    
    # First pass: get one from each priority category
    for category in category_priority:
        if category in by_category and len(selected_images) < max_selected:
            for img in by_category[category]:
                if img.id not in selected_indices:
                    selected_images.append(img)
                    selected_indices.add(img.id)
                    break
    
    # Second pass: fill remaining slots from priority categories
    for category in category_priority:
        if len(selected_images) >= max_selected:
            break
        for img in by_category.get(category, []):
            if img.id not in selected_indices and len(selected_images) < max_selected:
                selected_images.append(img)
                selected_indices.add(img.id)
    
    if len(selected_images) < max_selected:
        print(f"[CACHE] ❌ Could not select enough diverse images: {len(selected_images)} < {max_selected}")
        return None
    
    # Check that all selected images have AI tags
    images_with_ai_tags = [img for img in selected_images if img.ai_tags and len(img.ai_tags) > 0]
    
    if len(images_with_ai_tags) < max_selected:
        print(f"[CACHE] ❌ Not enough images with AI tags: {len(images_with_ai_tags)} < {max_selected}")
        return None
    
    # Use the selected images (they all have categories and AI tags at this point)
    final_images = selected_images[:max_selected]
    selected_photo_names = [img.photo_name for img in final_images]
    
    # Get cached analysis using these photo_names
    analysis = get_cached_restaurant_analysis(db, place_id, selected_photo_names)
    if not analysis:
        print(f"[CACHE] ❌ Failed to construct analysis from cached AI tags")
        return None
    
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
                'ai_tags': img.ai_tags
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

