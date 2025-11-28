from typing import List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.models.database import Restaurant, RestaurantImage
from app.services.storage_service import upload_image_to_gcs, StorageServiceError


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
    
    Args:
        db: Database session
        place_id: Google Places place_id
        max_images: Maximum number of images to return (None for all)
        
    Returns:
        List of RestaurantImage instances
    """
    restaurant = db.query(Restaurant).filter(Restaurant.place_id == place_id).first()
    
    if not restaurant:
        return []
    
    query = db.query(RestaurantImage).filter(
        RestaurantImage.restaurant_id == restaurant.id
    ).order_by(RestaurantImage.created_at.desc())
    
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
            except StorageServiceError as e:
                # Log error but continue with other images
                print(f"Warning: Failed to upload image {photo_name} to GCS: {str(e)}")
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
        
        # Refresh all images to get IDs
        for image in cached_images:
            db.refresh(image)
        
        return cached_images
        
    except Exception as e:
        db.rollback()
        raise DatabaseServiceError(f"Failed to cache images: {str(e)}")


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

