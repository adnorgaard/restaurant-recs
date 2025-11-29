"""
Debugging service for cache inspection and performance monitoring.
"""
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from app.models.database import Restaurant, RestaurantImage
from app.services.database_service import get_cached_images, get_cached_restaurant_analysis


def inspect_restaurant_cache(db: Session, place_id: str) -> Dict:
    """
    Inspect the cache status for a restaurant.
    
    Returns detailed information about what's cached and what's missing.
    """
    restaurant = db.query(Restaurant).filter(Restaurant.place_id == place_id).first()
    
    if not restaurant:
        return {
            "place_id": place_id,
            "exists": False,
            "restaurant_name": None,
            "images": {
                "total": 0,
                "with_categories": 0,
                "with_ai_tags": 0,
                "details": []
            },
            "cache_status": "NOT_FOUND"
        }
    
    images = get_cached_images(db, place_id)
    
    image_details = []
    with_categories = 0
    with_ai_tags = 0
    
    for img in images:
        has_category = bool(img.category)
        has_ai_tags = bool(img.ai_tags)
        
        if has_category:
            with_categories += 1
        if has_ai_tags:
            with_ai_tags += 1
        
        image_details.append({
            "id": img.id,
            "photo_name": img.photo_name[:50] + "..." if len(img.photo_name) > 50 else img.photo_name,
            "category": img.category,
            "has_ai_tags": has_ai_tags,
            "ai_tags_count": len(img.ai_tags) if img.ai_tags else 0,
            "gcs_url": img.gcs_url[:50] + "..." if img.gcs_url else None
        })
    
    # Check if we have complete cache
    cache_status = "INCOMPLETE"
    if len(images) >= 5:
        if with_categories >= 5 and with_ai_tags >= 5:
            cache_status = "COMPLETE"
        elif with_categories >= 5:
            cache_status = "MISSING_AI_TAGS"
        else:
            cache_status = "MISSING_CATEGORIES"
    elif len(images) > 0:
        cache_status = "INSUFFICIENT_IMAGES"
    else:
        cache_status = "NO_IMAGES"
    
    return {
        "place_id": place_id,
        "exists": True,
        "restaurant_name": restaurant.name,
        "restaurant_id": restaurant.id,
        "images": {
            "total": len(images),
            "with_categories": with_categories,
            "with_ai_tags": with_ai_tags,
            "details": image_details
        },
        "cache_status": cache_status,
        "can_serve_from_cache": cache_status == "COMPLETE"
    }


def get_cache_performance_stats(db: Session, place_id: str) -> Dict:
    """
    Get performance statistics for cache operations.
    """
    import time
    
    stats = {
        "place_id": place_id,
        "timings": {},
        "cache_hits": {},
        "cache_misses": {}
    }
    
    # Time database queries
    start = time.time()
    restaurant = db.query(Restaurant).filter(Restaurant.place_id == place_id).first()
    stats["timings"]["restaurant_query"] = time.time() - start
    
    if restaurant:
        start = time.time()
        images = get_cached_images(db, place_id)
        stats["timings"]["images_query"] = time.time() - start
        stats["cache_hits"]["images"] = len(images)
    else:
        stats["cache_misses"]["restaurant"] = True
        stats["cache_hits"]["images"] = 0
    
    return stats


def trace_request_flow(db: Session, query: str) -> Dict:
    """
    Trace the entire request flow to identify bottlenecks.
    """
    import time
    from app.services.places_service import extract_place_id_from_url
    
    trace = {
        "query": query,
        "steps": [],
        "total_time": 0,
        "place_id": None,
        "cache_status": None
    }
    
    start_total = time.time()
    
    # Step 1: Extract place_id
    step_start = time.time()
    try:
        place_id = extract_place_id_from_url(query, db=db)
        trace["place_id"] = place_id
        trace["steps"].append({
            "step": "extract_place_id",
            "time": time.time() - step_start,
            "success": True,
            "result": place_id
        })
    except Exception as e:
        trace["steps"].append({
            "step": "extract_place_id",
            "time": time.time() - step_start,
            "success": False,
            "error": str(e)
        })
        trace["total_time"] = time.time() - start_total
        return trace
    
    # Step 2: Check complete cache
    step_start = time.time()
    from app.services.database_service import get_complete_cached_restaurant_data
    cached_data = get_complete_cached_restaurant_data(db, place_id, max_images=10, max_selected=5)
    trace["steps"].append({
        "step": "check_complete_cache",
        "time": time.time() - step_start,
        "success": cached_data is not None,
        "has_cache": cached_data is not None
    })
    
    if cached_data:
        trace["cache_status"] = "COMPLETE"
        trace["total_time"] = time.time() - start_total
        return trace
    
    # Step 3: Check partial cache
    step_start = time.time()
    images = get_cached_images(db, place_id)
    trace["steps"].append({
        "step": "check_partial_cache",
        "time": time.time() - step_start,
        "images_count": len(images) if images else 0,
        "has_images": len(images) > 0 if images else False
    })
    
    # Step 4: Inspect cache details
    step_start = time.time()
    cache_inspection = inspect_restaurant_cache(db, place_id)
    trace["steps"].append({
        "step": "inspect_cache",
        "time": time.time() - step_start,
        "cache_status": cache_inspection.get("cache_status"),
        "details": cache_inspection
    })
    
    trace["cache_status"] = cache_inspection.get("cache_status")
    trace["total_time"] = time.time() - start_total
    
    return trace

