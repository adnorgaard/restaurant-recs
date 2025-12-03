from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import OperationalError, DisconnectionError
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import timedelta
import time
import traceback
import logging
from app.models.schemas import (
    ClassifyRequest, ClassifyResponse,
    UserCreate, UserResponse, Token,
    InteractionCreate, InteractionResponse,
    RecommendationResponse, RestaurantRecommendation,
    TextSearchRequest, TextSearchResponse
)
from app.database import get_db, check_database_connection
from app.models.database import User, UserRestaurantInteraction
from app.services.places_service import (
    get_place_id_by_name,
    get_place_images,
    get_place_images_with_urls,
    get_place_images_with_metadata,
    get_restaurant_name,
    extract_place_id_from_url,
    find_place_by_text,
    PlacesServiceError,
    GOOGLE_PLACES_API_KEY,
    GOOGLE_PLACES_PHOTOS_BASE_URL
)
from app.services.vision_service import (
    analyze_restaurant_images,
    categorize_images,
    select_diverse_images,
    select_images_by_quota,
    VisionServiceError
)
from app.services.database_service import (
    update_image_tags,
    get_image_by_id,
    DatabaseServiceError,
    backfill_images_from_gcs,
    mark_images_as_displayed,
    get_displayed_images,
    image_passes_quality_thresholds,
)
from app.services.auth_service import (
    create_user, authenticate_user, get_user_by_email,
    create_access_token, decode_access_token,
    AuthServiceError, ACCESS_TOKEN_EXPIRE_MINUTES
)
from app.services.embedding_service import (
    update_restaurant_embedding, EmbeddingServiceError
)
from app.services.recommendation_service import (
    find_similar_restaurants_by_tags,
    find_similar_restaurants_by_embedding,
    find_similar_restaurants_hybrid,
    search_restaurants_by_text,
    RecommendationServiceError
)

app = FastAPI(
    title="Restaurant Recommendation Engine",
    description="AI-powered restaurant classification and recommendation system",
    version="2.0.0"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_event():
    """Startup event - no blocking operations"""
    print("Starting up...")
    print("⚠️  Database connection will be checked on first use")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down...")


@app.exception_handler(OperationalError)
@app.exception_handler(DisconnectionError)
async def database_exception_handler(request: Request, exc: Exception):
    """Handle database connection errors gracefully"""
    logger.error(f"Database error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Database connection error. Please try again in a moment.",
            "error_type": type(exc).__name__
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions with detailed logging"""
    # Don't catch HTTPExceptions - let them pass through
    if isinstance(exc, HTTPException):
        raise exc
    error_traceback = traceback.format_exc()
    logger.error(f"Unhandled exception: {str(exc)}\n{error_traceback}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Internal server error: {str(exc)}",
            "error_type": type(exc).__name__
        }
    )

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Dependency to get current user
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Get the current authenticated user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
    
    email: str = payload.get("sub")
    if email is None:
        raise credentials_exception
    
    user = get_user_by_email(db, email=email)
    if user is None:
        raise credentials_exception
    
    return user


@app.get("/health")
async def health_check():
    """Health check endpoint that verifies database connectivity"""
    try:
        # Try to execute a simple query to verify database connection
        db = next(get_db())
        db.execute(text("SELECT 1"))
        db.close()
        return {
            "status": "healthy",
            "database": "connected"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "database": "disconnected",
                "error": str(e)
            }
        )


@app.get("/debug/cache/{place_id}")
async def debug_cache(place_id: str, db: Session = Depends(get_db)):
    """Debug endpoint to inspect cache status for a restaurant"""
    from app.services.debug_service import inspect_restaurant_cache, get_cache_performance_stats
    
    inspection = inspect_restaurant_cache(db, place_id)
    stats = get_cache_performance_stats(db, place_id)
    
    return {
        "inspection": inspection,
        "performance": stats
    }


@app.post("/debug/backfill-images/{place_id}")
async def debug_backfill_images(place_id: str, db: Session = Depends(get_db)):
    """
    Backfill database records for images that already exist in GCS for a given place_id.

    This is helpful when images were uploaded directly to the bucket without
    creating corresponding RestaurantImage rows.
    """
    from app.services.places_service import get_restaurant_name, PlacesServiceError

    restaurant_name: Optional[str] = None
    try:
        restaurant_name = get_restaurant_name(place_id)
    except PlacesServiceError:
        # Fallback to using place_id as name if Places API lookup fails
        restaurant_name = place_id

    try:
        result = backfill_images_from_gcs(db, place_id, restaurant_name=restaurant_name)
        return {
            "place_id": place_id,
            "restaurant_name": restaurant_name,
            "created": result["created"],
            "skipped": result["skipped"],
            "message": "Backfill completed",
        }
    except DatabaseServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/trace")
async def debug_trace(request: dict, db: Session = Depends(get_db)):
    """Debug endpoint to trace request flow and identify bottlenecks"""
    from app.services.debug_service import trace_request_flow
    
    query = request.get("query", "")
    if not query:
        raise HTTPException(status_code=400, detail="query parameter required")
    
    trace = trace_request_flow(db, query)
    return trace


@app.post("/debug/reset-cache/{place_id}")
async def debug_reset_cache(place_id: str, db: Session = Depends(get_db)):
    """
    Reset the cache for a restaurant by clearing categories and AI tags.
    This forces the next request to re-categorize and re-analyze images.
    Useful for testing the caching flow.
    """
    from app.models.database import Restaurant, RestaurantImage
    
    restaurant = db.query(Restaurant).filter(Restaurant.place_id == place_id).first()
    if not restaurant:
        raise HTTPException(status_code=404, detail=f"Restaurant with place_id {place_id} not found")
    
    # Clear categories and AI tags from all images
    images = db.query(RestaurantImage).filter(
        RestaurantImage.restaurant_id == restaurant.id
    ).all()
    
    reset_count = 0
    for img in images:
        img.category = None
        img.ai_tags = None
        reset_count += 1
    
    db.commit()
    
    return {
        "place_id": place_id,
        "restaurant_name": restaurant.name,
        "images_reset": reset_count,
        "message": "Cache cleared. Next request will re-categorize and re-analyze images."
    }


@app.post("/debug/force-recategorize/{place_id}")
async def debug_force_recategorize(place_id: str, db: Session = Depends(get_db)):
    """
    Force re-categorization of images for a restaurant.
    Downloads images from GCS and runs AI categorization.
    """
    import requests as http_requests
    from app.models.database import Restaurant, RestaurantImage
    
    restaurant = db.query(Restaurant).filter(Restaurant.place_id == place_id).first()
    if not restaurant:
        raise HTTPException(status_code=404, detail=f"Restaurant with place_id {place_id} not found")
    
    images = db.query(RestaurantImage).filter(
        RestaurantImage.restaurant_id == restaurant.id
    ).all()
    
    if not images:
        raise HTTPException(status_code=404, detail="No images found for this restaurant")
    
    results = []
    for img in images:
        try:
            # Download image from GCS
            response = http_requests.get(img.gcs_url, timeout=10)
            response.raise_for_status()
            image_bytes = response.content
            
            # Categorize with AI
            from app.services.vision_service import categorize_image
            category = categorize_image(image_bytes)
            
            # Update database
            img.category = category
            results.append({
                "photo_name": img.photo_name[:50] + "..." if len(img.photo_name) > 50 else img.photo_name,
                "category": category,
                "status": "success"
            })
        except Exception as e:
            results.append({
                "photo_name": img.photo_name[:50] + "..." if len(img.photo_name) > 50 else img.photo_name,
                "category": None,
                "status": f"error: {str(e)}"
            })
    
    db.commit()
    
    return {
        "place_id": place_id,
        "restaurant_name": restaurant.name,
        "results": results,
        "message": f"Re-categorized {len([r for r in results if r['status'] == 'success'])} images"
    }


@app.get("/api/places/photo/{photo_path:path}")
async def proxy_place_photo(photo_path: str, maxWidthPx: int = 800):
    """
    Proxy endpoint for Google Places API photos.
    This is needed because the Places API (New) requires authentication headers
    which browsers cannot send directly.
    """
    import requests
    
    # Reconstruct the full photo path
    # photo_path will be like "places/ChIJ.../photos/Aap_uEA..."
    url = f"{GOOGLE_PLACES_PHOTOS_BASE_URL}/{photo_path}/media"
    headers = {
        "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY
    }
    params = {
        "maxWidthPx": maxWidthPx
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        # Return the image with appropriate content type
        return Response(
            content=response.content,
            media_type=response.headers.get("Content-Type", "image/jpeg")
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=404, detail=f"Failed to fetch photo: {str(e)}")


class FindPlaceIdRequest(BaseModel):
    query: str  # Restaurant name, address, or Google Maps URL


class FindPlaceIdResponse(BaseModel):
    place_id: str
    restaurant_name: str
    message: str


@app.post("/find-place-id", response_model=FindPlaceIdResponse)
async def find_place_id(request: FindPlaceIdRequest):
    """
    Helper endpoint to find the place_id for a restaurant.
    Accepts restaurant name, address, or Google Maps URL.
    """
    try:
        # Get database session for cache checking
        db = next(get_db())
        try:
            # Try to get place_id (handles URLs and names) - check database first
            place_id = extract_place_id_from_url(request.query, db=db)
            
            # Get restaurant name from database if available, otherwise from API
            from app.models.database import Restaurant
            restaurant = db.query(Restaurant).filter(Restaurant.place_id == place_id).first()
            restaurant_name = restaurant.name if restaurant else get_restaurant_name(place_id)
        finally:
            db.close()
        
        return FindPlaceIdResponse(
            place_id=place_id,
            restaurant_name=restaurant_name,
            message=f"Found place_id for: {restaurant_name}"
        )
    except PlacesServiceError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/classify", response_model=ClassifyResponse)
async def classify_restaurant(request: ClassifyRequest, db: Session = Depends(get_db)):
    """
    Classify a restaurant based on images from Google Places API.
    Uses cached images when available to save API credits.
    
    Caching strategy:
    1. First request: Fetch images, categorize with AI, analyze with AI, cache everything
    2. Subsequent requests: Return cached data immediately (no API calls)
    
    Accepts either:
    - place_id: Direct Google Places place_id
    - name + location: Restaurant name and optional location for search
    """
    import requests as http_requests
    
    try:
        # Get place_id - check database first to avoid API calls
        if request.place_id:
            place_id = request.place_id
        elif request.name:
            # Try database first using semantic search
            search_query = request.name
            if request.location:
                search_query = f"{request.name} {request.location}"
            
            from app.services.database_service import find_restaurant_by_text
            cached_place_id = find_restaurant_by_text(db, search_query, min_similarity=0.75)
            
            if cached_place_id:
                print(f"[DEBUG] Found restaurant in database: {search_query} -> {cached_place_id}")
                place_id = cached_place_id
            else:
                # Not in database, use Places API
                if not request.location:
                    raise HTTPException(
                        status_code=400,
                        detail="location is required when using name-based search"
                    )
                place_id = get_place_id_by_name(request.name, request.location)
        else:
            raise HTTPException(
                status_code=400,
                detail="Either place_id or name must be provided"
            )
        
        # Check for complete cached data first - this allows instant responses
        # Quota: 2 exterior, 8 food, 8 interior, 2 drink = 20 total
        from app.services.database_service import (
            get_complete_cached_restaurant_data, get_cached_images, 
            get_cached_categories_and_tags, get_cached_restaurant_analysis
        )
        image_quota = {"food": 10, "interior": 10}
        cached_data = get_complete_cached_restaurant_data(db, place_id, max_images=50, max_selected=20, quota=image_quota, max_bar=2)
        
        if cached_data:
            # All data is cached - return immediately without any API calls
            print(f"[PERF] ✅ Returning fully cached data for {place_id}")
            return ClassifyResponse(
                restaurant_name=cached_data['restaurant_name'],
                tags=cached_data['analysis']['tags'],
                description=cached_data['analysis']['description']
            )
        
        # Not fully cached - build the cache
        print(f"[DEBUG] Building cache for {place_id}...")
        
        # Get restaurant name
        from app.models.database import Restaurant
        restaurant_record = db.query(Restaurant).filter(Restaurant.place_id == place_id).first()
        if restaurant_record:
            restaurant_name = restaurant_record.name
        else:
            restaurant_name = get_restaurant_name(place_id)
        
        # Get or fetch images (up to 50 for categorization, display 20)
        cached_images = get_cached_images(db, place_id, max_images=50)
        
        if cached_images and len(cached_images) >= 20:
            all_image_urls = [img.gcs_url for img in cached_images]
            photo_names = [img.photo_name for img in cached_images]
        else:
            # Fetch from Places API
            all_images, all_image_urls, photo_names = get_place_images_with_metadata(place_id, max_images=50, min_required=20, db=db)
            cached_images = get_cached_images(db, place_id, max_images=50)
        
        # Check which images need categorization
        cached_categories, _ = get_cached_categories_and_tags(db, place_id, photo_names)
        images_missing_categories = [pn for pn in photo_names if pn not in cached_categories]
        
        # Categorize images that don't have categories
        if images_missing_categories:
            print(f"[DEBUG] Categorizing {len(images_missing_categories)} images...")
            images_to_categorize = []
            photo_names_to_categorize = []
            
            for i, pn in enumerate(photo_names):
                if pn in images_missing_categories and i < len(all_image_urls):
                    try:
                        response = http_requests.get(all_image_urls[i], timeout=10)
                        response.raise_for_status()
                        images_to_categorize.append(response.content)
                        photo_names_to_categorize.append(pn)
                    except Exception as e:
                        print(f"[WARNING] Failed to download {pn[:30]}...: {str(e)}")
            
            if images_to_categorize:
                cache_quota = {"food": 10, "interior": 10}
                categorized_results = categorize_images(
                    images_to_categorize, db=db, place_id=place_id, 
                    photo_names=photo_names_to_categorize,
                    cache_quota=cache_quota, max_bar=2
                )
                for j, pn in enumerate(photo_names_to_categorize):
                    if j < len(categorized_results):
                        _, category = categorized_results[j]
                        cached_categories[pn] = category
        
        # Build categorized list for selection
        # Quota: 2 exterior, 8 food, 8 interior, 2 drink = 20 total
        categorized_images = [(b"", cached_categories.get(pn, "other")) for pn in photo_names]
        image_quota = {"food": 10, "interior": 10}
        selected_images, selected_indices = select_images_by_quota(categorized_images, quota=image_quota, max_bar=2, randomize=False)
        
        # Run AI analysis on all selected images
        analysis_indices = selected_indices  # Analyze ALL selected images
        selected_photo_names = [photo_names[i] for i in analysis_indices if i < len(photo_names)]
        
        # Mark selected images as displayed in the database
        if cached_images:
            photo_name_to_id = {img.photo_name: img.id for img in cached_images}
            selected_image_ids = [
                photo_name_to_id[pn] for pn in selected_photo_names 
                if pn in photo_name_to_id
            ]
            if selected_image_ids:
                mark_images_as_displayed(db, place_id, selected_image_ids, reset_others=True)
        
        # Check for cached AI analysis
        analysis = get_cached_restaurant_analysis(db, place_id, selected_photo_names)
        
        if not analysis:
            print(f"[DEBUG] Running AI analysis...")
            selected_images_bytes = []
            valid_photo_names = []
            
            for idx in analysis_indices:
                if idx < len(all_image_urls):
                    try:
                        response = http_requests.get(all_image_urls[idx], timeout=10)
                        response.raise_for_status()
                        selected_images_bytes.append(response.content)
                        valid_photo_names.append(photo_names[idx])
                    except Exception as e:
                        print(f"[WARNING] Failed to download image {idx}: {str(e)}")
            
            if selected_images_bytes:
                analysis = analyze_restaurant_images(
                    selected_images_bytes, db=db, place_id=place_id,
                    photo_names=valid_photo_names
                )
            else:
                raise HTTPException(status_code=500, detail="Failed to download images for analysis")
        
        # Generate embedding
        try:
            from app.services.database_service import get_or_create_restaurant
            restaurant_obj = get_or_create_restaurant(db, place_id, restaurant_name)
            if restaurant_obj:
                try:
                    update_restaurant_embedding(db, restaurant_obj.id)
                except Exception as e:
                    print(f"[WARNING] Failed to generate embedding: {str(e)}")
        except Exception as e:
            print(f"[WARNING] Failed to generate embedding: {str(e)}")
        
        return ClassifyResponse(
            restaurant_name=restaurant_name,
            tags=analysis["tags"],
            description=analysis["description"]
        )
        
    except PlacesServiceError as e:
        error_msg = str(e) if str(e) else f"PlacesServiceError: {type(e).__name__}"
        logger.error(f"PlacesServiceError in classify: {error_msg}", exc_info=True)
        raise HTTPException(status_code=404, detail=error_msg)
    except VisionServiceError as e:
        error_msg = str(e) if str(e) else f"VisionServiceError: {type(e).__name__}"
        logger.error(f"VisionServiceError in classify: {error_msg}", exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)
    except HTTPException:
        raise
    except Exception as e:
        error_traceback = traceback.format_exc()
        error_msg = str(e) if str(e) else f"{type(e).__name__} (no error message)"
        logger.error(f"Unexpected error in classify_restaurant: {error_msg}\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {error_msg}")


class TestRequest(BaseModel):
    url: str


class ImageMetadata(BaseModel):
    url: str
    category: str

class TestResponse(BaseModel):
    restaurant_name: str
    images: List[ImageMetadata]
    tags: List[str]
    description: str


@app.post("/test", response_model=TestResponse)
async def test_restaurant(request: TestRequest, db: Session = Depends(get_db)):
    """
    Test endpoint that accepts a restaurant URL and returns images and classification.
    Uses cached images when available to save API credits.
    
    Caching strategy:
    1. If complete cache exists (images + categories + AI tags) -> instant response
    2. If images cached but missing categories -> download from GCS, categorize, cache
    3. If images cached with categories but missing AI tags -> download selected, analyze, cache
    4. If no images cached -> fetch from Google API, cache everything
    """
    try:
        import requests as http_requests
        request_start = time.time()
        
        # Extract place_id from URL - check database first to avoid API calls
        step_start = time.time()
        place_id = extract_place_id_from_url(request.url, db=db)
        print(f"[PERF] extract_place_id took {time.time() - step_start:.3f}s")
        print(f"[DEBUG] Place ID: {place_id}")
        
        # Check for complete cached data first - this allows instant responses
        # Quota: 2 exterior, 8 food, 8 interior, 2 drink = 20 total
        step_start = time.time()
        from app.services.database_service import get_complete_cached_restaurant_data
        image_quota = {"food": 10, "interior": 10}
        cached_data = get_complete_cached_restaurant_data(db, place_id, max_images=50, max_selected=20, quota=image_quota, max_bar=2)
        print(f"[PERF] check_complete_cache took {time.time() - step_start:.3f}s")
        
        if cached_data:
            total_time = time.time() - request_start
            print(f"[PERF] TOTAL REQUEST TIME (fully cached): {total_time:.3f}s")
            print(f"[DEBUG] Found complete cached data for {place_id}")
            # All data is cached - return immediately without any API calls
            image_metadata = [
                ImageMetadata(
                    url=img['gcs_url'],
                    category=img['category']
                )
                for img in cached_data['images']
            ]
            
            return TestResponse(
                restaurant_name=cached_data['restaurant_name'],
                images=image_metadata,
                tags=cached_data['analysis']['tags'],
                description=cached_data['analysis']['description']
            )
        
        print(f"[DEBUG] No complete cached data, checking what we need...")
        
        # Get restaurant name from database if available (avoid API call)
        step_start = time.time()
        from app.models.database import Restaurant
        restaurant_record = db.query(Restaurant).filter(Restaurant.place_id == place_id).first()
        if restaurant_record:
            restaurant_name = restaurant_record.name
            print(f"[PERF] get_restaurant_name (from DB) took {time.time() - step_start:.3f}s")
        else:
            step_api = time.time()
            restaurant_name = get_restaurant_name(place_id)
            print(f"[PERF] get_restaurant_name (API call) took {time.time() - step_api:.3f}s")
        print(f"[PERF] restaurant_name lookup took {time.time() - step_start:.3f}s")
        
        # Check if we have cached images
        step_start = time.time()
        from app.services.database_service import get_cached_images, get_cached_categories_and_tags
        cached_images = get_cached_images(db, place_id, max_images=50)
        print(f"[PERF] get_cached_images took {time.time() - step_start:.3f}s")
        print(f"[DEBUG] Found {len(cached_images) if cached_images else 0} cached images")
        
        all_image_urls = []
        photo_names = []
        categorized_images = []
        
        if cached_images and len(cached_images) >= 20:
            print(f"[DEBUG] Using {len(cached_images)} cached images from database")
            all_image_urls = [img.gcs_url for img in cached_images]
            photo_names = [img.photo_name for img in cached_images]
            
            # Get cached categories
            cached_categories, cached_ai_tags = get_cached_categories_and_tags(db, place_id, photo_names)
            print(f"[DEBUG] Cached categories: {len(cached_categories)}, Cached AI tags: {len(cached_ai_tags)}")
            
            # Check how many images have valid categories (not None)
            images_with_categories = sum(1 for pn in photo_names if pn in cached_categories and cached_categories[pn])
            print(f"[DEBUG] Images with valid categories: {images_with_categories}")
            
            if images_with_categories >= 20:
                # All categories are cached - use them directly (FAST PATH)
                print(f"[DEBUG] Using cached categories (fast path)")
                categorized_images = [(b"", cached_categories.get(pn, "other")) for pn in photo_names]
            else:
                # Missing categories - need to download and categorize
                print(f"[DEBUG] Missing categories, downloading images from GCS for categorization...")
                step_start = time.time()
                
                all_images = []
                for img in cached_images:
                    try:
                        response = http_requests.get(img.gcs_url, timeout=10)
                        response.raise_for_status()
                        all_images.append(response.content)
                    except Exception as e:
                        print(f"[WARNING] Failed to download {img.gcs_url[:50]}...: {str(e)}")
                        all_images.append(b"")  # Placeholder for failed downloads
                
                print(f"[PERF] Downloaded {len([i for i in all_images if i])} images from GCS in {time.time() - step_start:.3f}s")
                
                # Categorize images - this will use cached categories where available
                # and only call AI for images without categories
                cache_quota = {"food": 10, "interior": 10}
                categorized_images = categorize_images(all_images, db=db, place_id=place_id, photo_names=photo_names, cache_quota=cache_quota, max_bar=2)
                print(f"[DEBUG] Categorization complete")
        else:
            print(f"[DEBUG] Not enough cached images ({len(cached_images) if cached_images else 0} < 20), fetching from Places API...")
            # Not enough cache - fetch from API (will combine with existing cached)
            all_images, all_image_urls, photo_names = get_place_images_with_metadata(place_id, max_images=50, min_required=20, db=db)
            
            # Refresh cached_images after potentially adding new ones
            cached_images = get_cached_images(db, place_id, max_images=50)
            
            if all_images:
                # Categorize and cache
                cache_quota = {"food": 10, "interior": 10}
                categorized_images = categorize_images(all_images, db=db, place_id=place_id, photo_names=photo_names, cache_quota=cache_quota, max_bar=2)
            else:
                # get_place_images_with_metadata returned empty bytes (images are already cached)
                if cached_images:
                    all_image_urls = [img.gcs_url for img in cached_images]
                    photo_names = [img.photo_name for img in cached_images]
                    
                    # Check for cached categories first
                    cached_categories, _ = get_cached_categories_and_tags(db, place_id, photo_names)
                    images_with_categories = sum(1 for pn in photo_names if pn in cached_categories and cached_categories[pn])
                    
                    # Only need 20 images with categories (not ALL images)
                    if images_with_categories >= 20:
                        # Enough categories are cached - use them directly (FAST PATH)
                        print(f"[DEBUG] Using cached categories for {images_with_categories} images (fast path)")
                        categorized_images = [(b"", cached_categories.get(pn, "other")) for pn in photo_names]
                    else:
                        # Download and categorize only what we need
                        print(f"[DEBUG] Only {images_with_categories} images have categories, need to categorize more...")
                        all_images = []
                        for img in cached_images:
                            try:
                                response = http_requests.get(img.gcs_url, timeout=10)
                                response.raise_for_status()
                                all_images.append(response.content)
                            except Exception as e:
                                all_images.append(b"")
                        cache_quota = {"food": 10, "interior": 10}
                        categorized_images = categorize_images(all_images, db=db, place_id=place_id, photo_names=photo_names, cache_quota=cache_quota, max_bar=2)
        
        # Select images based on quota: 2 exterior, 8 food, 8 interior, 2 drink = 20 total
        image_quota = {"food": 10, "interior": 10}
        selected_images, selected_indices = select_images_by_quota(categorized_images, quota=image_quota, max_bar=2, randomize=False)
        
        # Get photo_names for selected images
        selected_photo_names = [photo_names[i] for i in selected_indices if i < len(photo_names)]
        print(f"[DEBUG] Selected {len(selected_indices)} images for display (before quality filter)")
        
        # Apply quality filtering - only mark images that pass quality thresholds as displayed
        if cached_images:
            photo_name_to_image = {img.photo_name: img for img in cached_images}
            
            # Filter to only quality-passing images
            quality_passing_photo_names = []
            quality_failed_count = 0
            for pn in selected_photo_names:
                if pn in photo_name_to_image:
                    img = photo_name_to_image[pn]
                    if image_passes_quality_thresholds(img):
                        quality_passing_photo_names.append(pn)
                    else:
                        quality_failed_count += 1
                        print(f"[DEBUG] Image {pn} failed quality (people={img.people_confidence_score}, lighting={img.lighting_confidence_score})")
            
            if quality_failed_count > 0:
                print(f"[DEBUG] Quality filter removed {quality_failed_count} images, {len(quality_passing_photo_names)} passed")
            
            # Update selected to only include quality-passing images
            selected_photo_names = quality_passing_photo_names
            selected_indices = [i for i in selected_indices if i < len(photo_names) and photo_names[i] in quality_passing_photo_names]
            
            # Mark selected images as displayed in the database
            selected_image_ids = [
                photo_name_to_image[pn].id for pn in selected_photo_names 
                if pn in photo_name_to_image
            ]
            if selected_image_ids:
                mark_images_as_displayed(db, place_id, selected_image_ids, reset_others=True)
                print(f"[DEBUG] Marked {len(selected_image_ids)} quality-passing images as is_displayed=True")
        
        # Run AI analysis on all selected images
        analysis_indices = selected_indices  # Analyze ALL selected images
        analysis_photo_names = [photo_names[i] for i in analysis_indices if i < len(photo_names)]
        
        # Check for cached AI analysis
        from app.services.database_service import get_cached_restaurant_analysis
        analysis = get_cached_restaurant_analysis(db, place_id, analysis_photo_names)
        
        if not analysis:
            print(f"[DEBUG] No cached AI analysis, downloading selected images for analysis...")
            # Need to run AI analysis - download all 20 selected images for analysis
            selected_images_bytes = []
            valid_photo_names = []
            download_errors = []
            
            for idx in analysis_indices:
                if idx < len(all_image_urls) and idx < len(photo_names):
                    try:
                        response = http_requests.get(all_image_urls[idx], timeout=10)
                        response.raise_for_status()
                        selected_images_bytes.append(response.content)
                        valid_photo_names.append(photo_names[idx])
                    except Exception as e:
                        error_msg = f"Failed to download image {idx} from {all_image_urls[idx][:50]}...: {str(e)}"
                        print(f"[WARNING] {error_msg}")
                        download_errors.append(error_msg)
                        continue
            
            if selected_images_bytes:
                analysis = analyze_restaurant_images(
                    selected_images_bytes,
                    db=db,
                    place_id=place_id,
                    photo_names=valid_photo_names
                )
                print(f"[DEBUG] AI analysis complete and cached")
            else:
                # Provide helpful error message about GCS download failures
                error_detail = "Failed to analyze images: Could not download images from Google Cloud Storage"
                if download_errors:
                    error_detail += f". Errors: {'; '.join(download_errors[:2])}"
                error_detail += ". This usually means the GCS bucket or images are not publicly accessible. Please check your GCS configuration."
                logger.error(error_detail)
                raise HTTPException(status_code=500, detail=error_detail)
        else:
            print(f"[DEBUG] Using cached AI analysis")
        
        # Build image metadata with URLs and categories
        # Use photo_names to match URLs to avoid index mismatches
        image_metadata = []
        seen_urls = set()  # Deduplicate images
        seen_photo_names = set()  # Also dedupe by photo_name
        
        # Build a mapping from photo_name to URL for reliable lookup
        photo_name_to_url = {}
        if cached_images:
            for img in cached_images:
                photo_name_to_url[img.photo_name] = img.gcs_url
        else:
            for i, pn in enumerate(photo_names):
                if i < len(all_image_urls):
                    photo_name_to_url[pn] = all_image_urls[i]
        
        print(f"[DEBUG] Building image metadata: {len(selected_indices)} selected, {len(photo_name_to_url)} URLs available")
        
        for idx in selected_indices:
            if idx < len(categorized_images) and idx < len(photo_names):
                pn = photo_names[idx]
                url = photo_name_to_url.get(pn)
                _, category = categorized_images[idx]
                
                # Skip if no URL, already seen, or excluded category
                if not url:
                    print(f"[DEBUG] Skipping idx {idx}: no URL for photo_name")
                    continue
                if url in seen_urls or pn in seen_photo_names:
                    print(f"[DEBUG] Skipping idx {idx}: duplicate URL or photo_name")
                    continue
                if category in ("menu", "other", "skipped"):
                    print(f"[DEBUG] Skipping idx {idx}: excluded category '{category}'")
                    continue
                    
                image_metadata.append(ImageMetadata(
                    url=url,
                    category=category
                ))
                seen_urls.add(url)
                seen_photo_names.add(pn)
        
        print(f"[DEBUG] Final image_metadata count: {len(image_metadata)}")
        
        # Generate embedding for the restaurant (async, don't fail if it errors)
        try:
            from app.services.database_service import get_or_create_restaurant
            restaurant_obj = get_or_create_restaurant(db, place_id, restaurant_name)
            if restaurant_obj:
                try:
                    update_restaurant_embedding(db, restaurant_obj.id)
                except Exception as e:
                    print(f"Warning: Failed to generate embedding for restaurant {restaurant_obj.id}: {str(e)}")
        except Exception as e:
            print(f"Warning: Failed to generate embedding: {str(e)}")
        
        total_time = time.time() - request_start
        print(f"[PERF] TOTAL REQUEST TIME: {total_time:.3f}s")
        
        return TestResponse(
            restaurant_name=restaurant_name,
            images=image_metadata,
            tags=analysis["tags"],
            description=analysis["description"]
        )
        
    except PlacesServiceError as e:
        error_msg = str(e) if str(e) else f"PlacesServiceError: {type(e).__name__}"
        logger.error(f"PlacesServiceError: {error_msg}", exc_info=True)
        raise HTTPException(status_code=404, detail=error_msg)
    except VisionServiceError as e:
        error_msg = str(e) if str(e) else f"VisionServiceError: {type(e).__name__}"
        logger.error(f"VisionServiceError: {error_msg}", exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)
    except HTTPException as e:
        if not e.detail:
            error_msg = f"{type(e).__name__} (status {e.status_code})"
            logger.error(f"HTTPException without detail in test_restaurant: {error_msg}")
            raise HTTPException(status_code=e.status_code, detail=error_msg)
        raise
    except Exception as e:
        error_traceback = traceback.format_exc()
        error_msg = str(e) if str(e) else f"{type(e).__name__} (no error message)"
        logger.error(f"Unexpected error in test_restaurant: {error_msg}\n{error_traceback}")
        print(f"\n{'='*60}")
        print(f"ERROR in test_restaurant endpoint:")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {error_msg}")
        print(f"Traceback:\n{error_traceback}")
        print(f"{'='*60}\n")
        raise HTTPException(status_code=500, detail=f"Internal server error: {error_msg}")


class UpdateImageTagsRequest(BaseModel):
    category: Optional[str] = None
    ai_tags: Optional[List[str]] = None


class UpdateImageTagsResponse(BaseModel):
    id: int
    restaurant_id: int
    photo_name: str
    gcs_url: str
    category: Optional[str]
    ai_tags: Optional[List[str]]
    message: str


@app.put("/api/images/{image_id}/tags", response_model=UpdateImageTagsResponse)
async def update_image_tags_endpoint(
    image_id: int,
    request: UpdateImageTagsRequest,
    db: Session = Depends(get_db)
):
    """
    Update tags for a specific image.
    
    Both category and ai_tags are optional - only provided fields will be updated.
    
    Args:
        image_id: ID of the image to update
        request: UpdateImageTagsRequest with optional category and/or ai_tags
        db: Database session
        
    Returns:
        Updated image information
    """
    try:
        # Validate category if provided
        if request.category is not None:
            valid_categories = ["interior", "exterior", "food", "menu", "bar", "other"]
            if request.category not in valid_categories:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid category. Must be one of: {', '.join(valid_categories)}"
                )
        
        # Update the image tags
        updated_image = update_image_tags(
            db,
            image_id,
            category=request.category,
            ai_tags=request.ai_tags
        )
        
        return UpdateImageTagsResponse(
            id=updated_image.id,
            restaurant_id=updated_image.restaurant_id,
            photo_name=updated_image.photo_name,
            gcs_url=updated_image.gcs_url,
            category=updated_image.category,
            ai_tags=updated_image.ai_tags,
            message="Image tags updated successfully"
        )
        
    except DatabaseServiceError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ============================================================================
# Authentication Endpoints
# ============================================================================

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user.
    
    Args:
        user_data: User registration data (email and password)
        db: Database session
        
    Returns:
        Created user information
    """
    try:
        user = create_user(db, email=user_data.email, password=user_data.password)
        return UserResponse(
            id=user.id,
            email=user.email,
            is_active=user.is_active,
            created_at=user.created_at.isoformat()
        )
    except AuthServiceError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {str(e)}")


@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Login endpoint that returns a JWT access token.
    
    Args:
        form_data: OAuth2 form data (username=email, password)
        db: Database session
        
    Returns:
        JWT access token
    """
    user = authenticate_user(db, email=form_data.username, password=form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        is_active=current_user.is_active,
        created_at=current_user.created_at.isoformat()
    )


# ============================================================================
# User-Restaurant Interaction Endpoints
# ============================================================================

@app.post("/api/interactions", response_model=InteractionResponse, status_code=status.HTTP_201_CREATED)
async def create_interaction(
    interaction: InteractionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a user-restaurant interaction (like, dislike, or rating).
    
    Args:
        interaction: Interaction data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Created interaction information
    """
    try:
        # Validate interaction type
        valid_types = ["like", "dislike", "rating"]
        if interaction.interaction_type not in valid_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"interaction_type must be one of: {', '.join(valid_types)}"
            )
        
        # Validate rating if interaction_type is rating
        if interaction.interaction_type == "rating":
            if interaction.rating is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="rating is required when interaction_type is 'rating'"
                )
            if not (1.0 <= interaction.rating <= 5.0):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="rating must be between 1.0 and 5.0"
                )
        
        # Check if interaction already exists
        existing = db.query(UserRestaurantInteraction).filter(
            UserRestaurantInteraction.user_id == current_user.id,
            UserRestaurantInteraction.restaurant_id == interaction.restaurant_id
        ).first()
        
        if existing:
            # Update existing interaction
            existing.interaction_type = interaction.interaction_type
            existing.rating = interaction.rating
            db.commit()
            db.refresh(existing)
            return InteractionResponse(
                id=existing.id,
                user_id=existing.user_id,
                restaurant_id=existing.restaurant_id,
                interaction_type=existing.interaction_type,
                rating=existing.rating,
                created_at=existing.created_at.isoformat()
            )
        else:
            # Create new interaction
            new_interaction = UserRestaurantInteraction(
                user_id=current_user.id,
                restaurant_id=interaction.restaurant_id,
                interaction_type=interaction.interaction_type,
                rating=interaction.rating
            )
            db.add(new_interaction)
            db.commit()
            db.refresh(new_interaction)
            return InteractionResponse(
                id=new_interaction.id,
                user_id=new_interaction.user_id,
                restaurant_id=new_interaction.restaurant_id,
                interaction_type=new_interaction.interaction_type,
                rating=new_interaction.rating,
                created_at=new_interaction.created_at.isoformat()
            )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {str(e)}")


# ============================================================================
# Recommendation Endpoints
# ============================================================================

@app.get("/api/restaurants/{restaurant_id}/similar", response_model=RecommendationResponse)
async def get_similar_restaurants(
    restaurant_id: int,
    method: str = "hybrid",
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get restaurants similar to a given restaurant.
    
    Args:
        restaurant_id: ID of the reference restaurant
        method: Recommendation method - 'tags', 'embedding', or 'hybrid' (default)
        limit: Maximum number of recommendations
        db: Database session
        
    Returns:
        List of similar restaurants with similarity scores
    """
    try:
        if method == "tags":
            results = find_similar_restaurants_by_tags(db, restaurant_id, limit=limit)
        elif method == "embedding":
            results = find_similar_restaurants_by_embedding(db, restaurant_id, limit=limit)
        elif method == "hybrid":
            results = find_similar_restaurants_hybrid(db, restaurant_id, limit=limit)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="method must be one of: 'tags', 'embedding', 'hybrid'"
            )
        
        recommendations = [
            RestaurantRecommendation(
                id=restaurant.id,
                place_id=restaurant.place_id,
                name=restaurant.name,
                similarity_score=score
            )
            for restaurant, score in results
        ]
        
        return RecommendationResponse(
            recommendations=recommendations,
            method=method,
            reference_restaurant_id=restaurant_id
        )
    except RecommendationServiceError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {str(e)}")


@app.post("/api/search", response_model=TextSearchResponse)
async def search_restaurants_by_text(
    search_request: TextSearchRequest,
    db: Session = Depends(get_db)
):
    """
    Search restaurants using natural language text query (semantic search).
    
    Args:
        search_request: Search query and parameters
        db: Database session
        
    Returns:
        List of restaurants matching the query with similarity scores
    """
    try:
        results = search_restaurants_by_text(
            db,
            search_request.query,
            limit=search_request.limit,
            min_similarity=search_request.min_similarity
        )
        
        recommendations = [
            RestaurantRecommendation(
                id=restaurant.id,
                place_id=restaurant.place_id,
                name=restaurant.name,
                similarity_score=score
            )
            for restaurant, score in results
        ]
        
        return TextSearchResponse(
            results=recommendations,
            query=search_request.query
        )
    except RecommendationServiceError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {str(e)}")


@app.post("/api/restaurants/{restaurant_id}/generate-embedding", response_model=dict)
async def generate_restaurant_embedding_endpoint(
    restaurant_id: int,
    db: Session = Depends(get_db)
):
    """
    Generate and update embedding for a restaurant.
    This is useful for populating embeddings for existing restaurants.
    
    Args:
        restaurant_id: ID of the restaurant
        db: Database session
        
    Returns:
        Success message
    """
    try:
        restaurant = update_restaurant_embedding(db, restaurant_id)
        return {
            "message": "Embedding generated successfully",
            "restaurant_id": restaurant.id,
            "restaurant_name": restaurant.name
        }
    except EmbeddingServiceError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {str(e)}")


# ============================================================================
# Web Interface
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def test_interface():
    """Simple HTML interface for testing - served from templates/index.html"""
    import os
    template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates", "index.html")
    with open(template_path, "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

