"""
Bulk Processing Service

Handles bulk processing of restaurants with support for:
- Multiple input sources (CSV, JSON, database)
- Processing modes (net-new, refresh-stale, force)
- Parallel execution with rate limiting
- Progress tracking and error handling
- Provider abstraction (Google Places, SerpApi)
"""

import asyncio
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor
import requests

from sqlalchemy.orm import Session

from app.models.database import Restaurant, RestaurantImage
from app.config.ai_versions import (
    get_active_version,
    CATEGORY_VERSION,
    TAGS_VERSION,
    DESCRIPTION_VERSION,
    EMBEDDING_VERSION,
)
from app.services.database_service import (
    get_or_create_restaurant,
    get_cached_images,
    get_stale_restaurants,
    get_stale_images,
    get_restaurants_missing_data,
    get_images_missing_data,
    store_restaurant_description,
    store_ai_tags_for_images,
    update_image_tags,
    cache_images,
    delete_restaurant_images,
)
from app.services.places_service import (
    get_place_id_by_name,
    get_place_images_with_metadata,
    get_restaurant_name,
    PlacesServiceError,
)
from app.services.vision_service import (
    categorize_image,
    analyze_restaurant_images,
    select_images_by_quota,
    VisionServiceError,
)
from app.services.embedding_service import (
    update_restaurant_embedding,
    EmbeddingServiceError,
)
from app.services.photo_service import (
    PhotoService,
    get_cost_tracker,
    reset_cost_tracker,
    log_cost_summary,
)
from app.services.providers import ProviderError


class ProcessingMode(Enum):
    """Processing modes for bulk operations."""
    AUTO = "auto"                # Smart default: process missing OR outdated data
    NET_NEW = "net-new"          # Only process missing data (legacy)
    REFRESH_STALE = "refresh-stale"  # Only process outdated versions (legacy)
    FORCE = "force"              # Regenerate everything


class Component(Enum):
    """Components that can be processed."""
    CATEGORY = "category"
    TAGS = "tags"
    DESCRIPTION = "description"
    EMBEDDING = "embedding"
    QUALITY = "quality"  # Image quality scoring (people, lighting, blur)
    ALL = "all"


@dataclass
class RestaurantInput:
    """Input data for a restaurant to process."""
    place_id: Optional[str] = None
    name: Optional[str] = None
    location: Optional[str] = None
    
    def __post_init__(self):
        if not self.place_id and not self.name:
            raise ValueError("Either place_id or name must be provided")


@dataclass
class ProcessingResult:
    """Result of processing a single restaurant."""
    place_id: str
    name: str
    success: bool
    components_processed: List[str] = field(default_factory=list)
    components_skipped: List[str] = field(default_factory=list)
    error: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class BulkProcessingReport:
    """Summary report for bulk processing operation."""
    total: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    results: List[ProcessingResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Cost tracking
    total_cost: float = 0.0
    cost_by_provider: Dict[str, float] = field(default_factory=dict)
    api_calls_by_provider: Dict[str, int] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "processed": self.processed,
            "skipped": self.skipped,
            "failed": self.failed,
            "duration_seconds": self.duration_seconds,
            "total_cost": self.total_cost,
            "cost_by_provider": self.cost_by_provider,
            "api_calls_by_provider": self.api_calls_by_provider,
            "results": [
                {
                    "place_id": r.place_id,
                    "name": r.name,
                    "success": r.success,
                    "components_processed": r.components_processed,
                    "components_skipped": r.components_skipped,
                    "error": r.error,
                    "duration_seconds": r.duration_seconds,
                }
                for r in self.results
            ]
        }


class RateLimiter:
    """Async rate limiter using semaphores."""
    
    def __init__(
        self,
        openai_limit: int = 3,
        places_limit: int = 5,
        overall_limit: int = 5
    ):
        self.openai_semaphore = asyncio.Semaphore(openai_limit)
        self.places_semaphore = asyncio.Semaphore(places_limit)
        self.overall_semaphore = asyncio.Semaphore(overall_limit)
    
    async def acquire_openai(self):
        await self.openai_semaphore.acquire()
    
    def release_openai(self):
        self.openai_semaphore.release()
    
    async def acquire_places(self):
        await self.places_semaphore.acquire()
    
    def release_places(self):
        self.places_semaphore.release()
    
    async def acquire_overall(self):
        await self.overall_semaphore.acquire()
    
    def release_overall(self):
        self.overall_semaphore.release()


# =============================================================================
# Input Parsing
# =============================================================================

def parse_csv_input(file_path: str) -> List[RestaurantInput]:
    """
    Parse restaurant list from CSV file.
    
    Expected columns: place_id (optional), name, location (optional)
    """
    restaurants = []
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Normalize column names (case-insensitive)
            normalized = {k.lower().strip(): v.strip() for k, v in row.items() if v}
            
            place_id = normalized.get('place_id') or normalized.get('placeid')
            name = normalized.get('name') or normalized.get('restaurant_name')
            location = normalized.get('location') or normalized.get('address') or normalized.get('city')
            
            if place_id or name:
                restaurants.append(RestaurantInput(
                    place_id=place_id or None,
                    name=name or None,
                    location=location or None
                ))
    
    return restaurants


def parse_json_input(file_path: str) -> List[RestaurantInput]:
    """
    Parse restaurant list from JSON file.
    
    Expected format: Array of objects with place_id, name, location fields
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("JSON file must contain an array of restaurant objects")
    
    restaurants = []
    for item in data:
        if isinstance(item, dict):
            restaurants.append(RestaurantInput(
                place_id=item.get('place_id'),
                name=item.get('name'),
                location=item.get('location')
            ))
        elif isinstance(item, str):
            # Simple string format - treat as name
            restaurants.append(RestaurantInput(name=item))
    
    return restaurants


def get_restaurants_from_db(
    db: Session,
    mode: ProcessingMode,
    components: List[Component]
) -> List[RestaurantInput]:
    """
    Get restaurants from database based on processing mode.
    
    Args:
        db: Database session
        mode: Processing mode
        components: Components to check for staleness
        
    Returns:
        List of RestaurantInput for restaurants that need processing
    """
    restaurant_ids = set()
    
    if mode == ProcessingMode.FORCE:
        # All restaurants
        restaurants = db.query(Restaurant).all()
        return [RestaurantInput(place_id=r.place_id, name=r.name) for r in restaurants]
    
    if Component.ALL in components:
        components = [Component.CATEGORY, Component.TAGS, Component.DESCRIPTION, Component.EMBEDDING, Component.QUALITY]
    
    for component in components:
        if component == Component.QUALITY:
            current_version = get_active_version(db, "quality")
            
            if mode == ProcessingMode.AUTO:
                missing_images = get_images_missing_data(db, "quality")
                stale_images = get_stale_images(db, "quality", current_version)
                restaurant_ids.update(img.restaurant_id for img in missing_images)
                restaurant_ids.update(img.restaurant_id for img in stale_images)
            elif mode == ProcessingMode.NET_NEW:
                missing_images = get_images_missing_data(db, "quality")
                restaurant_ids.update(img.restaurant_id for img in missing_images)
            else:  # REFRESH_STALE
                stale_images = get_stale_images(db, "quality", current_version)
                restaurant_ids.update(img.restaurant_id for img in stale_images)
                
        elif component == Component.DESCRIPTION:
            current_version = get_active_version(db, "description")
            
            if mode == ProcessingMode.AUTO:
                # Get both missing AND stale
                missing = get_restaurants_missing_data(db, "description")
                stale = get_stale_restaurants(db, "description", current_version)
                restaurant_ids.update(r.id for r in missing)
                restaurant_ids.update(r.id for r in stale)
            elif mode == ProcessingMode.NET_NEW:
                missing = get_restaurants_missing_data(db, "description")
                restaurant_ids.update(r.id for r in missing)
            else:  # REFRESH_STALE
                stale = get_stale_restaurants(db, "description", current_version)
                restaurant_ids.update(r.id for r in stale)
            
        elif component == Component.EMBEDDING:
            current_version = get_active_version(db, "embedding")
            
            if mode == ProcessingMode.AUTO:
                missing = get_restaurants_missing_data(db, "embedding")
                stale = get_stale_restaurants(db, "embedding", current_version)
                restaurant_ids.update(r.id for r in missing)
                restaurant_ids.update(r.id for r in stale)
            elif mode == ProcessingMode.NET_NEW:
                missing = get_restaurants_missing_data(db, "embedding")
                restaurant_ids.update(r.id for r in missing)
            else:  # REFRESH_STALE
                stale = get_stale_restaurants(db, "embedding", current_version)
                restaurant_ids.update(r.id for r in stale)
            
        elif component in (Component.CATEGORY, Component.TAGS):
            comp_name = "category" if component == Component.CATEGORY else "tags"
            current_version = get_active_version(db, comp_name)
            
            if mode == ProcessingMode.AUTO:
                missing_images = get_images_missing_data(db, comp_name)
                stale_images = get_stale_images(db, comp_name, current_version)
                restaurant_ids.update(img.restaurant_id for img in missing_images)
                restaurant_ids.update(img.restaurant_id for img in stale_images)
            elif mode == ProcessingMode.NET_NEW:
                missing_images = get_images_missing_data(db, comp_name)
                restaurant_ids.update(img.restaurant_id for img in missing_images)
            else:  # REFRESH_STALE
                stale_images = get_stale_images(db, comp_name, current_version)
                restaurant_ids.update(img.restaurant_id for img in stale_images)
    
    # Fetch full restaurant records
    if restaurant_ids:
        restaurants = db.query(Restaurant).filter(Restaurant.id.in_(restaurant_ids)).all()
        return [RestaurantInput(place_id=r.place_id, name=r.name) for r in restaurants]
    
    return []


# =============================================================================
# Single Restaurant Processing
# =============================================================================

def should_process_component(
    db: Session,
    restaurant: Restaurant,
    component: Component,
    mode: ProcessingMode
) -> bool:
    """
    Determine if a component should be processed for a restaurant.
    
    Args:
        db: Database session
        restaurant: Restaurant to check
        component: Component to check
        mode: Processing mode
        
    Returns:
        True if component should be processed
    """
    if mode == ProcessingMode.FORCE:
        return True
    
    if component == Component.DESCRIPTION:
        is_missing = restaurant.description is None
        current_version = get_active_version(db, "description")
        is_stale = restaurant.description_version != current_version
        
        if mode == ProcessingMode.AUTO:
            return is_missing or is_stale
        elif mode == ProcessingMode.NET_NEW:
            return is_missing
        else:  # REFRESH_STALE
            return is_stale
            
    elif component == Component.EMBEDDING:
        is_missing = restaurant.embedding is None
        current_version = get_active_version(db, "embedding")
        is_stale = restaurant.embedding_version != current_version
        
        if mode == ProcessingMode.AUTO:
            return is_missing or is_stale
        elif mode == ProcessingMode.NET_NEW:
            return is_missing
        else:  # REFRESH_STALE
            return is_stale
    
    return True  # For image components, check at image level


def should_process_image_component(
    db: Session,
    image: RestaurantImage,
    component: Component,
    mode: ProcessingMode
) -> bool:
    """
    Determine if a component should be processed for an image.
    """
    if mode == ProcessingMode.FORCE:
        return True
    
    if component == Component.CATEGORY:
        is_missing = image.category is None
        current_version = get_active_version(db, "category")
        is_stale = image.category_version != current_version
        
        if mode == ProcessingMode.AUTO:
            return is_missing or is_stale
        elif mode == ProcessingMode.NET_NEW:
            return is_missing
        else:  # REFRESH_STALE
            return is_stale
            
    elif component == Component.TAGS:
        is_missing = image.ai_tags is None
        current_version = get_active_version(db, "tags")
        is_stale = image.tags_version != current_version
        
        if mode == ProcessingMode.AUTO:
            return is_missing or is_stale
        elif mode == ProcessingMode.NET_NEW:
            return is_missing
        else:  # REFRESH_STALE
            return is_stale
    
    elif component == Component.QUALITY:
        is_missing = image.quality_version is None
        current_version = get_active_version(db, "quality")
        is_stale = image.quality_version != current_version
        
        if mode == ProcessingMode.AUTO:
            return is_missing or is_stale
        elif mode == ProcessingMode.NET_NEW:
            return is_missing
        else:  # REFRESH_STALE
            return is_stale
    
    return False


def process_single_restaurant(
    db: Session,
    restaurant_input: RestaurantInput,
    mode: ProcessingMode,
    components: List[Component],
    rate_limiter: Optional[RateLimiter] = None,
    dry_run: bool = False,
    photo_provider: Optional[str] = None,
    refresh_images: bool = False,
) -> ProcessingResult:
    """
    Process a single restaurant - fetch images, categorize, analyze, embed.
    
    This is the main workhorse function that handles end-to-end processing.
    
    Args:
        db: Database session
        restaurant_input: Input data for the restaurant
        mode: Processing mode
        components: Components to process
        rate_limiter: Optional rate limiter (for async context)
        dry_run: If True, don't make any changes
        photo_provider: Override photo provider ("serpapi" or "google")
        refresh_images: If True, delete existing images and re-fetch from API
        
    Returns:
        ProcessingResult with details of what was done
    """
    import time
    start_time = time.time()
    
    components_processed = []
    components_skipped = []
    
    if Component.ALL in components:
        components = [Component.CATEGORY, Component.TAGS, Component.DESCRIPTION, Component.EMBEDDING, Component.QUALITY]
    
    # Initialize photo service with provider override
    photo_service = PhotoService(photo_provider_override=photo_provider)
    
    try:
        # Step 1: Resolve place_id if needed
        place_id = restaurant_input.place_id
        if not place_id:
            if not restaurant_input.name:
                raise ValueError("Either place_id or name must be provided")
            # Use photo_service for place search (supports both providers)
            try:
                result = photo_service.search_place(
                    restaurant_input.name,
                    restaurant_input.location,
                    db=db
                )
                place_id = result.place_id
            except ProviderError:
                # Fallback to existing method
                place_id = get_place_id_by_name(
                    restaurant_input.name,
                    restaurant_input.location
                )
        
        # Step 2: Get or create restaurant record
        restaurant_name = restaurant_input.name
        if not restaurant_name:
            restaurant_name = get_restaurant_name(place_id)
        
        restaurant = get_or_create_restaurant(db, place_id, restaurant_name)
        
        # Step 2.5: Delete existing images if refresh_images is True
        if refresh_images and not dry_run:
            print(f"[REFRESH] 🗑️ Deleting existing images for {restaurant_name}...")
            delete_result = delete_restaurant_images(db, place_id, delete_from_gcs=True)
            print(f"[REFRESH] Deleted {delete_result['deleted_db']} DB records, {delete_result['deleted_gcs']} GCS files")
            components_processed.append("images_deleted")
        
        if dry_run:
            # In dry-run mode, just report what would be done
            for comp in components:
                if comp in (Component.DESCRIPTION, Component.EMBEDDING):
                    if should_process_component(db, restaurant, comp, mode):
                        components_processed.append(comp.value)
                    else:
                        components_skipped.append(comp.value)
            
            # Check images
            cached_images = get_cached_images(db, place_id, max_images=50)
            for comp in [Component.CATEGORY, Component.TAGS, Component.QUALITY]:
                if comp in components:
                    needs_processing = any(
                        should_process_image_component(db, img, comp, mode)
                        for img in cached_images
                    )
                    if needs_processing:
                        components_processed.append(comp.value)
                    else:
                        components_skipped.append(comp.value)
            
            return ProcessingResult(
                place_id=place_id,
                name=restaurant_name,
                success=True,
                components_processed=components_processed,
                components_skipped=components_skipped,
                duration_seconds=time.time() - start_time
            )
        
        # Step 3: Smart fetch with inline quality scoring
        # This fetches from specific categories, scores each image, and stops when quota is met
        cached_images = get_cached_images(db, place_id, max_images=100)
        
        use_smart_fetch = Component.QUALITY in components and (
            refresh_images or not cached_images or len(cached_images) < 20
        )
        
        if use_smart_fetch:
            # Use new smart fetch that scores images inline
            from app.services.quality_service import smart_fetch_and_score
            
            print(f"[SMART FETCH] Using inline quality scoring for {restaurant_name}...")
            quota = {"food": 10, "interior": 10}
            
            quality_results = smart_fetch_and_score(
                db=db,
                place_id=place_id,
                quota=quota,
                max_per_category=30,
            )
            
            # Refresh cached images after smart fetch
            cached_images = get_cached_images(db, place_id, max_images=100)
            
            if quality_results:
                components_processed.append("quality")
                print(f"[SMART FETCH] ✅ Complete: {quality_results}")
        
        elif refresh_images or not cached_images or len(cached_images) < 20:
            # Fallback to old fetch method (when quality component not included)
            try:
                new_images = photo_service.fetch_and_cache_photos(
                    db=db,
                    place_id=place_id,
                    restaurant_name=restaurant_name,
                    max_photos=50,
                    restaurant=restaurant,
                )
                if new_images:
                    print(f"[PHOTO] Fetched {len(new_images)} photos via {photo_service.photo_provider.name}")
            except (ProviderError, Exception) as e:
                print(f"[WARNING] PhotoService failed: {e}")
            
            cached_images = get_cached_images(db, place_id, max_images=100)
        
        if not cached_images:
            raise ValueError(f"No images available for restaurant {place_id}")
        
        # Step 4: Process image categories (PARALLELIZED)
        # DISABLED: AI categorization is currently disabled because we use smart category
        # fetching from SerpAPI which pre-labels images. Re-enable by setting this to True.
        ENABLE_AI_CATEGORIZATION = False
        
        if Component.CATEGORY in components and ENABLE_AI_CATEGORIZATION:
            category_version = get_active_version(db, "category")
            images_to_categorize = [
                img for img in cached_images
                if should_process_image_component(db, img, Component.CATEGORY, mode)
            ]
            
            if images_to_categorize:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                
                def categorize_single_image(img):
                    """Download and categorize a single image."""
                    try:
                        response = requests.get(img.gcs_url, timeout=10)
                        response.raise_for_status()
                        image_bytes = response.content
                        category = categorize_image(image_bytes)
                        return {'id': img.id, 'category': category, 'success': True}
                    except Exception as e:
                        return {'id': img.id, 'category': None, 'success': False, 'error': str(e)}
                
                # Parallel categorization (limit to 5 concurrent to respect OpenAI rate limits)
                categorization_results = []
                max_category_workers = min(5, len(images_to_categorize))
                
                print(f"[PERF] Categorizing {len(images_to_categorize)} images in parallel (workers: {max_category_workers})")
                cat_start = time.time()
                
                with ThreadPoolExecutor(max_workers=max_category_workers) as executor:
                    futures = {executor.submit(categorize_single_image, img): img for img in images_to_categorize}
                    for future in as_completed(futures):
                        result = future.result()
                        categorization_results.append(result)
                
                cat_time = time.time() - cat_start
                print(f"[PERF] Categorization took {cat_time:.1f}s for {len(categorization_results)} images")
                
                # Update database with results
                for result in categorization_results:
                    if result['success']:
                        update_image_tags(
                            db, result['id'],
                            category=result['category'],
                            category_version=category_version
                        )
                    else:
                        print(f"[WARNING] Failed to categorize image {result['id']}: {result.get('error')}")
                
                components_processed.append("category")
                # Refresh cached images
                cached_images = get_cached_images(db, place_id, max_images=50)
            else:
                components_skipped.append("category")
        elif Component.CATEGORY in components:
            # AI categorization disabled - using pre-labeled categories from SerpAPI
            pre_categorized = sum(1 for img in cached_images if img.category)
            print(f"[SKIP] AI categorization disabled - {pre_categorized}/{len(cached_images)} images pre-categorized by SerpAPI")
            components_skipped.append("category")
        
        # Step 5: Quality scoring and display selection
        # If smart_fetch was used, scoring is already done - just need to select for display
        # If smart_fetch was NOT used, need to score existing images
        quality_selected_images = []
        
        if Component.QUALITY in components:
            from app.services.quality_service import apply_quality_filter_and_select
            from app.services.database_service import image_passes_quality_thresholds
            
            # Check if we already did quality scoring via smart_fetch
            quality_already_done = "quality" in components_processed
            
            if quality_already_done:
                # Smart fetch already scored images - just select for display
                print(f"[QUALITY] Using scores from smart fetch, selecting for display...")
                
                quality_selected_images = apply_quality_filter_and_select(
                    db, place_id,
                    quota={"food": 10, "interior": 10},
                    max_workers=5,
                )
                
                print(f"[QUALITY] Selected {len(quality_selected_images)} images for display")
                cached_images = get_cached_images(db, place_id, max_images=100)
            else:
                # Need to score existing images (legacy path when smart_fetch not used)
                images_need_scoring = any(
                    should_process_image_component(db, img, Component.QUALITY, mode)
                    for img in cached_images
                )
                
                if images_need_scoring:
                    print(f"[QUALITY] Scoring existing images for {restaurant_name}...")
                    
                    quality_selected_images = apply_quality_filter_and_select(
                        db, place_id,
                        quota={"food": 10, "interior": 10},
                        max_workers=5,
                    )
                    
                    print(f"[QUALITY] Selected {len(quality_selected_images)} images for display")
                    components_processed.append("quality")
                    cached_images = get_cached_images(db, place_id, max_images=100)
                else:
                    components_skipped.append("quality")
                    quality_selected_images = [img for img in cached_images if img.is_displayed]
        else:
            # Quality not in components - use already-displayed images
            quality_selected_images = [img for img in cached_images if img.is_displayed]
        
        # Step 6: Process AI tags and description together
        # (They use the same API call, so process together for efficiency)
        # IMPORTANT: Generate tags for quality-selected images (not arbitrary selection)
        process_tags = Component.TAGS in components
        process_description = Component.DESCRIPTION in components
        
        need_tags = process_tags and any(
            should_process_image_component(db, img, Component.TAGS, mode)
            for img in cached_images
        )
        need_description = process_description and should_process_component(
            db, restaurant, Component.DESCRIPTION, mode
        )
        
        if need_tags or need_description:
            # Use quality-selected images if available, otherwise select by quota
            if quality_selected_images:
                # Use the images that quality scoring selected for display
                selected_images = quality_selected_images
                print(f"[TAGS] Using {len(selected_images)} quality-selected images for AI analysis")
            else:
                # Fallback: select by quota from all images (when quality not run)
                # But prefer images that are already displayed
                displayed_images = [img for img in cached_images if img.is_displayed]
                if displayed_images and len(displayed_images) >= 5:
                    selected_images = displayed_images
                    print(f"[TAGS] Using {len(selected_images)} already-displayed images for AI analysis")
                else:
                    # Last resort: select by category quota
                    categorized = [(b"", img.category or "other") for img in cached_images]
                    quota = {"food": 10, "interior": 10}
                    _, selected_indices = select_images_by_quota(categorized, quota=quota, max_bar=2)
                    selected_images = [cached_images[i] for i in selected_indices if i < len(cached_images)]
                    print(f"[TAGS] Selected {len(selected_images)} images by category quota for AI analysis")
            
            # Download selected images (PARALLELIZED)
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def download_image(img):
                """Download a single image."""
                try:
                    response = requests.get(img.gcs_url, timeout=10)
                    response.raise_for_status()
                    return {'img': img, 'bytes': response.content, 'success': True}
                except Exception as e:
                    return {'img': img, 'bytes': None, 'success': False, 'error': str(e)}
            
            print(f"[PERF] Downloading {len(selected_images)} images for AI analysis in parallel")
            download_start = time.time()
            
            download_results = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(download_image, img): img for img in selected_images}
                for future in as_completed(futures):
                    download_results.append(future.result())
            
            download_time = time.time() - download_start
            print(f"[PERF] Image download took {download_time:.1f}s")
            
            # Separate successful downloads
            image_bytes_list = []
            valid_images = []
            for result in download_results:
                if result['success']:
                    image_bytes_list.append(result['bytes'])
                    valid_images.append(result['img'])
                else:
                    print(f"[WARNING] Failed to download image {result['img'].id}: {result.get('error')}")
            
            if image_bytes_list:
                # Run AI analysis
                tags_version = get_active_version(db, "tags")
                description_version = get_active_version(db, "description")
                
                analysis = analyze_restaurant_images(image_bytes_list)
                
                # Store tags
                if need_tags:
                    image_ai_tags = [(img.photo_name, analysis["tags"]) for img in valid_images]
                    store_ai_tags_for_images(db, place_id, image_ai_tags, version=tags_version)
                    components_processed.append("tags")
                else:
                    components_skipped.append("tags")
                
                # Store description
                if need_description:
                    store_restaurant_description(
                        db, place_id, analysis["description"],
                        version=description_version
                    )
                    components_processed.append("description")
                else:
                    components_skipped.append("description")
            else:
                if process_tags:
                    components_skipped.append("tags")
                if process_description:
                    components_skipped.append("description")
        else:
            if process_tags:
                components_skipped.append("tags")
            if process_description:
                components_skipped.append("description")
        
        # Step 7: Generate embedding
        if Component.EMBEDDING in components:
            if should_process_component(db, restaurant, Component.EMBEDDING, mode):
                embedding_version = get_active_version(db, "embedding")
                
                # Refresh restaurant to get updated data
                db.refresh(restaurant)
                
                update_restaurant_embedding(db, restaurant.id)
                
                # Update version tracking
                restaurant.embedding_version = embedding_version
                restaurant.embedding_updated_at = datetime.now(timezone.utc)
                db.commit()
                
                components_processed.append("embedding")
            else:
                components_skipped.append("embedding")
        
        return ProcessingResult(
            place_id=place_id,
            name=restaurant_name,
            success=True,
            components_processed=components_processed,
            components_skipped=components_skipped,
            duration_seconds=time.time() - start_time
        )
        
    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to process restaurant: {e}")
        print(traceback.format_exc())
        
        return ProcessingResult(
            place_id=restaurant_input.place_id or "unknown",
            name=restaurant_input.name or "unknown",
            success=False,
            components_processed=components_processed,
            components_skipped=components_skipped,
            error=str(e),
            duration_seconds=time.time() - start_time
        )


# =============================================================================
# Async Parallel Processing
# =============================================================================

async def process_restaurant_async(
    db_factory: Callable[[], Session],
    restaurant_input: RestaurantInput,
    mode: ProcessingMode,
    components: List[Component],
    rate_limiter: RateLimiter,
    dry_run: bool = False,
    photo_provider: Optional[str] = None,
    refresh_images: bool = False,
) -> ProcessingResult:
    """
    Async wrapper for processing a single restaurant with rate limiting.
    
    Args:
        db_factory: Callable that returns a new database session
        restaurant_input: Input data for the restaurant
        mode: Processing mode
        components: Components to process
        rate_limiter: Rate limiter for API calls
        dry_run: If True, don't make any changes
        photo_provider: Override photo provider ("serpapi" or "google")
        refresh_images: If True, delete existing images and re-fetch from API
        
    Returns:
        ProcessingResult
    """
    await rate_limiter.acquire_overall()
    
    try:
        # Run synchronous processing in a thread pool
        loop = asyncio.get_event_loop()
        
        def sync_process():
            db = db_factory()
            try:
                return process_single_restaurant(
                    db, restaurant_input, mode, components,
                    rate_limiter=rate_limiter, dry_run=dry_run,
                    photo_provider=photo_provider, refresh_images=refresh_images
                )
            finally:
                db.close()
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(executor, sync_process)
        
        return result
        
    finally:
        rate_limiter.release_overall()


async def process_restaurants_parallel(
    db_factory: Callable[[], Session],
    restaurants: List[RestaurantInput],
    mode: ProcessingMode,
    components: List[Component],
    concurrency: int = 5,
    openai_concurrency: int = 3,
    places_concurrency: int = 5,
    dry_run: bool = False,
    progress_callback: Optional[Callable[[int, int, ProcessingResult], None]] = None,
    photo_provider: Optional[str] = None,
    refresh_images: bool = False,
) -> BulkProcessingReport:
    """
    Process multiple restaurants in parallel with rate limiting.
    
    Args:
        db_factory: Callable that returns a new database session
        restaurants: List of restaurants to process
        mode: Processing mode
        components: Components to process
        concurrency: Overall concurrency limit
        openai_concurrency: OpenAI API concurrency limit
        places_concurrency: Google Places API concurrency limit
        dry_run: If True, don't make any changes
        progress_callback: Optional callback for progress updates
        photo_provider: Override photo provider ("serpapi" or "google")
        refresh_images: If True, delete existing images and re-fetch from API
        
    Returns:
        BulkProcessingReport with results
    """
    # Reset cost tracker for this batch
    reset_cost_tracker()
    
    report = BulkProcessingReport(
        total=len(restaurants),
        start_time=datetime.now(timezone.utc)
    )
    
    rate_limiter = RateLimiter(
        openai_limit=openai_concurrency,
        places_limit=places_concurrency,
        overall_limit=concurrency
    )
    
    async def process_with_progress(idx: int, restaurant: RestaurantInput):
        result = await process_restaurant_async(
            db_factory, restaurant, mode, components,
            rate_limiter, dry_run, photo_provider, refresh_images
        )
        
        if progress_callback:
            progress_callback(idx + 1, len(restaurants), result)
        
        return result
    
    # Create tasks for all restaurants
    tasks = [
        process_with_progress(i, r)
        for i, r in enumerate(restaurants)
    ]
    
    # Execute all tasks
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for result in results:
        if isinstance(result, Exception):
            report.failed += 1
            report.results.append(ProcessingResult(
                place_id="unknown",
                name="unknown",
                success=False,
                error=str(result)
            ))
        elif result.success:
            if result.components_processed:
                report.processed += 1
            else:
                report.skipped += 1
            report.results.append(result)
        else:
            report.failed += 1
            report.results.append(result)
    
    report.end_time = datetime.now(timezone.utc)
    
    # Add cost tracking to report
    cost_tracker = get_cost_tracker()
    report.total_cost = cost_tracker.total_cost
    report.cost_by_provider = dict(cost_tracker.costs_by_provider)
    report.api_calls_by_provider = dict(cost_tracker.api_calls_by_provider)
    
    return report


def run_bulk_processing(
    db_factory: Callable[[], Session],
    restaurants: List[RestaurantInput],
    mode: ProcessingMode,
    components: List[Component],
    concurrency: int = 5,
    openai_concurrency: int = 3,
    places_concurrency: int = 5,
    dry_run: bool = False,
    progress_callback: Optional[Callable[[int, int, ProcessingResult], None]] = None,
    photo_provider: Optional[str] = None,
    refresh_images: bool = False,
) -> BulkProcessingReport:
    """
    Synchronous entry point for bulk processing.
    
    Creates an event loop and runs the async processing.
    
    Args:
        photo_provider: Override photo provider ("serpapi" or "google")
        refresh_images: If True, delete existing images and re-fetch from API
    """
    return asyncio.run(
        process_restaurants_parallel(
            db_factory, restaurants, mode, components,
            concurrency, openai_concurrency, places_concurrency,
            dry_run, progress_callback, photo_provider, refresh_images
        )
    )

