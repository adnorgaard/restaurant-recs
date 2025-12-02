"""
Unified Photo Service

High-level service for fetching, processing, and caching restaurant photos.
Uses the provider abstraction layer and handles:
- Provider selection (SerpApi vs Google)
- Photo downloading (parallelized)
- GCS caching (parallelized)
- Cost tracking
- Fallback logic
"""

import os
import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from sqlalchemy.orm import Session

from app.services.providers import (
    get_photo_provider,
    get_place_provider,
    PhotoProvider,
    PlaceProvider,
    PhotoResult,
    PhotoFetchResult,
    PlaceResult,
    ProviderError,
    GooglePlacesProvider,
    SerpApiProvider,
)
from app.services.storage_service import upload_image_to_gcs, stream_upload_to_gcs
from app.models.database import Restaurant, RestaurantImage

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class CostTracker:
    """Tracks API costs across operations."""
    
    costs_by_provider: Dict[str, float] = field(default_factory=dict)
    api_calls_by_provider: Dict[str, int] = field(default_factory=dict)
    
    def add_cost(self, provider: str, cost: float, api_calls: int = 1):
        """Record cost for a provider."""
        self.costs_by_provider[provider] = self.costs_by_provider.get(provider, 0) + cost
        self.api_calls_by_provider[provider] = self.api_calls_by_provider.get(provider, 0) + api_calls
    
    @property
    def total_cost(self) -> float:
        return sum(self.costs_by_provider.values())
    
    @property
    def total_api_calls(self) -> int:
        return sum(self.api_calls_by_provider.values())
    
    def summary(self) -> str:
        lines = ["Cost Summary:"]
        for provider, cost in self.costs_by_provider.items():
            calls = self.api_calls_by_provider.get(provider, 0)
            lines.append(f"  {provider}: ${cost:.4f} ({calls} API calls)")
        lines.append(f"  TOTAL: ${self.total_cost:.4f} ({self.total_api_calls} calls)")
        return "\n".join(lines)


# Global cost tracker (can be reset per batch)
_cost_tracker = CostTracker()


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker."""
    return _cost_tracker


def reset_cost_tracker():
    """Reset the global cost tracker."""
    global _cost_tracker
    _cost_tracker = CostTracker()


class PhotoService:
    """
    High-level photo service with provider abstraction.
    
    Features:
    - Automatic provider selection based on config
    - Fallback from SerpApi to Google if needed
    - Cost tracking
    - GCS caching integration
    """
    
    def __init__(
        self,
        photo_provider: Optional[PhotoProvider] = None,
        place_provider: Optional[PlaceProvider] = None,
        photo_provider_override: Optional[str] = None,
        place_provider_override: Optional[str] = None,
        enable_fallback: bool = True,
    ):
        """
        Initialize PhotoService.
        
        Args:
            photo_provider: Explicit photo provider instance
            place_provider: Explicit place provider instance
            photo_provider_override: Override for photo provider (e.g., "serpapi")
            place_provider_override: Override for place provider (e.g., "google")
            enable_fallback: If True, fall back to Google if SerpApi fails
        """
        self.photo_provider = photo_provider or get_photo_provider(override=photo_provider_override)
        self.place_provider = place_provider or get_place_provider(override=place_provider_override)
        self.enable_fallback = enable_fallback
        self.cost_tracker = get_cost_tracker()
    
    def search_place(
        self,
        name: str,
        location: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> PlaceResult:
        """
        Search for a place and optionally store the result.
        
        Args:
            name: Restaurant name
            location: Optional location (city, address)
            db: Optional database session to cache serpapi_data_id
            
        Returns:
            PlaceResult with place_id and optionally serpapi_data_id
        """
        try:
            result = self.place_provider.search_place(name, location)
            self.cost_tracker.add_cost(result.provider, result.cost)
            
            # If we got a serpapi_data_id, cache it in the database
            if db and result.serpapi_data_id and result.place_id:
                self._cache_serpapi_data_id(db, result.place_id, result.serpapi_data_id)
            
            return result
            
        except ProviderError as e:
            self.cost_tracker.add_cost(e.provider, e.cost)
            
            # Try fallback if enabled and we're not already using Google
            if self.enable_fallback and self.place_provider.name != "google_places":
                logger.warning(f"Falling back to Google for place search: {e}")
                fallback = GooglePlacesProvider()
                result = fallback.search_place(name, location)
                self.cost_tracker.add_cost(result.provider, result.cost)
                return result
            
            raise
    
    def fetch_photos(
        self,
        place_id: str,
        max_photos: int = 50,
        db: Optional[Session] = None,
        restaurant: Optional[Restaurant] = None,
        use_smart_categories: bool = True,
        quota: Optional[dict] = None,
    ) -> PhotoFetchResult:
        """
        Fetch photos for a place.
        
        Args:
            place_id: Google Places ID
            max_photos: Maximum photos to fetch
            db: Optional database session
            restaurant: Optional restaurant record (for cached serpapi_data_id)
            use_smart_categories: If True and using SerpAPI, fetch by category for better quota coverage
            quota: Optional quota dict for smart category fetching
            
        Returns:
            PhotoFetchResult with photos
        """
        # Get cached serpapi_data_id if available
        serpapi_data_id = None
        if restaurant and restaurant.serpapi_data_id:
            serpapi_data_id = restaurant.serpapi_data_id
        
        try:
            # Use smart category fetching if available (SerpAPI only)
            if use_smart_categories and hasattr(self.photo_provider, 'fetch_photos_by_quota'):
                result = self.photo_provider.fetch_photos_by_quota(
                    place_id=place_id,
                    quota=quota,
                    max_total=max_photos,
                    serpapi_data_id=serpapi_data_id,
                )
            else:
                result = self.photo_provider.fetch_photos(
                    place_id=place_id,
                    max_photos=max_photos,
                    serpapi_data_id=serpapi_data_id,
                )
            
            self.cost_tracker.add_cost(result.provider, result.cost, result.api_calls)
            
            # Cache serpapi_data_id if we got one
            if db and result.provider == "serpapi" and restaurant:
                # The SerpApi provider may have looked up the data_id
                # We should cache it for future use
                pass  # TODO: Extract and cache data_id
            
            return result
            
        except ProviderError as e:
            self.cost_tracker.add_cost(e.provider, e.cost)
            
            # Try fallback if enabled
            if self.enable_fallback and self.photo_provider.name != "google_places":
                logger.warning(f"Falling back to Google for photos: {e}")
                fallback = GooglePlacesProvider()
                result = fallback.fetch_photos(place_id=place_id, max_photos=max_photos)
                self.cost_tracker.add_cost(result.provider, result.cost, result.api_calls)
                return result
            
            raise
    
    def fetch_and_cache_photos(
        self,
        db: Session,
        place_id: str,
        restaurant_name: str,
        max_photos: int = 50,
        restaurant: Optional[Restaurant] = None,
        max_workers: int = 10,
        use_smart_categories: bool = True,
    ) -> List[RestaurantImage]:
        """
        Fetch photos and cache them to GCS and database.
        
        Uses parallel downloads and uploads for performance.
        When using SerpAPI with smart categories, photos are pre-categorized
        to reduce AI categorization needs.
        
        Args:
            db: Database session
            place_id: Google Places ID
            restaurant_name: Restaurant name (for GCS path)
            max_photos: Maximum photos to fetch
            restaurant: Optional restaurant record
            max_workers: Maximum concurrent downloads/uploads (default: 10)
            use_smart_categories: Use smart category fetching (SerpAPI only)
            
        Returns:
            List of RestaurantImage records
        """
        import hashlib
        import time
        
        start_time = time.time()
        
        # Fetch photo URLs from provider (this is fast - just API call)
        result = self.fetch_photos(
            place_id=place_id,
            max_photos=max_photos,
            db=db,
            restaurant=restaurant,
            use_smart_categories=use_smart_categories,
        )
        
        if not result.photos:
            logger.warning(f"No photos found for {restaurant_name}")
            return []
        
        # Count pre-categorized photos
        pre_categorized = sum(1 for p in result.photos if p.category is not None)
        
        api_time = time.time() - start_time
        logger.info(f"[PERF] SerpApi fetch took {api_time:.1f}s for {len(result.photos)} photo URLs ({pre_categorized} pre-categorized)")
        
        # Get or create restaurant record
        if not restaurant:
            restaurant = db.query(Restaurant).filter(
                Restaurant.place_id == place_id
            ).first()
            
            if not restaurant:
                restaurant = Restaurant(
                    place_id=place_id,
                    name=restaurant_name,
                )
                db.add(restaurant)
                db.flush()
        
        # Pre-generate photo names and check for existing
        photo_tasks = []
        for photo in result.photos:
            photo_hash = hashlib.md5(photo.image_url.encode()).hexdigest()[:16]
            photo_name = f"{result.provider}_{photo_hash}"
            
            # Check if already cached
            existing = db.query(RestaurantImage).filter(
                RestaurantImage.restaurant_id == restaurant.id,
                RestaurantImage.photo_name == photo_name
            ).first()
            
            if existing:
                photo_tasks.append({
                    'photo': photo,
                    'photo_name': photo_name,
                    'existing': existing,
                    'skip': True
                })
            else:
                photo_tasks.append({
                    'photo': photo,
                    'photo_name': photo_name,
                    'existing': None,
                    'skip': False
                })
        
        # Count how many need processing
        to_process = [t for t in photo_tasks if not t['skip']]
        already_cached = [t for t in photo_tasks if t['skip']]
        
        logger.info(f"[PERF] {len(already_cached)} already cached, {len(to_process)} to download/upload")
        
        if not to_process:
            # All photos already cached
            return [t['existing'] for t in photo_tasks]
        
        # Parallel download and upload using ThreadPoolExecutor
        upload_start = time.time()
        
        def download_and_upload(task: dict) -> Optional[dict]:
            """Download image and upload to GCS in one operation."""
            try:
                gcs_url, gcs_path = stream_upload_to_gcs(
                    image_url=task['photo'].image_url,
                    place_id=place_id,
                    photo_name=task['photo_name'],
                )
                return {
                    'photo_name': task['photo_name'],
                    'gcs_url': gcs_url,
                    'gcs_path': gcs_path,
                    'category': task['photo'].category,
                    'success': True
                }
            except Exception as e:
                logger.warning(f"Failed to cache photo {task['photo_name']}: {e}")
                return {
                    'photo_name': task['photo_name'],
                    'success': False,
                    'error': str(e)
                }
        
        # Execute in parallel
        upload_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(download_and_upload, task): task 
                for task in to_process
            }
            for future in as_completed(futures):
                result_data = future.result()
                if result_data:
                    upload_results.append(result_data)
        
        upload_time = time.time() - upload_start
        successful = [r for r in upload_results if r.get('success')]
        logger.info(f"[PERF] Parallel upload took {upload_time:.1f}s for {len(successful)}/{len(to_process)} photos")
        
        # Create database records for successful uploads
        cached_images = []
        
        # Add already-cached images first
        for task in already_cached:
            cached_images.append(task['existing'])
        
        # Add newly uploaded images
        for result_data in successful:
            image_record = RestaurantImage(
                restaurant_id=restaurant.id,
                photo_name=result_data['photo_name'],
                gcs_url=result_data['gcs_url'],
                gcs_bucket_path=result_data['gcs_path'],
                category=result_data.get('category'),
            )
            db.add(image_record)
            cached_images.append(image_record)
        
        db.commit()
        
        total_time = time.time() - start_time
        logger.info(
            f"[PERF] Total photo caching: {total_time:.1f}s for {len(cached_images)} photos "
            f"(provider: {result.provider}, cost: ${result.cost:.4f})"
        )
        
        return cached_images
    
    def _cache_serpapi_data_id(
        self,
        db: Session,
        place_id: str,
        serpapi_data_id: str,
    ):
        """Cache serpapi_data_id for a restaurant."""
        try:
            restaurant = db.query(Restaurant).filter(
                Restaurant.place_id == place_id
            ).first()
            
            if restaurant and not restaurant.serpapi_data_id:
                restaurant.serpapi_data_id = serpapi_data_id
                db.commit()
                logger.debug(f"Cached serpapi_data_id for {restaurant.name}")
        except Exception as e:
            logger.warning(f"Failed to cache serpapi_data_id: {e}")


def log_cost_summary():
    """Log the current cost summary."""
    tracker = get_cost_tracker()
    logger.info(tracker.summary())
    print(tracker.summary())

