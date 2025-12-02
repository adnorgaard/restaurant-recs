"""
SerpApi provider implementation.

Uses SerpApi to scrape Google Maps for photos (100+ photos vs Google's 10 limit).
Supports category-based fetching for smart quota fulfillment.
"""

import os
import requests
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path

from .base import (
    PhotoProvider,
    PlaceProvider,
    PhotoResult,
    PhotoFetchResult,
    PlaceResult,
    ProviderError,
)

# Load environment
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
SERPAPI_BASE_URL = "https://serpapi.com/search.json"


# =============================================================================
# Google Maps Photo Categories
# =============================================================================
# These are the standard category IDs used by Google Maps
# Note: Category availability may vary by place type

@dataclass
class GoogleMapsCategory:
    """Google Maps photo category with ID and our internal mapping."""
    name: str           # Google's category name
    category_id: str    # SerpAPI category_id parameter
    our_category: str   # Our internal category (interior, food, etc.)


# Standard Google Maps photo categories
GMAPS_CATEGORIES = {
    "all": GoogleMapsCategory("All", "CgIgAQ", "all"),
    "food_drink": GoogleMapsCategory("Food & drink", "CgIYIA", "food"),
    "vibe": GoogleMapsCategory("Vibe", "CgIYIg", "interior"),
    "menu": GoogleMapsCategory("Menu", "CgIYIQ", "menu"),
}

# Mapping from our categories to Google Maps categories
OUR_CATEGORY_TO_GMAPS = {
    "food": "food_drink",
    "drink": "food_drink",  
    "interior": "vibe",
    "menu": "menu",
    # Note: No direct mapping for "exterior" - Google doesn't have this category
}


class SerpApiProvider(PhotoProvider, PlaceProvider):
    """
    SerpApi provider for Google Maps data.
    
    Advantages over Google Places API:
    - Returns 100+ photos per place (vs 10)
    - Supports pagination
    - Includes photo categories (food, vibe, etc.)
    - Same photos users see on Google Maps
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or SERPAPI_API_KEY
    
    @property
    def name(self) -> str:
        return "serpapi"
    
    @property
    def cost_per_request(self) -> float:
        # SerpApi: $50/mo for 5000 searches = $0.01 per search
        return 0.01
    
    def _get_api_key(self) -> str:
        if not self._api_key:
            raise ProviderError(
                "SERPAPI_API_KEY not configured. Add it to your .env file.",
                provider=self.name
            )
        return self._api_key
    
    def search_place(
        self,
        query: str,
        location: Optional[str] = None,
    ) -> PlaceResult:
        """Search for a place using SerpApi Google Maps."""
        api_key = self._get_api_key()
        
        search_query = query
        if location:
            search_query = f"{query} {location}"
        
        params = {
            "engine": "google_maps",
            "q": search_query,
            "api_key": api_key
        }
        
        try:
            response = requests.get(SERPAPI_BASE_URL, params=params, timeout=15)
            
            if response.status_code != 200:
                raise ProviderError(
                    f"SerpApi error ({response.status_code}): {response.text}",
                    provider=self.name,
                    cost=self.cost_per_request
                )
            
            data = response.json()
            
            # Check for place_results (single place) or local_results (multiple)
            if "place_results" in data:
                place = data["place_results"]
                return PlaceResult(
                    place_id=place.get("place_id", ""),
                    name=place.get("title", "Unknown"),
                    address=place.get("address"),
                    serpapi_data_id=place.get("data_id"),
                    provider=self.name,
                    cost=self.cost_per_request,
                    metadata={
                        "gps_coordinates": place.get("gps_coordinates"),
                        "rating": place.get("rating"),
                        "reviews": place.get("reviews"),
                    }
                )
            elif "local_results" in data and data["local_results"]:
                place = data["local_results"][0]
                return PlaceResult(
                    place_id=place.get("place_id", ""),
                    name=place.get("title", "Unknown"),
                    address=place.get("address"),
                    serpapi_data_id=place.get("data_id"),
                    provider=self.name,
                    cost=self.cost_per_request,
                    metadata={
                        "gps_coordinates": place.get("gps_coordinates"),
                        "rating": place.get("rating"),
                        "reviews": place.get("reviews"),
                    }
                )
            else:
                raise ProviderError(
                    f"No places found for query: {search_query}",
                    provider=self.name,
                    cost=self.cost_per_request
                )
                
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(
                f"Failed to search place: {str(e)}",
                provider=self.name,
                cost=self.cost_per_request
            )
    
    def get_place_details(
        self,
        place_id: str,
    ) -> PlaceResult:
        """Get place details using SerpApi."""
        api_key = self._get_api_key()
        
        params = {
            "engine": "google_maps",
            "place_id": place_id,
            "api_key": api_key
        }
        
        try:
            response = requests.get(SERPAPI_BASE_URL, params=params, timeout=15)
            
            if response.status_code != 200:
                raise ProviderError(
                    f"SerpApi error ({response.status_code}): {response.text}",
                    provider=self.name,
                    cost=self.cost_per_request
                )
            
            data = response.json()
            place = data.get("place_results", {})
            
            return PlaceResult(
                place_id=place.get("place_id", place_id),
                name=place.get("title", "Unknown"),
                address=place.get("address"),
                serpapi_data_id=place.get("data_id"),
                provider=self.name,
                cost=self.cost_per_request,
                metadata={
                    "gps_coordinates": place.get("gps_coordinates"),
                    "rating": place.get("rating"),
                    "reviews": place.get("reviews"),
                    "photos_link": place.get("photos_link"),
                }
            )
            
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(
                f"Failed to get place details: {str(e)}",
                provider=self.name,
                cost=self.cost_per_request
            )
    
    def _get_data_id(
        self,
        place_id: str,
        serpapi_data_id: Optional[str] = None,
    ) -> str:
        """Get SerpApi data_id, fetching it if necessary."""
        if serpapi_data_id:
            return serpapi_data_id
        
        # Need to look up the data_id from place_id
        place = self.get_place_details(place_id)
        if not place.serpapi_data_id:
            raise ProviderError(
                f"Could not find SerpApi data_id for place_id: {place_id}",
                provider=self.name,
                cost=self.cost_per_request
            )
        return place.serpapi_data_id
    
    def _fetch_photos_from_category(
        self,
        data_id: str,
        category_id: Optional[str] = None,
        category_label: Optional[str] = None,
        max_photos: int = 20,
    ) -> Tuple[List[PhotoResult], float, int]:
        """
        Fetch photos from a specific Google Maps category.
        
        Args:
            data_id: SerpAPI data_id for the place
            category_id: Google Maps category ID (e.g., "CgIYIg" for Vibe)
            category_label: Our category label to assign (e.g., "interior")
            max_photos: Maximum photos to fetch from this category
            
        Returns:
            Tuple of (photos, cost, api_calls)
        """
        api_key = self._get_api_key()
        
        photos = []
        total_cost = 0.0
        api_calls = 0
        next_token = None
        
        while len(photos) < max_photos:
            params = {
                "engine": "google_maps_photos",
                "data_id": data_id,
                "api_key": api_key
            }
            
            # Add category filter if specified
            if category_id:
                params["category_id"] = category_id
            
            if next_token:
                params["next_page_token"] = next_token
            
            try:
                response = requests.get(SERPAPI_BASE_URL, params=params, timeout=15)
                api_calls += 1
                total_cost += self.cost_per_request
                
                if response.status_code != 200:
                    break
                
                data = response.json()
                photo_list = data.get("photos", [])
                
                if not photo_list:
                    break
                
                for photo in photo_list:
                    if len(photos) >= max_photos:
                        break
                    
                    source = photo.get("source", {})
                    photos.append(PhotoResult(
                        image_url=photo.get("image", ""),
                        thumbnail_url=photo.get("thumbnail"),
                        category=category_label,  # Pre-assign category!
                        source_name=source.get("name"),
                        date=photo.get("date"),
                        provider=self.name,
                        photo_id=photo.get("image", "")[:100],
                        metadata={
                            "source_link": source.get("link"),
                            "source_reviews": source.get("reviews"),
                            "gmaps_category": category_id,
                        }
                    ))
                
                pagination = data.get("serpapi_pagination", {})
                next_token = pagination.get("next_page_token")
                
                if not next_token or len(photos) >= max_photos:
                    break
                    
            except Exception as e:
                print(f"[WARNING] SerpApi category fetch error: {e}")
                break
        
        return photos, total_cost, api_calls
    
    def get_data_id_for_place(
        self,
        place_id: str,
        serpapi_data_id: Optional[str] = None,
    ) -> str:
        """
        Public method to get SerpAPI data_id for a place.
        
        Args:
            place_id: Google Places ID
            serpapi_data_id: Optional cached data_id
            
        Returns:
            SerpAPI data_id string
        """
        return self._get_data_id(place_id, serpapi_data_id)
    
    def fetch_category_page(
        self,
        data_id: str,
        category_id: str,
        category_label: str,
        next_page_token: Optional[str] = None,
    ) -> Tuple[List[PhotoResult], Optional[str], float]:
        """
        Fetch a single page of photos from a category.
        
        This allows for incremental fetching with quality scoring between pages.
        
        Args:
            data_id: SerpAPI data_id for the place
            category_id: Google Maps category ID (e.g., "CgIYIg" for Vibe)
            category_label: Our category label to assign (e.g., "interior")
            next_page_token: Token from previous page, or None for first page
            
        Returns:
            Tuple of (photos, next_page_token, cost)
            next_page_token is None if no more pages available
        """
        api_key = self._get_api_key()
        
        params = {
            "engine": "google_maps_photos",
            "data_id": data_id,
            "api_key": api_key,
            "category_id": category_id,
        }
        
        if next_page_token:
            params["next_page_token"] = next_page_token
        
        try:
            response = requests.get(SERPAPI_BASE_URL, params=params, timeout=15)
            
            if response.status_code != 200:
                print(f"[WARNING] SerpApi error ({response.status_code})")
                return [], None, self.cost_per_request
            
            data = response.json()
            photo_list = data.get("photos", [])
            
            photos = []
            for photo in photo_list:
                source = photo.get("source", {})
                photos.append(PhotoResult(
                    image_url=photo.get("image", ""),
                    thumbnail_url=photo.get("thumbnail"),
                    category=category_label,
                    source_name=source.get("name"),
                    date=photo.get("date"),
                    provider=self.name,
                    photo_id=photo.get("image", "")[:100],
                    metadata={
                        "source_link": source.get("link"),
                        "source_reviews": source.get("reviews"),
                        "gmaps_category": category_id,
                    }
                ))
            
            pagination = data.get("serpapi_pagination", {})
            next_token = pagination.get("next_page_token")
            
            return photos, next_token, self.cost_per_request
            
        except Exception as e:
            print(f"[WARNING] SerpApi page fetch error: {e}")
            return [], None, self.cost_per_request
    
    def fetch_photos(
        self,
        place_id: str,
        max_photos: int = 50,
        categories: Optional[List[str]] = None,
        serpapi_data_id: Optional[str] = None,
    ) -> PhotoFetchResult:
        """
        Fetch photos using SerpApi Google Maps Photos.
        
        Returns up to max_photos, with pagination support for more.
        """
        api_key = self._get_api_key()
        
        # Get or lookup data_id
        try:
            data_id = self._get_data_id(place_id, serpapi_data_id)
        except ProviderError as e:
            # If we can't get data_id, return empty result
            return PhotoFetchResult(
                photos=[],
                total_available=0,
                has_more=False,
                provider=self.name,
                cost=e.cost
            )
        
        all_photos = []
        total_cost = 0.0
        api_calls = 0
        next_token = None
        
        while len(all_photos) < max_photos:
            params = {
                "engine": "google_maps_photos",
                "data_id": data_id,
                "api_key": api_key
            }
            if next_token:
                params["next_page_token"] = next_token
            
            try:
                response = requests.get(SERPAPI_BASE_URL, params=params, timeout=15)
                api_calls += 1
                total_cost += self.cost_per_request
                
                if response.status_code != 200:
                    break
                
                data = response.json()
                photos = data.get("photos", [])
                
                if not photos:
                    break
                
                # Convert to PhotoResult objects
                for photo in photos:
                    if len(all_photos) >= max_photos:
                        break
                    
                    # Try to determine category from SerpApi categories
                    category = None
                    if "categories" in data:
                        # SerpApi returns categories like "Food & drink", "Vibe", etc.
                        # We'll use the first matching one
                        pass  # Category assignment happens at image level later
                    
                    source = photo.get("source", {})
                    all_photos.append(PhotoResult(
                        image_url=photo.get("image", ""),
                        thumbnail_url=photo.get("thumbnail"),
                        source_name=source.get("name"),
                        date=photo.get("date"),
                        provider=self.name,
                        photo_id=photo.get("image", "")[:100],  # Use image URL as ID
                        metadata={
                            "source_link": source.get("link"),
                            "source_reviews": source.get("reviews"),
                        }
                    ))
                
                # Check for more pages
                pagination = data.get("serpapi_pagination", {})
                next_token = pagination.get("next_page_token")
                
                if not next_token or len(all_photos) >= max_photos:
                    break
                    
            except Exception as e:
                print(f"[WARNING] SerpApi photo fetch error: {e}")
                break
        
        return PhotoFetchResult(
            photos=all_photos,
            total_available=None,  # SerpApi doesn't tell us total
            has_more=next_token is not None,
            next_page_token=next_token,
            provider=self.name,
            cost=total_cost,
            api_calls=api_calls
        )
    
    def fetch_photos_by_quota(
        self,
        place_id: str,
        quota: Optional[Dict[str, int]] = None,
        max_total: int = 50,
        serpapi_data_id: Optional[str] = None,
    ) -> PhotoFetchResult:
        """
        Smart photo fetching that pulls from specific categories to meet quotas.
        
        This reduces AI categorization needs by pre-labeling photos from 
        Google Maps categories.
        
        Strategy:
        1. Fetch from "Vibe" category → interior photos
        2. Fetch from "Food & drink" category → food/drink photos  
        3. Fetch from "All" category → exterior + remaining (needs AI categorization)
        
        Args:
            place_id: Google Places ID
            quota: Target quota per category. Default: {"interior": 10, "food": 15, "exterior": 5}
            max_total: Maximum total photos to fetch
            serpapi_data_id: Optional cached data_id
            
        Returns:
            PhotoFetchResult with pre-categorized photos where possible
        """
        # Default quota focused on filling display needs
        # These are starting points - remaining photos come from "All" category
        if quota is None:
            quota = {
                "interior": 15,  # Fetch 15 from Vibe category (ambiance shots)
                "food": 20,      # Fetch 20 from Food & drink category (food + cocktails)
            }
        
        # Get or lookup data_id
        try:
            data_id = self._get_data_id(place_id, serpapi_data_id)
        except ProviderError as e:
            return PhotoFetchResult(
                photos=[],
                total_available=0,
                has_more=False,
                provider=self.name,
                cost=e.cost
            )
        
        all_photos = []
        total_cost = 0.0
        total_api_calls = 0
        seen_urls = set()  # Deduplicate across categories
        
        print(f"[SERPAPI] 🎯 Smart category fetching with quota: {quota}")
        
        # Step 1: Fetch from Vibe category → interior
        interior_target = quota.get("interior", 10)
        if interior_target > 0:
            vibe_cat = GMAPS_CATEGORIES["vibe"]
            photos, cost, calls = self._fetch_photos_from_category(
                data_id,
                category_id=vibe_cat.category_id,
                category_label="interior",
                max_photos=interior_target
            )
            total_cost += cost
            total_api_calls += calls
            
            for p in photos:
                if p.image_url not in seen_urls and len(all_photos) < max_total:
                    all_photos.append(p)
                    seen_urls.add(p.image_url)
            
            print(f"[SERPAPI] ✅ Vibe (interior): fetched {len(photos)} photos")
        
        # Step 2: Fetch from Food & drink category → food
        food_target = quota.get("food", 20)
        if food_target > 0:
            food_cat = GMAPS_CATEGORIES["food_drink"]
            photos, cost, calls = self._fetch_photos_from_category(
                data_id,
                category_id=food_cat.category_id,
                category_label="food",  # Will need AI to split food vs drink
                max_photos=food_target
            )
            total_cost += cost
            total_api_calls += calls
            
            for p in photos:
                if p.image_url not in seen_urls and len(all_photos) < max_total:
                    all_photos.append(p)
                    seen_urls.add(p.image_url)
            
            print(f"[SERPAPI] ✅ Food & drink: fetched {len(photos)} photos")
        
        # Step 3: Fetch from All category to fill remaining quota
        # The "All" category contains photos not in specific categories (exterior shots, 
        # staff photos, general ambiance, etc.) - always fetch to meet max_total
        remaining = max_total - len(all_photos)
        
        if remaining > 0:
            # Fetch from All - no category filter, no pre-labeling
            photos, cost, calls = self._fetch_photos_from_category(
                data_id,
                category_id=None,  # No filter = All
                category_label=None,  # Needs AI categorization
                max_photos=remaining  # Fetch enough to meet max_total
            )
            total_cost += cost
            total_api_calls += calls
            
            for p in photos:
                if p.image_url not in seen_urls and len(all_photos) < max_total:
                    all_photos.append(p)
                    seen_urls.add(p.image_url)
            
            print(f"[SERPAPI] ✅ All (general): fetched {len(photos)} photos (needs AI categorization)")
        
        # Summary
        pre_categorized = sum(1 for p in all_photos if p.category is not None)
        needs_ai = len(all_photos) - pre_categorized
        print(f"[SERPAPI] 📊 Total: {len(all_photos)} photos ({pre_categorized} pre-categorized, {needs_ai} need AI)")
        
        return PhotoFetchResult(
            photos=all_photos,
            total_available=None,
            has_more=False,  # We've fetched what we need
            provider=self.name,
            cost=total_cost,
            api_calls=total_api_calls
        )
    
    def fetch_more_photos(
        self,
        next_page_token: str,
        max_photos: int = 50,
    ) -> PhotoFetchResult:
        """Fetch additional photos using pagination token."""
        api_key = self._get_api_key()
        
        # The next_page_token contains the data_id info
        params = {
            "engine": "google_maps_photos",
            "next_page_token": next_page_token,
            "api_key": api_key
        }
        
        try:
            response = requests.get(SERPAPI_BASE_URL, params=params, timeout=15)
            
            if response.status_code != 200:
                raise ProviderError(
                    f"SerpApi error ({response.status_code}): {response.text}",
                    provider=self.name,
                    cost=self.cost_per_request
                )
            
            data = response.json()
            photos = data.get("photos", [])
            
            results = []
            for photo in photos[:max_photos]:
                source = photo.get("source", {})
                results.append(PhotoResult(
                    image_url=photo.get("image", ""),
                    thumbnail_url=photo.get("thumbnail"),
                    source_name=source.get("name"),
                    date=photo.get("date"),
                    provider=self.name,
                    photo_id=photo.get("image", "")[:100],
                    metadata={
                        "source_link": source.get("link"),
                        "source_reviews": source.get("reviews"),
                    }
                ))
            
            pagination = data.get("serpapi_pagination", {})
            
            return PhotoFetchResult(
                photos=results,
                has_more=bool(pagination.get("next_page_token")),
                next_page_token=pagination.get("next_page_token"),
                provider=self.name,
                cost=self.cost_per_request,
                api_calls=1
            )
            
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(
                f"Failed to fetch more photos: {str(e)}",
                provider=self.name,
                cost=self.cost_per_request
            )

