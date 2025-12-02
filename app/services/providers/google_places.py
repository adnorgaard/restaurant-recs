"""
Google Places API provider implementation.

Wraps the existing Google Places API functionality with the provider interface.
"""

import os
import requests
from typing import List, Optional
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

GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
GOOGLE_PLACES_BASE_URL = "https://places.googleapis.com/v1"


class GooglePlacesProvider(PhotoProvider, PlaceProvider):
    """
    Google Places API (New) provider.
    
    Limitations:
    - Returns max 10 photos per place
    - No pagination for photos
    - No category filtering
    """
    
    @property
    def name(self) -> str:
        return "google_places"
    
    @property
    def cost_per_request(self) -> float:
        # Approximate costs for Places API (New)
        # Place Details: ~$0.017
        # Photo: ~$0.007
        return 0.017
    
    @property
    def cost_per_photo(self) -> float:
        return 0.007
    
    def _get_api_key(self) -> str:
        if not GOOGLE_PLACES_API_KEY:
            raise ProviderError(
                "GOOGLE_PLACES_API_KEY not configured",
                provider=self.name
            )
        return GOOGLE_PLACES_API_KEY
    
    def search_place(
        self,
        query: str,
        location: Optional[str] = None,
    ) -> PlaceResult:
        """Search for a place using Google Places Text Search."""
        api_key = self._get_api_key()
        
        search_query = query
        if location:
            search_query = f"{query} {location}"
        
        url = f"{GOOGLE_PLACES_BASE_URL}/places:searchText"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress"
        }
        payload = {"textQuery": search_query}
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code != 200:
                raise ProviderError(
                    f"Google Places API error ({response.status_code}): {response.text}",
                    provider=self.name,
                    cost=self.cost_per_request
                )
            
            data = response.json()
            places = data.get("places", [])
            
            if not places:
                raise ProviderError(
                    f"No places found for query: {search_query}",
                    provider=self.name,
                    cost=self.cost_per_request
                )
            
            place = places[0]
            display_name = place.get("displayName", {})
            
            return PlaceResult(
                place_id=place["id"],
                name=display_name.get("text", "Unknown"),
                address=place.get("formattedAddress"),
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
        """Get place details including photo references."""
        api_key = self._get_api_key()
        
        url = f"{GOOGLE_PLACES_BASE_URL}/places/{place_id}"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "id,displayName,formattedAddress,photos"
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                raise ProviderError(
                    f"Google Places API error ({response.status_code}): {response.text}",
                    provider=self.name,
                    cost=self.cost_per_request
                )
            
            data = response.json()
            display_name = data.get("displayName", {})
            
            return PlaceResult(
                place_id=data["id"],
                name=display_name.get("text", "Unknown"),
                address=data.get("formattedAddress"),
                provider=self.name,
                cost=self.cost_per_request,
                metadata={"photos": data.get("photos", [])}
            )
            
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(
                f"Failed to get place details: {str(e)}",
                provider=self.name,
                cost=self.cost_per_request
            )
    
    def fetch_photos(
        self,
        place_id: str,
        max_photos: int = 50,
        categories: Optional[List[str]] = None,
        serpapi_data_id: Optional[str] = None,
    ) -> PhotoFetchResult:
        """
        Fetch photos from Google Places API.
        
        Note: Google Places API (New) returns max 10 photos, no pagination.
        """
        api_key = self._get_api_key()
        
        # First get place details to get photo references
        place_details = self.get_place_details(place_id)
        photo_refs = place_details.metadata.get("photos", [])
        
        if not photo_refs:
            return PhotoFetchResult(
                photos=[],
                total_available=0,
                has_more=False,
                provider=self.name,
                cost=place_details.cost
            )
        
        # Limit to max_photos (but Google only gives 10 anyway)
        photo_refs = photo_refs[:min(max_photos, 10)]
        
        photos = []
        total_cost = place_details.cost
        
        for photo_ref in photo_refs:
            photo_name = photo_ref.get("name")
            if not photo_name:
                continue
            
            # Build the media URL
            url = f"{GOOGLE_PLACES_BASE_URL}/{photo_name}/media"
            params = {"maxWidthPx": 800, "key": api_key}
            
            try:
                # Get the actual image URL (follows redirect)
                response = requests.get(
                    url, 
                    params=params, 
                    headers={"X-Goog-Api-Key": api_key},
                    allow_redirects=True,
                    timeout=10
                )
                
                if response.status_code == 200:
                    photos.append(PhotoResult(
                        image_url=response.url,
                        photo_id=photo_name,
                        provider=self.name,
                        metadata={"photo_name": photo_name}
                    ))
                    total_cost += self.cost_per_photo
                    
            except Exception as e:
                print(f"[WARNING] Failed to fetch photo {photo_name}: {e}")
                continue
        
        return PhotoFetchResult(
            photos=photos,
            total_available=len(photo_refs),
            has_more=False,  # Google doesn't support pagination
            provider=self.name,
            cost=total_cost,
            api_calls=1 + len(photos)  # 1 for details + N for photos
        )
    
    def fetch_more_photos(
        self,
        next_page_token: str,
        max_photos: int = 50,
    ) -> PhotoFetchResult:
        """
        Google Places doesn't support photo pagination.
        """
        raise ProviderError(
            "Google Places API does not support photo pagination",
            provider=self.name
        )

