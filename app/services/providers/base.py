"""
Base classes for provider abstraction layer.

Defines the interface that all providers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class ProviderType(Enum):
    """Supported provider types."""
    GOOGLE_PLACES = "google"
    SERPAPI = "serpapi"


class ProviderError(Exception):
    """Base exception for provider errors."""
    def __init__(self, message: str, provider: str, cost: float = 0.0):
        super().__init__(message)
        self.provider = provider
        self.cost = cost  # Track cost even on errors


@dataclass
class PhotoResult:
    """Result from fetching a single photo."""
    image_url: str  # Full-size image URL
    thumbnail_url: Optional[str] = None
    category: Optional[str] = None  # e.g., "food", "interior", "exterior"
    source_name: Optional[str] = None  # Who uploaded it
    date: Optional[str] = None  # When it was uploaded
    provider: str = "unknown"
    
    # For tracking/caching
    photo_id: Optional[str] = None  # Unique identifier from the provider
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhotoFetchResult:
    """Result from fetching photos for a place."""
    photos: List[PhotoResult]
    total_available: Optional[int] = None  # Total photos available (if known)
    has_more: bool = False  # Whether more photos can be fetched
    next_page_token: Optional[str] = None  # For pagination
    provider: str = "unknown"
    cost: float = 0.0  # Estimated cost of this request
    api_calls: int = 1  # Number of API calls made


@dataclass
class PlaceResult:
    """Result from searching for a place."""
    place_id: str  # Google Places ID
    name: str
    address: Optional[str] = None
    serpapi_data_id: Optional[str] = None  # SerpApi's data_id
    provider: str = "unknown"
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PhotoProvider(ABC):
    """Abstract base class for photo providers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging/tracking."""
        pass
    
    @property
    @abstractmethod
    def cost_per_request(self) -> float:
        """Estimated cost per API request in USD."""
        pass
    
    @abstractmethod
    def fetch_photos(
        self,
        place_id: str,
        max_photos: int = 50,
        categories: Optional[List[str]] = None,
        serpapi_data_id: Optional[str] = None,
    ) -> PhotoFetchResult:
        """
        Fetch photos for a place.
        
        Args:
            place_id: Google Places ID
            max_photos: Maximum number of photos to fetch
            categories: Optional list of categories to filter by
            serpapi_data_id: Optional SerpApi data_id (for SerpApi provider)
            
        Returns:
            PhotoFetchResult with photos and metadata
        """
        pass
    
    @abstractmethod
    def fetch_more_photos(
        self,
        next_page_token: str,
        max_photos: int = 50,
    ) -> PhotoFetchResult:
        """
        Fetch additional photos using pagination token.
        
        Args:
            next_page_token: Token from previous fetch
            max_photos: Maximum additional photos to fetch
            
        Returns:
            PhotoFetchResult with additional photos
        """
        pass
    
    def download_photo(self, photo: PhotoResult) -> bytes:
        """
        Download photo bytes from URL.
        
        Args:
            photo: PhotoResult with image_url
            
        Returns:
            Image bytes
        """
        import requests
        response = requests.get(photo.image_url, timeout=30)
        response.raise_for_status()
        return response.content


class PlaceProvider(ABC):
    """Abstract base class for place search providers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging/tracking."""
        pass
    
    @property
    @abstractmethod
    def cost_per_request(self) -> float:
        """Estimated cost per API request in USD."""
        pass
    
    @abstractmethod
    def search_place(
        self,
        query: str,
        location: Optional[str] = None,
    ) -> PlaceResult:
        """
        Search for a place by name/query.
        
        Args:
            query: Search query (restaurant name)
            location: Optional location to narrow search
            
        Returns:
            PlaceResult with place_id and details
        """
        pass
    
    @abstractmethod
    def get_place_details(
        self,
        place_id: str,
    ) -> PlaceResult:
        """
        Get details for a specific place.
        
        Args:
            place_id: Google Places ID
            
        Returns:
            PlaceResult with place details
        """
        pass

