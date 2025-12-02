"""
Provider abstraction layer for restaurant data services.

This module provides a unified interface for fetching restaurant data
from multiple providers (Google Places, SerpApi, etc.).
"""

from .base import (
    PhotoProvider,
    PlaceProvider,
    PhotoResult,
    PhotoFetchResult,
    PlaceResult,
    ProviderError,
    ProviderType,
)
from .google_places import GooglePlacesProvider
from .serpapi import SerpApiProvider
from .factory import get_photo_provider, get_place_provider

__all__ = [
    "PhotoProvider",
    "PlaceProvider", 
    "PhotoResult",
    "PhotoFetchResult",
    "PlaceResult",
    "ProviderError",
    "ProviderType",
    "GooglePlacesProvider",
    "SerpApiProvider",
    "get_photo_provider",
    "get_place_provider",
]

