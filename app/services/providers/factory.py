"""
Provider factory for creating provider instances.

Reads configuration from environment variables and returns appropriate providers.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path

from .base import PhotoProvider, PlaceProvider, ProviderType
from .google_places import GooglePlacesProvider
from .serpapi import SerpApiProvider

# Load environment
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)


def get_provider_type(env_var: str, default: str = "google") -> ProviderType:
    """Get provider type from environment variable."""
    value = os.getenv(env_var, default).lower()
    
    if value in ("serpapi", "serp"):
        return ProviderType.SERPAPI
    elif value in ("google", "google_places", "places"):
        return ProviderType.GOOGLE_PLACES
    else:
        print(f"[WARNING] Unknown provider '{value}' for {env_var}, defaulting to {default}")
        return ProviderType.GOOGLE_PLACES if default == "google" else ProviderType.SERPAPI


def get_photo_provider(
    provider_type: Optional[ProviderType] = None,
    override: Optional[str] = None,
) -> PhotoProvider:
    """
    Get a photo provider instance.
    
    Priority:
    1. override parameter (for CLI overrides)
    2. provider_type parameter (for programmatic control)
    3. PHOTO_PROVIDER environment variable
    4. Default: serpapi (better photo coverage)
    
    Args:
        provider_type: Explicit provider type
        override: String override (e.g., from CLI)
        
    Returns:
        PhotoProvider instance
    """
    # Handle override from CLI
    if override:
        if override.lower() in ("serpapi", "serp"):
            return SerpApiProvider()
        elif override.lower() in ("google", "google_places", "places"):
            return GooglePlacesProvider()
    
    # Handle explicit provider type
    if provider_type:
        if provider_type == ProviderType.SERPAPI:
            return SerpApiProvider()
        else:
            return GooglePlacesProvider()
    
    # Read from environment (default to serpapi for photos - it's better)
    env_type = get_provider_type("PHOTO_PROVIDER", default="serpapi")
    
    if env_type == ProviderType.SERPAPI:
        return SerpApiProvider()
    else:
        return GooglePlacesProvider()


def get_place_provider(
    provider_type: Optional[ProviderType] = None,
    override: Optional[str] = None,
) -> PlaceProvider:
    """
    Get a place search provider instance.
    
    Priority:
    1. override parameter (for CLI overrides)
    2. provider_type parameter (for programmatic control)
    3. PLACE_PROVIDER environment variable
    4. Default: google (more reliable for place search)
    
    Args:
        provider_type: Explicit provider type
        override: String override (e.g., from CLI)
        
    Returns:
        PlaceProvider instance
    """
    # Handle override from CLI
    if override:
        if override.lower() in ("serpapi", "serp"):
            return SerpApiProvider()
        elif override.lower() in ("google", "google_places", "places"):
            return GooglePlacesProvider()
    
    # Handle explicit provider type
    if provider_type:
        if provider_type == ProviderType.SERPAPI:
            return SerpApiProvider()
        else:
            return GooglePlacesProvider()
    
    # Read from environment (default to google for place search - more reliable)
    env_type = get_provider_type("PLACE_PROVIDER", default="google")
    
    if env_type == ProviderType.SERPAPI:
        return SerpApiProvider()
    else:
        return GooglePlacesProvider()


def get_all_providers() -> dict:
    """
    Get all available providers for status/debugging.
    
    Returns:
        Dict mapping provider name to instance
    """
    providers = {}
    
    # Try to create each provider
    try:
        providers["google_places"] = GooglePlacesProvider()
    except Exception as e:
        print(f"[WARNING] Could not create GooglePlacesProvider: {e}")
    
    try:
        providers["serpapi"] = SerpApiProvider()
    except Exception as e:
        print(f"[WARNING] Could not create SerpApiProvider: {e}")
    
    return providers

