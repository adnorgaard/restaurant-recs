import os
import re
import requests
from typing import Optional, List, Dict, Tuple
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from pathlib import Path

# Load .env file from project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
GOOGLE_PLACES_BASE_URL = "https://places.googleapis.com/v1"
GOOGLE_PLACES_PHOTOS_BASE_URL = "https://places.googleapis.com/v1"


class PlacesServiceError(Exception):
    """Custom exception for Places API errors"""
    pass


def find_place_by_text(text: str) -> str:
    """
    Find a place using the Places API (New) searchText endpoint.
    
    Args:
        text: Text query (can be restaurant name, address, or even URL)
        
    Returns:
        place_id string
        
    Raises:
        PlacesServiceError: If place not found or API error occurs
    """
    if not GOOGLE_PLACES_API_KEY:
        raise PlacesServiceError("GOOGLE_PLACES_API_KEY not configured")
    
    url = f"{GOOGLE_PLACES_BASE_URL}/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
        "X-Goog-FieldMask": "places.id,places.displayName"
    }
    payload = {
        "textQuery": text
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        
        # Check for HTTP errors first
        if response.status_code != 200:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", response.text)
            except:
                error_message = response.text
            
            raise PlacesServiceError(
                f"Google Places API error ({response.status_code}): {error_message}\n"
                f"This usually means:\n"
                f"1. Your API key is invalid or missing\n"
                f"2. Required APIs are not enabled (Places API (New))\n"
                f"3. API key restrictions are blocking requests\n"
                f"4. Billing is not enabled for your Google Cloud project\n"
                f"5. Request format is incorrect"
            )
        
        data = response.json()
        
        if "error" in data:
            error = data["error"]
            error_message = error.get("message", "API request denied")
            raise PlacesServiceError(
                f"Google Places API request denied: {error_message}\n"
                f"This usually means:\n"
                f"1. Your API key is invalid or missing\n"
                f"2. Required APIs are not enabled (Places API (New))\n"
                f"3. API key restrictions are blocking requests\n"
                f"4. Billing is not enabled for your Google Cloud project"
            )
        
        places = data.get("places", [])
        if not places:
            raise PlacesServiceError("Place not found: No results returned")
        
        return places[0]["id"]
    except PlacesServiceError:
        raise
    except requests.RequestException as e:
        raise PlacesServiceError(f"Failed to find place: {str(e)}")


def get_place_id_by_name(name: str, location: Optional[str] = None) -> str:
    """
    Search for a place by name and optionally location to get the place_id.
    Uses Places API (New) searchText endpoint.
    
    Args:
        name: Restaurant name
        location: Optional location/address to narrow down search
        
    Returns:
        place_id string
        
    Raises:
        PlacesServiceError: If place not found or API error occurs
    """
    if not GOOGLE_PLACES_API_KEY:
        raise PlacesServiceError("GOOGLE_PLACES_API_KEY not configured")
    
    query = name
    if location:
        query = f"{name} {location}"
    
    # Use the same searchText endpoint as find_place_by_text
    return find_place_by_text(query)


def get_place_details(place_id: str) -> Dict:
    """
    Get detailed information about a place including photos.
    Uses Places API (New).
    
    Args:
        place_id: Google Places place_id
        
    Returns:
        Dictionary with place details including photos
        
    Raises:
        PlacesServiceError: If place not found or API error occurs
    """
    if not GOOGLE_PLACES_API_KEY:
        raise PlacesServiceError("GOOGLE_PLACES_API_KEY not configured")
    
    url = f"{GOOGLE_PLACES_BASE_URL}/places/{place_id}"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
        "X-Goog-FieldMask": "id,displayName,photos"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        # Check for HTTP errors first
        if response.status_code != 200:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", response.text)
            except:
                error_message = response.text
            
            raise PlacesServiceError(
                f"Google Places API error ({response.status_code}): {error_message}\n"
                f"This usually means:\n"
                f"1. Your API key is invalid or missing\n"
                f"2. Required APIs are not enabled (Places API (New))\n"
                f"3. API key restrictions are blocking requests\n"
                f"4. Billing is not enabled for your Google Cloud project"
            )
        
        data = response.json()
        
        if "error" in data:
            error = data["error"]
            error_message = error.get("message", "API request denied")
            raise PlacesServiceError(
                f"Google Places API request denied: {error_message}\n"
                f"This usually means:\n"
                f"1. Your API key is invalid or missing\n"
                f"2. Required APIs are not enabled (Places API (New))\n"
                f"3. API key restrictions are blocking requests\n"
                f"4. Billing is not enabled for your Google Cloud project"
            )
        
        return data
    except PlacesServiceError:
        raise
    except requests.RequestException as e:
        raise PlacesServiceError(f"Failed to get place details: {str(e)}")


def get_place_images(place_id: str, max_images: int = 10, db: Optional[Session] = None) -> List[bytes]:
    """
    Fetch restaurant images from Google Places API, with optional caching.
    
    If db session is provided, checks cache first. If cached images exist, returns them.
    Otherwise, fetches from Google API and caches the results.
    
    Args:
        place_id: Google Places place_id
        max_images: Maximum number of images to fetch (default: 10)
        db: Optional database session for caching
        
    Returns:
        List of image bytes
        
    Raises:
        PlacesServiceError: If images cannot be fetched
    """
    # Check cache first if db session is provided
    if db:
        try:
            from app.services.database_service import get_cached_images
            cached_images = get_cached_images(db, place_id, max_images)
            
            if cached_images:
                # Download images from GCS URLs
                images = []
                for cached_image in cached_images:
                    try:
                        response = requests.get(cached_image.gcs_url, timeout=10)
                        response.raise_for_status()
                        images.append(response.content)
                    except requests.RequestException:
                        # If GCS URL fails, continue to next image
                        continue
                
                if images:
                    return images
        except Exception as e:
            # If cache check fails, fall through to API fetch
            print(f"Warning: Cache check failed: {str(e)}")
    
    # No cache or cache miss - fetch from Google API
    if not GOOGLE_PLACES_API_KEY:
        raise PlacesServiceError("GOOGLE_PLACES_API_KEY not configured")
    
    # Get place details to retrieve photo references
    place_details = get_place_details(place_id)
    photos = place_details.get("photos", [])
    
    if not photos:
        raise PlacesServiceError("No photos available for this restaurant")
    
    images = []
    photo_metadata = []  # Store (image_bytes, photo_name) for caching
    
    for photo in photos[:max_images]:
        photo_name = photo.get("name")
        if not photo_name:
            continue
        
        # Fetch the actual image using Places API (New) photo endpoint
        # photo_name already includes the full path like "places/ChIJ.../photos/Aap_uEA..."
        url = f"{GOOGLE_PLACES_PHOTOS_BASE_URL}/{photo_name}/media"
        headers = {
            "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY
        }
        params = {
            "maxWidthPx": 800  # Reasonable size for API analysis
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            image_bytes = response.content
            images.append(image_bytes)
            photo_metadata.append((image_bytes, photo_name))
        except requests.RequestException as e:
            # Continue with other images if one fails
            continue
    
    if not images:
        raise PlacesServiceError("Failed to download any images")
    
    # Cache the images if db session is provided
    if db and photo_metadata:
        try:
            from app.services.database_service import cache_images
            restaurant_name = get_restaurant_name(place_id)
            # Note: category will be set later by vision_service
            # For now, we'll cache with None category and update later
            cache_images(db, place_id, restaurant_name, [
                (img_bytes, photo_name, None) for img_bytes, photo_name in photo_metadata
            ])
        except Exception as e:
            # Log error but don't fail the request
            print(f"Warning: Failed to cache images: {str(e)}")
    
    return images


def get_restaurant_name(place_id: str) -> str:
    """
    Get the restaurant name from place_id.
    
    Args:
        place_id: Google Places place_id
        
    Returns:
        Restaurant name string
    """
    place_details = get_place_details(place_id)
    # In Places API (New), the name is in displayName.text
    display_name = place_details.get("displayName", {})
    if isinstance(display_name, dict):
        return display_name.get("text", "Unknown Restaurant")
    return str(display_name) if display_name else "Unknown Restaurant"


def extract_place_id_from_url(url: str, db: Optional[Session] = None) -> str:
    """
    Extract place_id from Google Maps or Google Places URL, or search by restaurant name.
    
    First checks the database using semantic search to avoid unnecessary API calls.
    
    Supports:
    - Restaurant names: "Joe's Pizza, New York" or just "Joe's Pizza"
    - Short URLs: https://maps.app.goo.gl/... or https://goo.gl/maps/... (will follow redirects)
    - Full URLs: https://www.google.com/maps/place/...
    - URLs with place_id: https://www.google.com/maps/place/?q=place_id:ChIJ...
    
    Args:
        url: Google Maps URL, share link, or restaurant name/address
        db: Optional database session to check for cached restaurants first
        
    Returns:
        place_id string
        
    Raises:
        PlacesServiceError: If place_id cannot be extracted
    """
    from urllib.parse import unquote, urlparse, parse_qs
    
    original_input = url.strip()
    url = original_input
    
    # Step 0.5: Check database first if it looks like a text query (not a URL)
    # This avoids API calls for restaurants we've already seen
    url_patterns = [
        r'^https?://',  # Standard http/https
        r'^maps://',    # Maps protocol
        r'^goo\.gl/',   # Short URL without protocol
        r'^maps\.app\.goo\.gl/',  # Maps app short URL
    ]
    
    is_url = any(re.match(pattern, url, re.IGNORECASE) for pattern in url_patterns)
    
    if not is_url and db:
        # This looks like a restaurant name/address, not a URL
        # Check database first using semantic search
        try:
            from app.services.database_service import find_restaurant_by_text
            cached_place_id = find_restaurant_by_text(db, url, min_similarity=0.75)
            if cached_place_id:
                print(f"[DEBUG] Found restaurant in database: {url} -> {cached_place_id}")
                return cached_place_id
        except Exception as e:
            # Log but continue to API search
            print(f"[WARNING] Database search failed, falling back to API: {str(e)}")
    
    # Step 0: Check if input is just a restaurant name (not a URL)
    # If it doesn't look like a URL, treat it as a restaurant name/address
    # Check for URL patterns more comprehensively
    url_patterns = [
        r'^https?://',  # Standard http/https
        r'^maps://',    # Maps protocol
        r'^goo\.gl/',   # Short URL without protocol
        r'^maps\.app\.goo\.gl/',  # Maps app short URL
    ]
    
    is_url = any(re.match(pattern, url, re.IGNORECASE) for pattern in url_patterns)
    
    if not is_url:
        # This looks like a restaurant name or address, not a URL
        # Try multiple search strategies with better error handling
        search_query = url.strip()
        
        # Strategy 1: Try Find Place API (most flexible, handles various formats)
        try:
            place_id = find_place_by_text(search_query)
            if place_id:
                return place_id
        except PlacesServiceError as e1:
            # If it's an API configuration error (REQUEST_DENIED), re-raise immediately
            if "request denied" in str(e1).lower():
                raise
            # Strategy 2: Try Text Search API (better for restaurant names)
            try:
                place_id = get_place_id_by_name(search_query, None)
                if place_id:
                    return place_id
            except PlacesServiceError as e2:
                # If it's an API configuration error, re-raise immediately
                if "request denied" in str(e2).lower():
                    raise
                # Strategy 3: If the query contains a comma, try splitting it
                # Format: "Restaurant Name, Location"
                if ',' in search_query:
                    parts = [p.strip() for p in search_query.split(',', 1)]
                    if len(parts) == 2:
                        name, location = parts
                        try:
                            place_id = get_place_id_by_name(name, location)
                            if place_id:
                                return place_id
                        except PlacesServiceError:
                            pass
                
                # All strategies failed - provide helpful error
                error_details = []
                if str(e1):
                    error_details.append(f"Find Place API: {str(e1)}")
                if str(e2):
                    error_details.append(f"Text Search API: {str(e2)}")
                
                error_msg = "Could not find restaurant"
                if error_details:
                    error_msg += f" ({'; '.join(error_details)})"
                
                raise PlacesServiceError(
                    f"{error_msg}. "
                    "Please try one of these options:\n"
                    "1. Use a Google Maps share link: https://maps.app.goo.gl/...\n"
                    "2. Use a full Google Maps URL: https://www.google.com/maps/place/Restaurant+Name\n"
                    "3. Enter restaurant name with location (e.g., 'Joe's Pizza, New York')\n"
                    "4. Use a URL with place_id: https://www.google.com/maps/place/?q=place_id:ChIJ..."
                )
    
    # Step 1: Handle short URLs (Google Maps share links)
    # These need to be resolved to their final destination
    short_url_patterns = [
        r'https?://(maps\.app\.)?goo\.gl/',
        r'https?://maps\.app\.goo\.gl/',
        r'^goo\.gl/',  # Without protocol
        r'^maps\.app\.goo\.gl/',  # Without protocol
    ]
    
    is_short_url = any(re.search(pattern, url, re.IGNORECASE) for pattern in short_url_patterns)
    if is_short_url:
        # Add protocol if missing
        if not url.startswith('http'):
            url = 'https://' + url
        try:
            # Use GET with redirects to get the final URL
            # Some servers don't respond to HEAD requests properly
            response = requests.get(url, allow_redirects=True, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            url = response.url  # Use the final redirected URL
        except requests.RequestException as e:
            # If redirect fails, try to continue with original URL
            # Some short URLs might have place_id in them directly
            pass
    
    # Pattern 1: Direct place_id parameter (case-insensitive)
    place_id_match = re.search(r'[?&]place_id=([A-Za-z0-9_-]+)', url, re.IGNORECASE)
    if place_id_match:
        return place_id_match.group(1)
    
    # Pattern 1b: query_place_id parameter (used in some Google Maps URLs)
    query_place_id_match = re.search(r'[?&]query_place_id=([A-Za-z0-9_-]+)', url, re.IGNORECASE)
    if query_place_id_match:
        return query_place_id_match.group(1)
    
    # Pattern 1c: cid parameter (sometimes used instead of place_id)
    cid_match = re.search(r'[?&]cid=([A-Za-z0-9_-]+)', url, re.IGNORECASE)
    if cid_match:
        # CID might be a place_id, try to use it
        cid_value = cid_match.group(1)
        if len(cid_value) >= 27:  # place_ids are typically 27+ chars
            return cid_value
    
    # Pattern 2: place_id in query parameter format place_id:ChIJ...
    place_id_colon_match = re.search(r'place_id:([A-Za-z0-9_-]+)', url, re.IGNORECASE)
    if place_id_colon_match:
        return place_id_colon_match.group(1)
    
    # Pattern 3: Extract from data parameter (encoded place_id)
    # Google Maps uses !1s followed by place_id in various formats
    # Try multiple patterns for the data parameter, ordered by specificity
    data_patterns = [
        r'!1s([A-Za-z0-9_-]{27})',  # Direct !1s pattern with 27 char place_id (most specific)
        r'!1s([A-Za-z0-9_-]{27,50})',  # !1s pattern with 27-50 char place_id
        r'1s([A-Za-z0-9_-]{27})',  # 1s pattern without ! (27 char)
        r'1s([A-Za-z0-9_-]{27,50})',  # 1s pattern without ! (variable length)
        r'/data=!([^&]+)/',  # Match data parameter more broadly
        r'data=!([^&"\']+)',  # Match data parameter (non-greedy)
    ]
    
    for pattern in data_patterns:
        match = re.search(pattern, url)
        if match:
            potential_id = match.group(1)
            # Google place_ids typically start with ChIJ and are 27 characters
            # But they can vary, so check if it looks like a place_id
            if potential_id.startswith('ChIJ') and 27 <= len(potential_id) <= 50:
                return potential_id
            # Also check for other valid place_id patterns (alphanumeric, reasonable length)
            if 27 <= len(potential_id) <= 50 and re.match(r'^[A-Za-z0-9_-]+$', potential_id):
                # Prefer IDs that start with ChIJ, but also accept others if they match the pattern
                return potential_id
            # Sometimes the place_id is embedded in a longer data string
            # Look for ChIJ pattern within the matched string
            chij_match = re.search(r'(ChIJ[A-Za-z0-9_-]{23,46})', potential_id)
            if chij_match:
                extracted_id = chij_match.group(1)
                if 27 <= len(extracted_id) <= 50:
                    return extracted_id
    
    # Pattern 4: Extract from q parameter (query parameter)
    try:
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        if 'q' in query_params:
            q_value = query_params['q'][0]
            # Check if q contains place_id: format
            place_id_in_q = re.search(r'place_id:([A-Za-z0-9_-]+)', q_value, re.IGNORECASE)
            if place_id_in_q:
                return place_id_in_q.group(1)
            # If q is just a place_id (without place_id: prefix)
            if re.match(r'^[A-Za-z0-9_-]{27,}$', q_value) and q_value.startswith('ChIJ'):
                return q_value
            # If q looks like a place_id (27+ chars, alphanumeric)
            if re.match(r'^[A-Za-z0-9_-]{27,50}$', q_value):
                return q_value
    except Exception:
        # If URL parsing fails, continue with other patterns
        pass
    
    # Pattern 5: Extract restaurant name from URL path and search
    # Format: /place/Restaurant+Name/@lat,lng
    # Also handle: /place/Restaurant+Name (without coordinates)
    # Handle both /place/ and /maps/place/ patterns
    name_patterns = [
        r'/maps/place/([^/@?]+)',
        r'/place/([^/@?]+)',
        r'place/([^/@?]+)',  # Without leading slash
    ]
    
    for pattern in name_patterns:
        name_match = re.search(pattern, url, re.IGNORECASE)
        if name_match:
            name = name_match.group(1)
            # Clean up the name (remove URL encoding)
            try:
                name = unquote(name)
            except:
                # Fallback to simple replacements
                name = name.replace('+', ' ').replace('%20', ' ').replace('%2C', ',')
            # Remove any trailing parameters or fragments
            name = name.split('?')[0].split('#')[0].strip()
            # Skip if name is too short or looks like coordinates/place_id
            if len(name) > 3 and not name.startswith('ChIJ') and not re.match(r'^-?\d+\.?\d*,-?\d+\.?\d*', name):
                # Try to search for this restaurant using Find Place API (more flexible)
                try:
                    place_id = find_place_by_text(name)
                    if place_id:
                        return place_id
                except PlacesServiceError:
                    # Fallback to text search
                    try:
                        place_id = get_place_id_by_name(name)
                        if place_id:
                            return place_id
                    except PlacesServiceError:
                        # If search fails, continue to other patterns
                        continue
    
    # Pattern 6: Extract from search query parameter
    query_match = re.search(r'[?&]query=([^&]+)', url)
    if query_match:
        query = query_match.group(1)
        try:
            query = unquote(query)
        except:
            pass
        # If query contains place_id: format
        place_id_in_query = re.search(r'place_id:([A-Za-z0-9_-]+)', query)
        if place_id_in_query:
            return place_id_in_query.group(1)
        # Otherwise, search by the query text using Find Place API
        try:
            return find_place_by_text(query)
        except PlacesServiceError:
            # Fallback to text search
            try:
                return get_place_id_by_name(query)
            except PlacesServiceError:
                pass
    
    # Pattern 7: Try to extract from the entire URL by looking for place_id patterns
    # Sometimes place_ids appear in various encoded forms
    # Look for ChIJ pattern specifically first (most reliable)
    chij_matches = re.findall(r'(ChIJ[A-Za-z0-9_-]{23,46})', url)
    for potential_id in chij_matches:
        if 27 <= len(potential_id) <= 50:
            # This looks like a valid place_id
            return potential_id
    
    # Pattern 8: Try broader search for any 27-char alphanumeric strings
    all_place_id_matches = re.findall(r'([A-Za-z0-9_-]{27})', url)
    for potential_id in all_place_id_matches:
        # Prefer IDs that start with common prefixes
        if potential_id.startswith(('ChIJ', 'Ei', 'Gh')):
            return potential_id
    
    # Pattern 9: Last resort - try to extract restaurant name from any part of URL
    # and search for it using Find Place API
    # Look for text that might be a restaurant name in the URL
    restaurant_name_patterns = [
        r'/place/([^/@?&]+)',
        r'place/([^/@?&]+)',
        r'/([A-Z][a-z]+(?:\+[A-Z][a-z]+)*)',  # Capitalized words separated by +
    ]
    
    for pattern in restaurant_name_patterns:
        match = re.search(pattern, url)
        if match:
            name = match.group(1)
            # Clean and validate
            try:
                name = unquote(name).replace('+', ' ').strip()
            except:
                name = name.replace('+', ' ').strip()
            if len(name) > 3 and not name.startswith('ChIJ') and not re.match(r'^-?\d+\.?\d*', name):
                try:
                    return find_place_by_text(name)
                except PlacesServiceError:
                    try:
                        return get_place_id_by_name(name)
                    except PlacesServiceError:
                        continue
    
    # Pattern 10: Final fallback - try using the entire URL or a cleaned version as search text
    # Sometimes the URL itself or parts of it can be used as a search query
    # Extract meaningful text from URL (remove protocol, domain, common paths)
    url_text = re.sub(r'https?://(www\.)?(maps\.)?(google\.com|goo\.gl)/', '', url, flags=re.IGNORECASE)
    url_text = re.sub(r'[/?&].*', '', url_text)  # Remove query params and fragments
    try:
        url_text = unquote(url_text)
    except:
        pass
    url_text = url_text.replace('+', ' ').replace('%20', ' ').strip()
    
    # If we have meaningful text (not just empty or coordinates), try searching
    if len(url_text) > 5 and not re.match(r'^-?\d+\.?\d*', url_text):
        try:
            place_id = find_place_by_text(url_text)
            if place_id:
                return place_id
        except PlacesServiceError:
            try:
                place_id = get_place_id_by_name(url_text)
                if place_id:
                    return place_id
            except PlacesServiceError:
                pass
    
    # Pattern 11: Last resort - try the original input as-is if it wasn't a URL
    # This handles edge cases where the input might have been misclassified
    if original_input != url and not is_url:
        try:
            place_id = find_place_by_text(original_input)
            if place_id:
                return place_id
        except PlacesServiceError:
            pass
    
    # All patterns failed - provide comprehensive error message
    raise PlacesServiceError(
        f"Could not extract place_id from input: '{original_input[:100]}...' (truncated if long). "
        "Please try one of these options:\n"
        "1. Use a Google Maps share link: https://maps.app.goo.gl/...\n"
        "2. Use a full Google Maps URL: https://www.google.com/maps/place/Restaurant+Name\n"
        "3. Enter just the restaurant name (e.g., 'Joe's Pizza, New York')\n"
        "4. Use a URL with place_id: https://www.google.com/maps/place/?q=place_id:ChIJ..."
    )


def get_place_images_with_urls(place_id: str, max_images: int = 10, db: Optional[Session] = None) -> Tuple[List[bytes], List[str]]:
    """
    Fetch restaurant images from Google Places API and return both image bytes and URLs, with optional caching.
    
    If db session is provided, checks cache first. If cached images exist, returns GCS URLs.
    Otherwise, fetches from Google API and caches the results.
    
    Args:
        place_id: Google Places place_id
        max_images: Maximum number of images to fetch (default: 10)
        db: Optional database session for caching
        
    Returns:
        Tuple of (list of image bytes, list of image URLs)
        
    Raises:
        PlacesServiceError: If images cannot be fetched
    """
    # Check cache first if db session is provided
    if db:
        try:
            from app.services.database_service import get_cached_images
            cached_images = get_cached_images(db, place_id, max_images)
            
            if cached_images:
                # Download images from GCS URLs and return GCS URLs
                images = []
                image_urls = []
                for cached_image in cached_images:
                    try:
                        response = requests.get(cached_image.gcs_url, timeout=10)
                        response.raise_for_status()
                        images.append(response.content)
                        image_urls.append(cached_image.gcs_url)
                    except requests.RequestException:
                        # If GCS URL fails, continue to next image
                        continue
                
                if images:
                    return images, image_urls
        except Exception as e:
            # If cache check fails, fall through to API fetch
            print(f"Warning: Cache check failed: {str(e)}")
    
    # No cache or cache miss - fetch from Google API
    if not GOOGLE_PLACES_API_KEY:
        raise PlacesServiceError("GOOGLE_PLACES_API_KEY not configured")
    
    # Get place details to retrieve photo references
    place_details = get_place_details(place_id)
    photos = place_details.get("photos", [])
    
    if not photos:
        raise PlacesServiceError("No photos available for this restaurant")
    
    images = []
    image_urls = []
    photo_metadata = []  # Store (image_bytes, photo_name) for caching
    
    for photo in photos[:max_images]:
        photo_name = photo.get("name")
        if not photo_name:
            continue
        
        # Build the image URL using our proxy endpoint
        # photo_name already includes the full path like "places/ChIJ.../photos/Aap_uEA..."
        # We use a proxy endpoint because Places API (New) requires authentication headers
        image_url = f"/api/places/photo/{photo_name}?maxWidthPx=800"
        
        # Fetch the actual image
        try:
            headers = {
                "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY
            }
            params = {
                "maxWidthPx": 800
            }
            response = requests.get(f"{GOOGLE_PLACES_PHOTOS_BASE_URL}/{photo_name}/media", headers=headers, params=params, timeout=10)
            response.raise_for_status()
            image_bytes = response.content
            images.append(image_bytes)
            image_urls.append(image_url)
            photo_metadata.append((image_bytes, photo_name))
        except requests.RequestException as e:
            # Continue with other images if one fails
            continue
    
    if not images:
        raise PlacesServiceError("Failed to download any images")
    
    # Cache the images if db session is provided
    if db and photo_metadata:
        try:
            from app.services.database_service import cache_images
            restaurant_name = get_restaurant_name(place_id)
            # Note: category will be set later by vision_service
            # For now, we'll cache with None category and update later
            cache_images(db, place_id, restaurant_name, [
                (img_bytes, photo_name, None) for img_bytes, photo_name in photo_metadata
            ])
            # Update URLs to use GCS URLs from cache
            cached_images = get_cached_images(db, place_id, max_images)
            if cached_images:
                image_urls = [img.gcs_url for img in cached_images[:len(images)]]
        except Exception as e:
            # Log error but don't fail the request
            print(f"Warning: Failed to cache images: {str(e)}")
    
    return images, image_urls


def get_place_images_with_metadata(place_id: str, max_images: int = 10, db: Optional[Session] = None) -> Tuple[List[bytes], List[str], List[str]]:
    """
    Fetch restaurant images from Google Places API and return image bytes, URLs, and photo_names.
    Similar to get_place_images_with_urls but also returns photo_names for tracking.
    
    Args:
        place_id: Google Places place_id
        max_images: Maximum number of images to fetch (default: 10)
        db: Optional database session for caching
        
    Returns:
        Tuple of (list of image bytes, list of image URLs, list of photo_names)
        
    Raises:
        PlacesServiceError: If images cannot be fetched
    """
    # Check cache first if db session is provided
    if db:
        try:
            from app.services.database_service import get_cached_images
            cached_images = get_cached_images(db, place_id, max_images)
            
            if cached_images:
                print(f"[DEBUG] Found {len(cached_images)} cached images in database")
                # Return cached URLs and photo_names directly - NO GCS downloads, NO API calls
                # The caller should check if they need image bytes (for categorization/AI)
                # If categories and AI tags are already cached, no bytes needed!
                image_urls = [img.gcs_url for img in cached_images]
                photo_names = [img.photo_name for img in cached_images]
                # Return empty bytes - caller will only download if needed for processing
                return [], image_urls, photo_names
        except Exception as e:
            # If cache check fails, fall through to API fetch
            import traceback
            print(f"[ERROR] Cache check failed: {str(e)}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
    
    # No cache or cache miss - fetch from Google API
    if not GOOGLE_PLACES_API_KEY:
        raise PlacesServiceError("GOOGLE_PLACES_API_KEY not configured")
    
    # Get place details to retrieve photo references
    place_details = get_place_details(place_id)
    photos = place_details.get("photos", [])
    
    if not photos:
        raise PlacesServiceError("No photos available for this restaurant")
    
    images = []
    image_urls = []
    photo_names = []
    photo_metadata = []  # Store (image_bytes, photo_name) for caching
    
    for photo in photos[:max_images]:
        photo_name = photo.get("name")
        if not photo_name:
            continue
        
        # Fetch the actual image from Google Places API
        try:
            headers = {
                "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY
            }
            params = {
                "maxWidthPx": 800
            }
            response = requests.get(f"{GOOGLE_PLACES_PHOTOS_BASE_URL}/{photo_name}/media", headers=headers, params=params, timeout=10)
            response.raise_for_status()
            image_bytes = response.content
            images.append(image_bytes)
            # Use placeholder URL for now - will be updated after GCS upload
            image_urls.append(None)
            photo_names.append(photo_name)
            photo_metadata.append((image_bytes, photo_name))
        except requests.RequestException as e:
            # Continue with other images if one fails
            print(f"[WARNING] Failed to fetch image from Places API: {str(e)}")
            continue
    
    if not images:
        raise PlacesServiceError("Failed to download any images")
    
    # Cache the images to GCS if db session is provided
    # This is REQUIRED for the URLs to work - we need GCS URLs, not proxy URLs
    if db and photo_metadata:
        try:
            from app.services.database_service import cache_images
            restaurant_name = get_restaurant_name(place_id)
            print(f"[DEBUG] Caching {len(photo_metadata)} images to GCS for {place_id}")
            # Note: category will be set later by vision_service
            cached_result = cache_images(db, place_id, restaurant_name, [
                (img_bytes, photo_name, None) for img_bytes, photo_name in photo_metadata
            ])
            print(f"[DEBUG] Successfully cached {len(cached_result)} images to GCS")
            
            # Get GCS URLs directly from the cached_result (RestaurantImage objects)
            # Build a mapping from photo_name to GCS URL
            photo_to_gcs_url = {img.photo_name: img.gcs_url for img in cached_result}
            
            # Update image_urls with GCS URLs in the same order as photo_names
            new_image_urls = []
            new_images = []
            new_photo_names = []
            
            for i, pn in enumerate(photo_names):
                gcs_url = photo_to_gcs_url.get(pn)
                if gcs_url:
                    new_image_urls.append(gcs_url)
                    new_images.append(images[i])
                    new_photo_names.append(pn)
                else:
                    print(f"[WARNING] No GCS URL for photo {pn[:50]}...")
            
            images = new_images
            image_urls = new_image_urls
            photo_names = new_photo_names
            print(f"[DEBUG] ✅ Updated {len(image_urls)} image URLs to use GCS URLs")
            
        except Exception as e:
            # Log error - this is a critical failure, images won't work without GCS URLs
            import traceback
            print(f"[ERROR] Failed to cache images to GCS: {str(e)}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            raise PlacesServiceError(f"Failed to upload images to GCS: {str(e)}")
    else:
        # No database session - can't cache, so we can't get proper URLs
        raise PlacesServiceError("Database session required to cache images to GCS")
    
    if not images or not image_urls:
        raise PlacesServiceError("Failed to cache any images to GCS")
    
    return images, image_urls, photo_names

