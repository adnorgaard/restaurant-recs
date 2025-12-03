"""
Image Quality Scoring Service

Uses GPT-4 Vision to score images on quality metrics for filtering poor images
before displaying them on the website.

SCORING METRICS (all 0.0-1.0, higher = better):
- People: Is a person the main subject? (1.0 = no people, 0.0 = people are focus)
- Image Quality: Is the image clear enough to display? (1.0 = clear/sharp, 0.0 = unusable)

METADATA TAGS (for future filtering):
- Time of Day: "day", "night", "unknown"
- Indoor/Outdoor: "indoor", "outdoor", "unknown"

THRESHOLDS (configured in app/config/ai_versions.py):
- QUALITY_PEOPLE_THRESHOLD = 0.6 (reject if people are main subject)
- IMAGE_QUALITY_THRESHOLD = 0.5 (reject if image not clear enough)

PROCESSING FLOW (Option B):
1. Cache ALL images from API
2. Score ALL images for quality
3. Filter to images passing ALL thresholds
4. Apply quota selection (e.g., 8 food, 8 interior, 2 exterior, 2 drink)
5. Mark selected images as is_displayed=True

See QUALITY_SCORING.md for full documentation.
"""

import os
import base64
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

from sqlalchemy.orm import Session
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

from app.models.database import RestaurantImage
from app.config.ai_versions import (
    QUALITY_VERSION,
    QUALITY_PEOPLE_THRESHOLD,
    IMAGE_QUALITY_THRESHOLD,
    get_active_version,
)

# Load .env file from project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class QualityServiceError(Exception):
    """Custom exception for Quality Service errors"""
    pass


@dataclass
class QualityScores:
    """Quality scores for an image."""
    people_score: float  # 1.0 = no people as subject, 0.0 = people are main subject
    image_quality_score: float  # 1.0 = clear/sharp, 0.0 = unusable
    time_of_day: str  # "day", "night", "unknown"
    indoor_outdoor: str  # "indoor", "outdoor", "unknown"
    
    def passes_thresholds(
        self,
        people_threshold: float = QUALITY_PEOPLE_THRESHOLD,
        image_quality_threshold: float = IMAGE_QUALITY_THRESHOLD,
    ) -> bool:
        """Check if all scores pass the configured thresholds."""
        return (
            self.people_score > people_threshold and
            self.image_quality_score > image_quality_threshold
        )
    
    def to_dict(self) -> Dict[str, any]:
        return {
            "people_score": self.people_score,
            "image_quality_score": self.image_quality_score,
            "time_of_day": self.time_of_day,
            "indoor_outdoor": self.indoor_outdoor,
        }


# Quality scoring prompt - single API call for all metrics
QUALITY_SCORING_PROMPT = """Analyze this restaurant image. Provide scores from 0.0 to 1.0:

1. PEOPLE_SCORE: How much are people NOT the main subject?
   - 1.0 = No people visible, or people are minor background elements
   - 0.0 = People are clearly the main subject (portrait, group photo, selfie)
   
   Score continuously between 0.0 and 1.0.

2. IMAGE_QUALITY_SCORE: Is this image clear enough to display on a restaurant website?
   
   Consider: Can a viewer clearly see and appreciate the content?
   - Is the subject visible and identifiable?
   - Is it reasonably sharp/in focus?
   - Can you see important details (textures, colors, shapes)?
   
   - 1.0 = Excellent - sharp, clear, all details visible
   - 0.0 = Unusable - cannot make out what's in the image
   
   Score continuously between 0.0 and 1.0.
   
   Do NOT penalize:
   - Ambient, moody, or nighttime lighting (if content is still visible)
   - Minor grain or noise (if content is still clear)

3. TIME_OF_DAY: When does this appear to be taken?
   - "day" = Daytime (natural daylight visible)
   - "night" = Nighttime (dark outside or evening ambiance, etc.)
   - "unknown" = Cannot determine

4. INDOOR_OUTDOOR: Where was this taken?
   - "indoor" = Inside a building
   - "outdoor" = Outside
   - "unknown" = Cannot determine

Respond in JSON format only:
{"people_score": 0.0, "image_quality_score": 0.0, "time_of_day": "unknown", "indoor_outdoor": "unknown"}"""


def score_image_quality(image_bytes: bytes) -> QualityScores:
    """
    Score a single image for quality metrics using GPT-4 Vision.
    
    Args:
        image_bytes: Image data as bytes
        
    Returns:
        QualityScores with people_score, image_quality_score, time_of_day, indoor_outdoor
        
    Raises:
        QualityServiceError: If scoring fails
    """
    if not OPENAI_API_KEY:
        raise QualityServiceError("OPENAI_API_KEY not configured")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Convert image to base64
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": QUALITY_SCORING_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=150,
            temperature=0.1  # Low temperature for consistent scoring
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        # Handle potential markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        
        # Validate and clamp scores to 0.0-1.0 range
        people_score = max(0.0, min(1.0, float(result.get("people_score", 0.5))))
        image_quality_score = max(0.0, min(1.0, float(result.get("image_quality_score", 0.5))))
        
        # Validate string fields
        time_of_day = result.get("time_of_day", "unknown")
        if time_of_day not in ("day", "night", "unknown"):
            time_of_day = "unknown"
        
        indoor_outdoor = result.get("indoor_outdoor", "unknown")
        if indoor_outdoor not in ("indoor", "outdoor", "unknown"):
            indoor_outdoor = "unknown"
        
        return QualityScores(
            people_score=people_score,
            image_quality_score=image_quality_score,
            time_of_day=time_of_day,
            indoor_outdoor=indoor_outdoor,
        )
        
    except json.JSONDecodeError as e:
        raise QualityServiceError(f"Failed to parse quality scores: {e}")
    except Exception as e:
        raise QualityServiceError(f"Failed to score image quality: {e}")


def score_image_quality_from_url(image_url: str, timeout: int = 10) -> QualityScores:
    """
    Download an image from URL and score its quality.
    
    Args:
        image_url: URL to download image from
        timeout: Download timeout in seconds
        
    Returns:
        QualityScores
        
    Raises:
        QualityServiceError: If download or scoring fails
    """
    try:
        response = requests.get(image_url, timeout=timeout)
        response.raise_for_status()
        return score_image_quality(response.content)
    except requests.RequestException as e:
        raise QualityServiceError(f"Failed to download image: {e}")


def score_images_batch(
    images: List[Tuple[int, str]],  # List of (image_id, gcs_url)
    max_workers: int = 5,
) -> Dict[int, QualityScores]:
    """
    Score multiple images in parallel.
    
    Args:
        images: List of (image_id, gcs_url) tuples
        max_workers: Maximum concurrent API calls (respect OpenAI rate limits)
        
    Returns:
        Dict mapping image_id to QualityScores (only successful scores included)
    """
    results = {}
    
    def score_single(image_id: int, url: str) -> Tuple[int, Optional[QualityScores], Optional[str]]:
        """Score a single image, returning (id, scores, error)."""
        try:
            scores = score_image_quality_from_url(url)
            return (image_id, scores, None)
        except Exception as e:
            return (image_id, None, str(e))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(score_single, img_id, url): img_id
            for img_id, url in images
        }
        
        for future in as_completed(futures):
            img_id, scores, error = future.result()
            if scores:
                results[img_id] = scores
            else:
                print(f"[WARNING] Failed to score image {img_id}: {error}")
    
    return results


def update_image_quality_scores(
    db: Session,
    image_id: int,
    scores: QualityScores,
    version: Optional[str] = None,
) -> RestaurantImage:
    """
    Update quality scores for an image in the database.
    
    Args:
        db: Database session
        image_id: ID of the image to update
        scores: QualityScores to store
        version: Quality version string (defaults to current QUALITY_VERSION)
        
    Returns:
        Updated RestaurantImage
        
    Raises:
        QualityServiceError: If image not found
    """
    image = db.query(RestaurantImage).filter(RestaurantImage.id == image_id).first()
    
    if not image:
        raise QualityServiceError(f"Image not found: {image_id}")
    
    image.people_confidence_score = scores.people_score
    image.image_quality_score = scores.image_quality_score
    image.time_of_day = scores.time_of_day
    image.indoor_outdoor = scores.indoor_outdoor
    image.quality_version = version or QUALITY_VERSION
    image.quality_scored_at = datetime.now(timezone.utc)
    
    db.commit()
    db.refresh(image)
    
    return image


def update_image_display_status(
    db: Session,
    image_id: int,
    is_displayed: bool,
) -> RestaurantImage:
    """
    Update the is_displayed flag for an image.
    
    Args:
        db: Database session
        image_id: ID of the image to update
        is_displayed: Whether the image should be displayed
        
    Returns:
        Updated RestaurantImage
    """
    image = db.query(RestaurantImage).filter(RestaurantImage.id == image_id).first()
    
    if not image:
        raise QualityServiceError(f"Image not found: {image_id}")
    
    image.is_displayed = is_displayed
    db.commit()
    db.refresh(image)
    
    return image


def get_images_needing_quality_scoring(
    db: Session,
    place_id: str,
    max_images: Optional[int] = None,
) -> List[RestaurantImage]:
    """
    Get images that need quality scoring (no scores or outdated version).
    
    Args:
        db: Database session
        place_id: Restaurant place_id
        max_images: Optional limit on number of images
        
    Returns:
        List of RestaurantImage records needing scoring
    """
    from app.models.database import Restaurant
    
    current_version = get_active_version(db, "quality")
    
    query = db.query(RestaurantImage).join(Restaurant).filter(
        Restaurant.place_id == place_id
    ).filter(
        # No quality scores OR outdated version
        (RestaurantImage.quality_version == None) |  # noqa: E711
        (RestaurantImage.quality_version != current_version)
    )
    
    if max_images:
        query = query.limit(max_images)
    
    return query.all()


def get_displayable_images(
    db: Session,
    place_id: str,
    people_threshold: float = QUALITY_PEOPLE_THRESHOLD,
    image_quality_threshold: float = IMAGE_QUALITY_THRESHOLD,
) -> List[RestaurantImage]:
    """
    Get images that pass quality thresholds and can be displayed.
    
    Args:
        db: Database session
        place_id: Restaurant place_id
        people_threshold: Minimum people score
        image_quality_threshold: Minimum image quality score
        
    Returns:
        List of RestaurantImage records that pass all thresholds
    """
    from app.models.database import Restaurant
    
    return db.query(RestaurantImage).join(Restaurant).filter(
        Restaurant.place_id == place_id,
        RestaurantImage.people_confidence_score > people_threshold,
        RestaurantImage.image_quality_score > image_quality_threshold,
    ).all()


def score_and_filter_images_for_restaurant(
    db: Session,
    place_id: str,
    max_workers: int = 5,
    people_threshold: float = QUALITY_PEOPLE_THRESHOLD,
    image_quality_threshold: float = IMAGE_QUALITY_THRESHOLD,
) -> Tuple[List[RestaurantImage], List[RestaurantImage]]:
    """
    Score all images for a restaurant and determine which pass quality thresholds.
    
    This is the main entry point for the Option B flow:
    1. Get all cached images
    2. Score images that need scoring
    3. Filter to images that pass thresholds
    4. Return (passing_images, all_scored_images)
    
    Args:
        db: Database session
        place_id: Restaurant place_id
        max_workers: Maximum concurrent API calls
        people_threshold: Minimum people score to pass
        image_quality_threshold: Minimum image quality score to pass
        
    Returns:
        Tuple of (images_passing_quality, all_images_with_scores)
    """
    from app.models.database import Restaurant
    from app.services.database_service import get_cached_images
    
    current_version = get_active_version(db, "quality")
    
    # Get all cached images for this restaurant
    all_images = get_cached_images(db, place_id, max_images=100)
    
    if not all_images:
        return [], []
    
    # Identify images needing scoring
    images_to_score = []
    for img in all_images:
        needs_scoring = (
            img.quality_version is None or
            img.quality_version != current_version
        )
        if needs_scoring:
            images_to_score.append((img.id, img.gcs_url))
    
    print(f"[QUALITY] {len(all_images)} total images, {len(images_to_score)} need scoring")
    
    # Score images in parallel
    if images_to_score:
        print(f"[QUALITY] Scoring {len(images_to_score)} images with GPT-4 Vision...")
        scored = score_images_batch(images_to_score, max_workers=max_workers)
        
        # Update database with scores
        for img_id, scores in scored.items():
            update_image_quality_scores(db, img_id, scores, version=current_version)
        
        print(f"[QUALITY] Successfully scored {len(scored)} images")
        
        # Refresh images from database
        all_images = get_cached_images(db, place_id, max_images=100)
    
    # Filter to images that pass thresholds
    passing_images = []
    for img in all_images:
        if (
            img.people_confidence_score is not None and
            img.image_quality_score is not None and
            img.people_confidence_score > people_threshold and
            img.image_quality_score > image_quality_threshold
        ):
            passing_images.append(img)
    
    print(f"[QUALITY] {len(passing_images)}/{len(all_images)} images pass quality thresholds")
    
    return passing_images, all_images


def apply_quality_filter_and_select(
    db: Session,
    place_id: str,
    quota: Optional[Dict[str, int]] = None,
    max_workers: int = 5,
    people_threshold: float = QUALITY_PEOPLE_THRESHOLD,
    image_quality_threshold: float = IMAGE_QUALITY_THRESHOLD,
) -> List[RestaurantImage]:
    """
    Full quality filtering flow: score images, filter by quality, then select by quota.
    
    This implements Option B:
    1. Score ALL cached images for quality
    2. Filter to only images that pass quality thresholds
    3. Apply quota selection to quality-passing images
    4. Mark selected images as is_displayed=True
    
    Args:
        db: Database session
        place_id: Restaurant place_id
        quota: Category quota for selection (default: food=10, interior=10)
        max_workers: Maximum concurrent API calls for scoring
        people_threshold: Minimum people score
        image_quality_threshold: Minimum image quality score
        
    Returns:
        List of RestaurantImage records selected for display
    """
    from app.services.vision_service import select_images_by_quota
    
    # Default quota
    if quota is None:
        quota = {"food": 10, "interior": 10}
    
    # Step 1 & 2: Score and filter images
    passing_images, all_images = score_and_filter_images_for_restaurant(
        db, place_id, max_workers,
        people_threshold, image_quality_threshold
    )
    
    if not passing_images:
        print(f"[QUALITY] ⚠️ No images passed quality thresholds for {place_id}")
        return []
    
    # Step 3: Apply quota selection to passing images
    # Build categorized list for select_images_by_quota
    categorized = [(b"", img.category or "other") for img in passing_images]
    _, selected_indices = select_images_by_quota(categorized, quota=quota, max_bar=2)
    
    selected_images = [passing_images[i] for i in selected_indices if i < len(passing_images)]
    
    # Step 4: Update is_displayed flags
    # First, reset all images for this restaurant to is_displayed=False
    for img in all_images:
        if img.is_displayed:
            img.is_displayed = False
    
    # Then mark selected images as displayed
    for img in selected_images:
        img.is_displayed = True
    
    db.commit()
    
    print(f"[QUALITY] ✅ Selected {len(selected_images)} images for display")
    
    return selected_images


# =============================================================================
# Smart Fetch with Inline Quality Scoring
# =============================================================================

# Category mapping for SerpAPI
CATEGORY_CONFIG = {
    "food": {
        "serpapi_category_id": "CgIYIA",  # "Food & drink"
        "serpapi_label": "food",
    },
    "interior": {
        "serpapi_category_id": "CgIYIg",  # "Vibe"
        "serpapi_label": "interior",
    },
}


def fetch_and_score_category(
    db: Session,
    place_id: str,
    restaurant_id: int,
    data_id: str,
    category: str,
    quota: int,
    max_images: int = 30,
    score_batch_size: int = 5,
) -> Tuple[int, int, int]:
    """
    Fetch images from a category, scoring them inline until quota is met.
    
    Args:
        db: Database session
        place_id: Google Places ID
        restaurant_id: Database restaurant ID
        data_id: SerpAPI data_id
        category: Category to fetch ("food" or "interior")
        quota: Target number of quality-passing images
        max_images: Maximum images to fetch before giving up (default 30)
        score_batch_size: Number of images to score in parallel (default 5)
        
    Returns:
        Tuple of (quality_passing_count, total_fetched, total_cached)
    """
    from app.services.providers.serpapi import SerpApiProvider, GMAPS_CATEGORIES
    from app.services.storage_service import upload_image_to_gcs, GCS_BUCKET_NAME
    
    config = CATEGORY_CONFIG.get(category)
    if not config:
        print(f"[FETCH] ❌ Unknown category: {category}")
        return 0, 0, 0
    
    provider = SerpApiProvider()
    category_id = config["serpapi_category_id"]
    category_label = config["serpapi_label"]
    
    quality_passing = 0
    total_fetched = 0
    total_cached = 0
    next_page_token = None
    seen_urls = set()
    
    current_version = get_active_version(db, "quality")
    
    print(f"[FETCH] 🎯 {category}: fetching until {quota} quality images (max {max_images})")
    
    while quality_passing < quota and total_fetched < max_images:
        # Fetch one page
        photos, next_token, cost = provider.fetch_category_page(
            data_id=data_id,
            category_id=category_id,
            category_label=category_label,
            next_page_token=next_page_token,
        )
        
        if not photos:
            print(f"[FETCH] {category}: no more photos available from SerpAPI")
            break
        
        # Filter duplicates
        new_photos = [p for p in photos if p.image_url not in seen_urls]
        for p in new_photos:
            seen_urls.add(p.image_url)
        
        if not new_photos:
            next_page_token = next_token
            if not next_token:
                break
            continue
        
        total_fetched += len(new_photos)
        print(f"[FETCH] {category}: fetched {len(new_photos)} new photos (total: {total_fetched})")
        
        # Download, cache, and score in batches
        for i in range(0, len(new_photos), score_batch_size):
            if quality_passing >= quota:
                print(f"[FETCH] {category}: quota met! ({quality_passing}/{quota})")
                break
            
            batch = new_photos[i:i + score_batch_size]
            
            # Download images in parallel
            def download_and_upload(photo):
                try:
                    response = requests.get(photo.image_url, timeout=15)
                    response.raise_for_status()
                    image_bytes = response.content
                    
                    # Generate photo_name from URL
                    import hashlib
                    url_hash = hashlib.md5(photo.image_url.encode()).hexdigest()[:16]
                    photo_name = f"serpapi_{url_hash}"
                    
                    # Upload to GCS - returns (public_url, bucket_path) tuple
                    gcs_result = upload_image_to_gcs(
                        image_bytes=image_bytes,
                        place_id=place_id,
                        photo_name=photo_name,
                    )
                    
                    # Handle both tuple and string returns
                    if isinstance(gcs_result, tuple):
                        gcs_url, bucket_path = gcs_result
                    else:
                        gcs_url = gcs_result
                        bucket_path = f"images/{place_id}/{photo_name}.jpg"
                    
                    return {
                        "photo": photo,
                        "photo_name": photo_name,
                        "gcs_url": gcs_url,
                        "bucket_path": bucket_path,
                        "image_bytes": image_bytes,
                        "success": True,
                    }
                except Exception as e:
                    print(f"[FETCH] Failed to download/upload: {e}")
                    return {"photo": photo, "success": False}
            
            # Download batch in parallel
            download_results = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(download_and_upload, p) for p in batch]
                for future in as_completed(futures):
                    download_results.append(future.result())
            
            # Cache successful downloads to database
            successful = [r for r in download_results if r["success"]]
            
            for result in successful:
                # Create database record directly (we've already uploaded to GCS)
                photo_category = result["photo"].category if result["photo"].category else category_label
                
                # Check if image already exists in database
                existing = db.query(RestaurantImage).filter(
                    RestaurantImage.restaurant_id == restaurant_id,
                    RestaurantImage.photo_name == result["photo_name"],
                ).first()
                
                if existing:
                    # Update category if needed
                    if not existing.category and photo_category:
                        existing.category = photo_category
                    total_cached += 1
                    continue
                
                # Use bucket_path from upload result
                gcs_url = result["gcs_url"]
                bucket_path = result.get("bucket_path", f"images/{place_id}/{result['photo_name']}.jpg")
                
                # Create new image record
                new_image = RestaurantImage(
                    restaurant_id=restaurant_id,
                    photo_name=result["photo_name"],
                    gcs_url=gcs_url,
                    gcs_bucket_path=bucket_path,
                    category=photo_category,
                )
                db.add(new_image)
                total_cached += 1
            
            db.commit()
            
            # Score for quality in parallel
            def score_single(result):
                try:
                    scores = score_image_quality(result["image_bytes"])
                    return {
                        "photo_name": result["photo_name"],
                        "scores": scores,
                        "passes": scores.passes_thresholds(),
                    }
                except Exception as e:
                    print(f"[FETCH] Failed to score: {e}")
                    return {"photo_name": result["photo_name"], "scores": None, "passes": False}
            
            score_results = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(score_single, r) for r in successful]
                for future in as_completed(futures):
                    score_results.append(future.result())
            
            # Update database with scores
            for result in score_results:
                if result["scores"]:
                    # Find the image in database and update
                    image = db.query(RestaurantImage).filter(
                        RestaurantImage.restaurant_id == restaurant_id,
                        RestaurantImage.photo_name == result["photo_name"],
                    ).first()
                    
                    if image:
                        image.people_confidence_score = result["scores"].people_score
                        image.image_quality_score = result["scores"].image_quality_score
                        image.time_of_day = result["scores"].time_of_day
                        image.indoor_outdoor = result["scores"].indoor_outdoor
                        image.quality_version = current_version
                        image.quality_scored_at = datetime.now(timezone.utc)
                    
                    if result["passes"]:
                        quality_passing += 1
            
            db.commit()
            
            print(f"[FETCH] {category}: scored {len(score_results)}, {sum(1 for r in score_results if r['passes'])} passed (total quality: {quality_passing}/{quota})")
        
        # Check if we should continue
        if quality_passing >= quota:
            break
        
        next_page_token = next_token
        if not next_token:
            print(f"[FETCH] {category}: no more pages available")
            break
    
    print(f"[FETCH] {category}: DONE - {quality_passing} quality images from {total_fetched} fetched, {total_cached} cached")
    return quality_passing, total_fetched, total_cached


def smart_fetch_and_score(
    db: Session,
    place_id: str,
    quota: Optional[Dict[str, int]] = None,
    max_per_category: int = 30,
) -> Dict[str, int]:
    """
    Smart fetch that pulls from specific categories until quotas are met.
    
    Fetches from "Food & drink" and "Vibe" categories sequentially,
    scoring images inline and stopping when quotas are met.
    
    Note: Categories are fetched sequentially to avoid SQLAlchemy session
    threading issues. Within each category, image downloads and scoring
    are parallelized for speed.
    
    Args:
        db: Database session
        place_id: Google Places ID
        quota: Target per category. Default: {"food": 10, "interior": 10}
        max_per_category: Max images to fetch per category before giving up
        
    Returns:
        Dict with quality counts per category
    """
    from app.services.providers.serpapi import SerpApiProvider
    from app.models.database import Restaurant
    
    if quota is None:
        quota = {"food": 10, "interior": 10}
    
    # Get restaurant
    restaurant = db.query(Restaurant).filter(Restaurant.place_id == place_id).first()
    if not restaurant:
        print(f"[FETCH] ❌ Restaurant not found for place_id: {place_id}")
        return {}
    
    # Get SerpAPI data_id
    provider = SerpApiProvider()
    try:
        data_id = provider.get_data_id_for_place(place_id, restaurant.serpapi_data_id)
        
        # Cache data_id if not already stored
        if not restaurant.serpapi_data_id and data_id:
            restaurant.serpapi_data_id = data_id
            db.commit()
    except Exception as e:
        print(f"[FETCH] ❌ Failed to get SerpAPI data_id: {e}")
        return {}
    
    print(f"[FETCH] 🚀 Smart fetch for {restaurant.name} with quota: {quota}")
    
    # Fetch categories sequentially (to avoid SQLAlchemy session threading issues)
    # Within each category, downloads and scoring are parallelized
    results = {}
    
    for category, target in quota.items():
        if category in CATEGORY_CONFIG:
            try:
                quality_count, fetched, cached = fetch_and_score_category(
                    db, place_id, restaurant.id, data_id,
                    category, target, max_per_category,
                )
                results[category] = quality_count
                print(f"[FETCH] ✅ {category}: {quality_count}/{target} quality images")
            except Exception as e:
                print(f"[FETCH] ❌ {category} failed: {e}")
                import traceback
                traceback.print_exc()
                results[category] = 0
    
    # Summary
    total_quality = sum(results.values())
    total_quota = sum(quota.values())
    print(f"[FETCH] 📊 Summary: {total_quality}/{total_quota} quality images across categories")
    
    return results

