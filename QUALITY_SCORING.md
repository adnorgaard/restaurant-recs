# Image Quality Scoring System

This document describes the AI-powered image quality scoring system that filters out poor-quality restaurant images before displaying them on the website.

## Overview

The quality scoring system uses GPT-4 Vision to analyze each image and assign scores for quality metrics, plus metadata tags for future filtering:

### Quality Scores (used for filtering)

| Metric | What It Measures | Score Direction |
|--------|------------------|-----------------|
| **People Score** | Are people the main subject? | Higher = less people-focused |
| **Image Quality Score** | Is the image clear enough to display? | Higher = clearer/sharper |

### Metadata Tags (for future filtering)

| Tag | What It Captures | Values |
|-----|------------------|--------|
| **Time of Day** | When was the photo taken? | `day`, `night`, `unknown` |
| **Indoor/Outdoor** | Where was it taken? | `indoor`, `outdoor`, `unknown` |

All scores range from **0.0 to 1.0**, where **higher is always better** (more displayable).

## Score Interpretation

### People Confidence Score

| Score | Interpretation |
|-------|----------------|
| 1.0 | No people visible, or people are minor background elements |
| 0.5 | People visible but not the focus (diners in background, staff partially visible) |
| 0.0 | People are clearly the main subject (portrait, group photo, selfie) |

### Image Quality Score

This is a unified score that evaluates overall image clarity and visibility:

| Score | Interpretation |
|-------|----------------|
| 1.0 | Excellent - sharp, clear, all details visible |
| 0.7 | Good - content clearly visible, minor issues acceptable |
| 0.5 | Adequate - content identifiable, some details lost |
| 0.3 | Poor - subject barely identifiable |
| 0.0 | Unusable - cannot make out what's in the image |

**Important**: The image quality score does NOT penalize:
- Ambient, moody, or nighttime lighting (if content is still visible)
- Minor grain or noise (if content is still clear)

A well-lit nighttime photo where you can clearly see the food should score just as high as a daytime photo.

## Configuration

Thresholds are configured in `app/config/ai_versions.py`:

```python
# Quality Scoring Thresholds
# Images must score ABOVE these thresholds to be displayed (higher = better)

QUALITY_PEOPLE_THRESHOLD = 0.6      # Display if people_confidence_score > 0.6
IMAGE_QUALITY_THRESHOLD = 0.5       # Display if image_quality_score > 0.5
```

### Adjusting Thresholds

- **Increase thresholds** → Stricter filtering, fewer images displayed
- **Decrease thresholds** → More lenient, more images displayed

Example: If too many photos are being filtered:
```python
IMAGE_QUALITY_THRESHOLD = 0.4  # More lenient on quality
```

## Database Schema

Quality scores are stored on the `restaurant_images` table:

| Column | Type | Description |
|--------|------|-------------|
| `people_confidence_score` | Float | 0.0-1.0, higher = less people-focused |
| `image_quality_score` | Float | 0.0-1.0, higher = clearer/sharper |
| `time_of_day` | String | `day`, `night`, or `unknown` |
| `indoor_outdoor` | String | `indoor`, `outdoor`, or `unknown` |
| `quality_version` | String | Version of the scoring prompt (e.g., "v1.1") |
| `quality_scored_at` | DateTime | When the image was scored |
| `is_displayed` | Boolean | Whether the image passes quality and is selected for display |

### Deprecated Columns

The following columns are deprecated as of v1.1 and are no longer written to:
- `lighting_confidence_score` - replaced by `image_quality_score`
- `blur_confidence_score` - replaced by `image_quality_score`

## Processing Flow

The quality scoring follows **Option B** (recommended):

```
1. Cache ALL images from API (SerpAPI/Google)
     ↓
2. Categorize images (food, interior, exterior, drink, bar, etc.)
     ↓
3. Score ALL cached images for quality (GPT-4 Vision)
   - people_score: Are people the main subject?
   - image_quality_score: Is the image clear enough?
   - time_of_day: When was it taken? (metadata)
   - indoor_outdoor: Where was it taken? (metadata)
     ↓
4. Filter to images that pass ALL thresholds
   (only quality images proceed)
     ↓
5. Apply quota selection to QUALITY-PASSING images by category
   Default quota: 10 food, 10 interior = 20 total
   
   For each category:
   - Take up to N images from quality-passing images in that category
   - If a category doesn't have enough quality images, remaining slots
     are filled from other quality-passing categories
     ↓
6. Mark selected images as is_displayed=True
     ↓
7. Generate AI tags/description for the DISPLAYED images
   (ensures tags match what's actually shown)
```

**Key point**: Quota selection happens AFTER quality filtering. This ensures:
- All displayed images pass quality thresholds
- Categories are filled with the best available images
- AI tags are generated for exactly the images being displayed

## Usage

### Via Bulk Processing CLI

```bash
# Run quality scoring on all restaurants that need it
python scripts/bulk_process.py --db-all --components quality

# Run quality scoring on specific restaurants
python scripts/bulk_process.py --place-ids ChIJxxx ChIJyyy --components quality

# Force re-score all images (ignore existing scores)
python scripts/bulk_process.py --db-all --components quality --force

# Dry run to preview what would be scored
python scripts/bulk_process.py --db-all --components quality --dry-run

# Run everything including quality (default behavior)
python scripts/bulk_process.py --db-all
```

### Programmatically

```python
from app.services.quality_service import (
    apply_quality_filter_and_select,
    score_and_filter_images_for_restaurant,
    get_displayable_images,
)
from app.database import SessionLocal

db = SessionLocal()

# Full flow: score, filter, select, and mark is_displayed
selected_images = apply_quality_filter_and_select(
    db=db,
    place_id="ChIJxxxxx",
    quota={"food": 10, "interior": 10},
    max_workers=5,  # Parallel API calls
)

# Or just score and filter (without quota selection)
passing_images, all_images = score_and_filter_images_for_restaurant(
    db=db,
    place_id="ChIJxxxxx",
)

# Or query already-scored images that pass thresholds
displayable = get_displayable_images(db, place_id="ChIJxxxxx")
```

### Custom Thresholds

```python
from app.services.quality_service import apply_quality_filter_and_select

# More lenient thresholds for a specific use case
selected = apply_quality_filter_and_select(
    db=db,
    place_id="ChIJxxxxx",
    people_threshold=0.4,         # Allow more people
    image_quality_threshold=0.3,  # Allow lower quality images
)
```

### Debug Tool

Use the debug script to analyze scoring results:

```bash
# Show score distribution across all images
python scripts/debug_quality_scores.py --distribution

# Show all images for a specific restaurant
python scripts/debug_quality_scores.py --place-id ChIJxxx

# Show only failed images (to check if good ones are being rejected)
python scripts/debug_quality_scores.py --place-id ChIJxxx --show-failed

# Re-score a specific image manually
python scripts/debug_quality_scores.py --rescore-image 123
```

## Version Tracking

Quality scoring uses the same version tracking system as other AI components:

- **Current version**: `QUALITY_VERSION = "v1.1"` in `app/config/ai_versions.py`
- **Stored in database**: `quality_version` column on each image
- **Prompt history**: Stored in `prompt_versions` table

When you update the scoring prompt:
1. Bump `QUALITY_VERSION` (e.g., `"v1.1"` → `"v1.2"`)
2. Run bulk processing with `--components quality`
3. Images with old version will be re-scored

## Cost Considerations

- **API**: Uses GPT-4 Vision (`gpt-4o`)
- **Cost**: ~$0.01-0.02 per image (depends on image size)
- **Optimization**: Scores are cached in database, only re-scored when version changes
- **Parallelization**: Uses `max_workers=5` by default to balance speed vs rate limits

## Troubleshooting

### Too many images being filtered

1. Check current thresholds in `app/config/ai_versions.py`
2. Run the debug script to see score distribution:
   ```bash
   python scripts/debug_quality_scores.py --distribution
   ```
3. Lower thresholds if needed

### Nighttime images being incorrectly filtered

The v1.1 prompt explicitly instructs the AI not to penalize nighttime lighting if content is visible. If you're still seeing issues:
1. Check the `image_quality_score` for nighttime images
2. If they're scoring low, the issue may be actual visibility, not just darkness
3. Use `--rescore-image` to manually re-score and inspect

### Images not being scored

1. Check if `OPENAI_API_KEY` is set in `.env`
2. Run with `--verbose` flag to see errors:
   ```bash
   python scripts/bulk_process.py --place-ids ChIJxxx --components quality --verbose
   ```

### Rescore specific images

```python
from app.services.quality_service import score_image_quality_from_url, update_image_quality_scores
from app.database import SessionLocal

db = SessionLocal()
image_url = "https://storage.googleapis.com/..."

# Score the image
scores = score_image_quality_from_url(image_url)
print(f"People: {scores.people_score}")
print(f"Image Quality: {scores.image_quality_score}")
print(f"Time of Day: {scores.time_of_day}")
print(f"Indoor/Outdoor: {scores.indoor_outdoor}")

# Update in database
update_image_quality_scores(db, image_id=123, scores=scores)
```

## Future Enhancements

Potential additions (not yet implemented):
- Watermark/logo detection
- Aspect ratio filtering
- Duplicate content detection
- Manual override for specific images
- Time-of-day based filtering in the frontend
