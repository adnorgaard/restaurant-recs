# Image Quality Scoring System

This document describes the AI-powered image quality scoring system that filters out poor-quality restaurant images before displaying them on the website.

## Overview

The quality scoring system uses GPT-4 Vision to analyze each image and assign confidence scores for three quality metrics:

| Metric | What It Measures | Score Direction |
|--------|------------------|-----------------|
| **People Score** | Are people the main subject? | Higher = less people-focused |
| **Lighting Score** | Is the image well-lit? | Higher = better lighting |
| **Blur Score** | Is the image sharp/in-focus? | Higher = sharper |

All scores range from **0.0 to 1.0**, where **higher is always better** (more displayable).

## Score Interpretation

### People Confidence Score

| Score | Interpretation |
|-------|----------------|
| 1.0 | No people visible, or people are minor background elements |
| 0.7 | People visible but not the focus (diners in background, staff partially visible) |
| 0.4 | People are prominent but sharing focus with food/interior |
| 0.0 | People are clearly the main subject (portrait, group photo, selfie) |

### Lighting Confidence Score

| Score | Interpretation |
|-------|----------------|
| 1.0 | Well-lit, all details clearly visible |
| 0.7 | Adequately lit, can see what's happening (dim ambient lighting is OK if intentional) |
| 0.4 | Somewhat dark but main subject still discernible |
| 0.0 | Too dark to make out what's in the image |

### Blur Confidence Score

| Score | Interpretation |
|-------|----------------|
| 1.0 | Sharp and crisp, good focus |
| 0.7 | Mostly sharp, minor softness acceptable |
| 0.4 | Noticeable blur but subject still identifiable |
| 0.0 | Very blurry, motion blur, or completely out of focus |

## Configuration

Thresholds are configured in `app/config/ai_versions.py`:

```python
# Quality Scoring Thresholds
# Images must score ABOVE these thresholds to be displayed (higher = better)

QUALITY_PEOPLE_THRESHOLD = 0.6      # Display if people_confidence_score > 0.6
QUALITY_LIGHTING_THRESHOLD = 0.5    # Display if lighting_confidence_score > 0.5
QUALITY_BLUR_THRESHOLD = 0.5        # Display if blur_confidence_score > 0.5
```

### Adjusting Thresholds

- **Increase thresholds** → Stricter filtering, fewer images displayed
- **Decrease thresholds** → More lenient, more images displayed

Example: If too many food photos are being filtered due to ambient restaurant lighting:
```python
QUALITY_LIGHTING_THRESHOLD = 0.4  # More lenient on lighting
```

## Database Schema

Quality scores are stored on the `restaurant_images` table:

| Column | Type | Description |
|--------|------|-------------|
| `people_confidence_score` | Float | 0.0-1.0, higher = less people-focused |
| `lighting_confidence_score` | Float | 0.0-1.0, higher = better lit |
| `blur_confidence_score` | Float | 0.0-1.0, higher = sharper |
| `quality_version` | String | Version of the scoring prompt (e.g., "v1.0") |
| `quality_scored_at` | DateTime | When the image was scored |
| `is_displayed` | Boolean | Whether the image passes quality and is selected for display |

## Processing Flow

The quality scoring follows **Option B** (recommended):

```
1. Cache ALL images from API (SerpAPI/Google)
     ↓
2. Categorize images (food, interior, exterior, drink, bar, etc.)
     ↓
3. Score ALL cached images for quality (GPT-4 Vision)
   - people_score: Are people the main subject?
   - lighting_score: Is it well-lit?
   - blur_score: Is it sharp?
     ↓
4. Filter to images that pass ALL thresholds
   (only quality images proceed)
     ↓
5. Apply quota selection to QUALITY-PASSING images by category
   Default quota: 8 food, 8 interior, 2 exterior, 2 drink = 20 total
   
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
    quota={"food": 8, "interior": 8, "exterior": 2, "drink": 2},
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
    people_threshold=0.4,    # Allow more people
    lighting_threshold=0.3,  # Allow darker images
    blur_threshold=0.4,      # Allow some blur
)
```

## Version Tracking

Quality scoring uses the same version tracking system as other AI components:

- **Current version**: `QUALITY_VERSION = "v1.0"` in `app/config/ai_versions.py`
- **Stored in database**: `quality_version` column on each image
- **Prompt history**: Stored in `prompt_versions` table

When you update the scoring prompt:
1. Bump `QUALITY_VERSION` (e.g., `"v1.0"` → `"v1.1"`)
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
2. Query images to see score distribution:
   ```sql
   SELECT 
     people_confidence_score,
     lighting_confidence_score,
     blur_confidence_score
   FROM restaurant_images
   WHERE restaurant_id = (SELECT id FROM restaurants WHERE place_id = 'ChIJxxx')
   ORDER BY quality_scored_at DESC;
   ```
3. Lower thresholds if needed

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
print(f"People: {scores.people_score}, Lighting: {scores.lighting_score}, Blur: {scores.blur_score}")

# Update in database
update_image_quality_scores(db, image_id=123, scores=scores)
```

## Future Enhancements

Potential additions (not yet implemented):
- Watermark/logo detection
- Aspect ratio filtering
- Duplicate content detection
- Manual override for specific images

