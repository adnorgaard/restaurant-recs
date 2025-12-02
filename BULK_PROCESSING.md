# Bulk Processing CLI Guide

The bulk processing CLI allows you to process restaurants at scale, supporting multiple input formats, processing modes, and parallel execution with rate limiting.

## Quick Start

```bash
# Process all restaurants that need updates (smart auto mode)
python scripts/bulk_process.py --db-all

# Import new restaurants from a CSV file
python scripts/bulk_process.py --input restaurants.csv

# Preview what would be processed (no changes made)
python scripts/bulk_process.py --db-all --dry-run

# Force regenerate specific components
python scripts/bulk_process.py --db-all --force --components description

# Completely refresh a restaurant's images (delete + re-fetch from API)
python scripts/bulk_process.py --place-ids ChIJxxx --refresh-images
```

## Installation

The bulk processing script requires `tqdm` for progress bars (optional but recommended):

```bash
pip install tqdm
```

## Input Formats

### CSV File

Create a CSV file with restaurant data. Supported columns:

| Column | Required | Description |
|--------|----------|-------------|
| `place_id` | No* | Google Places place_id |
| `name` | No* | Restaurant name |
| `location` | No | City/address for name lookup |

*Either `place_id` or `name` must be provided.

Example `restaurants.csv`:
```csv
place_id,name,location
ChIJN1t_tDeuEmsRUsoyG83frY4,,,
,Employees Only,New York
,The French Laundry,Yountville CA
```

### JSON File

Create a JSON file with an array of restaurant objects:

```json
[
  {"place_id": "ChIJN1t_tDeuEmsRUsoyG83frY4"},
  {"name": "Employees Only", "location": "New York"},
  {"name": "The French Laundry", "location": "Yountville, CA"}
]
```

Or a simple array of names:
```json
[
  "Employees Only, New York",
  "The French Laundry, Yountville CA"
]
```

### Database Query

Use `--db-all` to process restaurants already in the database:

```bash
# All restaurants needing updates
python scripts/bulk_process.py --db-all --mode refresh-stale

# Specific place_ids
python scripts/bulk_process.py --place-ids ChIJ123 ChIJ456 --mode force
```

## Processing Modes

| Mode | Flag | Description |
|------|------|-------------|
| **Auto** (default) | `--mode auto` or omit | Smart mode: process missing OR outdated data |
| Force | `--force` or `--mode force` | Regenerate everything regardless of version |

### When to Use Each Mode

- **auto** (default): Handles everything intelligently. Processes new restaurants AND updates stale data automatically.
- **force**: When you want to completely rebuild analysis (use sparingly - expensive)

### Legacy Modes (still available)

| Mode | Flag | Description |
|------|------|-------------|
| Net New | `--mode net-new` | Only process completely missing data |
| Refresh Stale | `--mode refresh-stale` | Only process outdated versions |

## Image Refresh

Use `--refresh-images` to completely refresh a restaurant's images:

```bash
# Refresh images for a specific restaurant
python scripts/bulk_process.py --place-ids ChIJCf9p82aBhYARQolIoRsK6fE --refresh-images

# Refresh images for multiple restaurants
python scripts/bulk_process.py --place-ids ChIJ123 ChIJ456 --refresh-images

# Refresh all restaurants (use with caution - expensive!)
python scripts/bulk_process.py --db-all --refresh-images
```

This will:
1. **Delete** all existing images from the database
2. **Delete** all images from Google Cloud Storage
3. **Clear** the restaurant's description and embedding
4. **Re-fetch** photos from the API (SerpApi for 100+ photos, or Google for 10)
5. **Re-run** all AI analysis (categorization, tags, description, embedding)

### When to Use

- Restaurant has too few images (e.g., only 10 from Google Places)
- Images are outdated or low quality
- Switching photo providers (e.g., from Google to SerpApi)
- Need to completely rebuild a restaurant's data

### Combined with Photo Provider

```bash
# Refresh using SerpApi (100+ photos)
python scripts/bulk_process.py --place-ids ChIJxxx --refresh-images --photo-provider serpapi

# Refresh using Google Places (10 photos max)
python scripts/bulk_process.py --place-ids ChIJxxx --refresh-images --photo-provider google
```

## Component Selection

Control which AI components are processed:

```bash
# Process all components (default)
python scripts/bulk_process.py --db-all --mode refresh-stale --components all

# Only regenerate descriptions
python scripts/bulk_process.py --db-all --mode force --components description

# Regenerate descriptions and embeddings
python scripts/bulk_process.py --db-all --mode refresh-stale --components description,embedding

# Only recategorize images
python scripts/bulk_process.py --db-all --mode force --components category
```

Available components:
- `category` - Image categorization (interior/exterior/food/etc)
- `tags` - AI-generated tags per image
- `description` - AI-generated restaurant description
- `embedding` - Vector embeddings for semantic search
- `all` - All of the above

## Rate Limiting

Control API call concurrency to stay within rate limits:

```bash
# Default settings
python scripts/bulk_process.py --db-all --mode refresh-stale

# Higher concurrency for faster processing
python scripts/bulk_process.py --db-all --mode net-new \
  --concurrency 10 \
  --openai-concurrency 5 \
  --places-concurrency 8

# Lower concurrency to avoid rate limits
python scripts/bulk_process.py --db-all --mode refresh-stale \
  --concurrency 2 \
  --openai-concurrency 1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--concurrency` | 5 | Overall max concurrent restaurants |
| `--openai-concurrency` | 3 | Max concurrent OpenAI API calls |
| `--places-concurrency` | 5 | Max concurrent Google Places API calls |

### Recommended Settings

| Scenario | Concurrency | OpenAI | Places |
|----------|-------------|--------|--------|
| Default/Safe | 5 | 3 | 5 |
| Fast processing | 10 | 5 | 8 |
| Rate limit issues | 2 | 1 | 2 |
| OpenAI tier 1 | 3 | 1 | 5 |

## Output Options

### Dry Run

Preview what would be processed without making any changes:

```bash
python scripts/bulk_process.py --db-all --mode refresh-stale --dry-run
```

### Results Report

Save detailed results to a JSON file:

```bash
python scripts/bulk_process.py --db-all --mode refresh-stale --output results.json
```

### Failed Items

Export failed items for retry:

```bash
python scripts/bulk_process.py --input restaurants.csv --mode net-new \
  --output-failures failed.json

# Later, retry failed items
python scripts/bulk_process.py --input failed.json --mode net-new
```

### Verbosity

```bash
# Quiet mode (minimal output)
python scripts/bulk_process.py --db-all --mode refresh-stale --quiet

# Verbose mode (detailed progress)
python scripts/bulk_process.py --db-all --mode refresh-stale --verbose
```

## Typical Workflows

### 1. Initial Setup - Import Restaurant List

```bash
# Create your restaurant list
cat > restaurants.csv << EOF
name,location
Employees Only,New York
The French Laundry,Yountville CA
Alinea,Chicago
EOF

# Import and process (auto mode handles everything)
python scripts/bulk_process.py --input restaurants.csv
```

### 2. Iterate on AI Prompts

After changing an AI prompt (e.g., the description prompt in `vision_service.py`):

```bash
# 1. Bump the version in app/config/ai_versions.py
#    DESCRIPTION_VERSION = "v1.1"  # was "v1.0"

# 2. Preview what would be updated
python scripts/bulk_process.py --db-all --components description --dry-run

# 3. Run the refresh (auto mode detects stale versions)
python scripts/bulk_process.py --db-all --components description
```

### 3. Full Rebuild

When you need to regenerate everything (e.g., after major prompt changes):

```bash
# Force regenerate all components
python scripts/bulk_process.py --db-all --force

# Or target specific components
python scripts/bulk_process.py --db-all --force --components tags,description,embedding
```

### 4. Nightly Sync

Add new restaurants and update stale data in one command:

```bash
#!/bin/bash
# nightly_sync.sh

# Auto mode handles both new AND stale data in one pass!
python scripts/bulk_process.py --input restaurants.csv
python scripts/bulk_process.py --db-all
```

## Version Tracking

The system tracks which version of AI logic generated each piece of data:

- `description_version` / `description_updated_at` on Restaurant
- `embedding_version` / `embedding_updated_at` on Restaurant  
- `category_version` / `category_updated_at` on RestaurantImage
- `tags_version` / `tags_updated_at` on RestaurantImage

Versions are defined in `app/config/ai_versions.py`:

```python
CATEGORY_VERSION = "v1.0"
TAGS_VERSION = "v1.0"
DESCRIPTION_VERSION = "v1.0"
EMBEDDING_VERSION = "v1.0"
```

When you change a prompt, bump the corresponding version. The `refresh-stale` mode will then identify and update all data generated with the old version.

## Troubleshooting

### "No restaurants to process"

- Check that your input file exists and is formatted correctly
- For `--db-all`, ensure restaurants exist in the database
- For `--mode refresh-stale`, ensure versions are properly tracked

### Rate Limit Errors

Reduce concurrency:
```bash
python scripts/bulk_process.py --db-all --mode refresh-stale \
  --concurrency 2 --openai-concurrency 1
```

### Out of Memory

Process in smaller batches or reduce concurrency:
```bash
# Process specific place_ids in batches
python scripts/bulk_process.py --place-ids ChIJ1 ChIJ2 ChIJ3 --mode net-new
```

### API Key Errors

Ensure your `.env` file has valid API keys:
```bash
OPENAI_API_KEY=sk-...
GOOGLE_PLACES_API_KEY=AIza...
```

### Database Connection Errors

Check your database URL in `.env`:
```bash
DATABASE_URL=postgresql://user:password@localhost:5432/restaurant_recs
```

## API Reference

### Command Line Options

```
usage: bulk_process.py [-h] (--input INPUT | --db-all | --place-ids PLACE_IDS [PLACE_IDS ...])
                       [--mode {auto,force,net-new,refresh-stale}]
                       [--force]
                       [--components COMPONENTS]
                       [--concurrency CONCURRENCY]
                       [--openai-concurrency OPENAI_CONCURRENCY]
                       [--places-concurrency PLACES_CONCURRENCY]
                       [--dry-run]
                       [--output OUTPUT]
                       [--output-failures OUTPUT_FAILURES]
                       [--photo-provider {serpapi,google}]
                       [--refresh-images]
                       [--quiet]
                       [--verbose]

Bulk process restaurants for AI analysis and caching

Input Source (required, mutually exclusive):
  --input, -i           Path to CSV or JSON file with restaurant data
  --db-all              Process all restaurants in the database
  --place-ids           Specific place_ids to process

Options:
  --mode, -m            Processing mode: auto (default), force
  --force, -f           Shorthand for --mode force
  --components, -c      Components to process: category,tags,description,embedding,all
  --concurrency         Overall concurrency limit (default: 5)
  --openai-concurrency  OpenAI API concurrency limit (default: 3)
  --places-concurrency  Google Places API concurrency limit (default: 5)
  --dry-run             Preview without making changes
  --output, -o          Output JSON file for results report
  --output-failures     Output JSON file for failed items
  --photo-provider      Photo provider: serpapi (100+ photos) or google (10 photos)
  --refresh-images      Delete existing images and re-fetch from API
  --quiet, -q           Minimal output
  --verbose, -v         Verbose output with detailed progress
```

