# Caching Architecture - How It Should Work

## Optimal Flow

### First Request (No Cache)
1. User searches "Employees Only, New York"
2. `extract_place_id_from_url` → API call to get place_id
3. `get_restaurant_name` → API call to get name
4. `get_place_images_with_metadata` → API calls to get images
5. Download images from Places API
6. `categorize_images` → OpenAI API calls to categorize
7. `analyze_restaurant_images` → OpenAI API calls for AI analysis
8. Store everything in database
9. Return response

**Time: ~10-30 seconds (multiple API calls)**

### Second Request (Full Cache) - OPTIMIZED ✅
1. User searches "Employees Only, New York" again
2. `extract_place_id_from_url` → **Database search first** (semantic search using embeddings)
3. If found in database: `get_complete_cached_restaurant_data` → Database query
4. Return instantly with cached data (no API calls, no image downloads)
5. If not found: Go through full flow again

**Time: < 1 second (just database queries, no API calls, no downloads)**

## Issues Fixed ✅

1. ✅ **Place ID lookup now cached** - `extract_place_id_from_url` checks database first using semantic search
2. ✅ **No unnecessary GCS downloads** - Images are only downloaded when needed for AI analysis
3. ✅ **AI tags properly cached** - AI analysis is checked before running, tags stored in database
4. ✅ **Smart partial cache** - Only fetches/downloads what's missing
5. ✅ **Database-first search** - Natural language search checks database before Places API

## Implementation Details

### Database-First Search
- `find_restaurant_by_text()`: Searches database using semantic embeddings before hitting Places API
- `extract_place_id_from_url()`: Now accepts `db` parameter and checks database first for text queries
- Uses restaurant name matching and semantic similarity (min 0.75) to find cached restaurants

### Optimized Image Fetching
- When cached images exist with categories: Uses GCS URLs directly, no downloads
- Only downloads images when needed for AI analysis (if tags not cached)
- `get_place_images_with_metadata()`: Returns empty bytes when cached, avoiding unnecessary downloads

### AI Tagging Caching
- `get_cached_restaurant_analysis()`: Checks if all selected images have AI tags before running analysis
- AI tags stored per image in `RestaurantImage.ai_tags` JSON field
- Analysis only runs when tags are missing, not on every request

