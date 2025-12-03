# 🍽️ Restaurant Recommendation Engine

An AI-powered restaurant discovery platform that analyzes restaurant photos to understand vibes, aesthetics, and cuisine—then uses that understanding to power smart recommendations and semantic search.

## What It Does

1. **Sees** — Fetches restaurant photos from Google Places API
2. **Understands** — GPT-4 Vision analyzes images to extract tags (vibe, cuisine, atmosphere, price tier)
3. **Remembers** — Caches everything (images in GCS, analysis in PostgreSQL) for instant repeat queries
4. **Recommends** — Finds similar restaurants using AI-generated tags and vector embeddings
5. **Searches** — Natural language queries like "romantic Italian spot with outdoor seating"

## Features

### 🔍 Image Analysis
- Automatic image categorization (interior, exterior, food, drink, bar, menu)
- AI-generated descriptive tags for each restaurant
- Natural language descriptions of restaurant vibes

### ⚡ Smart Caching
- Images cached in Google Cloud Storage
- Analysis results cached in PostgreSQL
- First request builds cache, subsequent requests are instant

### 🎯 Recommendations
- **Tag-based**: Jaccard similarity on AI-generated tags
- **Embedding-based**: Cosine similarity on vector embeddings
- **Hybrid**: Weighted combination of both approaches

### 🔎 Semantic Search
- Natural language queries powered by OpenAI embeddings
- "Find me a cozy coffee shop with good pastries"
- "Upscale steakhouse with a bar scene"

### 👤 User System
- JWT-based authentication
- Like/dislike/rate restaurants
- Personalized interaction history

## Tech Stack

- **API**: FastAPI (Python 3.8+)
- **Database**: PostgreSQL + Alembic migrations
- **AI**: OpenAI GPT-4 Vision + text-embedding-3-small
- **Storage**: Google Cloud Storage
- **External APIs**: Google Places API (New)

## Prerequisites

- Python 3.8+
- PostgreSQL database
- Google Cloud Storage bucket + service account
- Google Places API key
- OpenAI API key

## Quick Start

### 1. Clone and Install

```bash
git clone <repo-url>
cd restaurant-recs
pip install -r requirements.txt
```

### 2. Set Up Environment

Create a `.env` file:

```bash
# Required
GOOGLE_PLACES_API_KEY=your_google_places_api_key
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=postgresql://user:password@localhost:5432/restaurant_recs

# Google Cloud Storage
GCS_BUCKET_NAME=your-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# Optional (for auth)
JWT_SECRET_KEY=your-secret-key
```

### 3. Set Up Database

```bash
alembic upgrade head
```

### 4. Set Up GCS

See `GCS_SETUP.md` for detailed instructions, or run:

```bash
python scripts/validate_gcs_setup.py
```

### 5. Run

```bash
uvicorn app.main:app --reload
```

Open `http://localhost:8000` for the web interface.

## API Overview

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/classify` | POST | Analyze a restaurant from place_id or name |
| `/test` | POST | Analyze from Google Maps URL (web interface) |
| `/api/search` | POST | Semantic search with natural language |
| `/api/restaurants/{id}/similar` | GET | Get similar restaurants |

### Authentication

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/register` | POST | Create new user |
| `/token` | POST | Login, get JWT |
| `/me` | GET | Get current user (requires auth) |

### User Interactions

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/interactions` | POST | Like/dislike/rate a restaurant (requires auth) |

### Utilities

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/find-place-id` | POST | Get place_id from restaurant name or URL |
| `/health` | GET | Health check with database status |
| `/api/restaurants/{id}/generate-embedding` | POST | Generate embedding for a restaurant |

## Usage Examples

### Classify a Restaurant

```bash
# By place_id
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"place_id": "ChIJN1t_tDeuEmsRUsoyG83frY4"}'

# By name + location
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"name": "The French Laundry", "location": "Yountville, CA"}'
```

### Semantic Search

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "romantic Italian restaurant with candles", "limit": 5}'
```

### Get Similar Restaurants

```bash
# Hybrid (default) - combines tags and embeddings
curl "http://localhost:8000/api/restaurants/1/similar?method=hybrid&limit=10"

# Tags only
curl "http://localhost:8000/api/restaurants/1/similar?method=tags"

# Embeddings only
curl "http://localhost:8000/api/restaurants/1/similar?method=embedding"
```

### User Registration and Interaction

```bash
# Register
curl -X POST "http://localhost:8000/register" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123"}'

# Login
curl -X POST "http://localhost:8000/token" \
  -d "username=user@example.com&password=password123"

# Like a restaurant (with token)
curl -X POST "http://localhost:8000/api/interactions" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"restaurant_id": 1, "interaction_type": "like"}'
```

## Project Structure

```
restaurant-recs/
├── app/
│   ├── main.py                    # FastAPI app, all endpoints
│   ├── database.py                # Database connection
│   ├── config/
│   │   └── ai_versions.py         # AI prompt versions and quality thresholds
│   ├── models/
│   │   ├── database.py            # SQLAlchemy models
│   │   └── schemas.py             # Pydantic request/response schemas
│   └── services/
│       ├── places_service.py      # Google Places API integration
│       ├── vision_service.py      # GPT-4 Vision analysis
│       ├── storage_service.py     # Google Cloud Storage
│       ├── database_service.py    # Database operations + caching
│       ├── embedding_service.py   # Vector embeddings
│       ├── recommendation_service.py  # Similarity algorithms
│       ├── quality_service.py     # Image quality scoring (GPT-4 Vision)
│       ├── auth_service.py        # JWT authentication
│       └── debug_service.py       # Cache inspection tools
├── alembic/                       # Database migrations
├── scripts/                       # Setup and utility scripts
├── templates/                     # HTML templates
│   └── index.html                 # Web interface template
├── requirements.txt
└── .env                           # Your environment variables
```

## Database Schema

### restaurants
- `id`, `place_id`, `name`, `description`, `embedding`, `created_at`, `updated_at`

### restaurant_images  
- `id`, `restaurant_id`, `photo_name`, `gcs_url`, `gcs_bucket_path`, `category`, `ai_tags`, `created_at`, `updated_at`
- Quality scores: `people_confidence_score`, `image_quality_score`, `quality_version`, `quality_scored_at`
- Metadata tags: `time_of_day`, `indoor_outdoor`
- Display flag: `is_displayed`

### users
- `id`, `email`, `hashed_password`, `is_active`, `is_superuser`, `created_at`, `updated_at`

### user_restaurant_interactions
- `id`, `user_id`, `restaurant_id`, `interaction_type`, `rating`, `created_at`, `updated_at`

## Caching Architecture

The system uses a multi-tier caching strategy:

1. **Images**: Stored in GCS after first fetch (never re-downloaded from Google Places)
2. **Categories**: Stored per-image in PostgreSQL (AI categorization runs once)
3. **Tags & Description**: Stored per-restaurant (AI analysis runs once)
4. **Embeddings**: Stored per-restaurant (generated once from tags)

Result: First request may take 5-10s, subsequent requests < 100ms.

See `CACHING_ARCHITECTURE.md` for detailed documentation.

## Debug Endpoints

| Endpoint | Description |
|----------|-------------|
| `/debug/cache/{place_id}` | Inspect cache status for a restaurant |
| `/debug/trace` | Trace request flow and identify bottlenecks |
| `/debug/reset-cache/{place_id}` | Clear cache for re-analysis |
| `/debug/force-recategorize/{place_id}` | Re-run AI categorization |
| `/debug/backfill-images/{place_id}` | Create DB records for GCS images |

## Documentation

- **API Docs**: `http://localhost:8000/docs` (Swagger UI)
- **Alternative**: `http://localhost:8000/redoc` (ReDoc)
- **GCS Setup**: `GCS_SETUP.md`
- **Caching Details**: `CACHING_ARCHITECTURE.md`
- **Debug Guide**: `DEBUG_GUIDE.md`
- **Quality Scoring**: `QUALITY_SCORING.md`
- **Bulk Processing**: `BULK_PROCESSING.md`

## License

MIT
