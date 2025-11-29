# Restaurant Image Classifier Bot

A Python REST API that classifies and describes restaurants by analyzing images from Google Places API using OpenAI GPT-4 Vision.

## Features

- Fetches restaurant images from Google Places API
- **Caches images in Google Cloud Storage and PostgreSQL** to save API credits
- Analyzes images using OpenAI GPT-4 Vision
- Returns structured tags and natural language descriptions
- Supports both place_id and name-based restaurant lookup
- Web interface for easy testing with URL input
- Displays fetched images alongside classification results
- **Update image tags** (category and AI-generated tags) via API

## Prerequisites

- Python 3.8+
- Google Places API key
- OpenAI API key
- PostgreSQL database
- Google Cloud Storage bucket
- Google Cloud service account credentials (for GCS access)

## Setup

1. Clone the repository and navigate to the project directory:
```bash
cd restaurant-recs
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up PostgreSQL database:
   - Create a PostgreSQL database for the application
   - Note the connection string (e.g., `postgresql://user:password@localhost:5432/restaurant_recs`)

4. Set up Google Cloud Storage:
   - **See `GCS_SETUP.md` for detailed step-by-step instructions**
   - Quick summary:
     - Create a GCS bucket in Google Cloud Console
     - Create a service account with Storage Admin permissions
     - Download the service account JSON key file
   - After setup, validate your configuration:
     ```bash
     python scripts/validate_gcs_setup.py
     ```

5. Create a `.env` file in the root directory:
```bash
GOOGLE_PLACES_API_KEY=your_google_places_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_URL=postgresql://user:password@localhost:5432/restaurant_recs
GCS_BUCKET_NAME=your-gcs-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

6. Run database migrations:
```bash
alembic upgrade head
```

## Running the API

Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## Testing Interface

A simple web interface is available for testing at `http://localhost:8000`

Simply:
1. Open `http://localhost:8000` in your browser
2. Paste a Google Maps restaurant URL
3. Click "Classify Restaurant"
4. View the fetched images and AI-generated tags/description

The interface displays:
- All images fetched from Google Places
- AI-generated tags (e.g., "cozy", "upscale", "italian")
- Natural language description of the restaurant

## API Documentation

Once the server is running, you can access:
- Interactive API docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`
- Test interface: `http://localhost:8000`

## API Endpoints

### POST /classify

Classify a restaurant based on images from Google Places API.

**Request Body:**
```json
{
  "place_id": "ChIJN1t_tDeuEmsRUsoyG83frY4"
}
```

Or using name and location:
```json
{
  "name": "The French Laundry",
  "location": "Yountville, CA"
}
```

**Response:**
```json
{
  "restaurant_name": "The French Laundry",
  "tags": ["upscale", "fine-dining", "french", "romantic", "award-winning"],
  "description": "An elegant fine-dining restaurant with a sophisticated atmosphere, known for its exceptional French cuisine and impeccable service."
}
```

### POST /test

Test endpoint that accepts a restaurant URL and returns images and classification.

**Request Body:**
```json
{
  "url": "https://www.google.com/maps/place/Restaurant+Name"
}
```

**Response:**
```json
{
  "restaurant_name": "Restaurant Name",
  "image_urls": ["https://maps.googleapis.com/...", ...],
  "tags": ["cozy", "upscale", "italian"],
  "description": "A detailed description..."
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

### PUT /api/images/{image_id}/tags

Update tags for a specific cached image.

**Request Body:**
```json
{
  "category": "interior",
  "ai_tags": ["cozy", "romantic", "upscale"]
}
```

Both `category` and `ai_tags` are optional - only provided fields will be updated.

**Valid categories:** `interior`, `exterior`, `food`, `menu`, `bar`, `other`

**Response:**
```json
{
  "id": 1,
  "restaurant_id": 1,
  "photo_name": "places/ChIJ.../photos/Aap_uEA...",
  "gcs_url": "https://storage.googleapis.com/...",
  "category": "interior",
  "ai_tags": ["cozy", "romantic", "upscale"],
  "message": "Image tags updated successfully"
}
```

## Project Structure

```
restaurant-recs/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── database.py          # Database connection and session management
│   ├── services/
│   │   ├── __init__.py
│   │   ├── places_service.py    # Google Places API integration
│   │   ├── vision_service.py    # OpenAI Vision API integration
│   │   ├── storage_service.py   # Google Cloud Storage integration
│   │   └── database_service.py  # Database operations
│   └── models/
│       ├── __init__.py
│       ├── schemas.py           # Pydantic models for request/response
│       └── database.py          # SQLAlchemy database models
├── alembic/                   # Database migrations
│   ├── versions/
│   └── env.py
├── alembic.ini                # Alembic configuration
├── requirements.txt
├── .env                        # Your environment variables (not in git)
└── README.md
```

## Error Handling

The API handles various error scenarios:
- Missing or invalid place_id
- Restaurant not found
- No images available
- API rate limits
- Invalid API keys

All errors return appropriate HTTP status codes with descriptive error messages.

## Example Usage

Using curl:
```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"place_id": "ChIJN1t_tDeuEmsRUsoyG83frY4"}'
```

Using Python:
```python
import requests

response = requests.post(
    "http://localhost:8000/classify",
    json={"place_id": "ChIJN1t_tDeuEmsRUsoyG83frY4"}
)
print(response.json())
```

## Notes

- The API fetches up to 5 images per restaurant by default
- Images are analyzed together to provide comprehensive classification
- Tags and descriptions are generated based on ambiance, cuisine type, price level, and other characteristics
- **Images are cached after first fetch** - subsequent requests for the same restaurant will use cached images from Google Cloud Storage, saving Google Places API credits
- Categories and AI tags are automatically stored in the database after analysis
- You can manually update tags using the `/api/images/{image_id}/tags` endpoint

## Database Schema

The application uses two main tables:

- **restaurants**: Stores restaurant information (place_id, name)
- **restaurant_images**: Stores image metadata (GCS URL, category, AI tags) linked to restaurants

Run `alembic upgrade head` to create the database schema.

