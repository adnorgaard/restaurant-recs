from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session
from datetime import timedelta
from app.models.schemas import (
    ClassifyRequest, ClassifyResponse,
    UserCreate, UserResponse, Token,
    InteractionCreate, InteractionResponse,
    RecommendationResponse, RestaurantRecommendation,
    TextSearchRequest, TextSearchResponse
)
from app.database import get_db
from app.models.database import User, UserRestaurantInteraction
from app.services.places_service import (
    get_place_id_by_name,
    get_place_images,
    get_place_images_with_urls,
    get_place_images_with_metadata,
    get_restaurant_name,
    extract_place_id_from_url,
    find_place_by_text,
    PlacesServiceError,
    GOOGLE_PLACES_API_KEY,
    GOOGLE_PLACES_PHOTOS_BASE_URL
)
from app.services.vision_service import (
    analyze_restaurant_images,
    categorize_images,
    select_diverse_images,
    VisionServiceError
)
from app.services.database_service import (
    update_image_tags,
    get_image_by_id,
    DatabaseServiceError
)
from app.services.auth_service import (
    create_user, authenticate_user, get_user_by_email,
    create_access_token, decode_access_token,
    AuthServiceError, ACCESS_TOKEN_EXPIRE_MINUTES
)
from app.services.embedding_service import (
    update_restaurant_embedding, EmbeddingServiceError
)
from app.services.recommendation_service import (
    find_similar_restaurants_by_tags,
    find_similar_restaurants_by_embedding,
    find_similar_restaurants_hybrid,
    search_restaurants_by_text,
    RecommendationServiceError
)

app = FastAPI(
    title="Restaurant Recommendation Engine",
    description="AI-powered restaurant classification and recommendation system",
    version="2.0.0"
)

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Dependency to get current user
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Get the current authenticated user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
    
    email: str = payload.get("sub")
    if email is None:
        raise credentials_exception
    
    user = get_user_by_email(db, email=email)
    if user is None:
        raise credentials_exception
    
    return user


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/api/places/photo/{photo_path:path}")
async def proxy_place_photo(photo_path: str, maxWidthPx: int = 800):
    """
    Proxy endpoint for Google Places API photos.
    This is needed because the Places API (New) requires authentication headers
    which browsers cannot send directly.
    """
    import requests
    
    # Reconstruct the full photo path
    # photo_path will be like "places/ChIJ.../photos/Aap_uEA..."
    url = f"{GOOGLE_PLACES_PHOTOS_BASE_URL}/{photo_path}/media"
    headers = {
        "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY
    }
    params = {
        "maxWidthPx": maxWidthPx
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        # Return the image with appropriate content type
        return Response(
            content=response.content,
            media_type=response.headers.get("Content-Type", "image/jpeg")
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=404, detail=f"Failed to fetch photo: {str(e)}")


class FindPlaceIdRequest(BaseModel):
    query: str  # Restaurant name, address, or Google Maps URL


class FindPlaceIdResponse(BaseModel):
    place_id: str
    restaurant_name: str
    message: str


@app.post("/find-place-id", response_model=FindPlaceIdResponse)
async def find_place_id(request: FindPlaceIdRequest):
    """
    Helper endpoint to find the place_id for a restaurant.
    Accepts restaurant name, address, or Google Maps URL.
    """
    try:
        # Try to get place_id (handles URLs and names)
        place_id = extract_place_id_from_url(request.query)
        
        # Get restaurant name to confirm
        restaurant_name = get_restaurant_name(place_id)
        
        return FindPlaceIdResponse(
            place_id=place_id,
            restaurant_name=restaurant_name,
            message=f"Found place_id for: {restaurant_name}"
        )
    except PlacesServiceError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/classify", response_model=ClassifyResponse)
async def classify_restaurant(request: ClassifyRequest, db: Session = Depends(get_db)):
    """
    Classify a restaurant based on images from Google Places API.
    Uses cached images when available to save API credits.
    
    Accepts either:
    - place_id: Direct Google Places place_id
    - name + location: Restaurant name and optional location for search
    """
    try:
        # Get place_id
        if request.place_id:
            place_id = request.place_id
        elif request.name:
            if not request.location:
                raise HTTPException(
                    status_code=400,
                    detail="location is required when using name-based search"
                )
            place_id = get_place_id_by_name(request.name, request.location)
        else:
            raise HTTPException(
                status_code=400,
                detail="Either place_id or name must be provided"
            )
        
        # Get restaurant name
        restaurant_name = get_restaurant_name(place_id)
        
        # Fetch images with metadata (up to 10) - this will use cache if available
        all_images, _, photo_names = get_place_images_with_metadata(place_id, max_images=10, db=db)
        
        # Categorize images and select diverse set
        # Pass db and place_id to store categories
        categorized_images = categorize_images(all_images, db=db, place_id=place_id, photo_names=photo_names)
        selected_images, selected_indices = select_diverse_images(categorized_images, max_images=5)
        
        # Get photo_names for selected images
        selected_photo_names = [photo_names[i] for i in selected_indices if i < len(photo_names)]
        
        # Analyze selected images with AI
        # Pass db and place_id to store AI tags
        analysis = analyze_restaurant_images(
            selected_images,
            db=db,
            place_id=place_id,
            photo_names=selected_photo_names
        )
        
        # Generate embedding for the restaurant (async, don't fail if it errors)
        try:
            from app.services.database_service import get_or_create_restaurant
            restaurant = get_or_create_restaurant(db, place_id, restaurant_name)
            if restaurant:
                try:
                    update_restaurant_embedding(db, restaurant.id)
                except Exception as e:
                    # Log but don't fail the request
                    print(f"Warning: Failed to generate embedding for restaurant {restaurant.id}: {str(e)}")
        except Exception as e:
            # Log but don't fail the request
            print(f"Warning: Failed to generate embedding: {str(e)}")
        
        return ClassifyResponse(
            restaurant_name=restaurant_name,
            tags=analysis["tags"],
            description=analysis["description"]
        )
        
    except PlacesServiceError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except VisionServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


class TestRequest(BaseModel):
    url: str


class ImageMetadata(BaseModel):
    url: str
    category: str

class TestResponse(BaseModel):
    restaurant_name: str
    images: List[ImageMetadata]
    tags: List[str]
    description: str


@app.post("/test", response_model=TestResponse)
async def test_restaurant(request: TestRequest, db: Session = Depends(get_db)):
    """
    Test endpoint that accepts a restaurant URL and returns images and classification.
    Uses cached images when available to save API credits.
    """
    try:
        # Extract place_id from URL
        place_id = extract_place_id_from_url(request.url)
        
        # Get restaurant name
        restaurant_name = get_restaurant_name(place_id)
        
        # Fetch images with URLs and metadata (up to 10) - this will use cache if available
        all_images, all_image_urls, photo_names = get_place_images_with_metadata(place_id, max_images=10, db=db)
        
        # Categorize images and select diverse set
        # Pass db and place_id to store categories
        categorized_images = categorize_images(all_images, db=db, place_id=place_id, photo_names=photo_names)
        selected_images, selected_indices = select_diverse_images(categorized_images, max_images=5)
        
        # Build image metadata with URLs and categories
        # categorized_images is a list of (image_bytes, category) tuples
        image_metadata = []
        for idx in selected_indices:
            if idx < len(categorized_images) and idx < len(all_image_urls):
                _, category = categorized_images[idx]
                image_metadata.append(ImageMetadata(
                    url=all_image_urls[idx],
                    category=category
                ))
        
        # Get photo_names for selected images
        selected_photo_names = [photo_names[i] for i in selected_indices if i < len(photo_names)]
        
        # Analyze selected images with AI
        # Pass db and place_id to store AI tags
        analysis = analyze_restaurant_images(
            selected_images,
            db=db,
            place_id=place_id,
            photo_names=selected_photo_names
        )
        
        # Generate embedding for the restaurant (async, don't fail if it errors)
        try:
            from app.services.database_service import get_or_create_restaurant
            restaurant = get_or_create_restaurant(db, place_id, restaurant_name)
            if restaurant:
                try:
                    update_restaurant_embedding(db, restaurant.id)
                except Exception as e:
                    # Log but don't fail the request
                    print(f"Warning: Failed to generate embedding for restaurant {restaurant.id}: {str(e)}")
        except Exception as e:
            # Log but don't fail the request
            print(f"Warning: Failed to generate embedding: {str(e)}")
        
        return TestResponse(
            restaurant_name=restaurant_name,
            images=image_metadata,
            tags=analysis["tags"],
            description=analysis["description"]
        )
        
    except PlacesServiceError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except VisionServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


class UpdateImageTagsRequest(BaseModel):
    category: Optional[str] = None
    ai_tags: Optional[List[str]] = None


class UpdateImageTagsResponse(BaseModel):
    id: int
    restaurant_id: int
    photo_name: str
    gcs_url: str
    category: Optional[str]
    ai_tags: Optional[List[str]]
    message: str


@app.put("/api/images/{image_id}/tags", response_model=UpdateImageTagsResponse)
async def update_image_tags_endpoint(
    image_id: int,
    request: UpdateImageTagsRequest,
    db: Session = Depends(get_db)
):
    """
    Update tags for a specific image.
    
    Both category and ai_tags are optional - only provided fields will be updated.
    
    Args:
        image_id: ID of the image to update
        request: UpdateImageTagsRequest with optional category and/or ai_tags
        db: Database session
        
    Returns:
        Updated image information
    """
    try:
        # Validate category if provided
        if request.category is not None:
            valid_categories = ["interior", "exterior", "food", "menu", "bar", "other"]
            if request.category not in valid_categories:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid category. Must be one of: {', '.join(valid_categories)}"
                )
        
        # Update the image tags
        updated_image = update_image_tags(
            db,
            image_id,
            category=request.category,
            ai_tags=request.ai_tags
        )
        
        return UpdateImageTagsResponse(
            id=updated_image.id,
            restaurant_id=updated_image.restaurant_id,
            photo_name=updated_image.photo_name,
            gcs_url=updated_image.gcs_url,
            category=updated_image.category,
            ai_tags=updated_image.ai_tags,
            message="Image tags updated successfully"
        )
        
    except DatabaseServiceError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ============================================================================
# Authentication Endpoints
# ============================================================================

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user.
    
    Args:
        user_data: User registration data (email and password)
        db: Database session
        
    Returns:
        Created user information
    """
    try:
        user = create_user(db, email=user_data.email, password=user_data.password)
        return UserResponse(
            id=user.id,
            email=user.email,
            is_active=user.is_active,
            created_at=user.created_at.isoformat()
        )
    except AuthServiceError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {str(e)}")


@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Login endpoint that returns a JWT access token.
    
    Args:
        form_data: OAuth2 form data (username=email, password)
        db: Database session
        
    Returns:
        JWT access token
    """
    user = authenticate_user(db, email=form_data.username, password=form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        is_active=current_user.is_active,
        created_at=current_user.created_at.isoformat()
    )


# ============================================================================
# User-Restaurant Interaction Endpoints
# ============================================================================

@app.post("/api/interactions", response_model=InteractionResponse, status_code=status.HTTP_201_CREATED)
async def create_interaction(
    interaction: InteractionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a user-restaurant interaction (like, dislike, or rating).
    
    Args:
        interaction: Interaction data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Created interaction information
    """
    try:
        # Validate interaction type
        valid_types = ["like", "dislike", "rating"]
        if interaction.interaction_type not in valid_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"interaction_type must be one of: {', '.join(valid_types)}"
            )
        
        # Validate rating if interaction_type is rating
        if interaction.interaction_type == "rating":
            if interaction.rating is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="rating is required when interaction_type is 'rating'"
                )
            if not (1.0 <= interaction.rating <= 5.0):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="rating must be between 1.0 and 5.0"
                )
        
        # Check if interaction already exists
        existing = db.query(UserRestaurantInteraction).filter(
            UserRestaurantInteraction.user_id == current_user.id,
            UserRestaurantInteraction.restaurant_id == interaction.restaurant_id
        ).first()
        
        if existing:
            # Update existing interaction
            existing.interaction_type = interaction.interaction_type
            existing.rating = interaction.rating
            db.commit()
            db.refresh(existing)
            return InteractionResponse(
                id=existing.id,
                user_id=existing.user_id,
                restaurant_id=existing.restaurant_id,
                interaction_type=existing.interaction_type,
                rating=existing.rating,
                created_at=existing.created_at.isoformat()
            )
        else:
            # Create new interaction
            new_interaction = UserRestaurantInteraction(
                user_id=current_user.id,
                restaurant_id=interaction.restaurant_id,
                interaction_type=interaction.interaction_type,
                rating=interaction.rating
            )
            db.add(new_interaction)
            db.commit()
            db.refresh(new_interaction)
            return InteractionResponse(
                id=new_interaction.id,
                user_id=new_interaction.user_id,
                restaurant_id=new_interaction.restaurant_id,
                interaction_type=new_interaction.interaction_type,
                rating=new_interaction.rating,
                created_at=new_interaction.created_at.isoformat()
            )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {str(e)}")


# ============================================================================
# Recommendation Endpoints
# ============================================================================

@app.get("/api/restaurants/{restaurant_id}/similar", response_model=RecommendationResponse)
async def get_similar_restaurants(
    restaurant_id: int,
    method: str = "hybrid",
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get restaurants similar to a given restaurant.
    
    Args:
        restaurant_id: ID of the reference restaurant
        method: Recommendation method - 'tags', 'embedding', or 'hybrid' (default)
        limit: Maximum number of recommendations
        db: Database session
        
    Returns:
        List of similar restaurants with similarity scores
    """
    try:
        if method == "tags":
            results = find_similar_restaurants_by_tags(db, restaurant_id, limit=limit)
        elif method == "embedding":
            results = find_similar_restaurants_by_embedding(db, restaurant_id, limit=limit)
        elif method == "hybrid":
            results = find_similar_restaurants_hybrid(db, restaurant_id, limit=limit)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="method must be one of: 'tags', 'embedding', 'hybrid'"
            )
        
        recommendations = [
            RestaurantRecommendation(
                id=restaurant.id,
                place_id=restaurant.place_id,
                name=restaurant.name,
                similarity_score=score
            )
            for restaurant, score in results
        ]
        
        return RecommendationResponse(
            recommendations=recommendations,
            method=method,
            reference_restaurant_id=restaurant_id
        )
    except RecommendationServiceError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {str(e)}")


@app.post("/api/search", response_model=TextSearchResponse)
async def search_restaurants_by_text(
    search_request: TextSearchRequest,
    db: Session = Depends(get_db)
):
    """
    Search restaurants using natural language text query (semantic search).
    
    Args:
        search_request: Search query and parameters
        db: Database session
        
    Returns:
        List of restaurants matching the query with similarity scores
    """
    try:
        results = search_restaurants_by_text(
            db,
            search_request.query,
            limit=search_request.limit,
            min_similarity=search_request.min_similarity
        )
        
        recommendations = [
            RestaurantRecommendation(
                id=restaurant.id,
                place_id=restaurant.place_id,
                name=restaurant.name,
                similarity_score=score
            )
            for restaurant, score in results
        ]
        
        return TextSearchResponse(
            results=recommendations,
            query=search_request.query
        )
    except RecommendationServiceError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {str(e)}")


@app.post("/api/restaurants/{restaurant_id}/generate-embedding", response_model=dict)
async def generate_restaurant_embedding_endpoint(
    restaurant_id: int,
    db: Session = Depends(get_db)
):
    """
    Generate and update embedding for a restaurant.
    This is useful for populating embeddings for existing restaurants.
    
    Args:
        restaurant_id: ID of the restaurant
        db: Database session
        
    Returns:
        Success message
    """
    try:
        restaurant = update_restaurant_embedding(db, restaurant_id)
        return {
            "message": "Embedding generated successfully",
            "restaurant_id": restaurant.id,
            "restaurant_name": restaurant.name
        }
    except EmbeddingServiceError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {str(e)}")


# ============================================================================
# Web Interface
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def test_interface():
    """Simple HTML interface for testing"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Restaurant Image Classifier - Test</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                margin-bottom: 10px;
            }
            .subtitle {
                color: #666;
                margin-bottom: 30px;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: bold;
                color: #555;
            }
            input[type="text"] {
                width: 100%;
                padding: 12px;
                border: 2px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
                box-sizing: border-box;
            }
            input[type="text"]:focus {
                outline: none;
                border-color: #4CAF50;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
            }
            button:hover {
                background-color: #45a049;
            }
            button:disabled {
                background-color: #cccccc;
                cursor: not-allowed;
            }
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
                color: #666;
            }
            .loading.show {
                display: block;
            }
            .results {
                margin-top: 30px;
                display: none;
            }
            .results.show {
                display: block;
            }
            .restaurant-name {
                font-size: 24px;
                font-weight: bold;
                color: #333;
                margin-bottom: 20px;
            }
            .images-section {
                margin-bottom: 30px;
            }
            .images-section h3 {
                color: #555;
                margin-bottom: 15px;
            }
            .images-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            .image-container {
                border: 2px solid #ddd;
                border-radius: 4px;
                overflow: hidden;
                background: #f9f9f9;
            }
            .image-container img {
                width: 100%;
                height: 200px;
                object-fit: cover;
                display: block;
            }
            .image-metadata {
                padding: 8px 12px;
                background: #fff;
                border-top: 1px solid #ddd;
            }
            .category-label {
                display: inline-block;
                background-color: #2196F3;
                color: white;
                padding: 4px 10px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 600;
                text-transform: capitalize;
            }
            .classification-section {
                background: #f9f9f9;
                padding: 20px;
                border-radius: 4px;
            }
            .classification-section h3 {
                color: #555;
                margin-top: 0;
            }
            .tags {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-bottom: 15px;
            }
            .tag {
                background-color: #4CAF50;
                color: white;
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 14px;
            }
            .description {
                color: #333;
                line-height: 1.6;
                font-size: 15px;
            }
            .error {
                display: none;
                background-color: #ffebee;
                color: #c62828;
                padding: 15px;
                border-radius: 4px;
                margin-top: 20px;
            }
            .error.show {
                display: block;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🍽️ Restaurant Image Classifier</h1>
            <p class="subtitle">Enter a Google Maps URL, share link, or restaurant name to see images and AI classification</p>
            
            <form id="testForm">
                <div class="form-group">
                    <label for="restaurantUrl">Restaurant URL, Share Link, or Name:</label>
                    <input 
                        type="text" 
                        id="restaurantUrl" 
                        name="url" 
                        value="Employees Only, New York"
                        placeholder="https://maps.app.goo.gl/... or &quot;Joe's Pizza, New York&quot;" 
                        required
                    />
                    <small style="color: #666; display: block; margin-top: 5px;">
                        You can enter: Google Maps URL, share link, or restaurant name with location
                    </small>
                </div>
                <button type="submit">Classify Restaurant</button>
            </form>
            
            <div class="loading" id="loading">
                <p>Fetching images and analyzing restaurant... This may take a moment.</p>
            </div>
            
            <div class="error" id="error"></div>
            
            <div class="results" id="results">
                <div class="restaurant-name" id="restaurantName"></div>
                
                <div class="images-section">
                    <h3>📸 Fetched Images</h3>
                    <div class="images-grid" id="imagesGrid"></div>
                </div>
                
                <div class="classification-section">
                    <h3>🏷️ AI Classification</h3>
                    <div class="tags" id="tags"></div>
                    <div class="description" id="description"></div>
                </div>
            </div>
        </div>
        
        <script>
            document.getElementById('testForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const url = document.getElementById('restaurantUrl').value;
                const loading = document.getElementById('loading');
                const results = document.getElementById('results');
                const error = document.getElementById('error');
                
                // Reset UI
                loading.classList.add('show');
                results.classList.remove('show');
                error.classList.remove('show');
                document.querySelector('button').disabled = true;
                
                try {
                    const response = await fetch('/test', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ url: url })
                    });
                    
                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || 'Failed to classify restaurant');
                    }
                    
                    const data = await response.json();
                    
                    // Display restaurant name
                    document.getElementById('restaurantName').textContent = data.restaurant_name;
                    
                    // Display images with metadata
                    const imagesGrid = document.getElementById('imagesGrid');
                    imagesGrid.innerHTML = '';
                    data.images.forEach((imageData, index) => {
                        const container = document.createElement('div');
                        container.className = 'image-container';
                        
                        const img = document.createElement('img');
                        img.src = imageData.url;
                        img.alt = `Restaurant image ${index + 1}`;
                        container.appendChild(img);
                        
                        const metadata = document.createElement('div');
                        metadata.className = 'image-metadata';
                        const categoryLabel = document.createElement('span');
                        categoryLabel.className = 'category-label';
                        categoryLabel.textContent = imageData.category;
                        metadata.appendChild(categoryLabel);
                        container.appendChild(metadata);
                        
                        imagesGrid.appendChild(container);
                    });
                    
                    // Display tags
                    const tagsContainer = document.getElementById('tags');
                    tagsContainer.innerHTML = '';
                    data.tags.forEach(tag => {
                        const tagElement = document.createElement('span');
                        tagElement.className = 'tag';
                        tagElement.textContent = tag;
                        tagsContainer.appendChild(tagElement);
                    });
                    
                    // Display description
                    document.getElementById('description').textContent = data.description;
                    
                    results.classList.add('show');
                    
                } catch (err) {
                    error.textContent = 'Error: ' + err.message;
                    error.classList.add('show');
                } finally {
                    loading.classList.remove('show');
                    document.querySelector('button').disabled = false;
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

