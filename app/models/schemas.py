from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List


class ClassifyRequest(BaseModel):
    place_id: Optional[str] = Field(None, description="Google Places place_id")
    name: Optional[str] = Field(None, description="Restaurant name")
    location: Optional[str] = Field(None, description="Location/address for name-based search")

    class Config:
        json_schema_extra = {
            "example": {
                "place_id": "ChIJN1t_tDeuEmsRUsoyG83frY4"
            }
        }


class ClassifyResponse(BaseModel):
    restaurant_name: str
    tags: List[str]
    description: str

    class Config:
        json_schema_extra = {
            "example": {
                "restaurant_name": "Example Restaurant",
                "tags": ["cozy", "upscale", "italian", "romantic"],
                "description": "An elegant Italian restaurant with a warm, intimate atmosphere perfect for date nights."
            }
        }


# User and Authentication Schemas
class UserCreate(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    email: str
    is_active: bool
    created_at: str

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    email: Optional[str] = None


# User-Restaurant Interaction Schemas
class InteractionCreate(BaseModel):
    restaurant_id: int
    interaction_type: str = Field(..., description="Type: 'like', 'dislike', or 'rating'")
    rating: Optional[float] = Field(None, description="Rating value (1.0-5.0) if interaction_type is 'rating'")


class InteractionResponse(BaseModel):
    id: int
    user_id: int
    restaurant_id: int
    interaction_type: str
    rating: Optional[float]
    created_at: str

    class Config:
        from_attributes = True


# Recommendation Schemas
class RestaurantRecommendation(BaseModel):
    id: int
    place_id: str
    name: str
    similarity_score: float

    class Config:
        from_attributes = True


class RecommendationResponse(BaseModel):
    recommendations: List[RestaurantRecommendation]
    method: str
    reference_restaurant_id: Optional[int] = None


class TextSearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    limit: int = Field(10, ge=1, le=50, description="Maximum number of results")
    min_similarity: float = Field(0.5, ge=0.0, le=1.0, description="Minimum similarity score")


class TextSearchResponse(BaseModel):
    results: List[RestaurantRecommendation]
    query: str




