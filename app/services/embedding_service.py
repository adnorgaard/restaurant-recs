import os
from typing import List, Optional
from datetime import datetime, timezone
from openai import OpenAI
from sqlalchemy.orm import Session
from app.models.database import Restaurant, RestaurantImage
from dotenv import load_dotenv
from pathlib import Path

# Load .env file from project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"

# Import version configuration
from app.config.ai_versions import EMBEDDING_VERSION


class EmbeddingServiceError(Exception):
    """Custom exception for embedding service errors"""
    pass


def generate_text_embedding(text: str) -> List[float]:
    """
    Generate an embedding for a text string using OpenAI.
    
    Args:
        text: Text to generate embedding for
        
    Returns:
        List of floats representing the embedding vector
        
    Raises:
        EmbeddingServiceError: If embedding generation fails
    """
    if not OPENAI_API_KEY:
        raise EmbeddingServiceError("OPENAI_API_KEY not configured")
    
    if not text or not text.strip():
        raise EmbeddingServiceError("Text cannot be empty")
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text.strip()
        )
        return response.data[0].embedding
    except Exception as e:
        raise EmbeddingServiceError(f"Failed to generate embedding: {str(e)}")


def get_restaurant_text_content(db: Session, restaurant_id: int) -> str:
    """
    Aggregate all text content for a restaurant (tags, descriptions) to create embedding.
    
    Args:
        db: Database session
        restaurant_id: ID of the restaurant
        
    Returns:
        Combined text content for embedding generation
    """
    restaurant = db.query(Restaurant).filter(Restaurant.id == restaurant_id).first()
    if not restaurant:
        raise EmbeddingServiceError(f"Restaurant with id {restaurant_id} not found")
    
    # Get all images with their AI tags
    images = db.query(RestaurantImage).filter(
        RestaurantImage.restaurant_id == restaurant_id
    ).all()
    
    # Collect all tags and build text content
    all_tags = []
    for image in images:
        if image.ai_tags:
            all_tags.extend(image.ai_tags)
    
    # Remove duplicates and create text
    unique_tags = list(set(all_tags))
    text_parts = [restaurant.name]
    if unique_tags:
        text_parts.append(", ".join(unique_tags))
    
    return " ".join(text_parts)


def generate_restaurant_embedding(db: Session, restaurant_id: int) -> List[float]:
    """
    Generate an embedding for a restaurant based on its name and AI tags.
    
    Args:
        db: Database session
        restaurant_id: ID of the restaurant
        
    Returns:
        List of floats representing the embedding vector
        
    Raises:
        EmbeddingServiceError: If embedding generation fails
    """
    text_content = get_restaurant_text_content(db, restaurant_id)
    return generate_text_embedding(text_content)


def update_restaurant_embedding(db: Session, restaurant_id: int) -> Restaurant:
    """
    Generate and update the embedding for a restaurant.
    
    Args:
        db: Database session
        restaurant_id: ID of the restaurant
        
    Returns:
        Updated Restaurant instance
        
    Raises:
        EmbeddingServiceError: If embedding generation or update fails
    """
    try:
        restaurant = db.query(Restaurant).filter(Restaurant.id == restaurant_id).first()
        if not restaurant:
            raise EmbeddingServiceError(f"Restaurant with id {restaurant_id} not found")
        
        # Generate embedding
        embedding = generate_restaurant_embedding(db, restaurant_id)
        
        # Update restaurant with embedding and version tracking
        restaurant.embedding = embedding
        restaurant.embedding_version = EMBEDDING_VERSION
        restaurant.embedding_updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(restaurant)
        
        print(f"[EMBEDDING] ✅ Updated embedding for restaurant {restaurant_id} (version: {EMBEDDING_VERSION})")
        
        return restaurant
    except EmbeddingServiceError:
        raise
    except Exception as e:
        db.rollback()
        raise EmbeddingServiceError(f"Failed to update restaurant embedding: {str(e)}")


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (0-1)
    """
    import numpy as np
    
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))

