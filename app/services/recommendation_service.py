from typing import List, Optional, Dict, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_
from app.models.database import Restaurant, RestaurantImage, UserRestaurantInteraction
from app.services.embedding_service import cosine_similarity, generate_text_embedding, EmbeddingServiceError
import numpy as np


class RecommendationServiceError(Exception):
    """Custom exception for recommendation service errors"""
    pass


def get_restaurant_tags(db: Session, restaurant_id: int) -> List[str]:
    """
    Get all unique tags for a restaurant from its images.
    
    Args:
        db: Database session
        restaurant_id: ID of the restaurant
        
    Returns:
        List of unique tags
    """
    images = db.query(RestaurantImage).filter(
        RestaurantImage.restaurant_id == restaurant_id
    ).all()
    
    all_tags = []
    for image in images:
        if image.ai_tags:
            all_tags.extend(image.ai_tags)
    
    return list(set(all_tags))


def calculate_tag_similarity(tags1: List[str], tags2: List[str]) -> float:
    """
    Calculate Jaccard similarity between two sets of tags.
    
    Args:
        tags1: First set of tags
        tags2: Second set of tags
        
    Returns:
        Jaccard similarity score (0-1)
    """
    if not tags1 or not tags2:
        return 0.0
    
    set1 = set(tags1)
    set2 = set(tags2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    
    return intersection / union


def find_similar_restaurants_by_tags(
    db: Session,
    restaurant_id: int,
    limit: int = 10,
    min_similarity: float = 0.1
) -> List[Tuple[Restaurant, float]]:
    """
    Find restaurants similar to a given restaurant based on tag similarity.
    
    Args:
        db: Database session
        restaurant_id: ID of the reference restaurant
        limit: Maximum number of recommendations
        min_similarity: Minimum similarity score to include
        
    Returns:
        List of tuples (Restaurant, similarity_score) sorted by similarity
    """
    # Get tags for the reference restaurant
    reference_tags = get_restaurant_tags(db, restaurant_id)
    
    if not reference_tags:
        return []
    
    # Get all other restaurants
    all_restaurants = db.query(Restaurant).filter(
        Restaurant.id != restaurant_id
    ).all()
    
    # Calculate similarity for each restaurant
    similarities = []
    for restaurant in all_restaurants:
        restaurant_tags = get_restaurant_tags(db, restaurant.id)
        similarity = calculate_tag_similarity(reference_tags, restaurant_tags)
        
        if similarity >= min_similarity:
            similarities.append((restaurant, similarity))
    
    # Sort by similarity (descending) and return top results
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:limit]


def find_similar_restaurants_by_embedding(
    db: Session,
    restaurant_id: int,
    limit: int = 10,
    min_similarity: float = 0.5
) -> List[Tuple[Restaurant, float]]:
    """
    Find restaurants similar to a given restaurant based on embedding similarity.
    
    Args:
        db: Database session
        restaurant_id: ID of the reference restaurant
        limit: Maximum number of recommendations
        min_similarity: Minimum similarity score to include
        
    Returns:
        List of tuples (Restaurant, similarity_score) sorted by similarity
        
    Raises:
        RecommendationServiceError: If reference restaurant has no embedding
    """
    # Get reference restaurant and its embedding
    reference_restaurant = db.query(Restaurant).filter(
        Restaurant.id == restaurant_id
    ).first()
    
    if not reference_restaurant:
        raise RecommendationServiceError(f"Restaurant with id {restaurant_id} not found")
    
    if not reference_restaurant.embedding:
        raise RecommendationServiceError(f"Restaurant with id {restaurant_id} has no embedding")
    
    reference_embedding = reference_restaurant.embedding
    
    # Get all other restaurants with embeddings
    all_restaurants = db.query(Restaurant).filter(
        and_(
            Restaurant.id != restaurant_id,
            Restaurant.embedding.isnot(None)
        )
    ).all()
    
    # Calculate similarity for each restaurant
    similarities = []
    for restaurant in all_restaurants:
        if restaurant.embedding:
            similarity = cosine_similarity(reference_embedding, restaurant.embedding)
            
            if similarity >= min_similarity:
                similarities.append((restaurant, similarity))
    
    # Sort by similarity (descending) and return top results
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:limit]


def find_similar_restaurants_hybrid(
    db: Session,
    restaurant_id: int,
    limit: int = 10,
    embedding_weight: float = 0.7,
    tag_weight: float = 0.3
) -> List[Tuple[Restaurant, float]]:
    """
    Find similar restaurants using a hybrid approach combining embeddings and tags.
    
    Args:
        db: Database session
        restaurant_id: ID of the reference restaurant
        limit: Maximum number of recommendations
        embedding_weight: Weight for embedding similarity (0-1)
        tag_weight: Weight for tag similarity (0-1)
        
    Returns:
        List of tuples (Restaurant, combined_similarity_score) sorted by similarity
    """
    # Normalize weights
    total_weight = embedding_weight + tag_weight
    if total_weight > 0:
        embedding_weight = embedding_weight / total_weight
        tag_weight = tag_weight / total_weight
    
    # Get similarity scores from both methods
    embedding_similarities = {}
    tag_similarities = {}
    
    # Try embedding-based similarity
    try:
        embedding_results = find_similar_restaurants_by_embedding(
            db, restaurant_id, limit=limit * 2, min_similarity=0.0
        )
        embedding_similarities = {r.id: score for r, score in embedding_results}
    except RecommendationServiceError:
        # If no embedding, skip embedding-based similarity
        pass
    
    # Get tag-based similarity
    tag_results = find_similar_restaurants_by_tags(
        db, restaurant_id, limit=limit * 2, min_similarity=0.0
    )
    tag_similarities = {r.id: score for r, score in tag_results}
    
    # Combine scores
    all_restaurant_ids = set(embedding_similarities.keys()) | set(tag_similarities.keys())
    combined_scores = {}
    
    for rid in all_restaurant_ids:
        embedding_score = embedding_similarities.get(rid, 0.0)
        tag_score = tag_similarities.get(rid, 0.0)
        
        combined_score = (embedding_score * embedding_weight) + (tag_score * tag_weight)
        combined_scores[rid] = combined_score
    
    # Get restaurant objects and sort by combined score
    similarities = []
    for rid, score in combined_scores.items():
        restaurant = db.query(Restaurant).filter(Restaurant.id == rid).first()
        if restaurant:
            similarities.append((restaurant, score))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:limit]


def search_restaurants_by_text(
    db: Session,
    query_text: str,
    limit: int = 10,
    min_similarity: float = 0.5
) -> List[Tuple[Restaurant, float]]:
    """
    Search restaurants by natural language query using semantic search.
    
    Args:
        db: Database session
        query_text: Natural language search query
        limit: Maximum number of results
        min_similarity: Minimum similarity score to include
        
    Returns:
        List of tuples (Restaurant, similarity_score) sorted by similarity
    """
    try:
        # Generate embedding for the query
        query_embedding = generate_text_embedding(query_text)
        
        # Get all restaurants with embeddings
        all_restaurants = db.query(Restaurant).filter(
            Restaurant.embedding.isnot(None)
        ).all()
        
        # Calculate similarity for each restaurant
        similarities = []
        for restaurant in all_restaurants:
            if restaurant.embedding:
                similarity = cosine_similarity(query_embedding, restaurant.embedding)
                
                if similarity >= min_similarity:
                    similarities.append((restaurant, similarity))
        
        # Sort by similarity (descending) and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]
    except EmbeddingServiceError as e:
        raise RecommendationServiceError(f"Failed to search restaurants: {str(e)}")

