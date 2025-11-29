import os
import base64
from typing import List, Dict, Tuple, Optional
from sqlalchemy.orm import Session
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# Load .env file from project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class VisionServiceError(Exception):
    """Custom exception for Vision API errors"""
    pass


def categorize_image(image_bytes: bytes) -> str:
    """
    Quickly categorize a single image into one of: interior, exterior, food, menu, bar, other.
    
    Args:
        image_bytes: Image bytes to categorize
        
    Returns:
        Category string
    """
    if not OPENAI_API_KEY:
        raise VisionServiceError("OPENAI_API_KEY not configured")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    prompt = """Categorize this restaurant image into ONE of these categories:
- "interior": Inside the restaurant, dining area, seating, ambiance
- "exterior": Outside facade, entrance, building exterior, storefront
- "food": Food dishes, plates, individual menu items, close-up of food
- "menu": Menu boards, printed menus, menu displays
- "bar": Bar area, drinks, bartending, bar seating
- "other": Anything else that doesn't fit the above

Respond with ONLY the category name, nothing else."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=10,
            temperature=0.1
        )
        
        category = response.choices[0].message.content.strip().lower()
        
        # Validate category
        valid_categories = ["interior", "exterior", "food", "menu", "bar", "other"]
        if category not in valid_categories:
            return "other"
        
        return category
    except Exception as e:
        # Default to other if categorization fails
        return "other"


def categorize_images(images: List[bytes], db: Optional[Session] = None, place_id: Optional[str] = None, photo_names: Optional[List[str]] = None) -> List[Tuple[bytes, str]]:
    """
    Categorize a list of images. Uses cached categories when available.
    Only calls AI for images that have actual bytes AND don't have cached categories.
    
    Args:
        images: List of image bytes (can be empty bytes b"" for cached images)
        db: Optional database session to store categories
        place_id: Optional place_id for storing categories in database
        photo_names: Optional list of photo_names corresponding to images (must match length)
        
    Returns:
        List of tuples (image_bytes, category)
    """
    # Check for cached categories first
    cached_categories = {}
    if db and place_id and photo_names and len(photo_names) == len(images):
        try:
            from app.services.database_service import get_cached_categories_and_tags
            cached_categories, _ = get_cached_categories_and_tags(db, place_id, photo_names)
            print(f"[DEBUG] Found {len(cached_categories)} cached categories")
        except Exception as e:
            print(f"Warning: Failed to get cached categories: {str(e)}")
    
    categorized = []
    images_to_categorize = []
    indices_to_categorize = []
    
    # First pass: use cached categories where available
    for idx, img_bytes in enumerate(images):
        photo_name = photo_names[idx] if photo_names and idx < len(photo_names) else None
        
        # Check if we have a cached category
        if photo_name and photo_name in cached_categories and cached_categories[photo_name]:
            # Use cached category
            category = cached_categories[photo_name]
            categorized.append((img_bytes, category))
            print(f"[DEBUG] Using cached category for image {idx}: {category}")
        else:
            # Need to categorize this image
            # Only add to categorization queue if we have actual image bytes
            if img_bytes and len(img_bytes) > 0:
                categorized.append((img_bytes, "pending"))  # Placeholder
                images_to_categorize.append((idx, img_bytes))
                indices_to_categorize.append(idx)
            else:
                # No bytes and no cached category - default to "other"
                categorized.append((img_bytes, "other"))
                print(f"[WARNING] Image {idx} has no bytes and no cached category, defaulting to 'other'")
    
    # Second pass: categorize images that don't have cached categories
    if images_to_categorize:
        print(f"[DEBUG] Categorizing {len(images_to_categorize)} images with AI...")
        for idx, img_bytes in images_to_categorize:
            category = categorize_image(img_bytes)
            categorized[idx] = (img_bytes, category)
            print(f"[DEBUG] AI categorized image {idx} as: {category}")
    
    # Store new categories in database if db and place_id are provided
    if db and place_id and photo_names and len(photo_names) == len(images) and indices_to_categorize:
        try:
            from app.services.database_service import get_cached_images, update_image_tags
            cached_images = get_cached_images(db, place_id)
            print(f"[DEBUG] Storing {len(indices_to_categorize)} new categories to database")
            
            # Create a mapping of photo_name to cached image
            photo_to_image = {img.photo_name: img for img in cached_images}
            
            # Update categories for newly categorized images
            updated_count = 0
            for idx in indices_to_categorize:
                if idx < len(photo_names) and idx < len(categorized):
                    photo_name = photo_names[idx]
                    _, category = categorized[idx]
                    if photo_name in photo_to_image:
                        update_image_tags(db, photo_to_image[photo_name].id, category=category)
                        updated_count += 1
            print(f"[DEBUG] Successfully stored {updated_count} categories to database")
        except Exception as e:
            import traceback
            print(f"[ERROR] Failed to store categories in database: {str(e)}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
    
    return categorized


def select_diverse_images(categorized_images: List[Tuple[bytes, str]], max_images: int = 5, randomize: bool = True) -> Tuple[List[bytes], List[int]]:
    """
    Select diverse images from different categories, prioritizing category diversity.
    Ensures no duplicate indices are returned.
    
    Since Google Places API doesn't specify photo ordering (could be random, 
    popularity-based, or recency-based - we don't know), we randomize selection
    within each category to avoid always picking the same images.
    
    Args:
        categorized_images: List of (image_bytes, category) tuples
        max_images: Maximum number of images to select
        randomize: If True, randomly select from each category instead of always first
        
    Returns:
        Tuple of (selected image bytes, selected indices) - indices are unique
    """
    import random
    
    # Group images by category with their indices
    by_category = {}
    for idx, (img_bytes, category) in enumerate(categorized_images):
        # Normalize category - treat None, empty, or "pending" as "other"
        if not category or category == "pending":
            category = "other"
        if category not in by_category:
            by_category[category] = []
        by_category[category].append((idx, img_bytes))
    
    print(f"[DEBUG] select_diverse_images: categories found: {list(by_category.keys())}, counts: {[(k, len(v)) for k, v in by_category.items()]}")
    
    # Shuffle images within each category if randomize is True
    if randomize:
        for category in by_category:
            random.shuffle(by_category[category])
    
    # Priority order for categories (most informative first)
    category_priority = ["interior", "food", "exterior", "bar", "menu", "other"]
    
    selected = []
    selected_indices = set()  # Use set to prevent duplicates
    selected_indices_ordered = []  # Keep order
    
    # First pass: try to get at least one from each priority category
    for category in category_priority:
        if category in by_category and len(selected_indices) < max_images:
            for idx, img_bytes in by_category[category]:
                if idx not in selected_indices:
                    selected.append(img_bytes)
                    selected_indices.add(idx)
                    selected_indices_ordered.append(idx)
                    break
    
    # Second pass: fill remaining slots with more images from priority categories
    for category in category_priority:
        if len(selected_indices) >= max_images:
            break
        for idx, img_bytes in by_category.get(category, []):
            if idx not in selected_indices and len(selected_indices) < max_images:
                selected.append(img_bytes)
                selected_indices.add(idx)
                selected_indices_ordered.append(idx)
    
    # Fallback: if we still don't have enough, take any remaining
    if len(selected_indices) < max_images:
        for idx, (img_bytes, _) in enumerate(categorized_images):
            if idx not in selected_indices and len(selected_indices) < max_images:
                selected.append(img_bytes)
                selected_indices.add(idx)
                selected_indices_ordered.append(idx)
    
    if not selected:
        # Ultimate fallback: just take first N images
        selected = [img for img, _ in categorized_images[:max_images]]
        selected_indices_ordered = list(range(min(len(categorized_images), max_images)))
    
    print(f"[DEBUG] select_diverse_images: selected {len(selected_indices_ordered)} unique indices: {selected_indices_ordered}")
    
    return selected, selected_indices_ordered


def analyze_restaurant_images(images: List[bytes], db: Optional[Session] = None, place_id: Optional[str] = None, photo_names: Optional[List[str]] = None) -> Dict[str, any]:
    """
    Analyze restaurant images using OpenAI GPT-4 Vision to generate tags and description.
    
    Args:
        images: List of image bytes to analyze
        db: Optional database session to store AI tags
        place_id: Optional place_id for storing AI tags in database
        photo_names: Optional list of photo_names corresponding to images (must match length)
        
    Returns:
        Dictionary with 'tags' (list) and 'description' (string)
        
    Raises:
        VisionServiceError: If analysis fails
    """
    if not OPENAI_API_KEY:
        raise VisionServiceError("OPENAI_API_KEY not configured")
    
    if not images:
        raise VisionServiceError("No images provided for analysis")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Convert images to base64
    image_contents = []
    for img_bytes in images:
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
        image_contents.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    
    # Create prompt for restaurant classification
    prompt = """Analyze these restaurant images and provide:
1. A list of relevant tags/categories (e.g., "cozy", "upscale", "family-friendly", "italian", "romantic", "casual", "outdoor seating", "modern", "traditional", etc.)
2. A natural language description of the restaurant's ambiance, style, and characteristics

Focus on:
- Ambiance and atmosphere
- Cuisine type/style
- Price level indicators
- Decor and interior design
- Target audience
- Special features (outdoor seating, bar, etc.)

Respond in the following JSON format:
{
    "tags": ["tag1", "tag2", "tag3"],
    "description": "A detailed description of the restaurant..."
}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *image_contents
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        # Extract the response content
        content = response.choices[0].message.content
        
        # Try to parse JSON from the response
        import json
        try:
            # The response might be wrapped in markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            
            # Validate structure
            if "tags" not in result or "description" not in result:
                raise ValueError("Missing required fields in response")
            
            # Ensure tags is a list
            if not isinstance(result["tags"], list):
                result["tags"] = [result["tags"]]
            
            # Store AI tags and description in database if db and place_id are provided
            if db and place_id and photo_names and len(photo_names) == len(images):
                try:
                    from app.services.database_service import store_ai_tags_for_images, store_restaurant_description
                    # Store the same tags for all analyzed images
                    image_ai_tags = [(photo_name, result["tags"]) for photo_name in photo_names]
                    store_ai_tags_for_images(db, place_id, image_ai_tags)
                    # Store the AI-generated description on the restaurant
                    store_restaurant_description(db, place_id, result["description"])
                    print(f"[DEBUG] ✅ Stored AI tags and description for {len(photo_names)} images")
                except Exception as e:
                    # Log error but don't fail analysis
                    import traceback
                    print(f"[ERROR] Failed to store AI analysis in database: {str(e)}")
                    print(f"[ERROR] Traceback: {traceback.format_exc()}")
            
            return {
                "tags": result["tags"],
                "description": result["description"]
            }
        except json.JSONDecodeError:
            # Fallback: try to extract tags and description from text
            # This is a simple fallback - in production you might want more robust parsing
            lines = content.split("\n")
            tags = []
            description = ""
            
            for line in lines:
                if "tag" in line.lower() or any(char in line for char in ["-", "*"]):
                    # Try to extract tags
                    import re
                    tag_matches = re.findall(r'"([^"]+)"', line)
                    tags.extend(tag_matches)
                elif "description" in line.lower():
                    # Extract description
                    description = line.split(":", 1)[-1].strip()
            
            if not tags and not description:
                # Last resort: use the entire content as description
                description = content
                tags = ["restaurant"]  # Default tag
            
            result = {
                "tags": tags if tags else ["restaurant"],
                "description": description if description else content
            }
            
            # Store AI tags and description in database if db and place_id are provided
            if db and place_id and photo_names and len(photo_names) == len(images):
                try:
                    from app.services.database_service import store_ai_tags_for_images, store_restaurant_description
                    # Store the same tags for all analyzed images
                    image_ai_tags = [(photo_name, result["tags"]) for photo_name in photo_names]
                    store_ai_tags_for_images(db, place_id, image_ai_tags)
                    # Store the AI-generated description on the restaurant
                    store_restaurant_description(db, place_id, result["description"])
                    print(f"[DEBUG] ✅ Stored AI tags and description (fallback) for {len(photo_names)} images")
                except Exception as e:
                    # Log error but don't fail analysis
                    import traceback
                    print(f"[ERROR] Failed to store AI analysis in database: {str(e)}")
                    print(f"[ERROR] Traceback: {traceback.format_exc()}")
            
            return result
            
    except Exception as e:
        raise VisionServiceError(f"Failed to analyze images: {str(e)}")

