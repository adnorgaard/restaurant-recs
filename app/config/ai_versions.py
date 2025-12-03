"""
AI Version Configuration

Bump these version strings when you change the corresponding AI prompts or logic.
This allows tracking which version of logic generated each piece of data,
and enables bulk refresh of stale data.

Usage:
    from app.config.ai_versions import CATEGORY_VERSION, DESCRIPTION_VERSION
    
    # When storing AI-generated data, include the version:
    store_restaurant_description(db, place_id, description, version=DESCRIPTION_VERSION)
"""

from typing import Optional
from sqlalchemy.orm import Session

# =============================================================================
# Current Active Versions
# Bump these when you modify the corresponding AI prompts/logic
# =============================================================================

CATEGORY_VERSION = "v1.0"      # Image categorization prompt (interior/exterior/food/etc)
TAGS_VERSION = "v1.0"          # AI tags generation prompt
DESCRIPTION_VERSION = "v1.0"   # Restaurant description prompt
EMBEDDING_VERSION = "v1.0"     # Embedding generation logic
QUALITY_VERSION = "v1.1"       # Image quality scoring prompt (people, image_quality, time_of_day, indoor_outdoor)

# =============================================================================
# Quality Scoring Thresholds
# Images must score ABOVE these thresholds to be displayed (higher = better)
# 
# HOW TO ADJUST:
#   - Increase threshold → Stricter filtering, fewer images pass
#   - Decrease threshold → More lenient, more images pass
#
# SCORE MEANINGS:
#   - People: 1.0 = no people, 0.0 = people are main subject
#   - Image Quality: 1.0 = clear/sharp/visible, 0.0 = unusable
#
# See QUALITY_SCORING.md for full documentation.
# =============================================================================

QUALITY_PEOPLE_THRESHOLD = 0.6      # Reject if people are clearly the main subject
IMAGE_QUALITY_THRESHOLD = 0.5       # Reject if image is not clear enough to display

# DEPRECATED thresholds (kept for backwards compatibility, no longer used)
# These were replaced by IMAGE_QUALITY_THRESHOLD in v1.1
QUALITY_LIGHTING_THRESHOLD = 0.5    # DEPRECATED - use IMAGE_QUALITY_THRESHOLD
QUALITY_BLUR_THRESHOLD = 0.5        # DEPRECATED - use IMAGE_QUALITY_THRESHOLD

# Valid component names for version tracking
VALID_COMPONENTS = {"category", "tags", "description", "embedding", "quality"}


def get_active_version(db: Session, component: str) -> str:
    """
    Get the active version for a component.
    
    First checks the prompt_versions table for an active version,
    then falls back to the constants defined above.
    
    Args:
        db: Database session
        component: One of "category", "tags", "description", "embedding", "quality"
        
    Returns:
        Version string, e.g., "v1.0"
    """
    if component not in VALID_COMPONENTS:
        raise ValueError(f"Invalid component: {component}. Must be one of {VALID_COMPONENTS}")
    
    # Try to get from database first
    try:
        from app.models.database import PromptVersion
        
        prompt_version = db.query(PromptVersion).filter(
            PromptVersion.component == component,
            PromptVersion.is_active == True
        ).first()
        
        if prompt_version:
            return prompt_version.version
    except Exception:
        # If database query fails, fall back to constants
        pass
    
    # Fall back to constants
    version_map = {
        "category": CATEGORY_VERSION,
        "tags": TAGS_VERSION,
        "description": DESCRIPTION_VERSION,
        "embedding": EMBEDDING_VERSION,
        "quality": QUALITY_VERSION,
    }
    
    return version_map[component]


def get_all_active_versions(db: Session) -> dict:
    """
    Get all active versions as a dictionary.
    
    Args:
        db: Database session
        
    Returns:
        Dict mapping component name to version string
    """
    return {
        component: get_active_version(db, component)
        for component in VALID_COMPONENTS
    }


def is_version_current(db: Session, component: str, version: Optional[str]) -> bool:
    """
    Check if a stored version matches the current active version.
    
    Args:
        db: Database session
        component: One of "category", "tags", "description", "embedding", "quality"
        version: The version to check (can be None for unversioned data)
        
    Returns:
        True if version matches current active version, False otherwise
    """
    if version is None:
        return False
    
    current = get_active_version(db, component)
    return version == current

