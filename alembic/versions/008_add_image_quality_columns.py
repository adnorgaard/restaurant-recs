"""Add image_quality_score, time_of_day, indoor_outdoor columns

Revision ID: 008_add_image_quality_columns
Revises: 007_add_quality_scores
Create Date: 2024-12-03

This migration adds new unified image quality scoring columns and metadata tags.

DEPRECATION NOTE:
- lighting_confidence_score is now DEPRECATED (replaced by image_quality_score)
- blur_confidence_score is now DEPRECATED (replaced by image_quality_score)
These columns are kept for historical data but are no longer written to.
The new image_quality_score combines lighting and blur into a single holistic quality metric.

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '008_add_image_quality_columns'
down_revision = '007_add_quality_scores'
branch_labels = None
depends_on = None


# New quality scoring prompt v1.1
QUALITY_PROMPT_V1_1 = """Analyze this restaurant image. Provide scores from 0.0 to 1.0:

1. PEOPLE_SCORE: How much are people NOT the main subject?
   - 1.0 = No people visible, or people are minor background elements
   - 0.0 = People are clearly the main subject (portrait, group photo, selfie)
   
   Score continuously between 0.0 and 1.0.

2. IMAGE_QUALITY_SCORE: Is this image clear enough to display on a restaurant website?
   
   Consider: Can a viewer clearly see and appreciate the content?
   - Is the subject visible and identifiable?
   - Is it reasonably sharp/in focus?
   - Can you see important details (textures, colors, shapes)?
   
   - 1.0 = Excellent - sharp, clear, all details visible
   - 0.0 = Unusable - cannot make out what's in the image
   
   Score continuously between 0.0 and 1.0.
   
   Do NOT penalize:
   - Ambient, moody, or nighttime lighting (if content is still visible)
   - Minor grain or noise (if content is still clear)

3. TIME_OF_DAY: When does this appear to be taken?
   - "day" = Daytime (natural daylight visible)
   - "night" = Nighttime (dark outside or evening ambiance, etc.)
   - "unknown" = Cannot determine

4. INDOOR_OUTDOOR: Where was this taken?
   - "indoor" = Inside a building
   - "outdoor" = Outside
   - "unknown" = Cannot determine

Respond in JSON format only:
{"people_score": 0.0, "image_quality_score": 0.0, "time_of_day": "unknown", "indoor_outdoor": "unknown"}"""


def upgrade() -> None:
    # Add new unified image quality score column
    op.add_column('restaurant_images', sa.Column('image_quality_score', sa.Float(), nullable=True))
    
    # Add metadata tag columns
    op.add_column('restaurant_images', sa.Column('time_of_day', sa.String(), nullable=True))
    op.add_column('restaurant_images', sa.Column('indoor_outdoor', sa.String(), nullable=True))
    
    # Add index for efficient filtering by new quality score
    op.create_index('idx_image_quality_score', 'restaurant_images', ['image_quality_score'])
    
    # Add index for metadata filtering
    op.create_index('idx_time_of_day', 'restaurant_images', ['time_of_day'])
    op.create_index('idx_indoor_outdoor', 'restaurant_images', ['indoor_outdoor'])
    
    # Deactivate old v1.0 quality prompt and add new v1.1
    op.execute("UPDATE prompt_versions SET is_active = false WHERE component = 'quality' AND version = 'v1.0'")
    
    prompt_versions = sa.table(
        'prompt_versions',
        sa.column('component', sa.String),
        sa.column('version', sa.String),
        sa.column('prompt_text', sa.Text),
        sa.column('model', sa.String),
        sa.column('notes', sa.Text),
        sa.column('is_active', sa.Boolean),
    )
    
    op.bulk_insert(prompt_versions, [
        {
            'component': 'quality',
            'version': 'v1.1',
            'prompt_text': QUALITY_PROMPT_V1_1,
            'model': 'gpt-4o',
            'notes': 'Unified image_quality_score (replaces lighting+blur), added time_of_day and indoor_outdoor tags. Prompt reframed around visibility rather than brightness.',
            'is_active': True,
        },
    ])


def downgrade() -> None:
    # Remove indexes
    op.drop_index('idx_indoor_outdoor', table_name='restaurant_images')
    op.drop_index('idx_time_of_day', table_name='restaurant_images')
    op.drop_index('idx_image_quality_score', table_name='restaurant_images')
    
    # Remove columns
    op.drop_column('restaurant_images', 'indoor_outdoor')
    op.drop_column('restaurant_images', 'time_of_day')
    op.drop_column('restaurant_images', 'image_quality_score')
    
    # Remove v1.1 prompt and reactivate v1.0
    op.execute("DELETE FROM prompt_versions WHERE component = 'quality' AND version = 'v1.1'")
    op.execute("UPDATE prompt_versions SET is_active = true WHERE component = 'quality' AND version = 'v1.0'")

