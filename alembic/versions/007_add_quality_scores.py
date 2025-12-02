"""Add quality score columns to restaurant_images table

Revision ID: 007_add_quality_scores
Revises: 006_add_is_displayed_flag
Create Date: 2024-12-02

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '007_add_quality_scores'
down_revision = '006_add_is_displayed_flag'
branch_labels = None
depends_on = None


# Initial quality scoring prompt for seeding
QUALITY_PROMPT_V1 = """Analyze this restaurant image for quality metrics. Rate each on a scale of 0.0 to 1.0:

1. PEOPLE_SCORE: How much are people NOT the main subject?
   - 1.0 = No people visible, or people are minor background elements
   - 0.7 = People visible but not the focus (e.g., diners in background, staff partially visible)
   - 0.4 = People are prominent but sharing focus with food/interior
   - 0.0 = People are clearly the main subject (portrait, group photo, selfie)

2. LIGHTING_SCORE: How well-lit and visible is the content?
   - 1.0 = Well-lit, all details clearly visible
   - 0.7 = Adequately lit, can see what's happening (dim ambient lighting is OK if intentional)
   - 0.4 = Somewhat dark but main subject still discernible
   - 0.0 = Too dark to make out what's in the image

3. BLUR_SCORE: How sharp/in-focus is the image?
   - 1.0 = Sharp and crisp, good focus
   - 0.7 = Mostly sharp, minor softness acceptable
   - 0.4 = Noticeable blur but subject still identifiable
   - 0.0 = Very blurry, motion blur, or completely out of focus

Respond in JSON format only:
{"people_score": 0.0, "lighting_score": 0.0, "blur_score": 0.0}"""


def upgrade() -> None:
    # Add quality score columns to restaurant_images table
    op.add_column('restaurant_images', sa.Column('people_confidence_score', sa.Float(), nullable=True))
    op.add_column('restaurant_images', sa.Column('lighting_confidence_score', sa.Float(), nullable=True))
    op.add_column('restaurant_images', sa.Column('blur_confidence_score', sa.Float(), nullable=True))
    
    # Version tracking for quality scoring
    op.add_column('restaurant_images', sa.Column('quality_version', sa.String(), nullable=True))
    op.add_column('restaurant_images', sa.Column('quality_scored_at', sa.DateTime(timezone=True), nullable=True))
    
    # Add index for efficient filtering of displayable images by quality
    op.create_index('idx_quality_scores', 'restaurant_images', 
                    ['people_confidence_score', 'lighting_confidence_score', 'blur_confidence_score'])
    
    # Seed the quality prompt version
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
            'version': 'v1.0',
            'prompt_text': QUALITY_PROMPT_V1,
            'model': 'gpt-4o',
            'notes': 'Initial quality scoring prompt (people, lighting, blur)',
            'is_active': True,
        },
    ])


def downgrade() -> None:
    op.drop_index('idx_quality_scores', table_name='restaurant_images')
    op.drop_column('restaurant_images', 'quality_scored_at')
    op.drop_column('restaurant_images', 'quality_version')
    op.drop_column('restaurant_images', 'blur_confidence_score')
    op.drop_column('restaurant_images', 'lighting_confidence_score')
    op.drop_column('restaurant_images', 'people_confidence_score')
    
    # Remove the quality prompt version
    op.execute("DELETE FROM prompt_versions WHERE component = 'quality'")

