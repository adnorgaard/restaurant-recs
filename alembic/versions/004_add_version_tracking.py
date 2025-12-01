"""Add version tracking for AI-generated content

Revision ID: 004_add_version_tracking
Revises: 003_add_description
Create Date: 2024-01-04 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import func

# revision identifiers, used by Alembic.
revision: str = '004_add_version_tracking'
down_revision: Union[str, None] = '003_add_description'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Initial prompts for seeding (extracted from vision_service.py)
CATEGORY_PROMPT_V1 = """Categorize this restaurant image into ONE of these categories:
- "interior": Inside the restaurant, dining area, seating, ambiance
- "exterior": Outside facade, entrance, building exterior, storefront
- "food": Food dishes, plates, individual menu items, close-up of food
- "drink": Beverages, cocktails, wine glasses, coffee, drinks (product shots)
- "bar": Bar area, bar counter, bartending station, bar seating (the space, not drinks)
- "menu": Menu boards, printed menus, menu displays
- "other": Anything else that doesn't fit the above

Respond with ONLY the category name, nothing else."""

TAGS_DESCRIPTION_PROMPT_V1 = """Analyze these restaurant images and provide:

1. TAGS: Generate 4-10 simple, single-word or short-phrase tags that describe this restaurant.
   Focus on: cuisine type, atmosphere, and one or two standout features.
   Keep tags concise and easy to scan.

2. DESCRIPTION: Write a 2-3 sentence summary capturing the restaurant's essence, vibe, and what makes it special. Be evocative but natural.

Respond in JSON format:
{
    "tags": ["tag1", "tag2", ...],
    "description": "2-3 sentence summary..."
}"""


def upgrade() -> None:
    # Create prompt_versions table to track AI prompt history
    op.create_table(
        'prompt_versions',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('component', sa.String(), nullable=False, index=True),  # category, tags, description, embedding
        sa.Column('version', sa.String(), nullable=False),  # e.g., "v1.0"
        sa.Column('prompt_text', sa.Text(), nullable=True),  # The full prompt used
        sa.Column('model', sa.String(), nullable=True),  # e.g., "gpt-4o"
        sa.Column('notes', sa.Text(), nullable=True),  # Optional changelog notes
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),  # Whether this is the current version
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=func.now()),
        sa.UniqueConstraint('component', 'version', name='uq_component_version')
    )
    
    # Add version tracking columns to restaurants table
    op.add_column('restaurants', sa.Column('description_version', sa.String(), nullable=True))
    op.add_column('restaurants', sa.Column('description_updated_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('restaurants', sa.Column('embedding_version', sa.String(), nullable=True))
    op.add_column('restaurants', sa.Column('embedding_updated_at', sa.DateTime(timezone=True), nullable=True))
    
    # Add version tracking columns to restaurant_images table
    op.add_column('restaurant_images', sa.Column('category_version', sa.String(), nullable=True))
    op.add_column('restaurant_images', sa.Column('category_updated_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('restaurant_images', sa.Column('tags_version', sa.String(), nullable=True))
    op.add_column('restaurant_images', sa.Column('tags_updated_at', sa.DateTime(timezone=True), nullable=True))
    
    # Seed initial prompt versions
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
            'component': 'category',
            'version': 'v1.0',
            'prompt_text': CATEGORY_PROMPT_V1,
            'model': 'gpt-4o',
            'notes': 'Initial category classification prompt',
            'is_active': True,
        },
        {
            'component': 'tags',
            'version': 'v1.0',
            'prompt_text': TAGS_DESCRIPTION_PROMPT_V1,
            'model': 'gpt-4o',
            'notes': 'Initial tags generation prompt (shared with description)',
            'is_active': True,
        },
        {
            'component': 'description',
            'version': 'v1.0',
            'prompt_text': TAGS_DESCRIPTION_PROMPT_V1,
            'model': 'gpt-4o',
            'notes': 'Initial description generation prompt (shared with tags)',
            'is_active': True,
        },
        {
            'component': 'embedding',
            'version': 'v1.0',
            'prompt_text': None,  # Embeddings don't use a prompt
            'model': 'text-embedding-3-small',
            'notes': 'Initial embedding generation using restaurant name + tags',
            'is_active': True,
        },
    ])


def downgrade() -> None:
    # Remove version tracking columns from restaurant_images
    op.drop_column('restaurant_images', 'tags_updated_at')
    op.drop_column('restaurant_images', 'tags_version')
    op.drop_column('restaurant_images', 'category_updated_at')
    op.drop_column('restaurant_images', 'category_version')
    
    # Remove version tracking columns from restaurants
    op.drop_column('restaurants', 'embedding_updated_at')
    op.drop_column('restaurants', 'embedding_version')
    op.drop_column('restaurants', 'description_updated_at')
    op.drop_column('restaurants', 'description_version')
    
    # Drop prompt_versions table
    op.drop_table('prompt_versions')

