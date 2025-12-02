"""Add is_displayed flag to restaurant_images table

Revision ID: 006_add_is_displayed_flag
Revises: 005_add_serpapi_data_id
Create Date: 2024-12-02

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '006_add_is_displayed_flag'
down_revision = '005_add_serpapi_data_id'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add is_displayed column to restaurant_images table
    # Default to False - images are stored but not displayed until selected by quota
    op.add_column('restaurant_images', sa.Column('is_displayed', sa.Boolean(), nullable=False, server_default='false'))
    
    # Add index for faster filtering of displayed images
    op.create_index('idx_is_displayed', 'restaurant_images', ['is_displayed'])
    
    # Composite index for efficient queries: get displayed images for a restaurant
    op.create_index('idx_restaurant_displayed', 'restaurant_images', ['restaurant_id', 'is_displayed'])


def downgrade() -> None:
    op.drop_index('idx_restaurant_displayed', table_name='restaurant_images')
    op.drop_index('idx_is_displayed', table_name='restaurant_images')
    op.drop_column('restaurant_images', 'is_displayed')

