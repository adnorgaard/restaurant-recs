"""Initial schema

Revision ID: 001_initial
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create restaurants table
    op.create_table(
        'restaurants',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('place_id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_restaurants_id'), 'restaurants', ['id'], unique=False)
    op.create_index(op.f('ix_restaurants_place_id'), 'restaurants', ['place_id'], unique=True)

    # Create restaurant_images table
    op.create_table(
        'restaurant_images',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('restaurant_id', sa.Integer(), nullable=False),
        sa.Column('photo_name', sa.String(), nullable=False),
        sa.Column('gcs_url', sa.String(), nullable=False),
        sa.Column('gcs_bucket_path', sa.String(), nullable=False),
        sa.Column('category', sa.String(), nullable=True),
        sa.Column('ai_tags', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['restaurant_id'], ['restaurants.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('restaurant_id', 'photo_name', name='uq_restaurant_photo')
    )
    op.create_index(op.f('ix_restaurant_images_id'), 'restaurant_images', ['id'], unique=False)
    op.create_index('idx_restaurant_id', 'restaurant_images', ['restaurant_id'], unique=False)
    op.create_index('idx_photo_name', 'restaurant_images', ['photo_name'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_photo_name', table_name='restaurant_images')
    op.drop_index('idx_restaurant_id', table_name='restaurant_images')
    op.drop_index(op.f('ix_restaurant_images_id'), table_name='restaurant_images')
    op.drop_table('restaurant_images')
    op.drop_index(op.f('ix_restaurants_place_id'), table_name='restaurants')
    op.drop_index(op.f('ix_restaurants_id'), table_name='restaurants')
    op.drop_table('restaurants')

