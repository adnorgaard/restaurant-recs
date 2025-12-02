"""Add serpapi_data_id to restaurants table

Revision ID: 005_add_serpapi_data_id
Revises: 004_add_version_tracking
Create Date: 2024-12-02

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '005_add_serpapi_data_id'
down_revision = '004_add_version_tracking'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add serpapi_data_id column to restaurants table
    op.add_column('restaurants', sa.Column('serpapi_data_id', sa.String(), nullable=True))
    
    # Add index for faster lookups
    op.create_index('idx_serpapi_data_id', 'restaurants', ['serpapi_data_id'])


def downgrade() -> None:
    op.drop_index('idx_serpapi_data_id', table_name='restaurants')
    op.drop_column('restaurants', 'serpapi_data_id')

