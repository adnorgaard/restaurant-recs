"""Add restaurant description column

Revision ID: 003_add_description
Revises: 002_add_users_embeddings
Create Date: 2024-01-03 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '003_add_description'
down_revision: Union[str, None] = '002_add_users_embeddings'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add description column to restaurants table
    op.add_column('restaurants', sa.Column('description', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('restaurants', 'description')

