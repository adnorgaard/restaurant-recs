"""Add users and embeddings

Revision ID: 002_add_users_embeddings
Revises: 001_initial
Create Date: 2024-01-02 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '002_add_users_embeddings'
down_revision: Union[str, None] = '001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add embedding column to restaurants table
    op.add_column('restaurants', sa.Column('embedding', postgresql.JSON(astext_type=sa.Text()), nullable=True))
    
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('hashed_password', sa.String(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('is_superuser', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    
    # Create user_restaurant_interactions table
    op.create_table(
        'user_restaurant_interactions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('restaurant_id', sa.Integer(), nullable=False),
        sa.Column('interaction_type', sa.String(), nullable=False),
        sa.Column('rating', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.ForeignKeyConstraint(['restaurant_id'], ['restaurants.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'restaurant_id', name='uq_user_restaurant')
    )
    op.create_index(op.f('ix_user_restaurant_interactions_id'), 'user_restaurant_interactions', ['id'], unique=False)
    op.create_index('idx_user_id', 'user_restaurant_interactions', ['user_id'], unique=False)
    op.create_index('idx_interactions_restaurant_id', 'user_restaurant_interactions', ['restaurant_id'], unique=False)
    op.create_index('idx_interaction_type', 'user_restaurant_interactions', ['interaction_type'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_interaction_type', table_name='user_restaurant_interactions')
    op.drop_index('idx_interactions_restaurant_id', table_name='user_restaurant_interactions')
    op.drop_index('idx_user_id', table_name='user_restaurant_interactions')
    op.drop_index(op.f('ix_user_restaurant_interactions_id'), table_name='user_restaurant_interactions')
    op.drop_table('user_restaurant_interactions')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_table('users')
    op.drop_column('restaurants', 'embedding')

