from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, JSON, UniqueConstraint, Index, Float, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class Restaurant(Base):
    __tablename__ = "restaurants"

    id = Column(Integer, primary_key=True, index=True)
    place_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    embedding = Column(JSON, nullable=True)  # Vector embedding for semantic search (stored as JSON array)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationship to images
    images = relationship("RestaurantImage", back_populates="restaurant", cascade="all, delete-orphan")
    # Relationship to user interactions
    interactions = relationship("UserRestaurantInteraction", back_populates="restaurant", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Restaurant(id={self.id}, place_id='{self.place_id}', name='{self.name}')>"


class RestaurantImage(Base):
    __tablename__ = "restaurant_images"

    id = Column(Integer, primary_key=True, index=True)
    restaurant_id = Column(Integer, ForeignKey("restaurants.id"), nullable=False, index=True)
    photo_name = Column(String, nullable=False)  # Google Places photo name/path
    gcs_url = Column(String, nullable=False)  # Public URL to image in GCS
    gcs_bucket_path = Column(String, nullable=False)  # Path in bucket for reference
    category = Column(String, nullable=True)  # e.g., "interior", "exterior", "food", "menu", "bar", "other"
    ai_tags = Column(JSON, nullable=True)  # AI-generated tags as JSON array
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationship to restaurant
    restaurant = relationship("Restaurant", back_populates="images")

    # Unique constraint: one image per photo_name per restaurant
    __table_args__ = (
        UniqueConstraint('restaurant_id', 'photo_name', name='uq_restaurant_photo'),
        Index('idx_restaurant_id', 'restaurant_id'),
        Index('idx_photo_name', 'photo_name'),
    )

    def __repr__(self):
        return f"<RestaurantImage(id={self.id}, restaurant_id={self.restaurant_id}, photo_name='{self.photo_name}')>"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationship to interactions
    interactions = relationship("UserRestaurantInteraction", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}')>"


class UserRestaurantInteraction(Base):
    __tablename__ = "user_restaurant_interactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    restaurant_id = Column(Integer, ForeignKey("restaurants.id"), nullable=False, index=True)
    interaction_type = Column(String, nullable=False)  # 'like', 'dislike', 'rating'
    rating = Column(Float, nullable=True)  # For rating interactions (1.0-5.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="interactions")
    restaurant = relationship("Restaurant", back_populates="interactions")

    # Unique constraint: one interaction per user-restaurant pair
    __table_args__ = (
        UniqueConstraint('user_id', 'restaurant_id', name='uq_user_restaurant'),
        Index('idx_user_id', 'user_id'),
        Index('idx_restaurant_id', 'restaurant_id'),
        Index('idx_interaction_type', 'interaction_type'),
    )

    def __repr__(self):
        return f"<UserRestaurantInteraction(id={self.id}, user_id={self.user_id}, restaurant_id={self.restaurant_id}, type='{self.interaction_type}')>"

