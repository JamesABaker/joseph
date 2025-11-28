"""
Database models for users and results.
"""

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.database import Base


class User(Base):
    """User account model."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    oauth_provider = Column(String(50), nullable=False)  # 'google' or 'github'
    oauth_id = Column(String(255), nullable=False)  # OAuth provider's user ID
    name = Column(String(255), nullable=True)
    picture = Column(String(512), nullable=True)  # Profile picture URL
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationship to results
    results = relationship("Result", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.email}>"


class Result(Base):
    """Detection result model linked to a user."""

    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    text_analyzed = Column(Text, nullable=False)
    human_probability = Column(Float, nullable=False)
    ai_probability = Column(Float, nullable=False)
    prediction = Column(String(10), nullable=False)  # 'human' or 'ai'

    # ML scores
    ml_human_probability = Column(Float, nullable=False)
    ml_ai_probability = Column(Float, nullable=False)

    # Entropy metrics
    perplexity = Column(Float, nullable=False)
    shannon_entropy = Column(Float, nullable=False)
    burstiness = Column(Float, nullable=False)
    lexical_diversity = Column(Float, nullable=False)
    word_length_variance = Column(Float, nullable=False)
    punctuation_diversity = Column(Float, nullable=False)
    vocabulary_richness = Column(Float, nullable=False)
    entropy_ai_probability = Column(Float, nullable=False)
    entropy_human_probability = Column(Float, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship to user
    user = relationship("User", back_populates="results")

    def __repr__(self):
        return f"<Result {self.id} for User {self.user_id}>"
