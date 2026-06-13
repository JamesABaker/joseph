"""Tests for database models."""

from datetime import datetime

import pytest

from app.models import Result, User


def test_user_creation(db_session):
    """Test creating a user."""
    user = User(
        email="test@example.com",
        oauth_provider="github",
        oauth_id="123456",
        name="Test User",
        picture="https://example.com/pic.jpg",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    assert user.id is not None
    assert user.email == "test@example.com"
    assert user.oauth_provider == "github"
    assert user.oauth_id == "123456"
    assert user.name == "Test User"
    assert user.picture == "https://example.com/pic.jpg"
    assert user.is_active is True
    assert isinstance(user.created_at, datetime)


def test_user_unique_email(db_session):
    """Test that email must be unique."""
    user1 = User(email="duplicate@example.com", oauth_provider="github", oauth_id="111")
    db_session.add(user1)
    db_session.commit()

    user2 = User(email="duplicate@example.com", oauth_provider="github", oauth_id="222")
    db_session.add(user2)

    with pytest.raises(Exception):  # SQLAlchemy IntegrityError
        db_session.commit()


def test_result_creation(db_session):
    """Test creating a detection result."""
    # First create a user
    user = User(email="user@example.com", oauth_provider="github", oauth_id="789")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    # Create result
    result_data = {
        "text_analyzed": "This is a test text.",
        "human_probability": 75.5,
        "ai_probability": 24.5,
        "prediction": "human",
        "ml_human_probability": 80.0,
        "ml_ai_probability": 20.0,
        "perplexity": 50.0,
        "shannon_entropy": 4.5,
        "burstiness": 0.6,
        "lexical_diversity": 0.7,
        "word_length_variance": 0.5,
        "punctuation_diversity": 0.4,
        "vocabulary_richness": 0.8,
        "avg_sentence_length": 15.0,
        "sentence_length_std": 2.5,
        "special_char_ratio": 0.05,
        "uppercase_ratio": 0.1,
        "entropy_ai_probability": 30.0,
        "entropy_human_probability": 70.0,
    }
    result = Result(user_id=user.id, **result_data)
    db_session.add(result)
    db_session.commit()
    db_session.refresh(result)

    assert result.id is not None
    assert result.user_id == user.id
    assert result.text_analyzed == "This is a test text."
    assert result.prediction == "human"
    assert result.human_probability == 75.5
    assert result.ai_probability == 24.5
    assert isinstance(result.created_at, datetime)


def test_user_results_relationship(db_session):
    """Test user-results relationship."""
    user = User(email="relation@example.com", oauth_provider="github", oauth_id="999")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    # Add multiple results
    for i in range(3):
        result = Result(
            user_id=user.id,
            text_analyzed=f"Test {i}",
            human_probability=50.0,
            ai_probability=50.0,
            prediction="human",
            ml_human_probability=50.0,
            ml_ai_probability=50.0,
            perplexity=50.0,
            shannon_entropy=4.0,
            burstiness=0.5,
            lexical_diversity=0.5,
            word_length_variance=0.5,
            punctuation_diversity=0.5,
            vocabulary_richness=0.5,
            avg_sentence_length=10.0,
            sentence_length_std=2.0,
            special_char_ratio=0.05,
            uppercase_ratio=0.1,
            entropy_ai_probability=50.0,
            entropy_human_probability=50.0,
        )
        db_session.add(result)

    db_session.commit()
    db_session.refresh(user)

    assert len(user.results) == 3
    assert all(r.user_id == user.id for r in user.results)
