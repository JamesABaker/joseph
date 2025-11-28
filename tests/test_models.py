"""Tests for database models."""

from datetime import datetime

import pytest

from app.models import Result, User


def test_user_creation(test_db):
    """Test creating a user."""
    user = User(
        email="test@example.com",
        oauth_provider="github",
        oauth_id="123456",
        name="Test User",
        picture="https://example.com/pic.jpg",
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)

    assert user.id is not None
    assert user.email == "test@example.com"
    assert user.oauth_provider == "github"
    assert user.oauth_id == "123456"
    assert user.name == "Test User"
    assert user.picture == "https://example.com/pic.jpg"
    assert user.is_active is True
    assert isinstance(user.created_at, datetime)


def test_user_unique_email(test_db):
    """Test that email must be unique."""
    user1 = User(email="duplicate@example.com", oauth_provider="github", oauth_id="111")
    test_db.add(user1)
    test_db.commit()

    user2 = User(email="duplicate@example.com", oauth_provider="github", oauth_id="222")
    test_db.add(user2)

    with pytest.raises(Exception):  # SQLAlchemy IntegrityError
        test_db.commit()


def test_result_creation(test_db):
    """Test creating a detection result."""
    # First create a user
    user = User(email="user@example.com", oauth_provider="github", oauth_id="789")
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)

    # Create result
    result = Result(
        user_id=user.id,
        text_analyzed="This is a test text.",
        human_probability=75.5,
        ai_probability=24.5,
        prediction="human",
        ml_human_probability=80.0,
        ml_ai_probability=20.0,
        perplexity=50.0,
        shannon_entropy=4.5,
        burstiness=0.6,
        lexical_diversity=0.7,
        word_length_variance=0.5,
        punctuation_diversity=0.4,
        vocabulary_richness=0.8,
        entropy_ai_probability=30.0,
        entropy_human_probability=70.0,
    )
    test_db.add(result)
    test_db.flush()  # Use flush instead of commit for test isolation
    test_db.refresh(result)

    assert result.id is not None
    assert result.user_id == user.id
    assert result.text_analyzed == "This is a test text."
    assert result.prediction == "human"
    assert result.human_probability == 75.5
    assert result.ai_probability == 24.5
    assert isinstance(result.created_at, datetime)


def test_user_results_relationship(test_db):
    """Test user-results relationship."""
    user = User(email="relation@example.com", oauth_provider="github", oauth_id="999")
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)

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
            entropy_ai_probability=50.0,
            entropy_human_probability=50.0,
        )
        test_db.add(result)

    test_db.flush()  # Use flush instead of commit
    test_db.refresh(user)

    assert len(user.results) == 3
    assert all(r.user_id == user.id for r in user.results)
