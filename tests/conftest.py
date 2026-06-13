"""Pytest configuration and fixtures."""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Set test environment variables before importing app
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-testing-only")
os.environ.setdefault("OAUTH_REDIRECT_URI", "http://localhost:8000/auth/callback")

from app.database import Base, get_db  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def test_db_engine():
    """Create test database engine."""
    # Use in-memory SQLite for tests
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def test_db(test_db_engine):
    """Create test database session."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_db_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(scope="function")
def db_session(test_db):
    """Alias for test_db for backwards compatibility."""
    return test_db


@pytest.fixture
def client(test_db):
    """Create test client with database override."""

    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    # Mock the detector to avoid loading models and prevent startup DB connection
    with patch("app.main.detector") as mock_detector:
        mock_detector.detect.return_value = {
            "human_probability": 75.5,
            "ai_probability": 24.5,
            "prediction": "human",
            "text_length": 100,
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
            "sentence_length_std": 5.0,
            "special_char_ratio": 0.02,
            "uppercase_ratio": 0.05,
            "entropy_ai_probability": 30.0,
            "entropy_human_probability": 70.0,
        }

        app.dependency_overrides[get_db] = override_get_db

        # Patch the startup event to prevent model loading and DB connection
        with patch("app.main.startup_event"):
            with TestClient(app, raise_server_exceptions=False) as test_client:
                yield test_client

        app.dependency_overrides.clear()


@pytest.fixture
def mock_detector():
    """Mock detector for testing without loading models."""
    detector = MagicMock()
    detector.detect.return_value = {
        "human_probability": 75.5,
        "ai_probability": 24.5,
        "prediction": "human",
        "text_length": 100,
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
        "sentence_length_std": 5.0,
        "special_char_ratio": 0.02,
        "uppercase_ratio": 0.05,
        "entropy_ai_probability": 30.0,
        "entropy_human_probability": 70.0,
    }
    return detector


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return {
        "human": (
            "I walked through the park yesterday and saw a beautiful sunset. "
            "The colors were amazing!"
        ),
        "ai": (
            "The implementation of artificial intelligence systems requires careful "
            "consideration of ethical frameworks and potential societal impacts."
        ),
        "short": "Hi",
        "empty": "",
        "long": " ".join(["word"] * 1000),
    }
