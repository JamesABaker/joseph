"""Tests for authentication functionality."""

from datetime import datetime, timedelta

from jose import jwt

from app.auth import create_access_token, create_refresh_token, verify_token
from app.config import settings


def test_create_access_token():
    """Test access token creation."""
    data = {"sub": "123", "email": "test@example.com"}
    token = create_access_token(data)

    assert token is not None
    assert isinstance(token, str)

    # Decode and verify
    payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    assert payload["sub"] == "123"
    assert payload["email"] == "test@example.com"
    assert payload["type"] == "access"
    assert "exp" in payload


def test_create_refresh_token():
    """Test refresh token creation."""
    data = {"sub": "456"}
    token = create_refresh_token(data)

    assert token is not None
    assert isinstance(token, str)

    # Decode and verify
    payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    assert payload["sub"] == "456"
    assert payload["type"] == "refresh"
    assert "exp" in payload


def test_verify_token_valid_access():
    """Test verifying a valid access token."""
    data = {"sub": "789", "email": "user@example.com"}
    token = create_access_token(data)

    payload = verify_token(token, token_type="access")  # nosec B106

    assert payload is not None
    assert payload["sub"] == "789"
    assert payload["email"] == "user@example.com"
    assert payload["type"] == "access"


def test_verify_token_valid_refresh():
    """Test verifying a valid refresh token."""
    data = {"sub": "999"}
    token = create_refresh_token(data)

    payload = verify_token(token, token_type="refresh")  # nosec B106

    assert payload is not None
    assert payload["sub"] == "999"
    assert payload["type"] == "refresh"


def test_verify_token_wrong_type():
    """Test verifying token with wrong type."""
    data = {"sub": "111"}
    access_token = create_access_token(data)

    # Try to verify access token as refresh
    payload = verify_token(access_token, token_type="refresh")  # nosec B106

    assert payload is None


def test_verify_token_invalid():
    """Test verifying an invalid token."""
    invalid_token = "invalid.token.here"

    payload = verify_token(invalid_token, token_type="access")  # nosec B106

    assert payload is None


def test_verify_token_expired():
    """Test verifying an expired token."""
    data = {"sub": "222", "email": "expired@example.com"}
    expire = datetime.utcnow() - timedelta(minutes=1)  # Already expired
    to_encode = data.copy()
    to_encode.update({"exp": expire, "type": "access"})
    expired_token = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    payload = verify_token(expired_token, token_type="access")  # nosec B106

    assert payload is None
