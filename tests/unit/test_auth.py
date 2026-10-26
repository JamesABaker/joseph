"""Tests for authentication functionality."""

from datetime import datetime, timedelta

import pytest
from jose import jwt

from app.auth import (
    create_access_token,
    create_refresh_token,
    revoke_refresh_token,
    store_refresh_token,
    verify_refresh_token_in_db,
    verify_token,
)
from app.config import settings
from app.models import User


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


# ---------------------------------------------------------------------------
# DB-backed refresh token tests
# ---------------------------------------------------------------------------


@pytest.fixture
def db_user(test_db):
    """Get or create a minimal user for refresh-token tests."""
    user = test_db.query(User).filter(User.email == "token_test@example.com").first()
    if not user:
        user = User(
            email="token_test@example.com",
            oauth_provider="github",
            oauth_id="999",
        )
        test_db.add(user)
        test_db.commit()
        test_db.refresh(user)
    return user


def test_store_refresh_token(test_db, db_user):
    """Storing a refresh token creates a DB record with a hashed value."""
    token = create_refresh_token({"sub": str(db_user.id)})
    record = store_refresh_token(test_db, db_user.id, token)

    assert record.id is not None
    assert record.user_id == db_user.id
    assert record.revoked is False
    assert record.token_hash != token  # raw token must not be stored


def test_verify_refresh_token_in_db_valid(test_db, db_user):
    """A freshly stored token is returned as valid."""
    token = create_refresh_token({"sub": str(db_user.id)})
    store_refresh_token(test_db, db_user.id, token)

    result = verify_refresh_token_in_db(test_db, token)
    assert result is not None


def test_verify_refresh_token_in_db_revoked(test_db, db_user):
    """A revoked token is rejected."""
    token = create_refresh_token({"sub": str(db_user.id)})
    store_refresh_token(test_db, db_user.id, token)
    revoke_refresh_token(test_db, token)

    result = verify_refresh_token_in_db(test_db, token)
    assert result is None


def test_verify_refresh_token_in_db_expired(test_db, db_user):
    """An expired DB record is rejected."""
    token = create_refresh_token({"sub": str(db_user.id)})
    record = store_refresh_token(test_db, db_user.id, token)

    # Back-date the expiry to simulate expiry
    record.expires_at = datetime.utcnow() - timedelta(seconds=1)
    test_db.commit()

    result = verify_refresh_token_in_db(test_db, token)
    assert result is None


def test_verify_refresh_token_in_db_unknown(test_db, db_user):
    """A token that was never stored is rejected."""
    token = create_refresh_token({"sub": str(db_user.id)})
    # Not stored — should return None
    result = verify_refresh_token_in_db(test_db, token)
    assert result is None


def test_revoke_refresh_token_returns_true(test_db, db_user):
    """Revoking a known token returns True."""
    token = create_refresh_token({"sub": str(db_user.id)})
    store_refresh_token(test_db, db_user.id, token)

    assert revoke_refresh_token(test_db, token) is True


def test_revoke_refresh_token_unknown_returns_false(test_db, db_user):
    """Revoking an unknown token returns False."""
    token = create_refresh_token({"sub": str(db_user.id)})
    assert revoke_refresh_token(test_db, token) is False


def test_token_rotation(test_db, db_user):
    """Using a token revokes it and the old token no longer validates."""
    token = create_refresh_token({"sub": str(db_user.id)})
    store_refresh_token(test_db, db_user.id, token)

    # Simulate rotation: revoke old, issue new
    revoke_refresh_token(test_db, token)
    new_token = create_refresh_token({"sub": str(db_user.id)})
    store_refresh_token(test_db, db_user.id, new_token)

    assert verify_refresh_token_in_db(test_db, token) is None  # old revoked
    assert verify_refresh_token_in_db(test_db, new_token) is not None  # new valid
