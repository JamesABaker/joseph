"""
Authentication utilities for OAuth and JWT token management.
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.config import settings
from app.models import RefreshToken, User


def create_access_token(data: dict) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(data: dict) -> str:
    """Create a JWT refresh token. Each token is unique via a random jti claim."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh", "jti": secrets.token_urlsafe(16)})
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def verify_token(token: str, token_type: str = "access") -> Optional[dict]:  # nosec B107
    """
    Verify and decode a JWT token.

    Args:
        token: JWT token string
        token_type: Expected token type ('access' or 'refresh')

    Returns:
        Decoded token payload or None if invalid
    """
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        if payload.get("type") != token_type:
            return None
        return payload
    except JWTError:
        return None


def _hash_token(token: str) -> str:
    """Return the SHA-256 hex digest of a token (never store raw tokens)."""
    return hashlib.sha256(token.encode()).hexdigest()


def store_refresh_token(db: Session, user_id: int, token: str) -> RefreshToken:
    """
    Persist a refresh token hash so it can be revoked later.

    Args:
        db: Database session
        user_id: Owner of the token
        token: Raw refresh token JWT (only the hash is stored)

    Returns:
        Persisted RefreshToken record
    """
    expires_at = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    db_token = RefreshToken(
        user_id=user_id,
        token_hash=_hash_token(token),
        expires_at=expires_at,
    )
    db.add(db_token)
    db.commit()
    return db_token


def verify_refresh_token_in_db(db: Session, token: str) -> Optional[RefreshToken]:
    """
    Check that a refresh token exists in the DB, is not revoked, and has not expired.

    Args:
        db: Database session
        token: Raw refresh token JWT

    Returns:
        RefreshToken record if valid, None otherwise
    """
    return (
        db.query(RefreshToken)
        .filter(
            RefreshToken.token_hash == _hash_token(token),
            RefreshToken.revoked.is_(False),
            RefreshToken.expires_at > datetime.utcnow(),
        )
        .first()
    )


def revoke_refresh_token(db: Session, token: str) -> bool:
    """
    Revoke a refresh token so it can no longer be used.

    Args:
        db: Database session
        token: Raw refresh token JWT

    Returns:
        True if the token was found and revoked, False if not found
    """
    db_token = db.query(RefreshToken).filter(RefreshToken.token_hash == _hash_token(token)).first()
    if db_token:
        db_token.revoked = True
        db.commit()
        return True
    return False


def get_or_create_user(
    db: Session,
    email: str,
    oauth_provider: str,
    oauth_id: str,
    name: str = None,
    picture: str = None,
) -> User:
    """
    Get existing user or create new user from OAuth data.

    Args:
        db: Database session
        email: User email from OAuth
        oauth_provider: OAuth provider name ('google' or 'github')
        oauth_id: User ID from OAuth provider
        name: User's display name
        picture: Profile picture URL

    Returns:
        User object
    """
    # Try to find existing user by email
    user = db.query(User).filter(User.email == email).first()

    if user:
        # Update OAuth info if changed
        user.oauth_provider = oauth_provider
        user.oauth_id = oauth_id
        if name:
            user.name = name
        if picture:
            user.picture = picture
        user.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(user)
    else:
        # Create new user
        user = User(
            email=email,
            oauth_provider=oauth_provider,
            oauth_id=oauth_id,
            name=name,
            picture=picture,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    return user
