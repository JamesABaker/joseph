"""
Dependencies for protected routes.
"""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.auth import verify_token
from app.database import get_db
from app.models import User

security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """
    Dependency to get current authenticated user from JWT token.

    Args:
        credentials: HTTP Bearer token
        db: Database session

    Returns:
        Current user object

    Raises:
        HTTPException: If token is invalid or user not found
    """
    token = credentials.credentials
    payload = verify_token(token, token_type="access")  # nosec B106

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = int(payload.get("sub"))
    user = db.query(User).filter(User.id == user_id, User.is_active.is_(True)).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """
    Dependency to optionally get current authenticated user from JWT token.
    Returns None if no token or invalid token.

    Args:
        credentials: HTTP Bearer token (optional)
        db: Database session

    Returns:
        Current user object or None
    """
    if not credentials:
        return None

    token = credentials.credentials
    payload = verify_token(token, token_type="access")  # nosec B106

    if not payload:
        return None

    user_id = int(payload.get("sub"))
    user = db.query(User).filter(User.id == user_id, User.is_active.is_(True)).first()

    return user
