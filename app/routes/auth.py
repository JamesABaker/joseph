"""
Authentication routes for OAuth login and token management.
"""

import logging
import urllib.parse

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import create_access_token, create_refresh_token, get_or_create_user, verify_token
from app.config import settings
from app.database import get_db
from app.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


class TokenResponse(BaseModel):
    """Response model for token endpoints."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: dict


class RefreshRequest(BaseModel):
    """Request model for token refresh."""

    refresh_token: str


@router.get("/login/github")
async def login_github():
    """Initiate GitHub OAuth login flow."""
    if not settings.GITHUB_CLIENT_ID:
        raise HTTPException(status_code=501, detail="GitHub OAuth not configured")

    github_auth_url = (
        f"https://github.com/login/oauth/authorize?"
        f"client_id={settings.GITHUB_CLIENT_ID}&"
        f"redirect_uri={settings.OAUTH_REDIRECT_URI}&"
        f"scope=read:user user:email&"
        f"state=github"
    )
    return RedirectResponse(github_auth_url)


@router.get("/callback")
async def oauth_callback(
    code: str = Query(...),
    state: str = Query(...),
    db: Session = Depends(get_db),
):
    """
    Handle OAuth callback from GitHub.

    Args:
        code: Authorization code from OAuth provider
        state: State parameter (should be 'github')
        db: Database session
    """
    try:
        if state != "github":
            raise HTTPException(status_code=400, detail="Invalid OAuth provider")

        user_data = await _handle_github_callback(code)

        # Create or get user
        user = get_or_create_user(
            db=db,
            email=user_data["email"],
            oauth_provider="github",
            oauth_id=user_data["id"],
            name=user_data.get("name"),
            picture=user_data.get("picture"),
        )

        # Generate tokens
        access_token = create_access_token({"sub": str(user.id), "email": user.email})
        refresh_token = create_refresh_token({"sub": str(user.id)})

        # Redirect to frontend with tokens in hash
        user_json = urllib.parse.quote(
            f'{{"id": {user.id}, "email": "{user.email}", '
            f'"name": "{user.name or ""}", '
            f'"picture": "{user.picture or ""}"}}'
        )
        redirect_url = (
            f"/#access_token={access_token}"
            f"&refresh_token={refresh_token}"
            f"&token_type=bearer"
            f"&user={user_json}"
        )
        return RedirectResponse(redirect_url)

    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        raise HTTPException(status_code=500, detail="Authentication failed")


async def _handle_github_callback(code: str) -> dict:
    """Exchange GitHub authorization code for user data."""
    async with httpx.AsyncClient() as client:
        # Exchange code for access token
        token_response = await client.post(
            "https://github.com/login/oauth/access_token",
            data={
                "code": code,
                "client_id": settings.GITHUB_CLIENT_ID,
                "client_secret": settings.GITHUB_CLIENT_SECRET,
                "redirect_uri": settings.OAUTH_REDIRECT_URI,
            },
            headers={"Accept": "application/json"},
        )
        token_response.raise_for_status()
        token_data = token_response.json()

        # Get user info
        user_response = await client.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {token_data['access_token']}"},
        )
        user_response.raise_for_status()
        user_info = user_response.json()

        # Get primary email (GitHub may have multiple emails)
        email_response = await client.get(
            "https://api.github.com/user/emails",
            headers={"Authorization": f"Bearer {token_data['access_token']}"},
        )
        email_response.raise_for_status()
        emails = email_response.json()
        primary_email = next((e["email"] for e in emails if e["primary"]), emails[0]["email"])

        return {
            "id": str(user_info["id"]),
            "email": primary_email,
            "name": user_info.get("name"),
            "picture": user_info.get("avatar_url"),
        }


@router.post("/refresh", response_model=TokenResponse)
async def refresh_access_token(
    request: RefreshRequest,
    db: Session = Depends(get_db),
):
    """
    Refresh access token using refresh token.

    Args:
        request: Refresh token request
        db: Database session
    """
    payload = verify_token(request.refresh_token, token_type="refresh")  # nosec B106
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    user_id = int(payload.get("sub"))
    user = db.query(User).filter(User.id == user_id, User.is_active.is_(True)).first()

    if not user:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    # Generate new tokens
    access_token = create_access_token({"sub": str(user.id), "email": user.email})
    refresh_token = create_refresh_token({"sub": str(user.id)})

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user={
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "picture": user.picture,
        },
    )
