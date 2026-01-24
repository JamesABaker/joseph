"""
Application configuration from environment variables.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    DATABASE_URL: str

    # PostgreSQL connection settings (for docker-compose)
    POSTGRES_USER: str = "joseph"
    POSTGRES_PASSWORD: str = "joseph"  # nosec B106
    POSTGRES_DB: str = "joseph"

    # JWT
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # OAuth - GitHub
    GITHUB_CLIENT_ID: str = ""
    GITHUB_CLIENT_SECRET: str = ""
    OAUTH_REDIRECT_URI: str

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
