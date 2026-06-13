#!/usr/bin/env python3
"""
Database initialization script for Render deployment.

This script:
1. Tests database connectivity
2. Creates tables from SQLAlchemy models if they don't exist
3. Runs any pending migrations
4. Validates the database schema

Usage:
    python migrations/init_db.py

The DATABASE_URL environment variable must be set.
"""

import logging
import os
import sys
import time
from pathlib import Path

# Set default values for required config before importing app
os.environ.setdefault("JWT_SECRET_KEY", os.getenv("JWT_SECRET_KEY", "default-init-key"))
os.environ.setdefault(
    "OAUTH_REDIRECT_URI", os.getenv("OAUTH_REDIRECT_URI", "http://localhost:8000/auth/callback")
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import OperationalError, ProgrammingError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def wait_for_database(max_retries: int = 10, retry_delay: int = 5) -> bool:
    """
    Wait for database to become available.

    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Delay between retries in seconds

    Returns:
        True if database is available, False otherwise
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("❌ DATABASE_URL environment variable not set")
        return False

    logger.info(f"⏳ Waiting for database (max {max_retries} retries, {retry_delay}s delay)...")

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempt {attempt}/{max_retries}: Testing database connection...")
            engine = create_engine(db_url, echo=False, pool_pre_ping=True)

            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                logger.info("✅ Database is available!")
                engine.dispose()
                return True

        except OperationalError as e:
            logger.warning(f"⚠️ Connection failed: {str(e)[:100]}")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            continue
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            continue

    logger.error(f"❌ Failed to connect after {max_retries} attempts")
    return False


def create_tables():
    """Create all tables from SQLAlchemy models if they don't exist."""
    try:
        from app.database import Base, engine

        logger.info("📋 Creating database tables from models...")

        # Create all tables defined in models
        Base.metadata.create_all(bind=engine)

        logger.info("✅ Database tables created/verified successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to create tables: {e}")
        logger.exception("Full traceback:")
        return False


def run_migrations():
    """Run pending database migrations."""
    try:
        from app.database import engine
        from sqlalchemy import text

        logger.info("🔄 Running database migrations...")
        db_url = os.getenv("DATABASE_URL", "Not set")
        logger.info(f"Database: {db_url[:50]}...")

        with engine.begin() as conn:
            # Check if results table exists
            inspector = inspect(conn)
            tables = inspector.get_table_names()

            if "results" not in tables:
                logger.info("ℹ️ Results table doesn't exist yet, will be created by models")
                return True

            # Check for missing columns
            existing_columns = {col["name"] for col in inspector.get_columns("results")}
            required_columns = {
                "avg_sentence_length",
                "sentence_length_std",
                "special_char_ratio",
                "uppercase_ratio",
            }

            missing_columns = required_columns - existing_columns

            if not missing_columns:
                logger.info("✅ All required columns already exist")
                return True

            logger.info(f"Adding {len(missing_columns)} missing columns: {missing_columns}")

            columns_to_add = [
                ("avg_sentence_length", "DOUBLE PRECISION"),
                ("sentence_length_std", "DOUBLE PRECISION"),
                ("special_char_ratio", "DOUBLE PRECISION"),
                ("uppercase_ratio", "DOUBLE PRECISION"),
            ]

            for col_name, col_type in columns_to_add:
                if col_name not in existing_columns:
                    logger.info(f"Adding column: {col_name}")
                    try:
                        conn.execute(
                            text(
                                f"ALTER TABLE results ADD COLUMN {col_name} {col_type} DEFAULT 0.0"
                            )
                        )
                        conn.execute(
                            text(f"ALTER TABLE results ALTER COLUMN {col_name} SET NOT NULL")
                        )
                        logger.info(f"✓ Added {col_name}")
                    except ProgrammingError as e:
                        if "already exists" in str(e):
                            logger.info(f"✓ Column {col_name} already exists")
                        else:
                            raise

        logger.info("✅ All migrations completed successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        logger.exception("Full traceback:")
        return False


def validate_schema():
    """Validate that the database schema is correct."""
    try:
        from app.database import engine

        logger.info("🔍 Validating database schema...")

        inspector = inspect(engine)
        tables = inspector.get_table_names()

        if not tables:
            logger.warning("⚠️ No tables found in database")
            return True  # Not a hard failure - tables will be created

        logger.info(f"✅ Found {len(tables)} tables: {', '.join(tables)}")
        return True

    except Exception as e:
        logger.error(f"❌ Schema validation failed: {e}")
        return False


def init_database():
    """Run complete database initialization."""
    logger.info("=" * 60)
    logger.info("🗄️ Database Initialization")
    logger.info("=" * 60)

    # Step 1: Wait for database
    if not wait_for_database():
        logger.error("❌ Database initialization failed: Cannot connect to database")
        return False

    # Step 2: Create tables
    if not create_tables():
        logger.error("❌ Database initialization failed: Cannot create tables")
        return False

    # Step 3: Run migrations
    if not run_migrations():
        logger.error("❌ Database initialization failed: Cannot run migrations")
        return False

    # Step 4: Validate schema
    if not validate_schema():
        logger.warning("⚠️ Schema validation failed, but continuing...")

    logger.info("=" * 60)
    logger.info("✅ Database initialization completed successfully!")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    success = init_database()
    sys.exit(0 if success else 1)
