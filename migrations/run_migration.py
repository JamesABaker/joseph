#!/usr/bin/env python3
"""
Database migration script to add new entropy features.

This script adds the 4 new columns needed for the 10-feature model:
- avg_sentence_length
- sentence_length_std
- special_char_ratio
- uppercase_ratio

Usage:
    python migrations/run_migration.py

The DATABASE_URL environment variable must be set (automatically on Render).
"""

import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))  # noqa: E402

from sqlalchemy import text  # noqa: E402

from app.database import engine  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_migration():
    """Run the database migration to add new columns."""
    try:
        logger.info("Starting database migration...")
        db_url = os.getenv("DATABASE_URL", "Not set")
        logger.info(f"Database URL: {db_url[:50]}...")

        with engine.begin() as conn:
            # Check if columns already exist
            check_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'results'
            AND column_name IN ('avg_sentence_length', 'sentence_length_std',
                                'special_char_ratio', 'uppercase_ratio')
            """
            result = conn.execute(text(check_sql))
            existing_columns = {row[0] for row in result}

            if len(existing_columns) == 4:
                logger.info("✅ All 4 columns already exist, skipping migration")
                return True

            logger.info(f"Found {len(existing_columns)} existing columns: {existing_columns}")
            logger.info("Adding missing columns...")

            # Add columns one by one with better error handling
            columns_to_add = [
                ("avg_sentence_length", "DOUBLE PRECISION"),
                ("sentence_length_std", "DOUBLE PRECISION"),
                ("special_char_ratio", "DOUBLE PRECISION"),
                ("uppercase_ratio", "DOUBLE PRECISION"),
            ]

            for col_name, col_type in columns_to_add:
                if col_name not in existing_columns:
                    logger.info(f"Adding column: {col_name}")
                    conn.execute(
                        text(f"ALTER TABLE results ADD COLUMN {col_name} {col_type} DEFAULT 0.0")
                    )
                    conn.execute(text(f"ALTER TABLE results ALTER COLUMN {col_name} SET NOT NULL"))
                    logger.info(f"✓ Added {col_name}")
                else:
                    logger.info(f"✓ Column {col_name} already exists")

        logger.info("✅ Migration completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        logger.exception("Full traceback:")
        return False


if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1)
