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
    migration_sql = """
    -- Add new columns (nullable first to avoid breaking existing rows)
    ALTER TABLE results ADD COLUMN IF NOT EXISTS avg_sentence_length FLOAT;
    ALTER TABLE results ADD COLUMN IF NOT EXISTS sentence_length_std FLOAT;
    ALTER TABLE results ADD COLUMN IF NOT EXISTS special_char_ratio FLOAT;
    ALTER TABLE results ADD COLUMN IF NOT EXISTS uppercase_ratio FLOAT;

    -- Set default values for existing rows (if any)
    UPDATE results SET avg_sentence_length = 0.0 WHERE avg_sentence_length IS NULL;
    UPDATE results SET sentence_length_std = 0.0 WHERE sentence_length_std IS NULL;
    UPDATE results SET special_char_ratio = 0.0 WHERE special_char_ratio IS NULL;
    UPDATE results SET uppercase_ratio = 0.0 WHERE uppercase_ratio IS NULL;

    -- Make columns non-nullable after setting defaults
    ALTER TABLE results ALTER COLUMN avg_sentence_length SET NOT NULL;
    ALTER TABLE results ALTER COLUMN sentence_length_std SET NOT NULL;
    ALTER TABLE results ALTER COLUMN special_char_ratio SET NOT NULL;
    ALTER TABLE results ALTER COLUMN uppercase_ratio SET NOT NULL;
    """

    try:
        logger.info("Starting database migration...")
        logger.info(f"Database URL: {os.getenv('DATABASE_URL', 'Not set')[:30]}...")

        with engine.connect() as conn:
            # Execute each statement separately
            statements = [s.strip() for s in migration_sql.split(";") if s.strip()]

            for i, statement in enumerate(statements, 1):
                logger.info(f"Executing statement {i}/{len(statements)}...")
                conn.execute(text(statement))
                conn.commit()
                logger.info(f"✓ Statement {i} completed")

        logger.info("✅ Migration completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False


if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1)
