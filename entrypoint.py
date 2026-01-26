#!/usr/bin/env python3
"""
Entrypoint script for Docker deployment.
Runs database migrations before starting the FastAPI server.
"""

import logging
import os
import subprocess  # nosec B404
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_migrations():
    """Run database migrations."""
    logger.info("🚀 Starting deployment process...")
    logger.info("📊 Running database migrations...")

    migration_script = Path(__file__).parent / "migrations" / "run_migration.py"

    try:
        result = subprocess.run(  # nosec B603
            [sys.executable, str(migration_script)],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(result.stdout)
        logger.info("✅ Migrations completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Migration failed with exit code {e.returncode}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        # Don't fail deployment if migration fails (columns might already exist)
        logger.warning("⚠️  Continuing with deployment despite migration issues")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during migration: {e}")
        logger.warning("⚠️  Continuing with deployment")
        return False


def start_server():
    """Start the FastAPI server."""
    logger.info("🌐 Starting FastAPI server...")

    host = os.getenv("HOST", "0.0.0.0")  # nosec B104
    port = int(os.getenv("PORT", "8000"))

    # Use exec to replace the process
    os.execvp(  # nosec B606 B607
        "uvicorn",
        ["uvicorn", "app.main:app", "--host", host, "--port", str(port)],
    )


if __name__ == "__main__":
    run_migrations()
    start_server()
