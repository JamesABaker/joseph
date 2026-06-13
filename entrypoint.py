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
    """Run database initialization and migrations."""
    logger.info("🚀 Starting deployment process...")
    logger.info("🗄️ Initializing database...")

    init_script = Path(__file__).parent / "migrations" / "init_db.py"

    try:
        result = subprocess.run(  # nosec B603
            [sys.executable, str(init_script)],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(result.stdout)
        logger.info("✅ Database initialization completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Database initialization failed with exit code {e.returncode}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        # FAIL deployment if initialization fails - database is required
        logger.error("❌ Cannot start server without database initialization")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error during database initialization: {e}")
        logger.error("❌ Cannot start server without database initialization")
        sys.exit(1)


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
