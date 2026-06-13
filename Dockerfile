# Use Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml .

# Install dependencies (production only, no training deps)
# Install PyTorch CPU-only first to avoid CUDA dependencies
# Use --no-cache to reduce build size
RUN uv pip install --system --no-cache torch --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --system --no-cache .

# Copy application code
COPY app/ ./app/

# Copy trained model
COPY models/ ./models/

# Copy migrations
COPY migrations/ ./migrations/

# Copy entrypoint script
COPY entrypoint.py ./entrypoint.py

# Expose port
EXPOSE 8000

# Run the entrypoint script (migrations + application)
CMD ["python", "entrypoint.py"]
