# Use Python slim image
FROM python:3.11-slim

# Update the system CA store for outbound HTTPS.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml .

# Keep PyTorch in its own layer so dependency changes can reuse this large install.
RUN pip install --no-cache-dir --retries 10 --timeout 120 torch --index-url https://download.pytorch.org/whl/cpu

# Keep this separate from PyTorch to preserve its Docker cache layer.
# hadolint ignore=DL3059
RUN uv pip install --system --no-cache .

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
