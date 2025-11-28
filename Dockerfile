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
RUN uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --system .

# Copy application code
COPY app/ ./app/

# Copy trained Joseph model
COPY models/ ./models/

# Pre-download the model during build to avoid startup delays
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('Hello-SimpleAI/chatgpt-detector-roberta'); \
    AutoModelForSequenceClassification.from_pretrained('Hello-SimpleAI/chatgpt-detector-roberta')"

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
