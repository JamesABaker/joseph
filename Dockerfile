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

# Copy trained model
COPY models/ ./models/

# Verify the tuned GAN model loads correctly (10 features, no transformers)
RUN python -c "from app.gan_model import GANDetector; \
    model = GANDetector(feature_dim=10, latent_dim=128, hidden_dim=384); \
    model.load('models/gan_detector_tuned_best.pt'); \
    print('Tuned GAN model loaded successfully')"

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
