# Use conda base image
FROM continuumio/miniconda3:24.7.1-0

# Set working directory
WORKDIR /app

# Copy environment file
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml && \
    conda clean -afy

# Copy application code
COPY app/ ./app/

# Activate environment and set path
ENV PATH=/opt/conda/envs/verif/bin:$PATH

# Pre-download the model during build to avoid startup delays
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('Hello-SimpleAI/chatgpt-detector-roberta'); \
    AutoModelForSequenceClassification.from_pretrained('Hello-SimpleAI/chatgpt-detector-roberta')"

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
