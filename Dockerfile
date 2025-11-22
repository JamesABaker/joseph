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

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
