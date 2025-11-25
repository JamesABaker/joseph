# Joseph

[![Docker Build](https://github.com/JamesABaker/verif/actions/workflows/docker.yml/badge.svg)](https://github.com/JamesABaker/verif/actions/workflows/docker.yml)
[![pre-commit](https://github.com/JamesABaker/verif/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/JamesABaker/verif/actions/workflows/pre-commit.yml)

> *"There are simply too many notes."* — Emperor Joseph II to Mozart

**Joseph** is a hybrid AI text detection system combining machine learning with information theory. Named after Emperor Joseph II, who famously told Mozart his music had "too many notes"—a reminder that something trying to be too perfect can be detected.


## Features

- 🧠 **Hybrid Detection** - combines ML model with entropy analysis for broader insights
- 📊 **Information Theory** - perplexity, Shannon entropy, burstiness, lexical diversity
- 🎯 **Sensitive Against Modern LLMs** - entropy features work on GPT-4/GPT-5 output
- 🐳 **Fully containerized** - runs anywhere with Docker
- 🌐 **Web UI + REST API** - easy to use, easy to integrate
- 🚀 **Fast inference** - results in seconds
- 💾 **Model caching** - downloads once, runs forever
- 🔒 **Privacy-first** - runs completely locally, no external API calls

## Quick Start

### Run with Docker Compose (Recommended)

Install [Docker](https://docs.docker.com/get-docker/) if you haven't already.

```bash
# Clone or navigate to the project directory
cd verif

# Start the application
docker-compose up

# Or run in detached mode
docker-compose up -d
```

The application will be available at:
- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Run with Docker (Manual)

```bash
# Build the image
docker build -t verif .

# Run the container
docker run -p 8000:8000 verif
```

## Usage

### Web Interface

1. Open http://localhost:8000 in your browser
2. Paste or type text into the textarea
3. Click "Detect AI"
4. View the results

### REST API

**Detect Text:**
```bash
curl -X POST http://localhost:8000/api/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text to analyze here..."}'
```

**Response:**
```json
{
  "human_probability": 45.3,
  "ai_probability": 54.7,
  "prediction": "ai",
  "text_length": 123
}
```

**Get Model Info:**
```bash
curl http://localhost:8000/api/model-info
```

**Health Check:**
```bash
curl http://localhost:8000/health
```

### Interactive API Documentation

Visit http://localhost:8000/docs for interactive API documentation with a built-in testing interface.

## Development

### Local Development Setup

**Using Conda**
```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate verif

# Install pre-commit hooks
pre-commit install
```

## Model Information

- **ML Component**: RoBERTa-base trained on ChatGPT output
- **Entropy Features**:
  - **Perplexity**: Measures text predictability (lower = more AI-like)
  - **Shannon Entropy**: Character-level randomness measure
  - **Burstiness**: Sentence complexity variation (human writing varies more)
  - **Lexical Diversity**: Unique word ratio (type-token ratio)


## Future Enhancements

- [ ] Batch processing API endpoint
- [ ] API rate limiting
- [ ] User 2fa authentication
- [ ] Method refinement
- [ ] Deployment
- [ ] Volume data persistence for users


## License

MIT License - feel free to use this project for learning, development, or production.
