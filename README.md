# Joseph

[![Docker Build](https://github.com/JamesABaker/verif/actions/workflows/docker.yml/badge.svg)](https://github.com/JamesABaker/verif/actions/workflows/docker.yml)
[![pre-commit](https://github.com/JamesABaker/verif/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/JamesABaker/verif/actions/workflows/pre-commit.yml)

> *"Too many notes."* — Emperor Joseph II to Mozart

**Joseph** is a hybrid AI text detection system combining machine learning with information theory.

## Features

- 🧠 **Hybrid Detection** - combines ML model with entropy analysis for broader insights
- 📊 **Information Theory** - perplexity, Shannon entropy, burstiness, lexical diversity
  - 🎯 **Sensitive Against Modern LLMs** - entropy features work on GPT-4/GPT-5 output
- 🐳 **Fully containerized** - runs anywhere with Docker
- 🌐 **Web UI + REST API** - easy to use, easy to integrate
- 💾 **Model caching** - downloads once, runs forever

## Quick Start

### Run with Docker Compose (Recommended)

Install [Docker](https://docs.docker.com/get-docker/) if you haven't already.

```bash
git clone https://github.com/JamesABaker/joseph.git
cd joseph
docker compose up --build
```

The application will be available at:
- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Usage

### Web Interface

1. Open http://localhost:8000 in your browser
2. Paste or type text into the textarea
3. Click "Detect AI"
4. View the results

### Interactive API Documentation

Visit http://localhost:8000/docs for interactive API documentation with a built-in testing interface.

## Development

### Local Development Setup

**Using uv (recommended)**
```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install production dependencies
uv pip install -e .

# For training (optional)
uv pip install -e ".[training]"

# Install pre-commit hooks
pre-commit install
```

**Using pip**
```bash
# Production dependencies
pip install -e .

# With training dependencies
pip install -e ".[training]"
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
- [x] User 2fa authentication
- [ ] Method refinement
- [ ] Deployment
- [x] Volume data persistence for users


## License

MIT License - feel free to use this project for learning, development, or production.
