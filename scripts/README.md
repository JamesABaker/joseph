# Joseph Model Training

This directory contains scripts to train the GAN-based AI text detector on HC3 dataset features.

## Overview

The Joseph detector uses a **Generative Adversarial Network (GAN)** trained on **10 statistical features**:
- Shannon entropy, burstiness, lexical diversity, word length variance, punctuation diversity, vocabulary richness
- Average sentence length, sentence length standard deviation
- Special character ratio, uppercase ratio

No transformer models required - lightweight and memory-efficient (~80MB RAM at runtime).

## Current Production Model

**Model:** `gan_detector_tuned_best.pt`
**Hyperparameters:** hidden_dim=384, latent_dim=128, epochs=250
**Performance:** 91.8% accuracy, 87.2% F1-score on HC3 validation set

## Training Pipeline

### Prerequisites

Install training dependencies:

```bash
uv pip install -e ".[training]"
```

This installs PyTorch, scikit-learn, pandas, datasets, tqdm, and pyarrow.

### Step 1: Extract Features
```bash
uv run python scripts/prepare_features.py
```

**What it does:**
- Loads full HC3 dataset (human vs ChatGPT-3.5 text pairs)
- Extracts 10 statistical features for each sample using `EntropyDetector`
- Splits into train (70%), validation (15%), test (15%)
- Saves to `data/joseph_training/*.parquet`

**Time estimate:** ~10-20 minutes (no heavy models needed)

### Step 2: Train GAN Model
```bash
uv run python scripts/train_gan_model.py
```

**What it does:**
- Loads pre-extracted features from parquet files
- Trains adversarial GAN (Generator creates fake features, Discriminator detects them)
- Adversarial training makes the discriminator robust
- Saves to `models/gan_detector_best.pt`

**Time estimate:** ~15-20 minutes on CPU

### Step 3: Hyperparameter Tuning (Optional)
```bash
uv run python scripts/tune_hyperparameters.py
```

**What it does:**
- Grid search over hidden_dim, latent_dim, learning rates, epochs
- Tests each configuration on validation set
- Saves best model to `models/gan_detector_tuned_best.pt`
- Records results in `models/hyperparameter_tuning_results.csv`

**Time estimate:** ~2-4 hours (tests 50+ configurations)

### Step 4: Evaluate Model
```bash
uv run python scripts/evaluate_model.py
```

**What it does:**
- Loads trained model and test set
- Generates metrics (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix and error analysis

**Time estimate:** <1 minute

## Integration with App

The production model is loaded by [app/ml_model.py](app/ml_model.py):

```python
# In AIDetector.__init__()
self.gan_detector = GANDetector(feature_dim=10, latent_dim=128, hidden_dim=384)
self.gan_detector.load("models/gan_detector_tuned_best.pt")
```

## File Structure

```
scripts/
  prepare_features.py           # Extract 10 features from HC3
  train_gan_model.py            # Train base GAN
  tune_hyperparameters.py       # Optimize hyperparameters
  evaluate_model.py             # Test evaluation

data/
  joseph_training/
    train.parquet               # 70% training data
    val.parquet                 # 15% validation data
    test.parquet                # 15% test data

models/
  gan_detector_tuned_best.pt    # Production model (hyperparameter tuned)
  best_hyperparameters.json     # Optimal configuration
  hyperparameter_tuning_results.csv  # Grid search results
```

## Expected Performance

Based on HC3 dataset (ChatGPT-3.5 vs human):
- **Accuracy:** 91.8%
- **F1-Score:** 87.2%
- **ROC-AUC:** 97.1%
- **Memory:** ~80MB RAM

See [Model Evaluation Report](../docs/MODEL_EVALUATION.md) for detailed analysis.

## Retraining

To retrain from scratch:

```bash
# Extract features (if not already done)
uv run python scripts/prepare_features.py

# Train base model
uv run python scripts/train_gan_model.py

# Optional: Tune hyperparameters
uv run python scripts/tune_hyperparameters.py

# Evaluate
uv run python scripts/evaluate_model.py
```

## Docker Integration

Training runs **locally** with uv/pip. The production Docker image includes only the trained model and runtime dependencies.

The model file (`models/gan_detector_tuned_best.pt`) is included in the repository and Docker image.

## Notes

- **Feature consistency:** Training uses the same `EntropyDetector` as production
- **No transformers:** Lightweight statistical features only
- **Adversarial robustness:** GAN training creates a detector resistant to edge cases
- **Version control:** Production model is tracked in git (~15MB)
