# Joseph Model Training

This directory contains scripts to train the Joseph Random Forest model on HC3 dataset features.

## Overview

The Joseph model is trained on **8 features**:
- 7 entropy-based metrics (perplexity, shannon entropy, burstiness, lexical diversity, word length variance, punctuation diversity, vocabulary richness)
- 1 RoBERTa AI probability score

Instead of a hand-tuned weighted combination, we train a Random Forest classifier to learn optimal feature weighting from labeled HC3 data.

## Training Pipeline

### Prerequisites

Install training dependencies:

```bash
uv pip install -e ".[training]"
```

This installs scikit-learn, pandas, datasets, tqdm, and pyarrow.

### Step 1: Extract Features
```bash
uv run python scripts/prepare_features.py
```

**What it does:**
- Loads full HC3 dataset (human vs ChatGPT text pairs)
- Extracts 8 features for each sample using `EntropyDetector` and RoBERTa
- Splits into train (70%), validation (15%), test (15%)
- Saves to `data/joseph_training/*.parquet`

**Time estimate:** ~30-60 minutes depending on hardware (processes full HC3 dataset)

### Step 2: Train Model
```bash
uv run python scripts/train_joseph_model.py
```

**What it does:**
- Loads training and validation parquet files
- Trains Random Forest (100 trees, max_depth=10)
- Evaluates on validation set
- Saves trained model to `models/joseph_v1.pkl`
- Reports feature importances

**Time estimate:** ~5-10 minutes

### Step 3: Evaluate on Test Set
```bash
uv run python scripts/evaluate_model.py
```

**What it does:**
- Loads trained model and held-out test set
- Generates comprehensive metrics (accuracy, ROC-AUC, confusion matrix)
- Analyzes errors and prediction distributions
- Tests different decision thresholds

**Time estimate:** <1 minute

## Integration with App

Once trained, the model is automatically loaded by `app/ml_model.py`:

```python
# In AIDetector.__init__()
self.joseph_model = joblib.load("models/joseph_v1.pkl")
```

**Fallback behavior:** If `joseph_v1.pkl` doesn't exist, the app falls back to the old weighted hybrid approach (20% RoBERTa + 80% entropy).

## File Structure

```
scripts/
  prepare_features.py      # Step 1: Extract features from HC3
  train_joseph_model.py    # Step 2: Train Random Forest
  evaluate_model.py        # Step 3: Test set evaluation

data/
  joseph_training/
    train.parquet          # 70% of data
    val.parquet            # 15% of data
    test.parquet           # 15% of data

models/
  joseph_v1.pkl            # Trained Random Forest model
  training_metrics.json    # Validation metrics and feature importances
```

## Expected Performance

Based on HC3 dataset (ChatGPT vs human text):
- **Accuracy:** Should be >85% on test set
- **ROC-AUC:** Should be >0.90

The Random Forest will automatically learn which features are most predictive.

## Retraining

To retrain with different parameters:

1. Edit `train_joseph_model.py` - modify `RandomForestClassifier` parameters
2. Run training pipeline again (Steps 2-3)
3. Compare metrics with previous version

## Docker Integration

Training is meant to run **locally** with uv/pip, not in Docker. The production Docker image only includes runtime dependencies.

To train locally:

```bash
# Install training dependencies
uv pip install -e ".[training]"

# Run training pipeline
uv run python scripts/prepare_features.py
uv run python scripts/train_joseph_model.py
uv run python scripts/evaluate_model.py
```

The trained `models/joseph_v1.pkl` is automatically loaded by the app.

## Notes

- **Feature consistency:** Training uses the same `EntropyDetector` class as inference, ensuring identical feature extraction
- **Version control:** The trained model (`models/joseph_v1.pkl`) is included in git and the Docker image for production deployment
