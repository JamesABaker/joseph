"""
Train the Joseph Random Forest model on extracted entropy features.

This script:
1. Loads train/val parquet files
2. Trains Random Forest classifier on 8 features
3. Validates performance
4. Saves model artifact and feature stats
"""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data(data_dir):
    """Load train and validation parquet files."""
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")

    logger.info(f"Loaded {len(train_df)} training samples")
    logger.info(f"Loaded {len(val_df)} validation samples")

    return train_df, val_df


def prepare_features(df):
    """Separate features and labels."""
    feature_cols = [
        "perplexity",
        "shannon_entropy",
        "burstiness",
        "lexical_diversity",
        "word_length_variance",
        "punctuation_diversity",
        "vocabulary_richness",
        "roberta_ai_prob",
    ]

    X = df[feature_cols]
    y = df["label"]

    return X, y


def train_model(X_train, y_train):
    """Train Random Forest classifier."""
    logger.info("Training Random Forest model...")

    # Random Forest with reasonable defaults
    # n_estimators=100: number of trees
    # max_depth=10: prevent overfitting
    # random_state=42: reproducibility
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, verbose=1
    )

    model.fit(X_train, y_train)
    logger.info("Training complete")

    return model


def evaluate_model(model, X_val, y_val):
    """Evaluate model on validation set."""
    logger.info("Evaluating model on validation set...")

    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]  # Probability of AI class

    accuracy = accuracy_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    logger.info(f"Validation Accuracy: {accuracy:.4f}")
    logger.info(f"Validation ROC-AUC: {roc_auc:.4f}")

    logger.info("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=["Human", "AI"]))

    logger.info("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    # Feature importances
    feature_names = [
        "perplexity",
        "shannon_entropy",
        "burstiness",
        "lexical_diversity",
        "word_length_variance",
        "punctuation_diversity",
        "vocabulary_richness",
        "roberta_ai_prob",
    ]

    importances = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    logger.info("\nFeature Importances:")
    print(importances.to_string(index=False))

    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "feature_importances": importances.to_dict("records"),
    }


def save_model(model, metrics, output_dir):
    """Save trained model and metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / "joseph_v1.pkl"
    logger.info(f"Saving model to {model_path}")
    joblib.dump(model, model_path)

    # Save metrics
    metrics_path = output_dir / "training_metrics.json"
    logger.info(f"Saving metrics to {metrics_path}")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Model and metrics saved to {output_dir}")


def main():
    """Main training pipeline."""
    logger.info("Starting Joseph model training")

    # Paths
    data_dir = Path(__file__).parent.parent / "data" / "joseph_training"
    model_dir = Path(__file__).parent.parent / "models"

    # Load data
    train_df, val_df = load_data(data_dir)

    # Prepare features
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)

    logger.info(f"Feature shape: {X_train.shape}")
    logger.info(f"Training set class distribution: {y_train.value_counts().to_dict()}")

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate
    metrics = evaluate_model(model, X_val, y_val)

    # Save
    save_model(model, metrics, model_dir)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {model_dir / 'joseph_v1.pkl'}")
    logger.info("Next step: Run scripts/evaluate_model.py to test on held-out test set")


if __name__ == "__main__":
    main()
