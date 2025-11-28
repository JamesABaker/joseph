"""
Evaluate the trained Joseph model on held-out test set.

This script:
1. Loads trained model and test data
2. Generates predictions on test set
3. Reports comprehensive metrics and analysis
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_test_data(data_dir):
    """Load test parquet file."""
    test_df = pd.read_parquet(data_dir / "test.parquet")
    logger.info(f"Loaded {len(test_df)} test samples")
    return test_df


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


def evaluate(model, X_test, y_test):
    """Comprehensive evaluation on test set."""
    logger.info("Running evaluation on test set...")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # AI probability

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    logger.info("\n" + "=" * 60)
    logger.info("TEST SET RESULTS")
    logger.info("=" * 60)
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"ROC-AUC: {roc_auc:.4f}")

    logger.info("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Human", "AI"], digits=4))

    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Parse confusion matrix
    tn, fp, fn, tp = cm.ravel()
    logger.info(f"\nTrue Negatives (Human correctly identified): {tn}")
    logger.info(f"False Positives (Human misclassified as AI): {fp}")
    logger.info(f"False Negatives (AI misclassified as Human): {fn}")
    logger.info(f"True Positives (AI correctly identified): {tp}")

    # Error analysis
    logger.info("\nError Analysis:")
    logger.info(f"False Positive Rate: {fp/(fp+tn):.4f}")
    logger.info(f"False Negative Rate: {fn/(fn+tp):.4f}")

    # Probability distribution analysis
    logger.info("\nPrediction Probability Distribution:")
    human_probs = y_pred_proba[y_test == 0]
    ai_probs = y_pred_proba[y_test == 1]

    logger.info(f"Human samples - Mean AI prob: {human_probs.mean():.3f} (should be low)")
    logger.info(f"Human samples - Median AI prob: {pd.Series(human_probs).median():.3f}")

    logger.info(f"AI samples - Mean AI prob: {ai_probs.mean():.3f} (should be high)")
    logger.info(f"AI samples - Median AI prob: {pd.Series(ai_probs).median():.3f}")

    # Threshold analysis
    logger.info("\nThreshold Analysis (default is 0.5):")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        acc = accuracy_score(y_test, y_pred_thresh)
        logger.info(f"  Threshold {threshold:.1f}: Accuracy = {acc:.4f}")

    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def main():
    """Main evaluation pipeline."""
    logger.info("Starting Joseph model evaluation")

    # Paths
    data_dir = Path(__file__).parent.parent / "data" / "joseph_training"
    model_dir = Path(__file__).parent.parent / "models"
    model_path = model_dir / "joseph_v1.pkl"

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    # Load test data
    test_df = load_test_data(data_dir)

    # Prepare features
    X_test, y_test = prepare_features(test_df)

    logger.info(f"Test set class distribution: {y_test.value_counts().to_dict()}")

    # Evaluate
    evaluate(model, X_test, y_test)

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info("If metrics look good, integrate model into app/ml_model.py")


if __name__ == "__main__":
    main()
