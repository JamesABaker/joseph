"""
Evaluate trained models on GPT-4 test data.

This script evaluates both the Joseph Random Forest and GAN models
on the downloaded GPT test dataset to see how they perform on
newer AI text compared to the ChatGPT-3.5 training data.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))  # noqa: E402

from app.gan_model import GANDetector  # noqa: E402
from app.ml_model import AIDetector  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_features_from_texts(texts, entropy_detector, roberta_detector):
    """Extract features from raw texts."""
    logger.info("Extracting features from texts...")

    features_list = []

    for text in tqdm(texts, desc="Processing texts"):
        # Get entropy features
        entropy_result = entropy_detector.detect(text)

        # Get RoBERTa prediction
        inputs = roberta_detector.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        )
        with torch.no_grad():
            outputs = roberta_detector.roberta_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            roberta_ai_prob = probs[0][1].item() * 100

        # Combine features
        features = {
            "perplexity": entropy_result["perplexity"],
            "shannon_entropy": entropy_result["shannon_entropy"],
            "burstiness": entropy_result["burstiness"],
            "lexical_diversity": entropy_result["lexical_diversity"],
            "word_length_variance": entropy_result["word_length_variance"],
            "punctuation_diversity": entropy_result["punctuation_diversity"],
            "vocabulary_richness": entropy_result["vocabulary_richness"],
            "roberta_ai_prob": roberta_ai_prob,
        }
        features_list.append(features)

    return pd.DataFrame(features_list)


def evaluate_joseph_model(features_df, labels, joseph_model):
    """Evaluate Joseph Random Forest model."""
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATING JOSEPH RANDOM FOREST MODEL")
    logger.info("=" * 60)

    # Predict
    predictions = joseph_model.predict(features_df)
    probabilities = joseph_model.predict_proba(features_df)[:, 1]

    # Metrics
    accuracy = accuracy_score(labels, predictions)
    roc_auc = roc_auc_score(labels, probabilities)

    logger.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"ROC-AUC: {roc_auc:.4f}")

    logger.info("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=["Human", "AI"]))

    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(labels, predictions)
    print(cm)
    logger.info(f"  [[TN={cm[0, 0]}, FP={cm[0, 1]}],")
    logger.info(f"   [FN={cm[1, 0]}, TP={cm[1, 1]}]]")

    # Error analysis
    errors = labels != predictions
    logger.info("Error Analysis:")
    logger.info(
        f"  Total errors: {errors.sum()} / {len(labels)} ({errors.sum()/len(labels)*100:.2f}%)"
    )
    logger.info(
        f"  False Positives (Human → AI): {cm[0, 1]} ({cm[0, 1]/(cm[0, 0]+cm[0, 1])*100:.2f}%)"
    )
    logger.info(
        f"  False Negatives (AI → Human): {cm[1, 0]} ({cm[1, 0]/(cm[1, 0]+cm[1, 1])*100:.2f}%)"
    )

    return {
        "model": "Joseph Random Forest",
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "predictions": predictions,
        "probabilities": probabilities,
    }


def evaluate_gan_model(features_df, labels, gan_detector, device="cpu"):
    """Evaluate GAN model."""
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATING GAN DISCRIMINATOR MODEL")
    logger.info("=" * 60)

    gan_detector.eval()
    gan_detector.to_device(device)

    # Convert to tensor
    features_tensor = torch.FloatTensor(features_df.values).to(device)

    # Predict
    with torch.no_grad():
        probabilities = gan_detector.predict(features_tensor).cpu().numpy()

    predictions = (probabilities > 0.5).astype(int)

    # Metrics
    accuracy = accuracy_score(labels, predictions)
    roc_auc = roc_auc_score(labels, probabilities)

    logger.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"ROC-AUC: {roc_auc:.4f}")

    logger.info("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=["Human", "AI"]))

    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(labels, predictions)
    print(cm)
    logger.info(f"  [[TN={cm[0, 0]}, FP={cm[0, 1]}],")
    logger.info(f"   [FN={cm[1, 0]}, TP={cm[1, 1]}]]")

    # Error analysis
    errors = labels != predictions
    logger.info("Error Analysis:")
    logger.info(
        f"  Total errors: {errors.sum()} / {len(labels)} ({errors.sum()/len(labels)*100:.2f}%)"
    )
    logger.info(
        f"  False Positives (Human → AI): {cm[0, 1]} ({cm[0, 1]/(cm[0, 0]+cm[0, 1])*100:.2f}%)"
    )
    logger.info(
        f"  False Negatives (AI → Human): {cm[1, 0]} ({cm[1, 0]/(cm[1, 0]+cm[1, 1])*100:.2f}%)"
    )

    return {
        "model": "GAN Discriminator",
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "predictions": predictions,
        "probabilities": probabilities,
    }


def compare_models(joseph_results, gan_results):
    """Compare both models side by side."""
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)

    logger.info(f"\n{'Metric':<20} {'Joseph RF':<15} {'GAN':<15} {'Winner':<10}")
    logger.info("-" * 60)

    # Accuracy
    joseph_acc = joseph_results["accuracy"]
    gan_acc = gan_results["accuracy"]
    winner = "Joseph" if joseph_acc > gan_acc else "GAN" if gan_acc > joseph_acc else "Tie"
    logger.info(f"{'Accuracy':<20} {joseph_acc:.4f}          {gan_acc:.4f}          {winner:<10}")

    # ROC-AUC
    joseph_auc = joseph_results["roc_auc"]
    gan_auc = gan_results["roc_auc"]
    winner = "Joseph" if joseph_auc > gan_auc else "GAN" if gan_auc > joseph_auc else "Tie"
    logger.info(f"{'ROC-AUC':<20} {joseph_auc:.4f}          {gan_auc:.4f}          {winner:<10}")

    logger.info("\nConclusion:")
    if joseph_acc > gan_acc:
        logger.info("  → Joseph Random Forest performs better on this dataset")
    elif gan_acc > joseph_acc:
        logger.info("  → GAN Discriminator performs better on this dataset")
    else:
        logger.info("  → Both models perform equally well")


def main():
    """Main evaluation script."""
    project_root = Path(__file__).parent.parent
    test_data_path = project_root / "data" / "gpt4_test" / "gpt_wiki_test.parquet"
    models_dir = project_root / "models"

    # Check if test data exists
    if not test_data_path.exists():
        logger.error(f"Test data not found at {test_data_path}")
        logger.error("Run 'python scripts/download_gpt4_test_data.py' first")
        return

    # Check if models exist
    joseph_model_path = models_dir / "joseph_v1.pkl"
    gan_model_path = models_dir / "gan_detector_best.pt"

    if not joseph_model_path.exists():
        logger.error(f"Joseph model not found at {joseph_model_path}")
        return

    if not gan_model_path.exists():
        logger.error(f"GAN model not found at {gan_model_path}")
        return

    logger.info("=" * 60)
    logger.info("GPT TEST DATA EVALUATION")
    logger.info("=" * 60)

    # Load test data
    logger.info(f"\nLoading test data from {test_data_path}...")
    df = pd.read_parquet(test_data_path)
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"  Human: {(df['label'] == 0).sum()}")
    logger.info(f"  AI: {(df['label'] == 1).sum()}")

    # Initialize detectors for feature extraction
    logger.info("\nInitializing detectors...")
    ai_detector = AIDetector()

    # Extract features
    features_df = extract_features_from_texts(
        df["text"].tolist(), ai_detector.entropy_detector, ai_detector
    )

    logger.info("Feature statistics:")
    logger.info(
        f"  Perplexity - Human: {features_df[df['label'] == 0]['perplexity'].mean():.2f}, "
        f"AI: {features_df[df['label'] == 1]['perplexity'].mean():.2f}"
    )
    logger.info(
        f"  RoBERTa AI% - Human: {features_df[df['label'] == 0]['roberta_ai_prob'].mean():.2f}%, "
        f"AI: {features_df[df['label'] == 1]['roberta_ai_prob'].mean():.2f}%"
    )

    # Evaluate Joseph model
    joseph_results = evaluate_joseph_model(
        features_df, df["label"].values, ai_detector.joseph_model
    )

    # Evaluate GAN model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gan_detector = GANDetector(feature_dim=8, latent_dim=100)
    gan_detector.load(str(gan_model_path))
    gan_results = evaluate_gan_model(features_df, df["label"].values, gan_detector, device)

    # Compare models
    compare_models(joseph_results, gan_results)

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
