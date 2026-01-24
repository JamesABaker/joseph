"""
Train the GAN-based AI text detection model.

This script:
1. Loads extracted features from joseph_training parquet files
2. Trains the GAN detector with adversarial training
3. Validates performance on validation set
4. Saves trained model to models/gan_detector_v1.pt
"""

import json
import logging
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from tqdm import tqdm  # noqa: E402

from app.gan_model import GANDetector  # noqa: E402

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
    """Separate features and labels, convert to tensors."""
    # Expanded 10 features (lightweight, no transformers)
    feature_cols = [
        "shannon_entropy",
        "burstiness",
        "lexical_diversity",
        "word_length_variance",
        "punctuation_diversity",
        "vocabulary_richness",
        "avg_sentence_length",
        "sentence_length_std",
        "special_char_ratio",
        "uppercase_ratio",
    ]

    X = torch.FloatTensor(df[feature_cols].values)
    y = torch.FloatTensor(df["label"].values)

    return X, y


def create_dataloader(X, y, batch_size=64, shuffle=True):
    """Create PyTorch DataLoader."""
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def train_gan_detector(
    detector, train_loader, val_loader, epochs=50, device="cpu", checkpoint_dir=None
):
    """
    Train GAN detector with adversarial training.

    Args:
        detector: GANDetector instance
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
    """
    best_val_acc = 0.0
    history = {"train_loss": [], "val_acc": [], "val_auc": []}

    detector.to_device(device)
    detector.train()

    logger.info(f"Starting training for {epochs} epochs on {device}")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")

    for epoch in range(epochs):
        # Training phase
        detector.train()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0

        for batch_idx, (features, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        ):
            features = features.to(device)
            labels = labels.to(device)

            # Train discriminator more frequently than generator for stability
            train_generator = batch_idx % 5 == 0

            losses = detector.train_step(features, labels, train_generator=train_generator)
            epoch_d_loss += losses["discriminator_loss"]
            epoch_g_loss += losses["generator_loss"]

        avg_d_loss = epoch_d_loss / len(train_loader)
        avg_g_loss = epoch_g_loss / len(train_loader)

        # Validation phase
        detector.eval()
        val_preds = []
        val_probs = []
        val_labels = []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                predictions = detector.predict(features)
                val_probs.extend(predictions.cpu().numpy())
                val_preds.extend((predictions > 0.5).cpu().numpy())
                val_labels.extend(labels.numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_probs)

        history["train_loss"].append(avg_d_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if checkpoint_dir:
                best_path = checkpoint_dir / "gan_detector_best.pt"
                detector.save(str(best_path))
                logger.info(f"Saved best model (acc={val_acc:.4f}) to {best_path}")

    return history


def evaluate_model(detector, val_loader, device="cpu"):
    """Evaluate model on validation set."""
    logger.info("\nFinal Evaluation on Validation Set")
    logger.info("=" * 50)

    detector.eval()
    detector.to_device(device)

    val_preds = []
    val_probs = []
    val_labels = []

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            predictions = detector.predict(features)
            val_probs.extend(predictions.cpu().numpy())
            val_preds.extend((predictions > 0.5).cpu().numpy())
            val_labels.extend(labels.numpy())

    accuracy = accuracy_score(val_labels, val_preds)
    roc_auc = roc_auc_score(val_labels, val_probs)

    logger.info(f"Final Validation Accuracy: {accuracy:.4f}")
    logger.info(f"Final Validation ROC-AUC: {roc_auc:.4f}")

    logger.info("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=["Human", "AI"]))

    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(val_labels, val_preds)
    print(cm)
    logger.info(f"  [[TN={cm[0, 0]}, FP={cm[0, 1]}],")
    logger.info(f"   [FN={cm[1, 0]}, TP={cm[1, 1]}]]")

    return {"accuracy": accuracy, "roc_auc": roc_auc}


def main():
    """Main training script."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "joseph_training"
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    # Check if data exists
    if not data_dir.exists():
        logger.error(f"Training data not found at {data_dir}")
        logger.error("Please run 'python scripts/prepare_features.py' first")
        return

    # Load data
    logger.info("Loading training data...")
    train_df, val_df = load_data(data_dir)

    # Prepare features
    logger.info("Preparing features and labels...")
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)

    # Create data loaders
    train_loader = create_dataloader(X_train, y_train, batch_size=64, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=64, shuffle=False)

    # Initialize GAN detector
    logger.info("Initializing GAN detector...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = GANDetector(
        feature_dim=10,  # 10 lightweight entropy features
        latent_dim=100,
        hidden_dim=256,  # Increased from 128 for more capacity
        lr_generator=0.0002,
        lr_discriminator=0.0001,
    )

    # Train model
    logger.info(f"\nTraining on device: {device}")
    history = train_gan_detector(
        detector, train_loader, val_loader, epochs=200, device=device, checkpoint_dir=models_dir
    )

    # Save final model
    final_path = models_dir / "gan_detector_v1.pt"
    detector.save(str(final_path))
    logger.info(f"\nSaved final model to {final_path}")

    # Final evaluation
    metrics = evaluate_model(detector, val_loader, device=device)

    # Save training history and metrics
    history_path = models_dir / "gan_training_history.json"
    with open(history_path, "w") as f:
        json.dump(
            {
                "history": {
                    "train_loss": [float(x) for x in history["train_loss"]],
                    "val_acc": [float(x) for x in history["val_acc"]],
                    "val_auc": [float(x) for x in history["val_auc"]],
                },
                "final_metrics": metrics,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved training history to {history_path}")

    logger.info("\n" + "=" * 50)
    logger.info("Training complete!")
    logger.info(f"Best model saved to: {models_dir / 'gan_detector_best.pt'}")
    logger.info(f"Final model saved to: {final_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
