"""
Quick hyperparameter tuning for GAN-based AI text detector.

This script performs a focused grid search over key hyperparameters.
Reduced search space for faster iteration.
"""

import json
import logging
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # noqa: E402

from app.gan_model import GANDetector  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data(data_dir):
    """Load train and validation parquet files."""
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    return train_df, val_df


def prepare_features(df):
    """Separate features and labels, convert to tensors."""
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


def train_and_evaluate(detector, train_loader, val_loader, epochs, device):
    """
    Train model and return validation metrics.

    Returns:
        dict: Validation accuracy, F1, and AUC
    """
    detector.to_device(device)
    detector.train()

    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Train discriminator more frequently (5:1 ratio)
            detector.train_step(batch_X, batch_y, train_generator=(epoch % 5 == 0))

    # Evaluate on validation set
    detector.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            probs = detector.predict(batch_X)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    return {"accuracy": accuracy, "f1": f1, "auc": auc}


def main():
    """Run hyperparameter tuning with focused grid search."""
    logger.info("Starting quick hyperparameter tuning")

    # Load data
    data_dir = Path(__file__).parent.parent / "data" / "joseph_training"
    train_df, val_df = load_data(data_dir)

    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)

    train_loader = create_dataloader(X_train, y_train, batch_size=64, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=64, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Focused hyperparameter search space around current best (256)
    param_grid = {
        "hidden_dim": [256, 384, 512],  # Network capacity (around current 256)
        "latent_dim": [100, 128],  # Generator noise dimension
        "lr_generator": [0.0002, 0.0003],  # Generator learning rate
        "lr_discriminator": [0.0001, 0.00015],  # Discriminator learning rate
        "epochs": [200, 250],  # Training duration
    }

    # Calculate total combinations
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)

    logger.info(f"Testing {total_combinations} hyperparameter combinations")
    logger.info(f"Parameter grid: {param_grid}")

    # Store results
    results = []
    best_f1 = 0.0
    best_params = None

    # Grid search
    for i, (hidden_dim, latent_dim, lr_g, lr_d, epochs) in enumerate(
        product(
            param_grid["hidden_dim"],
            param_grid["latent_dim"],
            param_grid["lr_generator"],
            param_grid["lr_discriminator"],
            param_grid["epochs"],
        ),
        1,
    ):
        logger.info(
            f"\n[{i}/{total_combinations}] Testing: "
            f"hidden_dim={hidden_dim}, latent_dim={latent_dim}, "
            f"lr_g={lr_g}, lr_d={lr_d}, epochs={epochs}"
        )

        # Initialize model with current hyperparameters
        detector = GANDetector(
            feature_dim=10,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            lr_generator=lr_g,
            lr_discriminator=lr_d,
        )

        # Train and evaluate
        metrics = train_and_evaluate(detector, train_loader, val_loader, epochs, device)

        logger.info(
            f"Results: Accuracy={metrics['accuracy']:.4f}, "
            f"F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}"
        )

        # Store result
        result = {
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "lr_generator": lr_g,
            "lr_discriminator": lr_d,
            "epochs": epochs,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "auc": metrics["auc"],
        }
        results.append(result)

        # Track best
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_params = result
            logger.info(f"✨ New best F1: {best_f1:.4f}")

            # Save best model
            models_dir = Path(__file__).parent.parent / "models"
            best_model_path = models_dir / "gan_detector_tuned_best.pt"
            detector.save(str(best_model_path))
            logger.info(f"Saved best model to {best_model_path}")

    # Save all results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("f1", ascending=False)

    results_path = Path(__file__).parent.parent / "models" / "hyperparameter_tuning_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nSaved all results to {results_path}")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("HYPERPARAMETER TUNING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nBest F1 Score: {best_f1:.4f}")
    logger.info("Best Parameters:")
    for key, value in best_params.items():
        logger.info(f"  {key}: {value}")

    logger.info("\nTop 5 configurations:")
    logger.info(results_df.head(5).to_string(index=False))

    # Save best params as JSON for easy loading
    best_params_path = Path(__file__).parent.parent / "models" / "best_hyperparameters.json"
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"\nSaved best hyperparameters to {best_params_path}")


if __name__ == "__main__":
    main()
