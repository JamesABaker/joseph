"""
Lightweight AI text detection using GAN model on entropy-based features.
No heavy transformer models required - runs on minimal memory.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from app.entropy_detector import EntropyDetector
from app.gan_model import GANDetector

logger = logging.getLogger(__name__)


class AIDetector:
    """Lightweight AI detector using trained GAN model on 6 entropy features."""

    def __init__(self):
        """Initialize the AI detector (lightweight, ~50MB memory)."""
        logger.info("Initializing entropy detector...")
        self.entropy_detector = EntropyDetector()
        logger.info("Entropy detector ready")

        # Load trained GAN detector model (REQUIRED)
        gan_model_path = Path(__file__).parent.parent / "models" / "gan_detector_v1.pt"
        if not gan_model_path.exists():
            raise FileNotFoundError(
                f"GAN model not found at {gan_model_path}. "
                "Please run scripts/train_gan_model.py to train the model."
            )

        logger.info(f"Loading trained GAN model from {gan_model_path}")
        self.gan_detector = GANDetector(feature_dim=10)
        self.gan_detector.load(str(gan_model_path))
        logger.info("GAN model loaded successfully")

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detect AI-generated text using GAN model on entropy features.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with probabilities, entropy metrics, and prediction
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Get entropy-based analysis (10 features)
        entropy_results = self.entropy_detector.detect(text)

        # Prepare features for GAN model (10 features)
        features = np.array(
            [
                [
                    entropy_results["shannon_entropy"],
                    entropy_results["burstiness"],
                    entropy_results["lexical_diversity"],
                    entropy_results["word_length_variance"],
                    entropy_results["punctuation_diversity"],
                    entropy_results["vocabulary_richness"],
                    entropy_results["avg_sentence_length"],
                    entropy_results["sentence_length_std"],
                    entropy_results["special_char_ratio"],
                    entropy_results["uppercase_ratio"],
                ]
            ]
        )

        # Get prediction from GAN model (convert to tensor)
        features_tensor = torch.tensor(features, dtype=torch.float32)
        gan_ai_prob = self.gan_detector.predict(features_tensor)[0].item() * 100
        gan_human_prob = 100 - gan_ai_prob
        prediction = "ai" if gan_ai_prob > 50 else "human"

        return {
            # GAN model final scores
            "human_probability": round(gan_human_prob, 2),
            "ai_probability": round(gan_ai_prob, 2),
            "prediction": prediction,
            "text_length": len(text),
            # ML model scores (same as GAN in lightweight mode)
            "ml_human_probability": round(gan_human_prob, 2),
            "ml_ai_probability": round(gan_ai_prob, 2),
            # Perplexity (not computed in lightweight mode - no GPT-2)
            "perplexity": 0.0,
            # Entropy metrics
            "shannon_entropy": entropy_results["shannon_entropy"],
            "burstiness": entropy_results["burstiness"],
            "lexical_diversity": entropy_results["lexical_diversity"],
            "word_length_variance": entropy_results["word_length_variance"],
            "punctuation_diversity": entropy_results["punctuation_diversity"],
            "vocabulary_richness": entropy_results["vocabulary_richness"],
            "avg_sentence_length": entropy_results["avg_sentence_length"],
            "sentence_length_std": entropy_results["sentence_length_std"],
            "special_char_ratio": entropy_results["special_char_ratio"],
            "uppercase_ratio": entropy_results["uppercase_ratio"],
            # Individual entropy-based probability
            "entropy_ai_probability": entropy_results["ai_probability_entropy"],
            "entropy_human_probability": entropy_results["human_probability_entropy"],
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the detector."""
        info: Dict[str, Any] = {
            "model_name": "GAN Detector (Lightweight)",
            "architecture": "Generative Adversarial Network on 10 entropy features",
            "memory_usage": "~50MB",
            "entropy_features": [
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
            ],
            "labels": {"0": "Human-written", "1": "AI-generated"},
        }
        return info
