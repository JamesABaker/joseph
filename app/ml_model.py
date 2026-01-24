"""
Hybrid AI text detection combining ML model with entropy-based features.
Uses trained GAN detector model on entropy features + RoBERTa.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.entropy_detector import EntropyDetector
from app.gan_model import GANDetector

logger = logging.getLogger(__name__)


class AIDetector:
    """Hybrid AI detector using trained GAN model."""

    def __init__(self, load_roberta: bool = False):
        """
        Initialize the hybrid AI detector.

        Args:
            load_roberta: Whether to load RoBERTa model (uses ~500MB memory)
        """
        self.load_roberta = load_roberta

        if load_roberta:
            logger.info("Loading RoBERTa model: Hello-SimpleAI/chatgpt-detector-roberta")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Hello-SimpleAI/chatgpt-detector-roberta"
            )  # nosec B615
            self.roberta_model = AutoModelForSequenceClassification.from_pretrained(
                "Hello-SimpleAI/chatgpt-detector-roberta"
            )  # nosec B615
            self.roberta_model.eval()
            logger.info("RoBERTa model loaded successfully")
        else:
            logger.info("Skipping RoBERTa model (memory optimization)")
            self.tokenizer = None
            self.roberta_model = None

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
        self.gan_detector = GANDetector(feature_dim=8)
        self.gan_detector.load(str(gan_model_path))
        logger.info("GAN model loaded successfully")

    def detect(self, text: str, max_length: int = 512) -> Dict[str, Any]:
        """
        Hybrid detection using trained GAN model on entropy + RoBERTa features.

        Args:
            text: Input text to analyze
            max_length: Maximum token length (default 512)

        Returns:
            Dictionary with probabilities, entropy metrics, and prediction
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Get RoBERTa predictions (if loaded)
        if self.load_roberta:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length, padding=True
            )

            with torch.no_grad():
                outputs = self.roberta_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                ml_human_prob = probs[0][0].item() * 100
                ml_ai_prob = probs[0][1].item() * 100
        else:
            # Use entropy-based heuristic as RoBERTa substitute
            ml_human_prob = 50.0
            ml_ai_prob = 50.0

        # Get entropy-based analysis
        entropy_results = self.entropy_detector.detect(text)

        # Prepare features for GAN model (8 features)
        features = np.array(
            [
                [
                    entropy_results["perplexity"],
                    entropy_results["shannon_entropy"],
                    entropy_results["burstiness"],
                    entropy_results["lexical_diversity"],
                    entropy_results["word_length_variance"],
                    entropy_results["punctuation_diversity"],
                    entropy_results["vocabulary_richness"],
                    ml_ai_prob,  # RoBERTa as 8th feature
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
            # RoBERTa scores
            "ml_human_probability": round(ml_human_prob, 2),
            "ml_ai_probability": round(ml_ai_prob, 2),
            # Entropy metrics
            "perplexity": entropy_results["perplexity"],
            "shannon_entropy": entropy_results["shannon_entropy"],
            "burstiness": entropy_results["burstiness"],
            "lexical_diversity": entropy_results["lexical_diversity"],
            "word_length_variance": entropy_results["word_length_variance"],
            "punctuation_diversity": entropy_results["punctuation_diversity"],
            "vocabulary_richness": entropy_results["vocabulary_richness"],
            # Individual entropy-based probability
            "entropy_ai_probability": entropy_results["ai_probability_entropy"],
            "entropy_human_probability": entropy_results["human_probability_entropy"],
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the hybrid detector."""
        info: Dict[str, Any] = {
            "model_name": "GAN Detector",
            "architecture": "Generative Adversarial Network on 8 features (7 entropy + RoBERTa)",
            "roberta_loaded": self.load_roberta,
            "entropy_features": [
                "perplexity",
                "shannon_entropy",
                "burstiness",
                "lexical_diversity",
                "word_length_variance",
                "punctuation_diversity",
                "vocabulary_richness",
            ],
            "max_length": 512,
            "labels": {"0": "Human-written", "1": "AI-generated"},
        }
        return info
