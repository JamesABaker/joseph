"""
Hybrid AI text detection combining ML model with entropy-based features.
Uses trained Joseph Random Forest model on entropy features + RoBERTa.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.entropy_detector import EntropyDetector

logger = logging.getLogger(__name__)


class AIDetector:
    """Hybrid AI detector using trained Joseph Random Forest model."""

    def __init__(self, model_name: str = "Hello-SimpleAI/chatgpt-detector-roberta"):
        """
        Initialize the hybrid AI detector.

        Args:
            model_name: Hugging Face model identifier for RoBERTa classifier
        """
        logger.info(f"Loading RoBERTa model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)  # nosec B615
        self.roberta_model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )  # nosec B615
        self.roberta_model.eval()
        logger.info("RoBERTa model loaded successfully")

        logger.info("Initializing entropy detector...")
        self.entropy_detector = EntropyDetector()
        logger.info("Entropy detector ready")

        # Load trained Joseph Random Forest model (REQUIRED)
        joseph_model_path = Path(__file__).parent.parent / "models" / "joseph_v1.pkl"
        if not joseph_model_path.exists():
            raise FileNotFoundError(
                f"Joseph model not found at {joseph_model_path}. "
                "Please run scripts/prepare_features.py and "
                "scripts/train_joseph_model.py to train the model."
            )

        logger.info(f"Loading trained Joseph model from {joseph_model_path}")
        self.joseph_model = joblib.load(joseph_model_path)
        logger.info("Joseph model loaded successfully")

    def detect(self, text: str, max_length: int = 512) -> Dict[str, Any]:
        """
        Hybrid detection using trained Joseph model on entropy + RoBERTa features.

        Args:
            text: Input text to analyze
            max_length: Maximum token length (default 512)

        Returns:
            Dictionary with probabilities, entropy metrics, and prediction
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Get RoBERTa predictions
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length, padding=True
        )

        with torch.no_grad():
            outputs = self.roberta_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            ml_human_prob = probs[0][0].item() * 100
            ml_ai_prob = probs[0][1].item() * 100

        # Get entropy-based analysis
        entropy_results = self.entropy_detector.detect(text)

        # Prepare features for Joseph model (8 features)
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

        # Get prediction from Joseph model
        joseph_ai_prob = self.joseph_model.predict_proba(features)[0][1] * 100
        joseph_human_prob = 100 - joseph_ai_prob
        prediction = "ai" if joseph_ai_prob > 50 else "human"

        return {
            # Joseph model final scores
            "human_probability": round(joseph_human_prob, 2),
            "ai_probability": round(joseph_ai_prob, 2),
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
            "model_name": "Joseph Random Forest",
            "architecture": "Random Forest on 8 features (7 entropy + RoBERTa)",
            "roberta_model": self.model_name,
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
