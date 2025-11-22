"""
Hybrid AI text detection combining ML model with entropy-based features.
Uses RoBERTa classifier + information theory metrics for robust detection.
"""
import logging
from typing import Any, Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.entropy_detector import EntropyDetector

logger = logging.getLogger(__name__)


class AIDetector:
    """Hybrid AI detector combining ML model with entropy analysis."""

    def __init__(self, model_name: str = "Hello-SimpleAI/chatgpt-detector-roberta"):
        """
        Initialize the hybrid AI detector.

        Args:
            model_name: Hugging Face model identifier for ML classifier
        """
        logger.info(f"Loading ML model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        logger.info("ML model loaded successfully")

        logger.info("Initializing entropy detector...")
        self.entropy_detector = EntropyDetector()
        logger.info("Hybrid detector ready")

    def detect(self, text: str, max_length: int = 512) -> Dict[str, Any]:
        """
        Hybrid detection combining ML model and entropy analysis.

        Args:
            text: Input text to analyze
            max_length: Maximum token length (default 512)

        Returns:
            Dictionary with ML probabilities, entropy metrics, and hybrid score
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Get ML model predictions
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length, padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            ml_human_prob = probs[0][0].item() * 100
            ml_ai_prob = probs[0][1].item() * 100

        # Get entropy-based analysis
        entropy_results = self.entropy_detector.detect(text)

        # Hybrid score: weighted combination of ML and entropy
        # ML model gets 60% weight, entropy features get 40%
        hybrid_ai_prob = 0.6 * ml_ai_prob + 0.4 * entropy_results["ai_probability_entropy"]
        hybrid_human_prob = 100 - hybrid_ai_prob

        prediction = "ai" if hybrid_ai_prob > 50 else "human"

        return {
            # Hybrid final scores
            "human_probability": round(hybrid_human_prob, 2),
            "ai_probability": round(hybrid_ai_prob, 2),
            "prediction": prediction,
            # ML model scores
            "ml_human_probability": round(ml_human_prob, 2),
            "ml_ai_probability": round(ml_ai_prob, 2),
            # Entropy metrics
            "perplexity": entropy_results["perplexity"],
            "shannon_entropy": entropy_results["shannon_entropy"],
            "burstiness": entropy_results["burstiness"],
            "lexical_diversity": entropy_results["lexical_diversity"],
            # Individual entropy-based probability
            "entropy_ai_probability": entropy_results["ai_probability_entropy"],
            "entropy_human_probability": entropy_results["human_probability_entropy"],
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the hybrid detector."""
        info: Dict[str, Any] = {
            "model_name": self.model_name,
            "architecture": "Hybrid: RoBERTa + Entropy Analysis",
            "ml_model": "RoBERTa-base",
            "entropy_features": [
                "perplexity",
                "shannon_entropy",
                "burstiness",
                "lexical_diversity",
            ],
            "max_length": 512,
            "labels": {"0": "Human-written", "1": "AI-generated"},
        }
        return info
