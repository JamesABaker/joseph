"""
BERT-based AI text detection model wrapper.
Uses Open-Detector model from Hugging Face.
"""
import logging
from typing import Any, Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class AIDetector:
    """Wrapper for the BERT AI detection model."""

    def __init__(self, model_name: str = "Hello-SimpleAI/chatgpt-detector-roberta"):
        """
        Initialize the AI detector model.

        Args:
            model_name: Hugging Face model identifier
        """
        logger.info(f"Loading model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        logger.info("Model loaded successfully")

    def detect(self, text: str, max_length: int = 512) -> Dict[str, Any]:
        """
        Detect if text is AI-generated or human-written.

        Args:
            text: Input text to analyze
            max_length: Maximum token length (default 512)

        Returns:
            Dictionary with probabilities:
            {
                'human_probability': float (0-100),
                'ai_probability': float (0-100),
                'prediction': str ('human' or 'ai')
            }
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Tokenize input
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length, padding=True
        )

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Extract probabilities (label 0 = human, label 1 = AI)
            human_prob = probs[0][0].item() * 100
            ai_prob = probs[0][1].item() * 100

        prediction = "ai" if ai_prob > 50 else "human"

        return {
            "human_probability": round(human_prob, 2),
            "ai_probability": round(ai_prob, 2),
            "prediction": prediction,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info: Dict[str, Any] = {
            "model_name": self.model_name,
            "architecture": "RoBERTa-base",
            "max_length": 512,
            "labels": {"0": "Human-written", "1": "AI-generated"},
        }
        return info
