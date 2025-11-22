"""
Entropy and information theory-based AI text detection.
Implements perplexity, Shannon entropy, and burstiness analysis.
"""
import logging
import math
from typing import Any, Dict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class EntropyDetector:
    """Entropy-based AI text detector using information theory metrics."""

    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize the entropy detector with a language model for perplexity.

        Args:
            model_name: Hugging Face model for perplexity calculation (default: gpt2)
        """
        logger.info(f"Loading entropy detector with model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        logger.info("Entropy detector loaded successfully")

    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity of text using the language model.
        Lower perplexity suggests more predictable (AI-like) text.

        Args:
            text: Input text to analyze

        Returns:
            Perplexity score
        """
        encodings = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        max_length = self.model.config.n_positions
        stride = 512
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        perplexity = torch.exp(torch.stack(nlls).mean()).item()
        return perplexity

    def calculate_shannon_entropy(self, text: str) -> float:
        """
        Calculate Shannon entropy of the text at character level.

        Args:
            text: Input text to analyze

        Returns:
            Shannon entropy value
        """
        if not text:
            return 0.0

        # Count character frequencies
        char_freq = {}
        for char in text:
            char_freq[char] = char_freq.get(char, 0) + 1

        # Calculate entropy
        entropy = 0.0
        text_len = len(text)
        for freq in char_freq.values():
            probability = freq / text_len
            if probability > 0:
                entropy -= probability * math.log2(probability)

        return entropy

    def calculate_burstiness(self, text: str) -> float:
        """
        Calculate burstiness - variance in sentence complexity.
        Human writing tends to have higher burstiness (varied sentence complexity).

        Args:
            text: Input text to analyze

        Returns:
            Burstiness score (0-1, higher = more human-like variation)
        """
        # Split into sentences (simple approach)
        sentences = [
            s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()
        ]

        if len(sentences) < 2:
            return 0.0

        # Calculate sentence lengths
        lengths = [len(s.split()) for s in sentences]

        # Calculate coefficient of variation (std/mean)
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)

        if mean_length == 0:
            return 0.0

        burstiness = std_length / mean_length
        # Normalize to 0-1 range (typical values are 0-2)
        return min(burstiness / 2.0, 1.0)

    def calculate_lexical_diversity(self, text: str) -> float:
        """
        Calculate lexical diversity (type-token ratio).
        Higher diversity often indicates human writing.

        Args:
            text: Input text to analyze

        Returns:
            Lexical diversity score (0-1)
        """
        words = text.lower().split()
        if not words:
            return 0.0

        unique_words = set(words)
        return len(unique_words) / len(words)

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive entropy-based detection.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with entropy metrics and AI probability
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Calculate all metrics
        perplexity = self.calculate_perplexity(text)
        shannon_entropy = self.calculate_shannon_entropy(text)
        burstiness = self.calculate_burstiness(text)
        lexical_diversity = self.calculate_lexical_diversity(text)

        # Heuristic scoring (these thresholds are approximate and should be tuned)
        # Lower perplexity = more AI-like (typical AI: 20-50, human: 50-200)
        perplexity_score = 1.0 / (1.0 + math.exp((perplexity - 50) / 20))

        # Lower entropy = more AI-like
        entropy_score = 1.0 - min(shannon_entropy / 5.0, 1.0)

        # Lower burstiness = more AI-like
        burstiness_score = 1.0 - burstiness

        # Lower diversity = more AI-like
        diversity_score = 1.0 - lexical_diversity

        # Weighted average (can be tuned)
        ai_probability = (
            0.4 * perplexity_score
            + 0.2 * entropy_score
            + 0.25 * burstiness_score
            + 0.15 * diversity_score
        )

        return {
            "perplexity": round(perplexity, 2),
            "shannon_entropy": round(shannon_entropy, 3),
            "burstiness": round(burstiness, 3),
            "lexical_diversity": round(lexical_diversity, 3),
            "ai_probability_entropy": round(ai_probability * 100, 2),
            "human_probability_entropy": round((1 - ai_probability) * 100, 2),
        }
