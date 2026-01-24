"""Tests for entropy detection functionality."""

import pytest

from app.entropy_detector import EntropyDetector


@pytest.fixture(scope="module")
def detector():
    """Create entropy detector instance."""
    return EntropyDetector()


def test_entropy_detector_init(detector):
    """Test entropy detector initialization."""
    assert detector is not None
    assert detector.tokenizer is not None
    assert detector.model is not None


def test_calculate_perplexity(detector, sample_texts):
    """Test perplexity calculation."""
    perplexity = detector.calculate_perplexity(sample_texts["human"])

    assert perplexity > 0
    assert isinstance(perplexity, float)


def test_calculate_shannon_entropy(detector, sample_texts):
    """Test Shannon entropy calculation."""
    entropy = detector.calculate_shannon_entropy(sample_texts["human"])

    assert entropy >= 0
    assert isinstance(entropy, float)


def test_calculate_burstiness(detector, sample_texts):
    """Test burstiness calculation."""
    burstiness = detector.calculate_burstiness(sample_texts["human"])

    assert 0 <= burstiness <= 1
    assert isinstance(burstiness, float)


def test_calculate_lexical_diversity(detector, sample_texts):
    """Test lexical diversity calculation."""
    diversity = detector.calculate_lexical_diversity(sample_texts["human"])

    assert 0 <= diversity <= 1
    assert isinstance(diversity, float)


def test_calculate_word_length_variance(detector, sample_texts):
    """Test word length variance calculation."""
    variance = detector.calculate_word_length_variance(sample_texts["human"])

    assert 0 <= variance <= 1
    assert isinstance(variance, float)


def test_calculate_punctuation_diversity(detector, sample_texts):
    """Test punctuation diversity calculation."""
    diversity = detector.calculate_punctuation_diversity(sample_texts["human"])

    assert 0 <= diversity <= 1
    assert isinstance(diversity, float)


def test_calculate_vocabulary_richness(detector, sample_texts):
    """Test vocabulary richness calculation."""
    richness = detector.calculate_vocabulary_richness(sample_texts["human"])

    assert 0 <= richness <= 1
    assert isinstance(richness, float)


def test_detect_full_analysis(detector, sample_texts):
    """Test full detection analysis."""
    result = detector.detect(sample_texts["human"])

    # Check all expected keys are present
    assert "perplexity" in result
    assert "shannon_entropy" in result
    assert "burstiness" in result
    assert "lexical_diversity" in result
    assert "word_length_variance" in result
    assert "punctuation_diversity" in result
    assert "vocabulary_richness" in result
    assert "ai_probability_entropy" in result
    assert "human_probability_entropy" in result

    # Check probability sums to 100
    assert abs(result["ai_probability_entropy"] + result["human_probability_entropy"] - 100) < 0.1

    # Check ranges
    assert result["perplexity"] > 0
    assert result["shannon_entropy"] >= 0
    assert 0 <= result["burstiness"] <= 1
    assert 0 <= result["ai_probability_entropy"] <= 100
    assert 0 <= result["human_probability_entropy"] <= 100


def test_detect_empty_text(detector):
    """Test detection with empty text."""
    with pytest.raises(ValueError):
        detector.detect("")


def test_detect_whitespace_only(detector):
    """Test detection with whitespace only."""
    with pytest.raises(ValueError):
        detector.detect("   ")


def test_detect_short_text(detector, sample_texts):
    """Test detection with very short text."""
    result = detector.detect(sample_texts["short"])

    assert result is not None
    assert "perplexity" in result
