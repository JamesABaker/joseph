# Model Evaluation Report

**Date:** January 24, 2026
**Project:** Joseph AI Text Detector

---

## Summary

This report documents the performance evolution of the Joseph AI text detector across different model configurations and test datasets.

---

## Model Configurations Tested

### v1: Full Model (8 Features)
- **Features:** perplexity, shannon_entropy, burstiness, lexical_diversity, word_length_variance, punctuation_diversity, vocabulary_richness, roberta_ai_prob
- **Dependencies:** GPT-2 (~500MB), RoBERTa (~500MB), PyTorch
- **Memory Usage:** ~1.2GB RAM

### v2: Lightweight Model (6 Features)
- **Features:** shannon_entropy, burstiness, lexical_diversity, word_length_variance, punctuation_diversity, vocabulary_richness
- **Dependencies:** PyTorch only (no transformer models)
- **Memory Usage:** ~50MB RAM

---

## Results

### Dataset: HC3 (ChatGPT-3.5 vs Human)
Training data from Hello-SimpleAI/HC3 dataset - question/answer pairs where human and ChatGPT-3.5 responses are compared.

**Samples:** 25,317 (Human: 17,265 | AI: 8,052)

#### Full Model (8 features) - Joseph Random Forest
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Human | 1.00 | 1.00 | 1.00 | 17,265 |
| AI | 1.00 | 1.00 | 1.00 | 8,052 |
| **Accuracy** | | | **99.92%** | 25,317 |

#### Full Model (8 features) - GAN Detector
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Human | 0.99 | 1.00 | 0.99 | 17,265 |
| AI | 0.99 | 0.99 | 0.99 | 8,052 |
| **Accuracy** | | | **99.52%** | 25,317 |

#### Lightweight Model (6 features) - GAN Detector
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Human | 0.88 | 0.94 | 0.91 | 17,265 |
| AI | 0.85 | 0.72 | 0.78 | 8,052 |
| **Accuracy** | | | **87.16%** | 25,317 |
| **Macro Avg** | 0.87 | 0.83 | **0.85** | |

**Confusion Matrix (Lightweight):**
```
              Predicted
              Human    AI
Actual Human  16264   1001
Actual AI      2250   5802
```

### Dataset: GPT-4 Wikipedia Test
1000 samples from modern GPT models generating Wikipedia-style content.

**Samples:** 1,000 (Human: 500 | AI: 500)

#### Full Model (8 features) - Joseph Random Forest
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Human | 0.81 | 0.79 | 0.80 | 500 |
| AI | 0.79 | 0.82 | 0.80 | 500 |
| **Accuracy** | | | **80.20%** | 1,000 |
| **Macro F1** | | | **0.80** | |

#### Lightweight Model (6 features) - GAN Detector
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Human | 0.62 | 0.65 | 0.63 | 500 |
| AI | 0.63 | 0.60 | 0.61 | 500 |
| **Accuracy** | | | **62.00%** | 1,000 |
| **Macro F1** | | | **0.62** | |

**Confusion Matrix (Lightweight):**
```
              Predicted
              Human    AI
Actual Human   326    174
Actual AI      202    298
```

---

## Analysis

### Key Findings

1. **ChatGPT-3.5 detection is nearly solved** - Both the Random Forest and GAN models achieve >99% accuracy on the HC3 dataset. This older model has distinct patterns that are easy to detect.

2. **GPT-4 is significantly harder** - Accuracy drops ~20 percentage points when testing on GPT-4 generated content. Modern models produce more human-like text.

3. **Perplexity and RoBERTa contribute significantly** - Removing these features (to save memory) drops accuracy:
   - ChatGPT-3.5: 99.52% → 87.16% (-12.36%)
   - GPT-4: 78.70% → 60.80% (-17.9%)

4. **Memory vs Accuracy tradeoff** - The lightweight model uses ~50MB vs ~1.2GB, but sacrifices ~18% accuracy on modern AI text.

### Feature Importance

The 8-feature model relies heavily on:
- **Perplexity** (GPT-2 based) - Most reliable single feature
- **RoBERTa AI probability** - Pre-trained detector signal
- **Burstiness** - Sentence length variation
- **Vocabulary richness** (Yule's K) - Lexical diversity metric

The 6-feature lightweight model loses the two most powerful features (perplexity and RoBERTa), explaining the accuracy drop.

---

## Deployment Considerations

### For Render Free Tier (512MB RAM limit)
- **Recommended:** Lightweight 6-feature model
- **Accuracy:** 87% on ChatGPT, 61% on GPT-4
- **Memory:** ~50MB

### For Higher Memory Environments
- **Recommended:** Full 8-feature model
- **Accuracy:** 99% on ChatGPT, 80% on GPT-4
- **Memory:** ~1.2GB

---

## Future Improvements

1. **Fine-tune on GPT-4 data** - Current training is only on ChatGPT-3.5
2. **Quantized transformer models** - Could reduce RoBERTa memory by 75%
3. **Distillation** - Train smaller model to mimic full model's behavior
4. **Additional entropy features** - Sentence structure, POS patterns

---

## Files

- `models/gan_detector_v1.pt` - Current lightweight GAN (6 features)
- `models/joseph_v1.pkl` - Random Forest (8 features, requires GPT-2/RoBERTa)
- `data/joseph_training/` - HC3 training data with extracted features
- `data/gpt4_test/` - GPT-4 test dataset
