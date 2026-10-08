import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(
    r"[a-z]+(?:'[a-z]+)?|[0-9]+(?:\.[0-9]+)?|[^\w\s]",
    re.IGNORECASE,
)

MIN_TEXT_LENGTH = 50
RANDOM_SEED = 42
TARGET_COVERAGE = 0.98

START_TOKEN = "<START>"
END_TOKEN = "<END>"
RARE_TOKEN = "<RARE>"
NUM_TOKEN = "<NUM>"
URL_TOKEN = "<URL>"

EXPECTED_ORDER = [
    "Human_Story",
    "Gemma-2-9B",
    "Mistral-7B",
    "Qwen-2-72B",
    "Llama-8B",
    "Yi-Large",
    "GPT-4o",
]

TARGET_AI_LABEL = "GPT-4o"


def normalise_unicode_punctuation(text: str) -> str:
    return (
        text.replace("’", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("–", "-")
            .replace("—", "-")
            .replace("…", "...")
    )


def normalise_token(token: str) -> str:
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", token):
        return NUM_TOKEN
    if token == "url":
        return URL_TOKEN
    return token


def tokenise(text: str) -> list[str]:
    if not text:
        return []

    text = text.replace("\\n", " ")
    text = normalise_unicode_punctuation(text)
    text = re.sub(r"\s+", " ", text.lower().strip())
    raw_tokens = TOKEN_PATTERN.findall(text)
    tokens = [normalise_token(tok) for tok in raw_tokens]

    return [START_TOKEN, *tokens, END_TOKEN] if tokens else []


def is_valid_text(text, min_length: int = MIN_TEXT_LENGTH) -> bool:
    if not isinstance(text, str):
        return False
    return len(text.strip()) > min_length

def map_token(token: str, vocab: set[str]) -> str:
    return token if token in vocab else RARE_TOKEN


def map_sequence(tokens: list[str], vocab: set[str]) -> list[str]:
    return [map_token(token, vocab) for token in tokens]


def build_row_totals(transition_counter: Counter[tuple[str, str]]) -> dict[str, int]:
    row_totals = defaultdict(int)
    for (prev_token, next_token), count in transition_counter.items():
        row_totals[prev_token] += count
    return dict(row_totals)


def smoothed_transition_probability(
    prev_token: str,
    next_token: str,
    transition_counter: Counter[tuple[str, str]],
    row_totals: dict[str, int],
    vocab: set[str],
    alpha: float = 1.0,
) -> float:
    vocab_size = len(vocab)
    count = transition_counter.get((prev_token, next_token), 0)
    row_total = row_totals.get(prev_token, 0)
    return (count + alpha) / (row_total + alpha * vocab_size)


def sequence_log_likelihood(
    tokens: list[str],
    transition_counter: Counter[tuple[str, str]],
    row_totals: dict[str, int],
    vocab: set[str],
    alpha: float = 1.0,
    normalize_by_length: bool = True,
) -> float:
    if len(tokens) < 2:
        return 0.0

    log_prob = 0.0
    for prev_token, next_token in zip(tokens[:-1], tokens[1:]):
        p = smoothed_transition_probability(
            prev_token=prev_token,
            next_token=next_token,
            transition_counter=transition_counter,
            row_totals=row_totals,
            vocab=vocab,
            alpha=alpha,
        )
        log_prob += math.log(p)

    if normalize_by_length:
        return log_prob / (len(tokens) - 1)

    return log_prob


def inspect_dataset(df: pd.DataFrame, n: int = 10) -> None:
    logger.info("Columns: %s", list(df.columns))
    logger.info("Head:\n%s", df.head(n))
    logger.info("Label_B value counts:\n%s", df["Label_B"].value_counts())


def validate_block_structure(df: pd.DataFrame) -> None:
    block_size = len(EXPECTED_ORDER)
    n = len(df)

    if n % block_size != 0:
        raise ValueError(f"Dataset length {n} is not divisible by block size {block_size}")

    num_blocks = n // block_size
    bad_blocks = []

    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = start + block_size
        labels = df.iloc[start:end]["Label_B"].tolist()

        if labels != EXPECTED_ORDER:
            bad_blocks.append((block_idx, labels))
            if len(bad_blocks) >= 5:
                break

    if bad_blocks:
        raise ValueError(f"Block structure validation failed. Example bad blocks: {bad_blocks}")

    logger.info("Validated %d source blocks with expected label ordering.", num_blocks)


def build_grouped_pairs(df: pd.DataFrame, target_ai_label: str = TARGET_AI_LABEL) -> pd.DataFrame:
    block_size = len(EXPECTED_ORDER)
    num_blocks = len(df) // block_size

    rows = []

    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = start + block_size
        block = df.iloc[start:end].reset_index(drop=True)

        labels = block["Label_B"].tolist()
        if labels != EXPECTED_ORDER:
            raise ValueError(f"Unexpected labels in block {block_idx}: {labels}")

        label_to_text = dict(zip(block["Label_B"], block["Text"]))

        rows.append(
            {
                "group_id": block_idx,
                "human_text": label_to_text.get("Human_Story"),
                "ai_text": label_to_text.get(target_ai_label),
            }
        )

    paired_df = pd.DataFrame(rows)

    # Drop rows with missing or invalid text
    paired_df = paired_df[
        paired_df["human_text"].apply(lambda x: isinstance(x, str))
        & paired_df["ai_text"].apply(lambda x: isinstance(x, str))
    ].copy()

    logger.info("Constructed %d grouped Human-vs-%s pairs after dropping missing texts.", len(paired_df), target_ai_label)
    return paired_df

def build_vocab_from_training_pairs(
    paired_df: pd.DataFrame,
    train_indices: np.ndarray,
    target_coverage: float = TARGET_COVERAGE,
    min_length: int = MIN_TEXT_LENGTH,
) -> tuple[set[str], Counter]:
    token_counter = Counter()

    for idx in train_indices:
        row = paired_df.iloc[idx]

        human_text = row["human_text"]
        if is_valid_text(human_text, min_length):
            token_counter.update(tokenise(human_text))

        ai_text = row["ai_text"]
        if is_valid_text(ai_text, min_length):
            token_counter.update(tokenise(ai_text))

    total = sum(token_counter.values())
    vocab = {START_TOKEN, END_TOKEN, RARE_TOKEN, NUM_TOKEN, URL_TOKEN}

    if total == 0:
        return vocab, token_counter

    cumulative = 0.0
    for token, count in token_counter.most_common():
        if token in vocab:
            continue
        if count == 1:
            break

        vocab.add(token)
        cumulative += count / total
        if cumulative >= target_coverage:
            break

    logger.info("Vocabulary size: %d", len(vocab))
    logger.info("Approx. achieved coverage: %.4f", cumulative)
    return vocab, token_counter


def count_transitions_from_training_pairs(
    paired_df: pd.DataFrame,
    train_indices: np.ndarray,
    vocab: set[str],
    min_length: int = MIN_TEXT_LENGTH,
) -> tuple[Counter, int, Counter, int]:
    human_transition_counter = Counter()
    ai_transition_counter = Counter()
    human_total = 0
    ai_total = 0

    for idx in train_indices:
        row = paired_df.iloc[idx]

        human_text = row["human_text"]
        if is_valid_text(human_text, min_length):
            human_tokens = map_sequence(tokenise(human_text), vocab)
            if len(human_tokens) >= 2:
                human_transition_counter.update(zip(human_tokens[:-1], human_tokens[1:]))
                human_total += len(human_tokens) - 1

        ai_text = row["ai_text"]
        if is_valid_text(ai_text, min_length):
            ai_tokens = map_sequence(tokenise(ai_text), vocab)
            if len(ai_tokens) >= 2:
                ai_transition_counter.update(zip(ai_tokens[:-1], ai_tokens[1:]))
                ai_total += len(ai_tokens) - 1

    return human_transition_counter, human_total, ai_transition_counter, ai_total


def evaluate_on_pair_indices(
    paired_df: pd.DataFrame,
    eval_indices: np.ndarray,
    vocab: set[str],
    human_transition_counts: Counter,
    human_row_totals: dict[str, int],
    ai_transition_counts: Counter,
    ai_row_totals: dict[str, int],
    alpha: float,
    min_length: int = MIN_TEXT_LENGTH,
):
    true_vals = []
    predictions = []
    scores = []
    human_margins = []
    ai_margins = []

    for idx in eval_indices:
        row = paired_df.iloc[idx]

        human_text = row["human_text"]
        if is_valid_text(human_text, min_length):
            tokens = map_sequence(tokenise(human_text), vocab)
            human_score = sequence_log_likelihood(
                tokens, human_transition_counts, human_row_totals, vocab, alpha=alpha
            )
            ai_score = sequence_log_likelihood(
                tokens, ai_transition_counts, ai_row_totals, vocab, alpha=alpha
            )
            margin = ai_score - human_score
            human_margins.append(margin)

            prediction = 1 if ai_score > human_score else 0
            true_vals.append(0)
            predictions.append(prediction)
            scores.append(margin)

        ai_text = row["ai_text"]
        if is_valid_text(ai_text, min_length):
            tokens = map_sequence(tokenise(ai_text), vocab)
            human_score = sequence_log_likelihood(
                tokens, human_transition_counts, human_row_totals, vocab, alpha=alpha
            )
            ai_score = sequence_log_likelihood(
                tokens, ai_transition_counts, ai_row_totals, vocab, alpha=alpha
            )
            margin = ai_score - human_score
            ai_margins.append(margin)

            prediction = 1 if ai_score > human_score else 0
            true_vals.append(1)
            predictions.append(prediction)
            scores.append(margin)

    logger.info("Human margin min/max: %.4f / %.4f", min(human_margins), max(human_margins))
    logger.info("AI margin min/max: %.4f / %.4f", min(ai_margins), max(ai_margins))

    return true_vals, predictions, scores


def optimise_alpha(
    paired_df: pd.DataFrame,
    validation_indices: np.ndarray,
    vocab: set[str],
    human_transition_counts: Counter,
    human_row_totals: dict[str, int],
    ai_transition_counts: Counter,
    ai_row_totals: dict[str, int],
    alpha_grid: list[float],
    min_length: int = MIN_TEXT_LENGTH,
    metric: str = "f1",
) -> float:
    best_alpha = None
    best_score = -1.0

    for alpha in alpha_grid:
        true_vals, predictions, _ = evaluate_on_pair_indices(
            paired_df,
            validation_indices,
            vocab,
            human_transition_counts,
            human_row_totals,
            ai_transition_counts,
            ai_row_totals,
            alpha=alpha,
            min_length=min_length,
        )

        if metric == "accuracy":
            score = accuracy_score(true_vals, predictions)
        elif metric == "precision":
            score = precision_score(true_vals, predictions)
        elif metric == "recall":
            score = recall_score(true_vals, predictions)
        else:
            score = f1_score(true_vals, predictions)

        if score > best_score:
            best_score = score
            best_alpha = alpha

    return best_alpha


def exact_overlap_audit(
    paired_df: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
):
    def canon(text: str) -> str:
        return " ".join(str(text).lower().split())

    train_human = {canon(paired_df.iloc[i]["human_text"]) for i in train_indices}
    train_ai = {canon(paired_df.iloc[i]["ai_text"]) for i in train_indices}
    test_human = {canon(paired_df.iloc[i]["human_text"]) for i in test_indices}
    test_ai = {canon(paired_df.iloc[i]["ai_text"]) for i in test_indices}

    logger.info("train_human ∩ test_human: %d", len(train_human & test_human))
    logger.info("train_ai ∩ test_ai: %d", len(train_ai & test_ai))
    logger.info("train_human ∩ test_ai: %d", len(train_human & test_ai))
    logger.info("train_ai ∩ test_human: %d", len(train_ai & test_human))


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    split_path = "hf://datasets/Rajarshi-Roy-research/Defactify_Text_Dataset/data/train-00000-of-00001.parquet"
    df = pd.read_parquet(split_path)

    inspect_dataset(df)
    validate_block_structure(df)

    paired_df = build_grouped_pairs(df, target_ai_label=TARGET_AI_LABEL)

    num_pairs = len(paired_df)
    logger.info("Number of grouped pairs: %d", num_pairs)

    pair_indices = np.arange(num_pairs)
    rng.shuffle(pair_indices)

    fold_size = num_pairs // 5
    alpha_grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]

    all_fold_metrics = []

    for fold in range(5):
        logger.info("=== Fold %d ===", fold)

        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < 4 else num_pairs
        test_indices = pair_indices[test_start:test_end]

        val_fold = (fold + 1) % 5
        val_start = val_fold * fold_size
        val_end = (val_fold + 1) * fold_size if val_fold < 4 else num_pairs
        val_indices = pair_indices[val_start:val_end]

        test_set = set(test_indices.tolist())
        val_set = set(val_indices.tolist())
        train_indices = np.array([idx for idx in pair_indices if idx not in test_set and idx not in val_set])

        logger.info("Train groups: %d", len(train_indices))
        logger.info("Val groups: %d", len(val_indices))
        logger.info("Test groups: %d", len(test_indices))

        exact_overlap_audit(paired_df, train_indices, test_indices)

        vocab, _ = build_vocab_from_training_pairs(
            paired_df,
            train_indices,
            target_coverage=TARGET_COVERAGE,
            min_length=MIN_TEXT_LENGTH,
        )

        human_transition_counts, human_total, ai_transition_counts, ai_total = count_transitions_from_training_pairs(
            paired_df,
            train_indices,
            vocab,
            min_length=MIN_TEXT_LENGTH,
        )

        human_row_totals = build_row_totals(human_transition_counts)
        ai_row_totals = build_row_totals(ai_transition_counts)

        alpha = optimise_alpha(
            paired_df,
            val_indices,
            vocab,
            human_transition_counts,
            human_row_totals,
            ai_transition_counts,
            ai_row_totals,
            alpha_grid=alpha_grid,
            min_length=MIN_TEXT_LENGTH,
            metric="f1",
        )

        logger.info("Selected alpha: %.6f", alpha)

        true_vals, predictions, scores = evaluate_on_pair_indices(
            paired_df,
            test_indices,
            vocab,
            human_transition_counts,
            human_row_totals,
            ai_transition_counts,
            ai_row_totals,
            alpha=alpha,
            min_length=MIN_TEXT_LENGTH,
        )

        accuracy = accuracy_score(true_vals, predictions)
        precision = precision_score(true_vals, predictions)
        recall = recall_score(true_vals, predictions)
        f1 = f1_score(true_vals, predictions)
        cm = confusion_matrix(true_vals, predictions)

        logger.info("Accuracy: %.4f", accuracy)
        logger.info("Precision: %.4f", precision)
        logger.info("Recall: %.4f", recall)
        logger.info("F1: %.4f", f1)
        logger.info("Confusion matrix:\n%s", cm)

        all_fold_metrics.append(
            {
                "fold": fold,
                "alpha": alpha,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    metrics_df = pd.DataFrame(all_fold_metrics)
    logger.info("Fold metrics summary:\n%s", metrics_df)
    logger.info("Mean metrics:\n%s", metrics_df[["accuracy", "precision", "recall", "f1"]].mean())
    logger.info("Std metrics:\n%s", metrics_df[["accuracy", "precision", "recall", "f1"]].std())


if __name__ == "__main__":
    main()