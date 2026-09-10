import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

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
ALPHA_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
N_FOLDS = 5

START_TOKEN = "<START>"
END_TOKEN = "<END>"
RARE_TOKEN = "<RARE>"
NUM_TOKEN = "<NUM>"
URL_TOKEN = "<URL>"


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


def compute_raid_metrics(true_vals, scores):
    auroc = roc_auc_score(true_vals, scores)
    fpr, tpr, _ = roc_curve(true_vals, scores)

    def tpr_at_target_fpr(target_fpr: float) -> float:
        valid_indices = [i for i, f in enumerate(fpr) if f <= target_fpr]
        if not valid_indices:
            return 0.0
        return max(tpr[i] for i in valid_indices)

    return {
        "auroc": auroc,
        "tpr_at_fpr_5pct": tpr_at_target_fpr(0.05),
        "tpr_at_fpr_1pct": tpr_at_target_fpr(0.01),
    }


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


def build_vocab_from_training_pairs(
    paired_df: pd.DataFrame,
    train_indices: np.ndarray,
    target_coverage: float = TARGET_COVERAGE,
    min_length: int = MIN_TEXT_LENGTH,
) -> tuple[set[str], Counter]:
    token_counter = Counter()

    for idx in train_indices:
        row = paired_df.iloc[idx]
        if is_valid_text(row["human_text"], min_length):
            token_counter.update(tokenise(row["human_text"]))
        if is_valid_text(row["ai_text"], min_length):
            token_counter.update(tokenise(row["ai_text"]))

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

        if is_valid_text(row["human_text"], min_length):
            human_tokens = map_sequence(tokenise(row["human_text"]), vocab)
            if len(human_tokens) >= 2:
                human_transition_counter.update(zip(human_tokens[:-1], human_tokens[1:]))
                human_total += len(human_tokens) - 1

        if is_valid_text(row["ai_text"], min_length):
            ai_tokens = map_sequence(tokenise(row["ai_text"]), vocab)
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

        if is_valid_text(row["human_text"], min_length):
            tokens = map_sequence(tokenise(row["human_text"]), vocab)
            human_score = sequence_log_likelihood(
                tokens, human_transition_counts, human_row_totals, vocab, alpha=alpha
            )
            ai_score = sequence_log_likelihood(
                tokens, ai_transition_counts, ai_row_totals, vocab, alpha=alpha
            )
            margin = ai_score - human_score
            human_margins.append(margin)
            true_vals.append(0)
            predictions.append(1 if ai_score > human_score else 0)
            scores.append(margin)

        if is_valid_text(row["ai_text"], min_length):
            tokens = map_sequence(tokenise(row["ai_text"]), vocab)
            human_score = sequence_log_likelihood(
                tokens, human_transition_counts, human_row_totals, vocab, alpha=alpha
            )
            ai_score = sequence_log_likelihood(
                tokens, ai_transition_counts, ai_row_totals, vocab, alpha=alpha
            )
            margin = ai_score - human_score
            ai_margins.append(margin)
            true_vals.append(1)
            predictions.append(1 if ai_score > human_score else 0)
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


def evaluate_config_with_cv(
    paired_df: pd.DataFrame,
    random_seed: int = RANDOM_SEED,
):
    rng = np.random.default_rng(random_seed)
    all_fold_metrics = []
    fold_models = []

    num_pairs = len(paired_df)
    pair_indices = np.arange(num_pairs)
    rng.shuffle(pair_indices)

    fold_size = num_pairs // N_FOLDS

    for fold in range(N_FOLDS):
        logger.info("=== Fold %d ===", fold)

        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < N_FOLDS - 1 else num_pairs
        test_indices = pair_indices[test_start:test_end]

        val_fold = (fold + 1) % N_FOLDS
        val_start = val_fold * fold_size
        val_end = (val_fold + 1) * fold_size if val_fold < N_FOLDS - 1 else num_pairs
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
            alpha_grid=ALPHA_GRID,
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

        raid_metrics = compute_raid_metrics(true_vals, scores)

        accuracy = accuracy_score(true_vals, predictions)
        precision = precision_score(true_vals, predictions)
        recall = recall_score(true_vals, predictions)
        f1 = f1_score(true_vals, predictions)

        logger.info("AUROC: %.4f", raid_metrics["auroc"])
        logger.info("TPR@FPR=5%%: %.4f", raid_metrics["tpr_at_fpr_5pct"])
        logger.info("TPR@FPR=1%%: %.4f", raid_metrics["tpr_at_fpr_1pct"])
        logger.info("Accuracy: %.4f", accuracy)
        logger.info("Precision: %.4f", precision)
        logger.info("Recall: %.4f", recall)
        logger.info("F1: %.4f", f1)
        logger.info("Confusion matrix:\n%s", confusion_matrix(true_vals, predictions))

        all_fold_metrics.append(
            {
                "fold": fold,
                "num_pairs": num_pairs,
                "alpha": alpha,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auroc": raid_metrics["auroc"],
                "tpr_at_fpr_5pct": raid_metrics["tpr_at_fpr_5pct"],
                "tpr_at_fpr_1pct": raid_metrics["tpr_at_fpr_1pct"],
            }
        )

        fold_models.append(
            {
                "fold": fold,
                "test_source_ids": set(paired_df.iloc[test_indices]["source_id"].tolist()),
                "vocab": vocab,
                "human_transition_counts": human_transition_counts,
                "human_row_totals": human_row_totals,
                "ai_transition_counts": ai_transition_counts,
                "ai_row_totals": ai_row_totals,
                "alpha": alpha,
            }
        )

    return all_fold_metrics, fold_models


def build_generic_pairs_from_parquet_files(
    parquet_paths: list[str | Path],
    rng: np.random.Generator,
) -> pd.DataFrame:
    dfs = []
    for path in parquet_paths:
        try:
            df = pd.read_parquet(path)
            dfs.append(df)
        except Exception as e:
            logger.warning("Failed to read %s: %s", path, e)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    grouped_rows = []
    for source_id, group in combined.groupby("source_id", sort=False):
        idx = rng.integers(len(group))
        chosen = group.iloc[idx]

        grouped_rows.append(
            {
                "source_id": chosen["source_id"],
                "human_text": chosen["human_text"],
                "ai_text": chosen["ai_text"],
                "human_domain": chosen.get("human_domain"),
                "ai_domain": chosen.get("ai_domain"),
                "title": chosen.get("title"),
                "prompt": chosen.get("prompt"),
                "model": chosen["model"],
                "attack": chosen["attack"],
                "decoding": chosen["decoding"],
                "repetition_penalty": chosen["repetition_penalty"],
            }
        )

    return pd.DataFrame(grouped_rows)


def evaluate_trained_model_on_config_subset(
    paired_df: pd.DataFrame,
    model_artifacts: dict,
):
    if len(paired_df) == 0:
        return None

    eval_indices = np.arange(len(paired_df))

    true_vals, predictions, scores = evaluate_on_pair_indices(
        paired_df,
        eval_indices,
        model_artifacts["vocab"],
        model_artifacts["human_transition_counts"],
        model_artifacts["human_row_totals"],
        model_artifacts["ai_transition_counts"],
        model_artifacts["ai_row_totals"],
        alpha=model_artifacts["alpha"],
        min_length=MIN_TEXT_LENGTH,
    )

    raid_metrics = compute_raid_metrics(true_vals, scores)

    return {
        "accuracy": accuracy_score(true_vals, predictions),
        "precision": precision_score(true_vals, predictions),
        "recall": recall_score(true_vals, predictions),
        "f1": f1_score(true_vals, predictions),
        "auroc": raid_metrics["auroc"],
        "tpr_at_fpr_5pct": raid_metrics["tpr_at_fpr_5pct"],
        "tpr_at_fpr_1pct": raid_metrics["tpr_at_fpr_1pct"],
    }


def main():
    rng = np.random.default_rng(101)

    pair_dir = Path("raid_pairs")
    sorted_pair_files = sorted(pair_dir.glob("*.parquet"))

    generic_repeat_results = []
    generic_on_specific_results = []

    for repeat_idx in range(10):
        logger.info("=== Generic repeat %d ===", repeat_idx)

        selected_pair_files = [file for file in sorted_pair_files if rng.random() < 0.2]
        logger.info("Selected %d parquet files", len(selected_pair_files))

        paired_df = build_generic_pairs_from_parquet_files(selected_pair_files, rng)
        logger.info("Generic paired_df rows: %d", len(paired_df))

        if len(paired_df) == 0:
            logger.warning("Skipping repeat %d because generic paired_df is empty", repeat_idx)
            continue

        fold_metrics, fold_models = evaluate_config_with_cv(
            paired_df=paired_df,
            random_seed=RANDOM_SEED,
        )

        for row in fold_metrics:
            row["parquet_repeat"] = repeat_idx
            generic_repeat_results.append(row)

        for file in sorted_pair_files:
            try:
                full_config_df = pd.read_parquet(file)
            except Exception as e:
                logger.warning("Failed to read config parquet %s: %s", file, e)
                continue

            if len(full_config_df) == 0:
                continue

            config_model = full_config_df["model"].iloc[0]
            config_attack = full_config_df["attack"].iloc[0]
            config_decoding = full_config_df["decoding"].iloc[0]
            config_rep = full_config_df["repetition_penalty"].iloc[0]

            for fold_model in fold_models:
                fold = fold_model["fold"]
                test_source_ids = fold_model["test_source_ids"]

                config_df_fold = full_config_df[full_config_df["source_id"].isin(test_source_ids)].copy()

                metrics = evaluate_trained_model_on_config_subset(
                    config_df_fold,
                    fold_model,
                )

                if metrics is None:
                    continue

                generic_on_specific_results.append(
                    {
                        "model": config_model,
                        "attack": config_attack,
                        "decoding": config_decoding,
                        "repetition_penalty": config_rep,
                        "fold": fold,
                        "num_pairs": len(config_df_fold),
                        "alpha": fold_model["alpha"],
                        "parquet_repeat": repeat_idx,
                        **metrics,
                    }
                )

    output_dir = Path("raid_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    generic_repeat_df = pd.DataFrame(generic_repeat_results)
    generic_on_specific_df = pd.DataFrame(generic_on_specific_results)

    generic_repeat_df.to_csv(output_dir / "generic_clean_output.csv", index=False)
    generic_on_specific_df.to_csv(output_dir / "generic_model_specific_config.csv", index=False)

    logger.info("Saved %s", output_dir / "generic_clean_output.csv")
    logger.info("Saved %s", output_dir / "generic_model_specific_config.csv")


if __name__ == "__main__":
    main()