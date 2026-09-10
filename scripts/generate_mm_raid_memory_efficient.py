import logging
import math
import re
import sys
from collections import Counter, defaultdict
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import json
import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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
TRAIN_PROPORTION = 0.7
TARGET_COVERAGE = 0.98
TOP_K_OUTPUT = 1000
ALPHA = 1.0

START_TOKEN = "<START>"
END_TOKEN = "<END>"
RARE_TOKEN = "<RARE>"
NUM_TOKEN = "<NUM>"
URL_TOKEN = "<URL>"

metadata = {
    "dataset_name": "dmitva/human_ai_generated_text",
    "train_proportion": TRAIN_PROPORTION,
    "target_coverage": TARGET_COVERAGE,
    "min_text_length": MIN_TEXT_LENGTH,
    "random_seed": RANDOM_SEED,
    "special_tokens": [
        START_TOKEN,
        END_TOKEN,
        RARE_TOKEN,
        NUM_TOKEN,
        URL_TOKEN,
    ],
}

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
    
    text = re.sub(r"\s+", " ", text.lower().strip())
    text = text.replace("\\n", " ")
    text = re.sub(r"\s+", " ", text.lower().strip())
    text = normalise_unicode_punctuation(text)
    raw_tokens = TOKEN_PATTERN.findall(text)
    tokens = [normalise_token(tok) for tok in raw_tokens]

    return [START_TOKEN, *tokens, END_TOKEN] if tokens else []


def map_token(token: str, vocab: set[str]) -> str:
    return token if token in vocab else RARE_TOKEN


def map_sequence(tokens: list[str], vocab: set[str]) -> list[str]:
    return [map_token(token, vocab) for token in tokens]


def build_row_totals(
    transition_counter: Counter[tuple[str, str]]
) -> dict[str, int]:
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


def is_valid_text(text: str, min_length: int = MIN_TEXT_LENGTH) -> bool:
    return bool(text) and len(text.strip()) > min_length


def iter_training_rows(dataset, train_row_index_set: set[int]):
    for idx, row in enumerate(dataset):
        if idx in train_row_index_set:
            yield idx, row


def iter_test_rows(dataset, train_row_index_set: set[int]):
    for idx, row in enumerate(dataset):
        if idx not in train_row_index_set:
            yield idx, row

def count_unigrams_from_training_rows(
    dataset,
    train_selector,
    vocab: set[str],
    min_length: int = MIN_TEXT_LENGTH,
) -> tuple[Counter, int, Counter, int]:
    human_unigram_counts = Counter()
    ai_unigram_counts = Counter()
    human_total = 0
    ai_total = 0

    logger.info("Counting unigrams from training rows")

    for idx, row in enumerate(dataset):
        if idx in train_selector:
            continue

        human_text = row["human_text"]
        human_text = "".join(human_text)
        if is_valid_text(human_text, min_length):
            human_tokens = map_sequence(tokenise(human_text), vocab)
            human_unigram_counts.update(human_tokens)
            human_total += len(human_tokens)

        ai_text = row["ai_text"]
        ai_text = "".join(ai_text)
        if is_valid_text(ai_text, min_length):
            ai_tokens = map_sequence(tokenise(ai_text), vocab)
            ai_unigram_counts.update(ai_tokens)
            ai_total += len(ai_tokens)

    return human_unigram_counts, human_total, ai_unigram_counts, ai_total
def smoothed_unigram_probability(
    token: str,
    unigram_counts: Counter[str],
    total_count: int,
    vocab: set[str],
    alpha: float = 1.0,
) -> float:
    vocab_size = len(vocab)
    count = unigram_counts.get(token, 0)
    return (count + alpha) / (total_count + alpha * vocab_size)

def unigram_sequence_log_likelihood(
    tokens: list[str],
    unigram_counts: Counter[str],
    total_count: int,
    vocab: set[str],
    alpha: float = 1.0,
    normalize_by_length: bool = True,
) -> float:
    if not tokens:
        return 0.0

    log_prob = 0.0
    for token in tokens:
        p = smoothed_unigram_probability(
            token=token,
            unigram_counts=unigram_counts,
            total_count=total_count,
            vocab=vocab,
            alpha=alpha,
        )
        log_prob += math.log(p)

    if normalize_by_length:
        return log_prob / len(tokens)

    return log_prob
def evaluate_unigram_on_test_rows(
    dataset,
    train_selector,
    vocab: set[str],
    human_unigram_counts: Counter[str],
    human_total: int,
    ai_unigram_counts: Counter[str],
    ai_total: int,
    alpha: float = ALPHA,
    min_length: int = MIN_TEXT_LENGTH,
) -> tuple[list[int], list[int], list[float]]:
    true_vals = []
    predictions = []
    scores = []

    logger.info("Evaluating unigram baseline on test rows")

    for idx, row in enumerate(dataset):
        if idx not in train_selector:
            continue

        human_text = row["human_text"]
        if is_valid_text(human_text, min_length):
            tokens = map_sequence(tokenise(human_text), vocab)

            human_score = unigram_sequence_log_likelihood(
                tokens, human_unigram_counts, human_total, vocab, alpha=alpha
            )
            ai_score = unigram_sequence_log_likelihood(
                tokens, ai_unigram_counts, ai_total, vocab, alpha=alpha
            )

            prediction = 1 if ai_score > human_score else 0
            true_vals.append(0)
            predictions.append(prediction)
            scores.append(ai_score - human_score)

        ai_text = row["ai_text"]
        if is_valid_text(ai_text, min_length):
            tokens = map_sequence(tokenise(ai_text), vocab)

            human_score = unigram_sequence_log_likelihood(
                tokens, human_unigram_counts, human_total, vocab, alpha=alpha
            )
            ai_score = unigram_sequence_log_likelihood(
                tokens, ai_unigram_counts, ai_total, vocab, alpha=alpha
            )

            prediction = 1 if ai_score > human_score else 0
            true_vals.append(1)
            predictions.append(prediction)
            scores.append(ai_score - human_score)

    return true_vals, predictions, scores

def build_vocab_from_training_rows(
    dataset,
    train_row_index_set: set[int],
    target_coverage: float = TARGET_COVERAGE,
    min_length: int = MIN_TEXT_LENGTH,
) -> tuple[set[str], Counter]:
    token_counter = Counter()

    logger.info("Pass 1/3: building vocabulary from training rows")

    for i, human_text in enumerate(dataset["human_answers"]):
        if i not in train_row_index_set:
            continue
        try:
            human_text = "".join(human_text)
            ai_text = dataset["chatgpt_answers"].iloc[i]
            ai_text = "".join(ai_text)
        except:
            continue
        # print(row)
        # human_text = row["human_answers"]
        if is_valid_text(human_text, min_length):
            token_counter.update(tokenise(human_text))

        # ai_text = row["chatgpt_answers"]
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
    logger.info("Target coverage: %.4f", target_coverage)
    logger.info("Approx. achieved coverage: %.4f", cumulative)

    return vocab, token_counter


def count_transitions_from_training_rows(
    dataset,
    train_row_index_set: set[int],
    vocab: set[str],
    min_length: int = MIN_TEXT_LENGTH,
    rng = None,
    mislabel_error = 0,
    mislabel_bool = False
) -> tuple[Counter, int, Counter, int]:
    human_transition_counter = Counter()
    ai_transition_counter = Counter()
    human_total = 0
    ai_total = 0

    logger.info("Pass 2/3: counting transitions from training rows")

    for i, human_text in enumerate(dataset["human_answers"]):
        if i not in train_row_index_set:
            continue
        try:
            human_text = "".join(human_text)
            ai_text = dataset["chatgpt_answers"].iloc[i]
            ai_text = "".join(ai_text)
        except:
            continue
        # human_text = "".join(human_text)
        # if i not in train_row_index_set:
        #     continue
        if mislabel_bool and rng.random() < mislabel_error:
            human_text = dataset["chatgpt_answers"].iloc[i]
            human_text = "".join(human_text)
            ai_text = dataset["human_answers"].iloc[i]
            ai_text = "".join(ai_text)
        else:
            # human_text = row["human_answers"]
            human_text = "".join(human_text)
            ai_text = dataset["chatgpt_answers"].iloc[i]
            ai_text = "".join(ai_text)
        if is_valid_text(human_text, min_length):
            human_tokens = map_sequence(tokenise(human_text), vocab)
            if len(human_tokens) >= 2:
                human_transition_counter.update(zip(human_tokens[:-1], human_tokens[1:]))
                human_total += len(human_tokens) - 1

        if is_valid_text(ai_text, min_length):
            ai_tokens = map_sequence(tokenise(ai_text), vocab)
            if len(ai_tokens) >= 2:
                ai_transition_counter.update(zip(ai_tokens[:-1], ai_tokens[1:]))
                ai_total += len(ai_tokens) - 1

    logger.info("Total human transitions: %d", human_total)
    logger.info("Total AI transitions: %d", ai_total)

    return human_transition_counter, human_total, ai_transition_counter, ai_total

def optimise_alpha(
    dataset,
    validation_row_index_set: set[int],
    alpha_lims: list[float],
    num_alpha_steps: int,
    vocab: set[str],
    human_transition_counter: Counter,
    human_row_totals: Counter,
    ai_transition_counter: Counter,
    ai_row_totals: Counter,
    min_length: int = MIN_TEXT_LENGTH,
):
    log_alphas = [np.log(alpha_lims[0]) + (np.log(alpha_lims[1]) - np.log(alpha_lims[0]))*i/num_alpha_steps for i in range(num_alpha_steps + 1)]
    alphas = [np.exp(alpha) for alpha in log_alphas]
    recall_values = []
    for i, alpha in enumerate(alphas):
        print(i/num_alpha_steps, end = "\r")
        true_vals, predicitions, scores = evaluate_on_test_rows(
            dataset,
            validation_row_index_set,
            vocab,
            human_transition_counter,
            human_row_totals,
            ai_transition_counter,
            ai_row_totals,
            alpha,
            min_length,
        )
        recall = recall_score(true_vals, predicitions)  
        recall_values.append(recall)

    recall_values = np.array(recall_values)
    alpha_max = alphas[np.argmax(recall_values)]
    return alpha_max

def optimise_alpha_zooming(
    dataset,
    validation_row_index_set: set[int],
    alpha_lims: list[float],
    num_alpha_steps: int,
    num_zoom_steps: int,
    depth: int,
    vocab: set[str],
    human_transition_counter: Counter,
    human_row_totals: Counter,
    ai_transition_counter: Counter,
    ai_row_totals: Counter,
    min_length: int = MIN_TEXT_LENGTH,
):
    log_alphas = [np.log(alpha_lims[0]) + (np.log(alpha_lims[1]) - np.log(alpha_lims[0]))*i/num_alpha_steps for i in range(num_alpha_steps + 1)]
    alphas = [np.exp(alpha) for alpha in log_alphas]
    recall_values = []
    for i, alpha in enumerate(alphas):
        print(i/num_alpha_steps, end = "\r")
        true_vals, predicitions, scores = evaluate_on_test_rows(
            dataset,
            validation_row_index_set,
            vocab,
            human_transition_counter,
            human_row_totals,
            ai_transition_counter,
            ai_row_totals,
            alpha,
            min_length,
        )
        recall = recall_score(true_vals, predicitions)  
        recall_values.append(recall)

    if depth == num_zoom_steps:
        return alphas[np.argmax(recall_values)]
    else:
        for i, val in enumerate(recall_values[1:-1]):
            j = i + 1
            if val >= recall_values[j - 1] and val >= recall_values[j + 1]:
                alpha_lims_t = [alphas[j - 1], alphas[j + 1]]
                return optimise_alpha_zooming(
                    dataset,
                    validation_row_index_set,
                    alpha_lims_t,
                    num_alpha_steps,
                    num_zoom_steps,
                    depth + 1,
                    vocab,
                    human_transition_counter,
                    human_row_totals,
                    ai_transition_counter,
                    ai_row_totals,
                    min_length,
                )
            
        print("Failed to find local max, increase alpha width")
    return 
def save_vocab_with_metadata(
    vocab: set[str],
    path: str | Path,
    metadata: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": metadata or {},
        "vocab": sorted(vocab),
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return
def save_transition_artifacts(
    path: str | Path,
    human_transition_counts,
    ai_transition_counts,
    human_row_totals,
    ai_row_totals,
    human_total: int,
    ai_total: int,
    metadata: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": metadata or {},
        "human_transition_counts": human_transition_counts,
        "ai_transition_counts": ai_transition_counts,
        "human_row_totals": human_row_totals,
        "ai_row_totals": ai_row_totals,
        "human_total": human_total,
        "ai_total": ai_total,
    }

    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    return

def load_vocab_with_metadata(path: str | Path) -> tuple[set[str], dict]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    vocab = set(payload["vocab"])
    metadata = payload.get("metadata", {})
    return vocab, metadata

def load_transition_artifacts(path: str | Path) -> dict:
    path = Path(path)

    with path.open("rb") as f:
        payload = pickle.load(f)

    return payload

def evaluate_on_test_rows(
    dataset,
    test_row_index_set: set[int],
    vocab: set[str],
    human_transition_counts: Counter,
    human_row_totals: dict[str, int],
    ai_transition_counts: Counter,
    ai_row_totals: dict[str, int],
    alpha: float = ALPHA,
    min_length: int = MIN_TEXT_LENGTH,
) -> tuple[list[int], list[int], list[float]]:
    true_vals = []
    predictions = []
    scores = []
    human_margins = []
    ai_margins = []
    # logger.info("Pass 3/3: evaluating on test rows")

    test_counter = 0
    train_counter = 0
    testing_inds = []
    for i, human_text in enumerate(dataset["human_answers"]):
        
        try:
            human_text = "".join(human_text)
            ai_text = dataset["chatgpt_answers"].iloc[i]
            ai_text = "".join(ai_text)
        except:
            continue
        # human_text = "".join(human_text)
        # print(row)
        if i not in test_row_index_set:
            train_counter += 1
            continue
        testing_inds.append(i)
        test_counter += 1
        # ai_text = dataset["chatgpt_answers"].iloc[i]
        # ai_text = "".join(ai_text)
        if is_valid_text(human_text, min_length):
            tokens = map_sequence(tokenise(human_text), vocab)

            human_score = sequence_log_likelihood(
                tokens,
                human_transition_counts,
                human_row_totals,
                vocab,
                alpha=alpha,
            )
            ai_score = sequence_log_likelihood(
                tokens,
                ai_transition_counts,
                ai_row_totals,
                vocab,
                alpha=alpha,
            )
            margin = ai_score - human_score
            human_margins.append(margin)
            prediction = 1 if ai_score > human_score else 0

            true_vals.append(0)
            predictions.append(prediction)
            scores.append(ai_score - human_score)

        # ai_text = row["chatgpt_answers"]
        if is_valid_text(ai_text, min_length):
            tokens = map_sequence(tokenise(ai_text), vocab)

            human_score = sequence_log_likelihood(
                tokens,
                human_transition_counts,
                human_row_totals,
                vocab,
                alpha=alpha,
            )
            ai_score = sequence_log_likelihood(
                tokens,
                ai_transition_counts,
                ai_row_totals,
                vocab,
                alpha=alpha,
            )
            
            margin = ai_score - human_score
            ai_margins.append(margin)
            prediction = 1 if ai_score > human_score else 0

            true_vals.append(1)
            predictions.append(prediction)
            scores.append(ai_score - human_score)

    # logger.info("Human margin sample: %s", human_margins[:10])
    # logger.info("AI margin sample: %s", ai_margins[:10])
    # logger.info("Human margin min/max: %.4f / %.4f", min(human_margins), max(human_margins))
    # logger.info("AI margin min/max: %.4f / %.4f", min(ai_margins), max(ai_margins))

    # print("proportion testing")
    # print(test_counter/(train_counter + test_counter))
    # print(testing_inds)
    return true_vals, predictions, scores

def compute_transition_log_odds(
    human_transition_counts,
    human_row_totals,
    ai_transition_counts,
    ai_row_totals,
    vocab,
    alpha=1.0,
    min_total_count=5,
):
    all_transitions = set(human_transition_counts.keys()) | set(ai_transition_counts.keys())
    results = []

    for prev_token, next_token in all_transitions:
        human_count = human_transition_counts.get((prev_token, next_token), 0)
        ai_count = ai_transition_counts.get((prev_token, next_token), 0)
        total_count = human_count + ai_count

        if total_count < min_total_count:
            continue

        p_human = smoothed_transition_probability(
            prev_token, next_token, human_transition_counts, human_row_totals, vocab, alpha
        )
        p_ai = smoothed_transition_probability(
            prev_token, next_token, ai_transition_counts, ai_row_totals, vocab, alpha
        )

        score = math.log(p_ai) - math.log(p_human)

        results.append(
            {
                "transition": (prev_token, next_token),
                "human_count": human_count,
                "ai_count": ai_count,
                "total_count": total_count,
                "human_prob": p_human,
                "ai_prob": p_ai,
                "log_odds_ai_vs_human": score,
            }
        )

    return results

def top_transitions(counter: Counter, top_k: int) -> list[tuple[tuple[str, str], int]]:
    return counter.most_common(top_k)

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

def compute_raid_metrics(true_vals, scores):
    """
    Compute AUROC, TPR@FPR=5%, and TPR@FPR=1%.
    Assumes:
      - true_vals: 0 for human, 1 for AI
      - scores: higher means more AI-like
    """
    auroc = roc_auc_score(true_vals, scores)

    fpr, tpr, thresholds = roc_curve(true_vals, scores)

    def tpr_at_target_fpr(target_fpr: float) -> float:
        valid_indices = [i for i, f in enumerate(fpr) if f <= target_fpr]
        if not valid_indices:
            return 0.0
        return max(tpr[i] for i in valid_indices)

    tpr_at_5 = tpr_at_target_fpr(0.05)
    tpr_at_1 = tpr_at_target_fpr(0.01)

    return {
        "auroc": auroc,
        "tpr_at_fpr_5pct": tpr_at_5,
        "tpr_at_fpr_1pct": tpr_at_1,
    }

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

def build_grouped_pairs_for_config(
    dataset,
    target_model: str,
    target_attack: str = "none",
    target_decoding: str = "sampling",
    target_repetition_penalty: str = "no",
):
    pairs = {}

    for row in dataset:
        source_id = row["source_id"]
        model = row["model"]
        attack = row["attack"]
        decoding = row["decoding"]
        repetition_penalty = row["repetition_penalty"]

        if source_id not in pairs:
            pairs[source_id] = {}

        if model == "human" and attack == "none":
            pairs[source_id]["human_text"] = row["generation"]
            pairs[source_id]["domain"] = row["domain"]
            pairs[source_id]["title"] = row["title"]

        elif (
            model == target_model
            and attack == target_attack
            and decoding == target_decoding
            and repetition_penalty == target_repetition_penalty
        ):
            pairs[source_id]["ai_text"] = row["generation"]
            pairs[source_id]["prompt"] = row["prompt"]
            pairs[source_id]["ai_domain"] = row["domain"]

    paired_rows = []
    for source_id, item in pairs.items():
        if "human_text" in item and "ai_text" in item:
            paired_rows.append(
                {
                    "source_id": source_id,
                    "human_text": item["human_text"],
                    "ai_text": item["ai_text"],
                    "domain": item.get("domain"),
                    "title": item.get("title"),
                    "prompt": item.get("prompt"),
                    "target_model": target_model,
                    "target_attack": target_attack,
                    "target_decoding": target_decoding,
                    "target_repetition_penalty": target_repetition_penalty,
                }
            )

    return pd.DataFrame(paired_rows)

def main()-> None:
    rng = np.random.default_rng(RANDOM_SEED)

    MODELS = [
        "chatgpt", "gpt4", "gpt3", "gpt2",
        "llama-chat", "mistral", "mistral-chat",
        "mpt", "mpt-chat", "cohere", "cohere-chat",
    ]

    ATTACKS = [
        "none",
        "homoglyph",
        "number",
        "article_deletion",
        "insert_paragraphs",
        "perplexity_misspelling",
        "upper_lower",
        "whitespace",
        "zero_width_space",
        "synonym",
        "paraphrase",
        "alternative_spelling",
    ]

    DECODINGS = ["greedy", "sampling"]
    dataset = load_dataset("liamdugan/raid", "raid")["train"]
    all_fold_metrics = []

    #     # Step 1: inspect chatgpt configs
    for model_target in MODELS:
        combo_counter = Counter()
        for row in dataset:
            if row["model"] == model_target:
                combo = (row["attack"], row["decoding"], row["repetition_penalty"])
                combo_counter[combo] += 1

        print(f"Available {model_target} configs:")
        for combo, count in sorted(combo_counter.items()):
            print(combo, count)

        # Step 2: rebuild iterator if using non-streaming dataset
        pairs = {}

        for row in dataset:
            model = row["model"]
            attack = row["attack"]

            # if model not in {"human", "chatgpt"}:
                # continue
            if attack != "none":
                continue

            source_id = row["source_id"]

            if source_id not in pairs:
                pairs[source_id] = {}

            if model == "human":
                pairs[source_id]["human_text"] = row["generation"]

            elif (
                model == model_target
                and row["decoding"] == "sampling"
                and row["repetition_penalty"] == "no"
            ):
                pairs[source_id]["ai_text"] = row["generation"]
                pairs[source_id]["domain"] = row["domain"]
                pairs[source_id]["title"] = row["title"]
                pairs[source_id]["prompt"] = row["prompt"]

        paired_rows = []
        for source_id, item in pairs.items():
            if "human_text" in item and "ai_text" in item:
                paired_rows.append({
                    "source_id": source_id,
                    "human_text": item["human_text"],
                    "ai_text": item["ai_text"],
                    "domain": item.get("domain"),
                    "title": item.get("title"),
                    "prompt": item.get("prompt"),
                })

        paired_df = pd.DataFrame(paired_rows)
        # # paired_df.to_csv("../data/raid_paired_df.csv", index=False)
        
        # paired_df = pd.read_csv("../data/raid_paired_df.csv")
        print("Number of pairs:", len(paired_df))
        print(paired_df.head())

        num_pairs = len(paired_df)
        logger.info("Number of grouped pairs: %d", num_pairs)

        pair_indices = np.arange(num_pairs)
        rng.shuffle(pair_indices)

        fold_size = num_pairs // 5
        alpha_grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]

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

            raid_metrics = compute_raid_metrics(true_vals, scores)

            logger.info("AUROC: %.4f", raid_metrics["auroc"])
            logger.info("TPR@FPR=5%%: %.4f", raid_metrics["tpr_at_fpr_5pct"])
            logger.info("TPR@FPR=1%%: %.4f", raid_metrics["tpr_at_fpr_1pct"])

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
                    "model": model,
                    "fold": fold,
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

    metrics_df = pd.DataFrame(all_fold_metrics)
    logger.info("Fold metrics summary:\n%s", metrics_df)
    logger.info("Mean metrics:\n%s", metrics_df[["accuracy", "precision", "recall", "f1"]].mean())
    logger.info("Std metrics:\n%s", metrics_df[["accuracy", "precision", "recall", "f1"]].std())

    return
if __name__ == "__main__":
    main()