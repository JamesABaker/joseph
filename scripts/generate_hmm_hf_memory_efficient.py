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
    "dataset_name": "HC3/HC3",
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

        human_answers = row["human_answers"]
        human_answers = "".join(human_answers)
        if is_valid_text(human_answers, min_length):
            human_tokens = map_sequence(tokenise(human_answers), vocab)
            human_unigram_counts.update(human_tokens)
            human_total += len(human_tokens)

        chatgpt_answers = row["chatgpt_answers"]
        chatgpt_answers = "".join(chatgpt_answers)
        if is_valid_text(chatgpt_answers, min_length):
            ai_tokens = map_sequence(tokenise(chatgpt_answers), vocab)
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

        human_answers = row["human_answers"]
        if is_valid_text(human_answers, min_length):
            tokens = map_sequence(tokenise(human_answers), vocab)

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

        chatgpt_answers = row["chatgpt_answers"]
        if is_valid_text(chatgpt_answers, min_length):
            tokens = map_sequence(tokenise(chatgpt_answers), vocab)

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

    for i, human_answers in enumerate(dataset["human_answers"]):
        # print(human_answers)
        human_answers = "".join(human_answers)
        # print(row)
        if i not in train_row_index_set:
            continue
        chatgpt_answers = dataset["chatgpt_answers"][i]
        chatgpt_answers = "".join(chatgpt_answers)
        # human_answers = row["human_answers"]
        if is_valid_text(human_answers, min_length):
            token_counter.update(tokenise(human_answers))

        # chatgpt_answers = row["chatgpt_answers"]
        if is_valid_text(chatgpt_answers, min_length):
            token_counter.update(tokenise(chatgpt_answers))

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

    for i, human_answers in enumerate(dataset["human_answers"]):
        human_answers = "".join(human_answers)
        if i not in train_row_index_set:
            continue
        if mislabel_bool and rng.random() < mislabel_error:
            human_answers = dataset["chatgpt_answers"][i]
            human_answers = "".join(human_answers)
            chatgpt_answers = dataset["human_answers"][i]
            chatgpt_answers = "".join(chatgpt_answers)
        else:
            # human_answers = row["human_answers"]
            human_answers = "".join(human_answers)
            chatgpt_answers = dataset["chatgpt_answers"][i]
            chatgpt_answers = "".join(chatgpt_answers)
        if is_valid_text(human_answers, min_length):
            human_tokens = map_sequence(tokenise(human_answers), vocab)
            if len(human_tokens) >= 2:
                human_transition_counter.update(zip(human_tokens[:-1], human_tokens[1:]))
                human_total += len(human_tokens) - 1

        if is_valid_text(chatgpt_answers, min_length):
            ai_tokens = map_sequence(tokenise(chatgpt_answers), vocab)
            if len(ai_tokens) >= 2:
                ai_transition_counter.update(zip(ai_tokens[:-1], ai_tokens[1:]))
                ai_total += len(ai_tokens) - 1

    logger.info("Total human transitions: %d", human_total)
    logger.info("Total AI transitions: %d", ai_total)

    return human_transition_counter, human_total, ai_transition_counter, ai_total

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
    train_row_index_set: set[int],
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
    logger.info("Pass 3/3: evaluating on test rows")

    
    for i, human_answers in enumerate(dataset["human_answers"]):
        # print(row)
        # human_answers = "".join(human_answers)
        # print(row)
        if i in train_row_index_set:
            continue
        chatgpt_answers = dataset["chatgpt_answers"][i]
        # chatgpt_answers = "".join(chatgpt_answers)
        if is_valid_text(human_answers, min_length):
            tokens = map_sequence(tokenise(human_answers), vocab)

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

        # chatgpt_answers = row["chatgpt_answers"]
        if is_valid_text(chatgpt_answers, min_length):
            tokens = map_sequence(tokenise(chatgpt_answers), vocab)

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

    logger.info("Human margin sample: %s", human_margins[:10])
    logger.info("AI margin sample: %s", ai_margins[:10])
    logger.info("Human margin min/max: %.4f / %.4f", min(human_margins), max(human_margins))
    logger.info("AI margin min/max: %.4f / %.4f", min(ai_margins), max(ai_margins))

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


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    # logger.info("Loading Hugging Face dataset")
    # dataset_dict = load_dataset("Hello-SimpleAI/HC3")
    # dataset = dataset_dict["train"]
    dataset_dict = load_dataset("HC3/HC3")
    dataset = dataset_dict["train"]

    
    data_dir = Path(__file__).parent.parent / "data" / "all.jsonl"
    # dataset = load_dataset(str(data_dir))  # nosec B615
    df = pd.read_json(str(data_dir), lines = True)
    # print(df)
    dataset = df
    # print(df.columns)
    
    num_rows = len(dataset["human_answers"])
    row_indices = np.arange(num_rows)
    train_size = int(TRAIN_PROPORTION * num_rows)

    logger.info("Train rows: %d", train_size)
    logger.info("Test rows: %d", num_rows - train_size)

    row_indices = rng.choice(row_indices, size=train_size, replace=False)
    rng.shuffle(row_indices)
    test_size = int(np.floor(0.2*len(row_indices)))

    for i in range(1):
        if i != 4:
            test_inds = row_indices[i*test_size:(i+1)*test_size]
        else:
            test_inds = row_indices[i*test_size:]

        train_row_indices = []
        for ind in row_indices:
            if ind not in test_inds:
                train_row_indices.append(ind)
        train_row_index_set = set(train_row_indices)
        vocab_path = Path(__file__).parent.parent / "data" / f"vocab_HC3_{i}.json"
        artifact_path = Path(__file__).parent.parent / "data" / f"transition_artifacts_HC3_{i}.pkl"
        # vocab_path = Path(__file__).parent.parent / "data" / f"vocab.json"
        # artifact_path = Path(__file__).parent.parent / "data" / f"transition_artifacts.pkl"
        # logger.info("Dataset: %s", dataset)
        # logger.info("Num rows: %d", len(dataset))
        # logger.info("Features: %s", dataset.columns)

        if vocab_path.exists():
            logger.info("Loading vocab from %s", vocab_path)
            vocab, metadata = load_vocab_with_metadata(vocab_path)
        else:
            logger.info("Building vocab from training rows")
            vocab, token_counts = build_vocab_from_training_rows(
                dataset,
                train_row_index_set,
                target_coverage=TARGET_COVERAGE,
                min_length=MIN_TEXT_LENGTH,
            )

            metadata = {
                "dataset_name": "HC3/HC3",
                "target_coverage": TARGET_COVERAGE,
                "min_text_length": MIN_TEXT_LENGTH,
                "random_seed": RANDOM_SEED,
            }

            save_vocab_with_metadata(vocab, vocab_path, metadata)
        # vocab, token_counts = build_vocab_from_training_rows(
        #     dataset,
        #     train_row_index_set,
        #     target_coverage=TARGET_COVERAGE,
        #     min_length=MIN_TEXT_LENGTH,
        # )

        
        if artifact_path.exists():
            logger.info("Loading cached transition artifacts from %s", artifact_path)
            artifacts = load_transition_artifacts(artifact_path)

            human_transition_counts = artifacts["human_transition_counts"]
            ai_transition_counts = artifacts["ai_transition_counts"]
            human_row_totals = artifacts["human_row_totals"]
            ai_row_totals = artifacts["ai_row_totals"]
            human_total = artifacts["human_total"]
            ai_total = artifacts["ai_total"]

        else:
            logger.info("No cached transition artifacts found; computing them")

            human_transition_counts, human_total, ai_transition_counts, ai_total = count_transitions_from_training_rows(
                dataset,
                train_row_index_set,
                vocab,
                min_length=MIN_TEXT_LENGTH,
                rng = rng,
                mislabel_error = 0.1,
                mislabel_bool = False
            )

            human_row_totals = build_row_totals(human_transition_counts)
            ai_row_totals = build_row_totals(ai_transition_counts)

            metadata = {
                "dataset_name": "HC3/HC3",
                "target_coverage": TARGET_COVERAGE,
                "min_text_length": MIN_TEXT_LENGTH,
                "random_seed": RANDOM_SEED,
                "train_proportion": TRAIN_PROPORTION,
                "alpha": ALPHA,
                "vocab_size": len(vocab),
            }

            save_transition_artifacts(
                path=artifact_path,
                human_transition_counts=human_transition_counts,
                ai_transition_counts=ai_transition_counts,
                human_row_totals=human_row_totals,
                ai_row_totals=ai_row_totals,
                human_total=human_total,
                ai_total=ai_total,
                metadata=metadata,
            )

        true_vals, predictions, scores = evaluate_on_test_rows(
            dataset,
            train_row_index_set,
            vocab,
            human_transition_counts,
            human_row_totals,
            ai_transition_counts,
            ai_row_totals,
            alpha=ALPHA,
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

        top_human = top_transitions(human_transition_counts, TOP_K_OUTPUT)
        top_ai = top_transitions(ai_transition_counts, TOP_K_OUTPUT)

        # logger.info("Top 20 human transitions:")
        # for transition, count in top_human[:20]:
        #     logger.info("  %s -> %d", transition, count)

        # logger.info("Top 20 AI transitions:")
        # for transition, count in top_ai[:20]:
        #     logger.info("  %s -> %d", transition, count)

        # log_odds_results = compute_transition_log_odds(
        #     human_transition_counts=human_transition_counts,
        #     human_row_totals=human_row_totals,
        #     ai_transition_counts=ai_transition_counts,
        #     ai_row_totals=ai_row_totals,
        #     vocab=vocab,
        #     alpha=ALPHA,
        #     min_total_count=10,
        # )

        # top_ai_favored = sorted(
        #     log_odds_results,
        #     key=lambda x: x["log_odds_ai_vs_human"],
        #     reverse=True,
        # )[:50]

        # top_human_favored = sorted(
        #     log_odds_results,
        #     key=lambda x: x["log_odds_ai_vs_human"],
        # )[:50]

        
        # logger.info("Top 50 human favoured transitions:")
        # for row in top_human_favored:
        #     transition = row["transition"]
        #     human_count = row["human_count"]
        #     ai_count = row["ai_count"]
        #     score = row["log_odds_ai_vs_human"]

        #     logger.info(
        #         "%s | log_odds=%.4f | ai_count=%d | human_count=%d",
        #         transition,
        #         score,
        #         ai_count,
        #         human_count,
        #     )

        # logger.info("Top 50 AI favoured transitions:")
        # for row in top_ai_favored:
        #     transition = row["transition"]
        #     human_count = row["human_count"]
        #     ai_count = row["ai_count"]
        #     score = row["log_odds_ai_vs_human"]

        #     logger.info(
        #         "%s | log_odds=%.4f | ai_count=%d | human_count=%d",
        #         transition,
        #         score,
        #         ai_count,
        #         human_count,
        #     )

        # human_unigram_counts, human_total, ai_unigram_counts, ai_total = count_unigrams_from_training_rows(
        #     dataset,
        #     train_row_indices.tolist(),
        #     vocab,
        #     MIN_TEXT_LENGTH,
        # )

        # true_vals, predictions, scores = evaluate_unigram_on_test_rows(
        #     dataset,
        #     train_row_indices.tolist(),
        #     vocab,
        #     human_unigram_counts,
        #     human_total,
        #     ai_unigram_counts,
        #     ai_total,
        #     ALPHA,
        #     MIN_TEXT_LENGTH,
        # )
        
        # accuracy = accuracy_score(true_vals, predictions)
        # precision = precision_score(true_vals, predictions)
        # recall = recall_score(true_vals, predictions)
        # f1 = f1_score(true_vals, predictions)
        # cm = confusion_matrix(true_vals, predictions)

        # logger.info("Unigram Accuracy: %.4f", accuracy)
        # logger.info("Unigram Precision: %.4f", precision)
        # logger.info("Unigram Recall: %.4f", recall)
        # logger.info("Unigram F1: %.4f", f1)
        # logger.info("Unigram Confusion matrix:\n%s", cm)

if __name__ == "__main__":
    main()