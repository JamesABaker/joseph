import logging
import re
import sys
from collections import Counter
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import numpy as np
import pandas as pd
import math
from datasets import load_dataset


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
SAMPLE_SIZE_PER_CLASS = 1000
TOP_K_OUTPUT = 1000
RANDOM_SEED = 42

START_TOKEN = "<START>"
END_TOKEN = "<END>"
RARE_TOKEN = "<RARE>"
NUM_TOKEN = "<NUM>"
URL_TOKEN = "<URL>"

def build_row_totals(
    transition_counter: Counter[tuple[str, str]]
) -> dict[str, int]:
    """
    Compute total outgoing transition counts for each source token.
    """
    row_totals = defaultdict(int)

    for (prev_token, next_token), count in transition_counter.items():
        row_totals[prev_token] += count

    return dict(row_totals)


def conditional_probabilities_observed_only(
    transition_counter: Counter[tuple[str, str]],
    row_totals: dict[str, int],
) -> dict[tuple[str, str], float]:
    """
    Compute unsmoothed conditional probabilities for observed transitions only.
    Returns:
        P(next_token | prev_token)
    """
    probs = {}

    for (prev_token, next_token), count in transition_counter.items():
        total = row_totals[prev_token]
        if total > 0:
            probs[(prev_token, next_token)] = count / total

    return probs


def smoothed_transition_probability(
    prev_token: str,
    next_token: str,
    transition_counter: Counter[tuple[str, str]],
    row_totals: dict[str, int],
    vocab: set[str],
    alpha: float = 1.0,
) -> float:
    """
    Additive-smoothed conditional probability:
        P(next_token | prev_token)
      = (count(prev_token, next_token) + alpha)
        / (row_total(prev_token) + alpha * |V|)
    """
    vocab_size = len(vocab)
    count = transition_counter.get((prev_token, next_token), 0)
    row_total = row_totals.get(prev_token, 0)

    return (count + alpha) / (row_total + alpha * vocab_size)

def normalise_token(token: str) -> str:
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", token):
        return NUM_TOKEN
    if token == "url":
        return URL_TOKEN
    return token


def sequence_log_likelihood(
    tokens: list[str],
    transition_counter: Counter[tuple[str, str]],
    row_totals: dict[str, int],
    vocab: set[str],
    alpha: float = 1.0,
) -> float:
    """
    Compute the log-likelihood of a token sequence under a smoothed
    first-order Markov model.
    """
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

    return log_prob / (len(tokens) - 1)

def tokenise(text: str) -> list[str]:
    if not text:
        return []

    text = text.replace("\\n", " ")
    text = re.sub(r"\s+", " ", text.lower().strip())
    raw_tokens = TOKEN_PATTERN.findall(text)
    tokens = [normalise_token(tok) for tok in raw_tokens]

    return [START_TOKEN, *tokens, END_TOKEN] if tokens else []

def extract_valid_texts(answer_series: pd.Series, min_length: int = MIN_TEXT_LENGTH) -> list[str]:
    """Flatten answer lists and keep only sufficiently long texts."""
    valid_texts = []

    for answers in answer_series:
        if not answers:
            continue

        text = " ".join(answers)
        if text and len(text.strip()) > min_length:
            valid_texts.append(text)

    return valid_texts

def find_most_common_tokens(texts: list[str], target_coverage: float = 0.98):

    token_counter = Counter()
    # help(token_counter)
    for text in texts:
        tokens = tokenise(text)
        token_counter.update(tokens)
        
    total = sum(token_counter.values())
    vocab = {START_TOKEN, END_TOKEN, RARE_TOKEN}
    if total == 0:
        return vocab, token_counter

    cumulative = 0
    for token, count in token_counter.most_common():
        # print(cumulative)
        if count == 1:
            break
        vocab.add(token)
        cumulative += count / total
        if cumulative >= target_coverage:
            break
    # logger.info("Cumulative coverage achieved: ", cumulative)
    return vocab, token_counter

def map_token(token: str, vocab: set[str]) -> str:
    """Map a token into the shared vocabulary, replacing unknowns with <RARE>."""
    return token if token in vocab else RARE_TOKEN


def map_sequence(tokens: list[str], vocab: set[str]) -> list[str]:
    """Map a tokenized sequence into the shared vocabulary."""
    return [map_token(token, vocab) for token in tokens]

def count_transitions(texts: list[str], vocab: set[str]) -> tuple[Counter, int]:
    """Count bigram token transitions across a list of texts."""
    transition_counter = Counter()
    total_transitions = 0

    for text in texts:
        tokens = tokenise(text)
        if len(tokens) < 2:
            continue
        tokens = map_sequence(tokens, vocab)
        transitions = zip(tokens[:-1], tokens[1:])
        transition_counter.update(transitions)
        total_transitions += len(tokens) - 1

    return transition_counter, total_transitions


def top_transitions(counter: Counter, top_k: int) -> list[tuple[tuple[str, str], int]]:
    """Return the top-k most common transitions."""
    return counter.most_common(top_k)

def extract_texts_by_split(
    df: pd.DataFrame,
    train_row_index_set: set[int],
    min_length: int = MIN_TEXT_LENGTH,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Iterate once through the full dataframe and extract:
    - train human texts
    - train ai texts
    - test human texts
    - test ai texts
    based on row membership.
    """
    train_human_texts = []
    train_ai_texts = []
    test_human_texts = []
    test_ai_texts = []

    for row in df.itertuples(index=True):
        idx = row.Index
        target_human = train_human_texts if idx in train_row_index_set else test_human_texts
        target_ai = train_ai_texts if idx in train_row_index_set else test_ai_texts

        if row["human_text"]:
            human_text = " ".join(row["human_text"])
            if human_text and len(human_text.strip()) > min_length:
                target_human.append(human_text)

        if row["ai_text"]:
            ai_text = " ".join(row["ai_text"])
            if ai_text and len(ai_text.strip()) > min_length:
                target_ai.append(ai_text)

    return train_human_texts, train_ai_texts, test_human_texts, test_ai_texts

def extract_texts_by_split_from_hf_dataset(
    dataset,
    train_row_index_set: set[int],
    min_length: int = MIN_TEXT_LENGTH,
):
    train_human_texts = []
    train_ai_texts = []
    test_human_texts = []
    test_ai_texts = []

    for idx, human_text in enumerate(dataset["human_text"]):
        is_train = idx in train_row_index_set
        # print(human_text)
        # print(type(human_text))
        # print(len(human_text.strip()))
        # flop
        # human_text = row["human_text"]
        if human_text and len(human_text.strip()) > min_length:
            if is_train:
                train_human_texts.append(human_text)
            else:
                test_human_texts.append(human_text)

        ai_text = dataset["ai_text"][idx]
        if ai_text and len(ai_text.strip()) > min_length:
            if is_train:
                train_ai_texts.append(ai_text)
            else:
                test_ai_texts.append(ai_text)

    return train_human_texts, train_ai_texts, test_human_texts, test_ai_texts

def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    # data_path = Path(__file__).parent.parent / "data" / "all.jsonl"
    data_path = Path(__file__).parent.parent / "data" / "model_training_dataset.csv"
    output_path = Path(__file__).parent.parent / "data" / "transition_counts_2.csv"

    dataset = load_dataset("dmitva/human_ai_generated_text")
    # dataset = load_dataset(path = "csv", data_files = "../data/model_training_dataset.csv")
    train_split = dataset["train"]
    # print()
    logger.info("Loading HC3 data from %s", data_path)
    # df = pd.read_json(data_path, lines=True)
    # df = pd.read_csv(data_path)

    row_indices = np.arange(train_split.num_rows)
    train_size = int(0.7 * train_split.num_rows)

    train_row_indices = rng.choice(row_indices, size=train_size, replace=False)

    train_row_index_set = set(train_row_indices.tolist())
    # print(train_row_indices)
    # flop
    train_human_texts, train_ai_texts, test_human_texts, test_ai_texts = extract_texts_by_split_from_hf_dataset(
        df,
        train_row_index_set,
        min_length=MIN_TEXT_LENGTH,
    )
    print(train_human_texts)
    flop
    # logger.info("Columns: %s", list(df.columns))
    # flop

    # row_indices = np.arange(len(df))
    # train_size = int(0.7 * len(df))
    # print(train_size)
    # train_row_indices = rng.choice(row_indices, size=train_size, replace=False)
    # train_row_index_set = set(train_row_indices.tolist())

    # test_row_indices = [i for i in row_indices if i not in train_row_index_set]

    # print("making dfs")
    # train_df = df.iloc[train_row_indices].reset_index(drop=True)
    # print("made train df")
    # test_df = df.iloc[test_row_indices].reset_index(drop=True)
    # print("made test df")
    # # train_human_texts = extract_valid_texts(train_df["human_answers"])
    # train_ai_texts = extract_valid_texts(train_df["chatgpt_answers"])

    # test_human_texts = extract_valid_texts(test_df["human_answers"])
    # test_ai_texts = extract_valid_texts(test_df["chatgpt_answers"])
    
    # train_human_texts = extract_valid_texts(train_df["human_text"])
    # train_ai_texts = extract_valid_texts(train_df["ai_text"])

    # test_human_texts = extract_valid_texts(test_df["human_text"])
    # test_ai_texts = extract_valid_texts(test_df["ai_text"])

    logger.info("Extracting valid human and AI texts")
    # human_texts = extract_valid_texts(df["human_answers"])
    # ai_texts = extract_valid_texts(df["chatgpt_answers"])

    # logger.info("Valid human texts: %d", len(human_texts))
    # logger.info("Valid AI texts: %d", len(ai_texts))

    # ai_sample_size = min(int(training_proportion*len(ai_texts)), len(ai_texts))

    # logger.info("Sampling %d human texts", human_sample_size)
    # logger.info("Sampling %d AI texts", ai_sample_size)

    # sampled_human_indices = rng.choice(len(human_texts), size=human_sample_size, replace=False)
    # sampled_ai_indices = rng.choice(len(ai_texts), size=ai_sample_size, replace=False)

    # sampled_human_texts = [human_texts[ind] for ind in sampled_human_indices]
    # sampled_ai_texts = [ai_texts[ind] for ind in sampled_ai_indices]
    sampled_texts = train_human_texts + train_ai_texts
    # sampled_texts = sampled_human_texts + sampled_ai_texts
    target_coverage = 0.98
    vocab, token_counts = find_most_common_tokens(sampled_texts, target_coverage)
    logger.info("Counting human transitions")
    human_transition_counts, human_total = count_transitions(train_human_texts, vocab)
    # human_transition_counts, human_total = count_transitions(sampled_human_texts, vocab)

    logger.info("Counting AI transitions")
    ai_transition_counts, ai_total = count_transitions(train_ai_texts, vocab)
    # ai_transition_counts, ai_total = count_transitions(sampled_ai_texts, vocab)

    logger.info("Total human transitions counted: %d", human_total)
    logger.info("Total AI transitions counted: %d", ai_total)
    top_human = top_transitions(human_transition_counts, TOP_K_OUTPUT)
    top_ai = top_transitions(ai_transition_counts, TOP_K_OUTPUT)
    # print(human_transition_counts)
    rows = []
    max_rows = min(TOP_K_OUTPUT, len(top_human), len(top_ai))

    for rank in range(max_rows):
        human_transition, human_count = top_human[rank]
        ai_transition, ai_count = top_ai[rank]

        rows.append(
            {
                "frequency_ranking": rank,
                "human_transition": human_transition,
                "human_transition_rate": human_count/human_total,
                "ai_transition": ai_transition,
                "ai_transition_rate": ai_count/ai_total,
            }
        )

    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_path, index=False)

    logger.info("Saved top transition counts to %s", output_path)

    human_row_totals = build_row_totals(human_transition_counts)
    ai_row_totals = build_row_totals(ai_transition_counts)

    # test_texts = []
    # # not_sampled_human_texts = []
    # for i in range(len(human_texts)):
    #     if i not in sampled_human_indices:
    #         test_texts.append([human_texts[i], 0])
            
    # # not_sampled_ai_texts = []
    # for i in range(len(ai_texts)):
    #     if i not in sampled_ai_indices:
    #         test_texts.append([ai_texts[i], 1])

    test_texts = []
    for text in test_human_texts:
        test_texts.append([text, 0])
    for text in test_ai_texts:
        test_texts.append([text, 1])
        
    # test_texts = test_human_texts + test_ai_texts
    true_vals = []
    predictions = []
    for i, pair in enumerate(test_texts):
        true_vals.append(pair[1])
        text = pair[0]
        tokens = tokenise(text)
        tokens = map_sequence(tokens, vocab)
        human_score = sequence_log_likelihood(
            tokens,
            human_transition_counts,
            human_row_totals,
            vocab,
            alpha=1.0,
        )

        ai_score = sequence_log_likelihood(
            tokens,
            ai_transition_counts,
            ai_row_totals,
            vocab,
            alpha=1.0,
        )

        prediction = 1 if ai_score > human_score else 0
        predictions.append(prediction)

    score = 0
    for i, val in enumerate(true_vals):
        if val == predictions[i]:
            score += 1

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
    # print(score/len(true_vals))
if __name__ == "__main__":
    main()