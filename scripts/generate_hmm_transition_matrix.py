import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
import numpy as np

import re

TOKEN_PATTERN = re.compile(
    r"[a-z]+(?:'[a-z]+)?|[0-9]+(?:\.[0-9]+)?|[^\w\s]",
    re.IGNORECASE
)

def tokenise(text: str) -> list[str]:
    if not text:
        return []

    text = re.sub(r"\s+", " ", text.lower().strip())
    tokens = TOKEN_PATTERN.findall(text)
    return ["<START>"] + tokens + ["<END>"] if tokens else []

def main():
    rng = np.random.default_rng()
    """Main feature extraction pipeline."""
    # logger.info("Starting feature extraction from HC3 dataset")

    # # Load HC3 dataset from local cache
    # logger.info("Loading HC3 dataset...")
    data_dir = Path(__file__).parent.parent / "data" / "all.jsonl"
    # dataset = load_dataset(str(data_dir))  # nosec B615
    df = pd.read_json(str(data_dir), lines = True)
    print(df.columns)

    data_inds = [i for i in range(len(df["question"]))]
    texts = []
    human_texts = []
    human_inds = []
    human_i = 0
    chatgpt_texts = []
    chatgpt_inds = []
    chatgpt_i = 0
    labels = []
    for text in df["human_answers"]:
        human_text = "".join(text)
        # print(type(human_text))
        # print(str(human_text))
        # print(human_text)
        if human_text and len(human_text.strip()) > 50:  # Filter too-short samples
            human_inds.append(human_i)
            human_i += 1
            human_texts.append(human_text)
            texts.append(human_text)
            labels.append(0)  
    
    for text in df["chatgpt_answers"]:
        chatgpt_text = "".join(text)
        if chatgpt_text and len(chatgpt_text.strip()) > 50:  # Filter too-short samples
            chatgpt_inds.append(chatgpt_i)
            chatgpt_i += 1
            chatgpt_texts.append(chatgpt_text)
            texts.append(chatgpt_text)
            labels.append(1)

    training_proportion = 0.01
    print(chatgpt_i)
    print(human_i)
    human_training_size = int(human_i*training_proportion)
    human_training_size = int(1000)
    human_training_inds = np.random.choice(human_inds, size = human_training_size, replace = False)
    chatgpt_training_size = int(1000)
    chatgpt_training_inds = np.random.choice(chatgpt_inds, size = chatgpt_training_size, replace = False)

    human_token_seqs = []
    for text in human_texts:
        tokenised_text = tokenise(text)
        human_token_seqs.append(tokenised_text)
    # human_token_seqs = np.array(human_token_seqs)
    chatgpt_token_seqs = []
    for text in chatgpt_texts:
        tokenised_text = tokenise(text)
        chatgpt_token_seqs.append(tokenised_text)

    human_transition_frequencies = {}
    human_transition_count = 0
    for j, ind in enumerate(human_training_inds):
        print(j, end = "\r")
        tokenised_seq = human_token_seqs[ind]
        for i, token_1 in enumerate(tokenised_seq[:-1]):
            if (token_1, tokenised_seq[i + 1]) in list(human_transition_frequencies.keys()):
                human_transition_frequencies[(token_1, tokenised_seq[i + 1])] += 1
                human_transition_count += 1
            else:
                human_transition_frequencies[(token_1, tokenised_seq[i + 1])] = 1
                human_transition_count += 1
    # print(human_transition_frequencies.items())
    # print(list(human_transition_frequencies.items())[0])
    # print(list(human_transition_frequencies.items())[0][1])
    sorted_human_transition_frequencies = sorted(human_transition_frequencies.items(), key = lambda item: item[1], reverse = True)
    # print(sorted_human_transition_frequencies[:100])
    # flop
    
    chatgpt_transition_frequencies = {}
    chatgpt_transition_count = 0
    for j, ind in enumerate(chatgpt_training_inds):
        print(j, end = "\r")
        tokenised_seq = chatgpt_token_seqs[ind]
        for i, token_1 in enumerate(tokenised_seq[:-1]):
            if (token_1, tokenised_seq[i + 1]) in list(chatgpt_transition_frequencies.keys()):
                chatgpt_transition_frequencies[(token_1, tokenised_seq[i + 1])] += 1
                chatgpt_transition_count += 1 
            else:
                chatgpt_transition_frequencies[(token_1, tokenised_seq[i + 1])] = 1
                chatgpt_transition_count += 1 

    sorted_chatgpt_transition_frequencies = sorted(chatgpt_transition_frequencies.items(), key = lambda item: item[1], reverse= True)
    # print(sorted_chatgpt_transition_frequencies[:100])
    print(human_transition_count)
    print(chatgpt_transition_count)
    output = []
    for i in range(300):
        output.append([i, sorted_human_transition_frequencies[i][0], sorted_human_transition_frequencies[i][1], sorted_chatgpt_transition_frequencies[i][0], sorted_chatgpt_transition_frequencies[i][1]])
    
    df_output = pd.DataFrame(output)
    df_output =  df_output.rename(columns = {
        0:"frequency_ranking",
        1:"human_transition",
        2:"human_transition_count",
        3:"ai_transition",
        4:"ai_transition_count"
    })
    df_output.to_csv("../data/transition_counts.csv", index = False)


if __name__ == "__main__":
    main()
