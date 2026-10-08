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

import configparser

config = configparser.ConfigParser()
config.read("../config.ini")

target_dataset = ""

def load_data_using_config(config, dataset_label: str):

    human_text_label = "human_text"
    ai_text_label = "ai_text"

    if config(dataset_label, "is_online_dataset"):
        dataset = load_dataset(config(dataset_label, "file_path"))["train"]
        if config(dataset_label, "human_and_ai_text_columns"):
            human_text_label = config(dataset_label, "human_column")
            ai_text_label = config(dataset_label, "ai_column")
        else:
            if config(dataset_label, "matching"):
                df = []
                
    else:
        if config(dataset_label, "file_path")[-5:] == "jsonl":
            df = pd.read_json(config(dataset_label, "file_path"), lines = True)
        else:
            try:
                df = pd.read_csv(config(dataset_label, "file_path"), lines = True)
            except:
                raise NotImplementedError
        if config(dataset_label, "human_and_ai_text_columns"):
            human_text_label = config(dataset_label, "human_column")
            ai_text_label = config(dataset_label, "ai_column")


    
    return human_text_label, ai_text_label

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

