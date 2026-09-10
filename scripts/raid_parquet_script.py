import logging
import gc
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def preprocess_raid_once(dataset):
    """
    One pass over RAID:
      - human_texts[source_id]
      - generations[(source_id, model, attack, decoding, repetition_penalty)]
      - config_counter[(model, attack, decoding, repetition_penalty)]
    """
    human_texts = {}
    generations = {}
    config_counter = Counter()

    logger.info("Preprocessing RAID in one pass")

    for row in dataset:
        source_id = row["source_id"]
        model = row["model"]
        attack = row["attack"]
        decoding = row["decoding"]
        repetition_penalty = row["repetition_penalty"]

        if model == "human":
            if attack == "none":
                human_texts[source_id] = {
                    "human_text": row["generation"],
                    "human_domain": row["domain"],
                    "title": row["title"],
                }
        else:
            key = (source_id, model, attack, decoding, repetition_penalty)
            generations[key] = {
                "ai_text": row["generation"],
                "ai_domain": row["domain"],
                "prompt": row["prompt"],
            }
            config_counter[(model, attack, decoding, repetition_penalty)] += 1

    logger.info("Unique human source_ids: %d", len(human_texts))
    logger.info("Unique generation entries: %d", len(generations))
    logger.info("Unique configs: %d", len(config_counter))

    return human_texts, generations, config_counter


def print_available_configs(config_counter):
    by_model = defaultdict(list)

    for (model, attack, decoding, repetition_penalty), count in config_counter.items():
        by_model[model].append((attack, decoding, repetition_penalty, count))

    for model in sorted(by_model):
        logger.info("Available %s configs:", model)
        for attack, decoding, repetition_penalty, count in sorted(by_model[model]):
            logger.info(
                "  (%s, %s, %s) -> %d",
                attack,
                decoding,
                repetition_penalty,
                count,
            )


def build_pairs_from_preprocessed(
    human_texts,
    generations,
    target_model: str,
    target_attack: str,
    target_decoding: str,
    target_repetition_penalty: str,
):
    paired_rows = []

    for source_id, human_info in human_texts.items():
        key = (
            source_id,
            target_model,
            target_attack,
            target_decoding,
            target_repetition_penalty,
        )

        ai_info = generations.get(key)
        if ai_info is None:
            continue

        paired_rows.append(
            {
                "source_id": source_id,
                "human_text": human_info["human_text"],
                "ai_text": ai_info["ai_text"],
                "human_domain": human_info.get("human_domain"),
                "ai_domain": ai_info.get("ai_domain"),
                "title": human_info.get("title"),
                "prompt": ai_info.get("prompt"),
                "model": target_model,
                "attack": target_attack,
                "decoding": target_decoding,
                "repetition_penalty": target_repetition_penalty,
            }
        )

    return pd.DataFrame(paired_rows)


def main():
    output_dir = Path("raid_pairs")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading RAID dataset")
    dataset = load_dataset("liamdugan/raid", "raid")["train"]

    human_texts, generations, config_counter = preprocess_raid_once(dataset)
    print_available_configs(config_counter)

    # No longer need the original dataset in memory
    del dataset
    gc.collect()

    first_df = True
    sorted_configs = sorted(config_counter.items())

    for (model, attack, decoding, repetition_penalty), count in sorted_configs:
        if model in ["chatgpt", "cohere", "cohere-chat", "gpt2", "gpt3", "gpt4"]:
            continue
        logger.info(
            "Building pairs for model=%s attack=%s decoding=%s rep=%s",
            model,
            attack,
            decoding,
            repetition_penalty,
        )

        paired_df = build_pairs_from_preprocessed(
            human_texts,
            generations,
            target_model=model,
            target_attack=attack,
            target_decoding=decoding,
            target_repetition_penalty=repetition_penalty,
        )

        logger.info("Number of pairs: %d", len(paired_df))

        if len(paired_df) == 0:
            logger.warning("No pairs found; skipping save")
            continue

        filename = (
            f"raid_pairs__model={model}"
            f"__attack={attack}"
            f"__decoding={decoding}"
            f"__rep={repetition_penalty}.parquet"
        )
        output_path = output_dir / filename

        paired_df.to_parquet(output_path, index=False)
        logger.info("Saved %s", output_path)
        
        # if first_df:
        #     full_df = paired_df
        # else:
        #     full_df = pd.concat([full_df, paired_df], ignore_index = False)
        del paired_df
        # gc.collect()

    logger.info("Finished extracting RAID pair files")


if __name__ == "__main__":
    main()