import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, os
import pickle as pk
import json
from collections import Counter
import math
import re

START_TOKEN = "<START>"
END_TOKEN = "<END>"
RARE_TOKEN = "<RARE>"
NUM_TOKEN = "<NUM>"
URL_TOKEN = "<URL>"

TOKEN_PATTERN = re.compile(
    r"[a-z]+(?:'[a-z]+)?|[0-9]+(?:\.[0-9]+)?|[^\w\s]",
    re.IGNORECASE,
)

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

def sequence_log_likelihood(
    tokens: list[str],
    transition_counter: Counter[tuple[str, str]],
    row_totals: dict[str, int],
    vocab: set[str],
    alpha: float = 1.0,
    normalize_by_length: bool = True,
):
    if len(tokens) < 2:
        return 0.0, []

    log_prob = 0.0
    p_contributions = []
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
        p_contributions.append(math.log(p))

    if normalize_by_length:
        return log_prob / (len(tokens) - 1), p_contributions

    return log_prob, p_contributions

test_text_1 = """Technology has transformed nearly every part of modern life. It affects how people communicate, learn, work, travel, shop, and spend their free time. From smartphones and computers to medical devices and artificial intelligence, technology has created opportunities that were once unimaginable. At the same time, it has introduced new problems related to privacy, dependence, health, and human relationships. Because of this, the effects of technology are both positive and negative, and understanding both sides is important.

One of the most positive effects of technology is improved communication. People can now connect instantly across long distances through text messages, video calls, email, and social media. Families separated by geography can remain close, and businesses can operate globally with ease. Technology has also made access to information far easier. Students can research topics in seconds, people can learn new skills online, and news can spread around the world almost immediately. In education, online tools, digital resources, and virtual classes have expanded learning opportunities for many individuals.

Technology has also improved health and safety in significant ways. Medical technology has made it possible to detect illnesses earlier, perform advanced surgeries, and monitor patients more effectively. Devices such as heart monitors, hearing aids, and mobility tools have improved quality of life for millions of people. In transportation, technology has contributed to safer vehicles, GPS systems, and faster emergency response. These innovations show how technology can directly support human well-being.

Another major advantage is convenience. Daily tasks that once required much more time and effort can now be completed quickly. People can shop online, manage bank accounts, pay bills, navigate unfamiliar places, and work remotely with digital tools. Businesses can organize data, automate routine tasks, and reach customers more efficiently. For many people, technology saves time and increases productivity.

However, technology also has negative effects. One major concern is overdependence. Many people rely so heavily on phones and digital platforms that they struggle to disconnect. Constant notifications and screen time can reduce attention span and make it harder to focus on deep thinking or face-to-face interaction. Some individuals feel anxious when separated from their devices, which suggests that technology can sometimes control people more than people control technology.

Technology can also weaken personal relationships. Although digital communication makes connection easier, it does not always make it deeper. People may spend more time online than with those physically around them. Social media can create the illusion of connection while actually increasing loneliness, comparison, and insecurity. Instead of building genuine communication, it can encourage people to present idealized versions of themselves. This may damage self-esteem, especially among young people.

Another serious issue is privacy and security. Personal information is often stored online, making it vulnerable to hacking, identity theft, or misuse by companies and governments. Many people do not fully realize how much of their data is being collected through apps, websites, and devices. Technology has also made it easier to spread misinformation, scams, and harmful content. These problems show that increased access and convenience often come with hidden risks.

Technology can also affect physical and mental health. Long hours on screens may lead to eye strain, poor posture, sleep problems, and less physical activity. Excessive use of social media has been linked to stress, anxiety, and depression in some cases. Children and teenagers may be especially vulnerable because they are still developing socially and emotionally. When technology replaces exercise, outdoor play, or face-to-face interaction, it can negatively affect overall well-being.

In conclusion, technology has had a profound effect on people’s lives. It has improved communication, education, health care, safety, and convenience. At the same time, it has created challenges related to dependence, privacy, relationships, and health. Technology itself is not simply good or bad; its impact depends on how people use it. If used wisely and in balance, technology can greatly improve life. If used carelessly or excessively, it can create serious problems. A responsible approach is therefore essential in a world increasingly shaped by technology."""
test_text_2 = """Technology plays a major role in everyday life. People use it to communicate, study, work, travel, shop, and entertain themselves. In many ways, technology has made life easier and more efficient. At the same time, it has created problems that affect health, relationships, and privacy. Because of this, technology should be viewed as something that brings both benefits and drawbacks rather than something that is entirely good or entirely bad.

One of the clearest benefits of technology is communication. People can contact family, friends, coworkers, and teachers almost instantly through phones, messaging apps, email, and video calls. This allows relationships to continue across great distances. Technology also gives people quick access to information. A student can research a topic in minutes, and a worker can learn new skills online without having to attend in-person classes. These changes have made education and communication much more accessible.

Technology has also helped improve health and daily convenience. Medical equipment, digital records, and advanced treatments have made it easier to diagnose and treat illness. People can now schedule appointments online, use fitness trackers, and access health advice more easily than before. In daily life, technology lets people pay bills, shop, use maps, and handle banking from home. Thats one reason many people feel that modern technology saves both time and effort.

Even so, technology creates serious problems too. One concern is that people can become too dependent on it. Many individuals spend hours each day on phones, tablets, or computers. Constant screen use can make it harder to focus, rest, or spend time with others in person. Some people struggle to go even a short time without checking messages or social media. This shows that convenience can turn into dependence very quickly.

Technology also affects relationships. While online communication makes it easier to stay in touch, it does not always create deeper connection. People may spend more time looking at screens than talking face to face. Social media can add pressure by encouraging comparison and unrealistic self-presentation. Instead of helping people feel included, it can leave them feeling lonely or insecure. Young people, especially, may be affected by this constant need for approval online.

Another major issue is privacy and safety. Many websites and apps collect personal information, often without users fully understanding how much data is being stored or shared. Hackers, scams, and misinformation have also become more common because digital systems can be exploited. These problems show that the same tools that make life easier can also put people at risk.

Physical and mental health can suffer as well. Too much screen time can lead to poor sleep, eye strain, reduced physical activity, and posture problems. Heavy use of social media may also contribute to anxiety or low self-esteem in some users. If people replace exercise, outdoor time, or real conversation with constant technology use, the effects may become harmful over time.

In conclusion, technology has changed people’s lives in powerful ways. It has improved communication, education, health care, and convenience. Yet it has also brought dependence, weaker social interaction, privacy concerns, and health problems. Technology can be useful and even life-changing, but it needs to be used with balance and awareness. The real issue is not whether technology exists, but how wisely people choose to use it."""
test_text_3 = """Technology has had a profound impact on the way people live. It shapes communication, education, health care, work, entertainment, and everyday routines. Because of its broad influence, technology has become both incredibly helpful and, in some cases, harmful. It offers convenience, access, and innovation, yet it also creates challenges involving privacy, dependence, and human well-being. For this reason, the effects of technology should be examined from both the positive and negative sides.

One major positive effect of technology is improved communication. Smartphones, video calls, email, and messaging platforms allow people to stay connected across great distances. Families can maintain close contact, businesses can operate internationally, and students can engage with teachers and classmates more easily. Technology has also made access to information much faster. By providing search engines, online libraries, and learning platforms, it has opened the door to a wealth of knowledge that would otherwise be much harder to reach.

Technology has also led to meaningful progress in health and safety. Medical devices now help doctors diagnose illness earlier and treat conditions more effectively. People can monitor their health with digital tools, schedule care online, and benefit from improved hospital systems. In transportation, technology has contributed to safer vehicles, navigation systems, and faster emergency response. These advances show that technology can support both survival and quality of life.

Another advantage is convenience. Many tasks that once required time, travel, or paperwork can now be completed in minutes. People can shop, pay bills, work remotely, and manage finances through digital systems. In turn, this can create more efficiency in daily life. For businesses and individuals alike, technology is likely to increase productivity and reduce certain barriers of distance and time.

However, technology also has drawbacks. One of the biggest is overdependence. People may become so attached to devices that they struggle to focus, rest, or interact without screens. Constant notifications and digital stimulation can reduce attention span and make deep concentration more difficult. A person who spends large parts of the day online may lose sight of how much technology is controlling daily behavior.

Technology can also be detrimental to relationships. Although people are more connected than ever, online interaction does not always create genuine closeness. Social media may encourage users to compare themselves to idealized images of others, leading to insecurity and loneliness. Face-to-face conversation can decrease when screens dominate daily life. As a result, some relationships become shallower even while communication becomes more frequent.

Another serious issue involves privacy and security. Personal information is often collected, stored, and shared through apps, websites, and devices. Many users do not realize how much data they are giving away. At the same time, hacking, identity theft, and misinformation have become more common. These risks show that technological convenience often comes with hidden consequences.

Technology can also affect physical and mental health. Long hours on devices may lead to poor posture, sleep disruption, eye strain, and less physical activity. Social media use, especially when excessive, can increase stress and lower self-esteem. Younger people may be especially vulnerable because they are still developing habits, confidence, and emotional balance. If done without limits, technology use can weaken both body and mind.

Ultimately, technology has changed human life in major ways, bringing both impressive benefits and serious drawbacks. It has improved communication, health care, safety, and convenience. At the same time, it has created problems related to dependence, privacy, relationships, and health. The most reasonable conclusion is that technology is neither entirely beneficial nor entirely harmful. Its value depends on how it is used. People should strive to use technology wisely so that it improves life without taking control of it."""

def main() -> None:

    # transition_rates = pk.load("../data/transition_artifacts_dmitva_0.pk")
    with open("../data/transition_artifacts_dmitva_0.pkl", "rb") as f:
        transition_counts = pk.load(f)

    with open("../data/vocab_dmitva_0.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
        vocab = vocab["vocab"]

    alpha = 1e-10

    # print(transition_counts.keys())
    # # print(help(transition_counts["human_transition_counts"]))
    # # print(transition_counts[""])
    # print(transition_counts["human_transition_counts"][('<START>', '<RARE>')])
    
    # print(transition_counts["ai_transition_counts"][('<START>', '<RARE>')])
    # print(transition_counts["ai_row_totals"]["<RARE>"]/transition_counts["ai_total"])

    # # print(vocab)
    # human_cumulative_sum = 0
    # ai_cumulative_sum = 0
    # human_self_transitions = []
    # ai_self_transitions = []
    # for word in vocab:
    #     try:
    #         human_self_transitions.append(transition_counts["human_transition_counts"][(word, word)]/transition_counts["human_row_totals"][word])
    #         human_cumulative_sum += human_self_transitions[-1]
    #         if human_self_transitions[-1] > 0.05:
    #             print("human ", word)
    #     except:
    #         # print(word)
    #         a = 1
    #     try:
    #         ai_self_transitions.append(transition_counts["ai_transition_counts"][(word, word)]/transition_counts["ai_row_totals"][word])
    #         ai_cumulative_sum += ai_self_transitions[-1]
    #         if ai_self_transitions[-1] > 0.05:
    #             print("ai ",word)
    #     except:
    #         a = 1
            
    # human_vocab_dict = {}
    # for word in vocab:
    #     try:
    #         tranisition_count = transition_counts["human_row_totals"][word]
    #     except:
    #         tranisition_count = 0

    #     human_vocab_dict[word]  = tranisition_count

    # sorted_human_vocab_dict = dict(sorted(human_vocab_dict.items(), key = lambda item: item[1], reverse = True))

    # human_preference_sorted_vocab = list(sorted_human_vocab_dict.keys())

    # human_transition_probability_matrix = []
    # ai_transition_probability_matrix = []
    # diff_mat = []
    # curr_biggest_diff = 0
    # # print(human_preference_sorted_vocab[:10])
    # for i, curr_word in enumerate(human_preference_sorted_vocab[:100]):
    #     # print(i/len(human_preference_sorted_vocab), end = "\r")
    #     temp_row_human = []
    #     temp_row_ai = []
    #     temp_row_diff = []
    #     for j, next_word in enumerate(human_preference_sorted_vocab[:100]):
    #         human_prob = smoothed_transition_probability(
    #             curr_word,
    #             next_word,
    #             transition_counts["human_transition_counts"],
    #             transition_counts["human_row_totals"],
    #             vocab,
    #             alpha,
    #         )

    #         ai_prob = smoothed_transition_probability(
    #             curr_word,
    #             next_word,
    #             transition_counts["ai_transition_counts"],
    #             transition_counts["ai_row_totals"],
    #             vocab,
    #             alpha,
    #         )
    #         if abs(np.log(human_prob/ai_prob)) > curr_biggest_diff:
    #             curr_biggest_diff = abs(np.log(human_prob/ai_prob))
    #             print(curr_word, next_word, ai_prob, human_prob)
    #         temp_row_diff.append(np.log(human_prob/ai_prob))
    #         temp_row_human.append(np.log(human_prob))
    #         temp_row_ai.append(np.log((ai_prob)))
    #     human_transition_probability_matrix.append(temp_row_human)
    #     ai_transition_probability_matrix.append(temp_row_ai)
    #     diff_mat.append(temp_row_diff)
    # # for row in human_transition_probability_matrix:
    #     # print(row)

    # # print("\n\n")
    # # for row in ai_transition_probability_matrix:
    # #     print(row)

    # human_transition_probability_matrix = np.array(human_transition_probability_matrix)
    # ai_transition_probability_matrix = np.array(ai_transition_probability_matrix)
    # log_odds_results = compute_transition_log_odds(
    #     human_transition_counts=transition_counts["human_transition_counts"],
    #     human_row_totals=transition_counts["human_row_totals"],
    #     ai_transition_counts=transition_counts["ai_transition_counts"],
    #     ai_row_totals=transition_counts["ai_row_totals"],
    #     vocab=vocab,
    #     alpha=alpha,
    #     min_total_count=10,
    # )

    # top_ai_favored = sorted(
    #     log_odds_results,
    #     key=lambda x: x["log_odds_ai_vs_human"],
    #     reverse=True,
    # )[:150]

    # top_human_favored = sorted(
    #     log_odds_results,
    #     key=lambda x: x["log_odds_ai_vs_human"],
    # )[:150]

    
    # print("Top 50 human favoured transitions:")
    # for row in top_human_favored:
    #     transition = row["transition"]
    #     human_count = row["human_count"]
    #     ai_count = row["ai_count"]
    #     score = row["log_odds_ai_vs_human"]

    #     print(
    #         "%s | log_odds=%.4f | ai_count=%d | human_count=%d",
    #         transition,
    #         score,
    #         ai_count,
    #         human_count,
    #     )

    # diff_mat = np.array(diff_mat)
    # print(human_transition_probability_matrix)
    # print(np.max(human_transition_probability_matrix))
    # print(np.min(diff_mat))
    # print(np.max(diff_mat))
    # plt.matshow(human_transition_probability_matrix)
    # plt.savefig("../human_transmat.png")
    
    # plt.matshow(ai_transition_probability_matrix)
    # plt.savefig("../ai_transmat.png")
    # plt.matshow(diff_mat)
    # plt.savefig("../diff_mat.png")

    # with open("../data/top_human_transitions.txt", "w") as f:
    #     for row in top_human_favored:
    #         transition = row["transition"]
    #         f.write(f"From {transition[0]} to {transition[1]}\n")

    # with open("../data/top_ai_transitions.txt", "w") as f:
    #     for row in top_ai_favored:
    #         transition = row["transition"]
    #         f.write(f"From {transition[0]} to {transition[1]}\n")
    # print(sorted_human_vocab_dict)
    # print(sorted(human_self_transitions, reverse=True))
    # print(sorted(ai_self_transitions, reverse=True))
    test_tokens_1 = map_sequence(tokenise(test_text_1), vocab)
    test_tokens_2 = map_sequence(tokenise(test_text_2), vocab)
    test_tokens_3 = map_sequence(tokenise(test_text_3), vocab)

    tokens = [test_tokens_1, test_tokens_2, test_tokens_3]
    for tokens in tokens:
        human_score, human_contributions = sequence_log_likelihood(
            tokens,
            transition_counts["human_transition_counts"],
            transition_counts["human_row_totals"],
            vocab,
            alpha=alpha,
        )
        ai_score, ai_contributions = sequence_log_likelihood(
            tokens,
            transition_counts["ai_transition_counts"],
            transition_counts["ai_row_totals"],
            vocab,
            alpha=alpha,
        )
        margin = ai_score - human_score
        prediction = 1 if ai_score > human_score else 0
        print(prediction)
        print(margin)
    return

if __name__ == "__main__":
    main()