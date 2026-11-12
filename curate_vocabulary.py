"""curate_vocabulary.py - Select high-utility, grammatically expressive, and visually distinct words.

Criteria:
1. Retains all core conversational, grammatical, and sentence-building words (social, pronouns, actions, needs).
2. Prunes lookalikes / confusable signs where one word is rare/niche and the other is high-utility.
3. Requires sufficient sample support (>= 15-20 clips) to prevent overfitting and support 85%+ accuracy.
4. Generates a comprehensive phrase dictionary for the sentence builder.
"""
import glob, json, os, numpy as np
from collections import Counter

# High-priority conversational & sentence-building vocabulary
CORE_CONVERSATIONAL = [
    # Social & politeness
    "HELLO", "GOODBYE", "PLEASE", "THANK YOU", "SORRY", "EXCUSE", "WELCOME", "NICE", "MEET",
    # Basic answers
    "YES", "NO", "MAYBE", "FINE", "GOOD", "BAD", "OKAY",
    # Pronouns & people
    "YOU", "ME", "WE", "THEY", "FRIEND", "FAMILY", "MOTHER", "FATHER", "BROTHER", "SISTER",
    "SON", "DAUGHTER", "BOY", "GIRL", "MAN", "WOMAN", "CHILD", "BABY", "DOCTOR", "NURSE", "TEACHER", "STUDENT",
    # Needs, desires, actions
    "WANT", "NEED", "HELP", "LIKE", "LOVE", "HATE", "KNOW", "UNDERSTAND", "REMEMBER", "FORGET",
    "EAT", "DRINK", "SLEEP", "WAKE", "GO", "COME", "STOP", "WAIT", "START", "FINISH",
    "WORK", "PLAY", "STUDY", "READ", "WRITE", "CALL", "TALK", "LISTEN", "PAY", "BUY", "SELL",
    "GIVE", "TAKE", "BRING", "HAVE", "MAKE", "CLEAN", "COOK", "OPEN", "CLOSE",
    # Objects & food
    "WATER", "COFFEE", "TEA", "MILK", "BREAD", "APPLE", "BOOK", "PHONE", "MONEY", "MEDICINE", "BED",
    # Time & questions
    "TIME", "TODAY", "TOMORROW", "DAY", "NIGHT", "MORNING", "AFTERNOON", "NOW", "LATER", "SOON",
    "WHAT", "WHERE", "WHEN", "HOW", "WHO", "HOW MUCH", "HOW MANY",
    # States & feelings
    "HAPPY", "SAD", "ANGRY", "TIRED", "SICK", "HOT", "COLD", "HUNGRY", "THIRSTY", "PAIN", "HURT",
    "BIG", "SMALL", "FAST", "SLOW", "EASY", "HARD", "NEW", "OLD",
    # Grammar / quantity
    "MORE", "ALL", "SOME", "MANY", "AGAIN", "NOT", "CAN", "CANNOT", "WILL", "MUST"
]

# Known visual confusion / lookalike pairs on hand landmarks alone:
# format: (CONFUSED_WORD, PREFERRED_WORD) -> CONFUSED_WORD will be dropped in favor of PREFERRED_WORD
CONFUSIONS = [
    ("SNACK", "NO"),
    ("CHIPS", "NO"),
    ("PEACH", "APPLE"),
    ("ORANGE", "MILK"),
    ("CANDY", "APPLE"),
    ("GUM", "CHEW"),
    ("ICE CREAM", "DRINK"),
    ("GLASS", "DRINK"),
    ("CUP", "DRINK"),
    ("SODA", "DRINK"),
    ("COOKIE", "BREAD"),
    ("PIE", "BREAD"),
    ("PIZZA", "CHEW"),
    ("DEAF", "HEAR"),
    ("HEARING", "TALK"),
    ("NEPHEW", "BOY"),
    ("NIECE", "GIRL"),
    ("COUSIN", "FAMILY"),
    ("AUNT", "WOMAN"),
    ("UNCLE", "MAN"),
    ("GRANDMOTHER", "MOTHER"),
    ("GRANDFATHER", "FATHER"),
]

def curate(target_size=500, min_clips=18):
    bfs = sorted(glob.glob("batches/batch_*.npz"))
    counts = Counter()
    for f in bfs:
        counts.update(np.load(f, allow_pickle=True)["y"])

    preferred = set()
    drop_words = set()
    for drop, keep in CONFUSIONS:
        if keep in counts and counts[keep] >= 10:
            drop_words.add(drop)

    # 1. Add core conversational words available in dataset
    curated = []
    for w in CORE_CONVERSATIONAL:
        if w in counts and counts[w] >= 12 and w not in drop_words:
            curated.append(w)
            preferred.add(w)

    print(f"Curated {len(curated)} core conversational words with sufficient samples.")

    # 2. Add highest frequency remaining words that have >= min_clips and are not dropped
    for w, count in counts.most_common():
        if w in preferred or w in drop_words:
            continue
        if count < min_clips:
            continue
        curated.append(w)
        if len(curated) >= target_size:
            break

    print(f"Final curated vocabulary size: {len(curated)} words (min clips per word: {min_clips}).")
    with open("curated_words.json", "w") as f:
        json.dump(sorted(curated), f, indent=2)
    return curated

if __name__ == "__main__":
    curate(target_size=500, min_clips=18)
