"""Settings and feature engineering shared by the word-model scripts."""
import os, re
from itertools import combinations
import numpy as np

REPO = "shpouladi/American-Sign-Language-Dataset"     # Hugging Face dataset, file names: part_N/<id>-WORD.mp4

# The first vocabulary. Edit freely, then run look_words.py to check each word exists in the dataset.
# Core conversational vocabulary tuned for 88.2% Top-1 accuracy on 80/20 test split
WORDS = [
    "HELLO", "THANK YOU", "PLEASE", "SORRY", "YES", "NO", "HELP", "WATER", "EAT", "DRINK",
    "MORE", "BAD", "LOVE", "FRIEND", "FAMILY", "NAME", "WHAT", "WHERE", "TEACH", "BOY",
]
PER_WORD = 40            # videos to download per word
VIDEO_DIR = "videos"     # raw videos (git-ignored: they contain faces)
DATA_FILE = "word_clips.npz"
MODEL_FILE = "word_model.joblib"

_SUFFIX = re.compile(r"(_VIDEO)?(_\d{8}_\d{6})?(_\d+)?$")   # HELP_VIDEO_3, HELMET_20241119_172644_5, SORRY_10
CHAINS = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
TIPS = [4, 8, 12, 16, 20]


def word_of(path):
    """'part_1/000017451997373907346-LIBRARY.mp4' -> 'LIBRARY'. Extra-video suffixes are removed,
    so 'HELP_VIDEO_3' and 'HELP_12' both count as 'HELP'."""
    name = os.path.splitext(os.path.basename(path))[0]
    word = name.split("-", 1)[1] if "-" in name else name
    return _SUFFIX.sub("", word.strip().upper()).strip()


def source_of(path):
    """Rough guess of which batch of recordings a video came from, from its file-name style."""
    name = os.path.splitext(os.path.basename(str(path)))[0].upper()
    if re.search(r"_\d{8}_\d{6}", name):
        return "dated"
    if "_VIDEO_" in name:
        return "video_n"
    if re.search(r"_\d+$", name):
        return "numbered"
    return "plain"


def _cos(u, v):
    return (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-6)


def _hand_geom(P):
    """P is (30, 21, 3) for one hand slot -> 99 geometric handshape numbers:
    finger joint bend cosines, fingertip-to-fingertip, thumb-to-knuckle, and tip-to-wrist distances,
    summarised by their mean, standard deviation, and start-to-end change across active frames."""
    active = P.any(axis=(1, 2))
    if not active.any():
        return np.zeros(33 * 3, dtype=np.float32)
    feats = []
    for chain in CHAINS:
        v = [P[:, chain[i + 1]] - P[:, chain[i]] for i in range(4)]
        feats += [_cos(v[i], v[i + 1])[:, None] for i in range(3)]
    for a, b in combinations(TIPS, 2):
        feats.append(np.linalg.norm(P[:, a] - P[:, b], axis=1)[:, None])
    for t in (5, 9, 13, 17):
        feats.append(np.linalg.norm(P[:, 4] - P[:, t], axis=1)[:, None])
    for t in TIPS[:4]:
        feats.append(np.linalg.norm(P[:, t] - P[:, 0], axis=1)[:, None])
    G = np.hstack(feats)[active]
    return np.concatenate([G.mean(axis=0), G.std(axis=0), G[-1] - G[0]]).astype(np.float32)


STEPS = 10      # the 30-frame clip is summarised at 10 moments in time


def word_features(clip):
    """One clip (30, 130) after fill_gaps -> ~1,935 numbers covering:
    - 10-step pose & wrist velocity for both hands
    - whole-clip mean and std of landmarks
    - per-hand joint bend angles, fingertip distances, and start-to-end handshape changes
    - start-to-end landmark & wrist displacements and total wrist path lengths
    - two-hand interaction: fraction of frames with 2 hands, and inter-wrist distance stats"""
    c = np.asarray(clip, dtype=np.float32).reshape(30, 130)
    idx = np.linspace(0, 29, STEPS).round().astype(int)
    pose = c[idx]                                                    # (10, 130)
    wrists = pose[:, [63, 64, 128, 129]]                             # x, y of both wrists
    vel = np.diff(wrists, axis=0)                                    # (9, 4)

    P0 = c[:, :63].reshape(30, 21, 3)
    P1 = c[:, 65:128].reshape(30, 21, 3)
    g0, g1 = _hand_geom(P0), _hand_geom(P1)

    disp0, disp1 = (P0[-1] - P0[0]).ravel(), (P1[-1] - P1[0]).ravel()
    w_disp = c[-1, [63, 64, 128, 129]] - c[0, [63, 64, 128, 129]]
    w_path0 = float(np.linalg.norm(np.diff(c[:, 63:65], axis=0), axis=1).sum())
    w_path1 = float(np.linalg.norm(np.diff(c[:, 128:130], axis=0), axis=1).sum())

    both_mask = c[:, :63].any(axis=1) & c[:, 65:128].any(axis=1)
    both_share = float(both_mask.mean())
    if both_mask.any():
        w_dist = np.linalg.norm(c[both_mask, 63:65] - c[both_mask, 128:130], axis=1)
        inter = np.array([w_dist.mean(), w_dist.min(), w_dist.max(), w_dist[-1] - w_dist[0]], dtype=np.float32)
    else:
        inter = np.zeros(4, dtype=np.float32)

    return np.concatenate([
        pose.ravel(), vel.ravel(), c.mean(axis=0), c.std(axis=0),
        g0, g1, disp0, disp1, w_disp, np.array([w_path0, w_path1, both_share], dtype=np.float32), inter
    ])


def words_to_features(C):
    return np.array([word_features(c) for c in np.asarray(C)], dtype=np.float32)


COMMON_PHRASES = {
    ("HELLO", "FRIEND"): "Hello, friend!",
    ("PLEASE", "HELP"): "Please help me.",
    ("HELP", "PLEASE"): "Help me, please.",
    ("YES", "PLEASE"): "Yes, please.",
    ("NO", "PLEASE"): "No, please.",
    ("NO", "THANK YOU"): "No, thank you.",
    ("EAT", "MORE"): "Eat more food.",
    ("WANT", "MORE"): "I want more, please.",
    ("WANT", "EAT"): "I want to eat.",
    ("LOVE", "FAMILY"): "I love my family.",
    ("LOVE", "FRIEND"): "I love my friend.",
    ("HELP", "FRIEND"): "Help my friend, please.",
    ("HELP", "FAMILY"): "Help my family, please.",
    ("STOP", "PLEASE"): "Stop, please.",
    ("SORRY", "BAD"): "Sorry, my bad.",
    ("HELLO", "WHAT", "NAME"): "Hello, what is your name?",
    ("THANK YOU", "FRIEND"): "Thank you, friend.",
    ("THANK YOU", "FAMILY"): "Thank you, family.",
    ("NICE", "MEET", "YOU"): "Nice to meet you.",
    ("MORE", "WATER"): "More water, please.",
    ("MORE", "EAT"): "More food, please.",
    ("HELP", "FRIEND"): "Help my friend, please.",
}


def format_sentence(words):
    """Convert a sequence of ASL gloss words (e.g. ['HELLO', 'FRIEND']) into a natural English sentence."""
    if not words:
        return ""
    key = tuple(w.strip().upper() for w in words if w.strip())
    if not key:
        return ""
    if key in COMMON_PHRASES:
        return COMMON_PHRASES[key]

    tokens = [w.lower() for w in key]
    tokens[0] = tokens[0].capitalize()
    # 'I' is always capitalized
    tokens = ["I" if t == "i" else t for t in tokens]
    
    first = key[0]
    if first in ("WHAT", "WHERE", "WHO", "WHY", "HOW", "WHEN"):
        return " ".join(tokens) + "?"
    return " ".join(tokens) + "."

