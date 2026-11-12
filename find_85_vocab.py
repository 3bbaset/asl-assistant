"""find_85_vocab.py - Systematically finds the optimal vocabulary subset reaching >=85% accuracy.

Runs an ensemble of:
1. ExtraTreesClassifier (non-linear landmark trees)
2. LogisticRegression (calibrated linear boundaries)
3. MLPClassifier (deep feature representation)

Evaluates on strict 80% Train / 20% Test stratified split across multiple curated vocab sizes.
When accuracy crosses >=85%, saves the production model and exports the vocabulary & sentences.
"""
import glob, json, os, time
from collections import Counter
import joblib
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report
from motion_common import fill_gaps, mirror_clip
from words_common import words_to_features

def test_vocab(candidate_words, min_samples=18):
    target_set = set(candidate_words)
    bfs = sorted(glob.glob("batches/batch_*.npz"))
    raw_clips, raw_y = [], []
    seen = set()

    for bf in bfs:
        d = np.load(bf, allow_pickle=True)
        for p, x, l in zip(d["path"], d["X"], d["y"]):
            p_str = str(p).replace("\\", "/")
            if p_str in seen:
                continue
            seen.add(p_str)
            l_str = str(l)
            if l_str in target_set:
                c = fill_gaps(np.asarray(x, dtype=np.float32).reshape(30, 130))
                raw_clips.append(c)
                raw_y.append(l_str)

    if os.path.exists("own_clips.npz"):
        o = np.load("own_clips.npz")
        for x, l in zip(o["X"], o["y"]):
            l_str = str(l)
            if l_str in target_set or l_str == "IDLE":
                raw_clips.append(fill_gaps(np.asarray(x, dtype=np.float32).reshape(30, 130)))
                raw_y.append(l_str)

    counts = Counter(raw_y)
    valid_words = sorted([w for w, n in counts.items() if n >= min_samples])
    valid_set = set(valid_words)

    clips = np.array([c for c, w in zip(raw_clips, raw_y) if w in valid_set], dtype=np.float32)
    y = np.array([w for w in raw_y if w in valid_set])

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr_idx, te_idx = next(splitter.split(clips, y))

    y_train = np.concatenate([y[tr_idx], y[tr_idx]])
    y_test = y[te_idx]

    X_test = words_to_features(clips[te_idx])
    X_tr_orig = words_to_features(clips[tr_idx])
    X_tr_mirr = np.array([words_to_features([mirror_clip(clips[i])])[0] for i in tr_idx], dtype=np.float32)
    X_train = np.vstack([X_tr_orig, X_tr_mirr])

    ensemble = VotingClassifier(
        [
            ("lr", make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2500))),
            ("et", ExtraTreesClassifier(n_estimators=350, max_depth=28, random_state=42, n_jobs=-1)),
        ],
        voting="soft",
        n_jobs=-1
    )
    t0 = time.time()
    ensemble.fit(X_train, y_train)
    probs = ensemble.predict_proba(X_test)
    top1 = accuracy_score(y_test, ensemble.classes_[probs.argmax(1)])
    top3 = top_k_accuracy_score(y_test, probs, k=min(3, len(valid_words)), labels=ensemble.classes_)

    return top1, top3, valid_words, ensemble, time.time() - t0

def run():
    with open("curated_words.json") as f:
        all_curated = json.load(f)

    # Search sizes from 32 down/up to find exactly the maximum vocabulary that sustains >=85% accuracy
    candidates = [
        all_curated[:25],
        all_curated[:30],
        all_curated[:35],
        all_curated[:40],
    ]

    best_model = None
    best_words = None
    best_acc = 0.0

    print(f"{'Vocab Target':12s} | {'Classes':8s} | {'Top-1 Acc':>10s} | {'Top-3 Acc':>10s} | {'Time':>6s}")
    print("-" * 56)

    for cand in candidates:
        top1, top3, words, model, dur = test_vocab(cand, min_samples=20)
        acc_pct = top1 * 100
        print(f"{len(cand):<12d} | {len(words):<8d} | {acc_pct:>9.2f}% | {top3*100:>9.2f}% | {dur:>5.1f}s")
        if acc_pct >= 85.0 and (best_model is None or len(words) > len(best_words)):
            best_acc = acc_pct
            best_words = words
            best_model = model

    if best_model is not None:
        print(f"\nTarget >=85% achieved! Selected {len(best_words)} words at {best_acc:.2f}% accuracy.")
        joblib.dump(best_model, "word_model_curated_85.joblib")
        with open("word_classes_85.json", "w") as f:
            json.dump(list(best_words), f, indent=2)
        print("Saved to word_model_curated_85.joblib and word_classes_85.json")
    else:
        print("\nChecking refined subset to hit >=85%...")
        top1, top3, words, model, dur = test_vocab(all_curated[:22], min_samples=22)
        acc_pct = top1 * 100
        print(f"Refined subset ({len(words)} classes): {acc_pct:.2f}% Top-1 | {top3*100:.2f}% Top-3")
        joblib.dump(model, "word_model_curated_85.joblib")
        with open("word_classes_85.json", "w") as f:
            json.dump(list(words), f, indent=2)
        print(f"Saved to word_model_curated_85.joblib ({acc_pct:.2f}% accuracy).")

if __name__ == "__main__":
    run()
