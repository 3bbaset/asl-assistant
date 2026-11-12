"""train_curated.py - High-accuracy training on curated, conversational, distinct ASL vocabulary.

Tests multiple vocabularies (e.g. 50, 100, 200, 300 words) using 80/20 stratified split,
trajectory & geometric feature engineering, and high-performance gradient boosting / neural ensembles
to reach the target >=85% accuracy.
"""
import glob, json, os, time
from collections import Counter
import joblib
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report
import lightgbm as lgb
from motion_common import fill_gaps, mirror_clip
from words_common import words_to_features

def load_curated_data(vocab_file="curated_words.json", max_words=100):
    with open(vocab_file) as f:
        target_words = set(json.load(f)[:max_words])

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
            if l_str in target_words:
                c = fill_gaps(np.asarray(x, dtype=np.float32).reshape(30, 130))
                raw_clips.append(c)
                raw_y.append(l_str)

    if os.path.exists("own_clips.npz"):
        o = np.load("own_clips.npz")
        for x, l in zip(o["X"], o["y"]):
            l_str = str(l)
            if l_str in target_words or l_str == "IDLE":
                raw_clips.append(fill_gaps(np.asarray(x, dtype=np.float32).reshape(30, 130)))
                raw_y.append(l_str)

    counts = Counter(raw_y)
    valid_words = {w for w, n in counts.items() if n >= 15}
    clips = np.array([c for c, w in zip(raw_clips, raw_y) if w in valid_words], dtype=np.float32)
    y = np.array([w for w in raw_y if w in valid_words])
    return clips, y, sorted(valid_words)

def evaluate_vocab(n_words=100):
    print(f"\n=======================================================")
    print(f"  Training Curated Vocabulary (Target: {n_words} words)")
    print(f"=======================================================")
    clips, y, classes = load_curated_data(max_words=n_words)
    print(f"Loaded {len(y):,} clips across {len(classes)} classes")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(clips, y))

    y_train = np.concatenate([y[train_idx], y[train_idx]])
    y_test = y[test_idx]

    print("Computing geometric & trajectory features...", flush=True)
    X_test = words_to_features(clips[test_idx])
    X_tr_orig = words_to_features(clips[train_idx])
    X_tr_mirr = np.array([words_to_features([mirror_clip(clips[i])])[0] for i in train_idx], dtype=np.float32)
    X_train = np.vstack([X_tr_orig, X_tr_mirr])

    scaler = StandardScaler().fit(X_train)
    X_tr_s = scaler.transform(X_train)
    X_te_s = scaler.transform(X_test)

    # Label encoding for LightGBM
    label_to_id = {c: i for i, c in enumerate(classes)}
    y_tr_ids = np.array([label_to_id[c] for c in y_train])
    y_te_ids = np.array([label_to_id[c] for c in y_test])

    print("Training LightGBM classifier on GPU/Multi-core...", flush=True)
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.08,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    t0 = time.time()
    clf.fit(X_tr_s, y_tr_ids)
    dur = time.time() - t0

    probs = clf.predict_proba(X_te_s)
    top1 = accuracy_score(y_te_ids, probs.argmax(1))
    top3 = top_k_accuracy_score(y_te_ids, probs, k=min(3, len(classes)), labels=list(range(len(classes))))
    top5 = top_k_accuracy_score(y_te_ids, probs, k=min(5, len(classes)), labels=list(range(len(classes))))

    print(f"Results for {len(classes)} words (Trained in {dur:.1f}s):")
    print(f"  Top-1 Accuracy: {top1 * 100:.2f}%")
    print(f"  Top-3 Accuracy: {top3 * 100:.2f}%")
    print(f"  Top-5 Accuracy: {top5 * 100:.2f}%")
    return top1, top3, len(classes), clf, scaler, classes

if __name__ == "__main__":
    for n in (40, 70, 100):
        t1, t3, k, clf, sc, cls = evaluate_vocab(n)
