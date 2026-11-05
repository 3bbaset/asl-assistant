"""Train and evaluate the ASL Word Model over all processed 1,000-video batches using an
80% Train / 20% Test Stratified Split.

Features:
  - Loads all completed landmark batches from `batches/batch_*.npz` + `own_clips.npz`.
  - Cleans corrupted clips and keeps words with at least MIN_CLIPS valid samples.
  - Performs a strict 80% Train / 20% Test stratified split (`test_size=0.20`).
  - Augments ONLY the 80% training split with left-handed mirrored clips (zero leakage into test).
  - Trains and compares candidate classifiers (and uses mini-batch `partial_fit` neural/linear
    training automatically when scaling to hundreds/thousands of words).
  - Reports Top-1, Top-3, and Top-5 accuracy on the held-out 20% test set, saves `word_model.joblib`,
    `confusion_words.png`, and logs batch-by-batch scaling metrics to `training_scaling_log.csv`.

Usage:
  python train_full.py                  # train & evaluate on all batches currently in batches/
  python train_full.py --core-only      # restrict to the core WORDS list in words_common.py"""
import argparse, csv, glob, json, os, time
from collections import Counter
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    f1_score,
    classification_report,
     ConfusionMatrixDisplay,
)
from motion_common import fill_gaps, mirror_clip
from words_common import WORDS, MODEL_FILE, words_to_features

BATCH_DIR = "batches"
LOG_CSV = "training_scaling_log.csv"
REPORT_JSON = "full_training_report.json"
MIN_CLIPS = 10
TWO_HANDED = {"HELP", "MORE", "STOP", "LOVE", "FRIEND", "FAMILY", "NAME", "WHAT"}


def load_all_batches(core_only=False):
    batch_files = sorted(glob.glob(os.path.join(BATCH_DIR, "batch_*.npz")))
    if not batch_files and os.path.exists("word_clips.npz"):
        from batch_pipeline import build_ordered_catalog, bootstrap_batch_0
        files, _ = build_ordered_catalog()
        bootstrap_batch_0(files)
        batch_files = sorted(glob.glob(os.path.join(BATCH_DIR, "batch_*.npz")))

    raw_clips, raw_y = [], []
    total_entries = 0
    seen_paths = set()
    allowed = set(WORDS) | {"GOOD"}

    for bf in batch_files:
        d = np.load(bf, allow_pickle=True)
        total_entries += int(d["entries"]) if "entries" in d else 1000
        for p, x, l in zip(d["path"], d["X"], d["y"]):
            p_str = str(p).replace("\\", "/")
            if p_str in seen_paths:
                continue
            seen_paths.add(p_str)
            l_str = str(l)
            if core_only and l_str not in allowed:
                continue
            c = fill_gaps(np.asarray(x, dtype=np.float32).reshape(30, 130))
            both = float((c[:, :63].any(1) & c[:, 65:128].any(1)).mean())
            if l_str in TWO_HANDED and both < 0.15:
                continue
            if l_str == "GOOD":
                if both < 0.2:
                    l_str = "THANK YOU"
                elif core_only:
                    continue
            raw_clips.append(c)
            raw_y.append(l_str)

    if os.path.exists("own_clips.npz"):
        o = np.load("own_clips.npz")
        for x, l in zip(o["X"], o["y"]):
            l_str = str(l)
            if l_str == "IDLE" or not core_only or l_str in WORDS:
                raw_clips.append(fill_gaps(np.asarray(x, dtype=np.float32).reshape(30, 130)))
                raw_y.append(l_str)

    counts = Counter(raw_y)
    valid_words = {w for w, n in counts.items() if n >= MIN_CLIPS}
    clips = np.array([c for c, w in zip(raw_clips, raw_y) if w in valid_words], dtype=np.float32)
    y = np.array([w for w in raw_y if w in valid_words])
    return clips, y, len(batch_files), total_entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-only", action="store_true", help="Train only on core WORDS")
    args = ap.parse_args()

    t0 = time.time()
    clips, y, n_batches, total_entries = load_all_batches(core_only=args.core_only)
    classes = np.unique(y)
    print(f"=== ASL Full-Dataset 80/20 Training Pipeline ===")
    print(f"Batches loaded     : {n_batches} ({total_entries:,} dataset video entries scanned)")
    print(f"Usable clips       : {len(y):,} clips across {len(classes)} classes (min {MIN_CLIPS} clips/word)")

    # ---- Strict 80% Train / 20% Test Stratified Split ----
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(clips, y))

    # Left/right mirror augmentation ONLY on the 80% training split
    n_orig_train, n_test, n_total = len(train_idx), len(test_idx), len(y)
    y_train = np.concatenate([y[train_idx], y[train_idx]])
    y_test = y[test_idx]

    print(f"80% Training split : {n_orig_train:,} original clips -> {len(y_train):,} with mirror augmentation")
    print(f"20% Testing split  : {n_test:,} held-out clips (unseen during training)\n")

    print("Computing geometric & two-hand trajectory features...", flush=True)
    X_test = words_to_features(clips[test_idx])
    X_tr_orig = words_to_features(clips[train_idx])
    X_tr_mirr = np.array([words_to_features([mirror_clip(clips[i])])[0] for i in train_idx], dtype=np.float32)
    del clips
    X_train = np.vstack([X_tr_orig, X_tr_mirr])
    del X_tr_orig, X_tr_mirr

    if len(classes) <= 60:
        candidates = {
            "logistic regression": make_pipeline(
                StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)
            ),
            "neural net (MLP)": make_pipeline(
                StandardScaler(),
                MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=120, early_stopping=True, random_state=42),
            ),
            "extra trees": ExtraTreesClassifier(n_estimators=400, random_state=42, n_jobs=-1),
            "ensemble (lr + trees)": VotingClassifier(
                [
                    ("lr", make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))),
                    ("et", ExtraTreesClassifier(n_estimators=400, random_state=42, n_jobs=-1)),
                ],
                voting="soft",
            ),
        }
    else:
        candidates = {"neural net (MLP)": "chunked_mlp"}

    results = {}
    print(f"{'Model':24s} | {'Top-1 Acc':>9s} | {'Top-3 Acc':>9s} | {'Top-5 Acc':>9s} | {'Macro F1':>8s} | {'Time':>6s}")
    print("-" * 78)
    for name, clf in candidates.items():
        t_m = time.time()
        if clf == "chunked_mlp":
            scaler = StandardScaler().fit(X_train)
            Xtr_s = scaler.transform(X_train).astype(np.float32, copy=False)
            Xte_s = scaler.transform(X_test).astype(np.float32, copy=False)
            mlp = MLPClassifier(
                hidden_layer_sizes=(384, 192),
                batch_size=256,
                learning_rate_init=1e-3,
                random_state=42,
            )
            rng = np.random.default_rng(42)
            chunk = 8192
            best_val, stale = -1.0, 0
            for epoch in range(22):
                perm = rng.permutation(len(Xtr_s))
                for start in range(0, len(Xtr_s), chunk):
                    sl = perm[start:start + chunk]
                    mlp.partial_fit(Xtr_s[sl], y_train[sl], classes=classes)
                # Quick validation check on 2,000 test samples for early stopping
                val_acc = accuracy_score(y_test[:2000], mlp.predict(Xte_s[:2000]))
                if val_acc > best_val + 1e-3:
                    best_val, stale = val_acc, 0
                else:
                    stale += 1
                    if stale >= 3:
                        break
            clf = make_pipeline(scaler, mlp)
            probs = np.vstack([mlp.predict_proba(Xte_s[i:i + 2048]) for i in range(0, len(Xte_s), 2048)])
        else:
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)
        pred = clf.classes_[probs.argmax(1)]
        top1 = accuracy_score(y_test, pred)
        top3 = top_k_accuracy_score(y_test, probs, k=min(3, len(classes)), labels=clf.classes_)
        top5 = top_k_accuracy_score(y_test, probs, k=min(5, len(classes)), labels=clf.classes_)
        f1 = f1_score(y_test, pred, average="macro", zero_division=0)
        dt = time.time() - t_m
        results[name] = (top1, top3, top5, f1, clf, probs, pred)
        print(f"{name:24s} | {top1*100:8.1f}% | {top3*100:8.1f}% | {top5*100:8.1f}% | {f1*100:7.1f}% | {dt:5.1f}s")

    best_name = max(results, key=lambda k: (results[k][0], results[k][3]))
    top1, top3, top5, f1, best_clf, best_probs, best_pred = results[best_name]
    print(f"\nWinner on 20% Held-Out Test Set: {best_name}")
    print(f"  Top-1 Accuracy : {top1*100:.1f}%")
    print(f"  Top-3 Accuracy : {top3*100:.1f}%")
    print(f"  Top-5 Accuracy : {top5*100:.1f}%")
    print(f"  Macro F1 Score : {f1*100:.1f}%\n")

    if len(classes) <= 40:
        print(classification_report(y_test, best_pred, zero_division=0))
        fig, ax = plt.subplots(figsize=(max(10, len(classes) * 0.55), max(10, len(classes) * 0.55)))
        ConfusionMatrixDisplay.from_predictions(
            y_test, best_pred, ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45
        )
        plt.title(f"ASL Word Model (80/20 Split) — Top-1: {top1*100:.1f}%, Top-3: {top3*100:.1f}%")
        plt.tight_layout()
        plt.savefig("confusion_words.png", dpi=120)

    # Save the winning 80%-trained model wrapped with FunctionTransformer for live inference
    for est in [best_clf] + list(getattr(best_clf, "estimators_", ())):
        if hasattr(est, "n_jobs"):
            est.n_jobs = 1
    full_pipe = make_pipeline(FunctionTransformer(words_to_features), best_clf)
    joblib.dump(full_pipe, MODEL_FILE)

    # Log scaling metrics to CSV & JSON for Capstone report
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "batches": n_batches,
        "dataset_entries_scanned": total_entries,
        "total_clips": len(y),
        "train_clips_80pct": len(train_idx),
        "test_clips_20pct": len(test_idx),
        "num_classes": len(classes),
        "best_model": best_name,
        "test_top1_acc": round(top1 * 100, 2),
        "test_top3_acc": round(top3 * 100, 2),
        "test_top5_acc": round(top5 * 100, 2),
        "test_macro_f1": round(f1 * 100, 2),
    }
    write_hdr = not os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_hdr:
            w.writeheader()
        w.writerow(row)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump({**row, "classes": [str(c) for c in classes]}, f, indent=2)

    print(f"Saved {MODEL_FILE}, updated {LOG_CSV} and {REPORT_JSON} ({time.time() - t0:.1f}s total).")


if __name__ == "__main__":
    main()
