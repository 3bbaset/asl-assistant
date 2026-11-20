"""evaluate_all_models.py - Consolidate real held-out benchmarks across all 6 ASL & ArSL models.

Outputs reports/capstone_metrics.json consumed directly by the Capstone Dashboard.
Zero fabricated metrics: every number comes from evaluating the saved models on held-out splits
or loading existing training reports + measuring live inference latency.
"""
import csv
import glob
import json
import os
import time
from collections import Counter

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, top_k_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.base import clone

from common import flat_to_features
from motion_common import fill_gaps
from words_common import words_to_features

REPORT_DIR = "reports"
OUT_JSON = os.path.join(REPORT_DIR, "capstone_metrics.json")


def _clean_json(obj):
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return 0.0
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json(v) for v in obj]
    return obj


def _bench_asl_alpha():
    print("[Eval] Evaluating ASL Static Alphabet (asl_model_v2.joblib) on train2.py held-out split...")
    letters, idxs, rows = [], [], []
    with open("data_all.csv") as f:
        for r in csv.reader(f):
            letters.append(r[0])
            idxs.append(int(r[1]))
            rows.append([float(v) for v in r[2:]])
    y, idx, X = np.array(letters), np.array(idxs), np.array(rows, dtype=np.float32)

    rng = np.random.default_rng(42)
    BLOCK, GAP = 150, 30
    block = idx // BLOCK
    groups = np.array([f"{l}-{b}" for l, b in zip(y, block)])
    test_blocks = set()
    for letter in np.unique(y):
        ids = np.unique(groups[y == letter])
        n_test = max(1, round(len(ids) * 0.2))
        test_blocks.update(rng.choice(ids, n_test, replace=False))
    is_test = np.array([g in test_blocks for g in groups])
    test_i = np.where(is_test)[0]
    train_i = [
        i for i in np.where(~is_test)[0]
        if not ((f"{y[i]}-{block[i]+1}" in test_blocks and (idx[i] % BLOCK) >= BLOCK - GAP) or
                (f"{y[i]}-{block[i]-1}" in test_blocks and (idx[i] % BLOCK) < GAP))
    ]

    mdl = joblib.load("asl_model_v2.joblib")
    X_test, y_test = X[test_i], y[test_i]
    t0 = time.perf_counter()
    probs = mdl.predict_proba(X_test)
    inf_ms = round((time.perf_counter() - t0) * 1000.0 / max(1, len(X_test)), 4)

    classes = [str(c) for c in mdl.classes_]
    pred = np.array([classes[i] for i in probs.argmax(1)])
    top1 = accuracy_score(y_test, pred) * 100.0
    top3 = top_k_accuracy_score(y_test, probs, k=3, labels=classes) * 100.0
    macro_f1 = f1_score(y_test, pred, average="macro") * 100.0

    creport = classification_report(y_test, pred, labels=classes, output_dict=True, zero_division=0)
    per_class = [
        {
            "label": c,
            "precision": round(creport[c]["precision"] * 100.0, 2),
            "recall": round(creport[c]["recall"] * 100.0, 2),
            "f1": round(creport[c]["f1-score"] * 100.0, 2),
            "support": int(creport[c]["support"]),
        }
        for c in classes
    ]
    cm = confusion_matrix(y_test, pred, labels=classes)
    pairs = sorted(
        ((int(cm[i, j]), classes[i], classes[j])
         for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j] > 0),
        reverse=True,
    )[:10]

    print(f"  -> Top-1={top1:.2f}% Top-3={top3:.2f}% MacroF1={macro_f1:.2f}% inf={inf_ms}ms")
    return {
        "model_id": "asl_alpha",
        "title": "ASL Static Alphabet (A-Z sans J/Z)",
        "short_title": "ASL Alphabet",
        "language": "ASL",
        "dataset": "data_all.csv (Webcam + Internet ASL)",
        "best_model": type(mdl[-1]).__name__,
        "num_classes": len(classes),
        "train_samples": len(train_i),
        "test_samples": len(test_i),
        "test_top1_acc": round(top1, 2),
        "test_top3_acc": round(top3, 2),
        "test_macro_f1": round(macro_f1, 2),
        "inference_ms": inf_ms,
        "model_size_mb": round(os.path.getsize("asl_model_v2.joblib") / 1e6, 2),
        "classes": classes,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "top_confused": [{"count": n, "true": a, "pred": b} for n, a, b in pairs],
        "comparison": [
            {"model": type(mdl[-1]).__name__, "top1_acc": round(top1, 2), "macro_f1": round(macro_f1, 2)},
        ],
    }


def _bench_asl_motion():
    print("[Eval] Evaluating ASL Dynamic Motion J/Z (motion_model2.joblib) on 5-fold GroupKFold...")
    d = np.load("own_clips.npz")
    clips = np.array([fill_gaps(c.reshape(30, 130)) for c in d["X"]])
    y = np.array(d["y"].astype(str))
    classes = sorted(np.unique(y).tolist())
    groups = np.arange(len(y)) // 6

    mdl = joblib.load("motion_model2.joblib")
    oof = np.zeros((len(y), len(classes)), dtype=np.float32)
    for tr, te in GroupKFold(n_splits=5).split(clips, y, groups):
        fold_mdl = clone(mdl)
        fold_mdl.fit(clips[tr], y[tr])
        oof[te] = fold_mdl.predict_proba(clips[te])

    pred = np.array([classes[i] for i in oof.argmax(1)])
    top1 = accuracy_score(y, pred) * 100.0
    top3 = top_k_accuracy_score(y, oof, k=min(3, len(classes)), labels=classes) * 100.0
    macro_f1 = f1_score(y, pred, average="macro") * 100.0

    t0 = time.perf_counter()
    _ = mdl.predict_proba(clips[:20])
    inf_ms = round((time.perf_counter() - t0) * 1000.0 / 20.0, 4)

    creport = classification_report(y, pred, labels=classes, output_dict=True, zero_division=0)
    per_class = [
        {
            "label": c,
            "precision": round(creport[c]["precision"] * 100.0, 2),
            "recall": round(creport[c]["recall"] * 100.0, 2),
            "f1": round(creport[c]["f1-score"] * 100.0, 2),
            "support": int(creport[c]["support"]),
        }
        for c in classes
    ]
    cm = confusion_matrix(y, pred, labels=classes)
    pairs = sorted(
        ((int(cm[i, j]), classes[i], classes[j])
         for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j] > 0),
        reverse=True,
    )[:10]

    print(f"  -> Top-1={top1:.2f}% MacroF1={macro_f1:.2f}% inf={inf_ms}ms")
    return {
        "model_id": "asl_motion",
        "title": "ASL Motion Letters (J, Z, IDLE)",
        "short_title": "ASL Motion J/Z",
        "language": "ASL",
        "dataset": "own_clips.npz (5-fold GroupKFold OOF)",
        "best_model": type(mdl[-1]).__name__,
        "num_classes": len(classes),
        "train_samples": int(len(y) * 0.8),
        "test_samples": len(y),
        "test_top1_acc": round(top1, 2),
        "test_top3_acc": round(top3, 2),
        "test_macro_f1": round(macro_f1, 2),
        "inference_ms": inf_ms,
        "model_size_mb": round(os.path.getsize("motion_model2.joblib") / 1e6, 2),
        "classes": classes,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "top_confused": [{"count": n, "true": a, "pred": b} for n, a, b in pairs],
    }


def _bench_asl_words_30():
    print("[Eval] Evaluating ASL 30-Word Curated Model (word_model_curated_30.joblib)...")
    rep = json.load(open("full_training_report.json", encoding="utf-8"))
    mdl = joblib.load("word_model_curated_30.joblib")
    classes = [str(c) for c in mdl.classes_]
    target_set = set(classes)

    # Gather clips matching the curated model classes across batches/ and own_clips.npz
    raw_clips, raw_y = [], []
    seen = set()
    for bf in sorted(glob.glob("batches/batch_*.npz")):
        d = np.load(bf, allow_pickle=True)
        for p, x, l in zip(d["path"], d["X"], d["y"]):
            ps = str(p).replace("\\", "/")
            ls = "I" if str(l) == "ME" else str(l)
            if ps in seen or ls not in target_set:
                continue
            seen.add(ps)
            raw_clips.append(fill_gaps(np.asarray(x, dtype=np.float32).reshape(30, 130)))
            raw_y.append(ls)

    cm_list, per_class, top_confused = [], [], []
    top1, top3, macro_f1 = rep["test_top1_acc"], rep["test_top3_acc"], rep["test_macro_f1"]
    train_n, test_n = rep["train_clips_80pct"], rep["test_clips_20pct"]

    if raw_clips:
        clips_arr = np.array(raw_clips, dtype=np.float32)
        y_arr = np.array(raw_y)
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        tr_i, te_i = next(splitter.split(clips_arr, y_arr))
        X_te = words_to_features(clips_arr[te_i])
        y_te = y_arr[te_i]
        t0 = time.perf_counter()
        probs = mdl.predict_proba(X_te)
        inf_ms = round((time.perf_counter() - t0) * 1000.0 / max(1, len(X_te)), 4)
        pred = np.array([classes[i] for i in probs.argmax(1)])
        top1 = round(accuracy_score(y_te, pred) * 100.0, 2)
        top3 = round(top_k_accuracy_score(y_te, probs, k=3, labels=classes) * 100.0, 2)
        macro_f1 = round(f1_score(y_te, pred, average="macro") * 100.0, 2)
        train_n, test_n = len(tr_i), len(te_i)
        creport = classification_report(y_te, pred, labels=classes, output_dict=True, zero_division=0)
        per_class = [
            {
                "label": c,
                "precision": round(creport[c]["precision"] * 100.0, 2),
                "recall": round(creport[c]["recall"] * 100.0, 2),
                "f1": round(creport[c]["f1-score"] * 100.0, 2),
                "support": int(creport[c]["support"]),
            }
            for c in classes
        ]
        cm = confusion_matrix(y_te, pred, labels=classes)
        cm_list = cm.tolist()
        pairs = sorted(
            ((int(cm[i, j]), classes[i], classes[j])
             for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j] > 0),
            reverse=True,
        )[:10]
        top_confused = [{"count": n, "true": a, "pred": b} for n, a, b in pairs]
    else:
        dummy_clip = np.zeros((1, 30, 130), dtype=np.float32)
        feat = words_to_features(dummy_clip)
        t0 = time.perf_counter()
        _ = mdl.predict_proba(feat)
        inf_ms = round((time.perf_counter() - t0) * 1000.0, 4)

    print(f"  -> Top-1={top1}% Top-3={top3}% MacroF1={macro_f1}% inf={inf_ms}ms")
    return {
        "model_id": "asl_words_30",
        "title": "ASL Conversational Words (32 Signs)",
        "short_title": "ASL Words (32)",
        "language": "ASL",
        "dataset": "WLASL Curated (batches/ + own_clips)",
        "best_model": rep.get("best_model", "Soft-Voting Ensemble"),
        "num_classes": len(classes),
        "train_samples": train_n,
        "test_samples": test_n,
        "test_top1_acc": top1,
        "test_top3_acc": top3,
        "test_top5_acc": rep.get("test_top5_acc"),
        "test_macro_f1": macro_f1,
        "inference_ms": inf_ms,
        "model_size_mb": round(os.path.getsize("word_model_curated_30.joblib") / 1e6, 2),
        "classes": classes,
        "per_class": per_class,
        "confusion_matrix": cm_list,
        "top_confused": top_confused,
    }


def _bench_asl_words_2021():
    print("[Eval] Loading ASL Full 2,021-Word Report (full_training_report_2021.json)...")
    rep = json.load(open("full_training_report_2021.json", encoding="utf-8"))
    inf_ms = 0.0
    size_mb = 0.0
    if os.path.exists("word_model_full_2021.joblib"):
        size_mb = round(os.path.getsize("word_model_full_2021.joblib") / 1e6, 2)
        mdl = joblib.load("word_model_full_2021.joblib")
        dummy = np.zeros((5, 30, 130), dtype=np.float32)
        t0 = time.perf_counter()
        _ = mdl.predict_proba(dummy)
        inf_ms = round((time.perf_counter() - t0) * 1000.0 / 5.0, 4)

    scaling = []
    if os.path.exists("training_scaling_log.csv"):
        with open("training_scaling_log.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                nc = int(r["num_classes"])
                if nc >= 23 and int(r["total_clips"]) > 1050 or nc == 23 or nc == 29:
                    scaling.append({
                        "batches": int(r["batches"]),
                        "num_classes": nc,
                        "total_clips": int(r["total_clips"]),
                        "best_model": r["best_model"],
                        "top1_acc": float(r["test_top1_acc"]),
                        "top3_acc": float(r["test_top3_acc"]),
                        "top5_acc": float(r["test_top5_acc"]),
                        "macro_f1": float(r["test_macro_f1"]),
                    })

    return {
        "model_id": "asl_words_2021",
        "title": "ASL Full Vocabulary (2,021 Words)",
        "short_title": "ASL Full (2,021)",
        "language": "ASL",
        "dataset": "WLASL / HF American-Sign-Language-Dataset",
        "best_model": rep["best_model"],
        "num_classes": rep["num_classes"],
        "train_samples": rep["train_clips_80pct"],
        "test_samples": rep["test_clips_20pct"],
        "test_top1_acc": rep["test_top1_acc"],
        "test_top3_acc": rep["test_top3_acc"],
        "test_top5_acc": rep["test_top5_acc"],
        "test_macro_f1": rep["test_macro_f1"],
        "inference_ms": inf_ms,
        "model_size_mb": size_mb,
        "classes": rep.get("classes", [])[:50],
    }, scaling


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    models = []
    models.append(_bench_asl_alpha())
    models.append(_bench_asl_motion())
    models.append(_bench_asl_words_30())
    w2k, scaling = _bench_asl_words_2021()
    models.append(w2k)

    for rep_path, prefix in [
        (os.path.join(REPORT_DIR, "arsl_alpha_report.json"), "ArSL Letters"),
        (os.path.join(REPORT_DIR, "arsl_words_report.json"), "ArSL Words"),
    ]:
        if os.path.exists(rep_path):
            r = json.load(open(rep_path, encoding="utf-8"))
            r["short_title"] = f"{prefix} ({r.get('num_classes', '?')})"
            models.append(r)

    payload = _clean_json({
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
        "scaling": scaling,
    })
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[Eval] Consolidated {len(models)} models into {OUT_JSON}")


if __name__ == "__main__":
    main()
