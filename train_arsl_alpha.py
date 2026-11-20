"""train_arsl_alpha.py - Train 32-class Arabic Sign Language (ArSL) Alphabet model from arbml/ArSL21L.

Pipeline:
  1. Downloads arbml/ArSL21L parquet shards from Hugging Face into data_arsl/arsl21l/.
  2. Extracts 21x3 MediaPipe hand landmarks (multi-process, cached to data_arsl/arsl_alpha_landmarks.npz).
  3. Performs stratified 80/20 split, augments training split only (mirror + 3D tilt/stretch/jitter).
  4. Benchmarks RandomForest, HistGradientBoosting, and MLPClassifier on the held-out 20% test set.
  5. Saves arsl_alpha_model.joblib, reports/arsl_alpha_report.json, and reports/confusion_arsl_alpha.png.
"""
import io
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import mediapipe as mp  # import before pyarrow on Windows to avoid DLL init conflict
import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from common import detect, flat_to_features, make_detector, normalize, renorm

ARSL21L_CLASSES = [
    "ain", "al", "aleff", "bb", "dal", "dha", "dhad", "fa",
    "gaaf", "ghain", "ha", "haa", "jeem", "kaaf", "khaa", "la",
    "laam", "meem", "nun", "ra", "saad", "seen", "sheen", "ta",
    "taa", "thaa", "thal", "toot", "waw", "ya", "yaa", "zay",
]

DATA_DIR = "data_arsl"
CACHE_FILE = os.path.join(DATA_DIR, "arsl_alpha_landmarks.npz")
REPORT_DIR = "reports"
MODEL_OUT = "arsl_alpha_model.joblib"

_DET = None


def _init_worker():
    global _DET
    _DET = make_detector(min_conf=0.25)


def _extract_chunk(items):
    """Worker: list of (img_bytes, label_str) -> list of (feat63, label_str)."""
    out = []
    for raw_bytes, lbl in items:
        try:
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            rgb = np.asarray(img, dtype=np.uint8)
            lm = detect(_DET, rgb)
            if lm is None:
                # Upscale once if small or cropped tightly
                h, w = rgb.shape[:2]
                if max(h, w) < 320:
                    rgb2 = cv2.resize(rgb, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
                    lm = detect(_DET, rgb2)
            if lm is not None:
                out.append((normalize(lm).astype(np.float32).ravel(), lbl))
        except Exception:
            continue
    return out


def download_and_extract():
    os.makedirs(os.path.join(DATA_DIR, "arsl21l"), exist_ok=True)
    if os.path.exists(CACHE_FILE):
        d = np.load(CACHE_FILE, allow_pickle=True)
        print(f"[ArSL-Alpha] Loaded cached landmarks: {len(d['X']):,} samples across {len(np.unique(d['y']))} classes")
        return d["X"].astype(np.float32), d["y"].astype(str), int(d["total_images"])

    shards = ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"]
    tasks = []
    class_names = ARSL21L_CLASSES
    for shard in shards:
        print(f"[ArSL-Alpha] Downloading {shard} from arbml/ArSL21L ...", flush=True)
        path = hf_hub_download(
            repo_id="arbml/ArSL21L",
            filename=shard,
            repo_type="dataset",
            local_dir=os.path.join(DATA_DIR, "arsl21l"),
        )
        tbl = pq.read_table(path)
        # Check HF schema metadata if embedded
        meta = tbl.schema.metadata or {}
        if b"huggingface" in meta:
            try:
                hf_meta = json.loads(meta[b"huggingface"].decode("utf-8"))
                names = hf_meta["info"]["features"]["label"]["names"]
                if isinstance(names, list) and len(names) == 32:
                    class_names = names
            except Exception:
                pass
        imgs = tbl["image"].to_pylist()
        lbls = tbl["label"].to_pylist()
        for im, lb in zip(imgs, lbls):
            if isinstance(im, dict) and im.get("bytes"):
                lbl_str = class_names[int(lb)] if isinstance(lb, int) and 0 <= int(lb) < len(class_names) else str(lb)
                tasks.append((im["bytes"], lbl_str))

    total_images = len(tasks)
    print(f"[ArSL-Alpha] Extracting MediaPipe hand landmarks from {total_images:,} images ...", flush=True)
    chunk_size = 128
    chunks = [tasks[i:i + chunk_size] for i in range(0, total_images, chunk_size)]
    t0 = time.time()
    X_list, y_list = [], []
    with ProcessPoolExecutor(max_workers=6, initializer=_init_worker) as pool:
        for done, res in enumerate(pool.map(_extract_chunk, chunks), 1):
            for feat, lbl in res:
                X_list.append(feat)
                y_list.append(lbl)
            if done % 20 == 0 or done == len(chunks):
                print(f"  chunk {done}/{len(chunks)} -> {len(X_list):,} hands detected ({time.time() - t0:.1f}s)", flush=True)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    np.savez_compressed(CACHE_FILE, X=X, y=y, total_images=np.array(total_images))
    print(f"[ArSL-Alpha] Saved {CACHE_FILE}: {len(X):,}/{total_images:,} images detected ({len(X)/max(1,total_images)*100:.1f}%)")
    return X, y, total_images


def _rotation_matrices(ang):
    n = len(ang)
    cx, cy, cz = np.cos(ang).T
    sx, sy, sz = np.sin(ang).T
    Rx, Ry, Rz = np.zeros((n, 3, 3)), np.zeros((n, 3, 3)), np.zeros((n, 3, 3))
    Rx[:, 0, 0] = 1; Rx[:, 1, 1] = cx; Rx[:, 1, 2] = -sx; Rx[:, 2, 1] = sx; Rx[:, 2, 2] = cx
    Ry[:, 1, 1] = 1; Ry[:, 0, 0] = cy; Ry[:, 0, 2] = sy; Ry[:, 2, 0] = -sy; Ry[:, 2, 2] = cy
    Rz[:, 2, 2] = 1; Rz[:, 0, 0] = cz; Rz[:, 0, 1] = -sz; Rz[:, 1, 0] = sz; Rz[:, 1, 1] = cz
    return Rz @ Ry @ Rx


def augment(P, rng):
    n = len(P)
    ang = np.radians(np.column_stack([
        rng.uniform(-12, 12, n),
        rng.uniform(-12, 12, n),
        rng.uniform(-20, 20, n),
    ]))
    Q = np.einsum("nij,npj->npi", _rotation_matrices(ang), P)
    Q[:, :, 0] *= rng.uniform(0.9, 1.1, (n, 1))
    Q += rng.normal(0, 0.01, Q.shape)
    return renorm(Q)


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    X, y, total_images = download_and_extract()
    classes = sorted(np.unique(y).tolist())

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_i, test_i = next(splitter.split(X, y))

    rng = np.random.default_rng(42)
    P0 = X[train_i].reshape(-1, 21, 3)
    Pm = P0.copy()
    Pm[:, :, 0] *= -1.0
    P_aug = np.concatenate([P0, Pm, augment(P0, rng), augment(Pm, rng)])
    y_aug = np.tile(y[train_i], 4)

    X_aug = flat_to_features(P_aug.reshape(len(P_aug), -1)).astype(np.float32)
    X_test = flat_to_features(X[test_i]).astype(np.float32)
    y_test = y[test_i]
    print(f"[ArSL-Alpha] Split -> train={len(train_i):,} (aug={len(X_aug):,}), test={len(test_i):,}, classes={len(classes)}")

    candidates = [
        ("RandomForest", RandomForestClassifier(
            n_estimators=200, max_depth=24, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=42
        )),
        ("HistGradientBoosting", HistGradientBoostingClassifier(
            max_iter=150, class_weight="balanced", random_state=42
        )),
        ("Neural Net (MLP)", make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=80,
                          early_stopping=True, n_iter_no_change=6, random_state=42)
        )),
    ]

    comparison = []
    best_score = -1.0
    best_bundle = None

    for name, clf in candidates:
        t0 = time.time()
        clf.fit(X_aug, y_aug)
        train_s = round(time.time() - t0, 2)
        t1 = time.time()
        probs = clf.predict_proba(X_test)
        inf_ms = round((time.time() - t1) * 1000.0 / max(1, len(X_test)), 4)
        pred = clf.classes_[probs.argmax(1)]
        top1 = accuracy_score(y_test, pred) * 100.0
        top3 = top_k_accuracy_score(y_test, probs, k=min(3, len(classes)), labels=clf.classes_) * 100.0
        macro_f1 = f1_score(y_test, pred, average="macro") * 100.0
        size_mb = round(len(pickle.dumps(clf)) / 1e6, 2)
        print(f"  {name:24s} Top-1={top1:5.2f}%  Top-3={top3:5.2f}%  MacroF1={macro_f1:5.2f}%  inf={inf_ms:.3f}ms  size={size_mb}MB")
        comparison.append({
            "model": name,
            "top1_acc": round(top1, 2),
            "top3_acc": round(top3, 2),
            "macro_f1": round(macro_f1, 2),
            "train_sec": train_s,
            "inference_ms": inf_ms,
            "size_mb": size_mb,
        })
        if macro_f1 > best_score:
            best_score = macro_f1
            best_bundle = (name, clf, probs, pred, top1, top3, macro_f1, inf_ms)

    best_name, best_clf, probs, pred, top1, top3, macro_f1, inf_ms = best_bundle
    creport = classification_report(y_test, pred, labels=classes, output_dict=True, zero_division=0)
    per_class = []
    for c in classes:
        r = creport.get(c, {})
        per_class.append({
            "label": c,
            "precision": round(r.get("precision", 0.0) * 100.0, 2),
            "recall": round(r.get("recall", 0.0) * 100.0, 2),
            "f1": round(r.get("f1-score", 0.0) * 100.0, 2),
            "support": int(r.get("support", 0)),
        })

    cm = confusion_matrix(y_test, pred, labels=classes)
    pairs = sorted(
        ((int(cm[i, j]), classes[i], classes[j])
         for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j] > 0),
        reverse=True,
    )[:10]

    # Save pipeline taking raw 63-float normalized landmarks directly
    ft = FunctionTransformer(flat_to_features).fit(X[:1])
    pipe = make_pipeline(ft, best_clf)
    joblib.dump(pipe, MODEL_OUT, compress=3)
    model_mb = round(os.path.getsize(MODEL_OUT) / 1e6, 2)

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(13, 13))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=65
    )
    ax.set_title(f"ArSL 32-Letter Confusion Matrix ({best_name} - {top1:.1f}% Top-1)")
    plt.tight_layout()
    cm_path = os.path.join(REPORT_DIR, "confusion_arsl_alpha.png")
    plt.savefig(cm_path, dpi=140)
    plt.close(fig)

    report = {
        "model_id": "arsl_alpha",
        "title": "ArSL Alphabet (32 Letters)",
        "language": "ArSL",
        "dataset": "arbml/ArSL21L",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_images_scanned": int(total_images),
        "hands_detected": int(len(X)),
        "detection_rate_pct": round(len(X) / max(1, total_images) * 100.0, 2),
        "train_samples": int(len(train_i)),
        "train_augmented_samples": int(len(X_aug)),
        "test_samples": int(len(test_i)),
        "num_classes": len(classes),
        "best_model": best_name,
        "test_top1_acc": round(top1, 2),
        "test_top3_acc": round(top3, 2),
        "test_top5_acc": round(top_k_accuracy_score(y_test, probs, k=min(5, len(classes)), labels=best_clf.classes_) * 100.0, 2),
        "test_macro_f1": round(macro_f1, 2),
        "inference_ms": inf_ms,
        "model_size_mb": model_mb,
        "model_file": MODEL_OUT,
        "classes": classes,
        "comparison": comparison,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "top_confused": [{"count": n, "true": a, "pred": b} for n, a, b in pairs],
    }
    rep_path = os.path.join(REPORT_DIR, "arsl_alpha_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[ArSL-Alpha] Saved {MODEL_OUT} ({model_mb} MB), {rep_path}, and {cm_path}")


if __name__ == "__main__":
    main()
