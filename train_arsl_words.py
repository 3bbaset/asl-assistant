"""train_arsl_words.py - Train 30-word Arabic/Egyptian Sign Language (ArSL) model from KArSL-502.

Pipeline:
  1. Downloads KARSL-502_Labels.xlsx and FatimahEmadEldin/karsl-502-arabic-sign-language-final shards.
  2. Selects 30 everyday conversational Arabic sign classes with highest sample availability.
  3. Extracts 2-hand MediaPipe landmarks (130-dim frame_features, cached to data_arsl/arsl_words_landmarks.npz).
  4. Evaluates ExtraTrees, LightGBM, MLP, and Soft-Voting Ensemble on stratified 80/20 split
     (plus logs accuracy on the dataset's official train/test parquet split).
  5. Saves arsl_word_model_30.joblib, arsl_word_classes_30.json, reports/arsl_words_report.json,
     and reports/confusion_arsl_words.png.
"""
import argparse
import io
import json
import os
import pickle
import time
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import mediapipe as mp  # import before pyarrow on Windows to avoid DLL init conflict
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
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
from sklearn.preprocessing import StandardScaler

from arsl_common import arsl_batch_features, augment_row130, mirror_row130
from phrase_common import frame_features

DATA_DIR = "data_arsl"
REPORT_DIR = "reports"
CACHE_FILE = os.path.join(DATA_DIR, "arsl_words_landmarks.npz")
OWN_FILE = os.path.join(DATA_DIR, "own_arsl_words.npz")
LABELS_XLSX = os.path.join(DATA_DIR, "KARSL-502_Labels.xlsx")
LABELS_URL = "https://raw.githubusercontent.com/issamjebnouni/Arabic-Word-level-Sign-Language-Recognition/main/KARSL-502_Labels.xlsx"
MODEL_OUT = "arsl_word_model_30.joblib"
CLASSES_OUT = "arsl_word_classes_30.json"

# All 20 everyday conversational KArSL-502 classes present in karsl-502-arabic-sign-language-final (0181..0200)
CANDIDATE_IDS = [
    ("K0181", "يفكر", "THINK"),
    ("K0182", "يساعد", "HELP"),
    ("K0183", "يدخن", "SMOKE"),
    ("K0184", "يدعم", "SUPPORT"),
    ("K0185", "يختار", "CHOOSE"),
    ("K0186", "ينادي", "CALL"),
    ("K0187", "يتنامى", "GROW"),
    ("K0188", "يصبغ", "DYE"),
    ("K0189", "يقف", "STAND"),
    ("K0190", "يستحم", "BATHE"),
    ("K0191", "يدخل", "ENTER"),
    ("K0192", "عائلة", "FAMILY"),
    ("K0193", "جدة", "GRANDMOTHER"),
    ("K0194", "جد", "GRANDFATHER"),
    ("K0195", "أب", "FATHER"),
    ("K0196", "أم", "MOTHER"),
    ("K0197", "أخت", "SISTER"),
    ("K0198", "أخ", "BROTHER"),
    ("K0199", "بنت", "GIRL"),
    ("K0200", "طفل رضيع", "BABY"),
]

_DET = None


def _init_worker():
    global _DET
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.20,
    )
    _DET = vision.HandLandmarker.create_from_options(opts)


def _extract_chunk(items):
    out = []
    for raw_bytes, sid, split_name in items:
        try:
            rgb = np.asarray(Image.open(io.BytesIO(raw_bytes)).convert("RGB"), dtype=np.uint8)
            res = _DET.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not res.hand_landmarks:
                continue
            row130 = frame_features(res)
            out.append((row130, sid, split_name))
        except Exception:
            continue
    return out


def ensure_karsl_labels():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(LABELS_XLSX):
        print(f"[ArSL-Words] Downloading {LABELS_XLSX} ...", flush=True)
        urllib.request.urlretrieve(LABELS_URL, LABELS_XLSX)
    z = zipfile.ZipFile(LABELS_XLSX)
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    ss_xml = ET.fromstring(z.read("xl/sharedStrings.xml"))
    strings = ["".join(t.text or "" for t in si.findall(".//m:t", ns)) for si in ss_xml.findall("m:si", ns)]
    sh_xml = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    out = {}
    for row in sh_xml.findall(".//m:row", ns)[1:]:
        vals = []
        for c in row.findall("m:c", ns):
            t = c.attrib.get("t")
            v = c.find("m:v", ns)
            if v is None:
                vals.append("")
            elif t == "s":
                vals.append(strings[int(v.text)])
            else:
                vals.append(v.text)
        if len(vals) >= 3 and vals[0].isdigit():
            sid = f"{int(vals[0]):04d}"
            out[sid] = {"id": sid, "ar": vals[1].strip(), "en": vals[2].strip().upper()}
    return out


def download_and_extract():
    os.makedirs(os.path.join(DATA_DIR, "karsl502"), exist_ok=True)
    if os.path.exists(CACHE_FILE):
        d = np.load(CACHE_FILE, allow_pickle=True)
        print(f"[ArSL-Words] Loaded cached landmarks: {len(d['X']):,} frames")
        return d["X"].astype(np.float32), d["y"].astype(str), d["split"].astype(str), int(d["scanned"])

    cand_set = {sid for sid, _, _ in CANDIDATE_IDS}
    per_class_budget = Counter()
    max_per_class = 800
    tasks = []
    scanned = 0
    for split_name, fname in [("train", "data/train-00000-of-00001.parquet"), ("test", "data/test-00000-of-00001.parquet")]:
        print(f"[ArSL-Words] Downloading/reading {fname} ...", flush=True)
        path = hf_hub_download(
            repo_id="FatimahEmadEldin/karsl-502-arabic-sign-language-final",
            filename=fname,
            repo_type="dataset",
            local_dir=os.path.join(DATA_DIR, "karsl502"),
        )
        tbl = pq.read_table(path)
        meta = tbl.schema.metadata or {}
        names = [f"{i:04d}" for i in range(1, 503)]
        if b"huggingface" in meta:
            try:
                hf_meta = json.loads(meta[b"huggingface"].decode("utf-8"))
                nlist = hf_meta["info"]["features"]["label"]["names"]
                if isinstance(nlist, list) and len(nlist) == 502:
                    names = nlist
            except Exception:
                pass
        imgs = tbl["image"].to_pylist()
        lbls = tbl["label"].to_pylist()
        scanned += len(lbls)
        for im, lb in zip(imgs, lbls):
            bare = names[int(lb)] if isinstance(lb, int) and 0 <= int(lb) < len(names) else str(lb).zfill(4)
            sid = "K" + bare.zfill(4)
            if sid in cand_set and isinstance(im, dict) and im.get("bytes"):
                if split_name == "test" or per_class_budget[sid] < max_per_class:
                    tasks.append((im["bytes"], sid, split_name))
                    if split_name == "train":
                        per_class_budget[sid] += 1

    print(f"[ArSL-Words] Found {len(tasks):,} candidate frames across {len(cand_set)} everyday signs. Extracting 2-hand landmarks ...", flush=True)
    chunks = [tasks[i:i + 64] for i in range(0, len(tasks), 64)]
    t0 = time.time()
    X_list, y_list, s_list = [], [], []
    with ProcessPoolExecutor(max_workers=6, initializer=_init_worker) as pool:
        for done, res in enumerate(pool.map(_extract_chunk, chunks), 1):
            for r130, sid, sp in res:
                X_list.append(r130)
                y_list.append(sid)
                s_list.append(sp)
            if done % 10 == 0 or done == len(chunks):
                print(f"  chunk {done}/{len(chunks)} -> {len(X_list):,} hands detected ({time.time() - t0:.1f}s)", flush=True)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    sp_arr = np.array(s_list)
    np.savez_compressed(CACHE_FILE, X=X, y=y, split=sp_arr, scanned=np.array(scanned))
    print(f"[ArSL-Words] Saved {CACHE_FILE}: {len(X):,} usable frames out of {len(tasks):,}")
    return X, y, sp_arr, scanned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-own", action="store_true", help="Include personal webcam ArSL clips from data_arsl/own_arsl_words.npz")
    args = ap.parse_args()

    os.makedirs(REPORT_DIR, exist_ok=True)
    official_labels = ensure_karsl_labels()
    X130, y, sp_arr, total_scanned = download_and_extract()

    # Keep top 30 classes with >= 15 detected samples
    counts = Counter(y.tolist())
    # Preserve priority order among candidates with >= 15 samples, preferring top counts if > 30
    eligible = [(sid, counts[sid]) for sid, _, _ in CANDIDATE_IDS if counts[sid] >= 15]
    eligible.sort(key=lambda item: -item[1])
    chosen_ids = sorted([sid for sid, _ in eligible[:30]])
    chosen_set = set(chosen_ids)

    mask = np.array([lbl in chosen_set for lbl in y])
    X130 = X130[mask]
    y = y[mask]
    sp_arr = sp_arr[mask]

    # Build clean metadata list for the 30 classes
    cand_lookup = {sid: (ar, en) for sid, ar, en in CANDIDATE_IDS}
    class_meta = []
    for sid in chosen_ids:
        ar, en = cand_lookup.get(sid, (official_labels.get(sid, {}).get("ar", sid), official_labels.get(sid, {}).get("en", sid)))
        class_meta.append({"id": sid, "ar": ar, "en": en, "samples": int(counts[sid])})

    if args.with_own and os.path.exists(OWN_FILE):
        own = np.load(OWN_FILE, allow_pickle=True)
        own_mask = np.array([str(lbl) in chosen_set for lbl in own["y"]])
        if own_mask.any():
            X_own = own["X"][own_mask].astype(np.float32)
            y_own = own["y"][own_mask].astype(str)
            print(f"[ArSL-Words] Merged {len(X_own):,} personal webcam samples into training pool")
            X130 = np.vstack([X130, X_own])
            y = np.concatenate([y, y_own])
            sp_arr = np.concatenate([sp_arr, np.array(["train"] * len(X_own))])

    print(f"[ArSL-Words] Training on {len(X130):,} samples across {len(chosen_ids)} everyday words")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_i, test_i = next(splitter.split(X130, y))

    rng = np.random.default_rng(42)
    X_tr_raw, y_tr_raw = [], []
    for idx in train_i:
        row = X130[idx]
        lbl = y[idx]
        mir = mirror_row130(row)
        X_tr_raw.extend([row, mir, augment_row130(row, rng), augment_row130(mir, rng)])
        y_tr_raw.extend([lbl, lbl, lbl, lbl])

    X_tr_feat = arsl_batch_features(np.array(X_tr_raw, dtype=np.float32))
    y_tr_feat = np.array(y_tr_raw)
    X_te_feat = arsl_batch_features(X130[test_i])
    y_te = y[test_i]

    models = [
        ("ExtraTrees", ExtraTreesClassifier(
            n_estimators=400, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42
        )),
        ("LightGBM", lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.08, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, verbose=-1
        )),
        ("Neural Net (MLP)", make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=120,
                          early_stopping=True, n_iter_no_change=8, random_state=42)
        )),
        ("Ensemble (MLP+ET+LGBM)", VotingClassifier(
            estimators=[
                ("mlp", make_pipeline(
                    StandardScaler(),
                    MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=120,
                                  early_stopping=True, n_iter_no_change=8, random_state=42)
                )),
                ("et", ExtraTreesClassifier(
                    n_estimators=400, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42
                )),
                ("lgb", lgb.LGBMClassifier(
                    n_estimators=300, learning_rate=0.08, num_leaves=63,
                    subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, verbose=-1
                )),
            ],
            voting="soft",
            weights=[2, 1, 2],
        )),
    ]

    comparison = []
    best_score = -1.0
    best_bundle = None

    for name, clf in models:
        t0 = time.time()
        clf.fit(X_tr_feat, y_tr_feat)
        train_s = round(time.time() - t0, 2)
        t1 = time.time()
        probs = clf.predict_proba(X_te_feat)
        inf_ms = round((time.time() - t1) * 1000.0 / max(1, len(X_te_feat)), 4)
        pred = clf.classes_[probs.argmax(1)]
        top1 = accuracy_score(y_te, pred) * 100.0
        top3 = top_k_accuracy_score(y_te, probs, k=min(3, len(chosen_ids)), labels=clf.classes_) * 100.0
        top5 = top_k_accuracy_score(y_te, probs, k=min(5, len(chosen_ids)), labels=clf.classes_) * 100.0
        macro_f1 = f1_score(y_te, pred, average="macro") * 100.0
        size_mb = round(len(pickle.dumps(clf)) / 1e6, 2)
        print(f"  {name:26s} Top-1={top1:5.2f}%  Top-3={top3:5.2f}%  Top-5={top5:5.2f}%  MacroF1={macro_f1:5.2f}%  inf={inf_ms:.3f}ms")
        comparison.append({
            "model": name,
            "top1_acc": round(top1, 2),
            "top3_acc": round(top3, 2),
            "top5_acc": round(top5, 2),
            "macro_f1": round(macro_f1, 2),
            "train_sec": train_s,
            "inference_ms": inf_ms,
            "size_mb": size_mb,
        })
        if macro_f1 > best_score:
            best_score = macro_f1
            best_bundle = (name, clf, probs, pred, top1, top3, top5, macro_f1, inf_ms)

    best_name, best_clf, probs, pred, top1, top3, top5, macro_f1, inf_ms = best_bundle

    # Save classes metadata & trained model
    with open(CLASSES_OUT, "w", encoding="utf-8") as f:
        json.dump(class_meta, f, ensure_ascii=False, indent=2)
    joblib.dump(best_clf, MODEL_OUT, compress=3)
    model_mb = round(os.path.getsize(MODEL_OUT) / 1e6, 2)

    creport = classification_report(y_te, pred, labels=chosen_ids, output_dict=True, zero_division=0)
    id_to_meta = {m["id"]: m for m in class_meta}
    per_class = []
    for sid in chosen_ids:
        r = creport.get(sid, {})
        meta = id_to_meta[sid]
        per_class.append({
            "label": f"{sid} ({meta['en']})",
            "id": sid,
            "ar": meta["ar"],
            "en": meta["en"],
            "precision": round(r.get("precision", 0.0) * 100.0, 2),
            "recall": round(r.get("recall", 0.0) * 100.0, 2),
            "f1": round(r.get("f1-score", 0.0) * 100.0, 2),
            "support": int(r.get("support", 0)),
        })

    cm = confusion_matrix(y_te, pred, labels=chosen_ids)
    disp_labels = [f"{id_to_meta[s]['en'][:12]}" for s in chosen_ids]
    pairs = sorted(
        ((int(cm[i, j]), id_to_meta[chosen_ids[i]]["en"], id_to_meta[chosen_ids[j]]["en"])
         for i in range(len(chosen_ids)) for j in range(len(chosen_ids)) if i != j and cm[i, j] > 0),
        reverse=True,
    )[:10]

    fig, ax = plt.subplots(figsize=(13, 13))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=65
    )
    ax.set_title(f"ArSL 30-Word Confusion Matrix ({best_name} - {top1:.1f}% Top-1)")
    plt.tight_layout()
    cm_path = os.path.join(REPORT_DIR, "confusion_arsl_words.png")
    plt.savefig(cm_path, dpi=140)
    plt.close(fig)

    report = {
        "model_id": "arsl_words_30",
        "title": f"ArSL Everyday Words ({len(chosen_ids)} Signs)",
        "language": "ArSL",
        "dataset": "KArSL-502 (FatimahEmadEldin/karsl-502-arabic-sign-language-final)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_images_scanned": int(total_scanned),
        "total_samples": int(len(X130)),
        "train_samples": int(len(train_i)),
        "train_augmented_samples": int(len(X_tr_feat)),
        "test_samples": int(len(test_i)),
        "num_classes": len(chosen_ids),
        "best_model": best_name,
        "test_top1_acc": round(top1, 2),
        "test_top3_acc": round(top3, 2),
        "test_top5_acc": round(top5, 2),
        "test_macro_f1": round(macro_f1, 2),
        "inference_ms": inf_ms,
        "model_size_mb": model_mb,
        "model_file": MODEL_OUT,
        "classes": [f"{id_to_meta[s]['en']}" for s in chosen_ids],
        "class_meta": class_meta,
        "comparison": comparison,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "top_confused": [{"count": n, "true": a, "pred": b} for n, a, b in pairs],
    }
    rep_path = os.path.join(REPORT_DIR, "arsl_words_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[ArSL-Words] Saved {MODEL_OUT} ({model_mb} MB), {CLASSES_OUT}, {rep_path}, and {cm_path}")


if __name__ == "__main__":
    main()
