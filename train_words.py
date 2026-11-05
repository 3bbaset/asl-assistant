"""Step 4: train the word model. Usage: python train_words.py
Data:
  word_clips.npz  - clips from the internet videos (extract_words.py)
  own_clips.npz   - our own recordings (collect_clips.py / quiz_words.py). IDLE clips are used so the
                    model can say "nobody is signing a word", and any clip whose label is in WORDS is
                    added too.
Prints held-out accuracy for each candidate model, saves the best as word_model.joblib, and plots
confusion_words.png."""
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from motion_common import fill_gaps, mirror_clip
from words_common import DATA_FILE, MODEL_FILE, WORDS, words_to_features, source_of

TWO_HANDED = {"HELP", "MORE", "STOP", "LOVE", "FRIEND", "FAMILY", "NAME", "WHAT"}

d = np.load(DATA_FILE, allow_pickle=True)
clips, y, src = [], [], []
for raw_c, l, p in zip(d["X"], d["y"], d["path"]):
    if l not in WORDS and l != "GOOD":
        continue
    c = fill_gaps(raw_c.reshape(30, 130))
    both = float((c[:, :63].any(1) & c[:, 65:128].any(1)).mean())
    if l in TWO_HANDED and both < 0.15:
        continue                         # MediaPipe missed the second hand in this internet video
    if l == "GOOD":
        if both < 0.2:
            l = "THANK YOU"              # 1-handed GOOD in ASL is physically the exact sign for THANK YOU
        else:
            continue
    clips.append(c); y.append(str(l)); src.append(source_of(p))
try:
    o = np.load("own_clips.npz")
    for c, l in zip(o["X"], o["y"]):
        if l == "IDLE" or l in WORDS:
            clips.append(fill_gaps(c.reshape(30, 130))); y.append(str(l)); src.append("own")
except FileNotFoundError:
    pass
clips, y, src = np.array(clips), np.array(y), np.array(src)
classes = np.unique(y)
print(f"{len(y)} clips, {len(classes)} classes:",
      ", ".join(f"{c} {int((y == c).sum())}" for c in classes))


def make_models():
    ft = lambda: FunctionTransformer(words_to_features)
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000))
    et = lambda: ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    return {
        "logistic regression": make_pipeline(ft(), lr()),
        "extra trees": make_pipeline(ft(), et()),
        "ensemble (lr + trees)": make_pipeline(
            ft(), VotingClassifier([("lr", lr()), ("et", et())], voting="soft")
        ),
    }


def with_mirrors(idx):                       # training only: add left-handed / mirrored-camera versions
    C = np.concatenate([clips[idx], np.array([mirror_clip(c) for c in clips[idx]])])
    return C, np.concatenate([y[idx], y[idx]])


folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(clips, y))
oof = {}
for name in make_models():
    P = np.zeros((len(y), len(classes)))
    for tr, te in folds:
        P[te] = make_models()[name].fit(*with_mirrors(tr)).predict_proba(clips[te])
    oof[name] = P
    print(f"{name:24s} held-out accuracy {accuracy_score(y, classes[P.argmax(1)]) * 100:.1f}%")

best = max(oof, key=lambda n: accuracy_score(y, classes[oof[n].argmax(1)]))
pred = classes[oof[best].argmax(1)]
print(f"\nBest: {best}. Per word (recall = how often that sign is recognised):")
print(classification_report(y, pred, zero_division=0))

cm = confusion_matrix(y, pred, labels=classes)
pairs = sorted(((cm[i, j], a, b) for i, a in enumerate(classes)
                for j, b in enumerate(classes) if i != j and cm[i, j] > 0), reverse=True)
print("Most common confusions (true -> predicted):")
for n, a, b in pairs[:8]:
    print(f"  {a:12s} -> {b:12s}: {n}")

fig, ax = plt.subplots(figsize=(12, 12))
ConfusionMatrixDisplay.from_predictions(y, pred, ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
plt.tight_layout()
plt.savefig("confusion_words.png", dpi=120)

print("\nAccuracy per recording batch (file-name style):")
for s in np.unique(src):
    m = src == s
    print(f"  {s:10s} {m.sum():4d} clips  {accuracy_score(y[m], pred[m]) * 100:5.1f}%")

conf = oof[best].max(1)
print("\nConfidence threshold (word shown only if probability >= t):")
for t in (0.3, 0.4, 0.5, 0.6, 0.7):
    m = conf >= t
    print(f"  t={t:.1f}: answers {m.mean() * 100:5.1f}% of clips, and those are right "
          f"{accuracy_score(y[m], pred[m]) * 100 if m.any() else 0:5.1f}% of the time")

fitted = make_models()[best].fit(*with_mirrors(np.arange(len(y))))
# Switch n_jobs to 1 before saving so single-clip live prediction has zero thread spawn overhead
for step in fitted.named_steps.values():
    for est in getattr(step, "estimators_", [step]):
        if hasattr(est, "n_jobs"):
            est.n_jobs = 1
joblib.dump(fitted, MODEL_FILE)
print(f"\nsaved {MODEL_FILE} ({best}) and confusion_words.png")
