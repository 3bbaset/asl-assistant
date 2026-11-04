import numpy as np
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from motion_common import fill_gaps, mirror_clip, clips_to_features

d = np.load("own_clips.npz")
clips = np.array([fill_gaps(c.reshape(30, 130)) for c in d["X"]])
y = np.array(d["y"])
classes = np.unique(y)                       # ['IDLE' 'J' 'Z']
groups = np.arange(len(y)) // 6              # neighbours in the recording stay together

def make_models():
    ft = lambda: FunctionTransformer(clips_to_features)
    return {
        "logistic regression": make_pipeline(ft(), StandardScaler(),
                                             LogisticRegression(C=0.1, max_iter=3000)),
        "svm": make_pipeline(ft(), StandardScaler(),
                             SVC(C=3, probability=True, random_state=42)),
        "random forest": make_pipeline(ft(), RandomForestClassifier(
                             n_estimators=300, random_state=42, n_jobs=-1)),
    }

def with_mirrors(idx):                       # training only: add left-handed versions
    C = np.concatenate([clips[idx], np.array([mirror_clip(c) for c in clips[idx]])])
    return C, np.concatenate([y[idx], y[idx]])

oof = {}
for name in make_models():
    P = np.zeros((len(y), len(classes)))
    for tr, te in GroupKFold(n_splits=5).split(clips, y, groups):
        Ctr, ytr = with_mirrors(tr)
        P[te] = make_models()[name].fit(Ctr, ytr).predict_proba(clips[te])
    oof[name] = P
    pred = classes[P.argmax(1)]
    print(f"\n{name}: held-out accuracy {accuracy_score(y, pred) * 100:.1f}%")
    print(classes)
    print(confusion_matrix(y, pred, labels=classes))
    print("   wrong clips:", [int(i) for i in np.where(pred != y)[0]])

best = max(oof, key=lambda n: accuracy_score(y, classes[oof[n].argmax(1)]))
P = oof[best]
print(f"\nBest: {best}. What a confidence threshold does (J or Z only if probability >= t):")
print("  t    false triggers (IDLE called J/Z)   J/Z not right      accuracy")
for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    pred = classes[P.argmax(1)].copy()
    pred[P.max(1) < t] = "IDLE"
    ft_ = int(((y == "IDLE") & (pred != "IDLE")).sum())
    bad = int(((y != "IDLE") & (pred != y)).sum())
    print(f" {t:.1f}   {ft_:3d} of {(y == 'IDLE').sum()}"
          f"                      {bad:3d} of {(y != 'IDLE').sum()}"
          f"           {accuracy_score(y, pred) * 100:.1f}%")

Call, yall = with_mirrors(np.arange(len(y)))
joblib.dump(make_models()[best].fit(Call, yall), "motion_model2.joblib")
print(f"\nsaved motion_model2.joblib ({best})")
