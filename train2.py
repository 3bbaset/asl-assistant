import csv, time, pickle
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from common import flat_to_features, renorm

rng = np.random.default_rng(42)
BLOCK, GAP = 150, 30

# ---- 1. Load -------------------------------------------------------------
letters, idxs, rows = [], [], []
with open("data_all.csv") as f:
    for r in csv.reader(f):
        letters.append(r[0]); idxs.append(int(r[1]))
        rows.append([float(v) for v in r[2:]])
y, idx, X = np.array(letters), np.array(idxs), np.array(rows)
print("samples:", len(X))

# ---- 2. Fair split: whole stretches of the recording, with a gap ----------
block = idx // BLOCK
groups = np.array([f"{l}-{b}" for l, b in zip(y, block)])
test_blocks = set()
for letter in np.unique(y):                       # every letter gives ~20% of its stretches
    ids = np.unique(groups[y == letter])
    n_test = max(1, round(len(ids) * 0.2))
    test_blocks.update(rng.choice(ids, n_test, replace=False))
is_test = np.array([g in test_blocks for g in groups])
test_i = np.where(is_test)[0]
train_i = np.where(~is_test)[0]
keep = []
for i in train_i:
    b, p = block[i], idx[i] % BLOCK
    near = ((f"{y[i]}-{b+1}" in test_blocks and p >= BLOCK - GAP) or
            (f"{y[i]}-{b-1}" in test_blocks and p < GAP))
    if not near:
        keep.append(i)
train_i = np.array(keep)
print("train:", len(train_i), "| test:", len(test_i))

# ---- 3. Augmentation (training data only) ---------------------------------
def rotation_matrices(ang):
    n = len(ang)
    cx, cy, cz = np.cos(ang).T
    sx, sy, sz = np.sin(ang).T
    Rx, Ry, Rz = np.zeros((n, 3, 3)), np.zeros((n, 3, 3)), np.zeros((n, 3, 3))
    Rx[:, 0, 0] = 1; Rx[:, 1, 1] = cx; Rx[:, 1, 2] = -sx; Rx[:, 2, 1] = sx; Rx[:, 2, 2] = cx
    Ry[:, 1, 1] = 1; Ry[:, 0, 0] = cy; Ry[:, 0, 2] = sy; Ry[:, 2, 0] = -sy; Ry[:, 2, 2] = cy
    Rz[:, 2, 2] = 1; Rz[:, 0, 0] = cz; Rz[:, 0, 1] = -sz; Rz[:, 1, 0] = sz; Rz[:, 1, 1] = cz
    return Rz @ Ry @ Rx

def augment(P):
    n = len(P)
    ang = np.radians(np.column_stack([rng.uniform(-12, 12, n),     # tilt forward/back
                                      rng.uniform(-12, 12, n),     # tilt left/right
                                      rng.uniform(-20, 20, n)]))   # rotate in the picture
    Q = np.einsum("nij,npj->npi", rotation_matrices(ang), P)
    Q[:, :, 0] *= rng.uniform(0.9, 1.1, (n, 1))                    # slight stretch
    Q += rng.normal(0, 0.01, Q.shape)                              # tiny jitter
    return renorm(Q)

P0 = X[train_i].reshape(-1, 21, 3)
Pm = P0.copy(); Pm[:, :, 0] *= -1                                  # mirrored hands
P_aug = np.concatenate([P0, Pm, augment(P0), augment(Pm)])
y_aug = np.tile(y[train_i], 4)
X_aug = flat_to_features(P_aug.reshape(len(P_aug), -1))
X_test = flat_to_features(X[test_i])
y_test = y[test_i]
print("training samples after augmentation:", len(X_aug))

# ---- 4. Train and compare -------------------------------------------------
results = {}
def run(name, clf, Xtr, ytr, Xte):
    t = time.time()
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc, f1 = accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro")
    mb = len(pickle.dumps(clf)) / 1e6
    results[name] = (acc, f1, mb, clf, pred)
    print(f"{name:45s} acc {acc*100:5.1f}%  macroF1 {f1:.3f}  size {mb:6.1f} MB  ({time.time()-t:.0f}s)", flush=True)

run("OLD WAY: forest, raw 63, no augmentation",
    RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    X[train_i], y[train_i], X[test_i])
run("forest + features + augmentation",
    RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_leaf=5,
                           class_weight="balanced", n_jobs=-1, random_state=42),
    X_aug, y_aug, X_test)
run("gradient boosting + features + augmentation",
    HistGradientBoostingClassifier(max_iter=150, early_stopping=False,
                                   class_weight="balanced", random_state=42),
    X_aug, y_aug, X_test)
run("small neural network + features + augmentation",
    make_pipeline(StandardScaler(),
                  MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=60,
                                early_stopping=True, n_iter_no_change=5, random_state=42)),
    X_aug, y_aug, X_test)

# ---- 5. Pick the best new model and report ---------------------------------
candidates = {k: v for k, v in results.items() if not k.startswith("OLD")}
best_name = max(candidates, key=lambda k: candidates[k][1])
acc, f1, mb, best, pred = results[best_name]
print(f"\nBEST: {best_name}  (accuracy {acc*100:.1f}%)\n")
print(classification_report(y_test, pred, digits=3))

classes = np.unique(y)
cm = confusion_matrix(y_test, pred, labels=classes)
pairs = sorted(((cm[i, j], a, b) for i, a in enumerate(classes)
                for j, b in enumerate(classes) if i != j and cm[i, j] > 0), reverse=True)
print("Most common mistakes (true -> predicted):")
for n, a, b in pairs[:10]:
    print(f"  {a} -> {b}: {n}")

fig, ax = plt.subplots(figsize=(11, 11))
ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, cmap="Blues", colorbar=False)
plt.tight_layout(); plt.savefig("confusion2.png", dpi=120)

# ---- 6. Save: the file does its own feature calculation ---------------------
ft = FunctionTransformer(flat_to_features).fit(X[:1])
joblib.dump(make_pipeline(ft, best), "asl_model_v2.joblib")
print("saved asl_model_v2.joblib and confusion2.png")
