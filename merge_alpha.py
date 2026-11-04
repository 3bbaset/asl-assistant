"""Import letters from the landmark dataset sid220/asl-now-fingerspelling into data_all.csv.

Usage (inside ~/asl-assistant, venv active):
    python merge_alpha.py --replace            # replace M and N with the dataset's (default letters = MN)
    python merge_alpha.py --letters MNST       # choose other letters
    python merge_alpha.py --add                # keep your old rows too and just ADD the new ones

Then retrain:  python train2.py
data_all.csv is backed up to data_all_backup.csv first (only once, never overwritten).
"""
import argparse
import csv
import json
import os
import shutil
from types import SimpleNamespace

import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files
from common import normalize            # YOUR normalize, so the numbers are identical

REPO = "sid220/asl-now-fingerspelling"
CSV = "data_all.csv"
OFFSET = 150 * 100000      # new rows get indexes far from your old ones, in blocks of 30 spaced 2 blocks apart,
CHUNK, SPACING = 30, 300   # so train2.py's stretch-based test split treats them fairly

ap = argparse.ArgumentParser()
ap.add_argument("--letters", default="MN")
ap.add_argument("--replace", action="store_true", help="delete your existing rows for these letters first")
ap.add_argument("--add", action="store_true", help="keep your existing rows (default if --replace not given)")
ap.add_argument("--keep", type=float, default=0.8, help="fraction of samples to keep per letter (drops outliers)")
args = ap.parse_args()
letters = args.letters.upper()


def find_points(o):
    """Find 21 landmarks anywhere in the JSON -> list of (x, y, z)."""
    if isinstance(o, list) and len(o) == 21 and isinstance(o[0], dict):
        return [(p["x"], p["y"], p.get("z", 0.0)) for p in o]
    if isinstance(o, list) and len(o) == 21 and isinstance(o[0], (list, tuple)):
        return [(p[0], p[1], p[2] if len(p) > 2 else 0.0) for p in o]
    kids = o.values() if isinstance(o, dict) else (o if isinstance(o, list) else [])
    for v in kids:
        r = find_points(v)
        if r:
            return r
    return None


files = [f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".json")]
new_rows = []
for L in letters:
    paths = [f for f in files if f.split("/")[-2].upper() == L]
    vecs = []
    for p in paths:
        pts = find_points(json.load(open(hf_hub_download(REPO, p, repo_type="dataset"))))
        if pts:
            vecs.append(normalize([SimpleNamespace(x=a, y=b, z=c) for a, b, c in pts]).ravel())
    X = np.array(vecs)
    if len(X) == 0:
        print(f"{L}: nothing found!")
        continue
    # drop the samples furthest from the typical one (keeps the clean, consistent signs)
    d = np.linalg.norm(X - np.median(X, axis=0), axis=1)
    X = X[np.argsort(d)[: max(10, int(len(X) * args.keep))]]
    print(f"{L}: {len(paths)} samples -> kept {len(X)}")
    for j, v in enumerate(X):
        idx = OFFSET + (j // CHUNK) * SPACING + (j % CHUNK)
        new_rows.append([L, idx] + [repr(float(a)) for a in v])

if not os.path.exists("data_all_backup.csv"):
    shutil.copy(CSV, "data_all_backup.csv")
    print("backup saved: data_all_backup.csv")

old = list(csv.reader(open("data_all_backup.csv" if args.replace else CSV)))
if args.replace:
    old = [r for r in old if r[0] not in letters]
    print(f"removed old rows for {letters}")
# drop any rows imported by an earlier run of this script, so re-running never duplicates
old = [r for r in old if not (r[0] in letters and int(r[1]) >= OFFSET)]

with open(CSV, "w", newline="") as f:
    csv.writer(f).writerows(old + new_rows)
print(f"{CSV}: {len(old)} old + {len(new_rows)} new rows. Now run: python train2.py")
