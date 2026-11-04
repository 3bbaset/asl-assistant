"""Merge own1.csv (letter + 63 numbers) into data_all.csv (letter, index, 63 numbers).
Usage:  python merge_own.py            # ADD your samples, keep the Kaggle ones
        python merge_own.py --replace  # drop the Kaggle rows for those letters first
"""
import csv, sys
from collections import defaultdict

OFFSET = 200 * 100000      # far from the Kaggle and M/N indexes
CHUNK, SPACING = 30, 300   # same block trick as merge_alpha.py

own = defaultdict(list)
for r in csv.reader(open("own1.csv")):
    if r:
        own[r[0]].append(r[1:])

old = list(csv.reader(open("data_all.csv")))
old = [r for r in old if int(r[1]) < OFFSET]          # drop earlier runs, no duplicates
if "--replace" in sys.argv:
    old = [r for r in old if r[0] not in own]

new = []
for L, rows in own.items():
    print(L, len(rows), "own samples")
    for j, v in enumerate(rows):
        idx = OFFSET + (j // CHUNK) * SPACING + (j % CHUNK)
        new.append([L, idx] + v)

with open("data_all.csv", "w", newline="") as f:
    csv.writer(f).writerows(old + new)
print(f"data_all.csv: {len(old)} old + {len(new)} new rows")
