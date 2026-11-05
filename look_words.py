"""Step 1 of the word model: see which words exist in the Hugging Face dataset and how many videos each has.
Usage: python look_words.py            -> counts for the WORDS list in words_common.py
       python look_words.py HEL         -> also lists every word that starts with HEL (spelling check)
The file list (108k names) is cached in hf_files.txt so later runs are instant."""
import os, sys
from collections import Counter
from words_common import REPO, WORDS, word_of

CACHE = "hf_files.txt"
if not os.path.exists(CACHE):
    from huggingface_hub import HfApi
    print("listing the dataset (about 1 minute, only the first time)...")
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    with open(CACHE, "w", encoding="utf-8") as f:
        f.write("\n".join(x for x in files if x.lower().endswith(".mp4")))
with open(CACHE, encoding="utf-8") as f:
    files = f.read().split("\n")

count = Counter(word_of(x) for x in files)
print(f"{len(files)} videos, {len(count)} different words\n")
for w in WORDS:
    print(f"  {w:15s} {count.get(w, 0):4d} videos" + ("   <-- MISSING, check spelling" if w not in count else ""))
for prefix in sys.argv[1:]:
    print(f"\nwords starting with {prefix.upper()}:")
    for w, n in sorted(count.items()):
        if w.startswith(prefix.upper()):
            print(f"  {w:25s} {n}")
