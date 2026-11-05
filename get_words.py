"""Step 2: download up to PER_WORD videos for every word in WORDS into videos/WORD/.
About 40% of the files in the dataset are broken on Hugging Face's side (404). Those are skipped,
remembered in hf_broken.txt, and the next video of the same word is tried instead.
Usage: python get_words.py      (safe to re-run: files already downloaded are skipped)"""
import os, random, shutil, threading
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars
from words_common import REPO, WORDS, PER_WORD, VIDEO_DIR, word_of

disable_progress_bars()
BROKEN = "hf_broken.txt"
with open("hf_files.txt", encoding="utf-8") as f:        # made by look_words.py
    files = f.read().split("\n")
broken = set(open(BROKEN, encoding="utf-8").read().split("\n")) if os.path.exists(BROKEN) else set()
lock = threading.Lock()


def get(path, dst):
    try:
        shutil.copyfile(hf_hub_download(REPO, path, repo_type="dataset"), dst)
        return True
    except Exception:
        with lock:
            broken.add(path)
            with open(BROKEN, "a", encoding="utf-8") as f:
                f.write(path + "\n")
        return False


def fill_word(w):
    folder = os.path.join(VIDEO_DIR, w)
    os.makedirs(folder, exist_ok=True)
    mine = sorted(x for x in files if word_of(x) == w)
    random.Random(42).shuffle(mine)                      # same order on every run
    have = len(os.listdir(folder))
    for path in mine:
        if have >= PER_WORD:
            break
        dst = os.path.join(folder, path.replace("/", "_"))
        if os.path.exists(dst) or path in broken:
            continue
        have += get(path, dst)
    return w, have, len(mine)


with ThreadPoolExecutor(8) as pool:
    for w, have, total in pool.map(fill_word, WORDS):
        print(f"  {w:15s} {have:3d} videos" + (f"   (only {have} of {total} downloadable)" if have < PER_WORD else ""))
print(f"done. videos are in {VIDEO_DIR}/, {len(broken)} broken files remembered in {BROKEN}")
