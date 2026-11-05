"""High-speed 1,000-video streaming batch pipeline for the full Hugging Face ASL dataset.

Optimizations for 100k+ videos:
  1. Direct HTTP streaming (`resolve/main/...`) across 24 threads (no extra API metadata round-trips).
  2. Persistent MediaPipe HandLandmarker per worker process (initialized once per worker instead of
     once per video, cutting CPU extraction time by >2.5x).
  3. Uniform 320x240 frame-strided sampling (up to 20 frames per video -> resampled to 30x130 float16).
  4. Saves ~2 MB `batches/batch_XXX.npz` per 1,000 entries and immediately deletes raw `.mp4` files.
  5. Automatically runs 80/20 training checkpoints every `--train-every` batches.

Usage:
  python batch_pipeline.py --all --train             # run all batches continuously with periodic 80/20 training
  python batch_pipeline.py --batches 5 --train       # run next 5 batches (5,000 videos) then train
  python batch_pipeline.py --status                  # print current progress"""
import argparse, glob, os, shutil, subprocess, sys, threading, time, urllib.parse, urllib.request
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
from words_common import REPO, WORDS, VIDEO_DIR, DATA_FILE, word_of

BATCH_SIZE = 1000
BATCH_DIR = "batches"
TMP_DIR = "tmp_batch_mp4"
BROKEN_FILE = "hf_broken.txt"
CATALOG_FILE = "hf_files.txt"
MIN_VIDEOS_PER_WORD = 10
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/"

# Persistent worker globals (initialized once per worker process)
_WORKER_LMK = None
_WORKER_TS = 0


def _init_worker():
    global _WORKER_LMK, _WORKER_TS
    import cv2, mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    cv2.setNumThreads(1)
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _WORKER_LMK = vision.HandLandmarker.create_from_options(opts)
    _WORKER_TS = 0


def extract_one_video(job):
    """Worker function using the persistent _WORKER_LMK detector."""
    global _WORKER_LMK, _WORKER_TS
    rel_path, local_mp4 = job
    import cv2, mediapipe as mp
    from phrase_common import frame_features, clip_vector

    if _WORKER_LMK is None:
        _init_worker()

    cap = cv2.VideoCapture(local_mp4)
    if not cap.isOpened():
        return rel_path, None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 48
    step = max(1, n_frames // 20) if n_frames > 24 else 1

    seq, i = [], 0
    try:
        while i < 600:
            ok = cap.grab()
            if not ok:
                break
            if i % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                frame = cv2.resize(frame, (320, 240))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _WORKER_TS += 33
                res = _WORKER_LMK.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), _WORKER_TS
                )
                seq.append(frame_features(res))
            i += 1
    except Exception:
        # Re-initialize worker if MediaPipe state ever resets on a corrupted video stream
        try:
            _init_worker()
        except Exception:
            pass
        return rel_path, None
    finally:
        cap.release()

    seq = np.array(seq, dtype=np.float32)
    if len(seq) < 4:
        return rel_path, None
    seen = np.where(seq.any(axis=1))[0]
    if len(seen) < 4 or len(seen) / len(seq) < 0.25:
        return rel_path, None
    seq = seq[seen[0]:seen[-1] + 1]
    dur = (len(seq) * step) / max(10.0, min(60.0, fps))
    return rel_path, (clip_vector(seq).astype(np.float16), float(dur))


def build_ordered_catalog():
    if not os.path.exists(CATALOG_FILE):
        sys.exit(f"{CATALOG_FILE} not found. Run look_words.py first.")
    with open(CATALOG_FILE, encoding="utf-8") as f:
        files = [x.strip() for x in f if x.strip()]
    by_word = {}
    for path in files:
        w = word_of(path)
        by_word.setdefault(w, []).append(path)

    priority = list(WORDS) + (["GOOD"] if "GOOD" not in WORDS else [])
    ordered_words = [w for w in priority if w in by_word]
    rest = sorted(
        (w for w in by_word if w not in set(ordered_words) and len(by_word[w]) >= MIN_VIDEOS_PER_WORD),
        key=lambda w: (-len(by_word[w]), w),
    )
    ordered_words.extend(rest)

    ordered_files = []
    for w in ordered_words:
        ordered_files.extend(sorted(by_word[w]))
    return ordered_files, len(ordered_words)


def bootstrap_batch_0():
    os.makedirs(BATCH_DIR, exist_ok=True)
    b0 = os.path.join(BATCH_DIR, "batch_000.npz")
    if os.path.exists(b0) or not os.path.exists(DATA_FILE):
        return
    d = np.load(DATA_FILE, allow_pickle=True)
    np.savez_compressed(
        b0,
        path=d["path"],
        X=d["X"].astype(np.float16),
        y=d["y"],
        secs=d["secs"].astype(np.float16),
        entries=np.array(BATCH_SIZE, dtype=np.int32),
    )
    print(f"Bootstrapped {b0} from existing {DATA_FILE} ({len(d['y'])} valid clips).")


def process_batch(batch_idx, batch_files, broken, lock, pool):
    out_npz = os.path.join(BATCH_DIR, f"batch_{batch_idx:03d}.npz")
    if os.path.exists(out_npz):
        return False

    t0 = time.time()
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    downloaded = []
    new_broken = []

    def download_one(rel_path):
        if rel_path in broken:
            return None
        w = word_of(rel_path)
        existing = os.path.join(VIDEO_DIR, w, rel_path.replace("/", "_"))
        if os.path.exists(existing) and os.path.getsize(existing) > 1000:
            return (rel_path, existing)
        dst = os.path.join(TMP_DIR, rel_path.replace("/", "_"))
        url = BASE_URL + urllib.parse.quote(rel_path)
        for attempt in range(2):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as resp, open(dst, "wb") as out_f:
                    shutil.copyfileobj(resp, out_f)
                if os.path.getsize(dst) > 1000:
                    return (rel_path, dst)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    with lock:
                        broken.add(rel_path)
                        new_broken.append(rel_path)
                    return None
                time.sleep(0.5)
            except Exception:
                time.sleep(0.5)
        return None

    print(f"\n[Batch {batch_idx:03d}] Downloading {len(batch_files)} video entries (24 threads)...", flush=True)
    with ThreadPoolExecutor(24) as dl_pool:
        for res in dl_pool.map(download_one, batch_files):
            if res is not None:
                downloaded.append(res)

    if new_broken:
        with open(BROKEN_FILE, "a", encoding="utf-8") as bf:
            bf.write("\n".join(new_broken) + "\n")

    dl_time = time.time() - t0
    print(f"[Batch {batch_idx:03d}] {len(downloaded)} valid videos ready ({dl_time:.1f}s). "
          f"Extracting landmarks...", flush=True)

    paths, Xs, ys, secs = [], [], [], []
    t_ext = time.time()
    if downloaded:
        for k, (rel_path, result) in enumerate(pool.map(extract_one_video, downloaded, chunksize=8), 1):
            if result is not None:
                clip16, dur = result
                paths.append(rel_path)
                Xs.append(clip16)
                ys.append(word_of(rel_path))
                secs.append(dur)
            if k % 150 == 0 or k == len(downloaded):
                print(f"  [Batch {batch_idx:03d}] Extracted {k}/{len(downloaded)} videos "
                      f"({len(Xs)} with hands, {time.time() - t_ext:.0f}s)", flush=True)

    np.savez_compressed(
        out_npz,
        path=np.array(paths),
        X=np.array(Xs, dtype=np.float16) if Xs else np.zeros((0, 3900), dtype=np.float16),
        y=np.array(ys),
        secs=np.array(secs, dtype=np.float16),
        entries=np.array(len(batch_files), dtype=np.int32),
    )

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    mb = os.path.getsize(out_npz) / 1e6
    print(f"[Batch {batch_idx:03d}] DONE in {time.time() - t0:.1f}s -> saved {len(Xs)} clips "
          f"to {out_npz} ({mb:.1f} MB). Raw mp4s deleted.", flush=True)
    return True


def show_status(ordered_files, n_words):
    done_files = sorted(glob.glob(os.path.join(BATCH_DIR, "batch_*.npz")))
    total_batches = (len(ordered_files) + BATCH_SIZE - 1) // BATCH_SIZE
    total_clips, total_entries = 0, 0
    all_words = Counter()
    for f in done_files:
        d = np.load(f, allow_pickle=True)
        total_clips += len(d["y"])
        total_entries += int(d["entries"]) if "entries" in d else BATCH_SIZE
        all_words.update(d["y"])
    print(f"\n=== Full-Dataset Batch Pipeline Status ===", flush=True)
    print(f"Catalog size       : {len(ordered_files):,} videos across {n_words:,} words (>= {MIN_VIDEOS_PER_WORD} videos/word)")
    print(f"Batches completed  : {len(done_files)} / {total_batches} (batch size = {BATCH_SIZE:,} videos)")
    print(f"Dataset entries    : {total_entries:,} / {len(ordered_files):,} ({total_entries/len(ordered_files)*100:.1f}%)")
    print(f"Extracted clips    : {total_clips:,} clean landmark clips across {len(all_words):,} words", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=1, help="Number of 1,000-video batches to process now")
    ap.add_argument("--all", action="store_true", help="Process all remaining batches continuously")
    ap.add_argument("--train", action="store_true", help="Run 80/20 train_full.py checkpoints")
    ap.add_argument("--train-every", type=int, default=5, help="Run 80/20 training every N batches")
    ap.add_argument("--status", action="store_true", help="Print status and exit")
    args = ap.parse_args()

    ordered_files, n_words = build_ordered_catalog()
    bootstrap_batch_0()

    if args.status:
        show_status(ordered_files, n_words)
        return

    broken = set()
    if os.path.exists(BROKEN_FILE):
        with open(BROKEN_FILE, encoding="utf-8") as f:
            broken = {x.strip() for x in f if x.strip()}
    lock = threading.Lock()

    total_batches = (len(ordered_files) + BATCH_SIZE - 1) // BATCH_SIZE
    target_new = total_batches if args.all else args.batches
    n_workers = min(8, max(2, (os.cpu_count() or 4) - 2))
    ran = 0

    with ProcessPoolExecutor(n_workers, initializer=_init_worker) as pool:
        for b_idx in range(total_batches):
            if ran >= target_new:
                break
            start = b_idx * BATCH_SIZE
            batch_slice = ordered_files[start:start + BATCH_SIZE]
            if process_batch(b_idx, batch_slice, broken, lock, pool):
                ran += 1
                if args.train and (ran % args.train_every == 0):
                    show_status(ordered_files, n_words)
                    subprocess.run([sys.executable, "train_full.py"], check=False)

    show_status(ordered_files, n_words)
    if args.train and ran > 0 and (ran % args.train_every != 0):
        subprocess.run([sys.executable, "train_full.py"], check=False)


if __name__ == "__main__":
    main()
