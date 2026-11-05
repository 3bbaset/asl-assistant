"""Step 3: run MediaPipe on every video in videos/WORD/ and save one clip per video to word_clips.npz.
Each clip = 30 frames x 130 numbers (phrase_common.frame_features), the SAME format as own_clips.npz,
so the live app can feed the word model exactly like the J/Z model.
Usage: python extract_words.py      (re-run after adding videos; already-done videos are reused)"""
import os, glob
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from words_common import VIDEO_DIR, DATA_FILE


def video_to_clip(path):
    import cv2, mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    from phrase_common import frame_features, clip_vector
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    seq, i = [], 0
    with vision.HandLandmarker.create_from_options(opts) as lmk:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if max(h, w) > 640:                               # same scale as the webcam, and faster
                s = 640 / max(h, w)
                frame = cv2.resize(frame, (int(w * s), int(h * s)))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)       # NOT flipped, like collect_clips.py
            res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                       int(i * 1000 / fps))
            seq.append(frame_features(res))
            i += 1
    cap.release()
    seq = np.array(seq)
    if len(seq) < 5:
        return None
    seen = np.where(seq.any(axis=1))[0]
    if len(seen) < 5 or len(seen) / len(seq) < 0.3:
        return None                                           # hands hardly visible: useless clip
    seq = seq[seen[0]:seen[-1] + 1]                           # cut the no-hand start and end
    return clip_vector(seq).astype(np.float32), len(seq) / fps


if __name__ == "__main__":
    paths = sorted(glob.glob(os.path.join(VIDEO_DIR, "*", "*.mp4")))
    old = {}
    if os.path.exists(DATA_FILE):
        d = np.load(DATA_FILE, allow_pickle=True)
        old = {p: (x, y, s) for p, x, y, s in zip(d["path"], d["X"], d["y"], d["secs"])}
    todo = [p for p in paths if p not in old]
    print(f"{len(paths)} videos, {len(todo)} new to process")

    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for k, (p, r) in enumerate(zip(todo, pool.map(video_to_clip, todo, chunksize=4)), 1):
            if r is None:
                print("  skipped (no hands):", p)
            else:
                old[p] = (r[0], os.path.basename(os.path.dirname(p)), r[1])
            if k % 50 == 0:
                print(f"  {k}/{len(todo)}")

    keep = [p for p in paths if p in old]
    np.savez(DATA_FILE, path=np.array(keep), X=np.array([old[p][0] for p in keep]),
             y=np.array([old[p][1] for p in keep]), secs=np.array([old[p][2] for p in keep]))
    ys, counts = np.unique([old[p][1] for p in keep], return_counts=True)
    print(f"saved {len(keep)} clips to {DATA_FILE}")
    for w, n in zip(ys, counts):
        print(f"  {w:15s} {n}")
    secs = np.array([old[p][2] for p in keep])
    print(f"sign length: median {np.median(secs):.2f} s, 90% under {np.percentile(secs, 90):.2f} s")
