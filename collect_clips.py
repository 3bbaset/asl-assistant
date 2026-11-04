"""Record motion clips. Usage: python collect_clips.py J
SPACE = record 1.5 s, ESC = quit. Saves to own_clips.npz (appends)."""
import os, sys, time
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import cv2, numpy as np, mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from phrase_common import frame_features, clip_vector

label = sys.argv[1].upper()
OUT, DUR = "own_clips.npz", 1.5
opts = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO, num_hands=2)

X, y = [], []
if os.path.exists(OUT):
    d = np.load(OUT); X, y = list(d["X"]), list(d["y"])

cap, t0, seq, rec_start = cv2.VideoCapture(0), time.time(), [], None
with vision.HandLandmarker.create_from_options(opts) as lmk:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # NOT flipped, like the training videos
        res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                   int((now - t0) * 1000))
        if rec_start is not None:
            seq.append(frame_features(res))
            if now - rec_start >= DUR:
                arr = np.array(seq)
                if len(arr) >= 5 and arr.any(axis=1).mean() > 0.3:
                    X.append(clip_vector(arr)); y.append(label)
                    np.savez(OUT, X=np.array(X), y=np.array(y))
                else:
                    print("no hand seen, clip discarded")
                rec_start, seq = None, []
        show = cv2.flip(frame, 1)
        n = sum(1 for l in y if l == label)
        msg = f"{label}: {n} clips  " + ("REC" if rec_start else "SPACE to record")
        cv2.putText(show, msg, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("clips", show)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == 32 and rec_start is None:
            rec_start, seq = time.time(), []
cap.release(); cv2.destroyAllWindows()
