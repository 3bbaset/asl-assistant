"""collect_arsl.py - Optional webcam fine-tuning recorder for personal Egyptian/Arabic Sign Language words.

Records landmark samples matching arsl_common / phrase_common.frame_features into
data_arsl/own_arsl_words.npz. Retrain afterward with:
    .venv\\Scripts\\python.exe train_arsl_words.py --with-own
"""
import json
import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np

from common import CONNECTIONS
from phrase_common import frame_features

CLASSES_FILE = "arsl_word_classes_30.json"
OUT_FILE = os.path.join("data_arsl", "own_arsl_words.npz")
SAMPLES_PER_WORD = 20


def main():
    if not os.path.exists(CLASSES_FILE):
        print(f"Missing {CLASSES_FILE}. Run train_arsl_words.py first.")
        return
    with open(CLASSES_FILE, encoding="utf-8") as f:
        meta = json.load(f)

    os.makedirs("data_arsl", exist_ok=True)
    X_existing, y_existing = [], []
    if os.path.exists(OUT_FILE):
        d = np.load(OUT_FILE, allow_pickle=True)
        X_existing = list(d["X"])
        y_existing = list(d["y"])

    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.35,
    )
    det = vision.HandLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    idx = 0
    recording = False
    captured = 0
    ts_ms = 0
    print("Controls: SPACE = capture 20 samples | n/p = next/prev word | q = save & quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        view = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts_ms = max(ts_ms + 33, int(time.time() * 1000))
        res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)

        h, w = view.shape[:2]
        if res.hand_landmarks:
            for lm in res.hand_landmarks:
                pts = [(int((1 - p.x) * w), int(p.y * h)) for p in lm]
                for a, b in CONNECTIONS:
                    cv2.line(view, pts[a], pts[b], (0, 255, 0), 2)

        item = meta[idx]
        if recording and res.hand_landmarks:
            X_existing.append(frame_features(res))
            y_existing.append(item["id"])
            captured += 1
            if captured >= SAMPLES_PER_WORD:
                recording = False
                captured = 0
                idx = (idx + 1) % len(meta)

        status = f"REC {captured}/{SAMPLES_PER_WORD}" if recording else "READY (press SPACE)"
        cv2.rectangle(view, (0, 0), (w, 72), (15, 23, 42), -1)
        cv2.putText(view, f"[{idx+1}/{len(meta)}] {item['id']} {item['en']}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(view, f"{status} | Saved total: {len(y_existing)} | n=next p=prev q=quit", (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 180), 1)
        cv2.imshow("ArSL Word Collector", view)

        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        elif k == 32 and not recording:
            recording = True
            captured = 0
        elif k == ord("n"):
            idx = (idx + 1) % len(meta)
        elif k == ord("p"):
            idx = (idx - 1) % len(meta)

    cap.release()
    det.close()
    cv2.destroyAllWindows()
    if X_existing:
        np.savez_compressed(OUT_FILE, X=np.array(X_existing, dtype=np.float32), y=np.array(y_existing))
        print(f"Saved {len(y_existing)} personal ArSL samples to {OUT_FILE}")


if __name__ == "__main__":
    main()
