import os, subprocess, time
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import cv2, joblib, numpy as np, mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from phrase_common import frame_features, clip_vector
from motion_common import fill_gaps

DUR = 1.5
THRESH = 0.6      # J or Z is only reported if the model is at least this sure

model = joblib.load("motion_model2.joblib")
opts = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO, num_hands=2)

def speak(s):
    subprocess.Popen(["espeak-ng", "-s", "130", s],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def draw_text(img, msg, org, scale=1.0, color=(0, 255, 255), thick=2):
    cv2.putText(img, msg, org, cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.putText(img, msg, org, cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)

cap, t0, last_ts = cv2.VideoCapture(0), time.time(), 0
cv2.namedWindow("try J / Z", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow("try J / Z", 1024, 768)
seq, rec_start, shown, detail, shown_until = [], None, "", "", 0.0

with vision.HandLandmarker.create_from_options(opts) as lmk:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        ts = max(int((now - t0) * 1000), last_ts + 1)
        last_ts = ts
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     # NOT flipped, like the training clips
        res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        if rec_start is not None:
            seq.append(frame_features(res))
            if now - rec_start >= DUR:
                arr = np.array(seq)
                if len(arr) >= 5 and arr.any(axis=1).mean() > 0.3:
                    clip = fill_gaps(clip_vector(arr).reshape(30, 130))
                    probs = model.predict_proba(np.array([clip]))[0]
                    best = int(probs.argmax())
                    label = str(model.classes_[best])
                    if label != "IDLE" and probs[best] < THRESH:
                        label = "IDLE"                    # not sure enough, so ignore it
                    shown = label
                    detail = "   ".join(f"{c} {p:.0%}" for c, p in zip(model.classes_, probs))
                    if label != "IDLE":
                        speak(label)
                else:
                    shown, detail = "no hand seen", ""
                shown_until = now + 3.0
                rec_start, seq = None, []
        show = cv2.flip(frame, 1)                         # mirrored view for you to watch
        h, w = show.shape[:2]
        if rec_start is not None:
            draw_text(show, "RECORDING - sign now", (12, 40), 1.1, (0, 0, 255))
        elif now < shown_until:
            draw_text(show, f"Result: {shown}", (12, 40), 1.2)
            draw_text(show, detail, (12, 80), 0.8, (255, 255, 255))
        else:
            draw_text(show, "SPACE = record 1.5 s", (12, 40), 1.1)
        cv2.imshow("try J / Z", show)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == 32 and rec_start is None:
            rec_start, seq = time.time(), []
cap.release()
cv2.destroyAllWindows()
