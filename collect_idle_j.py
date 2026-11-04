"""Record IDLE clips that LOOK like J but are not J (pinky out / I hand, no hook).
Usage: python collect_idle_j.py        # one pass through all prompts (about 25 clips)
       python collect_idle_j.py 10     # start at prompt 10
Each clip starts by itself after a 3 s countdown and lasts 1.5 s.
ESC = quit (saved clips are kept), s = skip a prompt, u = undo the last clip saved in this run.
Appends to own_clips.npz with the label IDLE."""
import os, sys, textwrap, time
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import cv2, numpy as np, mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from phrase_common import frame_features, clip_vector

OUT, DUR, PREP = "own_clips.npz", 1.5, 3.0

PROMPTS = [
    # still I hand (pinky up), must NOT count as J
    "Hold the I hand (pinky up) STILL, center",
    "Hold I STILL, left side of the picture",
    "Hold I STILL, right side of the picture",
    "Hold I STILL, close to the camera",
    "Hold I STILL, far from the camera",
    "Hold I STILL, high in the picture",
    "Hold I STILL, low in the picture",
    # I hand moving, but NO hook
    "I hand: straight line DOWN, then stop (NO hook!)",
    "I hand: straight line UP",
    "I hand: line left to right",
    "I hand: line right to left",
    "I hand: small CIRCLE",
    "I hand: big CIRCLE",
    "I hand: wave left and right",
    "I hand: tiny shaking",
    "I hand: slow drift across the picture",
    "I hand: tap forward toward the camera",
    "I hand: tilt the wrist side to side",
    # getting into and out of the I hand
    "Start with a fist, raise the pinky, then STOP (no drawing)",
    "Hold I, then lower the hand out of the picture",
    "Hand comes in from the bottom already making I, then stops",
    "Make I, then relax into an open hand",
    # Y hand (thumb and pinky out)
    "Y hand (thumb and pinky out) STILL",
    "Y hand: wave left and right",
]


def draw_text(img, msg, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(img, msg, org, cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.putText(img, msg, org, cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)


X, y = [], []
if os.path.exists(OUT):
    d = np.load(OUT)
    X, y = list(d["X"]), list(d["y"])
print(f"{sum(1 for l in y if l == 'IDLE')} IDLE clips already in the file, {len(PROMPTS)} prompts in this pass")

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
saved_now = 0

opts = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO, num_hands=2)

cap, t0, last_ts = cv2.VideoCapture(0), time.time(), 0
cv2.namedWindow("IDLE (J lookalikes)", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow("IDLE (J lookalikes)", 1024, 768)
phase, phase_start, seq, trail = "prep", time.time(), [], []
flash, flash_until, trail_until = "", 0.0, 0.0

with vision.HandLandmarker.create_from_options(opts) as lmk:
    while idx < len(PROMPTS):
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        ts = max(int((now - t0) * 1000), last_ts + 1)
        last_ts = ts
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)          # NOT flipped, like all the other clips
        res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        h, w = frame.shape[:2]

        if phase == "prep" and now - phase_start >= PREP:
            phase, phase_start, seq, trail = "rec", now, [], []
        elif phase == "rec":
            seq.append(frame_features(res))
            if res.hand_landmarks:                            # pinky-tip trail, drawn mirrored
                tip = res.hand_landmarks[0][20]
                trail.append((int((1 - tip.x) * w), int(tip.y * h)))
            if now - phase_start >= DUR:
                arr = np.array(seq)
                if len(arr) >= 5 and arr.any(axis=1).mean() > 0.3:
                    X.append(clip_vector(arr)); y.append("IDLE")
                    np.savez(OUT, X=np.array(X), y=np.array(y))
                    idx += 1
                    saved_now += 1
                    flash = "saved"
                else:
                    flash = "no hand seen, repeating this one"
                flash_until = trail_until = now + 1.2
                phase, phase_start = "prep", now

        show = cv2.flip(frame, 1)                             # mirrored view for you to watch
        if len(trail) > 1 and (phase == "rec" or now < trail_until):
            cv2.polylines(show, [np.array(trail, dtype=np.int32)], False, (0, 255, 255), 4, cv2.LINE_AA)
        if idx < len(PROMPTS):
            for i, line in enumerate(textwrap.wrap(PROMPTS[idx], 42)):
                draw_text(show, line, (12, 36 + 32 * i), 0.75, (0, 255, 255))
        if phase == "prep":
            draw_text(show, f"Get ready: {max(0, PREP - (now - phase_start)):.0f}",
                      (12, h - 50), 1.0, (0, 165, 255))
        else:
            cv2.circle(show, (w - 40, 40), 16, (0, 0, 255), -1)
            draw_text(show, "RECORDING - do it now", (12, h - 50), 1.0, (0, 0, 255))
        draw_text(show, f"clip {idx + 1}/{len(PROMPTS)}   {flash if now < flash_until else ''}",
                  (12, h - 14), 0.6)
        cv2.imshow("IDLE (J lookalikes)", show)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == ord("s"):
            idx += 1
            phase, phase_start = "prep", now
        elif k == ord("u") and saved_now > 0:
            X.pop(); y.pop()
            idx -= 1
            saved_now -= 1
            np.savez(OUT, X=np.array(X), y=np.array(y))
            flash, flash_until = "last clip removed", now + 1.2
            phase, phase_start = "prep", now

cap.release()
cv2.destroyAllWindows()
print(f"done. saved {saved_now} clips this run. IDLE clips in the file now: {sum(1 for l in y if l == 'IDLE')}")
