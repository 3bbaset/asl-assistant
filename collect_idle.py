import os, textwrap, time
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import cv2, numpy as np, mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from phrase_common import frame_features, clip_vector

OUT, DUR, PREP = "own_clips.npz", 1.5, 3.0

PROMPTS = [
    # A. still handshapes
    "Hold I (pinky up) STILL, center",
    "Hold I STILL, left side of the picture",
    "Hold I STILL, right side, close to the camera",
    "Hold a POINTING index finger STILL, center",
    "Hold the pointing index STILL, high in the picture",
    "Hold the pointing index STILL, far from the camera",
    "Hold L STILL",
    "Hold A (fist, thumb at the side) STILL",
    "Hold W (three fingers up) STILL",
    "Hold a plain fist STILL, low in the picture",
    # B. resting and transitions
    "Hand rests low in the picture, relaxed",
    "Hand rests on the left, relaxed",
    "Hand ENTERS from the bottom, then stops",
    "Hand LEAVES: move it down and out of the picture",
    "Open relaxed hand, doing nothing",
    "Adjust your sleeve",
    "Scratch your face or touch your hair",
    "Hand half out of the picture, still",
    # C. pinky out, but NOT a J
    "Pinky out: straight line UP (no hook!)",
    "Pinky out: straight line SIDEWAYS",
    "Pinky out: small CIRCLE",
    "Pinky out: wave left and right",
    "Pinky out: tiny shaking",
    # D. index out, but NOT a Z
    "Index finger: straight line UP or DOWN (no Z!)",
    "Index finger: a CIRCLE",
    "Index finger: side to side",
    "Index finger: small tap forward",
    "Index finger: one slow horizontal line only",
    # E. open hand movements
    "Open hand: wave hello",
    "Open hand: slow sweep across the picture",
    "Open hand: fast shake",
    "Twist your wrist, hand stays in place",
    # F. other letters while moving
    "Make A and move the hand slowly up and down",
    "Make B (flat hand) and move it sideways",
    "Make C and move it slowly",
    "Make O and move it slowly",
    # G. two hands
    "BOTH hands resting in view",
    "BOTH hands moving",
    "One hand still, the other waving",
    "Rub your hands together",
]

def draw_text(img, msg, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(img, msg, org, cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.putText(img, msg, org, cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)

X, y = [], []
if os.path.exists(OUT):
    d = np.load(OUT)
    X, y = list(d["X"]), list(d["y"])
idx = sum(1 for l in y if l == "IDLE")          # resume where you stopped
print(f"{idx} IDLE clips already saved, {len(PROMPTS)} prompts in total")

opts = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO, num_hands=2)

cap, t0, last_ts = cv2.VideoCapture(0), time.time(), 0
cv2.namedWindow("IDLE recorder", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow("IDLE recorder", 1024, 768)
phase, phase_start, seq, flash, flash_until = "prep", time.time(), [], "", 0.0

with vision.HandLandmarker.create_from_options(opts) as lmk:
    while idx < len(PROMPTS):
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        ts = max(int((now - t0) * 1000), last_ts + 1)
        last_ts = ts
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)          # NOT flipped, like the J/Z clips
        res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        if phase == "prep" and now - phase_start >= PREP:
            phase, phase_start, seq = "rec", now, []
        elif phase == "rec":
            seq.append(frame_features(res))
            if now - phase_start >= DUR:
                arr = np.array(seq)
                if len(arr) >= 5 and arr.any(axis=1).mean() > 0.3:
                    X.append(clip_vector(arr)); y.append("IDLE")
                    np.savez(OUT, X=np.array(X), y=np.array(y))
                    idx += 1
                    flash = "saved"
                else:
                    flash = "no hand seen, repeating this one"
                flash_until = now + 1.2
                phase, phase_start = "prep", now

        show = cv2.flip(frame, 1)                              # mirrored view for you to watch
        h, w = show.shape[:2]
        if idx < len(PROMPTS):
            for i, line in enumerate(textwrap.wrap(PROMPTS[idx], 42)):
                draw_text(show, line, (12, 36 + 32 * i), scale=0.75, color=(0, 255, 255))
        if phase == "prep":
            left = max(0, PREP - (now - phase_start))
            draw_text(show, f"Get ready: {left:.0f}", (12, h - 50), scale=1.0, color=(0, 165, 255))
        else:
            cv2.circle(show, (w - 40, 40), 16, (0, 0, 255), -1)
            draw_text(show, "RECORDING - do it now", (12, h - 50), scale=1.0, color=(0, 0, 255))
        draw_text(show, f"clip {idx + 1}/{len(PROMPTS)}   {flash if now < flash_until else ''}",
                  (12, h - 14), scale=0.6)
        cv2.imshow("IDLE recorder", show)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:                                            # ESC quits
            break
        elif k == ord("s"):                                    # s skips this prompt
            idx += 1
            phase, phase_start = "prep", now
        elif k == ord("u") and y and y[-1] == "IDLE":          # u undoes the last saved clip
            X.pop(); y.pop(); idx = max(0, idx - 1)
            np.savez(OUT, X=np.array(X), y=np.array(y))
            flash, flash_until = "last clip removed", now + 1.2
            phase, phase_start = "prep", now

cap.release()
cv2.destroyAllWindows()
print("done. IDLE clips saved:", sum(1 for l in y if l == "IDLE"))
