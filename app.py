import os
if os.name == "posix":                            # Linux only: avoids the Wayland warning (breaks Windows)
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import subprocess, sys, threading, time
from collections import Counter, deque
from types import SimpleNamespace
import cv2
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from common import make_detector, detect, normalize, draw_hand
from phrase_common import frame_features, clip_vector
from motion_common import fill_gaps

SOURCE = 0        # 0 = laptop camera. Later this becomes the Pi's stream address.
SHARE = "--share" in sys.argv   # run python app.py --share: ONE hand detector for both models (faster)
# ---- letters (unchanged) ----
WINDOW = 0.6      # seconds: how long a letter must be held
AGREE = 0.8       # share of that window the same letter must win
MIN_CONF = 0.6    # ignore guesses the model is unsure about
UNLOCK_AFTER = 0.5  # seconds without a hand before the same letter can be typed again
# ---- J / Z, found automatically in the last 1.5 s of movement ----
MOTION_DUR = 1.5        # length of the clip the model was trained on
EVAL_EVERY = 0.2        # ask the model this often (seconds)
MOTION_MIN_SEEN = 0.4   # the hand must be tracked in at least this share of the window
MOVE_MIN = 0.10         # a fingertip must travel at least this far (1.0 = the picture width). UNTUNED: press d
MOTION_THRESH = 0.6     # J or Z only if the model is at least this sure
CONFIRM = 2             # ...and this many checks in a row must agree (stops one-off false alarms)
MOTION_COOLDOWN = 1.0   # seconds of silence after a J/Z
QUIET_AFTER = 1.0       # seconds after a J/Z when letters are not typed
# J starts as an I hand and Z as a pointing hand, so the letter model may type that lookalike
# first. If a J/Z is found right after, the lookalike is replaced. UNTUNED guesses: edit if needed.
REPLACE = {"J": {"I"}, "Z": {"D", "G", "L", "X"}}
REPLACE_SLACK = 0.3     # a lookalike typed up to this long before the movement started still counts

model = joblib.load("asl_model_v2.joblib")
motion_model = joblib.load("motion_model2.joblib")

def one_thread(m):
    """Predicting ONE sample on many threads is slower than on one (thread start-up cost on every
    call). The saved models remember n_jobs=-1 from training, so switch it off."""
    for step in (getattr(m, "named_steps", None) or {"model": m}).values():
        if hasattr(step, "n_jobs"):
            step.n_jobs = 1
one_thread(model)
one_thread(motion_model)
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)      # same idea for the OpenMP/BLAS threads inside boosting / neural nets
except ImportError:
    pass

detector = None if SHARE else make_detector(min_conf=0.6)
motion_detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO, num_hands=2))

def speak(s):
    """Say s out loud: Windows' built-in voice, or espeak-ng on Linux. Never crashes if there is no voice."""
    s = "".join(ch for ch in s if ch.isalnum() or ch == " ")      # letters and spaces only
    if not s.strip():
        return
    try:
        if os.name == "nt":
            cmd = ["powershell", "-NoProfile", "-Command",
                   "Add-Type -AssemblyName System.Speech; "
                   f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{s}')"]
        else:
            cmd = ["espeak-ng", "-s", "130", s]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=0x08000000 if os.name == "nt" else 0)   # no console window flash
    except OSError:
        pass

def draw_text(img, msg, org, scale=1.0, color=(0, 255, 255), thick=2):
    """Text with a thick black outline, readable on any background."""
    cv2.putText(img, msg, org, cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.putText(img, msg, org, cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)

def tip_positions(res):
    """Index and pinky fingertip (x, y) of the first hand in the picture, or None."""
    if not res.hand_landmarks:
        return None
    lm = res.hand_landmarks[0]
    return (lm[8].x, lm[8].y, lm[20].x, lm[20].y)

def path_length(positions):
    """How far the index or pinky tip travelled (the larger of the two), in picture widths."""
    best = [0.0, 0.0]
    prev = None
    for p in positions:
        if p is not None and prev is not None:
            best[0] += float(np.hypot(p[0] - prev[0], p[1] - prev[1]))
            best[1] += float(np.hypot(p[2] - prev[2], p[3] - prev[3]))
        prev = p
    return max(best)

def evaluate(buf):
    """Look at the last 1.5 s. Returns (label, probability, probabilities text, movement).
    label is J, Z, IDLE, or STILL / NO HAND when the model was not even asked."""
    arr = np.array([f for _, f, _ in buf])
    move = path_length([p for _, _, p in buf])
    if arr.any(axis=1).mean() < MOTION_MIN_SEEN:
        return "NO HAND", 0.0, "", move
    if move < MOVE_MIN:
        return "STILL", 0.0, "", move
    clip = fill_gaps(clip_vector(arr).reshape(30, 130))
    probs = motion_model.predict_proba(np.array([clip]))[0]
    best = int(probs.argmax())
    detail = "   ".join(f"{c} {p:.0%}" for c, p in zip(motion_model.classes_, probs))
    return str(motion_model.classes_[best]), float(probs[best]), detail, move

class LatestFrame:
    """Reads the camera in its own thread and keeps ONLY the newest picture. The old loop read one
    picture per turn, so when the models were slower than the camera, the unread pictures queued up
    and the video showed the past (the lag). Now slow processing just skips pictures."""
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.lock, self.frame, self.alive = threading.Lock(), None, self.cap.isOpened()
        self.fps, self._n, self._t = 0.0, 0, time.time()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        fails = 0
        while self.alive:
            ok, f = self.cap.read()
            if not ok:
                fails += 1
                if fails > 100:                 # about 5 s without a picture: give up
                    self.alive = False
                time.sleep(0.05)
                continue
            fails = 0
            with self.lock:
                self.frame = f
            self._n += 1
            if time.time() - self._t >= 1.0:
                self.fps, self._n, self._t = self._n / (time.time() - self._t), 0, time.time()

    def read(self):
        """The newest picture not seen yet, or None."""
        with self.lock:
            f, self.frame = self.frame, None
        return f

    def close(self):
        self.alive = False
        self.thread.join(1.0)
        self.cap.release()

ms = {"motion": 0.0, "alpha": 0.0, "eval": 0.0}   # smoothed milliseconds per step, shown with d
def lap(key, t_start):
    ms[key] = 0.9 * ms[key] + 0.1 * (time.perf_counter() - t_start) * 1000

text = ""
history = deque()      # (time, letter) pairs from the last WINDOW seconds
locked = None          # letter just typed; hand must change or leave before typing again
last_seen = time.time()
last_typed = (0.0, "", "")             # (time, character, "alpha" or "motion")
mbuf = deque()                         # (time, 130 numbers, fingertip positions), last 1.5 s
t0, last_ts = time.time(), 0
last_eval, cooldown_until, alpha_block_until = 0.0, 0.0, 0.0
streak_label, streak = None, 0
flash_label, flash_until = "", 0.0
dbg_move, dbg_detail, debug = 0.0, "", False

camera = LatestFrame(SOURCE)
if not camera.alive:
    print(f"could not open the camera {SOURCE!r}")
cv2.namedWindow("ASL assistant", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow("ASL assistant", 1024, 768)       # starting size, 4:3 shape
fullscreen = False
fps_n, fps_t, app_fps = 0, time.time(), 0.0

while True:
    raw = camera.read()               # the newest picture only; None = nothing new yet
    if raw is None:
        if not camera.alive:
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        time.sleep(0.002)
        continue
    frame = cv2.flip(raw, 1)          # letters use the mirrored picture, J/Z use the raw one
    h, w = frame.shape[:2]
    now = time.time()

    # ---- J / Z: keep the last 1.5 s of raw (not mirrored) hand data, check it every 0.2 s ----
    ts = max(int((now - t0) * 1000), last_ts + 1)          # must always grow
    last_ts = ts
    t1 = time.perf_counter()
    res = motion_detector.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)), ts)
    lap("motion", t1)
    mbuf.append((now, frame_features(res), tip_positions(res)))
    while now - mbuf[0][0] > MOTION_DUR:
        mbuf.popleft()

    if (now >= cooldown_until and now - last_eval >= EVAL_EVERY
            and now - mbuf[0][0] >= MOTION_DUR * 0.9):         # the window is full
        last_eval = now
        t1 = time.perf_counter()
        label, prob, detail, dbg_move = evaluate(mbuf)
        lap("eval", t1)
        dbg_detail = detail
        if label in ("J", "Z") and prob >= MOTION_THRESH:
            streak = streak + 1 if label == streak_label else 1
            streak_label = label
        else:
            streak_label, streak = None, 0
        if streak >= CONFIRM:
            lt, ch, src = last_typed
            if (src == "alpha" and lt >= mbuf[0][0] - REPLACE_SLACK
                    and ch in REPLACE.get(label, ()) and text.endswith(ch)):
                text = text[:-1]                               # drop the lookalike letter
            text += label
            speak(label)
            last_typed = (now, label, "motion")
            flash_label, flash_until = label, now + 1.5
            cooldown_until = now + MOTION_COOLDOWN
            alpha_block_until = now + QUIET_AFTER
            mbuf.clear()
            history.clear()
            streak_label, streak = None, 0

    # ---- letters ----
    t1 = time.perf_counter()
    if SHARE:      # reuse the J/Z detector's hand, with x mirrored (the same as flipping the picture)
        lm = ([SimpleNamespace(x=1 - p.x, y=p.y, z=p.z) for p in res.hand_landmarks[0]]
              if res.hand_landmarks else None)
    else:
        lm = detect(detector, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    letter, top2 = None, ""
    if lm:
        last_seen = now
        probs = model.predict_proba([normalize(lm).flatten()])[0]
        order = probs.argsort()[::-1]
        first, second = order[0], order[1]
        top2 = (f"{model.classes_[first]} {probs[first]:.0%}   "
                f"{model.classes_[second]} {probs[second]:.0%}")
        if probs[first] >= MIN_CONF:
            letter = model.classes_[first]
        draw_hand(frame, lm)
    else:
        if now - last_seen > UNLOCK_AFTER:   # hand really left, not just a flicker
            locked = None
    lap("alpha", t1)

    if now < alpha_block_until:
        # just typed a J/Z: type no letters, and do not type the hand that is still showing
        history.clear()
        if letter:
            locked = letter
    else:
        history.append((now, letter))
        # keep one picture from just before the window starts; without it, a slower frame
        # rate (the camera now feeds two hand detectors) could never fill the window
        while len(history) > 1 and now - history[1][0] >= WINDOW:
            history.popleft()

        if history and now - history[0][0] >= WINDOW * 0.9:        # window is full
            top, n = Counter(l for _, l in history).most_common(1)[0]
            if top is not None and n / len(history) >= AGREE and top != locked:
                text += top
                speak(top)
                locked = top
                last_typed = (now, str(top), "alpha")

    # faint dark band behind the guess, then outlined text on top
    strip = frame.copy()
    cv2.rectangle(strip, (0, 0), (w, 56), (0, 0, 0), -1)
    frame = cv2.addWeighted(strip, 0.45, frame, 0.55, 0)
    if now < flash_until:
        draw_text(frame, f"Detected: {flash_label}", (12, 40), scale=1.1, color=(0, 255, 0))
    else:
        draw_text(frame, top2 or "-", (12, 40), scale=1.1, color=(0, 255, 255))
    if debug:
        draw_text(frame, f"move {dbg_move:.2f} (needs {MOVE_MIN:.2f})   {dbg_detail}", (12, 84),
                  scale=0.65, color=(255, 255, 255))
        draw_text(frame, f"camera {camera.fps:.0f} fps | app {app_fps:.0f} fps | {w}x{h}"
                         f"{' | ONE detector' if SHARE else ''}", (12, 118), scale=0.6, color=(255, 255, 255))
        draw_text(frame, f"ms per picture: hand {ms['motion']:.0f}  letters {ms['alpha']:.0f}"
                         f"  J/Z check {ms['eval']:.0f} (every {EVAL_EVERY:.1f} s)",
                  (12, 150), scale=0.6, color=(255, 255, 255))
    cv2.rectangle(frame, (0, h - 56), (w, h), (0, 0, 0), -1)
    draw_text(frame, text[-30:], (12, h - 16), scale=1.1, color=(255, 255, 255))
    draw_text(frame, "d = debug", (w - 130, h - 16), scale=0.6, color=(180, 180, 180), thick=1)
    cv2.imshow("ASL assistant", frame)
    fps_n += 1
    if now - fps_t >= 1.0:
        app_fps, fps_n, fps_t = fps_n / (now - fps_t), 0, now

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == 32:                 # SPACE: add a space
        text += " "
    elif key == 8:                  # BACKSPACE: delete last character
        text = text[:-1]
    elif key == 13:                 # ENTER: say the whole text
        speak(text)
    elif key == ord("c"):           # c: clear
        text = ""
    elif key == ord("d"):           # d: show / hide how much the hand moved and the J/Z guess
        debug = not debug
    elif key == ord("f"):           # f: toggle fullscreen
        fullscreen = not fullscreen
        cv2.setWindowProperty("ASL assistant", cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

camera.close()
motion_detector.close()
cv2.destroyAllWindows()
