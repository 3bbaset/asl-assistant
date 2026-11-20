import argparse
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
from words_common import MODEL_FILE as WORD_MODEL_FILE, format_sentence
from words_live import WordSegmenter, classify as classify_word
from arsl_common import classify_arsl_word
import translator

parser = argparse.ArgumentParser(description="ASL Assistant: Multi-Mode Recognition (Alphabets, Spelling, Words, Sentences)")
parser.add_argument("--source", default=0, help="Camera index or video stream URL")
parser.add_argument("--share", action="store_true", help="Share one hand detector for both models (faster)")
parser.add_argument("--mode", choices=["alphabets", "spelling", "words", "sentences"], default=None,
                    help="Starting mode: alphabets (inspection), spelling (fingerspelling typing), words (whole-word ASL), sentences (continuous ASL)")
parser.add_argument("--words", action="store_true", help="Start in words mode (legacy shortcut)")
parser.add_argument("--sentences", action="store_true", help="Start in continuous sentences mode")
parser.add_argument("--full-vocab", action="store_true", help="Use 2,021-word model instead of high-accuracy conversational")
args, _ = parser.parse_known_args()

SOURCE = int(args.source) if str(args.source).isdigit() else args.source
SHARE = args.share or ("--share" in sys.argv)
FULL_VOCAB = args.full_vocab or ("--full-vocab" in sys.argv)
ACTIVE_WORD_MODEL = "word_model_full_2021.joblib" if (FULL_VOCAB and os.path.exists("word_model_full_2021.joblib")) else WORD_MODEL_FILE
WORD_THRESH = 0.20 if FULL_VOCAB else 0.35  # minimum confidence to accept a signed word
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
word_model = joblib.load(ACTIVE_WORD_MODEL) if os.path.exists(ACTIVE_WORD_MODEL) else None
arsl_alpha_model = joblib.load("arsl_alpha_model.joblib") if os.path.exists("arsl_alpha_model.joblib") else None
arsl_word_model = joblib.load("arsl_word_model_30.joblib") if os.path.exists("arsl_word_model_30.joblib") else None
sign_lang = "asl"
out_lang = "en"

def one_thread(m):
    """Predicting ONE sample on many threads is slower than on one (thread start-up cost on every
    call). The saved models remember n_jobs=-1 from training, so switch it off."""
    if m is None:
        return
    for step in (getattr(m, "named_steps", None) or {"model": m}).values():
        if hasattr(step, "n_jobs"):
            step.n_jobs = 1
        for sub in getattr(step, "estimators_", ()):
            if hasattr(sub, "n_jobs"):
                sub.n_jobs = 1
one_thread(model)
one_thread(motion_model)
one_thread(word_model)
one_thread(arsl_alpha_model)
one_thread(arsl_word_model)
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

if args.mode:
    mode = args.mode
elif args.sentences or ("--sentences" in sys.argv):
    mode = "sentences" if word_model is not None else "spelling"
elif args.words or ("--words" in sys.argv):
    mode = "words" if word_model is not None else "spelling"
else:
    mode = "spelling"

word_seg = WordSegmenter()
word_top3 = ""
sentence_words = []
sentence_text = ""
sentence_last_time = 0.0
sentence_spoken = True

def switch_mode(new_mode, cur_time):
    global mode, word_seg, streak_label, streak, flash_label, flash_until
    global sentence_words, sentence_text, sentence_spoken
    has_word = (arsl_word_model is not None) if sign_lang == "arsl" else (word_model is not None)
    if new_mode in ("words", "sentences") and not has_word:
        flash_label, flash_until = "Word model not found!", cur_time + 1.5
        return
    mode = new_mode
    mbuf.clear()
    history.clear()
    streak_label, streak = None, 0
    word_seg = WordSegmenter(max_sign=3.0 if new_mode == "sentences" else 4.0)
    if mode == "sentences":
        sentence_words.clear()
        sentence_text = ""
        sentence_spoken = True
    mode_titles = {
        "alphabets": "1: ALPHABETS (Live Inspection)",
        "spelling": "2: SPELLING (Fingerspelling Typing)",
        "words": "3: WORDS (Whole-Word)",
        "sentences": "4: SENTENCES (Continuous)",
    }
    flash_label, flash_until = f"{sign_lang.upper()}>{out_lang.upper()} {mode_titles.get(mode, mode.upper())}", cur_time + 1.5

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

    # ---- J / Z or WORDS: motion_detector runs on the raw (not mirrored) picture ----
    ts = max(int((now - t0) * 1000), last_ts + 1)          # must always grow
    last_ts = ts
    t1 = time.perf_counter()
    res = motion_detector.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)), ts)
    lap("motion", t1)

    active_wm = arsl_word_model if sign_lang == "arsl" else word_model
    if mode in ("words", "sentences") and active_wm is not None:
        for h_lm in res.hand_landmarks:
            draw_hand(frame, [SimpleNamespace(x=1 - p.x, y=p.y, z=p.z) for p in h_lm])
        t1 = time.perf_counter()
        w_clip = word_seg.update(res, now)
        if w_clip is not None:
            if sign_lang == "arsl":
                w_id, w_prob, w_ranked = classify_arsl_word(active_wm, w_clip)
                word_top3 = "   ".join(f"{translator.translate_arsl_word(w, 'en')[:10]} {p:.0%}" for w, p in w_ranked[:3])
                dbg_detail = word_top3
                w_disp = translator.translate_arsl_word(w_id, out_lang)
                w_en = translator.translate_arsl_word(w_id, "en")
                if w_prob >= 0.25:
                    if mode == "words":
                        text = w_disp
                        speak(w_disp if out_lang == "en" else w_en)
                        last_typed = (now, w_disp, "word")
                        flash_label, flash_until = f"{w_en} ({w_prob:.0%})", now + 2.0
                    else:
                        if not (sentence_words and sentence_words[-1] == w_en and now - sentence_last_time < 1.2):
                            sentence_words.append(w_en)
                            sentence_last_time = now
                            sentence_spoken = False
                            sentence_text = format_sentence(sentence_words)
                            flash_label, flash_until = f"+ {w_en} ({len(sentence_words)} words)", now + 1.8
                else:
                    flash_label, flash_until = f"? ({w_en} {w_prob:.0%})", now + 1.5
            else:
                w_pred, w_prob, w_ranked = classify_word(active_wm, w_clip)
                word_top3 = "   ".join(f"{w} {p:.0%}" for w, p in w_ranked[:3])
                dbg_detail = word_top3
                if w_pred != "IDLE" and w_prob >= WORD_THRESH:
                    if mode == "words":
                        text = w_pred
                        speak(w_pred)
                        last_typed = (now, w_pred, "word")
                        flash_label, flash_until = f"{w_pred} ({w_prob:.0%})", now + 2.0
                    else:  # sentences mode: successive words accumulation
                        if not (sentence_words and sentence_words[-1] == w_pred and now - sentence_last_time < 1.2):
                            sentence_words.append(w_pred)
                            sentence_last_time = now
                            sentence_spoken = False
                            sentence_text = format_sentence(sentence_words)
                            flash_label, flash_until = f"+ {w_pred} ({len(sentence_words)} words)", now + 1.8
                else:
                    flash_label, flash_until = f"? ({w_pred} {w_prob:.0%})", now + 1.5
        lap("eval", t1)

        # In sentences mode: auto-speak when user pauses for 1.8s after signing words
        if mode == "sentences" and sentence_words and not sentence_spoken and not word_seg.active:
            if now - sentence_last_time >= 1.8:
                speak(sentence_text)
                sentence_spoken = True
                flash_label, flash_until = f"Spoken: \"{sentence_text}\"", now + 2.5

        if mode == "words":
            top2 = (f"SIGNING WORD... {now - word_seg.start:.1f}s (drop hands when done)"
                    if word_seg.active else (word_top3 or "Raise hands & sign a word"))
        else: # sentences
            if word_seg.active:
                top2 = f"SIGNING WORD {len(sentence_words) + 1}... {now - word_seg.start:.1f}s"
            elif sentence_words:
                top2 = " -> ".join(sentence_words[-3:])
            else:
                top2 = "Sign successive words..."
    else:
        # ---- Mode 1: alphabets (inspection) OR Mode 2: spelling (typing) ----
        mbuf.append((now, frame_features(res), tip_positions(res)))
        while now - mbuf[0][0] > MOTION_DUR:
            mbuf.popleft()

        motion_label, motion_prob = None, 0.0
        if (now >= cooldown_until and now - last_eval >= EVAL_EVERY
                and now - mbuf[0][0] >= MOTION_DUR * 0.9):         # the window is full
            last_eval = now
            t1 = time.perf_counter()
            label, prob, detail, dbg_move = evaluate(mbuf)
            lap("eval", t1)
            dbg_detail = detail
            if label in ("J", "Z") and prob >= MOTION_THRESH:
                motion_label, motion_prob = label, prob
                streak = streak + 1 if label == streak_label else 1
                streak_label = label
            else:
                streak_label, streak = None, 0

            if mode == "spelling" and streak >= CONFIRM:
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
        active_am = arsl_alpha_model if (sign_lang == "arsl" and arsl_alpha_model is not None) else model
        if lm and active_am is not None:
            last_seen = now
            probs = active_am.predict_proba([normalize(lm).flatten()])[0]
            order = probs.argsort()[::-1]
            first, second = order[0], order[1]
            c1 = translator.translate_arsl_letter(active_am.classes_[first], "en") if sign_lang == "arsl" else str(active_am.classes_[first])
            c2 = translator.translate_arsl_letter(active_am.classes_[second], "en") if sign_lang == "arsl" else str(active_am.classes_[second])
            top2 = f"{c1} {probs[first]:.0%}   {c2} {probs[second]:.0%}"
            if probs[first] >= MIN_CONF:
                letter = translator.translate_arsl_letter(active_am.classes_[first], out_lang) if sign_lang == "arsl" else str(active_am.classes_[first])
            draw_hand(frame, lm)
        else:
            if now - last_seen > UNLOCK_AFTER:   # hand really left, not just a flicker
                locked = None
        lap("alpha", t1)

        # In Mode 2 (spelling): fingerspelling typing accumulation
        if mode == "spelling":
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
        elif mode == "alphabets":
            # In Mode 1 (alphabets): inspection only, show motion if J/Z gesture detected
            if motion_label:
                top2 = f"Motion: {motion_label} {motion_prob:.0%}"

    # faint dark band behind the guess, then outlined text on top
    strip = frame.copy()
    cv2.rectangle(strip, (0, 0), (w, 56), (0, 0, 0), -1)
    frame = cv2.addWeighted(strip, 0.45, frame, 0.55, 0)
    if now < flash_until:
        draw_text(frame, f"{flash_label}", (12, 40), scale=1.0, color=(0, 255, 0))
    elif mode == "alphabets":
        msg = f"Letter: {top2}" if top2 else "Show a hand sign (A-Z, J/Z)"
        draw_text(frame, msg, (12, 40), scale=0.9, color=(0, 255, 255))
    elif mode == "spelling":
        msg = f"Hold: {top2}" if top2 else "Fingerspell to type..."
        draw_text(frame, msg, (12, 40), scale=0.9, color=(0, 255, 255))
    elif mode == "words":
        draw_text(frame, top2 or "Raise hands & sign a word", (12, 40), scale=0.9, color=(0, 255, 255))
    elif mode == "sentences":
        if sentence_words:
            w_chain = " -> ".join(sentence_words[-3:])
            timer_hint = " [Spoken]" if sentence_spoken else f" [Pause {max(0.0, 1.8 - (now - sentence_last_time)):.1f}s to speak]"
            draw_text(frame, f"Chain: {w_chain}{timer_hint}", (12, 40), scale=0.85, color=(0, 255, 255))
        else:
            draw_text(frame, "Sign successive words (continuous sentence)...", (12, 40), scale=0.85, color=(0, 255, 255))

    mode_badges = {
        "alphabets": "1: ALPHABETS",
        "spelling": "2: SPELLING",
        "words": "3: WORDS",
        "sentences": "4: SENTENCES",
    }
    badge = f"[{sign_lang.upper()}>{out_lang.upper()} {mode_badges.get(mode, mode.upper())}]"
    draw_text(frame, badge, (w - 300, 40), scale=0.65, color=(0, 220, 255), thick=2)

    if debug:
        draw_text(frame, f"move {dbg_move:.2f} (needs {MOVE_MIN:.2f})   {dbg_detail}", (12, 84),
                  scale=0.65, color=(255, 255, 255))
        draw_text(frame, f"camera {camera.fps:.0f} fps | app {app_fps:.0f} fps | {w}x{h}"
                         f"{' | ONE detector' if SHARE else ''}", (12, 118), scale=0.6, color=(255, 255, 255))
        draw_text(frame, f"ms per picture: hand {ms['motion']:.0f}  letters {ms['alpha']:.0f}"
                         f"  J/Z check {ms['eval']:.0f} (every {EVAL_EVERY:.1f} s)",
                  (12, 150), scale=0.6, color=(255, 255, 255))

    cv2.rectangle(frame, (0, h - 56), (w, h), (0, 0, 0), -1)
    if mode == "alphabets":
        display_text = f"Buffer: {text[-24:]}" if text else "(Inspection Mode - typing paused)"
        draw_text(frame, display_text, (12, h - 16), scale=0.85, color=(180, 180, 180))
    elif mode == "sentences":
        if out_lang == "ar" and sentence_text:
            ar_s = translator.shape_arabic(translator.translate_en_to_ar(sentence_text))
            from PIL import Image, ImageDraw, ImageFont
            pil_f = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fnt = ImageFont.truetype(r"C:\Windows\Fonts\tahoma.ttf", 22) if os.path.exists(r"C:\Windows\Fonts\tahoma.ttf") else ImageFont.load_default()
            ImageDraw.Draw(pil_f).text((12, h - 46), ar_s, font=fnt, fill=(255, 255, 255))
            frame = cv2.cvtColor(np.asarray(pil_f), cv2.COLOR_RGB2BGR)
        else:
            display_text = f'Sentence: "{sentence_text}"' if sentence_text else 'Sentence: (waiting for words...)'
            draw_text(frame, display_text, (12, h - 16), scale=0.95, color=(255, 255, 255))
    else:
        raw_shown = text[-30:]
        if out_lang == "ar" and raw_shown:
            ar_s = translator.shape_arabic(raw_shown if any("\u0600" <= ch <= "\u06ff" for ch in raw_shown) else translator.translate_en_to_ar(raw_shown))
            from PIL import Image, ImageDraw, ImageFont
            pil_f = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fnt = ImageFont.truetype(r"C:\Windows\Fonts\tahoma.ttf", 22) if os.path.exists(r"C:\Windows\Fonts\tahoma.ttf") else ImageFont.load_default()
            ImageDraw.Draw(pil_f).text((12, h - 46), ar_s, font=fnt, fill=(255, 255, 255))
            frame = cv2.cvtColor(np.asarray(pil_f), cv2.COLOR_RGB2BGR)
        else:
            draw_text(frame, raw_shown, (12, h - 16), scale=1.1, color=(255, 255, 255))

    shortcuts = "1-4:Mode | l:ASL/ArSL | t:EN/AR | m:cycle"
    draw_text(frame, shortcuts, (w - 430, h - 16), scale=0.52, color=(0, 255, 180), thick=1)
    cv2.imshow("ASL assistant", frame)
    fps_n += 1
    if now - fps_t >= 1.0:
        app_fps, fps_n, fps_t = fps_n / (now - fps_t), 0, now

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("l"):
        sign_lang = "arsl" if sign_lang == "asl" else "asl"
        if sign_lang == "arsl" and out_lang == "en":
            out_lang = "ar"
        text = ""
        sentence_words.clear()
        sentence_text = ""
        flash_label, flash_until = f"Input: {sign_lang.upper()} -> {out_lang.upper()}", now + 1.5
    elif key == ord("t"):
        out_lang = "ar" if out_lang == "en" else "en"
        flash_label, flash_until = f"Output Language: {out_lang.upper()}", now + 1.5
    elif key == ord("1"):
        switch_mode("alphabets", now)
    elif key == ord("2"):
        switch_mode("spelling", now)
    elif key == ord("3"):
        switch_mode("words", now)
    elif key == ord("4"):
        switch_mode("sentences", now)
    elif key in (ord("m"), 9):  # m or TAB: cycle between available modes
        available = ["alphabets", "spelling"] + (["words", "sentences"] if word_model is not None else [])
        idx = (available.index(mode) + 1) % len(available) if mode in available else 0
        switch_mode(available[idx], now)
    elif key == 32:                 # SPACE: add a space
        if mode != "sentences":
            text += " "
    elif key == 8:                  # BACKSPACE: delete last character or word
        if mode == "sentences":
            if sentence_words:
                sentence_words.pop()
                sentence_text = format_sentence(sentence_words)
                sentence_spoken = False
        else:
            text = text[:-1]
    elif key == 13:                 # ENTER: say the whole text or sentence
        if mode == "sentences":
            if sentence_text:
                speak(sentence_text)
                sentence_spoken = True
                flash_label, flash_until = f"Spoken: \"{sentence_text}\"", now + 2.0
        else:
            speak(text)
    elif key == ord("c"):           # c: clear
        if mode == "sentences":
            sentence_words.clear()
            sentence_text = ""
            sentence_spoken = True
        else:
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
