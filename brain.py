"""brain.py - ALL the recognition logic. No camera, no network, no screen.

One frame in (unflipped BGR image) -> one dict out.

Two models work together:
  * alphabet model (static letters): looks at ONE frame, mirrored like app.py does.
  * J/Z model (motion): looks at the last 1.5 s of hand landmarks, NOT mirrored,
    like the clips it was trained on.

It imports your existing helper files (common.py, phrase_common.py, motion_common.py),
so the numbers are computed exactly as in training.
"""
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from common import normalize
from phrase_common import frame_features, clip_vector
from motion_common import fill_gaps
from words_common import format_sentence
from words_live import WordSegmenter, classify as classify_word


class Config:
    # --- alphabet typing (same values as app.py) ---
    WINDOW = 0.6          # seconds a letter must be held
    AGREE = 0.8           # share of the window the same letter must win
    MIN_CONF = 0.6        # ignore guesses the model is unsure about
    UNLOCK_AFTER = 0.5    # seconds without a hand before the same letter can repeat
    # --- J/Z motion ---
    MOTION_DUR = 1.5      # clip length the model was trained on
    MOTION_EVERY = 0.4    # how often to ask the model while a hand is moving
    MOTION_MIN_SEEN = 0.6 # hand must be tracked in this share of the clip
    MOTION_MIN_PATH = 0.05  # fingertip must travel at least this far (cheap gate; UNTUNED)
    MOTION_THRESH = 0.6   # J or Z only if the model is at least this sure (as try_motion.py)
    MOTION_COOLDOWN = 1.5 # seconds of silence after a J/Z
    # J starts as an I hand and Z starts as a pointing hand, so the alphabet model may
    # type that lookalike first. If a J/Z follows, the lookalike is replaced.
    REPLACE = {"J": {"I"}, "Z": {"D", "G", "L", "X"}}
    REPLACE_WITHIN = 2.5  # seconds
    WORD_THRESH = 0.35    # minimum probability to accept a signed word (96%+ precision)
    MAX_TEXT = 500


class Brain:
    def __init__(self, alpha_model, motion_model, alpha_detect, motion_detect,
                 cfg=None, word_model=None):
        """alpha_detect(rgb_flipped) -> list of 21 landmarks or None
        motion_detect(rgb_unflipped, ts_ms) -> MediaPipe-style result (.hand_landmarks)"""
        self.cfg = cfg or Config()
        self.alpha_model, self.motion_model = alpha_model, motion_model
        self.word_model = word_model
        # The alphabet detector and the motion detector are separate MediaPipe objects, so in
        # spelling/alphabets mode they run in parallel (halves the time per frame).
        self._alpha_detect_raw, self.motion_detect = alpha_detect, motion_detect
        self.alpha_detect = self._alpha_detect_cached
        self._alpha_future = None
        self._pool = ThreadPoolExecutor(max_workers=1)
        self.mode = "spelling"            # "alphabets", "spelling", "words", or "sentences"
        self.word_seg = WordSegmenter()
        self.word_top3 = ""
        self.text = ""
        self.sentence_words = []
        self.sentence_text = ""
        self.sentence_last_time = 0.0
        self.sentence_spoken = True
        self.history = deque()           # (time, letter) from the last WINDOW seconds
        self.locked = None
        self.last_seen = time.time()
        self.alpha_block_until = 0.0
        self.mbuf = deque()              # (time, 130 features)
        self.last_eval = 0.0
        self.motion_cooldown_until = 0.0
        self.motion_info = (0.0, "")     # (time, "J 87%   IDLE 10%   Z 3%")
        self.last_typed = (0.0, "", "")  # (time, char, source)
        self.pending_say = False

    # ---------------- commands from the device ----------------
    def set_mode(self, mode):
        valid = ["alphabets", "spelling"] + (["words", "sentences"] if self.word_model is not None else [])
        if mode in valid:
            self.mode = mode
            self.mbuf.clear()
            self.history.clear()
            self.word_seg = WordSegmenter()
            if mode == "sentences":
                self.sentence_words.clear()
                self.sentence_text = ""
                self.sentence_spoken = True

    def command(self, cmd):
        if cmd == "space":
            if self.mode != "sentences":
                self.text += " "
        elif cmd == "backspace":
            if self.mode == "sentences":
                if self.sentence_words:
                    self.sentence_words.pop()
                    self.sentence_text = format_sentence(self.sentence_words)
                    self.sentence_spoken = False
            else:
                self.text = self.text[:-1]
        elif cmd == "clear":
            if self.mode == "sentences":
                self.sentence_words.clear()
                self.sentence_text = ""
                self.sentence_spoken = True
            else:
                self.text = ""
        elif cmd == "speak":
            self.pending_say = True
        elif cmd == "mode":
            modes = ["alphabets", "spelling"]
            if self.word_model is not None:
                modes.extend(["words", "sentences"])
            idx = (modes.index(self.mode) + 1) % len(modes) if self.mode in modes else 0
            self.set_mode(modes[idx])
        elif cmd in ("mode_alphabets", "alphabets", "1"):
            self.set_mode("alphabets")
        elif cmd in ("mode_spelling", "spelling", "2"):
            self.set_mode("spelling")
        elif cmd in ("mode_words", "words", "3"):
            self.set_mode("words")
        elif cmd in ("mode_sentences", "sentences", "4"):
            self.set_mode("sentences")

    def _type(self, ch, source, now):
        self.text = (self.text + ch)[-self.cfg.MAX_TEXT:]
        self.last_typed = (now, ch, source)
        self._typed.append(ch)

    def _alpha_detect_cached(self, rgb_flipped):
        fut, self._alpha_future = self._alpha_future, None
        return fut.result() if fut is not None else self._alpha_detect_raw(rgb_flipped)

    # ---------------- one frame ----------------
    def process(self, frame_bgr, now=None):
        now = time.time() if now is None else now
        t_start = time.time()
        self._typed = []
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.mode == "sentences" and self.word_model is not None:
            hand_pts, top2 = self._sentence_step(rgb, now)
            letter = None
            current_text = self.sentence_text
        elif self.mode == "words" and self.word_model is not None:
            hand_pts, top2 = self._word_step(rgb, now)
            letter = None
            current_text = self.text
        elif self.mode == "alphabets":
            flipped = cv2.flip(rgb, 1)
            self._alpha_future = self._pool.submit(self._alpha_detect_raw, flipped)
            hand_pts = self._motion_step_inspect(rgb, now)
            letter, top2 = self._alpha_step_inspect(flipped, now)
            self._alpha_future = None
            current_text = self.text
        else: # "spelling"
            flipped = cv2.flip(rgb, 1)
            self._alpha_future = self._pool.submit(self._alpha_detect_raw, flipped)
            hand_pts = self._motion_step(rgb, now)
            letter, top2 = self._alpha_step(flipped, now)
            self._alpha_future = None
            current_text = self.text
        say = current_text if self.pending_say else ""
        self.pending_say = False
        mt, mtxt = self.motion_info
        return {
            "text": current_text,
            "mode": self.mode,
            "words": list(self.sentence_words) if self.mode == "sentences" else [],
            "letter": letter,                 # what the alphabet model sees right now
            "top2": top2,
            "motion": mtxt if now - mt < 3.0 else "",
            "typed": "".join(self._typed),    # NEW characters since the last frame (speak these)
            "say": say,                       # whole text, only when the device sent "speak"
            "hand": hand_pts,                 # 21 (x, y) in the UNFLIPPED picture, or []
            "ms": round((time.time() - t_start) * 1000, 1),
        }

    def _sentence_step(self, rgb_unflipped, now):
        res = self.motion_detect(rgb_unflipped, int(now * 1000))
        pts = [[round(p.x, 4), round(p.y, 4)] for p in res.hand_landmarks[0]] if res.hand_landmarks else []
        clip = self.word_seg.update(res, now)
        if clip is not None:
            w_pred, w_prob, w_ranked = classify_word(self.word_model, clip)
            self.word_top3 = "   ".join(f"{w} {p:.0%}" for w, p in w_ranked[:3])
            self.motion_info = (now, self.word_top3)
            if w_pred != "IDLE" and w_prob >= self.cfg.WORD_THRESH:
                # Deduplication check: do not repeat immediately if same word within 1.2s
                if not (self.sentence_words and self.sentence_words[-1] == w_pred and now - self.sentence_last_time < 1.2):
                    self.sentence_words.append(w_pred)
                    self.sentence_last_time = now
                    self.sentence_spoken = False
                    self.sentence_text = format_sentence(self.sentence_words)
                    self._typed.append(f" {w_pred}")

        # Natural pause auto-speak (1.8s pause after signing sentence)
        if self.sentence_words and not self.sentence_spoken and not self.word_seg.active:
            if now - self.sentence_last_time >= 1.8:
                self.pending_say = True
                self.sentence_spoken = True

        if self.word_seg.active:
            top2 = f"SIGNING WORD... {now - self.word_seg.start:.1f}s"
        elif self.sentence_words:
            top2 = " -> ".join(self.sentence_words[-3:])
        else:
            top2 = self.word_top3 or "Sign successive words..."
        return pts, top2

    def _word_step(self, rgb_unflipped, now):
        res = self.motion_detect(rgb_unflipped, int(now * 1000))
        pts = [[round(p.x, 4), round(p.y, 4)] for p in res.hand_landmarks[0]] if res.hand_landmarks else []
        clip = self.word_seg.update(res, now)
        if clip is not None:
            w_pred, w_prob, w_ranked = classify_word(self.word_model, clip)
            self.word_top3 = "   ".join(f"{w} {p:.0%}" for w, p in w_ranked[:3])
            self.motion_info = (now, self.word_top3)
            if w_pred != "IDLE" and w_prob >= self.cfg.WORD_THRESH:
                prefix = " " if (self.text and not self.text.endswith(" ")) else ""
                self._type(prefix + w_pred + " ", "word", now)
        top2 = (f"SIGNING... {now - self.word_seg.start:.1f}s"
                if self.word_seg.active else (self.word_top3 or "Sign a word"))
        return pts, top2

    # ---------------- static letters ----------------
    def _alpha_step_inspect(self, rgb_flipped, now):
        c = self.cfg
        lm = self.alpha_detect(rgb_flipped)
        letter, top2 = None, ""
        if lm:
            probs = self.alpha_model.predict_proba([normalize(lm).flatten()])[0]
            order = probs.argsort()[::-1]
            a, b = order[0], order[1]
            top2 = (f"{self.alpha_model.classes_[a]} {probs[a]:.0%}   "
                    f"{self.alpha_model.classes_[b]} {probs[b]:.0%}")
            if probs[a] >= c.MIN_CONF:
                letter = self.alpha_model.classes_[a]
        return letter, top2

    # ---------------- static letters ----------------
    def _alpha_step(self, rgb_flipped, now):
        c = self.cfg
        lm = self.alpha_detect(rgb_flipped)
        letter, top2 = None, ""
        if lm:
            self.last_seen = now
            probs = self.alpha_model.predict_proba([normalize(lm).flatten()])[0]
            order = probs.argsort()[::-1]
            a, b = order[0], order[1]
            top2 = (f"{self.alpha_model.classes_[a]} {probs[a]:.0%}   "
                    f"{self.alpha_model.classes_[b]} {probs[b]:.0%}")
            if probs[a] >= c.MIN_CONF:
                letter = self.alpha_model.classes_[a]
        elif now - self.last_seen > c.UNLOCK_AFTER:      # hand really left, not a flicker
            self.locked = None

        if now < self.alpha_block_until:                 # just typed a J/Z: stay quiet,
            self.history.clear()                         # and don't re-type the hand
            if letter:                                   # that is still showing
                self.locked = letter
            return letter, top2

        self.history.append((now, letter))
        while self.history and now - self.history[0][0] > c.WINDOW:
            self.history.popleft()
        if self.history and now - self.history[0][0] >= c.WINDOW * 0.9:   # window is full
            top, n = Counter(l for _, l in self.history).most_common(1)[0]
            if top is not None and n / len(self.history) >= c.AGREE and top != self.locked:
                self._type(str(top), "alpha", now)
                self.locked = top
        return letter, top2

    # ---------------- J / Z ----------------
    def _path_len(self, arr):
        """Longest distance an index or pinky fingertip travelled in the clip."""
        best = 0.0
        for s in (0, 1):
            o = 65 * s
            seen = arr[:, o:o + 63].any(axis=1)
            for tip in (8, 20):
                pos = arr[:, o + 63:o + 65] + 0.15 * arr[:, o + 3 * tip:o + 3 * tip + 2]
                pos = pos[seen]
                if len(pos) >= 2:
                    best = max(best, float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()))
        return best

    def _motion_step(self, rgb_unflipped, now):
        c = self.cfg
        res = self.motion_detect(rgb_unflipped, int(now * 1000))
        feat = frame_features(res)
        self.mbuf.append((now, feat))
        while self.mbuf and now - self.mbuf[0][0] > c.MOTION_DUR:
            self.mbuf.popleft()
        pts = []
        if res.hand_landmarks:
            pts = [[round(p.x, 4), round(p.y, 4)] for p in res.hand_landmarks[0]]

        if now < self.motion_cooldown_until or now - self.last_eval < c.MOTION_EVERY:
            return pts
        if now - self.mbuf[0][0] < c.MOTION_DUR * 0.9:
            return pts
        arr = np.array([f for _, f in self.mbuf])
        if arr.any(axis=1).mean() < c.MOTION_MIN_SEEN or self._path_len(arr) < c.MOTION_MIN_PATH:
            return pts
        self.last_eval = now
        clip = fill_gaps(clip_vector(arr).reshape(30, 130))
        probs = self.motion_model.predict_proba(np.array([clip]))[0]
        k = int(probs.argmax())
        label = str(self.motion_model.classes_[k])
        self.motion_info = (now, "   ".join(f"{cl} {p:.0%}" for cl, p in
                                            zip(self.motion_model.classes_, probs)))
        if label != "IDLE" and probs[k] >= c.MOTION_THRESH:
            lt, ch, src = self.last_typed
            if src == "alpha" and now - lt <= c.REPLACE_WITHIN and ch in c.REPLACE.get(label, ()):
                self.text = self.text[:-1]               # drop the lookalike letter
            self._type(label, "motion", now)
            self.motion_cooldown_until = now + c.MOTION_COOLDOWN
            self.alpha_block_until = now + 1.0
            self.mbuf.clear()
        return pts

    def _motion_step_inspect(self, rgb_unflipped, now):
        c = self.cfg
        res = self.motion_detect(rgb_unflipped, int(now * 1000))
        feat = frame_features(res)
        self.mbuf.append((now, feat))
        while self.mbuf and now - self.mbuf[0][0] > c.MOTION_DUR:
            self.mbuf.popleft()
        pts = []
        if res.hand_landmarks:
            pts = [[round(p.x, 4), round(p.y, 4)] for p in res.hand_landmarks[0]]

        if now < self.motion_cooldown_until or now - self.last_eval < c.MOTION_EVERY:
            return pts
        if now - self.mbuf[0][0] < c.MOTION_DUR * 0.9:
            return pts
        arr = np.array([f for _, f in self.mbuf])
        if arr.any(axis=1).mean() < c.MOTION_MIN_SEEN or self._path_len(arr) < c.MOTION_MIN_PATH:
            return pts
        self.last_eval = now
        clip = fill_gaps(clip_vector(arr).reshape(30, 130))
        probs = self.motion_model.predict_proba(np.array([clip]))[0]
        k = int(probs.argmax())
        label = str(self.motion_model.classes_[k])
        self.motion_info = (now, "   ".join(f"{cl} {p:.0%}" for cl, p in
                                            zip(self.motion_model.classes_, probs)))
        # In inspection/alphabets mode, we do NOT append to text
        return pts
