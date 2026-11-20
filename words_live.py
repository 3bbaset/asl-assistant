"""Live helpers shared by try_words.py and quiz_words.py: speech, text drawing, and the word segmenter."""
import os, subprocess
import cv2
import numpy as np
from phrase_common import frame_features, clip_vector
from motion_common import fill_gaps

GONE_AFTER = 0.18   # seconds without hands = the word is finished
MIN_SIGN = 0.22     # shorter than this = a flicker, ignored
MAX_SIGN = 2.0      # longer than this = finish anyway


def speak(s):
    """Say s out loud (Windows voice, or espeak-ng on Linux). Never crashes."""
    s = "".join(ch for ch in s if ch.isalnum() or ch == " ")
    if not s.strip():
        return
    try:
        if os.name == "nt":
            cmd = ["powershell", "-NoProfile", "-Command", "Add-Type -AssemblyName System.Speech; "
                   f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{s}')"]
        else:
            cmd = ["espeak-ng", "-s", "130", s]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=0x08000000 if os.name == "nt" else 0)
    except OSError:
        pass


def draw_text(img, msg, org, scale=1.0, color=(0, 255, 255), thick=2):
    cv2.putText(img, msg, org, cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.putText(img, msg, org, cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)


class WordSegmenter:
    """Feed it one MediaPipe result per frame. A word = from "hands appear" to "hands gone"
    (or held steady for `still_after` seconds when enabled), trimmed like extract_words.py.
    update() returns the finished clip (3900 numbers) once, otherwise None."""
    def __init__(self, max_sign=MAX_SIGN, still_after=None, still_thresh=0.016, min_still_start=0.35):
        self.max_sign = max_sign
        self.still_after = still_after
        self.still_thresh = still_thresh
        self.min_still_start = min_still_start
        self.seq, self.start, self.last_hand = [], None, 0.0
        self.still_since, self.prev_feat = None, None

    @property
    def active(self):
        return self.start is not None

    def update(self, res, now):
        hand = bool(res.hand_landmarks)
        feat = frame_features(res)
        if hand:
            self.last_hand = now
            if self.start is None:
                self.start, self.seq = now, []
                self.still_since = None
            elif self.still_after is not None and self.prev_feat is not None:
                mask = (feat != 0) | (self.prev_feat != 0)
                delta = float(np.mean(np.abs(feat[mask] - self.prev_feat[mask]))) if mask.any() else 1.0
                if now - self.start >= self.min_still_start and delta < self.still_thresh:
                    self.still_since = self.still_since or now
                else:
                    self.still_since = None
            self.prev_feat = feat
        else:
            self.still_since = None
            self.prev_feat = None
        if self.start is None:
            return None
        self.seq.append(feat)
        still_done = (
            self.still_after is not None
            and self.still_since is not None
            and now - self.still_since >= self.still_after
        )
        if not ((not hand and now - self.last_hand >= GONE_AFTER) or now - self.start >= self.max_sign or still_done):
            return None
        long_enough = self.last_hand - self.start >= MIN_SIGN
        arr, self.seq, self.start = np.array(self.seq), [], None
        self.still_since = None
        self.prev_feat = None
        if not long_enough:
            return None
        seen = np.where(arr.any(axis=1))[0]
        if len(seen) == 0:
            return None
        return clip_vector(arr[seen[0]:seen[-1] + 1]).astype(np.float32)


def classify(model, clip):
    """clip (3900 numbers) -> (word, probability, list of (word, probability) best first)."""
    raw_clip = fill_gaps(clip.reshape(30, 130))
    # Check if model has a FunctionTransformer pipeline that expects (30, 130) raw clips:
    first_step = getattr(model, "steps", None)
    if first_step and hasattr(first_step[0][1], "func") and first_step[0][1].func.__name__ == "words_to_features":
        inp = np.array([raw_clip], dtype=np.float32)
    else:
        # Otherwise compute engineered feature vector:
        from words_common import words_to_features
        inp = words_to_features([raw_clip])
    probs = model.predict_proba(inp)[0]
    order = probs.argsort()[::-1]
    ranked = [(str(model.classes_[i]), float(probs[i])) for i in order]
    return ranked[0][0], ranked[0][1], ranked
