"""Live helpers shared by try_words.py and quiz_words.py: speech, text drawing, and the word segmenter."""
import os, subprocess
import cv2
import numpy as np
from phrase_common import frame_features, clip_vector
from motion_common import fill_gaps

GONE_AFTER = 0.4    # seconds without hands = the word is finished
MIN_SIGN = 0.3      # shorter than this = a flicker, ignored
MAX_SIGN = 4.0      # longer than this = finish anyway


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
    """Feed it one MediaPipe result per frame. A word = from "hands appear" to "hands gone",
    trimmed the same way as the training videos (extract_words.py).
    update() returns the finished clip (3900 numbers) once, otherwise None."""
    def __init__(self):
        self.seq, self.start, self.last_hand = [], None, 0.0

    @property
    def active(self):
        return self.start is not None

    def update(self, res, now):
        hand = bool(res.hand_landmarks)
        if hand:
            self.last_hand = now
            if self.start is None:
                self.start, self.seq = now, []
        if self.start is None:
            return None
        self.seq.append(frame_features(res))
        if not ((not hand and now - self.last_hand >= GONE_AFTER) or now - self.start >= MAX_SIGN):
            return None
        long_enough = self.last_hand - self.start >= MIN_SIGN
        arr, self.seq, self.start = np.array(self.seq), [], None
        if not long_enough:
            return None
        seen = np.where(arr.any(axis=1))[0]
        return clip_vector(arr[seen[0]:seen[-1] + 1]).astype(np.float32)


def classify(model, clip):
    """clip (3900 numbers) -> (word, probability, list of (word, probability) best first)."""
    probs = model.predict_proba(np.array([fill_gaps(clip.reshape(30, 130))]))[0]
    order = probs.argsort()[::-1]
    ranked = [(str(model.classes_[i]), float(probs[i])) for i in order]
    return ranked[0][0], ranked[0][1], ranked
