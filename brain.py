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
from arsl_common import classify_arsl_word
from translator import (
    format_arabic_sentence,
    translate_ar_to_en,
    translate_arsl_letter,
    translate_arsl_word,
    translate_en_to_ar,
)


class Config:
    # --- alphabet typing (tuned for responsive real-time speed) ---
    WINDOW = 0.35         # seconds a letter must be held
    AGREE = 0.75          # share of the window the same letter must win
    MIN_CONF = 0.55       # ignore guesses the model is unsure about when typing
    PREVIEW_CONF = 0.35   # minimum confidence to show live preview in GUESS box
    UNLOCK_AFTER = 0.28   # seconds without a hand before the same letter can repeat
    # --- J/Z motion ---
    MOTION_DUR = 1.5      # clip length the model was trained on
    MOTION_EVERY = 0.30   # how often to ask the model while a hand is moving
    MOTION_MIN_SEEN = 0.6 # hand must be tracked in this share of the clip
    MOTION_MIN_PATH = 0.05  # fingertip must travel at least this far (cheap gate; UNTUNED)
    MOTION_THRESH = 0.6   # J or Z only if the model is at least this sure (as try_motion.py)
    MOTION_COOLDOWN = 1.0 # seconds of silence after a J/Z
    # J starts as an I hand and Z starts as a pointing hand, so the alphabet model may
    # type that lookalike first. If a J/Z follows, the lookalike is replaced.
    REPLACE = {"J": {"I"}, "Z": {"D", "G", "L", "X"}}
    REPLACE_WITHIN = 1.8  # seconds
    WORD_THRESH = 0.35    # minimum probability to accept a signed word
    MAX_TEXT = 500


class Brain:
    def __init__(self, alpha_model, motion_model, alpha_detect, motion_detect,
                 cfg=None, word_model=None, word_model_full=None,
                 arsl_alpha_model=None, arsl_word_model=None):
        """alpha_detect(rgb_flipped) -> list of 21 landmarks or None
        motion_detect(rgb_unflipped, ts_ms) -> MediaPipe-style result (.hand_landmarks)"""
        self.cfg = cfg or Config()
        self.alpha_model, self.motion_model = alpha_model, motion_model
        self.word_model_30 = word_model
        self.word_model_full = word_model_full
        self.word_model = word_model
        self.arsl_alpha_model = arsl_alpha_model
        self.arsl_word_model = arsl_word_model
        self.sign_lang = "asl"            # "asl" or "arsl"
        self.out_lang = "en"              # "en" or "ar"
        self.word_vocab_mode = "30"       # "30" (core 80-86% acc) or "2000" (full vocabulary)
        # The alphabet detector and the motion detector are separate MediaPipe objects, so in
        # spelling/alphabets mode they run in parallel (halves the time per frame).
        self._alpha_detect_raw, self.motion_detect = alpha_detect, motion_detect
        self.alpha_detect = self._alpha_detect_cached
        self._alpha_future = None
        self._pool = ThreadPoolExecutor(max_workers=1)
        self.mode = "spelling"            # "alphabets", "spelling", "words", or "sentences"
        self.word_seg = WordSegmenter()
        self.word_top3 = ""
        self.word_top3_ar = ""
        self.last_top3 = []
        self.last_hint_ar = ""
        self.last_guess = ""
        self.last_guess_ar = ""
        self.last_word_guess = (0.0, "", "")
        self.text = ""
        self.sentence_words = []
        self.sentence_words_ar = []
        self.sentence_text = ""
        self.sentence_last_time = 0.0
        self.sentence_spoken = True
        self._arsl_last_word_id = None
        self._trans_cache = ("", "")
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

    def set_sign_lang(self, lang):
        lang = str(lang).strip().lower()
        if lang not in ("asl", "arsl") or lang == self.sign_lang:
            return
        self.sign_lang = lang
        if lang == "arsl" and self.out_lang == "en":
            self.out_lang = "ar"
        elif lang == "asl" and self.mode in ("words", "sentences"):
            pass
        self.mbuf.clear()
        self.history.clear()
        self.locked = None
        self.text = ""
        self._arsl_last_word_id = None
        self.sentence_words.clear()
        self.sentence_words_ar.clear()
        self.sentence_text = ""
        self.sentence_spoken = True
        self.word_top3 = ""
        self.word_top3_ar = ""
        self.last_top3 = []
        self.last_hint_ar = ""
        self.last_guess = ""
        self.last_guess_ar = ""
        self.last_word_guess = (0.0, "", "")
        self.word_seg = WordSegmenter(max_sign=1.6 if self.sign_lang == "arsl" else 2.0)

    def set_out_lang(self, lang):
        lang = str(lang).strip().lower()
        if lang not in ("en", "ar") or lang == self.out_lang:
            return
        self.out_lang = lang
        self._trans_cache = ("", "")
        if self.sign_lang == "arsl":
            if self.mode == "words" and self._arsl_last_word_id:
                self.text = translate_arsl_word(self._arsl_last_word_id, self.out_lang)
            elif self.mode == "sentences":
                if self.out_lang == "ar":
                    self.sentence_text = format_arabic_sentence(self.sentence_words_ar)
                else:
                    self.sentence_text = format_sentence(self.sentence_words)

    def toggle_word_model(self):
        """Toggle between high-accuracy 30-word model and full 2,000-word model."""
        if self.word_model_full is None:
            return "30"
        if self.word_vocab_mode == "30":
            self.word_model = self.word_model_full
            self.word_vocab_mode = "2000"
            self.cfg.WORD_THRESH = 0.20   # Lower threshold for 2000-class spread
        else:
            self.word_model = self.word_model_30
            self.word_vocab_mode = "30"
            self.cfg.WORD_THRESH = 0.35
        self.word_top3 = ""
        return self.word_vocab_mode

    # ---------------- commands from the device ----------------
    def set_mode(self, mode):
        # Allow passing "words_30", "words_2000", "sentences", "sentences_2000"
        if mode in ("words_30", "words [30]"):
            self.word_model = self.word_model_30
            self.word_vocab_mode = "30"
            self.cfg.WORD_THRESH = 0.35
            mode = "words"
        elif mode in ("words_2000", "words [2000]", "words_full"):
            if self.word_model_full is not None:
                self.word_model = self.word_model_full
                self.word_vocab_mode = "2000"
                self.cfg.WORD_THRESH = 0.20
            mode = "words"
        elif mode in ("sentences", "sentences_30"):
            self.word_model = self.word_model_30
            self.word_vocab_mode = "30"
            self.cfg.WORD_THRESH = 0.35
            mode = "sentences"
        elif mode in ("sentences_2000", "sentences_full"):
            if self.word_model_full is not None:
                self.word_model = self.word_model_full
                self.word_vocab_mode = "2000"
                self.cfg.WORD_THRESH = 0.20
            mode = "sentences"
        has_word = (self.word_model is not None) or (self.arsl_word_model is not None)
        valid = ["alphabets", "spelling"] + (["words", "sentences"] if has_word else [])
        if mode in valid:
            self.mode = mode
            self.mbuf.clear()
            self.history.clear()
            self.locked = None
            self.last_top3 = []
            self.word_top3 = ""
            self.last_guess = ""
            self.last_guess_ar = ""
            self.last_word_guess = (0.0, "", "")
            self._arsl_last_word_id = None
            self.word_seg = WordSegmenter(max_sign=1.6 if self.sign_lang == "arsl" else 2.0)
            if mode == "words":
                self.text = ""
            elif mode == "sentences":
                self.sentence_words.clear()
                self.sentence_words_ar.clear()
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
                    if self.sentence_words_ar:
                        self.sentence_words_ar.pop()
                    if self.sign_lang == "arsl" and self.out_lang == "ar":
                        self.sentence_text = format_arabic_sentence(self.sentence_words_ar)
                    else:
                        self.sentence_text = format_sentence(self.sentence_words)
                    self.sentence_spoken = False
            else:
                self.text = self.text[:-1]
        elif cmd == "clear":
            self._arsl_last_word_id = None
            if self.mode == "sentences":
                self.sentence_words.clear()
                self.sentence_words_ar.clear()
                self.sentence_text = ""
                self.sentence_spoken = True
            else:
                self.text = ""
        elif cmd == "speak":
            self.pending_say = True
        elif cmd == "mode":
            has_word = (self.word_model is not None) or (self.arsl_word_model is not None)
            modes = ["alphabets", "spelling"]
            if has_word:
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
        elif cmd in ("toggle_vocab", "vocab", "switch_vocab"):
            self.toggle_word_model()
        elif cmd in ("sign:asl", "sign:arsl"):
            self.set_sign_lang(cmd.split(":", 1)[1])
        elif cmd == "toggle_sign":
            self.set_sign_lang("arsl" if self.sign_lang == "asl" else "asl")
        elif cmd in ("lang:en", "lang:ar"):
            self.set_out_lang(cmd.split(":", 1)[1])
        elif cmd == "toggle_lang":
            self.set_out_lang("ar" if self.out_lang == "en" else "en")

    def _type(self, ch, source, now):
        self.text = (self.text + ch)[-self.cfg.MAX_TEXT:]
        self.last_typed = (now, ch, source)
        self._typed.append(ch)

    def _alpha_detect_cached(self, rgb_flipped):
        fut, self._alpha_future = self._alpha_future, None
        return fut.result() if fut is not None else self._alpha_detect_raw(rgb_flipped)

    def _translate_cached(self, text):
        if not text:
            return ""
        ck = (self.sign_lang, self.out_lang, text)
        if self._trans_cache[0] == ck:
            return self._trans_cache[1]
        if self.sign_lang == "asl" and self.out_lang == "ar":
            val = translate_en_to_ar(text)
        elif self.sign_lang == "arsl" and self.out_lang == "ar":
            val = text
        elif self.sign_lang == "arsl" and self.out_lang == "en":
            val = translate_ar_to_en(text) if any("\u0600" <= ch <= "\u06ff" for ch in text) else text
        else:
            val = ""
        self._trans_cache = (ck, val)
        return val

    # ---------------- one frame ----------------
    def process(self, frame_bgr, now=None):
        now = time.time() if now is None else now
        t_start = time.time()
        self._typed = []
        self._hands = []
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        active_wmodel = self.arsl_word_model if self.sign_lang == "arsl" else self.word_model
        if self.mode == "sentences" and active_wmodel is not None:
            hand_pts, top2 = self._sentence_step(rgb, now)
            letter = None
            current_text = self.sentence_text
        elif self.mode == "words" and active_wmodel is not None:
            hand_pts, top2 = self._word_step(rgb, now)
            letter = None
            current_text = self.text
        elif self.mode == "alphabets":
            flipped = cv2.flip(rgb, 1)
            if self.sign_lang == "arsl":
                lm = self._alpha_detect_raw(flipped)
                hand_pts = [[round(1.0 - p.x, 4), round(p.y, 4)] for p in lm] if lm else []
                self._hands = [hand_pts] if hand_pts else []
                letter, top2 = self._alpha_step_inspect(flipped, now, lm=lm)
            else:
                self._alpha_future = self._pool.submit(self._alpha_detect_raw, flipped)
                hand_pts = self._motion_step_inspect(rgb, now)
                letter, top2 = self._alpha_step_inspect(flipped, now)
                self._alpha_future = None
            current_text = self.text
        else: # "spelling"
            flipped = cv2.flip(rgb, 1)
            if self.sign_lang == "arsl":
                lm = self._alpha_detect_raw(flipped)
                hand_pts = [[round(1.0 - p.x, 4), round(p.y, 4)] for p in lm] if lm else []
                self._hands = [hand_pts] if hand_pts else []
                letter, top2 = self._alpha_step(flipped, now, lm=lm)
            else:
                self._alpha_future = self._pool.submit(self._alpha_detect_raw, flipped)
                hand_pts = self._motion_step(rgb, now)
                letter, top2 = self._alpha_step(flipped, now)
                self._alpha_future = None
            current_text = self.text

        translated_text = self._translate_cached(current_text)
        typed_str = "".join(self._typed)
        if self.sign_lang == "asl" and self.out_lang == "ar":
            say_src = translated_text if self.pending_say else ""
            if typed_str and self.mode in ("words", "alphabets"):
                typed_str = translate_en_to_ar(typed_str)
        else:
            say_src = current_text if self.pending_say else ""
        self.pending_say = False
        mt, mtxt = self.motion_info
        return {
            "text": current_text,
            "translated_text": translated_text,
            "sign_lang": self.sign_lang,
            "out_lang": self.out_lang,
            "speak_lang": self.out_lang,
            "mode": self.mode,
            "words": list(self.sentence_words) if self.mode == "sentences" else [],
            "letter": letter,                 # what the alphabet model sees right now
            "guess": self.last_guess,
            "top2": top2,
            "hint_ar": self.last_hint_ar,
            "guess_ar": self.last_guess_ar,
            "top3": list(self.last_top3),
            "motion": mtxt if now - mt < 3.0 else "",
            "typed": typed_str,               # NEW text since the last frame (speak these)
            "say": say_src,                   # whole text, only when the device sent "speak"
            "hand": hand_pts,                 # 21 (x, y) in the UNFLIPPED picture, or []
            "hands": self._hands,             # all detected hands (up to 2) in word/sentence mode
            "ms": round((time.time() - t_start) * 1000, 1),
        }

    def _sentence_step(self, rgb_unflipped, now):
        res = self.motion_detect(rgb_unflipped, int(now * 1000))
        self._hands = [[[round(p.x, 4), round(p.y, 4)] for p in h] for h in res.hand_landmarks] if res.hand_landmarks else []
        pts = self._hands[0] if self._hands else []
        clip = self.word_seg.update(res, now)
        if clip is not None:
            if self.sign_lang == "arsl" and self.arsl_word_model is not None:
                w_id, w_prob, w_ranked = classify_arsl_word(self.arsl_word_model, clip)
                self.last_top3 = [
                    {"label": translate_arsl_word(w, self.out_lang), "id": w, "prob": round(float(p), 4)}
                    for w, p in w_ranked[:3]
                ]
                self.word_top3 = "   ".join(f"{translate_arsl_word(w, 'en')[:10]} {p:.0%}" for w, p in w_ranked[:3])
                self.word_top3_ar = "  ".join(f"{translate_arsl_word(w, 'ar')} {p:.0%}" for w, p in w_ranked[:2])
                self.motion_info = (now, self.word_top3)
                thresh = 0.25
                w_en = translate_arsl_word(w_id, "en")
                w_ar = translate_arsl_word(w_id, "ar")
                g_ar = w_ar.split()[0] if len(w_ar.split()[0]) <= 6 else f"{w_prob:.0%}"
                self.last_word_guess = (now, f"{w_prob:.0%}", g_ar)
                if w_prob >= thresh:
                    if not (self.sentence_words and self.sentence_words[-1] == w_en and now - self.sentence_last_time < 0.9):
                        self.sentence_words.append(w_en)
                        self.sentence_words_ar.append(w_ar)
                        self.sentence_last_time = now
                        self.sentence_spoken = False
                        self.sentence_text = (
                            format_arabic_sentence(self.sentence_words_ar)
                            if self.out_lang == "ar"
                            else format_sentence(self.sentence_words)
                        )
                        self._typed.append(" " + (w_ar if self.out_lang == "ar" else w_en))
            else:
                w_pred, w_prob, w_ranked = classify_word(self.word_model, clip)
                self.last_top3 = [
                    {"label": str(w), "prob": round(float(p), 4)}
                    for w, p in w_ranked[:3]
                ]
                self.word_top3 = "   ".join(f"{w} {p:.0%}" for w, p in w_ranked[:3])
                self.word_top3_ar = "  ".join(f"{translate_en_to_ar(str(w))} {p:.0%}" for w, p in w_ranked[:2])
                self.motion_info = (now, self.word_top3)
                g_ar = translate_en_to_ar(str(w_pred)).split()[0]
                if len(g_ar) > 6:
                    g_ar = f"{w_prob:.0%}"
                self.last_word_guess = (now, f"{w_prob:.0%}", g_ar)
                if w_pred != "IDLE" and w_prob >= self.cfg.WORD_THRESH:
                    if not (self.sentence_words and self.sentence_words[-1] == w_pred and now - self.sentence_last_time < 0.9):
                        self.sentence_words.append(w_pred)
                        self.sentence_words_ar.append(translate_en_to_ar(w_pred))
                        self.sentence_last_time = now
                        self.sentence_spoken = False
                        self.sentence_text = format_sentence(self.sentence_words)
                        self._typed.append(f" {w_pred}")

        # Natural pause auto-speak (1.2s pause after signing sentence)
        if self.sentence_words and not self.sentence_spoken and not self.word_seg.active:
            if now - self.sentence_last_time >= 1.2:
                self.pending_say = True
                self.sentence_spoken = True

        if self.word_seg.active:
            dur = f"{now - self.word_seg.start:.1f}s"
            top2 = f"SIGNING WORD... {dur}"
            self.last_hint_ar = f"تسجيل {now - self.word_seg.start:.1f}"
            self.last_guess = dur
            self.last_guess_ar = f"{now - self.word_seg.start:.1f}"
        else:
            gt, gen, gar = self.last_word_guess
            self.last_guess = gen if now - gt < 3.0 else ""
            self.last_guess_ar = gar if now - gt < 3.0 else ""
            if self.sentence_words:
                top2 = " -> ".join(self.sentence_words[-3:])
                self.last_hint_ar = " ".join(self.sentence_words_ar[-2:]) if self.sentence_words_ar else (self.word_top3_ar or "")
            else:
                top2 = self.word_top3 or "Sign successive words..."
                self.last_hint_ar = self.word_top3_ar or "إشارة جملة"
        return pts, top2

    def _word_step(self, rgb_unflipped, now):
        res = self.motion_detect(rgb_unflipped, int(now * 1000))
        self._hands = [[[round(p.x, 4), round(p.y, 4)] for p in h] for h in res.hand_landmarks] if res.hand_landmarks else []
        pts = self._hands[0] if self._hands else []
        clip = self.word_seg.update(res, now)
        if clip is not None:
            if self.sign_lang == "arsl" and self.arsl_word_model is not None:
                w_id, w_prob, w_ranked = classify_arsl_word(self.arsl_word_model, clip)
                self.last_top3 = [
                    {"label": translate_arsl_word(w, self.out_lang), "id": w, "prob": round(float(p), 4)}
                    for w, p in w_ranked[:3]
                ]
                self.word_top3 = "   ".join(f"{translate_arsl_word(w, 'en')[:10]} {p:.0%}" for w, p in w_ranked[:3])
                self.word_top3_ar = "  ".join(f"{translate_arsl_word(w, 'ar')} {p:.0%}" for w, p in w_ranked[:2])
                self.motion_info = (now, self.word_top3)
                w_ar = translate_arsl_word(w_id, "ar")
                g_ar = w_ar.split()[0] if len(w_ar.split()[0]) <= 6 else f"{w_prob:.0%}"
                self.last_word_guess = (now, f"{w_prob:.0%}", g_ar)
                if w_prob >= 0.25:
                    self._arsl_last_word_id = w_id
                    shown = translate_arsl_word(w_id, self.out_lang)
                    self.text = shown
                    self.last_typed = (now, shown, "word")
                    self._typed.append(shown)
            else:
                w_pred, w_prob, w_ranked = classify_word(self.word_model, clip)
                self.last_top3 = [
                    {"label": str(w), "prob": round(float(p), 4)}
                    for w, p in w_ranked[:3]
                ]
                self.word_top3 = "   ".join(f"{w} {p:.0%}" for w, p in w_ranked[:3])
                self.word_top3_ar = "  ".join(f"{translate_en_to_ar(str(w))} {p:.0%}" for w, p in w_ranked[:2])
                self.motion_info = (now, self.word_top3)
                g_ar = translate_en_to_ar(str(w_pred)).split()[0]
                if len(g_ar) > 6:
                    g_ar = f"{w_prob:.0%}"
                self.last_word_guess = (now, f"{w_prob:.0%}", g_ar)
                if w_pred != "IDLE" and w_prob >= self.cfg.WORD_THRESH:
                    self.text = w_pred
                    self.last_typed = (now, w_pred, "word")
                    self._typed.append(w_pred)
        if self.word_seg.active:
            dur = f"{now - self.word_seg.start:.1f}s"
            top2 = f"SIGNING... {dur}"
            self.last_hint_ar = f"تسجيل {now - self.word_seg.start:.1f}"
            self.last_guess = dur
            self.last_guess_ar = f"{now - self.word_seg.start:.1f}"
        else:
            gt, gen, gar = self.last_word_guess
            self.last_guess = gen if now - gt < 3.0 else ""
            self.last_guess_ar = gar if now - gt < 3.0 else ""
            top2 = self.word_top3 or "Sign a word"
            self.last_hint_ar = self.word_top3_ar or "إشارة كلمة"
        return pts, top2

    def _active_alpha_model(self):
        if self.sign_lang == "arsl" and self.arsl_alpha_model is not None:
            return self.arsl_alpha_model
        return self.alpha_model

    def _fmt_letter(self, raw_cls):
        if self.sign_lang == "arsl" and self.arsl_alpha_model is not None:
            return translate_arsl_letter(raw_cls, self.out_lang)
        return str(raw_cls)

    # ---------------- static letters ----------------
    def _alpha_step_inspect(self, rgb_flipped, now, lm=None):
        c = self.cfg
        mdl = self._active_alpha_model()
        if lm is None and self.sign_lang != "arsl":
            lm = self.alpha_detect(rgb_flipped)
        letter, top2 = None, ""
        self.last_hint_ar, self.last_guess, self.last_guess_ar = "", "", ""
        if lm and mdl is not None:
            probs = mdl.predict_proba([normalize(lm).flatten()])[0]
            order = probs.argsort()[::-1]
            self.last_top3 = [
                {"label": self._fmt_letter(mdl.classes_[i]), "prob": round(float(probs[i]), 4)}
                for i in order[:3]
            ]
            a, b = order[0], order[1]
            la = translate_arsl_letter(mdl.classes_[a], "en") if self.sign_lang == "arsl" else str(mdl.classes_[a])
            lb = translate_arsl_letter(mdl.classes_[b], "en") if self.sign_lang == "arsl" else str(mdl.classes_[b])
            ara = translate_arsl_letter(mdl.classes_[a], "ar") if self.sign_lang == "arsl" else str(mdl.classes_[a])
            arb = translate_arsl_letter(mdl.classes_[b], "ar") if self.sign_lang == "arsl" else str(mdl.classes_[b])
            top2 = f"{la} {probs[a]:.0%}   {lb} {probs[b]:.0%}"
            self.last_hint_ar = f"{ara} {probs[a]:.0%}  {arb} {probs[b]:.0%}"
            if probs[a] >= c.PREVIEW_CONF:
                self.last_guess = self._fmt_letter(mdl.classes_[a])
                self.last_guess_ar = ara
            if probs[a] >= c.MIN_CONF:
                letter = self._fmt_letter(mdl.classes_[a])
        return letter, top2

    # ---------------- static letters ----------------
    def _alpha_step(self, rgb_flipped, now, lm=None):
        c = self.cfg
        mdl = self._active_alpha_model()
        if lm is None and self.sign_lang != "arsl":
            lm = self.alpha_detect(rgb_flipped)
        letter, top2 = None, ""
        self.last_hint_ar, self.last_guess, self.last_guess_ar = "", "", ""
        if lm and mdl is not None:
            self.last_seen = now
            probs = mdl.predict_proba([normalize(lm).flatten()])[0]
            order = probs.argsort()[::-1]
            self.last_top3 = [
                {"label": self._fmt_letter(mdl.classes_[i]), "prob": round(float(probs[i]), 4)}
                for i in order[:3]
            ]
            a, b = order[0], order[1]
            la = translate_arsl_letter(mdl.classes_[a], "en") if self.sign_lang == "arsl" else str(mdl.classes_[a])
            lb = translate_arsl_letter(mdl.classes_[b], "en") if self.sign_lang == "arsl" else str(mdl.classes_[b])
            ara = translate_arsl_letter(mdl.classes_[a], "ar") if self.sign_lang == "arsl" else str(mdl.classes_[a])
            arb = translate_arsl_letter(mdl.classes_[b], "ar") if self.sign_lang == "arsl" else str(mdl.classes_[b])
            top2 = f"{la} {probs[a]:.0%}   {lb} {probs[b]:.0%}"
            self.last_hint_ar = f"{ara} {probs[a]:.0%}  {arb} {probs[b]:.0%}"
            if probs[a] >= c.PREVIEW_CONF:
                self.last_guess = self._fmt_letter(mdl.classes_[a])
                self.last_guess_ar = ara
            if probs[a] >= c.MIN_CONF:
                letter = self._fmt_letter(mdl.classes_[a])
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
                if self.sign_lang == "arsl" and self.out_lang == "en":
                    self._type(str(top) + " ", "alpha", now)
                else:
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
        pts = []
        if res.hand_landmarks:
            self._hands = [[[round(p.x, 4), round(p.y, 4)] for p in h] for h in res.hand_landmarks]
            pts = self._hands[0]
        if self.sign_lang == "arsl":
            return pts
        feat = frame_features(res)
        self.mbuf.append((now, feat))
        while self.mbuf and now - self.mbuf[0][0] > c.MOTION_DUR:
            self.mbuf.popleft()

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
        pts = []
        if res.hand_landmarks:
            self._hands = [[[round(p.x, 4), round(p.y, 4)] for p in h] for h in res.hand_landmarks]
            pts = self._hands[0]
        if self.sign_lang == "arsl":
            return pts
        feat = frame_features(res)
        self.mbuf.append((now, feat))
        while self.mbuf and now - self.mbuf[0][0] > c.MOTION_DUR:
            self.mbuf.popleft()

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
