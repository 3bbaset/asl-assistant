"""Try the word model live. Usage: python try_words.py
No key needed: raise your hand(s), sign ONE word, then drop your hands out of the picture.
Everything between "hands appear" and "hands gone" is one word - the same way the training
videos were trimmed in extract_words.py. ESC = quit, c = clear the sentence, ENTER = speak it."""
import os, sys, time
if os.name == "posix":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import cv2, joblib, mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from words_common import MODEL_FILE
from words_live import WordSegmenter, classify, draw_text, speak

FULL_VOCAB = "--full-vocab" in sys.argv
ACTIVE_MODEL = "word_model_full_2021.joblib" if (FULL_VOCAB and os.path.exists("word_model_full_2021.joblib")) else MODEL_FILE
THRESH = 0.20 if FULL_VOCAB else 0.35  # word shown only if the model is at least this sure

model = joblib.load(ACTIVE_MODEL)
opts = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO, num_hands=2)

cap, t0, last_ts = cv2.VideoCapture(0), time.time(), 0
cv2.namedWindow("try words", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow("try words", 1024, 768)
seg = WordSegmenter()
result, detail, result_until, sentence = "", "", 0.0, []

with vision.HandLandmarker.create_from_options(opts) as lmk:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        ts = max(int((now - t0) * 1000), last_ts + 1)
        last_ts = ts
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)          # NOT flipped, like the training clips
        res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        clip = seg.update(res, now)
        if clip is not None:
            word, prob, ranked = classify(model, clip)
            detail = "   ".join(f"{w} {p:.0%}" for w, p in ranked[:3])
            if word != "IDLE" and prob >= THRESH:
                result = word
                sentence.append(word)
                speak(word)
            else:
                result = "? (not sure)"
            result_until = now + 3.0

        show = cv2.flip(frame, 1)                             # mirrored view for you to watch
        h, w = show.shape[:2]
        if seg.active:
            draw_text(show, f"SIGNING... {now - seg.start:.1f} s  (drop hands to finish)", (12, 40), 0.9, (0, 0, 255))
        elif now < result_until:
            draw_text(show, f"Word: {result}", (12, 40), 1.2, (0, 255, 0))
            draw_text(show, detail, (12, 80), 0.7, (255, 255, 255))
        else:
            draw_text(show, "Raise your hands and sign one word", (12, 40), 0.9)
        cv2.rectangle(show, (0, h - 56), (w, h), (0, 0, 0), -1)
        draw_text(show, " ".join(sentence)[-40:], (12, h - 16), 1.0, (255, 255, 255))
        cv2.imshow("try words", show)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == ord("c"):
            sentence = []
        elif k == 13:
            speak(" ".join(sentence))
cap.release()
cv2.destroyAllWindows()
