"""Practice & accuracy test for ASL words.
Shows a reference video on the left and your live camera on the right.
Usage:
  python quiz_words.py            -> cycle through all words in WORDS
  python quiz_words.py HELLO YES  -> practice specific words

Controls:
  n / SPACE : next word
  p         : previous word
  v         : switch to a different reference video for the same word
  k         : keep/save your last attempt into own_clips.npz (for retraining)
  r         : reset accuracy scoreboard
  ESC / q   : quit and print final accuracy summary"""
import glob, os, sys, time
if os.name == "posix":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import cv2, joblib, numpy as np, mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from words_common import WORDS, VIDEO_DIR, MODEL_FILE
from words_live import WordSegmenter, classify, draw_text, speak

OUT_OWN = "own_clips.npz"
words = [w.upper() for w in sys.argv[1:]] or list(WORDS)
words = [w for w in words if os.path.isdir(os.path.join(VIDEO_DIR, w))]
if not words:
    sys.exit(f"No reference videos found in {VIDEO_DIR}/. Run get_words.py first.")

model = joblib.load(MODEL_FILE)
opts = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO, num_hands=2)


class RefPlayer:
    """Loops a reference mp4 video so you can copy the signer."""
    def __init__(self):
        self.cap, self.paths, self.idx, self.vid_idx = None, [], 0, 0
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.last_step = 0.0
        self.fps = 30.0

    def set_word(self, word, vid_idx=0):
        if self.cap is not None:
            self.cap.release()
        self.paths = sorted(glob.glob(os.path.join(VIDEO_DIR, word, "*.mp4")))
        self.vid_idx = vid_idx % max(1, len(self.paths))
        if self.paths:
            self.cap = cv2.VideoCapture(self.paths[self.vid_idx])
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        else:
            self.cap = None

    def read(self, now, size=(640, 480)):
        if self.cap is not None and now - self.last_step >= 1.0 / max(10.0, min(60.0, self.fps)):
            self.last_step = now
            ok, f = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, f = self.cap.read()
            if ok:
                # Mirror reference video so the signer faces you like a mirror, matching your webcam view
                self.frame = cv2.flip(cv2.resize(f, size), 1)
        return self.frame.copy()

    def close(self):
        if self.cap is not None:
            self.cap.release()


def save_own_clip(clip, label):
    X, y = [], []
    if os.path.exists(OUT_OWN):
        d = np.load(OUT_OWN)
        X, y = list(d["X"]), list(d["y"])
    X.append(clip)
    y.append(label)
    np.savez(OUT_OWN, X=np.array(X), y=np.array(y))
    return sum(1 for l in y if l == label)


player = RefPlayer()
w_idx = 0
player.set_word(words[w_idx])

# stats[word] = [attempts, top1_hits, top3_hits]
stats = {w: [0, 0, 0] for w in words}
last_clip, last_word, saved_msg, saved_until = None, None, "", 0.0
status_msg, detail_msg, status_color, status_until = "", "", (0, 255, 255), 0.0

cap, t0, last_ts = cv2.VideoCapture(0), time.time(), 0
cv2.namedWindow("ASL Word Quiz (Left: Reference | Right: You)", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow("ASL Word Quiz (Left: Reference | Right: You)", 1280, 560)
seg = WordSegmenter()

with vision.HandLandmarker.create_from_options(opts) as lmk:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        ts = max(int((now - t0) * 1000), last_ts + 1)
        last_ts = ts
        target = words[w_idx]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        clip = seg.update(res, now)

        if clip is not None:
            last_clip, last_word = clip, target
            pred, prob, ranked = classify(model, clip)
            top3_words = [w for w, _ in ranked[:3]]
            target_prob = dict(ranked).get(target, 0.0)

            stats[target][0] += 1
            if pred == target:
                stats[target][1] += 1
                stats[target][2] += 1
                status_msg = f"CORRECT! {pred} ({prob:.0%})"
                status_color = (0, 255, 0)
                speak(target)
            elif target in top3_words:
                stats[target][2] += 1
                status_msg = f"CLOSE (Top 3)! Guessed {pred} ({prob:.0%}), {target} was {target_prob:.0%}"
                status_color = (0, 200, 255)
            else:
                status_msg = f"MISS: Guessed {pred} ({prob:.0%}), {target} was {target_prob:.0%}"
                status_color = (0, 80, 255)

            detail_msg = "Top 3: " + "   ".join(f"{w} {p:.0%}" for w, p in ranked[:3])
            status_until = now + 4.5

        cam_view = cv2.resize(cv2.flip(frame, 1), (640, 480))
        ref_view = player.read(now, (640, 480))

        # Draw labels on reference pane (left)
        cv2.rectangle(ref_view, (0, 0), (640, 52), (0, 0, 0), -1)
        draw_text(ref_view, f"TARGET: {target}  ({w_idx + 1}/{len(words)})", (12, 36), 0.9, (0, 255, 255))
        draw_text(ref_view, f"Demo video {player.vid_idx + 1}/{len(player.paths)} (v = change)",
                  (12, 465), 0.65, (220, 220, 220), 1)

        # Draw status on camera pane (right)
        cv2.rectangle(cam_view, (0, 0), (640, 52), (0, 0, 0), -1)
        if seg.active:
            draw_text(cam_view, f"RECORDING... {now - seg.start:.1f}s (drop hands when done)",
                      (12, 36), 0.75, (0, 80, 255))
        elif now < status_until:
            draw_text(cam_view, status_msg, (12, 36), 0.7, status_color)
            draw_text(cam_view, detail_msg, (12, 80), 0.65, (255, 255, 255))
            draw_text(cam_view, "Press 'k' to save this attempt to training data", (12, 112), 0.6, (180, 255, 180), 1)
        else:
            draw_text(cam_view, "Copy the signer on the left, then drop hands", (12, 36), 0.7, (255, 255, 255))

        if now < saved_until:
            draw_text(cam_view, saved_msg, (12, 465), 0.7, (0, 255, 0))

        # Bottom scoreboard bar across both panes
        board = np.zeros((64, 1280, 3), dtype=np.uint8)
        tot_a = sum(s[0] for s in stats.values())
        tot_1 = sum(s[1] for s in stats.values())
        tot_3 = sum(s[2] for s in stats.values())
        w_a, w_1, w_3 = stats[target]
        score_txt = (f"This word ({target}): {w_1}/{w_a} exact, {w_3}/{w_a} in top-3   |   "
                     f"Overall: {tot_1}/{tot_a} ({tot_1 / tot_a * 100 if tot_a else 0:.0f}%) exact, "
                     f"{tot_3}/{tot_a} ({tot_3 / tot_a * 100 if tot_a else 0:.0f}%) top-3")
        draw_text(board, score_txt, (12, 26), 0.65, (255, 255, 255), 1)
        draw_text(board, "Keys: [SPACE/n] next word  [p] prev  [v] another demo video  [k] keep clip  [r] reset  [q/ESC] quit",
                  (12, 54), 0.58, (180, 180, 180), 1)

        combined = np.vstack([np.hstack([ref_view, cam_view]), board])
        cv2.imshow("ASL Word Quiz (Left: Reference | Right: You)", combined)

        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        elif k in (32, ord("n")):
            w_idx = (w_idx + 1) % len(words)
            player.set_word(words[w_idx])
            status_until = 0.0
        elif k == ord("p"):
            w_idx = (w_idx - 1) % len(words)
            player.set_word(words[w_idx])
            status_until = 0.0
        elif k == ord("v"):
            player.set_word(target, player.vid_idx + 1)
        elif k == ord("k") and last_clip is not None:
            cnt = save_own_clip(last_clip, last_word)
            saved_msg = f"Saved to {OUT_OWN}! ({last_word}: {cnt} own clips)"
            saved_until = now + 3.0
            last_clip = None
        elif k == ord("r"):
            stats = {w: [0, 0, 0] for w in words}

player.close()
cap.release()
cv2.destroyAllWindows()

# Print final summary table in terminal
tot_a = sum(s[0] for s in stats.values())
if tot_a > 0:
    print("\n=== Session Accuracy Summary ===")
    print(f"{'WORD':15s} {'ATTEMPTS':>8s} {'EXACT (Top-1)':>15s} {'CLOSE (Top-3)':>15s}")
    for w in words:
        a, h1, h3 = stats[w]
        if a > 0:
            print(f"{w:15s} {a:8d} {h1:7d} ({h1/a*100:4.0f}%) {h3:7d} ({h3/a*100:4.0f}%)")
    tot_1 = sum(s[1] for s in stats.values())
    tot_3 = sum(s[2] for s in stats.values())
    print("-" * 56)
    print(f"{'TOTAL':15s} {tot_a:8d} {tot_1:7d} ({tot_1/tot_a*100:4.0f}%) {tot_3:7d} ({tot_3/tot_a*100:4.0f}%)")
