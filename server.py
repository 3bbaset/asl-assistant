"""server.py - run this on the LAPTOP. It is the "cloud".

    python server.py                       # listens on 0.0.0.0:8000
    python server.py --token mysecret      # require a shared password header
    python server.py --stub                # fake models, only to test the plumbing

Protocol (plain HTTP, no extra packages):
    POST /frame   body = one JPEG, NOT mirrored
                  optional header  X-Cmd: space | backspace | clear | speak
                  reply = JSON from Brain.process(): text, letter, top2, motion, typed, say, hand, ms
    GET  /state   reply = {"text": ...}   (handy to check from a phone browser)
    POST /cmd     header X-Cmd only, no frame
Keep it on your home Wi-Fi: there is no encryption. The token only stops accidents.
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if os.name == "posix":                      # Linux only (breaks Windows)
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import cv2
import numpy as np

from brain import Brain

LOCK = threading.Lock()      # one frame at a time: the Brain keeps state between frames
BRAIN = None
TOKEN = None


def _one_thread(m):
    if m is None:
        return
    steps = list((getattr(m, "named_steps", None) or {"model": m}).values())
    for step in steps:
        if hasattr(step, "n_jobs"):
            step.n_jobs = 1
        for sub in list(getattr(step, "estimators_", ())) + (list(step.estimators) if isinstance(getattr(step, "estimators", None), list) else []):
            if hasattr(sub, "n_jobs"):
                sub.n_jobs = 1
            if hasattr(sub, "named_steps"):
                _one_thread(sub)


class FastHGBClassifier:
    """Bit-exact vectorized NumPy walker for HistGradientBoostingClassifier Pipelines."""

    def __init__(self, pipeline_or_model):
        self._orig = pipeline_or_model
        steps = list(getattr(pipeline_or_model, "named_steps", {}).values())
        if steps:
            self._pre_steps = steps[:-1]
            hgb = steps[-1]
        else:
            self._pre_steps = []
            hgb = pipeline_or_model
        self.classes_ = hgb.classes_
        preds = hgb._predictors
        self._n_iter = len(preds)
        self._n_cls = len(preds[0])
        T = self._n_iter * self._n_cls
        max_nodes = max(len(preds[i][k].nodes) for i in range(self._n_iter) for k in range(self._n_cls))
        self._feat = np.zeros((T, max_nodes), dtype=np.int32)
        self._thresh = np.zeros((T, max_nodes), dtype=np.float64)
        self._left = np.zeros((T, max_nodes), dtype=np.int32)
        self._right = np.zeros((T, max_nodes), dtype=np.int32)
        self._val = np.zeros((T, max_nodes), dtype=np.float64)
        self._is_leaf = np.ones((T, max_nodes), dtype=bool)
        self._miss_left = np.zeros((T, max_nodes), dtype=bool)
        for i in range(self._n_iter):
            for k in range(self._n_cls):
                t = i * self._n_cls + k
                nd = preds[i][k].nodes
                L = len(nd)
                self._feat[t, :L] = nd["feature_idx"]
                self._thresh[t, :L] = nd["num_threshold"]
                self._left[t, :L] = nd["left"]
                self._right[t, :L] = nd["right"]
                self._val[t, :L] = nd["value"]
                self._is_leaf[t, :L] = nd["is_leaf"].astype(bool)
                self._miss_left[t, :L] = nd["missing_go_to_left"].astype(bool)
        self._t_arange = np.arange(T, dtype=np.int32)
        self._base = np.asarray(hgb._baseline_prediction, dtype=np.float64).ravel()

    def predict_proba(self, X):
        X_arr = np.asarray(X)
        if len(X_arr) != 1:
            return self._orig.predict_proba(X)
        Xf = X_arr
        for st in self._pre_steps:
            Xf = st.transform(Xf)
        xf = np.asarray(Xf[0], dtype=np.float64)
        T = len(self._t_arange)
        idx = np.zeros(T, dtype=np.int32)
        ta = self._t_arange
        while True:
            leaf = self._is_leaf[ta, idx]
            if leaf.all():
                break
            fv = xf[self._feat[ta, idx]]
            nan_mask = np.isnan(fv)
            go_left = np.where(nan_mask, self._miss_left[ta, idx], fv <= self._thresh[ta, idx])
            nxt = np.where(go_left, self._left[ta, idx], self._right[ta, idx])
            idx = np.where(leaf, idx, nxt)
        raw = self._val[ta, idx].reshape(self._n_iter, self._n_cls).sum(axis=0) + self._base
        if self._n_cls == 1:
            p1 = 1.0 / (1.0 + np.exp(-raw))
            return np.column_stack([1.0 - p1, p1])
        raw -= raw.max()
        e = np.exp(raw)
        return (e / e.sum())[None, :]


def _wrap_fast_hgb(mdl):
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        last = list(getattr(mdl, "named_steps", {}).values())[-1] if hasattr(mdl, "named_steps") else mdl
        if not isinstance(last, HistGradientBoostingClassifier):
            return mdl
        fast = FastHGBClassifier(mdl)
        n_in = getattr(mdl, "n_features_in_", 63)
        test_x = np.random.default_rng(0).normal(size=(1, n_in)).astype(np.float32)
        if float(np.max(np.abs(fast.predict_proba(test_x) - mdl.predict_proba(test_x)))) < 1e-6:
            return fast
    except Exception:
        pass
    return mdl


def build_real_brain():
    import joblib
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    alpha_model = _wrap_fast_hgb(joblib.load("asl_model_v2.joblib"))
    motion_model = joblib.load("motion_model2.joblib")
    m30_path = "word_model_curated_30.joblib" if os.path.exists("word_model_curated_30.joblib") else "word_model_curated_85.joblib"
    m_full_path = "word_model_full_2021.joblib"
    word_model_30 = joblib.load(m30_path) if os.path.exists(m30_path) else None
    word_model_full = joblib.load(m_full_path) if os.path.exists(m_full_path) else None
    arsl_alpha_model = _wrap_fast_hgb(joblib.load("arsl_alpha_model.joblib")) if os.path.exists("arsl_alpha_model.joblib") else None
    arsl_word_model = joblib.load("arsl_word_model_30.joblib") if os.path.exists("arsl_word_model_30.joblib") else None
    for mdl in (alpha_model, motion_model, word_model_30, word_model_full, arsl_alpha_model, arsl_word_model):
        _one_thread(mdl)

    # Use VIDEO tracking mode for both detectors so MediaPipe tracks the hand ROI across frames (~3x faster)
    alpha_opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    alpha_lmk = vision.HandLandmarker.create_from_options(alpha_opts)
    alpha_last = [0]

    def alpha_detect(rgb):
        ts_ms = max(int(time.time() * 1000), alpha_last[0] + 1)
        alpha_last[0] = ts_ms
        res = alpha_lmk.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
        return res.hand_landmarks[0] if res.hand_landmarks else None

    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO, num_hands=2)
    motion_lmk = vision.HandLandmarker.create_from_options(opts)
    last = [0]

    def motion_detect(rgb, ts_ms):
        ts_ms = max(ts_ms, last[0] + 1)                # VIDEO mode needs strictly rising times
        last[0] = ts_ms
        return motion_lmk.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)

    return Brain(alpha_model, motion_model,
                 alpha_detect, motion_detect,
                 word_model=word_model_30, word_model_full=word_model_full,
                 arsl_alpha_model=arsl_alpha_model, arsl_word_model=arsl_word_model)


def build_stub_brain():
    """No models, no MediaPipe: a hand that never appears. Only tests the network."""
    from types import SimpleNamespace

    class Nothing:
        classes_ = np.array(["IDLE", "J", "Z"])

        def predict_proba(self, X):
            return np.array([[1.0, 0.0, 0.0]] * len(X))

    return Brain(Nothing(), Nothing(), lambda rgb: None,
                 lambda rgb, ts: SimpleNamespace(hand_landmarks=[]))


class QuietServer(ThreadingHTTPServer):
    """Same server, but a device that unplugs or drops Wi-Fi mid-request is not an error."""
    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"          # keep the connection open between frames

    def log_message(self, *a):             # quiet
        pass

    def _reply(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        if TOKEN and self.headers.get("X-Token") != TOKEN:
            self._reply(401, {"error": "bad token"})
            return False
        return True

    def do_GET(self):
        if not self._authorized():
            return
        if self.path == "/state":
            with LOCK:
                self._reply(200, {"text": BRAIN.text})
        else:
            self._reply(404, {"error": "unknown path"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(n) if n else b""          # always read the body first
        if not self._authorized():
            return
        cmd = self.headers.get("X-Cmd")
        if self.path == "/cmd":
            with LOCK:
                BRAIN.command(cmd)
                self._reply(200, {"text": BRAIN.text})
            return
        if self.path != "/frame":
            self._reply(404, {"error": "unknown path"})
            return
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self._reply(400, {"error": "not a JPEG"})
            return
        with LOCK:
            if cmd:
                BRAIN.command(cmd)
            out = BRAIN.process(frame)
        self._reply(200, out)


def main():
    global BRAIN, TOKEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--token", default=os.environ.get("ASL_TOKEN"))
    ap.add_argument("--stub", action="store_true")
    a = ap.parse_args()
    TOKEN = a.token
    BRAIN = build_stub_brain() if a.stub else build_real_brain()
    print(f"server ready on {a.host}:{a.port}  token={'ON' if TOKEN else 'off'}"
          f"{'  (STUB models)' if a.stub else ''}", flush=True)
    QuietServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
