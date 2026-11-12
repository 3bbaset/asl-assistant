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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if os.name == "posix":                      # Linux only (breaks Windows)
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import cv2
import numpy as np

from brain import Brain

LOCK = threading.Lock()      # one frame at a time: the Brain keeps state between frames
BRAIN = None
TOKEN = None


def build_real_brain():
    import joblib
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    from common import make_detector, detect

    alpha_model = joblib.load("asl_model_v2.joblib")
    motion_model = joblib.load("motion_model2.joblib")
    m30_path = "word_model_curated_30.joblib" if os.path.exists("word_model_curated_30.joblib") else "word_model_curated_85.joblib"
    m_full_path = "word_model_full_2021.joblib"
    word_model_30 = joblib.load(m30_path) if os.path.exists(m30_path) else None
    word_model_full = joblib.load(m_full_path) if os.path.exists(m_full_path) else None
    alpha_det = make_detector(min_conf=0.6)            # IMAGE mode, 1 hand, as in app.py
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
                 lambda rgb: detect(alpha_det, rgb), motion_detect,
                 word_model=word_model_30, word_model_full=word_model_full)


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
