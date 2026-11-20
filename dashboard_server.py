"""dashboard_server.py - Localhost Capstone Dashboard & Live Prototype Telemetry Server.

Can run standalone to browse model benchmarks, confusion matrices, and export Capstone
results, OR run embedded inside ai_engine.py to stream live video, predictions, and controls.

Usage:
    .\\.venv\\Scripts\\python.exe dashboard_server.py --port 8080
"""
import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import queue
import sys
import threading
import time

import translator

HERE = os.path.dirname(os.path.abspath(__file__))
DASH_DIR = os.path.join(HERE, "dashboard")
METRICS_FILE = os.path.join(HERE, "reports", "capstone_metrics.json")
ASL_WORDS_FILE = os.path.join(HERE, "word_classes_30.json")


class TelemetryHub:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_cond = threading.Condition()
        self.latest_jpeg = None
        self.frame_seq = 0
        self.video_clients = 0
        self.cmd_queue = queue.Queue()
        self.logs = deque(maxlen=40)
        self.state = {
            "connected": False,
            "pi_host": "",
            "fps": 0.0,
            "latency_ms": 0.0,
            "hand": False,
            "num_hands": 0,
            "mode": "spelling",
            "sign_lang": "asl",
            "out_lang": "en",
            "text": "",
            "translated_text": "",
            "letter": "",
            "top2": "",
            "top3": [],
            "updated_at": time.time(),
        }

    def update(self, **kwargs):
        with self.lock:
            self.state.update(kwargs)
            self.state["updated_at"] = time.time()

    def add_log(self, tag, msg):
        line = f"{time.strftime('%H:%M:%S')} [{tag}] {msg}"
        with self.lock:
            self.logs.append(line)

    def push_frame(self, jpeg_bytes):
        if not jpeg_bytes:
            return
        with self.frame_cond:
            self.latest_jpeg = bytes(jpeg_bytes)
            self.frame_seq += 1
            self.frame_cond.notify_all()

    def push_cmd(self, cmd):
        c = str(cmd or "").strip()
        if c:
            self.cmd_queue.put(c)
            self.add_log("Web", f"cmd -> {c}")

    def drain_cmds(self):
        out = []
        while True:
            try:
                out.append(self.cmd_queue.get_nowait())
            except queue.Empty:
                break
        return out

    def snapshot(self):
        with self.lock:
            snap = dict(self.state)
            snap["logs"] = list(self.logs)
            return snap


def _load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def build_vocab_payload():
    asl_raw = _load_json(ASL_WORDS_FILE, [])
    asl_words = [
        {"en": str(w), "ar": translator.translate_en_to_ar(str(w))}
        for w in asl_raw
    ]
    arsl_words = list(translator.load_arsl_word_map().values())
    arsl_letters = [
        {"key": k, "ar": v["ar"], "name_ar": v["name_ar"], "en": v["en"]}
        for k, v in translator.ARSL_LETTER_MAP.items()
    ]
    return {
        "asl_words": asl_words,
        "arsl_words": arsl_words,
        "arsl_letters": arsl_letters,
    }


def _make_handler(hub):
    class DashboardHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send_bytes(self, code, data, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self._send_bytes(code, body, "application/json; charset=utf-8")

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                return self._serve_file("index.html", "text/html; charset=utf-8")
            if path == "/styles.css":
                return self._serve_file("styles.css", "text/css; charset=utf-8")
            if path == "/app.js":
                return self._serve_file("app.js", "application/javascript; charset=utf-8")
            if path == "/api/metrics":
                metrics = _load_json(METRICS_FILE, {"models": [], "scaling": [], "status": "missing"})
                return self._send_json(200, metrics)
            if path == "/api/vocab":
                return self._send_json(200, build_vocab_payload())
            if path == "/api/state":
                return self._send_json(200, hub.snapshot())
            if path == "/api/stream":
                return self._serve_sse()
            if path == "/api/video":
                return self._serve_mjpeg()
            return self._send_json(404, {"error": "not found"})

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            if path == "/api/cmd":
                try:
                    payload = json.loads(raw.decode("utf-8"))
                    cmd = payload.get("cmd", "")
                except Exception:
                    return self._send_json(400, {"error": "invalid json"})
                hub.push_cmd(cmd)
                return self._send_json(200, {"ok": True, "cmd": cmd})
            return self._send_json(404, {"error": "not found"})

        def _serve_file(self, fname, ctype):
            full = os.path.join(DASH_DIR, fname)
            if not os.path.exists(full):
                return self._send_bytes(404, b"Missing dashboard asset", "text/plain")
            with open(full, "rb") as f:
                return self._send_bytes(200, f.read(), ctype)

        def _serve_sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                last_payload = ""
                last_ping = 0.0
                while True:
                    snap = hub.snapshot()
                    payload = json.dumps(snap, ensure_ascii=False)
                    now = time.time()
                    if payload != last_payload or (now - last_ping >= 1.5):
                        msg = f"data: {payload}\n\n".encode("utf-8")
                        self.wfile.write(msg)
                        self.wfile.flush()
                        last_payload = payload
                        last_ping = now
                    time.sleep(0.12)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass

        def _serve_mjpeg(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with hub.lock:
                hub.video_clients += 1
            try:
                last_seq = -1
                while True:
                    with hub.frame_cond:
                        hub.frame_cond.wait_for(lambda: hub.frame_seq != last_seq, timeout=1.5)
                        jpg = hub.latest_jpeg
                        last_seq = hub.frame_seq
                    if jpg:
                        hdr = (
                            b"--frame\r\nContent-Type: image/jpeg\r\n"
                            + f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
                        )
                        self.wfile.write(hdr + jpg + b"\r\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            finally:
                with hub.lock:
                    hub.video_clients = max(0, hub.video_clients - 1)

    return DashboardHandler


class QuietDashHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


class DashboardServer:
    def __init__(self, host="127.0.0.1", port=8080, hub=None):
        self.host = host
        self.port = port
        self.hub = hub or TelemetryHub()
        self.httpd = None

    def start_in_thread(self):
        self.httpd = QuietDashHTTPServer((self.host, self.port), _make_handler(self.hub))
        t = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        t.start()
        return self.hub

    def serve_forever(self):
        self.httpd = QuietDashHTTPServer((self.host, self.port), _make_handler(self.hub))
        print(f"[Dashboard] Serving Capstone Prototype Dashboard at http://{self.host}:{self.port}", flush=True)
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.httpd.server_close()


def main():
    ap = argparse.ArgumentParser(description="ASL / ArSL Capstone Prototype Dashboard")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    DashboardServer(host=args.host, port=args.port).serve_forever()


if __name__ == "__main__":
    main()
