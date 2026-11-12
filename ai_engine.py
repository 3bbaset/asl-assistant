"""ai_engine.py - the laptop's ONLY job: run the AI model.

The Raspberry Pi is the device (camera, touch screen, speaker). It listens on
TCP port 9000; this program connects out to it over Wi-Fi, receives camera
frames, runs the recognizer, and sends the result back. Nothing here touches
the ESP32 or any hardware.

    .venv\\Scripts\\python.exe ai_engine.py              # auto-finds the Pi
    .venv\\Scripts\\python.exe ai_engine.py --pi 192.168.0.23
    .venv\\Scripts\\python.exe ai_engine.py --gui        # optional debug window
"""
import argparse
import json
import os
import socket
import struct
import threading
import time

import cv2
import numpy as np

import common
import server

PI_PORT = 9000
HOST_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pi_host")


def log(tag, msg):
    print(f"{time.strftime('%H:%M:%S')} [{tag}] {msg}", flush=True)


def recv_exact(s, n):
    out = bytearray()
    while len(out) < n:
        chunk = s.recv(n - len(out))
        if not chunk:
            raise ConnectionError("Pi closed the connection")
        out += chunk
    return bytes(out)


# ---------------------------------------------------------------- finding the Pi
def port_open(host, timeout=0.6):
    try:
        with socket.create_connection((host, PI_PORT), timeout=timeout):
            return True
    except OSError:
        return False


def scan_subnets():
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    prefixes = sorted({ip.rsplit(".", 1)[0] for ip in ips if not ip.startswith(("127.", "169.254."))})
    found = []

    def probe(h):
        if port_open(h, 0.5):
            found.append(h)

    threads = [threading.Thread(target=probe, args=(f"{p}.{i}",), daemon=True)
               for p in prefixes for i in range(1, 255)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=3)
    return found


def find_pi(explicit):
    cands = [explicit] if explicit else []
    cands += [os.environ.get("ASL_PI"), "raspberrypi.local"]
    try:
        cands.append(open(HOST_CACHE).read().strip())
    except OSError:
        pass
    for h in filter(None, cands):
        if port_open(h, 1.5):
            return h
    log("Net", "scanning the Wi-Fi network for the Pi...")
    hits = scan_subnets()
    return hits[0] if hits else None


# ---------------------------------------------------------------- recognition
def current_text(b):
    return b.sentence_text if b.mode == "sentences" else b.text


def _combined_mode(b):
    if b.mode == "words":
        return "words_2000" if b.word_vocab_mode == "2000" else "words"
    if b.mode == "sentences":
        return "sentences_2000" if b.word_vocab_mode == "2000" else "sentences"
    return b.mode


def handle(b, hdr, jpg, last, gui):
    req = hdr.get("mode", "")
    current_combined = _combined_mode(b)
    if req and req != current_combined:
        b.set_mode("words_30" if req == "words" else req)
        log("Device", f"mode -> {b.mode} (vocab={b.word_vocab_mode})")
    for c in hdr.get("cmds", []):
        if c.startswith("mode:"):
            m = c[5:]
            b.set_mode("words_30" if m == "words" else m)
            log("Device", f"mode -> {b.mode} (vocab={b.word_vocab_mode})")
        else:
            b.command(c)
            log("Device", c)
    speak = ""
    if jpg:
        frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            res = b.process(frame)
            last["hand"] = 1 if res.get("hand") else 0
            last["guess"] = "" if b.mode in ("words", "sentences") else (res.get("letter") or "")
            last["hint"] = str(res.get("top2") or res.get("motion") or "")[:39]
            if res.get("typed") and b.mode in ("words", "alphabets"):
                speak = res["typed"]
            if res.get("say"):
                speak = res["say"]
            if gui:
                view = cv2.flip(frame, 1)
                h, w = view.shape[:2]
                if res.get("hand"):
                    pts = [(int((1 - x) * w), int(y * h)) for x, y in res["hand"]]
                    for p1, p2 in common.CONNECTIONS:
                        cv2.line(view, pts[p1], pts[p2], (0, 255, 0), 2)
                cv2.putText(view, f"{b.mode.upper()}  {current_text(b)[-30:]}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("AI engine (debug)", view)
                cv2.waitKey(1)
    text = current_text(b)
    if text != last.get("text"):
        log("AI", f"[{b.mode}] {text[-60:]!r}")
    last["text"] = text
    disp_mode = _combined_mode(b)
    return {"seq": hdr.get("seq", 0), "req_mode": req, "mode": disp_mode, "text": text,
            "hand": last["hand"], "guess": last["guess"], "hint": last["hint"], "speak": speak}


def session(b, host, gui):
    s = socket.create_connection((host, PI_PORT), timeout=5)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.settimeout(15)
    try:
        open(HOST_CACHE, "w").write(host)
    except OSError:
        pass
    log("Net", f"connected to the Pi at {host}. Use the device touch screen; Ctrl+C here to stop.")
    last = {"hand": 0, "guess": "", "hint": "", "text": None}
    n, t0, busy = 0, time.time(), 0.0
    while True:
        hlen, jlen = struct.unpack(">II", recv_exact(s, 8))
        hdr = json.loads(recv_exact(s, hlen))
        jpg = recv_exact(s, jlen) if jlen else b""
        t = time.time()
        out = json.dumps(handle(b, hdr, jpg, last, gui)).encode()
        s.sendall(struct.pack(">I", len(out)) + out)
        if jpg:
            n += 1
            busy += time.time() - t
        if time.time() - t0 >= 10:
            if n:
                log("AI", f"{n / (time.time() - t0):.1f} frames/s, {busy / n * 1000:.0f} ms/frame")
            n, t0, busy = 0, time.time(), 0.0


def main():
    ap = argparse.ArgumentParser(description="ASL AI engine (laptop side)")
    ap.add_argument("--pi", default=None, help="Pi address (default: auto-detect)")
    ap.add_argument("--gui", action="store_true", help="show a debug camera window")
    args = ap.parse_args()

    log("AI", "loading models...")
    b = server.build_real_brain()
    log("AI", "models ready")
    try:
        while True:
            host = find_pi(args.pi)
            if not host:
                log("Net", "Pi not found; is it powered and on the same Wi-Fi? retrying in 3 s")
                time.sleep(3)
                continue
            try:
                session(b, host, args.gui)
            except (OSError, ConnectionError, struct.error, ValueError) as e:
                log("Net", f"link lost ({e}); reconnecting")
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        log("AI", "stopped")


if __name__ == "__main__":
    main()
