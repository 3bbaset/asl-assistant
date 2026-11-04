"""device.py - the "prototype". Runs on the laptop now, on the Raspberry Pi later.

It only does three things: grab the newest camera frame, send it to the server,
show / speak what comes back. No AI in here.

    python device.py                                   # webcam 0 -> server on this laptop
    python device.py --server http://192.168.1.20:8000 # laptop's address, from the Pi
    python device.py --camera "http://PHONE:8080/video"    # phone as a test camera
    python device.py --camera picamera                 # Pi camera (needs picamera2)
    python device.py --display console                 # no window (headless Pi)

Keys in the window: q quit | SPACE space | BACKSPACE delete | ENTER say it all | c clear | f fullscreen
"""
import argparse
import http.client
import json
import os
import subprocess
import threading
import time
from urllib.parse import urlparse

if os.name == "posix":                      # Linux only (breaks Windows)
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import cv2
import numpy as np


# ------------------------------------------------------------------ camera
class Camera:
    """Reads in a background thread and keeps ONLY the newest frame,
    so a slow network or a slow server never builds up a delay."""

    def __init__(self, source):
        self.frame, self.lock, self.running = None, threading.Lock(), True
        self.picam = None
        self.is_file = False
        self.opened = True
        if source == "picamera":
            from picamera2 import Picamera2          # only exists on the Pi
            self.picam = Picamera2()
            self.picam.configure(self.picam.create_video_configuration(
                main={"size": (640, 480), "format": "BGR888"}))
            self.picam.start()
        else:
            self.cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.opened = self.cap.isOpened()
            # a video FILE works as a fake camera (plays in real time, loops): handy for tests
            self.is_file = os.path.isfile(str(source))
            self.file_dt = 1.0 / (self.cap.get(cv2.CAP_PROP_FPS) or 20)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            if self.picam:
                f = self.picam.capture_array()
                ok = f is not None
            else:
                ok, f = self.cap.read()
                if self.is_file:
                    time.sleep(self.file_dt)
                    if not ok:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
            if not ok:
                time.sleep(0.05)
                continue
            with self.lock:
                self.frame = f

    def latest(self):
        with self.lock:
            f, self.frame = self.frame, None        # never send the same frame twice
        return f

    def close(self):
        self.running = False


# ------------------------------------------------------------------ server link
class Link:
    def __init__(self, url, token):
        u = urlparse(url)
        self.host, self.port, self.token = u.hostname, u.port or 80, token
        self.conn = None

    def _headers(self, cmd):
        h = {"Content-Type": "image/jpeg"}
        if self.token:
            h["X-Token"] = self.token
        if cmd:
            h["X-Cmd"] = cmd
        return h

    def post(self, path, body, cmd=None):
        for attempt in (1, 2):                      # one reconnect if the connection dropped
            try:
                if self.conn is None:
                    self.conn = http.client.HTTPConnection(self.host, self.port, timeout=3)
                self.conn.request("POST", path, body=body, headers=self._headers(cmd))
                r = self.conn.getresponse()
                data = r.read()
                if r.status != 200:
                    raise RuntimeError(f"server said {r.status}: {data[:80]!r}")
                return json.loads(data)
            except (OSError, http.client.HTTPException):
                self.conn = None
                if attempt == 2:
                    raise


# ------------------------------------------------------------------ output
def speak(s):
    """Windows' built-in voice, or espeak-ng on Linux / the Pi. Never crashes if there is no voice."""
    s = "".join(ch for ch in s if ch.isalnum() or ch == " ")      # letters and spaces only
    if not s.strip():
        return
    try:
        if os.name == "nt":
            cmd = ["powershell", "-NoProfile", "-Command",
                   "Add-Type -AssemblyName System.Speech; "
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


BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
         (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]


def render(frame, res, status):
    """Mirrored preview with the guesses on top and the text at the bottom.
    On the Pi this is the part you replace with an OLED/LCD driver."""
    show = cv2.flip(frame, 1)
    h, w = show.shape[:2]
    if res.get("hand"):
        pts = [(int((1 - x) * w), int(y * h)) for x, y in res["hand"]]   # mirror x
        for a, b in BONES:
            cv2.line(show, pts[a], pts[b], (0, 255, 0), 2)
    cv2.rectangle(show, (0, 0), (w, 84), (0, 0, 0), -1)
    draw_text(show, res.get("top2") or "-", (12, 36), 0.9)
    draw_text(show, res.get("motion") or "", (12, 74), 0.7, (255, 255, 255))
    cv2.rectangle(show, (0, h - 56), (w, h), (0, 0, 0), -1)
    draw_text(show, res.get("text", "")[-30:], (12, h - 16), 1.1, (255, 255, 255))
    draw_text(show, status, (w - 230, 120), 0.6, (0, 165, 255))
    return show


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--camera", default="0", help="0, a stream URL, or 'picamera'")
    ap.add_argument("--token", default=os.environ.get("ASL_TOKEN"))
    ap.add_argument("--width", type=int, default=640, help="frames are shrunk to this width")
    ap.add_argument("--quality", type=int, default=70, help="JPEG quality 1-100")
    ap.add_argument("--display", choices=["window", "console"], default="window")
    ap.add_argument("--no-speech", action="store_true")
    a = ap.parse_args()

    cam, link = Camera(a.camera), Link(a.server, a.token)
    window = a.display == "window"
    if window:
        cv2.namedWindow("ASL device", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("ASL device", 1024, 768)
    fullscreen, cmd, res, last_text = False, None, {}, ""
    sent, t_fps, fps, err = 0, time.time(), 0.0, ""
    t_frame, t_warn = time.time(), 0.0              # when the camera last gave a frame
    if not cam.opened:
        print(f"WARNING: could not open the camera source {a.camera!r}", flush=True)

    try:
        while True:
            frame = cam.latest()
            if frame is None:
                waited = time.time() - t_frame
                if waited > 3.0:                    # say WHY the window is empty
                    if time.time() - t_warn > 5.0:
                        t_warn = time.time()
                        print(f"No frames from the camera for {waited:.0f} s (source {a.camera!r}). "
                              "Check that this address shows video in a browser, "
                              "the phone screen is on and the app is open.", flush=True)
                    if window:
                        wait_img = np.zeros((480, 640, 3), np.uint8)
                        draw_text(wait_img, "Waiting for the camera...", (20, 230), 1.0, (0, 165, 255))
                        draw_text(wait_img, "no frames received", (20, 280), 0.7, (255, 255, 255))
                        cv2.imshow("ASL device", wait_img)
                time.sleep(0.005)
                if window and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            t_frame = time.time()
            small = frame
            if frame.shape[1] > a.width:
                small = cv2.resize(frame, (a.width, int(frame.shape[0] * a.width / frame.shape[1])))
            ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            try:
                res = link.post("/frame", jpg.tobytes(), cmd)
                cmd, err = None, ""
            except Exception as e:                  # server off / Wi-Fi gone: keep trying
                err = f"no server: {type(e).__name__}"
                time.sleep(0.5)

            sent += 1
            if time.time() - t_fps >= 1.0:
                fps, sent, t_fps = sent / (time.time() - t_fps), 0, time.time()

            if not a.no_speech:
                if res.get("typed"):
                    speak(res["typed"])
                if res.get("say"):
                    speak(res["say"])
            if a.display == "console" and res.get("text", "") != last_text:
                last_text = res["text"]
                print(f"[{fps:4.1f} fps, {res.get('ms', 0):5.1f} ms] {last_text}", flush=True)

            if window:
                status = err or f"{fps:.0f} fps  {res.get('ms', 0):.0f} ms"
                cv2.imshow("ASL device", render(small, res, status))
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                elif k == 32:
                    cmd = "space"
                elif k == 8:
                    cmd = "backspace"
                elif k == 13:
                    cmd = "speak"
                elif k == ord("c"):
                    cmd = "clear"
                elif k == ord("f"):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty("ASL device", cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        if window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
