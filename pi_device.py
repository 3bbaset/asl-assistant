#!/usr/bin/env python3
"""pi_device.py - the ASL Communicator device. Runs on the Raspberry Pi, starts at boot.

The Pi owns every piece of hardware:
  * Pi camera          -> captures MJPEG frames
  * ESP32 touch screen -> UART /dev/serial0 (3 jumpers: GND, TX->IO22, RX<-IO27)
  * Speaker            -> espeak-ng on the 3.5 mm jack
The laptop is only the AI: it connects to this Pi over Wi-Fi (TCP port 9000),
receives camera frames, and answers with the recognition result.

Wire format (TCP, big endian)
  Pi -> AI : [u32 header_len][u32 jpeg_len][header json][jpeg bytes]
             header = {"seq":n, "mode":"spelling", "cmds":["space","backspace","clear"]}
  AI -> Pi : [u32 len][result json]
             result = {"seq","req_mode","mode","text","hand","guess","hint","speak"}
Pi -> ESP32 (UART, text lines):  S|m=..|t=..|h=0/1|c=0/1|a=0/1|g=..|i=..
ESP32 -> Pi:                     CMD:MODE:<mode>  CMD:SPACE  CMD:DEL  CMD:CLEAR  CMD:SPEAK  READY  PONG
"""
import json
import queue
import select
import socket
import struct
import subprocess
import threading
import time

try:
    import serial
except ImportError:  # installed by deploy_pi.py
    serial = None

NET_PORT = 9000
UART_DEV = "/dev/serial0"
UART_BAUD = 115200
CAM_CMD = ["rpicam-vid", "-t", "0", "-n", "--codec", "mjpeg", "--width", "640", "--height", "480",
           "--framerate", "30", "--quality", "65", "--flush", "-o", "-"]


def log(tag, msg):
    print(f"[{tag}] {msg}", flush=True)


def clean(s):
    """ESP32 font is plain ASCII and '|' is our field separator."""
    return "".join(ch if 32 <= ord(ch) < 127 and ch != "|" else " " for ch in str(s))


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.mode = "spelling"
        self.text = ""
        self.hand = 0
        self.guess = ""
        self.hint = ""
        self.ai = 0                # laptop AI engine connected and answering
        self.frame = None          # newest camera JPEG
        self.frame_ts = 0.0
        self.cmds = []             # edit commands waiting to go to the AI with the next frame
        self.seq = 0               # sequence number of the last message sent to the AI
        self.text_gate = 0         # ignore text from results older than the last local edit
        self.mode_gate = 0         # ignore mode from results older than the last local mode tap
        self.lang_gate = 0         # ignore sign/out lang older than the last local lang tap
        self.sign_lang = "asl"     # "asl" or "arsl"
        self.out_lang = "en"       # "en" or "ar"
        self.ar_hex = ""           # pre-shaped 16px glyph indices for ESP32 text box
        self.ar_hint_hex = ""      # pre-shaped Arabic hint indices for ESP32 bottom bar
        self.ar_guess_hex = ""     # pre-shaped Arabic guess indices for ESP32 guess box
        self.speak_text = ""       # full UTF-8 text (including Arabic) for Speech
        self.vocab_mode = "30"

    def cam_ok(self):
        return time.time() - self.frame_ts < 1.5


# ------------------------------------------------------------------ camera
def camera_thread(st):
    while True:
        p = None
        try:
            subprocess.run(["pkill", "-x", "rpicam-vid"], stderr=subprocess.DEVNULL)
            time.sleep(0.3)
            p = subprocess.Popen(CAM_CMD, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
            log("Camera", "started")
            buf = bytearray()
            while True:
                chunk = p.stdout.read(65536)
                if not chunk:
                    raise RuntimeError("camera pipe closed")
                buf += chunk
                while True:
                    a = buf.find(b"\xff\xd8")
                    if a < 0:
                        del buf[:-1]
                        break
                    b = buf.find(b"\xff\xd9", a + 2)
                    if b < 0:
                        del buf[:a]
                        break
                    frame = bytes(buf[a:b + 2])
                    del buf[:b + 2]
                    with st.lock:
                        st.frame, st.frame_ts = frame, time.time()
        except Exception as e:
            log("Camera", f"error: {e}; restarting in 2 s")
        finally:
            if p:
                try:
                    p.kill()
                except Exception:
                    pass
        time.sleep(2)


# ------------------------------------------------------------------ speech
class Speech:
    def __init__(self):
        self.q = queue.Queue()
        subprocess.run(["amixer", "sset", "PCM", "100%", "unmute"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, s, lang="en", interrupt=False):
        s = "".join(
            ch for ch in str(s)
            if ch.isalnum() or ch in " '.,?!،؟" or "\u0600" <= ch <= "\u06ff"
        ).strip()
        if not s:
            return
        voice = "ar" if (lang == "ar" or any("\u0600" <= ch <= "\u06ff" for ch in s)) else "en-us"
        speed = "135" if voice == "ar" else "150"
        if interrupt:
            while not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
        elif self.q.qsize() > 3:
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
        self.q.put((voice, speed, s))

    def _run(self):
        while True:
            voice, speed, s = self.q.get()
            log("Speech", f"({voice}) {s}")
            subprocess.run(["espeak-ng", "-v", voice, "-s", speed, "-a", "200", s],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ------------------------------------------------------------------ ESP32 touch screen
class Esp:
    def __init__(self, st, speech):
        self.st, self.speech = st, speech
        self.ser = None
        self.buf = b""
        self.sent = {}
        self.last_tx = self.last_hb = self.last_ping = 0.0
        self.esp_alive = False
        threading.Thread(target=self.run, daemon=True).start()

    def view(self):
        st = self.st
        with st.lock:
            cam, ai = st.cam_ok(), st.ai
            if not ai:
                hint = "Start the AI engine on the laptop"
            elif not cam:
                hint = "Camera is not sending frames"
            else:
                hint = st.hint
            return {"m": st.mode, "l": st.sign_lang, "o": st.out_lang,
                    "t": clean(st.text)[-60:], "u": st.ar_hex,
                    "ia": st.ar_hint_hex, "ga": st.ar_guess_hex,
                    "h": int(bool(st.hand and ai and cam)),
                    "c": int(cam), "a": int(ai), "g": clean(st.guess)[:6], "i": clean(hint)[:26]}

    def queue_cmd(self, c):
        with self.st.lock:
            self.st.cmds.append(c)
            self.st.text_gate = self.st.seq + 1
            if c == "clear":
                self.st.text = ""
                self.st.ar_hex = ""
                self.st.ar_hint_hex = ""
                self.st.ar_guess_hex = ""
                self.st.speak_text = ""

    def on_line(self, line):
        st = self.st
        if line.startswith("CMD:"):
            cmd = line[4:]
            log("Touch", cmd)
            if cmd.startswith("MODE:"):
                m = cmd[5:].strip()
                with st.lock:
                    st.mode = m
                    st.hint = ""
                    st.mode_gate = st.seq + 1
                    st.cmds.append("mode:" + m)
            elif cmd.startswith("SIGN:"):
                sl = cmd[5:].strip().lower()
                if sl in ("asl", "arsl"):
                    with st.lock:
                        st.sign_lang = sl
                        if sl == "arsl" and st.out_lang == "en":
                            st.out_lang = "ar"
                        st.text = ""
                        st.ar_hex = ""
                        st.ar_hint_hex = ""
                        st.ar_guess_hex = ""
                        st.speak_text = ""
                        st.hint = ""
                        st.lang_gate = st.seq + 1
                        st.text_gate = st.seq + 1
                        st.cmds.append("sign:" + sl)
            elif cmd.startswith("LANG:"):
                ol = cmd[5:].strip().lower()
                if ol in ("en", "ar"):
                    with st.lock:
                        st.out_lang = ol
                        st.lang_gate = st.seq + 1
                        st.cmds.append("lang:" + ol)
            elif cmd == "SPACE":
                self.queue_cmd("space")
            elif cmd == "DEL":
                self.queue_cmd("backspace")
            elif cmd == "CLEAR":
                self.queue_cmd("clear")
            elif cmd == "SPEAK":
                with st.lock:
                    t = st.speak_text or st.text
                    ol = st.out_lang
                self.speech.say(t if t.strip() else "Nothing to say", lang=ol, interrupt=True)
            elif cmd == "TOGGLE_VOCAB":
                self.queue_cmd("toggle_vocab")
        elif line == "READY":
            log("ESP32", "screen booted")
            self.sent = {}
        elif line == "PONG":
            if not self.esp_alive:
                log("ESP32", "link OK (PONG received)")
            self.esp_alive = True

    def write(self, payload):
        self.ser.write(("S|" + "|".join(f"{k}={v}" for k, v in payload.items()) + "\n").encode("ascii", "ignore"))

    def run(self):
        while True:
            if self.ser is None:
                try:
                    self.ser = serial.Serial(UART_DEV, UART_BAUD, timeout=0.02)
                    self.sent = {}
                    log("ESP32", f"UART {UART_DEV} open")
                except Exception as e:
                    log("ESP32", f"cannot open {UART_DEV}: {e}")
                    time.sleep(3)
                    continue
            try:
                data = self.ser.read(self.ser.in_waiting or 1)
                if data:
                    self.buf += data
                    while b"\n" in self.buf:
                        line, self.buf = self.buf.split(b"\n", 1)
                        line = line.decode("utf-8", "ignore").strip()
                        if line:
                            self.on_line(line)
                    if len(self.buf) > 400:
                        self.buf = b""
                now = time.time()
                if now - self.last_ping > 10:
                    self.last_ping = now
                    self.ser.write(b"PING\n")
                view = self.view()
                diff = {k: v for k, v in view.items() if self.sent.get(k) != v}
                if diff and now - self.last_tx >= 0.08:
                    self.write(diff)
                    self.sent.update(diff)
                    self.last_tx = self.last_hb = now
                elif now - self.last_hb > 1.0:
                    self.write({"c": view["c"]})
                    self.last_hb = now
            except Exception as e:
                log("ESP32", f"UART error: {e}")
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None
                time.sleep(1)


# ------------------------------------------------------------------ link to the laptop AI
def recv_exact(conn, n):
    out = bytearray()
    while len(out) < n:
        chunk = conn.recv(n - len(out))
        if not chunk:
            raise ConnectionError("AI engine closed the connection")
        out += chunk
    return bytes(out)


def serve_ai(st, speech, conn):
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.settimeout(8)
    waiting, sent_at, last_ts = False, 0.0, 0.0
    while True:
        now = time.time()
        if not waiting:
            with st.lock:
                frame, ts = st.frame, st.frame_ts
                new = frame is not None and ts != last_ts
                cmds = st.cmds
                send = new or bool(cmds) or now - sent_at > 2.0     # idle ping every 2 s
                if send:
                    st.cmds = []
                    st.seq += 1
                    seq, mode, sl, ol = st.seq, st.mode, st.sign_lang, st.out_lang
            if send:
                hdr = json.dumps({"seq": seq, "mode": mode, "sign_lang": sl, "out_lang": ol, "cmds": cmds}).encode()
                jpg = frame if new else b""
                conn.sendall(struct.pack(">II", len(hdr), len(jpg)) + hdr + jpg)
                if new:
                    last_ts = ts
                waiting, sent_at = True, now
        readable, _, _ = select.select([conn], [], [], 0.02)
        if readable:
            n = struct.unpack(">I", recv_exact(conn, 4))[0]
            res = json.loads(recv_exact(conn, n).decode("utf-8"))
            with st.lock:
                st.ai = 1
                st.hand = 1 if res.get("hand") else 0
                st.guess, st.hint = res.get("guess", ""), res.get("hint", "")
                st.ar_hint_hex = res.get("ar_hint_hex", "")
                st.ar_guess_hex = res.get("ar_guess_hex", "")
                if res.get("seq", 0) >= st.text_gate:
                    st.text = res.get("text", "")
                    st.ar_hex = res.get("ar_hex", "")
                    st.speak_text = res.get("full_text", res.get("text", ""))
                if res.get("seq", 0) >= st.mode_gate and res.get("mode"):
                    st.mode = res["mode"]
                if res.get("seq", 0) >= st.lang_gate:
                    if res.get("sign_lang"):
                        st.sign_lang = res["sign_lang"]
                    if res.get("out_lang"):
                        st.out_lang = res["out_lang"]
            if res.get("speak"):
                speech.say(res["speak"], lang=res.get("speak_lang", "en"))
            waiting = False
        elif waiting and time.time() - sent_at > 6:
            raise TimeoutError("AI engine stopped answering")


def net_thread(st, speech):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", NET_PORT))
    srv.listen(2)
    log("Net", f"waiting for the laptop AI engine on port {NET_PORT}")
    while True:
        conn, addr = srv.accept()
        log("Net", f"AI engine connected from {addr[0]}")
        try:
            serve_ai(st, speech, conn)
        except Exception as e:
            log("Net", f"AI engine disconnected: {e}")
        finally:
            conn.close()
            with st.lock:
                st.ai, st.hand, st.guess = 0, 0, ""


def main():
    st = State()
    speech = Speech()
    Esp(st, speech)
    threading.Thread(target=camera_thread, args=(st,), daemon=True).start()
    speech.say("Communicator ready")
    net_thread(st, speech)


if __name__ == "__main__":
    main()
