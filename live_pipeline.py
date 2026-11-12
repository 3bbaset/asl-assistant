"""live_pipeline.py - ASL Communicator V10 background engine.

The hardware is the product:
  [Pi Camera] -> [Raspberry Pi] --ethernet--> [this laptop: recognition only]
  [ESP32 touch screen] <--USB serial--> [this laptop]

Everything the user does happens on the ESP32 screen (mode tabs, SPACE, DEL,
CLEAR, SPEAK). The laptop runs headless; pass --gui to see a debug window.

Run:  .venv\\Scripts\\python.exe live_pipeline.py [--gui] [--esp-port COM14]
"""
import argparse
import os
import subprocess
import threading
import time

import cv2
import numpy as np
import paramiko
import serial
from serial.tools import list_ports

import common
import server

PI_HOSTS = ["192.168.0.23", "169.254.83.50", "fe80::76bf:bb6e:2582:f5e9%3", "raspberrypi.local"]
PI_USER, PI_PASS, PI_PORT = "pi", "raspberry", 8888
MODES = ("alphabets", "spelling", "words", "sentences")


def log(tag, msg):
    print(f"{time.strftime('%H:%M:%S')} [{tag}] {msg}", flush=True)


# ---------------------------------------------------------------- speech
class Speaker:
    """Speaks directly on the Raspberry Pi's audio hardware using espeak-ng,
    with local fallback to Windows TTS if the Pi connection is offline."""
    def __init__(self, pi_cam=None):
        self.pi_cam = pi_cam
        self.local_p = None
        try:
            self.local_p = subprocess.Popen(
                ["powershell", "-NoProfile", "-Command",
                 "Add-Type -AssemblyName System.Speech;"
                 "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                 "while(($l=[Console]::In.ReadLine()) -ne $null){ if($l.Trim()){ $s.Speak($l) } }"],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                text=True, creationflags=0x08000000 if os.name == "nt" else 0)
        except Exception as e:
            log("TTS", f"local fallback unavailable: {e}")

    def say(self, s):
        s = "".join(ch for ch in s if ch.isalnum() or ch in " '").strip()
        if not s:
            return

        spoken_on_pi = False
        if self.pi_cam and self.pi_cam.ssh:
            try:
                # Trigger speech on Raspberry Pi audio hardware
                self.pi_cam.ssh.exec_command(f'nohup espeak-ng -v en -s 145 "{s}" >/dev/null 2>&1 &')
                spoken_on_pi = True
            except Exception:
                spoken_on_pi = False

        if not spoken_on_pi and self.local_p and self.local_p.poll() is None:
            try:
                self.local_p.stdin.write(s + "\n")
                self.local_p.stdin.flush()
            except Exception:
                pass

    def close(self):
        if self.local_p:
            self.local_p.kill()


# ---------------------------------------------------------------- ESP32 link
def find_esp_port():
    for p in list_ports.comports():
        if "CH340" in (p.description or "") or (p.vid == 0x1A86):
            return p.device
    return None


class ESP32Link:
    """Background thread: keeps the serial port open, sends state, collects touch commands."""
    def __init__(self, port=None):
        self.fixed_port = port
        self.ser = None
        self.lock = threading.Lock()
        self.state = {}
        self.sent = {}
        self.commands = []
        self.running = True
        self.last_send = 0.0
        threading.Thread(target=self._run, daemon=True).start()

    def set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                v = str(v).replace("|", "/").replace("\n", " ")
                self.state[k] = v

    def pop_commands(self):
        with self.lock:
            c, self.commands = self.commands, []
        return c

    def raw(self, line):
        if self.ser:
            try:
                self.ser.write((line + "\n").encode())
            except Exception:
                pass

    def _open(self):
        port = self.fixed_port or find_esp_port()
        if not port:
            return False
        try:
            s = serial.Serial()
            s.port, s.baudrate, s.timeout = port, 115200, 0
            s.dtr = False; s.rts = False          # do not reset the board
            s.open()
            self.ser = s
            self.sent = {}
            log("ESP32", f"connected on {port}")
            return True
        except Exception as e:
            log("ESP32", f"cannot open {port}: {e}")
            return False

    def _run(self):
        buf = b""
        while self.running:
            if self.ser is None:
                if not self._open():
                    time.sleep(2); continue
            try:
                data = self.ser.read(256)
                if data:
                    buf += data
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.decode("utf-8", "ignore").strip()
                        if line.startswith("CMD:"):
                            with self.lock:
                                self.commands.append(line[4:])
                        elif line in ("READY", "CAL:OK"):
                            log("ESP32", line); self.sent = {}
                with self.lock:
                    diff = {k: v for k, v in self.state.items() if self.sent.get(k) != v}
                now = time.time()
                if diff or now - self.last_send > 1.0:  # changes immediately, heartbeat every 1 s
                    payload = diff if diff else {"c": self.state.get("c", "0")}
                    msg = "S|" + "|".join(f"{k}={v}" for k, v in payload.items()) + "\n"
                    self.ser.write(msg.encode("utf-8"))
                    self.sent.update(payload)
                    self.last_send = now
                time.sleep(0.02)
            except Exception as e:
                log("ESP32", f"link lost ({e}); reconnecting")
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None
                time.sleep(1)

    def close(self):
        self.running = False
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass


# ---------------------------------------------------------------- Pi camera
class PiCamera:
    """SSH to the Pi, run rpicam-vid as an MJPEG TCP server, tunnel it, decode frames.
    Reconnects automatically if the stream stalls."""
    def __init__(self):
        self.frame, self.frame_time = None, 0.0
        self.lock = threading.Lock()
        self.running = True
        self.ssh = None
        threading.Thread(target=self._run, daemon=True).start()

    def latest(self):
        with self.lock:
            return self.frame, self.frame_time

    def ok(self):
        return time.time() - self.frame_time < 1.5

    def _connect(self):
        for host in PI_HOSTS:
            try:
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(host, username=PI_USER, password=PI_PASS, timeout=5,
                            banner_timeout=5, auth_timeout=5)
                log("Pi", f"SSH connected ({host})")
                return ssh
            except Exception:
                continue
        return None

    def _run(self):
        while self.running:
            self.ssh = self._connect()
            if not self.ssh:
                log("Pi", "not reachable, retrying in 3 s"); time.sleep(3); continue
            try:
                self.ssh.exec_command("pkill -9 rpicam-vid; sleep 0.3; nohup rpicam-vid -t 0 -n "
                                      "--codec mjpeg --width 640 --height 480 --framerate 30 -l "
                                      f"-o tcp://127.0.0.1:{PI_PORT} >/dev/null 2>&1 &")
                chan = None
                for _ in range(20):
                    time.sleep(0.3)
                    try:
                        chan = self.ssh.get_transport().open_channel(
                            "direct-tcpip", ("127.0.0.1", PI_PORT), ("127.0.0.1", 0))
                        break
                    except Exception:
                        pass
                if chan is None:
                    raise RuntimeError("camera server did not start")
                chan.settimeout(3.0)
                log("Pi", "camera stream live")
                self._read(chan)
            except Exception as e:
                log("Pi", f"stream error: {e}")
            try:
                self.ssh.exec_command("pkill -9 rpicam-vid")
                self.ssh.close()
            except Exception:
                pass
            time.sleep(1)

    def _read(self, chan):
        buf = b""
        while self.running:
            data = chan.recv(65536)
            if not data:
                raise RuntimeError("stream closed")
            buf += data
            while True:
                a = buf.find(b"\xff\xd8")
                if a < 0:
                    buf = buf[-1:]; break
                b = buf.find(b"\xff\xd9", a + 2)
                if b < 0:
                    buf = buf[a:]; break
                jpg, buf = buf[a:b + 2], buf[b + 2:]
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    with self.lock:
                        self.frame, self.frame_time = img, time.time()

    def close(self):
        self.running = False
        try:
            self.ssh.exec_command("pkill -9 rpicam-vid"); self.ssh.close()
        except Exception:
            pass


# ---------------------------------------------------------------- main loop
def apply_command(b, c, speaker):
    if c.startswith("MODE:"):
        m = c[5:].strip()
        b.set_mode(m)
        log("Device", f"mode -> {b.mode}" + ("" if b.mode == m else f" (requested {m}, unavailable)"))
    elif c == "SPACE":
        b.command("space")
    elif c == "DEL":
        b.command("backspace")
    elif c == "CLEAR":
        b.command("clear")
    elif c == "SPEAK":
        b.command("speak")
    log("Device", c) if not c.startswith("MODE:") else None


def main():
    ap = argparse.ArgumentParser(description="ASL Communicator background engine")
    ap.add_argument("--mode", default="spelling", choices=MODES)
    ap.add_argument("--esp-port", default=None, help="default: auto-detect CH340")
    ap.add_argument("--gui", action="store_true", help="show a debug camera window on the laptop")
    args = ap.parse_args()

    esp = ESP32Link(args.esp_port)
    esp.set(m=args.mode, t="", h=0, c=0, g="", i="Starting engine...")
    cam = PiCamera()
    speaker = Speaker(pi_cam=cam)

    log("AI", "loading models...")
    b = server.build_real_brain()
    b.set_mode(args.mode)
    log("AI", f"ready, mode={b.mode}. Control everything from the device screen. Ctrl+C to stop.")

    last_ts, fps, n, t0 = 0.0, 0.0, 0, time.time()
    try:
        while True:
            for c in esp.pop_commands():
                apply_command(b, c, speaker)

            frame, ts = cam.latest()
            if frame is None or ts == last_ts:
                if not cam.ok():
                    esp.set(c=0, h=0, m=b.mode, i="Waiting for camera...")
                time.sleep(0.005)
                continue
            last_ts = ts

            res = b.process(frame)
            n += 1
            if time.time() - t0 >= 1.0:
                fps, n, t0 = n / (time.time() - t0), 0, time.time()

            if res.get("typed") and b.mode in ("words", "sentences", "alphabets"):
                speaker.say(res["typed"])
            if res.get("say"):
                speaker.say(res["say"])

            hand = bool(res.get("hand"))
            guess = res.get("letter") or ""
            if b.mode in ("words", "sentences"):
                guess = ""
            hint = res.get("top2") or res.get("motion") or ""
            esp.set(m=b.mode, t=res.get("text", ""), h=int(hand), c=1, g=guess,
                    i=f"{hint}"[:39])

            if args.gui:
                view = cv2.flip(frame, 1)
                h, w = view.shape[:2]
                if hand:
                    pts = [(int((1 - x) * w), int(y * h)) for x, y in res["hand"]]
                    for p1, p2 in common.CONNECTIONS:
                        cv2.line(view, pts[p1], pts[p2], (0, 255, 0), 2)
                cv2.putText(view, f"{b.mode.upper()} {fps:.0f}fps  {res.get('text','')[-30:]}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("ASL engine (debug)", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        esp.set(c=0, h=0, i="Engine stopped")
        time.sleep(0.2)
        cam.close(); esp.close(); speaker.close()
        cv2.destroyAllWindows()
        log("System", "stopped")


if __name__ == "__main__":
    main()
