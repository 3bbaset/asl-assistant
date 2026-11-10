"""live_pipeline.py - Complete End-to-End ASL Communicator Pipeline.

Architecture:
  [Pi Camera OV5647] -> [Raspberry Pi 2B] -> [Direct Ethernet Link] -> [Laptop AI Brain]
                                                                                |
  [ESP32 2.8" Screen & LEDs] <- [UART IO22/27 & USB COM15] <-------------------+
                               + [Windows TTS Speech Engine] <------------------+
"""
import sys
import os
import time
import socket
import threading
import subprocess
import argparse
import paramiko
import serial
import cv2
import numpy as np

def speak(s):
    """Speaks text asynchronously via Windows TTS."""
    s = "".join(ch for ch in s if ch.isalnum() or ch == " ")
    if not s.strip():
        return
    try:
        cmd = ["powershell", "-NoProfile", "-Command",
               f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{s}')"]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=0x08000000 if os.name == "nt" else 0)
    except Exception:
        pass

# ASL Recognition Modules
import common
import brain
import server

PI_HOST = "fe80::76bf:bb6e:2582:f5e9%3"
PI_USER = "pi"
PI_PASS = "raspberry"
PI_PORT = 8888

class ESP32Bridge:
    def __init__(self, port="COM15", baud=115200):
        self.port = port
        self.baud = baud
        self.ser = None
        self.running = True
        self.touch_queue = []
        self.lock = threading.Lock()

        # State mailbox
        self.pending_frame = None
        self.pending_text = None
        self.pending_sub = None
        self.pending_mode = None
        self.pending_led = None

        try:
            self.ser = serial.Serial()
            self.ser.port = port
            self.ser.baudrate = baud
            self.ser.dtr = False
            self.ser.rts = False
            self.ser.timeout = 0.05
            self.ser.open()
            print(f"[ESP32] Connected on {port} (Asynchronous Thread Active)")
            time.sleep(1.2) # Allow ESP32 to finish boot cleanly
        except Exception as e:
            print(f"[ESP32] Note: Direct USB port {port} not opened ({e}).")

        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def update(self, frame_bgr=None, mode=None, text=None, sub=None, led=None):
        with self.lock:
            if frame_bgr is not None:
                self.pending_frame = frame_bgr
            if mode is not None:
                self.pending_mode = mode
            if text is not None:
                self.pending_text = text
            if sub is not None:
                self.pending_sub = sub
            if led is not None:
                self.pending_led = led

    def send(self, mode=None, text=None, sub=None, led=None):
        self.update(mode=mode, text=text, sub=sub, led=led)

    def poll_touch(self):
        with self.lock:
            if self.touch_queue:
                return self.touch_queue.pop(0)
        return None

    def _worker(self):
        last_frame_sent = 0
        last_sent_text = ""
        last_sent_mode = ""
        last_sent_led = ""

        while self.running:
            if not (self.ser and self.ser.is_open):
                time.sleep(0.05)
                continue

            try:
                # 1. Read touch events from screen
                if self.ser.in_waiting:
                    line = self.ser.readline().decode("utf-8", "ignore").strip()
                    if line.startswith("CMD:"):
                        with self.lock:
                            self.touch_queue.append(line)

                # 2. Check if text/mode/led changed
                with self.lock:
                    cur_text = self.pending_text
                    cur_sub = self.pending_sub
                    cur_mode = self.pending_mode
                    cur_led = self.pending_led
                    cur_frame = self.pending_frame
                    self.pending_frame = None

                if (cur_text != last_sent_text or cur_mode != last_sent_mode or cur_led != last_sent_led) and cur_text is not None:
                    parts = []
                    if cur_mode: parts.append(f"mode:{cur_mode}")
                    if cur_text is not None: parts.append(f"text:{cur_text}")
                    if cur_sub is not None: parts.append(f"sub:{cur_sub}")
                    if cur_led: parts.append(f"led:{cur_led}")
                    msg = ";".join(parts) + "\n"
                    self.ser.write(msg.encode("utf-8"))
                    last_sent_text = cur_text
                    last_sent_mode = cur_mode
                    last_sent_led = cur_led

                # 3. Stream 60x45 camera frame at ~6 FPS
                now = time.time()
                if cur_frame is not None and (now - last_frame_sent) >= 0.16:
                    last_frame_sent = now
                    small = cv2.resize(cur_frame, (60, 45))
                    r = (small[:, :, 2] >> 3).astype(np.uint16)
                    g = (small[:, :, 1] >> 2).astype(np.uint16)
                    b = (small[:, :, 0] >> 3).astype(np.uint16)
                    rgb565 = ((r << 11) | (g << 5) | b).byteswap().tobytes()
                    self.ser.write(b"IMG:5400\n" + rgb565)

                time.sleep(0.01)
            except Exception:
                time.sleep(0.05)

    def close(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()

class PiCameraStream:
    def __init__(self, host=PI_HOST, user=PI_USER, password=PI_PASS, port=PI_PORT):
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.ssh = None
        self.chan = None
        self.running = True
        self.frame = None
        self.lock = threading.Lock()

        print("[Pi] Connecting to Raspberry Pi 2 via SSH...")
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(self.host, username=self.user, password=self.password, timeout=6)
        print("[Pi] Connected! Starting CSI camera pipeline...")

        # Kill any prior rpicam instances
        self.ssh.exec_command("killall -9 rpicam-vid 2>/dev/null")
        time.sleep(0.3)

        # Start rpicam-vid TCP server on port 8888
        def _cam_worker():
            cmd = f"rpicam-vid -t 0 --codec mjpeg --width 640 --height 480 --framerate 30 -l -o tcp://127.0.0.1:{self.port} --nopreview"
            self.ssh.exec_command(cmd)

        self.cam_thread = threading.Thread(target=_cam_worker, daemon=True)
        self.cam_thread.start()
        time.sleep(2.0)

        # Connect direct-tcpip channel through SSH
        transport = self.ssh.get_transport()
        self.chan = transport.open_channel("direct-tcpip", dest_addr=("127.0.0.1", self.port), src_addr=("127.0.0.1", 0))
        print("[Pi] Camera stream channel open and listening at 30 FPS.")

        self.stream_thread = threading.Thread(target=_stream_reader, args=(self,), daemon=True)
        self.stream_thread.start()

    def latest(self):
        with self.lock:
            f = self.frame
        return f

    def send_uart(self, line):
        if self.ssh:
            # Escape and send to /dev/serial0
            self.ssh.exec_command(f'echo "{line}" > /dev/serial0 2>/dev/null')

    def close(self):
        self.running = False
        if self.chan:
            self.chan.close()
        if self.ssh:
            self.ssh.exec_command("killall -9 rpicam-vid 2>/dev/null")
            self.ssh.close()

def _stream_reader(streamer):
    buf = b""
    while streamer.running:
        try:
            data = streamer.chan.recv(8192)
            if not data:
                time.sleep(0.01)
                continue
            buf += data
            while True:
                start = buf.find(b"\xff\xd8")
                if start == -1:
                    buf = buf[-2:]
                    break
                end = buf.find(b"\xff\xd9", start + 2)
                if end == -1:
                    buf = buf[start:]
                    break
                jpg = buf[start:end+2]
                buf = buf[end+2:]
                img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    with streamer.lock:
                        streamer.frame = img
        except Exception:
            time.sleep(0.05)

def main():
    parser = argparse.ArgumentParser(description="ASL Communicator Live System")
    parser.add_argument("--mode", default="words", choices=["alphabets", "spelling", "words", "sentences"])
    parser.add_argument("--esp-port", default="COM15")
    args = parser.parse_args()

    esp = ESP32Bridge(port=args.esp_port)
    esp.send(mode=args.mode, text="STARTING...", sub="Connecting Camera", led="yellow")

    cam = PiCameraStream()
    print("[AI] Loading ASL Neural Models (Alphabet, Dynamic J/Z, and Words)...")
    b = server.build_real_brain()
    b.mode = args.mode
    print(f"[AI] Models loaded! Initial mode: {b.mode.upper()}")

    cv2.namedWindow("ASL Communicator V9 - Live System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ASL Communicator V9 - Live System", 960, 720)

    last_text = ""
    last_led = ""
    t_fps = time.time()
    frames = 0
    fps = 30.0

    print("\n" + "="*60)
    print("ASL COMMUNICATOR V9 IS LIVE!")
    print("Keyboard shortcuts in camera window:")
    print("  '1': Alphabets mode (Single-letter inspection)")
    print("  '2': Spelling mode (Fingerspelling word buffer)")
    print("  '3': Words mode (Whole-word dynamic signs)")
    print("  '4': Sentences mode (Continuous multi-word grammar)")
    print("  'c': Clear sentence/word buffer")
    print("  'q': Quit")
    print("="*60 + "\n")

    esp.send(mode=b.mode, text="READY TO SIGN", sub="Tracking Active", led="green")

    try:
        while True:
            frame = cam.latest()
            if frame is None:
                time.sleep(0.01)
                continue

            frames += 1
            if time.time() - t_fps >= 1.0:
                fps = frames / (time.time() - t_fps)
                frames = 0
                t_fps = time.time()

            # 1. Check for hardware touch commands from ESP32 screen
            t_cmd = esp.poll_touch()
            if t_cmd:
                if t_cmd.startswith("CMD:MODE:"):
                    m = t_cmd.split(":", 2)[2].strip()
                    b.set_mode(m)
                    print(f"[Hardware Touch Screen] Mode -> {m.upper()}")
                elif t_cmd == "CMD:CLEAR":
                    b.command("clear")
                    print("[Hardware Touch Screen] Buffer Cleared")

            res = b.process(frame)

            # Check if text or mode updated
            current_text = res.get("text", "") or res.get("top2", "")
            current_mode = b.mode
            led_color = "green" if res.get("hand") else "blue"

            # Trigger TTS speech if engine produced a spoken token
            if res.get("typed"):
                speak(res["typed"])
            if res.get("say"):
                speak(res["say"])

            # Render UI with hand skeleton
            h, w = frame.shape[:2]
            preview = cv2.flip(frame, 1)

            # Draw hand skeleton if detected
            if res.get("hand"):
                pts = [(int((1 - x) * w), int(y * h)) for x, y in res["hand"]]
                for p1, p2 in common.CONNECTIONS:
                    cv2.line(preview, pts[p1], pts[p2], (0, 255, 0), 2)
                for pt in pts:
                    cv2.circle(preview, pt, 4, (0, 0, 255), -1)

            # 2. Asynchronously update ESP32 screen (camera viewport, caption, and LED)
            sub_msg = f"{fps:.0f} FPS | Hand: {'YES' if res.get('hand') else 'NO'}"
            esp.update(frame_bgr=preview, mode=current_mode, text=current_text[:16], sub=sub_msg, led=led_color)

            # Overlay banners
            cv2.rectangle(preview, (0, 0), (w, 60), (15, 23, 42), -1)
            cv2.putText(preview, f"MODE: {b.mode.upper()}  |  {fps:.1f} FPS", (15, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 220, 255), 2)

            cv2.rectangle(preview, (0, h - 70), (w, h), (15, 23, 42), -1)
            disp_txt = current_text if current_text else "(No sign in view)"
            cv2.putText(preview, disp_txt, (15, h - 25),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

            cv2.imshow("ASL Communicator V9 - Live System", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                b.set_mode("alphabets")
                esp.send(mode="alphabets", sub="Mode: Alphabets")
            elif key == ord('2'):
                b.set_mode("spelling")
                esp.send(mode="spelling", sub="Mode: Spelling")
            elif key == ord('3'):
                b.set_mode("words")
                esp.send(mode="words", sub="Mode: Words")
            elif key == ord('4'):
                b.set_mode("sentences")
                esp.send(mode="sentences", sub="Mode: Sentences")
            elif key == ord('c'):
                b.command("clear")
                esp.send(text="", sub="Buffer Cleared")
            elif key == ord(' '):
                b.command("space")
            elif key == 8: # Backspace
                b.command("backspace")
            elif key == 13: # Enter
                b.command("speak")

    finally:
        cam.close()
        esp.close()
        cv2.destroyAllWindows()
        print("[System] Clean shutdown complete.")

if __name__ == "__main__":
    main()
