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
import translator
from dashboard_server import DashboardServer

PI_PORT = 9000
HOST_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pi_host")
_AR_FONT = None
_BANNER_CACHE = (None, None)


def _get_ar_font():
    global _AR_FONT
    if _AR_FONT is None:
        from PIL import ImageFont
        for p in (r"C:\Windows\Fonts\tahoma.ttf", r"C:\Windows\Fonts\arial.ttf"):
            if os.path.exists(p):
                _AR_FONT = ImageFont.truetype(p, 26)
                break
        if _AR_FONT is None:
            _AR_FONT = ImageFont.load_default()
    return _AR_FONT


def _draw_gui_banner(view, b, text, translated):
    global _BANNER_CACHE
    h, w = view.shape[:2]
    badge = f"{b.sign_lang.upper()}>{b.out_lang.upper()} | {b.mode.upper()}"
    has_ar = any("\u0600" <= ch <= "\u06ff" for ch in (translated or text or ""))
    if has_ar:
        strip_h = min(66, h)
        cache_key = (w, strip_h, badge, (text or "")[-36:], (translated or "")[-36:])
        if _BANNER_CACHE[0] == cache_key:
            view[:strip_h, :] = _BANNER_CACHE[1]
            return view
        from PIL import Image, ImageDraw
        pil_strip = Image.new("RGB", (w, strip_h), (15, 23, 42))
        draw = ImageDraw.Draw(pil_strip)
        font = _get_ar_font()
        draw.text((12, 8), badge, font=font, fill=(45, 212, 191))
        shown_ar = translator.shape_arabic((translated or text)[-36:])
        draw.text((12, 34), shown_ar, font=font, fill=(255, 255, 255))
        if translated and text and translated != text and not any("\u0600" <= ch <= "\u06ff" for ch in text):
            draw.text((max(12, w - 260), 8), text[-22:], font=font, fill=(245, 158, 11))
        bgr_strip = cv2.cvtColor(np.asarray(pil_strip), cv2.COLOR_RGB2BGR)
        _BANNER_CACHE = (cache_key, bgr_strip)
        view[:strip_h, :] = bgr_strip
        return view
    cv2.rectangle(view, (0, 0), (w, 46), (15, 23, 42), -1)
    cv2.putText(view, f"[{badge}]  {text[-32:]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
    return view


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


def handle(b, hdr, jpg, last, gui, hub=None):
    req = hdr.get("mode", "")
    current_combined = _combined_mode(b)
    if req and req != current_combined:
        b.set_mode("words_30" if req == "words" else req)
        log("Device", f"mode -> {b.mode} (vocab={b.word_vocab_mode})")
        if hub:
            hub.add_log("Device", f"mode -> {b.mode} (vocab={b.word_vocab_mode})")
    req_sign = hdr.get("sign_lang", "")
    if req_sign and req_sign != b.sign_lang:
        b.set_sign_lang(req_sign)
        log("Device", f"sign -> {b.sign_lang}")
    req_out = hdr.get("out_lang", "")
    if req_out and req_out != b.out_lang:
        b.set_out_lang(req_out)
        log("Device", f"out -> {b.out_lang}")

    all_cmds = list(hdr.get("cmds", [])) + (hub.drain_cmds() if hub else [])
    for c in all_cmds:
        if c.startswith("mode:"):
            m = c[5:]
            b.set_mode("words_30" if m == "words" else m)
            log("Device", f"mode -> {b.mode} (vocab={b.word_vocab_mode})")
        else:
            b.command(c)
            log("Device", c)
    speak = ""
    top3 = list(b.last_top3)
    res = {}
    if jpg:
        frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            res = b.process(frame)
            last["hand"] = 1 if res.get("hand") else 0
            last["num_hands"] = len(res.get("hands") or ([res["hand"]] if res.get("hand") else []))
            last["ms"] = float(res.get("ms") or 0.0)
            last["guess"] = str(res.get("guess") or res.get("letter") or "")
            last["hint"] = str(res.get("top2") or res.get("motion") or "")[:39]
            last["hint_ar"] = res.get("hint_ar") or ""
            last["guess_ar"] = res.get("guess_ar") or ""
            top3 = res.get("top3") or top3
            if res.get("typed") and b.mode in ("words", "alphabets"):
                speak = res["typed"]
            if res.get("say"):
                speak = res["say"]
            want_vid = bool(hub and hub.video_clients > 0 and (time.time() - last.get("vid_t", 0.0) >= 0.08))
            if gui or want_vid:
                view = cv2.flip(frame, 1)
                h, w = view.shape[:2]
                for hand in (res.get("hands") or ([res["hand"]] if res.get("hand") else [])):
                    pts = [(int((1 - x) * w), int(y * h)) for x, y in hand]
                    for p1, p2 in common.CONNECTIONS:
                        cv2.line(view, pts[p1], pts[p2], (0, 255, 0), 2)
                text_now = current_text(b)
                trans_now = res.get("translated_text") or ""
                view = _draw_gui_banner(view, b, text_now, trans_now)
                if want_vid:
                    ok_enc, enc_jpg = cv2.imencode(".jpg", view, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    if ok_enc:
                        hub.push_frame(enc_jpg.tobytes())
                        last["vid_t"] = time.time()
                if gui:
                    cv2.imshow("AI engine (debug)", view)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("l"):
                        b.command("toggle_sign")
                        log("GUI", f"sign -> {b.sign_lang}")
                    elif k == ord("t"):
                        b.command("toggle_lang")
                        log("GUI", f"out -> {b.out_lang}")

    text = current_text(b)
    translated = res.get("translated_text") if res else b._translate_cached(text)
    if text != last.get("text"):
        log("AI", f"[{b.sign_lang}/{b.out_lang}/{b.mode}] {text[-60:]!r}" + (f" -> {translated!r}" if translated and translated != text else ""))
        if hub:
            hub.add_log("AI", f"[{b.sign_lang}/{b.out_lang}/{b.mode}] {translated or text}")
    last["text"] = text
    disp_mode = _combined_mode(b)
    ar_display = (translated or text) if b.out_lang == "ar" else ""
    if ar_display != last.get("_ar_disp"):
        last["_ar_disp"] = ar_display
        last["_ar_hex"] = translator.encode_esp_arabic(ar_display, max_chars=24) if ar_display else ""
    ar_hex = last["_ar_hex"]

    hint_ar_src = (last.get("hint_ar") or "") if b.out_lang == "ar" else ""
    if hint_ar_src != last.get("_hint_ar_src"):
        last["_hint_ar_src"] = hint_ar_src
        last["_ar_hint_hex"] = translator.encode_esp_arabic(hint_ar_src, max_chars=24) if hint_ar_src else ""
    ar_hint_hex = last["_ar_hint_hex"]

    guess_ar_src = (last.get("guess_ar") or "") if b.out_lang == "ar" else ""
    if guess_ar_src != last.get("_guess_ar_src"):
        last["_guess_ar_src"] = guess_ar_src
        last["_ar_guess_hex"] = translator.encode_esp_arabic(guess_ar_src, max_chars=6) if guess_ar_src else ""
    ar_guess_hex = last["_ar_guess_hex"]
    # ASCII-safe fallback for Pi state when output is Arabic
    esp_ascii = text if not any(ord(ch) >= 127 for ch in text) else ""
    if hub:
        hub.update(
            connected=True,
            latency_ms=last.get("ms", 0.0),
            hand=bool(last["hand"]),
            num_hands=last.get("num_hands", 0),
            mode=disp_mode,
            sign_lang=b.sign_lang,
            out_lang=b.out_lang,
            text=text,
            translated_text=translated or (text if b.out_lang == "ar" else ""),
            letter=last["guess"],
            top2=last["hint"],
            top3=top3,
        )
    return {
        "seq": hdr.get("seq", 0),
        "req_mode": req,
        "mode": disp_mode,
        "sign_lang": b.sign_lang,
        "out_lang": b.out_lang,
        "speak_lang": b.out_lang,
        "text": esp_ascii if ar_hex else text,
        "full_text": translated or text,
        "translated_text": translated or "",
        "ar_hex": ar_hex,
        "ar_hint_hex": ar_hint_hex,
        "ar_guess_hex": ar_guess_hex,
        "hand": last["hand"],
        "guess": last["guess"] if not any(ord(ch) >= 127 for ch in str(last["guess"])) else "",
        "hint": last["hint"],
        "speak": speak,
    }


def session(b, host, gui, hub=None):
    s = socket.create_connection((host, PI_PORT), timeout=5)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.settimeout(15)
    try:
        open(HOST_CACHE, "w").write(host)
    except OSError:
        pass
    log("Net", f"connected to the Pi at {host}. Use the device touch screen or dashboard; Ctrl+C here to stop.")
    if hub:
        hub.update(connected=True, pi_host=host)
        hub.add_log("Net", f"connected to Pi at {host}")
    last = {"hand": 0, "num_hands": 0, "ms": 0.0, "guess": "", "hint": "", "text": None}
    n, t0, busy = 0, time.time(), 0.0
    while True:
        hlen, jlen = struct.unpack(">II", recv_exact(s, 8))
        hdr = json.loads(recv_exact(s, hlen))
        jpg = recv_exact(s, jlen) if jlen else b""
        t = time.time()
        out = json.dumps(handle(b, hdr, jpg, last, gui, hub=hub), ensure_ascii=False).encode("utf-8")
        s.sendall(struct.pack(">I", len(out)) + out)
        if jpg:
            n += 1
            busy += time.time() - t
        if time.time() - t0 >= 5:
            fps_now = round(n / (time.time() - t0), 1)
            if hub:
                hub.update(fps=fps_now)
            if n and (time.time() - t0 >= 10):
                log("AI", f"{fps_now:.1f} frames/s, {busy / n * 1000:.0f} ms/frame")
            if time.time() - t0 >= 10:
                n, t0, busy = 0, time.time(), 0.0


def main():
    ap = argparse.ArgumentParser(description="ASL / ArSL AI engine + Localhost Dashboard (laptop side)")
    ap.add_argument("--pi", default=None, help="Pi address (default: auto-detect)")
    ap.add_argument("--gui", action="store_true", help="show a debug camera window (press l=ASL/ArSL, t=EN/AR)")
    ap.add_argument("--dashboard-port", type=int, default=8080, help="localhost dashboard port (default: 8080)")
    ap.add_argument("--no-dashboard", action="store_true", help="disable embedded localhost dashboard")
    args = ap.parse_args()

    log("AI", "loading models...")
    b = server.build_real_brain()
    log("AI", "models ready")
    hub = None
    if not args.no_dashboard:
        try:
            ds = DashboardServer(host="127.0.0.1", port=args.dashboard_port)
            hub = ds.start_in_thread()
            hub.add_log("AI", "models loaded & ready")
            log("Dashboard", f"open http://127.0.0.1:{args.dashboard_port} in your browser")
        except OSError as e:
            log("Dashboard", f"port {args.dashboard_port} unavailable ({e}); continuing without dashboard")
    try:
        while True:
            host = find_pi(args.pi)
            if not host:
                if hub:
                    hub.update(connected=False)
                log("Net", "Pi not found; is it powered and on the same Wi-Fi? retrying in 3 s")
                time.sleep(3)
                continue
            try:
                session(b, host, args.gui, hub=hub)
            except (OSError, ConnectionError, struct.error, ValueError) as e:
                if hub:
                    hub.update(connected=False)
                    hub.add_log("Net", f"link lost ({e})")
                log("Net", f"link lost ({e}); reconnecting")
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        log("AI", "stopped")


if __name__ == "__main__":
    main()
