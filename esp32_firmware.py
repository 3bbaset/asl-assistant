"""esp32_firmware.py - ASL Communicator V10 device firmware (ESP32-2432S028R "CYD").

The device is the user interface. The laptop is only a background recognizer.

Screen (320x240 landscape):
  +-------+-------+-------+-------+
  | ALPHA | SPELL | WORDS | SENT  |   mode tabs (tap)
  +-------+-------+-------+-------+
  | CAMERA  | HAND    |  GUESS    |   live status indicators
  +---------------------------------+
  |  recognized text (2 lines)      |
  |  hint / top guesses             |
  +-------+-------+-------+-------+
  | SPACE |  DEL  | CLEAR | SPEAK |   action buttons (tap)
  +-------+-------+-------+-------+

Protocol (text only, newline terminated, over USB serial and Pi UART):
  host -> device : S|m=<mode>|t=<text>|h=<0/1>|c=<0/1>|g=<guess>|i=<hint>
                   CAL            (start touch calibration)
                   REPL           (stop firmware, return to REPL for flashing)
                   PING
  device -> host : CMD:MODE:<mode>  CMD:SPACE  CMD:DEL  CMD:CLEAR  CMD:SPEAK
                   READY  PONG  TOUCH:x,y
Hold anywhere on the text area for 3 s to recalibrate the touch screen.
"""
import micropython
micropython.kbd_intr(-1)  # never let a stray 0x03 byte kill the firmware

from machine import Pin, SPI, SoftSPI, UART
import time, struct, framebuf, sys, select

# ------------------------------------------------------------------ LEDs
bl = Pin(21, Pin.OUT, value=1)
led_r = Pin(4, Pin.OUT, value=1)
led_g = Pin(16, Pin.OUT, value=1)
led_b = Pin(17, Pin.OUT, value=1)

def set_led(r, g, b):  # active low
    led_r.value(0 if r else 1); led_g.value(0 if g else 1); led_b.value(0 if b else 1)

# ------------------------------------------------------------------ Display
spi = SPI(1, baudrate=40000000, polarity=0, phase=0, sck=Pin(14), mosi=Pin(13), miso=Pin(12))
cs = Pin(15, Pin.OUT, value=1)
dc = Pin(2, Pin.OUT, value=1)

def cmd(c, data=None):
    cs.value(0); dc.value(0); spi.write(bytes([c]))
    if data:
        dc.value(1); spi.write(data)
    cs.value(1)

def init_tft():
    cmd(0x01); time.sleep_ms(120)
    for c, d in ((0xCF, b"\x00\xC1\x30"), (0xED, b"\x64\x03\x12\x81"), (0xE8, b"\x85\x00\x78"),
                 (0xCB, b"\x39\x2C\x00\x34\x02"), (0xF7, b"\x20"), (0xEA, b"\x00\x00"),
                 (0xC0, b"\x23"), (0xC1, b"\x10"), (0xC5, b"\x3e\x28"), (0xC7, b"\x86"),
                 (0x36, b"\x28"), (0x3A, b"\x55"), (0xB1, b"\x00\x18"), (0xB6, b"\x08\x82\x27")):
        cmd(c, d)
    cmd(0x11); time.sleep_ms(120); cmd(0x29)

def window(x, y, w, h):
    cmd(0x2A, struct.pack(">HH", x, x + w - 1))
    cmd(0x2B, struct.pack(">HH", y, y + h - 1))
    cs.value(0); dc.value(0); spi.write(b"\x2C"); dc.value(1)

def fill(x, y, w, h, c):
    window(x, y, w, h)
    n = w * h
    chunk = struct.pack(">H", c) * min(n, 512)
    while n > 0:
        k = min(n, 512)
        spi.write(chunk if k == 512 else chunk[:k * 2])
        n -= k
    cs.value(1)

def _sw(c):  # framebuf stores little endian, panel wants big endian
    return ((c & 0xFF) << 8) | (c >> 8)

@micropython.native
def _scale_row(src, off, bw, s, dst):
    d = 0
    for col in range(bw):
        b0 = src[off + 2 * col]
        b1 = src[off + 2 * col + 1]
        for _ in range(s):
            dst[d] = b0
            dst[d + 1] = b1
            d += 2

def text(x, y, w, h, s, fg, bg, scale=1, align="c"):
    """Draw text centred (or left aligned) in a box, redrawing the whole box."""
    bw, bh = w // scale, h // scale
    buf = bytearray(bw * bh * 2)
    fb = framebuf.FrameBuffer(buf, bw, bh, framebuf.RGB565)
    fb.fill(_sw(bg))
    lines = s.split("\n")
    ty = max(0, (bh - 10 * len(lines) + 2) // 2)
    for ln in lines:
        tx = 2 if align == "l" else max(0, (bw - len(ln) * 8) // 2)
        fb.text(ln, tx, ty, _sw(fg))
        ty += 10
    if scale == 1:
        window(x, y, bw, bh); spi.write(buf); cs.value(1)
        return
    line = bytearray(bw * scale * 2)
    window(x, y, bw * scale, bh * scale)
    for r in range(bh):
        _scale_row(buf, r * bw * 2, bw, scale, line)
        for _ in range(scale):
            spi.write(line)
    cs.value(1)

# Colors (RGB565)
BG = 0x0841; PANEL = 0x18E3; TAB = 0x2945; TAB_ON = 0x05FF; WHITE = 0xFFFF
GREY = 0x8410; DARK = 0x0000; GREEN = 0x07E0; DKGREEN = 0x0320; RED = 0xF800
DKRED = 0x7800; YELLOW = 0xFFE0; ORANGE = 0xFD20; CYAN = 0x07FF; BLUE = 0x001F

# ------------------------------------------------------------------ Touch (XPT2046)
t_cs = Pin(33, Pin.OUT, value=1)
t_irq = Pin(36, Pin.IN)
t_spi = SoftSPI(baudrate=1000000, polarity=0, phase=0, sck=Pin(25), mosi=Pin(32), miso=Pin(39))

def _tread(c):
    t_cs.value(0)
    t_spi.write(bytes([c]))
    v = int.from_bytes(t_spi.read(2), "big") >> 3
    t_cs.value(1)
    return v

def raw_touch():
    """Median of 5 samples, or None if not pressed."""
    if t_irq.value() != 0 and _tread(0xB0) < 100:
        return None
    xs, ys = [], []
    for _ in range(5):
        xs.append(_tread(0xD0)); ys.append(_tread(0x90))
    if _tread(0xB0) < 60 and t_irq.value() != 0:
        return None  # released while sampling
    xs.sort(); ys.sort()
    a, b = xs[2], ys[2]
    if a < 50 or a > 4050 or b < 50 or b > 4050:
        return None
    return a, b

CAL_FILE = "cal.txt"
cal = None  # (swap, ax, bx, ay, by)

def load_cal():
    global cal
    try:
        with open(CAL_FILE) as f:
            p = f.read().split(",")
        cal = (p[0] == "1", float(p[1]), float(p[2]), float(p[3]), float(p[4]))
    except Exception:
        cal = None

def to_screen(r):
    swap, ax, bx, ay, by = cal
    a, b = (r[1], r[0]) if swap else r
    return (max(0, min(319, int(ax * a + bx))), max(0, min(239, int(ay * b + by))))

def wait_tap():
    while raw_touch():
        time.sleep_ms(20)
    while True:
        r = raw_touch()
        if r:
            time.sleep_ms(60)
            r2 = raw_touch() or r
            while raw_touch():
                time.sleep_ms(20)
            return ((r[0] + r2[0]) // 2, (r[1] + r2[1]) // 2)
        time.sleep_ms(15)

def calibrate():
    global cal
    pts = ((20, 20), (300, 20), (20, 220))
    raws = []
    for i, (px, py) in enumerate(pts):
        fill(0, 0, 320, 240, BG)
        text(40, 100, 240, 40, "TOUCH CALIBRATION\nTap the cross (%d/3)" % (i + 1), WHITE, BG)
        fill(px - 10, py - 1, 21, 3, YELLOW); fill(px - 1, py - 10, 3, 21, YELLOW)
        raws.append(wait_tap())
    (a1, b1), (a2, b2), (a3, b3) = raws
    swap = abs(b2 - b1) > abs(a2 - a1)  # which raw channel moves with screen X
    if swap:
        a1, b1, a2, b2, a3, b3 = b1, a1, b2, a2, b3, a3
    ax = 280 / ((a2 - a1) or 1); bx = 20 - ax * a1
    ay = 200 / ((b3 - b1) or 1); by = 20 - ay * b1
    cal = (swap, ax, bx, ay, by)
    with open(CAL_FILE, "w") as f:
        f.write("%d,%f,%f,%f,%f" % (1 if swap else 0, ax, bx, ay, by))
    out("CAL:OK")

# ------------------------------------------------------------------ Links
uart = UART(2, baudrate=115200, tx=Pin(27), rx=Pin(22), timeout=0, rxbuf=2048)
poll = select.poll()
poll.register(sys.stdin.buffer, select.POLLIN)

def out(msg):
    sys.stdout.write(msg + "\n")
    try:
        uart.write(msg + "\n")
    except Exception:
        pass

# ------------------------------------------------------------------ UI state
MODES = (("ALPHA", "alphabets"), ("SPELL", "spelling"), ("W:30", "words"), ("W:2K", "words_2000"), ("SENT", "sentences"))
TAB_W = 64
BUTTONS = (("SPACE", "CMD:SPACE", 0x2A69), ("DEL", "CMD:DEL", 0x52AA),
           ("CLEAR", "CMD:CLEAR", DKRED), ("SPEAK", "CMD:SPEAK", 0x0451))
st = {"m": "", "l": "asl", "o": "en", "u": "", "t": None, "h": None, "c": None, "a": None, "g": None, "i": None, "link": None}
last_rx = 0
try:
    with open("ar_font.bin", "rb") as _af:
        _AR_FONT = _af.read()
except Exception:
    _AR_FONT = None


@micropython.native
def _raster_ar_row(font_bytes, g_list, ng, r, fg_hi, fg_lo, bg_hi, bg_lo, dst):
    di = 0
    for gi in range(ng):
        g = g_list[gi]
        off = g * 32 + r * 2
        b0 = font_bytes[off]
        b1 = font_bytes[off + 1]
        mask = 0x80
        for _ in range(8):
            if b0 & mask:
                dst[di] = fg_hi; dst[di + 1] = fg_lo
            else:
                dst[di] = bg_hi; dst[di + 1] = bg_lo
            di += 2
            mask >>= 1
        mask = 0x80
        for _ in range(8):
            if b1 & mask:
                dst[di] = fg_hi; dst[di + 1] = fg_lo
            else:
                dst[di] = bg_hi; dst[di + 1] = bg_lo
            di += 2
            mask >>= 1


def draw_ar_text(x, y, w, h, hex_str, fg, bg):
    """Render up to 2 lines of pre-shaped 16x16 Arabic glyphs from hex indices ('FF' = newline)."""
    fill(x, y, w, h, bg)
    if not _AR_FONT or not hex_str:
        return
    max_g = len(_AR_FONT) // 32
    lines = []
    for seg in hex_str.split("FF"):
        gs = []
        for i in range(0, len(seg) - 1, 2):
            idx = int(seg[i:i + 2], 16)
            gs.append(idx if idx < max_g else 0)
        if gs:
            lines.append(gs[:19])
    lines = lines[:2]
    if not lines:
        return
    fg_hi, fg_lo = fg >> 8, fg & 0xFF
    bg_hi, bg_lo = bg >> 8, bg & 0xFF
    total_h = len(lines) * 18 - 2
    cy = y + max(0, (h - total_h) // 2)
    for gs in lines:
        ng = len(gs)
        lw = ng * 16
        gx = x + max(4, w - lw - 6)
        row_buf = bytearray(lw * 2)
        window(gx, cy, lw, 16)
        for r in range(16):
            _raster_ar_row(_AR_FONT, gs, ng, r, fg_hi, fg_lo, bg_hi, bg_lo, row_buf)
            spi.write(row_buf)
        cs.value(1)
        cy += 18

def draw_tabs(active):
    for i, (label, key) in enumerate(MODES):
        if key == "words":
            on = active in ("words", "sentences")
        elif key == "words_2000":
            on = active in ("words_2000", "sentences_2000")
        elif key == "sentences":
            on = active in ("sentences", "sentences_2000")
        else:
            on = (key == active)
        text(i * TAB_W + 1, 2, TAB_W - 2, 34, label, DARK if on else WHITE, TAB_ON if on else TAB, 1)

def draw_cam():
    if not st["link"]:
        text(2, 40, 104, 54, "PI\nOFFLINE", DARK, YELLOW)
    elif st["c"]:
        text(2, 40, 104, 54, "CAMERA\nOK", DARK, GREEN)
    else:
        text(2, 40, 104, 54, "CAMERA\nNO SIGNAL", WHITE, RED)

def draw_hand():
    if not st["link"]:
        text(108, 40, 104, 54, "-", GREY, PANEL, 2)
        set_led(1, 1, 0)                       # yellow: Pi not talking
    elif not st["a"]:
        text(108, 40, 104, 54, "AI OFFLINE\nstart laptop", DARK, YELLOW)
        set_led(1, 0, 0)                       # red: laptop AI engine missing
    elif st["c"] and st["h"]:
        text(108, 40, 104, 54, "HAND\nDETECTED", DARK, GREEN)
        set_led(0, 1, 0)                       # green: hand tracked
    else:
        text(108, 40, 104, 54, "NO HAND\nshow hand", WHITE, PANEL)
        set_led(0, 0, 1)                       # blue: ready

def draw_guess():
    g = (st["g"] or "")[:6]
    if len(g) <= 2:
        text(214, 40, 104, 54, g or "-", CYAN, PANEL, 3 if g else 2)
    else:
        text(214, 40, 104, 54, g, CYAN, PANEL, 2 if len(g) <= 4 else 1)

def draw_text():
    if st["o"] == "ar" and st["u"] and _AR_FONT:
        draw_ar_text(2, 98, 316, 64, st["u"], WHITE, DARK)
        return
    t = st["t"] or ""
    t = t[-40:]
    if not t:
        msg = "AR ready..." if st["o"] == "ar" else "Start signing..."
        text(2, 98, 316, 64, msg, GREY, DARK, 2)
        return
    l1, l2 = (t[:20], t[20:]) if len(t) > 20 else (t, "")
    if l2 == "":
        l1, l2 = "", l1
    text(2, 98, 316, 64, l1 + "\n" + l2, WHITE, DARK, 2, "l")

def draw_hint():
    text(2, 164, 206, 30, (st["i"] or "")[:25], 0xBDF7, PANEL, 1)

def draw_lang_toggles():
    sl_on = (st["l"] == "arsl")
    ol_on = (st["o"] == "ar")
    text(210, 164, 53, 30, "IN:AR" if sl_on else "IN:ASL", DARK if sl_on else WHITE, ORANGE if sl_on else TAB, 1)
    text(265, 164, 53, 30, "OUT:AR" if ol_on else "OUT:EN", DARK if ol_on else WHITE, GREEN if ol_on else TAB, 1)

def draw_buttons(pressed=-1):
    for i, (label, _, col) in enumerate(BUTTONS):
        text(i * 80 + 2, 198, 76, 40, label, DARK if i == pressed else WHITE,
             WHITE if i == pressed else col, 1)

def draw_all():
    fill(0, 0, 320, 240, BG)
    draw_tabs(st["m"]); draw_cam(); draw_hand(); draw_guess(); draw_text(); draw_hint(); draw_lang_toggles(); draw_buttons()

# ------------------------------------------------------------------ Message handling
def handle_line(line):
    global last_rx
    line = line.strip()
    if not line:
        return
    last_rx = time.ticks_ms()
    if line == "PING":
        out("PONG"); return
    if line == "REPL":  # maintenance: hand the port back to the REPL for flashing
        micropython.kbd_intr(3)
        raise SystemExit
    if line == "CAL":
        calibrate(); draw_all(); return
    if not line.startswith("S|"):
        return
    if not st["link"]:
        st["link"] = True
        draw_cam(); draw_hand()
    new = {}
    for part in line[2:].split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            new[k] = v
    if "m" in new and new["m"] != st["m"]:
        st["m"] = new["m"]; draw_tabs(st["m"])
    lang_dirty = False
    if "l" in new and new["l"] in ("asl", "arsl") and new["l"] != st["l"]:
        st["l"] = new["l"]; lang_dirty = True
    if "o" in new and new["o"] in ("en", "ar") and new["o"] != st["o"]:
        st["o"] = new["o"]; lang_dirty = True
    if lang_dirty:
        draw_lang_toggles()
    for k in ("c", "h", "a"):
        if k in new:
            v = new[k] == "1"
            if v != st[k]:
                st[k] = v; draw_cam(); draw_hand()
    if "g" in new and new["g"] != st["g"]:
        st["g"] = new["g"]; draw_guess()
    text_dirty = False
    if "u" in new and new["u"] != st["u"]:
        st["u"] = new["u"]; text_dirty = True
    if "t" in new and new["t"] != st["t"]:
        st["t"] = new["t"]; text_dirty = True
    if text_dirty or lang_dirty:
        draw_text()
    if "i" in new and new["i"] != st["i"]:
        st["i"] = new["i"]; draw_hint()

def on_tap(x, y):
    if y < 38:
        i = min(4, x // TAB_W)
        key = MODES[i][1]
        cur = st["m"]
        if key == "words":
            target = "sentences" if cur in ("sentences", "sentences_2000") else "words"
        elif key == "words_2000":
            target = "sentences_2000" if cur in ("sentences", "sentences_2000") else "words_2000"
        elif key == "sentences":
            if cur == "sentences":
                target = "words"
            elif cur == "sentences_2000":
                target = "words_2000"
            elif cur == "words_2000":
                target = "sentences_2000"
            else:
                target = "sentences"
        else:
            target = key
        st["m"] = target
        draw_tabs(target)
        out("CMD:MODE:" + target)
    elif 162 <= y < 196 and x >= 208:
        if x < 264:
            st["l"] = "arsl" if st["l"] == "asl" else "asl"
            if st["l"] == "arsl" and st["o"] == "en":
                st["o"] = "ar"
            st["u"] = ""; st["t"] = ""
            draw_lang_toggles(); draw_text()
            out("CMD:SIGN:" + st["l"])
        else:
            st["o"] = "ar" if st["o"] == "en" else "en"
            draw_lang_toggles(); draw_text()
            out("CMD:LANG:" + st["o"])
    elif y >= 196:
        i = min(3, x // 80)
        draw_buttons(i)
        out(BUTTONS[i][1])
        if BUTTONS[i][1] == "CMD:CLEAR":
            st["t"] = ""; st["u"] = ""; draw_text()
        time.sleep_ms(90)
        draw_buttons()

# ------------------------------------------------------------------ Boot
init_tft()
load_cal()
if cal is None:
    calibrate()
draw_all()
out("READY")

rx_usb = bytearray(); rx_uart = bytearray()
pressed_since = None
press_handled = False
last_touch = 0
last_link_check = 0

while True:
    now = time.ticks_ms()

    # --- USB serial from laptop (non blocking, byte by byte)
    while poll.poll(0):
        b = sys.stdin.buffer.read(1)
        if not b:
            break
        if b == b"\n" or b == b"\r":
            if rx_usb:
                handle_line(rx_usb.decode("utf-8", "ignore")); rx_usb = bytearray()
        elif len(rx_usb) < 200:
            rx_usb.extend(b)

    # --- UART from Raspberry Pi (same protocol)
    n = uart.any()
    if n:
        for c in uart.read(n):
            if c in (10, 13):
                if rx_uart:
                    handle_line(rx_uart.decode("utf-8", "ignore")); rx_uart = bytearray()
            elif len(rx_uart) < 200:
                rx_uart.append(c)

    # --- Touch: edge triggered tap, 3 s hold on text area = recalibrate
    if time.ticks_diff(now, last_touch) >= 25:
        last_touch = now
        r = raw_touch()
        if r:
            x, y = to_screen(r)
            if pressed_since is None:
                pressed_since = now; press_handled = False
                out("TOUCH:%d,%d" % (x, y))
                if not (98 <= y < 196):
                    on_tap(x, y); press_handled = True
            elif not press_handled and time.ticks_diff(now, pressed_since) > 3000:
                press_handled = True
                calibrate(); draw_all()
        else:
            pressed_since = None

    # --- Link watchdog: no message from laptop for 3 s => offline
    if time.ticks_diff(now, last_link_check) > 500:
        last_link_check = now
        alive = time.ticks_diff(now, last_rx) < 3000 and last_rx != 0
        if alive != st["link"]:
            st["link"] = alive; draw_cam(); draw_hand()

    time.sleep_ms(2)
