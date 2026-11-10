"""esp32_firmware.py - Robust, Crash-Proof ASL Communicator V9 Firmware

Features:
  - 60x45 RGB565 Live Camera Viewport (scaled 2x to 120x90 on screen)
  - Real-time ASL Word & Sentence Captions
  - Interactive Resistive Touch Buttons (Mode switching & Clear)
  - RGB Status LED
  - Non-blocking, watchdog-safe FreeRTOS loop
"""
from machine import Pin, SPI, SoftSPI, UART
import time, struct, framebuf, sys

# ---------------- Hardware Pins (ESP32-2432S028R) ----------------
bl = Pin(21, Pin.OUT, value=1)
led_r = Pin(4, Pin.OUT, value=1)
led_g = Pin(16, Pin.OUT, value=1)
led_b = Pin(17, Pin.OUT, value=0) # Blue on boot

def set_led(color):
    if color == "green":
        led_r.value(1); led_g.value(0); led_b.value(1)
    elif color == "yellow":
        led_r.value(0); led_g.value(0); led_b.value(1)
    elif color == "red":
        led_r.value(0); led_g.value(1); led_b.value(1)
    elif color == "blue":
        led_r.value(1); led_g.value(1); led_b.value(0)
    elif color == "cyan":
        led_r.value(1); led_g.value(0); led_b.value(0)
    else:
        led_r.value(1); led_g.value(1); led_b.value(1)

# ILI9341 Display on HSPI (SPI 1)
spi_tft = SPI(1, baudrate=40000000, polarity=0, phase=0, sck=Pin(14), mosi=Pin(13), miso=Pin(12))
cs_tft = Pin(15, Pin.OUT, value=1)
dc_tft = Pin(2, Pin.OUT, value=1)

def tft_cmd(c, data=None):
    cs_tft.value(0); dc_tft.value(0)
    spi_tft.write(bytes([c]))
    if data:
        dc_tft.value(1); spi_tft.write(data)
    cs_tft.value(1)

def init_tft():
    tft_cmd(0x01)
    time.sleep_ms(100)
    tft_cmd(0xCF, b"\x00\xC1\x30")
    tft_cmd(0xED, b"\x64\x03\x12\x81")
    tft_cmd(0xE8, b"\x85\x00\x78")
    tft_cmd(0xCB, b"\x39\x2C\x00\x34\x02")
    tft_cmd(0xF7, b"\x20")
    tft_cmd(0xEA, b"\x00\x00")
    tft_cmd(0xC0, b"\x23")
    tft_cmd(0xC1, b"\x10")
    tft_cmd(0xC5, b"\x3e\x28")
    tft_cmd(0xC7, b"\x86")
    tft_cmd(0x36, b"\x28") # Landscape 320x240 BGR
    tft_cmd(0x3A, b"\x55") # 16-bit RGB565
    tft_cmd(0xB1, b"\x00\x18")
    tft_cmd(0xB6, b"\x08\x82\x27")
    tft_cmd(0x11)
    time.sleep_ms(100)
    tft_cmd(0x29)

def fill_rect(x, y, w, h, color565):
    tft_cmd(0x2A, struct.pack(">HH", x, x + w - 1))
    tft_cmd(0x2B, struct.pack(">HH", y, y + h - 1))
    cs_tft.value(0); dc_tft.value(0)
    spi_tft.write(b"\x2C")
    dc_tft.value(1)
    chunk = struct.pack(">H", color565) * min(w * h, 256)
    rem = w * h
    while rem > 0:
        n = min(rem, 256)
        spi_tft.write(chunk[:n*2])
        rem -= n
    cs_tft.value(1)

def draw_text_box(x, y, w, h, text, fg, bg, scale=1):
    bw = max(1, w // scale)
    bh = max(1, h // scale)
    buf = bytearray(bw * bh * 2)
    fb = framebuf.FrameBuffer(buf, bw, bh, framebuf.RGB565)
    bg_sw = ((bg & 0xff) << 8) | (bg >> 8)
    fg_sw = ((fg & 0xff) << 8) | (fg >> 8)
    fb.fill(bg_sw)
    tx = max(0, (bw - len(text) * 8) // 2)
    ty = max(0, (bh - 8) // 2)
    fb.text(text, tx, ty, fg_sw)
    if scale == 1:
        tft_cmd(0x2A, struct.pack(">HH", x, x + bw - 1))
        tft_cmd(0x2B, struct.pack(">HH", y, y + bh - 1))
        cs_tft.value(0); dc_tft.value(0)
        spi_tft.write(b"\x2C")
        dc_tft.value(1)
        spi_tft.write(buf)
        cs_tft.value(1)
    else:
        line_buf = bytearray(bw * scale * 2)
        tft_cmd(0x2A, struct.pack(">HH", x, x + bw * scale - 1))
        tft_cmd(0x2B, struct.pack(">HH", y, y + bh * scale - 1))
        cs_tft.value(0); dc_tft.value(0)
        spi_tft.write(b"\x2C")
        dc_tft.value(1)
        for row in range(bh):
            row_off = row * bw * 2
            idx = 0
            for col in range(bw):
                b0 = buf[row_off + col*2]
                b1 = buf[row_off + col*2 + 1]
                for _ in range(scale):
                    line_buf[idx] = b0
                    line_buf[idx+1] = b1
                    idx += 2
            for _ in range(scale):
                spi_tft.write(line_buf)
        cs_tft.value(1)

# ---------------- XPT2046 Resistive Touch ----------------
sck_touch = Pin(25, Pin.OUT)
mosi_touch = Pin(32, Pin.OUT)
miso_touch = Pin(39, Pin.IN)
cs_touch = Pin(33, Pin.OUT, value=1)
irq_touch = Pin(36, Pin.IN)
spi_touch = SoftSPI(baudrate=1000000, polarity=0, phase=0, sck=sck_touch, mosi=mosi_touch, miso=miso_touch)

def read_touch():
    if irq_touch.value() == 0:
        cs_touch.value(0)
        spi_touch.write(b"\x90")
        rx = int.from_bytes(spi_touch.read(2), "big") >> 3
        spi_touch.write(b"\xd0")
        ry = int.from_bytes(spi_touch.read(2), "big") >> 3
        cs_touch.value(1)
        # 320x240 calibration
        x = max(0, min(319, (rx - 300) * 320 // 3500))
        y = max(0, min(239, (ry - 300) * 240 // 3500))
        return x, y
    return None

# ---------------- UI Layout & State ----------------
# 60x45 native image scaled 2x to 120x90 in the screen center
CAM_W = 120
CAM_H = 90
CAM_X = 100
CAM_Y = 38

MODES = [("ALPHA", 4, 74), ("SPELL", 82, 74), ("WORDS", 160, 74), ("SENT", 238, 78)]
current_mode = "words"
current_text = ""
current_sub = ""

def render_mode_bar(active):
    for name, x, w in MODES:
        is_active = (name.lower() == active.lower() or
                     (name == "ALPHA" and active == "alphabets") or
                     (name == "SENT" and active in ("sentence", "sentences")) or
                     (name == "WORDS" and active == "words") or
                     (name == "SPELL" and active == "spelling"))
        fg = 0x0000 if is_active else 0x9CD3
        bg = 0x266B if is_active else 0x1947
        draw_text_box(x, 4, w, 28, name, fg, bg, scale=1)

# Pre-allocated line buffer for scaling 60 pixels to 120 pixels (240 bytes)
scale_line = bytearray(120 * 2)

def blit_cam_60x45(raw_5400):
    """Takes 60x45 RGB565 (5,400 bytes) and blits scaled 2x to 120x90 on screen."""
    tft_cmd(0x2A, struct.pack(">HH", CAM_X, CAM_X + CAM_W - 1))
    tft_cmd(0x2B, struct.pack(">HH", CAM_Y, CAM_Y + CAM_H - 1))
    cs_tft.value(0); dc_tft.value(0)
    spi_tft.write(b"\x2C")
    dc_tft.value(1)
    for row in range(45):
        row_offset = row * 60 * 2
        idx = 0
        for col in range(60):
            p0 = raw_5400[row_offset + col*2]
            p1 = raw_5400[row_offset + col*2 + 1]
            scale_line[idx] = p0; scale_line[idx+1] = p1
            scale_line[idx+2] = p0; scale_line[idx+3] = p1
            idx += 4
        # Write line twice for 2x vertical scale
        spi_tft.write(scale_line)
        spi_tft.write(scale_line)
    cs_tft.value(1)

def update_caption(text=None, sub=None):
    global current_text, current_sub
    if text is not None and text != current_text:
        current_text = text
        fill_rect(10, 134, 235, 48, 0x10E4)
        disp = text if text else "READY"
        draw_text_box(14, 140, 227, 36, disp[:16], 0xFFFF, 0x10E4, scale=2)
    if sub is not None and sub != current_sub:
        current_sub = sub
        fill_rect(10, 184, 235, 24, 0x1947)
        draw_text_box(14, 186, 227, 20, sub[:28], 0x3FF7, 0x1947, scale=1)

def init_screen():
    init_tft()
    fill_rect(0, 0, 320, 240, 0x08A5)
    fill_rect(0, 0, 320, 36, 0x0F17)
    render_mode_bar("words")

    # Border for camera preview
    fill_rect(CAM_X - 2, CAM_Y - 2, CAM_W + 4, CAM_H + 4, 0x266B)
    fill_rect(CAM_X, CAM_Y, CAM_W, CAM_H, 0x0000)

    # Caption card
    fill_rect(10, 134, 235, 74, 0x10E4)
    update_caption("READY TO SIGN", "Camera & AI Linked")

    # Clear button [CLR]
    fill_rect(252, 134, 58, 74, 0x8000)
    draw_text_box(252, 150, 58, 40, "CLR", 0xFFFF, 0x8000, scale=2)

    # Footer
    fill_rect(0, 212, 320, 28, 0x0F17)
    draw_text_box(10, 216, 300, 20, "V9 PORTABLE COMMUNICATOR", 0x266B, 0x0F17, scale=1)

init_screen()

# UART from Pi
uart_pi = UART(2, baudrate=115200, tx=Pin(27), rx=Pin(22), timeout=2)

def send_to_host(msg):
    sys.stdout.write(msg + "\n")
    try:
        uart_pi.write((msg + "\n").encode())
    except Exception:
        pass

last_touch_time = 0

def handle_touch(x, y):
    global last_touch_time, current_mode
    now = time.ticks_ms()
    if time.ticks_diff(now, last_touch_time) < 400:
        return
    last_touch_time = now

    if y <= 40:
        if x < 80:
            current_mode = "alphabets"
            render_mode_bar("alphabets")
            send_to_host("CMD:MODE:alphabets")
        elif x < 160:
            current_mode = "spelling"
            render_mode_bar("spelling")
            send_to_host("CMD:MODE:spelling")
        elif x < 240:
            current_mode = "words"
            render_mode_bar("words")
            send_to_host("CMD:MODE:words")
        else:
            current_mode = "sentences"
            render_mode_bar("sentences")
            send_to_host("CMD:MODE:sentences")
    elif x >= 245 and y >= 130:
        update_caption("", "Buffer Cleared")
        send_to_host("CMD:CLEAR")

def handle_cmd(line):
    line = line.strip()
    if not line: return
    sys.stdout.write("ACK: " + line + "\n")
    for part in line.split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            k = k.strip().lower()
            v = v.strip()
            if k == "mode":
                render_mode_bar(v)
            elif k == "text":
                update_caption(text=v)
            elif k == "sub":
                update_caption(sub=v)
            elif k == "led":
                set_led(v.lower())

import select

print("ESP32_ASL_V9_ONLINE")

# Pre-allocated image buffer (5,400 bytes = 60x45 RGB565)
img_buffer = bytearray(5400)
line_buffer = bytearray(128)
line_len = 0

poll_obj = select.poll()
poll_obj.register(sys.stdin, select.POLLIN)

while True:
    # 1. Check Touch Screen
    pt = read_touch()
    if pt:
        handle_touch(pt[0], pt[1])

    # 2. Check incoming USB (sys.stdin)
    if poll_obj.poll(0):
        c = sys.stdin.read(1)
        if c:
            if c == "\n":
                line_str = bytes(line_buffer[:line_len]).decode("utf-8", "ignore").strip()
                line_len = 0
                if line_str.startswith("IMG:5400"):
                    bytes_read = 0
                    t_start = time.ticks_ms()
                    while bytes_read < 5400 and time.ticks_diff(time.ticks_ms(), t_start) < 200:
                        chunk = sys.stdin.buffer.read(5400 - bytes_read)
                        if chunk:
                            img_buffer[bytes_read:bytes_read+len(chunk)] = chunk
                            bytes_read += len(chunk)
                        else:
                            time.sleep_ms(2)
                    if bytes_read == 5400:
                        blit_cam_60x45(img_buffer)
                else:
                    handle_cmd(line_str)
            elif c != "\r" and line_len < 120:
                line_buffer[line_len] = ord(c)
                line_len += 1

    # 3. Check incoming Pi UART (IO22)
    if uart_pi.any():
        p_line = uart_pi.readline()
        if p_line:
            handle_cmd(p_line.decode("utf-8", "ignore"))

    time.sleep_ms(2)
