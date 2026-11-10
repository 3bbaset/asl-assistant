from machine import Pin, SPI, UART
import time, struct, framebuf, select, sys

# Backlight (GPIO 21)
bl = Pin(21, Pin.OUT, value=1)

# RGB LED (active-low on GPIO 4, 16, 17)
led_r = Pin(4, Pin.OUT, value=1)
led_g = Pin(16, Pin.OUT, value=1)
led_b = Pin(17, Pin.OUT, value=0) # Blue on ready

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
cs = Pin(15, Pin.OUT, value=1)
dc = Pin(2, Pin.OUT, value=1)

def cmd(c, data=None):
    cs.value(0); dc.value(0)
    spi_tft.write(bytes([c]))
    if data:
        dc.value(1); spi_tft.write(data)
    cs.value(1)

def init_tft():
    cmd(0x01)
    time.sleep_ms(100)
    cmd(0xCF, b"\x00\xC1\x30")
    cmd(0xED, b"\x64\x03\x12\x81")
    cmd(0xE8, b"\x85\x00\x78")
    cmd(0xCB, b"\x39\x2C\x00\x34\x02")
    cmd(0xF7, b"\x20")
    cmd(0xEA, b"\x00\x00")
    cmd(0xC0, b"\x23")
    cmd(0xC1, b"\x10")
    cmd(0xC5, b"\x3e\x28")
    cmd(0xC7, b"\x86")
    cmd(0x36, b"\x28") # Landscape 320x240 BGR
    cmd(0x3A, b"\x55") # 16-bit RGB565
    cmd(0xB1, b"\x00\x18")
    cmd(0xB6, b"\x08\x82\x27")
    cmd(0x11)
    time.sleep_ms(100)
    cmd(0x29)

def fill_rect(x, y, w, h, color565):
    cmd(0x2A, struct.pack(">HH", x, x + w - 1))
    cmd(0x2B, struct.pack(">HH", y, y + h - 1))
    cs.value(0); dc.value(0)
    spi_tft.write(b"\x2C")
    dc.value(1)
    chunk = struct.pack(">H", color565) * min(w * h, 1024)
    rem = w * h
    while rem > 0:
        n = min(rem, 1024)
        spi_tft.write(chunk[:n*2])
        rem -= n
    cs.value(1)

def draw_text_box(x, y, w, h, text, fg, bg, scale=2):
    bw = w // scale
    bh = h // scale
    buf = bytearray(bw * bh * 2)
    fb = framebuf.FrameBuffer(buf, bw, bh, framebuf.RGB565)
    bg_sw = ((bg & 0xff) << 8) | (bg >> 8)
    fg_sw = ((fg & 0xff) << 8) | (fg >> 8)
    fb.fill(bg_sw)
    tx = max(0, (bw - len(text) * 8) // 2)
    ty = max(0, (bh - 8) // 2)
    fb.text(text, tx, ty, fg_sw)
    if scale == 1:
        cmd(0x2A, struct.pack(">HH", x, x + bw - 1))
        cmd(0x2B, struct.pack(">HH", y, y + bh - 1))
        cs.value(0); dc.value(0)
        spi_tft.write(b"\x2C")
        dc.value(1)
        spi_tft.write(buf)
        cs.value(1)
    else:
        line_buf = bytearray(bw * scale * 2)
        cmd(0x2A, struct.pack(">HH", x, x + bw * scale - 1))
        cmd(0x2B, struct.pack(">HH", y, y + bh * scale - 1))
        cs.value(0); dc.value(0)
        spi_tft.write(b"\x2C")
        dc.value(1)
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
        cs.value(1)

MODES = [("ALPHA", 8, 72), ("SPELL", 85, 72), ("WORDS", 162, 72), ("SENTENCE", 239, 73)]

def render_mode_bar(active):
    for name, x, w in MODES:
        is_active = (name.lower() == active.lower() or (name == "ALPHA" and active == "alphabets") or (name == "SENTENCE" and active == "sentences"))
        fg = 0x0000 if is_active else 0x9CD3
        bg = 0x266B if is_active else 0x1947
        draw_text_box(x, 48, w, 24, name, fg, bg, scale=1)

current_mode = "words"
current_text = ""
current_sub = ""

def update_ui(mode=None, text=None, sub=None, led=None):
    global current_mode, current_text, current_sub
    if led:
        set_led(led)
    if mode and mode != current_mode:
        current_mode = mode
        render_mode_bar(mode)
    if text is not None and text != current_text:
        current_text = text
        fill_rect(10, 80, 300, 52, 0x10E4)
        disp_txt = text if text else "READY"
        draw_text_box(10, 88, 300, 36, disp_txt[:18], 0xFFFF, 0x10E4, scale=2)
    if sub is not None and sub != current_sub:
        current_sub = sub
        fill_rect(10, 134, 300, 48, 0x10E4)
        draw_text_box(10, 142, 300, 20, sub[:36], 0x3FF7, 0x10E4, scale=1)

# Draw base layout
init_tft()
fill_rect(0, 0, 320, 240, 0x08A5)
fill_rect(0, 0, 320, 40, 0x1947)
draw_text_box(10, 8, 300, 24, "ASL COMMUNICATOR V9", 0x3FF7, 0x1947, scale=2)
render_mode_bar("words")
fill_rect(10, 80, 300, 105, 0x10E4)
update_ui(mode="words", text="SYSTEM READY", sub="Pi 2B + Cam + Laptop Linked", led="green")
fill_rect(0, 200, 320, 40, 0x1947)
draw_text_box(10, 210, 300, 20, "Edge Communicator Online", 0x266B, 0x1947, scale=1)

# UART from Raspberry Pi (Pin 22 = RX, Pin 27 = TX)
uart_pi = UART(2, baudrate=115200, tx=Pin(27), rx=Pin(22), timeout=10)
poll_obj = select.poll()
poll_obj.register(sys.stdin, select.POLLIN)

def parse_line(line):
    # e.g.: MODE:WORDS;TEXT:HELLO;SUB:Conf 92%;LED:GREEN
    line = line.strip()
    if not line:
        return
    parts = line.split(";")
    kwargs = {}
    for p in parts:
        if ":" in p:
            k, v = p.split(":", 1)
            k = k.strip().lower()
            v = v.strip()
            if k in ("mode", "text", "sub", "led"):
                kwargs[k] = v
    if kwargs:
        update_ui(**kwargs)

print("ESP32_ASL_V9_ONLINE")
buf_pi = b""
buf_usb = ""

while True:
    # 1. Check Pi UART (IO22/27)
    if uart_pi.any():
        c = uart_pi.read(1)
        if c == b"\n":
            parse_line(buf_pi.decode("utf-8", "ignore"))
            buf_pi = b""
        elif c != b"\r":
            buf_pi += c

    # 2. Check USB UART
    if poll_obj.poll(0):
        ch = sys.stdin.read(1)
        if ch == "\n":
            parse_line(buf_usb)
            buf_usb = ""
        elif ch != "\r":
            buf_usb += ch

    time.sleep_ms(10)
