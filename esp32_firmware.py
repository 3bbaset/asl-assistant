"""esp32_firmware.py - Production-Grade ASL Communicator V9 Firmware

Features:
  - 32x24 RGB565 Live Camera Viewport (scaled 4x to 128x96 via native emitter, ~38ms render)
  - Row-packet streaming with read_exact and kbd_intr(-1)
  - Interactive Touch Screen Mode Tabs (ALPHA, SPELL, WORDS, SENT) & Clear (CLR)
  - High-visibility ASL Word & Subtitle Captions
  - RGB Status LED (Green = Hand Tracked, Blue = Ready, Yellow = Connecting)
  - CMD:REPL maintenance command for effortless firmware flashing
"""
import micropython
micropython.kbd_intr(-1) # Disable Ctrl-C capture so binary bytes never interrupt!

from machine import Pin, SPI, SoftSPI, UART
import time, struct, framebuf, sys, select

# ---------------- Hardware Pins (ESP32-2432S028R) ----------------
bl = Pin(21, Pin.OUT, value=1)
led_r = Pin(4, Pin.OUT, value=1)
led_g = Pin(16, Pin.OUT, value=1)
led_b = Pin(17, Pin.OUT, value=0) # Blue on boot

def set_led(color):
    c = color.lower()
    if c == "green":
        led_r.value(1); led_g.value(0); led_b.value(1)
    elif c == "yellow":
        led_r.value(0); led_g.value(0); led_b.value(1)
    elif c == "red":
        led_r.value(0); led_g.value(1); led_b.value(1)
    elif c == "blue":
        led_r.value(1); led_g.value(1); led_b.value(0)
    elif c == "cyan":
        led_r.value(1); led_g.value(0); led_b.value(0)
    elif c == "off":
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

def read_channel(cmd):
    cs_touch.value(0)
    spi_touch.write(bytes([cmd]))
    v = int.from_bytes(spi_touch.read(2), "big") >> 3
    cs_touch.value(1)
    return v

def read_touch():
    # Only read if IRQ is low (active touch) or quick Z1 check
    if irq_touch.value() == 0:
        is_pressed = True
    else:
        z1 = read_channel(0xB0)
        is_pressed = (z1 > 60)
    
    if is_pressed:
        raw_y = read_channel(0x90) # Y-channel
        raw_x = read_channel(0xD0) # X-channel
        if raw_x == 0 and raw_y == 4095:
            return None
        # Map to landscape 320x240
        x = max(0, min(319, (raw_y - 250) * 320 // 3550))
        y = max(0, min(239, (raw_x - 300) * 240 // 3500))
        return x, y
    return None

# ---------------- UI Layout & State ----------------
CAM_W = 128
CAM_H = 96
CAM_X = 96
CAM_Y = 36

MODES = [("ALPHA", 4, 74), ("SPELL", 82, 74), ("WORDS", 160, 74), ("SENT", 238, 78)]
current_mode = "words"
current_text = ""
current_sub = ""

def render_mode_bar(active):
    global current_mode
    current_mode = active.lower()
    for name, x, w in MODES:
        is_active = (name.lower() == current_mode or
                     (name == "ALPHA" and current_mode.startswith("alpha")) or
                     (name == "SENT" and current_mode.startswith("sent")) or
                     (name == "WORDS" and current_mode == "words") or
                     (name == "SPELL" and current_mode == "spelling"))
        fg = 0x0000 if is_active else 0x9CD3
        bg = 0x266B if is_active else 0x1947
        draw_text_box(x, 4, w, 28, name, fg, bg, scale=1)

# Pre-allocated line buffer for scaling 32 pixels to 128 pixels (256 bytes)
scale_line = bytearray(128 * 2)

@micropython.native
def scale_row_32_to_128(src, dst):
    d_idx = 0
    for col in range(32):
        b0 = src[col * 2]
        b1 = src[col * 2 + 1]
        dst[d_idx] = b0; dst[d_idx+1] = b1
        dst[d_idx+2] = b0; dst[d_idx+3] = b1
        dst[d_idx+4] = b0; dst[d_idx+5] = b1
        dst[d_idx+6] = b0; dst[d_idx+7] = b1
        d_idx += 8

def blit_single_row(row_idx, row_bytes_64):
    """Blits one 32-pixel row scaled 4x vertically and horizontally (128x4 pixels)"""
    y_start = CAM_Y + row_idx * 4
    scale_row_32_to_128(row_bytes_64, scale_line)
    tft_cmd(0x2A, struct.pack(">HH", CAM_X, CAM_X + CAM_W - 1))
    tft_cmd(0x2B, struct.pack(">HH", y_start, y_start + 3))
    cs_tft.value(0); dc_tft.value(0)
    spi_tft.write(b"\x2C")
    dc_tft.value(1)
    spi_tft.write(scale_line)
    spi_tft.write(scale_line)
    spi_tft.write(scale_line)
    spi_tft.write(scale_line)
    cs_tft.value(1)

def update_caption(text=None, sub=None):
    global current_text, current_sub
    if text is not None and text != current_text:
        current_text = text
        fill_rect(10, 136, 235, 46, 0x10E4)
        disp = text if text else "READY"
        draw_text_box(14, 140, 227, 36, disp[:16], 0xFFFF, 0x10E4, scale=2)
    if sub is not None and sub != current_sub:
        current_sub = sub
        fill_rect(10, 184, 235, 24, 0x1947)
        draw_text_box(14, 186, 227, 20, sub[:28], 0x3FF7, 0x1947, scale=1)

def init_screen():
    init_tft()
    fill_rect(0, 0, 320, 240, 0x08A5)
    fill_rect(0, 0, 320, 34, 0x0F17)
    render_mode_bar("words")

    # Border for camera preview
    fill_rect(CAM_X - 2, CAM_Y - 2, CAM_W + 4, CAM_H + 4, 0x266B)
    fill_rect(CAM_X, CAM_Y, CAM_W, CAM_H, 0x0000)

    # Caption card
    fill_rect(10, 136, 235, 72, 0x10E4)
    update_caption("READY TO SIGN", "Camera & AI Linked")

    # Clear button [CLR]
    fill_rect(252, 136, 58, 72, 0x8000)
    draw_text_box(252, 152, 58, 40, "CLR", 0xFFFF, 0x8000, scale=2)

    # Footer
    fill_rect(0, 214, 320, 26, 0x0F17)
    draw_text_box(10, 218, 300, 18, "V9 PORTABLE COMMUNICATOR", 0x266B, 0x0F17, scale=1)

init_screen()

# UART to Raspberry Pi
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
    if time.ticks_diff(now, last_touch_time) < 300:
        return
    last_touch_time = now

    if y <= 38:
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
    elif x >= 240 and y >= 130:
        update_caption("", "Buffer Cleared")
        send_to_host("CMD:CLEAR")

def handle_cmd(line):
    line = line.strip()
    if not line: return
    if line == "CMD:REPL":
        micropython.kbd_intr(3)
        sys.exit()
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
                set_led(v)

print("ESP32_ASL_V9_ONLINE")

poll_obj = select.poll()
poll_obj.register(sys.stdin.buffer, select.POLLIN)

line_buf = bytearray(128)
line_len = 0
last_touch_check = 0

def read_exact(n, timeout_ms=35):
    buf = bytearray(n)
    idx = 0
    t0 = time.ticks_ms()
    while idx < n and time.ticks_diff(time.ticks_ms(), t0) < timeout_ms:
        if poll_obj.poll(3):
            chunk = sys.stdin.buffer.read(min(n - idx, 64))
            if chunk:
                buf[idx:idx+len(chunk)] = chunk
                idx += len(chunk)
    return buf if idx == n else None

while True:
    now = time.ticks_ms()

    # 1. Check Touch Screen every 40ms
    if time.ticks_diff(now, last_touch_check) >= 40:
        last_touch_check = now
        pt = read_touch()
        if pt:
            handle_touch(pt[0], pt[1])

    # 2. Check incoming USB (sys.stdin.buffer)
    if poll_obj.poll(0):
        b = sys.stdin.buffer.read(1)
        if b:
            if b == b"L":
                # Row packet: 1 byte row_idx + 64 bytes pixel data = 65 bytes
                pkt = read_exact(65, timeout_ms=35)
                if pkt is not None:
                    row_id = pkt[0]
                    if row_id < 24:
                        blit_single_row(row_id, pkt[1:])
                        if row_id == 23:
                            sys.stdout.write("IMG_OK\n")
            elif b in (b"\n", b"\r"):
                if line_len > 0:
                    cmd_str = bytes(line_buf[:line_len]).decode("utf-8", "ignore").strip()
                    line_len = 0
                    if cmd_str.startswith("SET:"):
                        handle_cmd(cmd_str[4:])
                    else:
                        handle_cmd(cmd_str)
            else:
                if line_len < 120:
                    line_buf[line_len] = b[0]
                    line_len += 1

    # 3. Check incoming Pi UART (IO22)
    if uart_pi.any():
        p_line = uart_pi.readline()
        if p_line:
            cmd = p_line.decode("utf-8", "ignore").strip()
            if cmd.startswith("SET:"):
                handle_cmd(cmd[4:])
            else:
                handle_cmd(cmd)

    time.sleep_ms(1)
