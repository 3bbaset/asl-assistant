"""build_ar_font.py - Generate crisp proportional 20px-tall Bold MONO_HMSB Arabic bitmap atlas for ESP32 CYD.

Each glyph occupies 61 bytes:
  - byte 0      : advance width `adv` in pixels (1..24)
  - bytes 1..60 : 20x24 MONO_HMSB rows (3 bytes/row, MSB first, glyph anchored at x=0)

Outputs:
  - ar_font.bin      (N * 61 bytes)
  - ar_font_map.json (dict mapping Unicode char -> glyph index 0..253)
"""
import json
import os
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\tahomabd.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def load_font(size=17):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    raise FileNotFoundError("No TrueType font with Arabic glyph support found.")


def build_charset():
    chars = [" "]  # index 0 = space
    for ch in "0123456789%.,!?:-+/()،؟؛":
        if ch not in chars:
            chars.append(ch)
    for cp in range(0x0660, 0x066A):
        chars.append(chr(cp))
    for cp in list(range(0x0621, 0x063B)) + list(range(0x0641, 0x064B)):
        ch = chr(cp)
        if ch not in chars:
            chars.append(ch)
    for cp in range(0xFE80, 0xFEFD):
        ch = chr(cp)
        if ch not in chars:
            chars.append(ch)
    for cp in range(ord("A"), ord("Z") + 1):
        ch = chr(cp)
        if ch not in chars:
            chars.append(ch)
    assert len(chars) <= 254, f"Too many glyphs ({len(chars)}); max 254 (0..253, 255=newline)."
    return chars


def _measure_band(font, baseline_y=32):
    """Find optimal 20px vertical band centered around Arabic ascenders + descenders."""
    probes = ["\uFEE3", "\uFE83", "\uFEAA", "\uFEF2", "\uFEFB", "\uFEB5", "0", "%"]
    ymin, ymax = 999, -999
    for ch in probes:
        c = Image.new("L", (64, 64), 0)
        ImageDraw.Draw(c).text((32, baseline_y), ch, fill=255, font=font, anchor="ms")
        bb = c.getbbox()
        if bb:
            ymin = min(ymin, bb[1])
            ymax = max(ymax, bb[3])
    span = ymax - ymin
    band_top = ymin - max(0, (20 - span) // 2)
    return band_top, band_top + 20


def rasterize_glyph(ch, font, band_top, band_bot, baseline_y=32):
    """Render char -> 61 bytes ([adv] + 60-byte 20x24 MONO_HMSB bitmap aligned at fixed baseline)."""
    if ch == " ":
        return bytes([6]) + bytes(60)
    canvas = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(canvas).text((32, baseline_y), ch, fill=255, font=font, anchor="ms")
    bbox = canvas.getbbox()
    out_img = Image.new("L", (24, 20), 0)
    adv = 6
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        bw = max(1, x1 - x0)
        # Keep baseline aligned across all glyphs; only nudge slice vertically if a tall glyph exceeds 20px
        st = band_top
        if y0 < st:
            st = max(0, min(y0, band_top))
        elif y1 > st + 20:
            st = min(44, y1 - 20)
        crop = canvas.crop((x0, st, x1, st + 20))
        if bw > 24:
            crop = crop.resize((24, 20), Image.Resampling.BILINEAR)
            bw = 24
        out_img.paste(crop, (0, 0))
        is_arabic_form = (0xFE80 <= ord(ch) <= 0xFEFC) or (0x0621 <= ord(ch) <= 0x064A)
        adv = max(2, min(24, bw if is_arabic_form else min(24, bw + 1)))

    px = out_img.load()
    rows = bytearray(61)
    rows[0] = adv
    for y in range(20):
        b0 = b1 = b2 = 0
        for x in range(8):
            if px[x, y] >= 100:
                b0 |= (0x80 >> x)
            if px[8 + x, y] >= 100:
                b1 |= (0x80 >> x)
            if px[16 + x, y] >= 100:
                b2 |= (0x80 >> x)
        off = 1 + y * 3
        rows[off] = b0
        rows[off + 1] = b1
        rows[off + 2] = b2
    return bytes(rows)


def main():
    font = load_font(size=16)
    band_top, band_bot = _measure_band(font, baseline_y=32)
    chars = build_charset()
    charmap = {}
    blob = bytearray()
    for idx, ch in enumerate(chars):
        charmap[ch] = idx
        blob.extend(rasterize_glyph(ch, font, band_top, band_bot, baseline_y=32))
    with open("ar_font.bin", "wb") as f:
        f.write(blob)
    with open("ar_font_map.json", "w", encoding="utf-8") as f:
        json.dump(charmap, f, ensure_ascii=False, indent=2)
    print(f"Generated ar_font.bin ({len(blob)} bytes, {len(chars)} glyphs x 61B, band={band_top}..{band_bot})")


if __name__ == "__main__":
    main()
