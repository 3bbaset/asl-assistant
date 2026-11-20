"""build_ar_font.py - Generate proportional 16px-tall MONO_HMSB Arabic bitmap atlas for ESP32 CYD.

Each glyph occupies 33 bytes:
  - byte 0      : advance width `adv` in pixels (1..16)
  - bytes 1..32 : 16x16 MONO_HMSB rows (2 bytes/row, MSB first, glyph anchored at x=0)

Outputs:
  - ar_font.bin      (N * 33 bytes)
  - ar_font_map.json (dict mapping Unicode char -> glyph index 0..254)
"""
import json
import os
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(size=13):
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


def rasterize_glyph(ch, font):
    """Render char -> 33 bytes ([adv] + 32-byte 16x16 MONO_HMSB bitmap aligned at baseline & x=0)."""
    if ch == " ":
        return bytes([5]) + bytes(32)
    # Fixed baseline at y=24 on 48x48 canvas -> baseline lands at y=11 in 16px box (rows 13..28)
    canvas = Image.new("L", (48, 48), 0)
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 24), ch, fill=255, font=font, anchor="ms")
    bbox = canvas.getbbox()
    out_img = Image.new("L", (16, 16), 0)
    adv = 5
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        bw = max(1, x1 - x0)
        # Preserve consistent vertical baseline band rows [13..29) -> [0..16)
        band_top, band_bot = 13, 29
        if y0 < band_top or y1 > band_bot:
            # Clamp vertical slice cleanly if unusually tall
            shift = 0
            if y0 < band_top:
                shift = band_top - y0
            elif y1 > band_bot:
                shift = band_bot - y1
            slice_top = max(0, band_top - shift)
            crop = canvas.crop((x0, slice_top, x1, slice_top + 16))
        else:
            crop = canvas.crop((x0, band_top, x1, band_bot))

        if bw > 16:
            crop = crop.resize((16, 16), Image.Resampling.BILINEAR)
            bw = 16
        out_img.paste(crop, (0, 0))
        is_arabic_form = (0xFE80 <= ord(ch) <= 0xFEFC) or (0x0621 <= ord(ch) <= 0x064A)
        adv = max(1, min(16, bw if is_arabic_form else min(16, bw + 1)))

    px = out_img.load()
    rows = bytearray(33)
    rows[0] = adv
    for y in range(16):
        hi = 0
        lo = 0
        for x in range(8):
            if px[x, y] >= 96:
                hi |= (0x80 >> x)
            if px[8 + x, y] >= 96:
                lo |= (0x80 >> x)
        rows[1 + y * 2] = hi
        rows[1 + y * 2 + 1] = lo
    return bytes(rows)


def main():
    font = load_font(size=13)
    chars = build_charset()
    charmap = {}
    blob = bytearray()
    for idx, ch in enumerate(chars):
        charmap[ch] = idx
        blob.extend(rasterize_glyph(ch, font))
    with open("ar_font.bin", "wb") as f:
        f.write(blob)
    with open("ar_font_map.json", "w", encoding="utf-8") as f:
        json.dump(charmap, f, ensure_ascii=False, indent=2)
    print(f"Generated ar_font.bin ({len(blob)} bytes, {len(chars)} glyphs x 33B) and ar_font_map.json")


if __name__ == "__main__":
    main()
