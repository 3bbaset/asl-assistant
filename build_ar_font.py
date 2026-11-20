"""build_ar_font.py - Generate 16x16 MONO_HMSB Arabic bitmap atlas for ESP32 CYD.

Outputs:
  - ar_font.bin      (N * 32 bytes; each 16x16 glyph stored MSB-first row-major)
  - ar_font_map.json (dict mapping Unicode char -> glyph index 0..254)
"""
import json
import os
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\seguiemj.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(size=14):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    raise FileNotFoundError("No TrueType font with Arabic glyph support found.")


def build_charset():
    chars = [" "]  # index 0 = space
    # Standard digits & punctuation (helpful when mixed in translated Arabic sentences)
    for ch in "0123456789.,!?:-+/()،؟؛":
        if ch not in chars:
            chars.append(ch)
    # Arabic-Indic digits
    for cp in range(0x0660, 0x066A):
        chars.append(chr(cp))
    # Basic Arabic letters (fallback if an unshaped codepoint appears)
    for cp in list(range(0x0621, 0x063B)) + list(range(0x0641, 0x064B)):
        ch = chr(cp)
        if ch not in chars:
            chars.append(ch)
    # Arabic Presentation Forms-B (all isolated/final/initial/medial glyphs from arabic_reshaper)
    for cp in range(0xFE80, 0xFEFD):
        ch = chr(cp)
        if ch not in chars:
            chars.append(ch)
    # ASCII uppercase fallback (if transliterated / proper noun slips in)
    for cp in range(ord("A"), ord("Z") + 1):
        ch = chr(cp)
        if ch not in chars:
            chars.append(ch)
    assert len(chars) <= 255, f"Too many glyphs ({len(chars)}); max 255 for 1-byte indexing."
    return chars


def rasterize_glyph(ch, font):
    """Render single character into 32-byte 16x16 MONO_HMSB bitmap."""
    if ch == " ":
        return bytes(32)
    canvas = Image.new("L", (40, 40), 0)
    draw = ImageDraw.Draw(canvas)
    # Anchor baseline consistently so letters/descenders align nicely
    draw.text((20, 22), ch, fill=255, font=font, anchor="ms")
    bbox = canvas.getbbox()
    out_img = Image.new("L", (16, 16), 0)
    if bbox is not None:
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        crop = canvas.crop(bbox)
        # Scale down only if glyph exceeds 15px in either dimension; keep baseline y=11 relative
        if bw > 15 or bh > 15:
            scale = min(15.0 / bw, 15.0 / bh)
            nw = max(1, int(round(bw * scale)))
            nh = max(1, int(round(bh * scale)))
            crop = crop.resize((nw, nh), Image.Resampling.BILINEAR)
            ox = (16 - nw) // 2
            oy = (16 - nh) // 2
        else:
            ox = (16 - bw) // 2
            # Preserve vertical alignment relative to baseline (22 -> 12 in 16px box)
            oy = max(0, min(16 - bh, bbox[1] - 22 + 12))
        out_img.paste(crop, (ox, oy))

    px = out_img.load()
    rows = bytearray(32)
    for y in range(16):
        hi = 0
        lo = 0
        for x in range(8):
            if px[x, y] >= 96:
                hi |= (0x80 >> x)
            if px[8 + x, y] >= 96:
                lo |= (0x80 >> x)
        rows[y * 2] = hi
        rows[y * 2 + 1] = lo
    return bytes(rows)


def main():
    font = load_font(size=14)
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
    print(f"Generated ar_font.bin ({len(blob)} bytes, {len(chars)} glyphs) and ar_font_map.json")


if __name__ == "__main__":
    main()
