#!/usr/bin/env python3
"""
preview_endcard.py - Build and preview the endcard image only.
Run: python3 preview_endcard.py
"""
from PIL import Image, ImageDraw, ImageFont

BG_PATH  = "/Users/jackson/Documents/GitHub/Daily/backgrounds/ChatGPT-Image-VoR.png"
OUT_PATH = "/Users/jackson/Documents/GitHub/Daily/templates/endcard-template.png"
W, H = 1920, 1080

# Load forest background and crop out the "Voices of Recovery" top band
bg = Image.open(BG_PATH).convert("RGB")
iw, ih = bg.size
scale = W / iw
bg = bg.resize((int(iw * scale), int(ih * scale)), Image.LANCZOS)
# The dark band with "Voices of Recovery" takes up roughly the top 27% of the image
crop_top = int(bg.height * 0.27)
bg = bg.crop((0, crop_top, W, crop_top + H))

# Semi-transparent dark box over the text area only
overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
ImageDraw.Draw(overlay).rectangle([(80, 100), (1840, 760)], fill=(0, 0, 0, 110))
combined = Image.alpha_composite(bg.convert("RGBA"), overlay).convert("RGB")
draw = ImageDraw.Draw(combined)

def load_font(size):
    for p in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]:
        try: return ImageFont.truetype(p, size)
        except: pass
    return ImageFont.load_default()

def draw_centered(draw, text, font, y, color=(255, 255, 255)):
    bb = draw.textbbox((0, 0), text, font=font)
    x = (W - (bb[2] - bb[0])) // 2
    draw.text((x, y), text, font=font, fill=color)

# "To learn more about SAA, visit:"
draw_centered(draw, "To learn more about SAA, visit:", load_font(54), 170)

# "saa-recovery.org" — large and prominent
draw_centered(draw, "saa-recovery.org", load_font(108), 245, color=(255, 255, 255))

# SAA purpose quote
quote_lines = [
    '"A fellowship of individuals who share their',
    'experience, strength, and hope with each other',
    'so they may overcome their sexual addiction',
    'and help others recover from sexual addiction',
    'or dependency."',
]
y = 410
for line in quote_lines:
    draw_centered(draw, line, load_font(40), y, color=(210, 225, 210))
    y += 54

# Attribution
draw_centered(draw, "Video: Pexels.com   |   Narration: Microsoft Azure TTS",
              load_font(21), 720, color=(210, 225, 210))

combined.save(OUT_PATH)
print(f"✅ Saved: {OUT_PATH}")

import subprocess
subprocess.run(["open", OUT_PATH])
