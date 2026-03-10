#!/usr/bin/env python3
from PIL import Image, ImageDraw, ImageFont

BG_PATH = "/Users/jackson/Documents/GitHub/Daily/backgrounds/ChatGPT-Image-VoR.png"
OUT_PATH = "/Users/jackson/Documents/GitHub/Daily/templates/endcard-template.png"
W, H = 1920, 1080

bg = Image.open(BG_PATH).convert("RGB")
iw, ih = bg.size
scale = W / iw
bg = bg.resize((int(iw*scale), int(ih*scale)), Image.LANCZOS)
bg = bg.crop((0, 0, W, H))

# Subtle dark overlay over the whole image for text legibility
overlay = Image.new("RGBA", (W, H), (0, 0, 0, 80))
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

# "To learn more about SAA, visit:" — upper area
draw_centered(draw, "To learn more about SAA, visit:", load_font(54), 230)

# "saa-recovery.org" — large, bold, prominent
draw_centered(draw, "saa-recovery.org", load_font(108), 305, color=(255, 255, 255))

# SAA purpose quote
quote_lines = [
    '"A fellowship of individuals who share their',
    'experience, strength, and hope with each other',
    'so they may overcome their sexual addiction',
    'and help others recover from sexual addiction',
    'or dependency."',
]
quote_y = 480
for line in quote_lines:
    draw_centered(draw, line, load_font(40), quote_y, color=(220, 230, 220))
    quote_y += 54

# Attribution at bottom
draw_centered(draw, "Video: Pexels.com   |   Narration: Microsoft Azure TTS",
              load_font(28), 900, color=(170, 180, 170))

combined.save(OUT_PATH)
print(f"✅ Saved: {OUT_PATH}")
