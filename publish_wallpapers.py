"""
publish_wallpapers.py

Version: 2026-11-21.01

Renders today's Voices of Recovery reading as two wallpaper images and
(optionally) uploads them to the public R2 bucket behind daily.step12saa.org:

    today-desktop.png   3840 x 2160  (panel centred)
    today-phone.png     1290 x 2796  (text below the Lock Screen clock,
                                      above the flashlight/camera buttons)

Text comes from meditations/MM-DD.txt, written earlier in the run by
daily_poster.py. The background is drawn here (a soft teal gradient with
gentle curves), so no third-party wallpaper is redistributed.

Usage:
    python publish_wallpapers.py [MM-DD] [--upload] [--out DIR]

Without --upload it only writes the PNGs to --out (default output/wallpapers)
— safe for testing. With --upload it also puts both files in R2 with a short
cache lifetime so phones and Macs pick up the new day promptly. Needs env:
    R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT
Optional env: R2_BUCKET (default steptwelve-daily)
"""

from __future__ import annotations

import math
import os
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent
MED_DIR = ROOT / "meditations"

QUOTES = ('"', "“", "”", "‘", "’")
INK = (28, 52, 56)
SOFT = (60, 95, 100)

FONT_PATHS = {
    "regular": [
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    ],
    "bold": [
        "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    ],
    "italic": [
        "/System/Library/Fonts/Supplemental/Georgia Italic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
    ],
}

# Layout boxes as fractions of the image: (left, top, width, max_bottom)
LAYOUTS = {
    "desktop": {"size": (3840, 2160), "box": (0.33, 0.06, 0.34, 0.94), "pad": 70,
                "max_font": 48, "min_font": 22, "radius": 46},
    "phone":   {"size": (1290, 2796), "box": (0.06, 0.34, 0.88, 0.86), "pad": 56,
                "max_font": 46, "min_font": 26, "radius": 50},
}


def font(kind: str, size: int):
    for p in FONT_PATHS[kind]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# ── Text ──────────────────────────────────────────────────────────────────────

def split_reading(full: str):
    """-> (title, [(paragraph, style)]); opening quote and closing line italic."""
    parts = [" ".join(p.split()) for p in full.split("\n\n") if p.strip()]
    parts = [p for p in parts if any(c.isalnum() for c in p)]  # drop lone "–"
    title, rest = parts[0], parts[1:]
    out = []
    for i, p in enumerate(rest):
        italic = (i == 0 and p.startswith(QUOTES)) or (i == len(rest) - 1 and len(rest) > 2)
        out.append((p, "italic" if italic else "regular"))
    return title, out


def wrap(draw, text, fnt, width):
    lines, line = [], ""
    for w in text.split():
        trial = f"{line} {w}".strip()
        if draw.textlength(trial, font=fnt) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines


# ── Drawing ───────────────────────────────────────────────────────────────────

def background(size):
    """Soft teal gradient with two broad curved bands — drawn, not borrowed."""
    W, H = size
    small = (W // 8, H // 8)
    img = Image.new("RGB", small)
    px = img.load()
    top, bottom = (214, 246, 244), (150, 214, 206)
    for y in range(small[1]):
        for x in range(small[0]):
            t = y / small[1]
            # two gentle waves to give depth
            w1 = 0.5 + 0.5 * math.sin((x / small[0]) * math.pi * 1.2 + t * 2.4)
            w2 = 0.5 + 0.5 * math.cos((x / small[0]) * math.pi * 0.7 - t * 3.1)
            k = min(1.0, max(0.0, 0.55 * t + 0.25 * w1 + 0.2 * w2))
            px[x, y] = tuple(int(top[i] * (1 - k) + bottom[i] * k) for i in range(3))
    img = img.filter(ImageFilter.GaussianBlur(6))
    return img.resize(size, Image.BICUBIC)


def render(kind: str, title: str, paras, out_path: Path):
    L = LAYOUTS[kind]
    bg = background(L["size"])
    W, H = bg.size
    bx, by, bw, bbottom = L["box"]
    pad = L["pad"]
    px0, pw = int(W * bx), int(W * bw)
    py0 = int(H * by)
    max_h = int(H * bbottom) - py0
    inner = pw - 2 * pad
    d = ImageDraw.Draw(bg)

    for size in range(L["max_font"], L["min_font"] - 1, -1):
        lh = int(size * 1.42)
        gap = int(lh * 0.5)
        f = {k: font(k, size) for k in ("regular", "italic")}
        laid = [(wrap(d, t, f[st], inner), st) for t, st in paras]
        title_f = font("bold", int(size * 1.35))
        sub_f = font("italic", int(size * 0.8))
        head = int(size * 1.35 * 1.3) + int(size * 0.8 * 1.6) + int(size * 0.9)
        body = sum(len(ls) * lh for ls, _ in laid) + gap * (len(laid) - 1)
        ph = pad + head + body + pad
        if ph <= max_h:
            break
    ph = min(ph, max_h)

    box = (px0, py0, px0 + pw, py0 + ph)
    region = bg.crop(box).filter(ImageFilter.GaussianBlur(30))
    region = Image.blend(region, Image.new("RGB", region.size, "white"), 0.58)
    mask = Image.new("L", region.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, pw, ph), L["radius"], fill=255)
    bg.paste(region, box[:2], mask)
    d.rounded_rectangle(box, L["radius"], outline="white", width=3)

    x, y = px0 + pad, py0 + pad
    d.text((x, y), title, font=title_f, fill=INK)
    y += int(size * 1.35 * 1.3)
    d.text((x, y), "Voices of Recovery", font=sub_f, fill=SOFT)
    y += int(size * 0.8 * 1.6)
    d.line((x, y, x + inner, y), fill="white", width=3)
    y += int(size * 0.9)
    for ls, st in laid:
        for line in ls:
            if y + lh > py0 + ph:
                break
            d.text((x, y), line, font=f[st], fill=INK)
            y += lh
        y += gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bg.save(out_path, "PNG", optimize=True)
    return out_path


# ── Upload ────────────────────────────────────────────────────────────────────

def upload(paths):
    import boto3  # installed by the workflow step

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    bucket = os.environ.get("R2_BUCKET", "steptwelve-daily")
    for p in paths:
        s3.upload_file(
            str(p), bucket, p.name,
            ExtraArgs={"ContentType": "image/png",
                       "CacheControl": "public, max-age=300"},
        )
        print(f"Uploaded {p.name} -> {bucket}")


def main():
    args = sys.argv[1:]
    do_upload = "--upload" in args
    out_dir = ROOT / "output" / "wallpapers"
    if "--out" in args:
        out_dir = Path(args[args.index("--out") + 1])
    pos = [a for a in args if not a.startswith("--") and a != str(out_dir)]
    mmdd = pos[0] if pos else datetime.now().strftime("%m-%d")

    txt = MED_DIR / f"{mmdd}.txt"
    if not txt.exists():
        sys.exit(f"No text for {mmdd} at {txt}")
    title, paras = split_reading(txt.read_text(encoding="utf-8"))

    files = [
        render("desktop", title, paras, out_dir / "today-desktop.png"),
        render("phone", title, paras, out_dir / "today-phone.png"),
    ]
    for f in files:
        print(f"Rendered {f}")
    if do_upload:
        upload(files)


if __name__ == "__main__":
    main()
