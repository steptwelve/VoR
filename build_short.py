#!/usr/bin/env python3
"""
build_short.py — Vertical YouTube Short of the daily meditation.

Usage:
    python3 build_short.py MM-DD YYYY
    VOR_LANG=es python3 build_short.py MM-DD YYYY
    BACKGROUND_VIDEO=/tmp/background.mp4 python3 build_short.py MM-DD YYYY

Output: output/MM-DD-SHORT.mp4  (or MM-DD-es-SHORT.mp4)

Differences from build-video.sh (landscape):
  - 1080x1920 (9:16), so YouTube classifies it as a Short.
  - No static title card: the background video starts at frame 1 with a
    header overlay, so scrollers see motion immediately.
  - Burned-in captions (most Shorts are watched muted), rendered with
    Pillow so no libass/drawtext is needed in ffmpeg.
  - Hard 3:00 limit. If the build would exceed it, nothing is written and
    the script exits 3 (skip + warn, never truncate).

Voice: uses VOICE env var if set; otherwise the voice build-video.sh just
used (current last_voice_idx in .vor-state.json — read, never advanced),
so the Short and the landscape video match.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
FPS = 25
MAX_SECONDS = 179.0  # YouTube Shorts limit is 3:00; keep a margin
NARRATION_DELAY = 0.3  # seconds of silence before the voice starts

DAILY_DIR = Path(__file__).resolve().parent
MMDD = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%m-%d")
YEAR = sys.argv[2] if len(sys.argv) > 2 else str(datetime.now().year)
LANG = os.environ.get("VOR_LANG", "en")

MED_DIR = DAILY_DIR / "meditations"
OUT_DIR = DAILY_DIR / "output"
BRAND_IMG = DAILY_DIR / "backgrounds" / "ChatGPT-Image-VoR.png"
TEMPLATES = DAILY_DIR / "templates"

suffix = "-es" if LANG == "es" else ""
MED_TEXT = MED_DIR / f"{MMDD}{suffix}.txt"
OUTPUT = OUT_DIR / f"{MMDD}{suffix}-SHORT.mp4"

EN_VOICES = ["en-US-GuyNeural", "en-US-JennyNeural", "en-US-ChristopherNeural",
             "en-US-AriaNeural", "en-US-EricNeural", "en-US-MichelleNeural"]
ES_VOICES = ["es-MX-JorgeNeural", "es-MX-DaliaNeural", "es-US-AlonsoNeural",
             "es-US-PalomaNeural", "es-AR-TomasNeural", "es-AR-ElenaNeural"]

SPANISH_MONTHS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
                  "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

# Text safe area for the Shorts player UI: the top ~150px holds icons, the
# right ~150px holds the like/comment column, and the bottom ~420px holds the
# title/channel overlay. Captions stay inside this box.
SAFE_LEFT, SAFE_RIGHT = 80, 920
CAPTION_CENTER_Y = 960
CAPTION_FONT_SIZE = 66
CAPTION_MAX_LINES = 2


def log(msg):
    print(msg, flush=True)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"❌ Command failed: {' '.join(str(c) for c in cmd[:6])} ...")
        log(r.stderr[-3000:])
        sys.exit(1)
    return r


def duration(path):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    return float(r.stdout.strip())


def font(size, bold=False):
    bold_paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    reg_paths = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in (bold_paths if bold else reg_paths) + bold_paths:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def text_w(draw, text, f):
    b = draw.textbbox((0, 0), text, font=f)
    return b[2] - b[0]


def wrap(draw, text, f, max_w):
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and text_w(draw, trial, f) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def pick_voice():
    if os.environ.get("VOICE"):
        return os.environ["VOICE"]
    voices = ES_VOICES if LANG == "es" else EN_VOICES
    idx = 0
    state = DAILY_DIR / ".vor-state.json"
    if state.exists():
        try:
            idx = int(json.loads(state.read_text()).get("last_voice_idx", 0))
        except (ValueError, json.JSONDecodeError):
            idx = 0
    return voices[idx % len(voices)]


def tts(text_file, voice, mp3, srt=None):
    cmd = [sys.executable, "-m", "edge_tts", "--voice", voice,
           "--file", str(text_file), "--write-media", str(mp3)]
    if srt:
        cmd += ["--write-subtitles", str(srt)]
    run(cmd)


def parse_srt(path):
    def ts(s):
        h, m, rest = s.split(":")
        sec, ms = rest.replace(".", ",").split(",")
        return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000
    cues = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip()):
        lines = [l for l in block.splitlines() if l.strip()]
        for i, l in enumerate(lines):
            if "-->" in l:
                a, b = [x.strip() for x in l.split("-->")]
                text = " ".join(lines[i + 1:]).strip()
                if text:
                    cues.append((ts(a), ts(b), text))
                break
    return cues


def chunk_cues(cues, draw, f, max_w):
    """Split sentence-level cues into caption chunks of <= CAPTION_MAX_LINES
    lines, timing each chunk in proportion to its share of the characters."""
    out = []
    for start, end, text in cues:
        chunks, cur = [], []
        for word in text.split():
            trial = cur + [word]
            if cur and len(wrap(draw, " ".join(trial), f, max_w)) > CAPTION_MAX_LINES:
                chunks.append(" ".join(cur))
                cur = [word]
            else:
                cur = trial
        if cur:
            chunks.append(" ".join(cur))
        total = sum(len(c) for c in chunks) or 1
        t = start
        for c in chunks:
            d = (end - start) * len(c) / total
            out.append((t, t + d, c))
            t += d
    return out


def render_caption(text, path, draw_probe, f):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    lines = wrap(draw_probe, text, f, SAFE_RIGHT - SAFE_LEFT)
    line_h = int(CAPTION_FONT_SIZE * 1.25)
    y = CAPTION_CENTER_Y - (line_h * len(lines)) // 2
    cx = (SAFE_LEFT + SAFE_RIGHT) // 2
    for line in lines:
        x = cx - text_w(d, line, f) // 2
        d.text((x + 3, y + 4), line, font=f, fill=(0, 0, 0, 150))  # soft shadow
        d.text((x, y), line, font=f, fill=(255, 255, 255, 255),
               stroke_width=5, stroke_fill=(0, 0, 0, 255))
        y += line_h
    img.save(path)


def render_header(path, date_str):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for y in range(0, 560):  # top gradient for legibility
        d.line([(0, y), (W, y)], fill=(5, 20, 5, int(190 * (1 - y / 560))))
    title = "Meditación Diaria" if LANG == "es" else "Daily Meditation"
    sub = "de Voces de Recuperación" if LANG == "es" else "from Voices of Recovery"
    for text, f, y, col in [(title, font(78, True), 175, (255, 255, 255)),
                            (sub, font(46), 275, (215, 235, 215)),
                            (date_str, font(46), 340, (215, 235, 215))]:
        d.text(((W - text_w(d, text, f)) // 2, y), text, font=f, fill=col,
               stroke_width=2, stroke_fill=(0, 0, 0))
    img.save(path)


def render_endcard(path):
    bg = Image.open(BRAND_IMG).convert("RGB")
    iw, ih = bg.size
    bg = bg.crop((0, int(ih * 0.26), iw, ih))          # drop the title band
    s = H / bg.height
    bg = bg.resize((int(bg.width * s), H), Image.LANCZOS)
    left = int(bg.width * 0.40 - W / 2)                 # keep the sunburst
    left = max(0, min(left, bg.width - W))
    bg = bg.crop((left, 0, left + W, H)).convert("RGBA")

    if LANG == "es":
        cta = "Para saber más sobre SAA, visita:"
        quote = ('"Una confraternidad de personas que comparten su experiencia, '
                 'fortaleza y esperanza entre sí para superar su adicción sexual '
                 'y ayudar a otros a recuperarse de la adicción o dependencia sexual."')
        meet = "Encuentra una reunión: saa-meetings.org"
        attr = "Video: Pexels.com  |  Narración: Microsoft Azure TTS"
    else:
        cta = "To learn more about SAA, visit:"
        quote = ('"A fellowship of individuals who share their experience, strength, '
                 'and hope with each other so they may overcome their sexual addiction '
                 'and help others recover from sexual addiction or dependency."')
        meet = "Find a meeting: saa-meetings.org"
        attr = "Video: Pexels.com  |  Narration: Microsoft Azure TTS"

    qf = font(40)
    probe = ImageDraw.Draw(bg)
    quote_lines = wrap(probe, quote, qf, W - 200)
    top = 520
    bottom = top + 320 + 54 * len(quote_lines) + 170   # box fits the content
    box = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(box).rounded_rectangle([(60, top), (W - 60, bottom)], 28,
                                          fill=(0, 0, 0, 140))
    img = Image.alpha_composite(bg, box)
    d = ImageDraw.Draw(img)

    def center(text, f, y, col=(255, 255, 255)):
        d.text(((W - text_w(d, text, f)) // 2, y), text, font=f, fill=col)

    center(cta, font(46), top + 70)
    center("saa-recovery.org", font(92, True), top + 145)
    y = top + 320
    for line in quote_lines:
        center(line, qf, y, (210, 225, 210))
        y += 54
    center(meet, font(42, True), y + 35)
    center(attr, font(24), y + 115, (210, 225, 210))
    img.convert("RGB").save(path)


def main():
    if not MED_TEXT.exists():
        log(f"❌ Meditation text not found: {MED_TEXT}")
        sys.exit(1)
    bg_video = Path(os.environ.get("BACKGROUND_VIDEO",
                                   DAILY_DIR / "backgrounds" / "background.mp4"))
    if not bg_video.exists():
        log(f"❌ Background video not found: {bg_video}")
        sys.exit(1)

    d = datetime.strptime(f"{MMDD}-{YEAR}", "%m-%d-%Y")
    if LANG == "es":
        date_str = f"{d.day} de {SPANISH_MONTHS[d.month - 1]} de {d.year}"
        intro = f"Esta es la Meditación Diaria del {date_str}, de Voces de Recuperación."
        end_file = TEMPLATES / "endcard-narration-es.txt"
        if not end_file.exists():
            end_file = TEMPLATES / "endcard-narration.txt"
    else:
        date_str = d.strftime("%B %-d, %Y")
        intro = f"Here's the Daily Meditation for {date_str} from Voices of Recovery."
        end_file = TEMPLATES / "endcard-narration.txt"

    voice = pick_voice()
    log(f"📱 Building Short for {date_str} ({LANG}) — voice {voice}")
    work = Path(tempfile.mkdtemp(prefix=f"vor-short-{MMDD}-{LANG}-"))
    OUT_DIR.mkdir(exist_ok=True)

    # 1. Narration (intro + meditation) with sentence timings, and end card
    narr_txt = work / "narration.txt"
    narr_txt.write_text(intro + "\n\n" + MED_TEXT.read_text(encoding="utf-8"),
                        encoding="utf-8")
    tts(narr_txt, voice, work / "narration.mp3", work / "narration.srt")
    tts(end_file, voice, work / "endcard.mp3")
    main_dur = duration(work / "narration.mp3") + 0.6
    end_dur = duration(work / "endcard.mp3") + 1.0
    total = main_dur + end_dur
    log(f"   Narration {main_dur:.1f}s + end card {end_dur:.1f}s = {total:.1f}s")
    if total > MAX_SECONDS:
        log(f"⚠️  {total:.0f}s is over the 3:00 Shorts limit — skipping the Short.")
        Path("/tmp/short_warning.txt").write_text(
            f"{MMDD} Short skipped: {total:.0f}s exceeds the 3:00 Shorts limit.")
        sys.exit(3)

    # 2. Overlays: header, caption frames, end card
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    cf = font(CAPTION_FONT_SIZE, True)
    render_header(work / "header.png", date_str)
    render_endcard(work / "endcard.png")
    blank = work / "cap_blank.png"
    Image.new("RGBA", (W, H), (0, 0, 0, 0)).save(blank)

    chunks = chunk_cues(parse_srt(work / "narration.srt"), probe, cf,
                        SAFE_RIGHT - SAFE_LEFT)
    entries, t = [], 0.0
    for i, (a, b, text) in enumerate(chunks):
        a, b = a + NARRATION_DELAY, b + NARRATION_DELAY  # audio is delayed too
        a, b = max(a, t), max(b, a + 0.3)
        if a - t > 0.05:
            entries.append((blank, a - t))
        p = work / f"cap_{i:03d}.png"
        render_caption(text, p, probe, cf)
        entries.append((p, b - a))
        t = b
    entries.append((blank, max(main_dur - t, 0.1)))
    with open(work / "captions.ffconcat", "w") as fh:
        fh.write("ffconcat version 1.0\n")
        for p, dur in entries:
            fh.write(f"file '{p}'\nduration {dur:.3f}\n")
        fh.write(f"file '{blank}'\n")
    log(f"   {len(chunks)} caption frames")

    # 3. Main segment: vertical-cropped background + header + captions + voice
    vf = (f"[0:v]scale=-2:{H},crop={W}:{H},fps={FPS},"
          f"eq=brightness=-0.07:saturation=1.05,setsar=1[bg];"
          f"[bg][1:v]overlay=0:0[h];"
          f"[2:v]format=rgba[c];[h][c]overlay=0:0:eof_action=repeat,format=yuv420p[v]")
    run(["ffmpeg", "-y", "-stream_loop", "-1", "-i", str(bg_video),
         "-loop", "1", "-i", str(work / "header.png"),
         "-f", "concat", "-safe", "0", "-i", str(work / "captions.ffconcat"),
         "-i", str(work / "narration.mp3"),
         "-filter_complex", vf, "-map", "[v]", "-map", "3:a",
         "-af", f"adelay={int(NARRATION_DELAY*1000)}:all=1,apad",
         "-c:v", "libx264", "-preset", "fast", "-crf", "21",
         "-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", "128k",
         "-t", f"{main_dur:.3f}", str(work / "main.mp4")])

    # 4. End card segment
    run(["ffmpeg", "-y", "-loop", "1", "-i", str(work / "endcard.png"),
         "-i", str(work / "endcard.mp3"),
         "-vf", f"scale={W}:{H},fps={FPS},format=yuv420p",
         "-af", "adelay=1000:all=1,apad",
         "-c:v", "libx264", "-preset", "fast", "-crf", "21",
         "-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", "128k",
         "-t", f"{end_dur:.3f}", str(work / "end.mp4")])

    # 5. Join
    run(["ffmpeg", "-y", "-i", str(work / "main.mp4"), "-i", str(work / "end.mp4"),
         "-filter_complex", "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]",
         "-map", "[v]", "-map", "[a]",
         "-c:v", "libx264", "-preset", "fast", "-crf", "21",
         "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(OUTPUT)])

    final = duration(OUTPUT)
    if final > MAX_SECONDS:
        OUTPUT.unlink()
        log(f"⚠️  Final {final:.0f}s exceeds the Shorts limit — removed.")
        sys.exit(3)
    log(f"🎉 Short ready: {OUTPUT}  ({final:.0f}s)")


if __name__ == "__main__":
    main()
