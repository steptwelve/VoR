#!/usr/bin/env python3
"""
build_short.py — Vertical YouTube Short of the daily meditation.

Usage:
    python3 build_short.py MM-DD YYYY
    VOR_LANG=es python3 build_short.py MM-DD YYYY
    BACKGROUND_VIDEO=/tmp/background.mp4 python3 build_short.py MM-DD YYYY
    SHORT_BRANDS=vor,scisaa python3 build_short.py MM-DD YYYY

Output: output/MM-DD-SHORT.mp4  (or MM-DD-es-SHORT.mp4)
        output/MM-DD-SCISAA-SHORT.mp4 when "scisaa" is in SHORT_BRANDS
        (same narration and captions; SCISAA end card and closing words)

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
import shutil
import subprocess
import sys
import tempfile
import time
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
TEMPLATES = DAILY_DIR / "templates"

suffix = "-es" if LANG == "es" else ""
MED_TEXT = MED_DIR / f"{MMDD}{suffix}.txt"

# One narrated main segment, one end card per brand. SHORT_BRANDS lists the
# brands to build (e.g. "vor,scisaa"); each gets its own output file:
#   vor    -> output/MM-DD[-es]-SHORT.mp4         (@StepTwelveSAA)
#   scisaa -> output/MM-DD[-es]-SCISAA-SHORT.mp4  (SCISAA channel)
BRANDS = {
    "vor": {
        "image": DAILY_DIR / "backgrounds" / "ChatGPT-Image-VoR.png",
        "crop_top": 0.26,     # drop the "Voices of Recovery" title band
        "center_x": 0.40,     # keep the sunburst
        "tag": "",
        "narration": "endcard-narration",
        "photo_credit": "",
    },
    "scisaa": {
        "image": DAILY_DIR / "backgrounds" / "scisaa-la-skyline.jpg",
        "crop_top": 0.0,
        "center_x": 0.30,     # downtown skyline + the palm-lined road
        "tag": "-SCISAA",
        "narration": "endcard-narration-scisaa",
        "photo_credit": "Photo: Venti Views / Unsplash  |  ",
    },
}
SHORT_BRANDS = [b.strip() for b in os.environ.get("SHORT_BRANDS", "vor").split(",")
                if b.strip()]
for _b in SHORT_BRANDS:
    if _b not in BRANDS:
        sys.exit(f"Unknown brand in SHORT_BRANDS: {_b}")


def output_for(brand):
    return OUT_DIR / f"{MMDD}{suffix}{BRANDS[brand]['tag']}-SHORT.mp4"

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


TTS_RETRY_DELAYS = (30, 60, 120)  # seconds between attempts


def tts(text_file, voice, mp3, srt=None):
    """edge-tts with backoff: Microsoft's endpoint intermittently returns
    HTTP 429 to GitHub runner IPs and clears within seconds to minutes."""
    cmd = [sys.executable, "-m", "edge_tts", "--voice", voice,
           "--file", str(text_file), "--write-media", str(mp3)]
    if srt:
        cmd += ["--write-subtitles", str(srt)]
    for attempt, delay in enumerate(TTS_RETRY_DELAYS, start=1):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return r
        tail = (r.stderr.strip().splitlines() or ["(no stderr)"])[-1]
        log(f"⚠️  edge-tts failed (attempt {attempt}): {tail[:200]}")
        log(f"   retrying in {delay}s...")
        time.sleep(delay)
    return run(cmd)  # final attempt; exits with full stderr on failure


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


def render_endcard(path, brand="vor"):
    cfg = BRANDS[brand]
    bg = Image.open(cfg["image"]).convert("RGB")
    iw, ih = bg.size
    bg = bg.crop((0, int(ih * cfg["crop_top"]), iw, ih))
    s = H / bg.height
    bg = bg.resize((int(bg.width * s), H), Image.LANCZOS)
    left = int(bg.width * cfg["center_x"] - W / 2)
    left = max(0, min(left, bg.width - W))
    bg = bg.crop((left, 0, left + W, H)).convert("RGBA")

    if LANG == "es":
        quote = ('"Una confraternidad de personas que comparten su experiencia, '
                 'fortaleza y esperanza entre sí para superar su adicción sexual '
                 'y ayudar a otros a recuperarse de la adicción o dependencia sexual."')
        attr = cfg["photo_credit"] + "Video: Pexels.com  |  Narración: Microsoft Azure TTS"
        if brand == "scisaa":
            cta = "Reuniones de SAA en el Sur de California:"
            url = "scisaa.org"
            meet = "En todo el mundo: saa-recovery.org"
            sub_lines = ["Suscríbete a nuestro boletín:", "scisaa.org/subscribe"]
        else:
            cta = "Para saber más sobre SAA, visita:"
            url = "saa-recovery.org"
            meet = "Encuentra una reunión: saa-meetings.org"
            sub_lines = ["Suscríbete: @StepTwelveSAA"]
    else:
        quote = ('"A fellowship of individuals who share their experience, strength, '
                 'and hope with each other so they may overcome their sexual addiction '
                 'and help others recover from sexual addiction or dependency."')
        attr = cfg["photo_credit"] + "Video: Pexels.com  |  Narration: Microsoft Azure TTS"
        if brand == "scisaa":
            cta = "SAA meetings in Southern California:"
            url = "scisaa.org"
            meet = "Worldwide: saa-recovery.org"
            sub_lines = ["Subscribe to our bulletin:", "scisaa.org/subscribe"]
        else:
            cta = "To learn more about SAA, visit:"
            url = "saa-recovery.org"
            meet = "Find a meeting: saa-meetings.org"
            sub_lines = ["Subscribe: @StepTwelveSAA"]

    qf = font(40)
    probe = ImageDraw.Draw(bg)
    quote_lines = wrap(probe, quote, qf, W - 200)
    top = 520
    bottom = top + 320 + 54 * len(quote_lines) + 170 + 52 * len(sub_lines)   # box fits the content
    box = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(box).rounded_rectangle([(60, top), (W - 60, bottom)], 28,
                                          fill=(0, 0, 0, 140))
    img = Image.alpha_composite(bg, box)
    d = ImageDraw.Draw(img)

    def center(text, f, y, col=(255, 255, 255)):
        d.text(((W - text_w(d, text, f)) // 2, y), text, font=f, fill=col)

    center(cta, font(46), top + 70)
    center(url, font(110 if len(url) < 12 else 92, True), top + 140)
    y = top + 320
    for line in quote_lines:
        center(line, qf, y, (210, 225, 210))
        y += 54
    center(meet, font(42, True), y + 35)
    sy = y + 105
    for line in sub_lines:
        center(line, font(42, True), sy)
        sy += 52
    af = font(24)
    while text_w(d, attr, af) > W - 160 and af.size > 16:
        af = font(af.size - 1)
    center(attr, af, sy + 25, (210, 225, 210))
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
    else:
        date_str = d.strftime("%B %-d, %Y")

    voice = pick_voice()
    log(f"📱 Building Short for {date_str} ({LANG}) — voice {voice} — "
        f"brands {', '.join(SHORT_BRANDS)}")
    work = Path(tempfile.mkdtemp(prefix=f"vor-short-{MMDD}-{LANG}-"))
    OUT_DIR.mkdir(exist_ok=True)

    # 1. Narration (meditation body) with sentence timings
    #
    # Reuse audio built by build-video.sh when available. The landscape build
    # exports /tmp/vor-audio-LANG/narration.{mp3,srt} — the meditation body
    # (no intro sentence). When those files are present we skip the TTS call
    # entirely, eliminating the duplicate Microsoft edge-tts request.
    shared_dir = Path(f"/tmp/vor-audio-{LANG}")
    shared_mp3 = shared_dir / "narration.mp3"
    shared_srt = shared_dir / "narration.srt"

    if shared_mp3.exists() and shared_srt.exists():
        shutil.copy(shared_mp3, work / "narration.mp3")
        shutil.copy(shared_srt, work / "narration.srt")
        log(f"   ✅ Reusing shared audio from {shared_dir} (skipping TTS call)")
    else:
        log(f"   ℹ️  No shared audio found at {shared_dir} — calling TTS")
        tts(MED_TEXT, voice, work / "narration.mp3", work / "narration.srt")

    main_dur = duration(work / "narration.mp3") + 0.6

    # Pre-recorded outro audio — no TTS call needed for end cards
    end_durs, skipped = {}, []
    for brand in SHORT_BRANDS:
        outro_src = TEMPLATES / "audio" / f"outro-{brand}-{voice}.mp3"
        if not outro_src.exists():
            log(f"   ⚠️  Pre-recorded outro not found: {outro_src}")
            log(f"   Run: python3 generate_static_audio.py --voice {voice} --lang {LANG}")
            sys.exit(1)
        dest = work / f"endcard-{brand}.mp3"
        shutil.copy(outro_src, dest)
        log(f"   ✅ [{brand}] outro copied from templates/audio/outro-{brand}-{voice}.mp3")
        end_durs[brand] = duration(dest) + 1.0
        total = main_dur + end_durs[brand]
        log(f"   [{brand}] narration {main_dur:.1f}s + end card "
            f"{end_durs[brand]:.1f}s = {total:.1f}s")
        if total > MAX_SECONDS:
            skipped.append(f"{brand} ({total:.0f}s)")
            del end_durs[brand]
    if skipped:
        log(f"⚠️  Over the 3:00 Shorts limit — skipping: {', '.join(skipped)}")
        Path("/tmp/short_warning.txt").write_text(
            f"{MMDD} Short skipped: {', '.join(skipped)} exceeds the 3:00 Shorts limit.")
    if not end_durs:
        sys.exit(3)

    # 2. Overlays: header, caption frames
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    cf = font(CAPTION_FONT_SIZE, True)
    render_header(work / "header.png", date_str)
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

    # 4-5. Per brand: end card segment, then join with the shared main segment
    built = 0
    for brand, end_dur in end_durs.items():
        png, end_mp4, out = (work / f"endcard-{brand}.png", work / f"end-{brand}.mp4",
                             output_for(brand))
        render_endcard(png, brand)
        run(["ffmpeg", "-y", "-loop", "1", "-i", str(png),
             "-i", str(work / f"endcard-{brand}.mp3"),
             "-vf", f"scale={W}:{H},fps={FPS},format=yuv420p",
             "-af", "adelay=1000:all=1,apad",
             "-c:v", "libx264", "-preset", "fast", "-crf", "21",
             "-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", "128k",
             "-t", f"{end_dur:.3f}", str(end_mp4)])
        run(["ffmpeg", "-y", "-i", str(work / "main.mp4"), "-i", str(end_mp4),
             "-filter_complex", "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]",
             "-map", "[v]", "-map", "[a]",
             "-c:v", "libx264", "-preset", "fast", "-crf", "21",
             "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out)])
        final = duration(out)
        if final > MAX_SECONDS:
            out.unlink()
            log(f"⚠️  [{brand}] final {final:.0f}s exceeds the Shorts limit — removed.")
            continue
        built += 1
        log(f"🎉 Short ready: {out}  ({final:.0f}s)")
    if not built:
        sys.exit(3)


if __name__ == "__main__":
    main()
