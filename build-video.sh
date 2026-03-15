#!/bin/bash
# ============================================================
# build-video.sh — Daily Meditation Video Builder
# @StepTwelveSAA YouTube Channel
#
# Usage:
#   ./build-video.sh MM-DD YYYY
#   BACKGROUND_VIDEO=/tmp/bg.mp4 ./build-video.sh MM-DD YYYY   (GitHub Actions)
#   VOR_LANG=es ./build-video.sh MM-DD YYYY                      (Spanish)
#
# Output: ~/Documents/GitHub/Daily/output/MM-DD-FINAL.mp4
#         ~/Documents/GitHub/Daily/output/MM-DD-titlecard.png
# ============================================================

set -e
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

# Activate venv if present (local dev)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
[[ -f "$SCRIPT_DIR/.venv/bin/activate" ]] && source "$SCRIPT_DIR/.venv/bin/activate"

# Self-healing: ensure this script stays executable
chmod +x "$0" 2>/dev/null || true

# ── Arguments ────────────────────────────────────────────────
MMDD="${1:-$(date +%m-%d)}"
YEAR="${2:-$(date +%Y)}"
# Use VOR_LANG to avoid collision with the system $LANG locale variable
VOR_LANG="${VOR_LANG:-en}"

# Date string for display (e.g. "March 10, 2026")
MONTH_DAY=$(date -j -f "%m-%d-%Y" "${MMDD}-${YEAR}" "+%B %-d, %Y" 2>/dev/null \
    || python3 -c "from datetime import datetime; print(datetime.strptime('${MMDD}-${YEAR}', '%m-%d-%Y').strftime('%B %-d, %Y'))")

if [ "$VOR_LANG" = "es" ]; then
    echo "🎬 Building SPANISH meditation video for ${MONTH_DAY}..."
else
    echo "🎬 Building meditation video for ${MONTH_DAY}..."
fi
echo ""

# ── Paths ─────────────────────────────────────────────────────
# Use SCRIPT_DIR so this works both locally and on GitHub Actions runners
# (where $HOME is /home/runner, not ~/Documents/GitHub/Daily)
DAILY_DIR="$SCRIPT_DIR"
TEMPLATES="$DAILY_DIR/templates"
BACKGROUNDS="$DAILY_DIR/backgrounds"
MEDITATIONS="$DAILY_DIR/meditations"
OUTPUT_DIR="$DAILY_DIR/output"
WORK_DIR="/tmp/vor-build-${MMDD}-${VOR_LANG}"

CHATGPT_IMAGE="$BACKGROUNDS/ChatGPT-Image-VoR.png"
ENDCARD_NARRATION="$TEMPLATES/endcard-narration.txt"

# Language-specific source file
if [ "$VOR_LANG" = "es" ]; then
    MEDITATION_TEXT="$MEDITATIONS/${MMDD}-es.txt"
    FINAL_OUTPUT="$OUTPUT_DIR/${MMDD}-es-FINAL.mp4"
    TITLECARD_OUTPUT="$OUTPUT_DIR/${MMDD}-es-titlecard.png"
else
    MEDITATION_TEXT="$MEDITATIONS/${MMDD}.txt"
    FINAL_OUTPUT="$OUTPUT_DIR/${MMDD}-FINAL.mp4"
    TITLECARD_OUTPUT="$OUTPUT_DIR/${MMDD}-titlecard.png"
fi

STATE_FILE="$DAILY_DIR/.vor-state.json"
# Use venv python if available, fall back to system python3
if [[ -f "$DAILY_DIR/.venv/bin/python3" ]]; then
    PYTHON="$DAILY_DIR/.venv/bin/python3"
    EDGE_TTS="$DAILY_DIR/.venv/bin/edge-tts"
elif [[ -f "$DAILY_DIR/venv/bin/python3" ]]; then
    PYTHON="$DAILY_DIR/venv/bin/python3"
    EDGE_TTS="$DAILY_DIR/venv/bin/edge-tts"
else
    PYTHON="python3"
    EDGE_TTS="edge-tts"
fi

# Allow BACKGROUND_VIDEO to be passed in (for GitHub Actions)
# Falls back to local backgrounds/ directory
if [ -z "$BACKGROUND_VIDEO" ]; then
    BACKGROUND_VIDEO=""
fi

# ── Auto-download background (local dev only) ─────────────────
# In GitHub Actions, BACKGROUND_VIDEO is passed in explicitly.
# Locally, we run download_background.py to rotate to the next Pexels video.
if [ -z "$BACKGROUND_VIDEO" ] && [ -z "$CI" ]; then
    echo "📥 Downloading next background video from Pexels..."
    LOCAL_BG="$BACKGROUNDS/background.mp4"
    if $PYTHON "$DAILY_DIR/download_background.py" --output "$LOCAL_BG"; then
        BACKGROUND_VIDEO="$LOCAL_BG"
        echo ""
    else
        echo "⚠️  Pexels download failed — falling back to existing background video"
        # Fall through to rotation logic below
    fi
fi

# Always start fresh — stale files in WORK_DIR cause silent audio bugs
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR" "$OUTPUT_DIR"

# ── Voice + Video rotation ────────────────────────────────────
read -r VOICE BACKGROUND_VIDEO_ROTATED < <($PYTHON << PYEOF
import json, os, glob

state_file = "$STATE_FILE"
bg_dir = "$BACKGROUNDS"
lang = "$VOR_LANG"

# English voices
en_voices = [
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-ChristopherNeural",
    "en-US-AriaNeural",
    "en-US-EricNeural",
    "en-US-MichelleNeural",
]

# Spanish voices
es_voices = [
    "es-MX-JorgeNeural",
    "es-MX-DaliaNeural",
    "es-US-AlonsoNeural",
    "es-US-PalomaNeural",
    "es-AR-TomasNeural",
    "es-AR-ElenaNeural",
]

voices = es_voices if lang == "es" else en_voices

# Videos: all .mov and .mp4 in backgrounds/, sorted for consistency
video_exts = ('.mov', '.mp4')
videos = sorted([f for f in glob.glob(os.path.join(bg_dir, '*'))
                 if os.path.splitext(f)[1].lower() in video_exts])

# Load state
state = {}
if os.path.exists(state_file):
    try:
        state = json.loads(open(state_file).read())
    except:
        state = {}

last_voice_idx = state.get('last_voice_idx', -1)
last_video_idx = state.get('last_video_idx', -1)

next_voice_idx = (last_voice_idx + 1) % len(voices)
next_video_idx = (last_video_idx + 1) % len(videos) if videos else 0

voice = voices[next_voice_idx]
video = videos[next_video_idx] if videos else ""

# Save state
state['last_voice_idx'] = next_voice_idx
state['last_video_idx'] = next_video_idx
open(state_file, 'w').write(json.dumps(state, indent=2))

print(voice, video)
PYEOF
)

# If BACKGROUND_VIDEO was passed in externally (GitHub Actions), use it
if [ -z "$BACKGROUND_VIDEO" ]; then
    BACKGROUND_VIDEO="$BACKGROUND_VIDEO_ROTATED"
fi

echo "🎙️  Voice:      $VOICE"
echo "🎬 Background: $(basename "$BACKGROUND_VIDEO")"
echo ""

# ── Step 0: Verify meditation text exists ────────────────────
if [ ! -f "$MEDITATION_TEXT" ]; then
    echo "❌ Meditation text not found: $MEDITATION_TEXT"
    if [ "$VOR_LANG" = "es" ]; then
        echo "   Run: python3 translate_meditation.py --date $MMDD"
    else
        echo "   Run: python3 daily_poster.py --test $MMDD"
    fi
    exit 1
fi

if [ ! -f "$BACKGROUND_VIDEO" ]; then
    echo "❌ Background video not found: $BACKGROUND_VIDEO"
    exit 1
fi

echo "✅ All source files found"
echo ""

# ── Step 1: Generate title card PNG ──────────────────────────
echo "🖼️  Step 1: Generating title card for ${MONTH_DAY}..."
$PYTHON << PYEOF
from PIL import Image, ImageDraw, ImageFont

bg = Image.open("$CHATGPT_IMAGE").convert("RGB")
W, H = 1920, 1080
iw, ih = bg.size
scale = W / iw
bg = bg.resize((int(iw*scale), int(ih*scale)), Image.LANCZOS)
bg = bg.crop((0, 0, W, H))

overlay = Image.new("RGBA", (W, H), (0, 0, 0, 40))
combined = Image.alpha_composite(bg.convert("RGBA"), overlay)

box_layer = Image.new("RGBA", (W, H), (0,0,0,0))
ImageDraw.Draw(box_layer).rectangle([(160, 470), (1760, 950)], fill=(15, 40, 12, 160))
combined = Image.alpha_composite(combined, box_layer).convert("RGB")
draw = ImageDraw.Draw(combined)

def load_font(size):
    for p in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try: return ImageFont.truetype(p, size)
        except: pass
    return ImageFont.load_default()

def draw_centered(draw, text, font, y):
    bb = draw.textbbox((0,0), text, font=font)
    draw.text(((W - (bb[2]-bb[0])) // 2, y), text, font=font, fill=(255,255,255))

start_y = 565
draw_centered(draw, "Daily Meditation",                                           load_font(72), start_y)
draw_centered(draw, "from Voices of Recovery",                                    load_font(52), start_y + 95)
draw_centered(draw, "$MONTH_DAY",                                                 load_font(52), start_y + 175)
draw_centered(draw, "saa-recovery.org/daily-meditation-from-voices-of-recovery/", load_font(28), start_y + 265)
combined.save("$WORK_DIR/titlecard.png")
print("  ✅ Title card generated")
PYEOF

# ── Step 2: Title card narration ─────────────────────────────
echo "🎤 Step 2: Generating title card narration..."
echo "Here's the Daily Meditation for ${MONTH_DAY} from Voices of Recovery" \
    > "$WORK_DIR/titlecard-narration.txt"
$EDGE_TTS --voice "$VOICE" \
    --file "$WORK_DIR/titlecard-narration.txt" \
    --write-media "$WORK_DIR/titlecard-narration.mp3"

# ── Step 3: Narrated title card video ────────────────────────
echo "🎬 Step 3: Creating narrated title card video..."
ffmpeg -y \
    -loop 1 -i "$WORK_DIR/titlecard.png" \
    -i "$WORK_DIR/titlecard-narration.mp3" \
    -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p \
    -c:a aac -ar 44100 -ac 2 -b:a 128k \
    -vf "scale=1920:1080,fps=25" \
    -shortest \
    "$WORK_DIR/titlecard-video.mp4" 2>/dev/null
echo "   ✅ Title card: $(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$WORK_DIR/titlecard-video.mp4" 2>/dev/null | xargs printf "%.1f")s"

# ── Step 4: Meditation narration ─────────────────────────────
echo "🎤 Step 4: Generating meditation narration..."

# 0.5s silence prefix so narration doesn't start cold
ffmpeg -y -f lavfi -i anullsrc=r=44100:cl=stereo -t 0.5 \
    "$WORK_DIR/silence.mp3" 2>/dev/null

$EDGE_TTS --voice "$VOICE" \
    --file "$MEDITATION_TEXT" \
    --write-media "$WORK_DIR/med-raw.mp3"

printf "file '$WORK_DIR/silence.mp3'\nfile '$WORK_DIR/med-raw.mp3'\n" \
    > "$WORK_DIR/audio-concat.txt"
ffmpeg -y -f concat -safe 0 -i "$WORK_DIR/audio-concat.txt" \
    -c:a libmp3lame -q:a 2 "$WORK_DIR/meditation-audio.mp3" 2>/dev/null

# ── Step 5: Background video + meditation audio ───────────────
echo "🎞️  Step 5: Combining background video + narration..."
RAW_DUR=$(ffprobe -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 "$WORK_DIR/med-raw.mp3" \
    | xargs printf "%.3f")
MED_DUR=$(echo "$RAW_DUR + 0.5" | bc)

ffmpeg -y -t "$MED_DUR" -i "$WORK_DIR/meditation-audio.mp3" \
    -ar 44100 -ac 2 -c:a pcm_s16le "$WORK_DIR/meditation-audio-trimmed.wav" 2>/dev/null

ffmpeg -y \
    -stream_loop -1 -i "$BACKGROUND_VIDEO" \
    -i "$WORK_DIR/meditation-audio-trimmed.wav" \
    -map 0:v -map 1:a \
    -vf "scale=1920:1080,fps=25" \
    -c:v libx264 -preset fast -crf 23 \
    -c:a aac -ar 44100 -ac 2 -b:a 128k \
    -t "$MED_DUR" \
    "$WORK_DIR/meditation-main.mp4" 2>/dev/null
echo "   ✅ Meditation: $(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$WORK_DIR/meditation-main.mp4" 2>/dev/null | xargs printf "%.1f")s"

# ── Step 6: End card video ────────────────────────────────────
echo "🖼️  Step 6: Creating end card..."

# Overlay text onto the branded forest background
$PYTHON << PYEOF
from PIL import Image, ImageDraw, ImageFont

# Use the branded forest image — crop out the top "Voices of Recovery" band
bg = Image.open("$CHATGPT_IMAGE").convert("RGB")
W, H = 1920, 1080
iw, ih = bg.size
scale = W / iw
bg = bg.resize((int(iw*scale), int(ih*scale)), Image.LANCZOS)
# Crop starting below the "Voices of Recovery" text band (~28% from top)
crop_top = int(bg.height * 0.28)
bg = bg.crop((0, crop_top, W, crop_top + H))

# Semi-transparent dark box for readability
overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
box_draw = ImageDraw.Draw(overlay)
box_draw.rectangle([(80, 100), (1840, 760)], fill=(0, 0, 0, 110))
combined = Image.alpha_composite(bg.convert("RGBA"), overlay).convert("RGB")
draw = ImageDraw.Draw(combined)

def load_font(size):
    for p in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try: return ImageFont.truetype(p, size)
        except: pass
    return ImageFont.load_default()

def draw_centered(draw, text, font, y, color=(255,255,255)):
    bb = draw.textbbox((0,0), text, font=font)
    x = (W - (bb[2]-bb[0])) // 2
    draw.text((x, y), text, font=font, fill=color)

lang = "$VOR_LANG"

if lang == "es":
    cta_line1 = "Para saber más sobre SAA, visita:"
    cta_url   = "saa-recovery.org"
    quote_lines = [
        '"Una confraternidad de personas que comparten',
        'su experiencia, fortaleza y esperanza entre sí',
        'para superar su adicción sexual y ayudar a otros',
        'a recuperarse de la adicción o dependencia sexual."',
    ]
    attribution = "Video: Pexels.com   |   Narración: Microsoft Azure TTS"
else:
    cta_line1 = "To learn more about SAA, visit:"
    cta_url   = "saa-recovery.org"
    quote_lines = [
        '"A fellowship of individuals who share their',
        'experience, strength, and hope with each other',
        'so they may overcome their sexual addiction',
        'and help others recover from sexual addiction',
        'or dependency."',
    ]
    attribution = "Video: Pexels.com   |   Narration: Microsoft Azure TTS"

draw_centered(draw, cta_line1, load_font(54), 170)
draw_centered(draw, cta_url, load_font(108), 245, color=(255,255,255))

y = 410
for line in quote_lines:
    draw_centered(draw, line, load_font(40), y, color=(210,225,210))
    y += 54

draw_centered(draw, attribution, load_font(21), 720, color=(210,225,210))

combined.save("$WORK_DIR/endcard-frame.png")
print("  ✅ End card frame generated")
PYEOF

if [ "$VOR_LANG" = "es" ]; then
    ENDCARD_NARRATION_FILE="$TEMPLATES/endcard-narration-es.txt"
    if [ ! -f "$ENDCARD_NARRATION_FILE" ]; then
        ENDCARD_NARRATION_FILE="$ENDCARD_NARRATION"
    fi
else
    ENDCARD_NARRATION_FILE="$ENDCARD_NARRATION"
fi

$EDGE_TTS --voice "$VOICE" \
    --file "$ENDCARD_NARRATION_FILE" \
    --write-media "$WORK_DIR/endcard-narration-raw.mp3"

ffmpeg -y -f lavfi -i anullsrc=r=44100:cl=stereo -t 1.0 \
    "$WORK_DIR/endcard-silence.mp3" 2>/dev/null
printf "file '$WORK_DIR/endcard-silence.mp3'\nfile '$WORK_DIR/endcard-narration-raw.mp3'\n" \
    > "$WORK_DIR/endcard-concat.txt"
ffmpeg -y -f concat -safe 0 -i "$WORK_DIR/endcard-concat.txt" \
    -c copy "$WORK_DIR/endcard-narration.mp3" 2>/dev/null

ENDCARD_DUR=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$WORK_DIR/endcard-narration.mp3" | xargs printf "%.3f")
ffmpeg -y \
    -loop 1 -i "$WORK_DIR/endcard-frame.png" \
    -i "$WORK_DIR/endcard-narration.mp3" \
    -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p \
    -c:a aac -ar 44100 -ac 2 -b:a 128k \
    -vf "scale=1920:1080,fps=25" \
    -t "$ENDCARD_DUR" \
    "$WORK_DIR/endcard-video.mp4" 2>/dev/null
echo "   ✅ End card: $(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$WORK_DIR/endcard-video.mp4" 2>/dev/null | xargs printf "%.1f")s"

# ── Step 7: Concatenate ───────────────────────────────────────
echo "🔗 Step 7: Concatenating title + meditation + end card..."
ffmpeg -y \
    -i "$WORK_DIR/titlecard-video.mp4" \
    -i "$WORK_DIR/meditation-main.mp4" \
    -i "$WORK_DIR/endcard-video.mp4" \
    -filter_complex \
      "[0:v][0:a][1:v][1:a][2:v][2:a]concat=n=3:v=1:a=1[vout][aout]" \
    -map "[vout]" -map "[aout]" \
    -c:v libx264 -preset fast -crf 23 \
    -c:a aac -b:a 128k \
    -movflags +faststart \
    "$FINAL_OUTPUT" 2>/dev/null

# ── Save thumbnail ────────────────────────────────────────────
cp "$WORK_DIR/titlecard.png" "$TITLECARD_OUTPUT"

# ── Done ──────────────────────────────────────────────────────
echo ""
echo "🎉 SUCCESS!"
echo ""
DURATION=$(ffprobe -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 "$FINAL_OUTPUT" 2>/dev/null | xargs printf "%.0f")
echo "   📹 Output:   $FINAL_OUTPUT"
echo "   ⏱️  Duration: ${DURATION}s"
echo "   💾 Size:     $(du -sh "$FINAL_OUTPUT" | cut -f1)"
echo ""
echo "Ready to upload! 🚀"
