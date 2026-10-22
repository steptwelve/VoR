#!/usr/bin/env python3
"""
generate_static_audio.py — One-time script to pre-record all static narration clips.

Generates intro and outro MP3s for every EN and ES voice and stores them in
templates/audio/. Once committed to the repo, build-video.sh and build_short.py
will use these files instead of making TTS calls, eliminating 429 throttling risk
for the intro and outro segments.

Files generated (24 total, 12 EN + 12 ES):
  templates/audio/intro-{voice}.mp3            — timeless intro for title card
  templates/audio/outro-vor-{voice}.mp3        — VoR end card narration
  templates/audio/outro-scisaa-{voice}.mp3     — SCISAA end card narration

Run once locally:
    cd ~/Documents/GitHub/Daily
    python3 generate_static_audio.py

Re-run to regenerate a specific voice only:
    python3 generate_static_audio.py --voice en-US-GuyNeural
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

DAILY_DIR = Path(__file__).resolve().parent
TEMPLATES = DAILY_DIR / "templates"
AUDIO_DIR = TEMPLATES / "audio"

EN_VOICES = [
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-ChristopherNeural",
    "en-US-AriaNeural",
    "en-US-EricNeural",
    "en-US-MichelleNeural",
]

ES_VOICES = [
    "es-MX-JorgeNeural",
    "es-MX-DaliaNeural",
    "es-US-AlonsoNeural",
    "es-US-PalomaNeural",
    "es-AR-TomasNeural",
    "es-AR-ElenaNeural",
]

# Timeless intro text — no date (the meditation body always opens with the date)
INTRO_EN = "Here's today's Daily Meditation from Voices of Recovery."
INTRO_ES = "Esta es la Meditación Diaria de hoy de Voces de Recuperación."

# Outro text — read from existing template files so there's one source of truth
OUTRO_FILES = {
    "vor-en":    TEMPLATES / "endcard-narration.txt",
    "vor-es":    TEMPLATES / "endcard-narration-es.txt",
    "scisaa-en": TEMPLATES / "endcard-narration-scisaa.txt",
    "scisaa-es": TEMPLATES / "endcard-narration-scisaa-es.txt",
}

RETRY_DELAYS = (0, 30, 60, 120)


def tts(voice: str, text_or_file: str, out_mp3: Path, is_file: bool = False) -> bool:
    """Call edge-tts with backoff. Returns True on success."""
    cmd = [sys.executable, "-m", "edge_tts", "--voice", voice,
           "--write-media", str(out_mp3)]
    if is_file:
        cmd += ["--file", text_or_file]
    else:
        cmd += ["--text", text_or_file]

    for attempt, delay in enumerate(RETRY_DELAYS):
        if delay:
            print(f"   ⚠️  Attempt {attempt + 1} failed — retrying in {delay}s...")
            time.sleep(delay)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return True
        tail = (r.stderr.strip().splitlines() or ["(no stderr)"])[-1]
        print(f"   ❌ edge-tts error: {tail[:200]}")
    return False


def generate(voice: str, label: str, text_or_file: str, out_mp3: Path,
             is_file: bool = False, overwrite: bool = False) -> bool:
    if out_mp3.exists() and not overwrite:
        print(f"   ⏭  {out_mp3.name} already exists — skipping")
        return True
    print(f"   🎤 {label} → {out_mp3.name}")
    ok = tts(voice, text_or_file, out_mp3, is_file=is_file)
    if ok:
        size = out_mp3.stat().st_size // 1024
        print(f"      ✅ {out_mp3.name} ({size} KB)")
    else:
        print(f"      ❌ FAILED: {out_mp3.name}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Pre-record static intro/outro audio clips.")
    parser.add_argument("--voice", help="Only generate for this voice (e.g. en-US-GuyNeural)")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if file exists")
    parser.add_argument("--lang", choices=["en", "es", "all"], default="all",
                        help="Which language(s) to generate (default: all)")
    args = parser.parse_args()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Verify outro template files exist
    for key, path in OUTRO_FILES.items():
        if not path.exists():
            print(f"⚠️  Missing outro template: {path} — skipping {key} outros")

    failures = []

    def do_voice(voice: str, lang: str):
        intro_text = INTRO_ES if lang == "es" else INTRO_EN

        # Intro
        ok = generate(
            voice,
            f"[{lang}] intro",
            intro_text,
            AUDIO_DIR / f"intro-{voice}.mp3",
            is_file=False,
            overwrite=args.overwrite,
        )
        if not ok:
            failures.append(f"intro-{voice}")

        # VoR outro
        outro_key = f"vor-{lang}"
        if OUTRO_FILES[outro_key].exists():
            ok = generate(
                voice,
                f"[{lang}] outro-vor",
                str(OUTRO_FILES[outro_key]),
                AUDIO_DIR / f"outro-vor-{voice}.mp3",
                is_file=True,
                overwrite=args.overwrite,
            )
            if not ok:
                failures.append(f"outro-vor-{voice}")

        # SCISAA outro
        scisaa_key = f"scisaa-{lang}"
        if OUTRO_FILES[scisaa_key].exists():
            ok = generate(
                voice,
                f"[{lang}] outro-scisaa",
                str(OUTRO_FILES[scisaa_key]),
                AUDIO_DIR / f"outro-scisaa-{voice}.mp3",
                is_file=True,
                overwrite=args.overwrite,
            )
            if not ok:
                failures.append(f"outro-scisaa-{voice}")

    voices_en = EN_VOICES if args.lang in ("en", "all") else []
    voices_es = ES_VOICES if args.lang in ("es", "all") else []

    if args.voice:
        # Filter to just the requested voice
        if args.voice in EN_VOICES:
            voices_en = [args.voice]
            voices_es = []
        elif args.voice in ES_VOICES:
            voices_en = []
            voices_es = [args.voice]
        else:
            sys.exit(f"Unknown voice: {args.voice}")

    total = len(voices_en) * 3 + len(voices_es) * 3
    print(f"🎙️  Generating {total} static audio clips → {AUDIO_DIR}\n")

    for v in voices_en:
        print(f"\n── {v} (EN) ──")
        do_voice(v, "en")

    for v in voices_es:
        print(f"\n── {v} (ES) ──")
        do_voice(v, "es")

    print(f"\n{'─'*50}")
    if failures:
        print(f"❌ {len(failures)} failures:")
        for f in failures:
            print(f"   • {f}")
        sys.exit(1)
    else:
        n = sum(1 for p in AUDIO_DIR.glob("*.mp3"))
        print(f"✅ Done — {n} MP3s in {AUDIO_DIR}")
        print("\nNext steps:")
        print("  git add templates/audio/")
        print("  git commit -m 'chore: add pre-recorded static intro/outro audio'")
        print("  git push")


if __name__ == "__main__":
    main()
