#!/usr/bin/env python3
"""
post_scisaa_bsky.py — Post today's SCISAA-branded Short to the SCISAA Bluesky
account (@scisaa.org) as a native video, in English or Spanish.

Runs as its own workflow step after the SCISAA Short is built (Step 4d for
English, Step 5d for Spanish), and only when the repo variables ENABLE_SCISAA
and ENABLE_SCISAA_BSKY are both 'true'. It does not touch daily_poster.py or
the @ocisaa.org Bluesky post, so a problem here can never affect the main run
(or the ISO, which uses this code).

Flow:
  1. Find output/MM-DD[-es]-SCISAA-SHORT.mp4 and meditations/MM-DD[-es].txt.
  2. Build the post text: opening quote + attribution, a meetings line that
     points to scisaa.org, and a few hashtags. Bluesky allows 300 characters;
     the quote is the only part that is ever shortened (with an ellipsis).
  3. Log in as @scisaa.org with an app password and post the video, with
     rich-text facets so the hashtags and the link are clickable.
  4. Write meditations/MM-DD-scisaa-bsky-{lang}.posted so re-runs never
     double-post (the workflow's commit step adds meditations/*.posted).

Usage:  python post_scisaa_bsky.py [MM-DD] [--lang en|es] [--dry-run] [--force]
          --dry-run  build and print the post, check the video, post nothing
          --force    post even if today's .posted flag already exists
Env:    BSKY_SCISAA_APP_PASSWORD  required to post (an app password, never the
                                  account password)
        BSKY_SCISAA_USERNAME      default: scisaa.org
        BSKY_VIDEO_MAX_MB         default: 100 (Bluesky's video size limit)
Exit:   0 = posted, already posted, or dry run; 1 = failed (the workflow then
        sends a Pushover warning).
"""

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MED_DIR = PROJECT_ROOT / "meditations"
OUT_DIR = PROJECT_ROOT / "output"

try:  # local runs read the same .env as daily_poster.py; Actions uses secrets
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
except Exception:
    pass

BSKY_LIMIT = 300
SAFE_LIMIT = 296                      # small margin under Bluesky's 300
MAX_VIDEO_MB = float(os.environ.get("BSKY_VIDEO_MAX_MB", "100"))
SHORT_WIDTH, SHORT_HEIGHT = 1080, 1920   # build_short.py renders 9:16
DEFAULT_USERNAME = "scisaa.org"

SCISAA_URL = "https://scisaa.org"
SCISAA_DISPLAY = "scisaa.org"

LANGS = {
    "en": {
        "suffix": "",
        "cta": "SAA meetings in Southern California:",
        "hashtags": "#SAA #recovery #sobriety",
        "alt": ("Daily Voices of Recovery meditation from the Southern "
                "California Intergroup of SAA (SCISAA)."),
    },
    "es": {
        "suffix": "-es",
        "cta": "Reuniones en el Sur de California:",
        "hashtags": "#SAA #recuperación #sobriedad",
        "alt": ("Meditación diaria de Voces de Recuperación del Intergrupo "
                "del Sur de California de SAA (SCISAA)."),
    },
}


def log(msg):
    print(msg, flush=True)


# ── Paths ────────────────────────────────────────────────────────────────────

def text_path(mmdd, lang):
    return MED_DIR / f"{mmdd}{LANGS[lang]['suffix']}.txt"


def short_path(mmdd, lang):
    return OUT_DIR / f"{mmdd}{LANGS[lang]['suffix']}-SCISAA-SHORT.mp4"


def flag_path(mmdd, lang):
    return MED_DIR / f"{mmdd}-scisaa-bsky-{lang}.posted"


# ── Post text ────────────────────────────────────────────────────────────────

def parse_meditation(full_text):
    """Return (quote, attribution) from a meditations/MM-DD[-es].txt file.

    Layout: date line, opening quote, attribution line, body... The quote is
    the second paragraph (same as build_bsky_text() in daily_poster.py). The
    third paragraph is used as an attribution only if it looks like one:
    short and not ending like a sentence.
    """
    parts = [p.strip() for p in full_text.split("\n\n") if p.strip()]
    if len(parts) > 1:
        quote = parts[1]
    elif parts:
        quote = parts[0]
    else:
        quote = ""
    quote = " ".join(quote.split())

    attribution = ""
    if len(parts) > 2:
        cand = " ".join(parts[2].replace("*", "").split())
        cand = re.sub(r"\s+,", ",", cand)
        if cand and len(cand) <= 100 and not cand.endswith((".", "!", "?")):
            attribution = cand
    return quote, attribution


def build_text(quote, attribution, lang):
    """Quote + attribution + meetings line + hashtags, within 300 characters.
    Only the quote is shortened if needed (word by word, with an ellipsis)."""
    s = LANGS[lang]
    attrib = f"\n— {attribution}" if attribution else ""
    tail = f"\n\n{s['cta']} {SCISAA_DISPLAY}\n{s['hashtags']}"

    def assemble(q):
        return f"{q}{attrib}{tail}"

    text = assemble(quote)
    if len(text) <= SAFE_LIMIT:
        return text

    words = quote.split()
    while words:
        words.pop()
        shortened = " ".join(words).rstrip(",;:—- ") + "…"
        text = assemble(shortened)
        if len(text) <= SAFE_LIMIT:
            return text
    raise ValueError("Post text cannot be made to fit in 300 characters")


def facet_specs(text):
    """Hashtag and link positions as (kind, value, byte_start, byte_end).

    AT Protocol facets use UTF-8 BYTE offsets, not character positions, so
    curly quotes, em dashes and accented letters must be counted in bytes
    (the same bug daily_poster.py fixed for @ocisaa.org).
    """
    def byte_range(start, end):
        return (len(text[:start].encode("utf-8")), len(text[:end].encode("utf-8")))

    specs = []
    for m in re.finditer(r"#\w+", text):          # str regex: matches é, ó, ñ…
        b0, b1 = byte_range(m.start(), m.end())
        specs.append(("tag", m.group()[1:], b0, b1))
    i = text.find(SCISAA_DISPLAY)
    if i != -1:
        b0, b1 = byte_range(i, i + len(SCISAA_DISPLAY))
        specs.append(("link", SCISAA_URL, b0, b1))
    return specs


# ── Posting ──────────────────────────────────────────────────────────────────

def post_video(text, specs, video, lang, username, password):
    from atproto import Client, models      # imported late: dry runs need no SDK

    F = models.AppBskyRichtextFacet
    facets = []
    for kind, value, b0, b1 in specs:
        feature = F.Tag(tag=value) if kind == "tag" else F.Link(uri=value)
        facets.append(F.Main(index=F.ByteSlice(byte_start=b0, byte_end=b1),
                             features=[feature]))

    client = Client()
    log(f"Logging into Bluesky as {username}")
    client.login(username, password)
    log(f"Uploading {video.name} ({video.stat().st_size / 1e6:.1f} MB)…")
    resp = client.send_video(
        text=text,
        video=video.read_bytes(),
        video_alt=LANGS[lang]["alt"],
        video_aspect_ratio=models.AppBskyEmbedDefs.AspectRatio(
            width=SHORT_WIDTH, height=SHORT_HEIGHT),
        langs=[lang],
        facets=facets,
    )
    return getattr(resp, "uri", None)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mmdd", nargs="?", help="MM-DD (default: today)")
    ap.add_argument("--lang", choices=sorted(LANGS), default="en")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and print the post, post nothing")
    ap.add_argument("--force", action="store_true",
                    help="post even if the .posted flag exists")
    args = ap.parse_args()
    if args.mmdd is None:
        args.mmdd = datetime.now().strftime("%m-%d")
    if not re.fullmatch(r"\d{2}-\d{2}", args.mmdd):
        ap.error("date must look like MM-DD, e.g. 10-25")
    return args


def main():
    args = parse_args()
    mmdd, lang = args.mmdd, args.lang
    flag = flag_path(mmdd, lang)
    txt = text_path(mmdd, lang)
    video = short_path(mmdd, lang)

    if not args.dry_run and not args.force and flag.exists():
        log(f"Already posted ({flag.name}) — nothing to do. "
            f"Delete it or use --force to post again.")
        return 0

    if not txt.exists():
        log(f"ERROR: meditation text not found: {txt}")
        return 1

    quote, attribution = parse_meditation(txt.read_text(encoding="utf-8"))
    if not quote:
        log(f"ERROR: could not find a quote in {txt.name}")
        return 1
    text = build_text(quote, attribution, lang)
    specs = facet_specs(text)

    log("─" * 70)
    log(f"SCISAA Bluesky [{lang}] {mmdd}  ({len(text)}/{BSKY_LIMIT} chars)")
    log("─" * 70)
    log(text)
    log("─" * 70)
    for kind, value, b0, b1 in specs:
        log(f"  facet {kind:<4} {value}  bytes {b0}-{b1}")
    if video.exists():
        log(f"Video: {video.name}  {video.stat().st_size / 1e6:.1f} MB")
    else:
        log(f"Video: MISSING — {video}")

    if args.dry_run:
        log("DRY RUN: nothing posted.")
        return 0

    if not video.exists():
        log("ERROR: no SCISAA Short to post (see the Short build step).")
        return 1
    size_mb = video.stat().st_size / 1e6
    if size_mb > MAX_VIDEO_MB:
        log(f"ERROR: Short is {size_mb:.1f} MB, over the {MAX_VIDEO_MB:.0f} MB "
            f"Bluesky limit — not posting.")
        return 1

    password = os.environ.get("BSKY_SCISAA_APP_PASSWORD", "").strip()
    if not password:
        log("ERROR: BSKY_SCISAA_APP_PASSWORD is not set.")
        return 1
    username = os.environ.get("BSKY_SCISAA_USERNAME", "").strip() or DEFAULT_USERNAME

    try:
        uri = post_video(text, specs, video, lang, username, password)
    except Exception as e:
        # Print the error type and message only — never credentials.
        log(f"ERROR: Bluesky post failed: {type(e).__name__}: {e}")
        return 1

    log(f"✅ Posted to @{username}: {uri or '<no uri returned>'}")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    flag.write_text(f"posted: {stamp}\naccount: {username}\nlang: {lang}\n"
                    f"uri: {uri or ''}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
