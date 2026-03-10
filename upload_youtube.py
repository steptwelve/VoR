#!/usr/bin/env python3
"""
upload_youtube.py - Upload today's meditation video to YouTube.

Usage:
    python3 upload_youtube.py               # uploads today's date
    python3 upload_youtube.py 03-10 2026    # uploads specific date
    python3 upload_youtube.py 03-10 2026 es # uploads Spanish version

Reads credentials from youtube_token.json and youtube_client_secrets.json.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# ── Date / Language args ──────────────────────────────────────
MMDD   = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%m-%d")
YEAR   = sys.argv[2] if len(sys.argv) > 2 else str(datetime.now().year)
LANG   = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("LANG_VOR", "en")

DAILY_DIR  = Path(__file__).parent
OUTPUT_DIR = DAILY_DIR / "output"

# File paths
if LANG == "es":
    VIDEO_FILE     = OUTPUT_DIR / f"{MMDD}-es-FINAL.mp4"
    THUMBNAIL_FILE = OUTPUT_DIR / f"{MMDD}-es-titlecard.png"
else:
    VIDEO_FILE     = OUTPUT_DIR / f"{MMDD}-FINAL.mp4"
    THUMBNAIL_FILE = OUTPUT_DIR / f"{MMDD}-titlecard.png"

TOKEN_FILE          = DAILY_DIR / "youtube_token.json"
CLIENT_SECRETS_FILE = DAILY_DIR / "youtube_client_secrets.json"

# ── Validate files exist ──────────────────────────────────────
if not VIDEO_FILE.exists():
    print(f"❌ Video not found: {VIDEO_FILE}")
    sys.exit(1)
if not THUMBNAIL_FILE.exists():
    print(f"⚠️  Thumbnail not found: {THUMBNAIL_FILE} (uploading without thumbnail)")
    THUMBNAIL_FILE = None

# ── Build title and description ───────────────────────────────
# Parse date for display
try:
    date_obj   = datetime.strptime(f"{MMDD}-{YEAR}", "%m-%d-%Y")
    date_str   = date_obj.strftime("%B %-d, %Y")   # e.g. "March 10, 2026"
    month_year = date_obj.strftime("%B %Y")
except Exception:
    date_str   = f"{MMDD}-{YEAR}"
    month_year = YEAR

if LANG == "es":
    title       = f"Voces de Recuperación — {date_str}"
    description = f"""Meditación Diaria de Voces de Recuperación para el {date_str}.

Voces de Recuperación es un lector diario para personas en recuperación de la adicción sexual, publicado por Sexoadictos Anónimos® (SAA).

📖 Lectura de hoy: https://saa-recovery.org/daily-meditation-from-voices-of-recovery/

Sexoadictos Anónimos (SAA) es una confraternidad de personas que comparten su experiencia, fortaleza y esperanza con el objetivo de superar su adicción sexual.

🌐 Más información: https://saa-recovery.org
🤝 Encuentra una reunión: https://saa-meetings.org

#recuperacion #adiccion #SAA #meditacion #VocesDeRecuperacion"""
    tags        = ["recuperación", "adicción sexual", "SAA", "meditación diaria",
                   "Sexoadictos Anónimos", "voces de recuperación", "espiritualidad"]
    playlist_id = os.environ.get("YT_PLAYLIST_ID_ES", "")
else:
    title       = f"Voices of Recovery — {date_str}"
    description = f"""Daily Meditation from Voices of Recovery for {date_str}.

Voices of Recovery is a daily reader for people in recovery from sex addiction, published by Sex Addicts Anonymous® (SAA).

📖 Today's reading: https://saa-recovery.org/daily-meditation-from-voices-of-recovery/

Sex Addicts Anonymous (SAA) is a fellowship of individuals who share their experience, strength, and hope with each other so they may overcome their sexual addiction and help others recover from sexual addiction or dependency.

🌐 Learn more: https://saa-recovery.org
🤝 Find a meeting: https://saa-meetings.org

#recovery #sobriety #addiction #SAA #dailymeditation #VoicesOfRecovery #sexaddict"""
    tags        = ["recovery", "sex addiction", "SAA", "daily meditation",
                   "Sex Addicts Anonymous", "voices of recovery", "sobriety",
                   "spirituality", "12 step"]
    playlist_id = os.environ.get("YT_PLAYLIST_ID_EN",
                                 "PL8OsToLnGRr_XtJlzH-Md6QmFkkcpuMSl")

# ── Google API auth ───────────────────────────────────────────
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/youtube",
]

def get_credentials():
    """Load credentials from token file, refreshing if needed."""
    creds = None

    # Try token file first
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    # Try env var (GitHub Actions)
    if not creds and os.environ.get("YT_TOKEN_JSON"):
        token_data = json.loads(os.environ["YT_TOKEN_JSON"])
        with open(TOKEN_FILE, "w") as f:
            json.dump(token_data, f)
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds:
        print("❌ No YouTube credentials found.")
        print("   Run: python3 get_youtube_token.py")
        sys.exit(1)

    # Refresh if expired
    if creds.expired and creds.refresh_token:
        print("🔄 Refreshing YouTube token...")
        creds.refresh(Request())
        TOKEN_FILE.write_text(creds.to_json())

    return creds

# ── Upload ────────────────────────────────────────────────────
print(f"📤 Uploading: {VIDEO_FILE.name}")
print(f"   Title:     {title}")
print(f"   Playlist:  {playlist_id or '(none)'}")
print()

creds   = get_credentials()
youtube = build("youtube", "v3", credentials=creds)

# Upload video
body = {
    "snippet": {
        "title":       title,
        "description": description,
        "tags":        tags,
        "categoryId":  "22",  # People & Blogs
    },
    "status": {
        "privacyStatus":           "public",
        "selfDeclaredMadeForKids": False,
    },
}

media = MediaFileUpload(str(VIDEO_FILE), mimetype="video/mp4",
                        resumable=True, chunksize=1024*1024*10)

print("⬆️  Uploading video (this may take a minute)...")
request  = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
response = None
while response is None:
    status, response = request.next_chunk()
    if status:
        pct = int(status.progress() * 100)
        print(f"   {pct}%...", end="\r")

video_id = response["id"]
print(f"\n✅ Uploaded: https://www.youtube.com/watch?v={video_id}")

# Upload thumbnail
if THUMBNAIL_FILE:
    print("🖼️  Uploading thumbnail...")
    # Compress thumbnail if needed (YouTube max 2MB)
    thumb_path = str(THUMBNAIL_FILE)
    thumb_size = THUMBNAIL_FILE.stat().st_size
    if thumb_size > 1.9 * 1024 * 1024:
        from PIL import Image
        import tempfile
        img = Image.open(THUMBNAIL_FILE).convert("RGB")
        img = img.resize((1280, 720), Image.LANCZOS)
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(tmp.name, "JPEG", quality=85)
        thumb_path = tmp.name
        print(f"   (Compressed from {thumb_size//1024}KB)")

    youtube.thumbnails().set(
        videoId=video_id,
        media_body=MediaFileUpload(thumb_path, mimetype="image/jpeg")
    ).execute()
    print("   ✅ Thumbnail set")

# Add to playlist
if playlist_id:
    print(f"📋 Adding to playlist {playlist_id}...")
    youtube.playlistItems().insert(
        part="snippet",
        body={
            "snippet": {
                "playlistId": playlist_id,
                "resourceId": {
                    "kind":    "youtube#video",
                    "videoId": video_id,
                },
            }
        }
    ).execute()
    print("   ✅ Added to playlist")

print()
print(f"🎉 Done! Watch at: https://www.youtube.com/watch?v={video_id}")
