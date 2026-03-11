"""
download_background.py

Version: 2026-03-11.02

Changes in .02 (2026-03-11):
- When a Pexels video ID returns 404 and a fallback is used, writes a
  warning message to /tmp/pexels_warning.txt. The GitHub Actions workflow
  checks for this file after the download step and sends a quiet Pushover
  notification (priority -1) so Jackson is informed without alarm.
- No change to exit behavior: script still exits 0 on any successful
  download, exits 1 only if ALL IDs in backgrounds.txt fail.

Version: 2026-03-11.01 (original)

Downloads the next background video from Pexels for the daily build.
Uses backgrounds.txt for the video ID list and .vor-state.json for rotation state.

Usage:
  python download_background.py --output /tmp/background.mp4

Requires:
  PEXELS_API_KEY in environment or .env file
"""

import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
STATE_FILE   = PROJECT_ROOT / ".vor-state.json"
BACKGROUNDS  = PROJECT_ROOT / "backgrounds.txt"

# File written when a 404 fallback occurs; checked by workflow for Pushover warning
PEXELS_WARNING_FILE = Path("/tmp/pexels_warning.txt")

load_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.getenv("PEXELS_API_KEY")


def load_video_ids():
    ids = []
    for line in BACKGROUNDS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        vid_id = parts[0].strip()
        if vid_id.upper() == "PENDING":
            continue
        try:
            ids.append(int(vid_id))
        except ValueError:
            print(f"[download_background] Skipping invalid ID: {vid_id}")
    return ids


def get_next_video_id(ids):
    state = {}
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text())
        except Exception:
            state = {}

    last_idx = state.get("last_video_idx", -1)
    next_idx = (last_idx + 1) % len(ids)
    state["last_video_idx"] = next_idx
    STATE_FILE.write_text(json.dumps(state, indent=2))
    return ids[next_idx], next_idx


def get_download_url(video_id):
    headers = {"Authorization": API_KEY}
    resp = requests.get(
        f"https://api.pexels.com/videos/videos/{video_id}",
        headers=headers,
        timeout=30
    )
    if resp.status_code == 404:
        return None, None
    resp.raise_for_status()
    data = resp.json()

    files = data.get("video_files", [])
    hd = next((f for f in files if f.get("width") == 1920 and f.get("height") == 1080), None)
    if not hd:
        hd = next((f for f in files if f.get("height") == 1080), None)
    if not hd:
        hd = sorted(files, key=lambda f: f.get("width", 0) * f.get("height", 0), reverse=True)[0]

    return hd["link"], data.get("user", {}).get("name", "Pexels")


def download_video(url, output_path):
    print(f"[download_background] Downloading from Pexels...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = int(downloaded / total * 100)
                print(f"  ⬇️  {pct}%", end="\r")

    print(f"\n  ✅ Downloaded: {output_path} ({downloaded / 1024 / 1024:.1f} MB)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/background.mp4")
    args = parser.parse_args()

    if not API_KEY:
        print("❌ PEXELS_API_KEY not set")
        sys.exit(1)

    ids = load_video_ids()
    if not ids:
        print("❌ No valid video IDs in backgrounds.txt")
        sys.exit(1)

    skipped_ids = []

    # Try IDs in rotation order, skip any that return 404
    for attempt in range(len(ids)):
        video_id, idx = get_next_video_id(ids)
        print(f"[download_background] Video {idx + 1} of {len(ids)}: Pexels ID {video_id}")
        url, author = get_download_url(video_id)
        if url is None:
            print(f"  ⚠️  ID {video_id} not found on Pexels (404), trying next...")
            skipped_ids.append(video_id)
            continue

        # If we had to skip any IDs to get here, write a warning file so the
        # workflow can send a quiet Pushover notification to Jackson.
        if skipped_ids:
            skipped_str = ", ".join(str(i) for i in skipped_ids)
            warning_msg = (
                f"Pexels 404 on {len(skipped_ids)} video ID(s): {skipped_str}. "
                f"Used ID {video_id} instead. "
                f"Consider removing stale IDs from backgrounds.txt."
            )
            PEXELS_WARNING_FILE.write_text(warning_msg)
            print(f"  ⚠️  Warning written to {PEXELS_WARNING_FILE}")

        print(f"[download_background] Author: {author}")
        download_video(url, args.output)
        print(f"[download_background] Ready: {args.output}")
        sys.exit(0)

    print("❌ All video IDs failed. Check backgrounds.txt for stale IDs.")
    sys.exit(1)


if __name__ == "__main__":
    main()
