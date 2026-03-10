"""
download_background.py

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

    video_id, idx = get_next_video_id(ids)
    print(f"[download_background] Video {idx + 1} of {len(ids)}: Pexels ID {video_id}")

    url, author = get_download_url(video_id)
    print(f"[download_background] Author: {author}")
    download_video(url, args.output)
    print(f"[download_background] Ready: {args.output}")


if __name__ == "__main__":
    main()
