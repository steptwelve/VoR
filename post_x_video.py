#!/usr/bin/env python3
"""
post_x_video.py — Post today's meditation to X with the YouTube Short as a
native video (via Buffer), falling back to the usual image post.

Runs as its own workflow step after the Short is built, when the repo
variable X_VIDEO == 'true'. In that mode daily_poster.py is run with
DEFER_X=1, so it posts to Bluesky only and writes "x_twitter: deferred" in
meditations/MM-DD.posted. This script only acts when it sees "deferred",
so re-runs never double-post.

Why native video: X shows native video to far more people than link posts.

Flow:
  1. If output/MM-DD-SHORT.mp4 exists and is <= 140s (X's limit without
     Premium), upload it to a GitHub release ("x-media") to get a public URL
     (Buffer only accepts media by URL), then post text + video.
  2. Otherwise — or if Buffer reports the video post failed — post the
     usual text + image (imgbb), exactly like daily_poster.py does.
  3. Update the .posted flag with the outcome; exit 1 if nothing posted.

Usage:  python post_x_video.py [MM-DD]
Env:    BUFFER_API_KEY, IMGBB_API_KEY, GH_TOKEN (for the release upload),
        GITHUB_REPOSITORY (set automatically in Actions)
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

import daily_poster as dp

X_VIDEO_MAX_SECONDS = 140          # X limit for non-Premium accounts
RELEASE_TAG = "x-media"            # rolling release that hosts the MP4s
KEEP_DAYS = 7                      # prune older release assets
BUFFER_URL = "https://api.buffer.com"


def log(msg):
    print(msg, flush=True)


def video_duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def gh(*args, check=True):
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=check)


def host_on_release(path: Path) -> str:
    """Upload the MP4 to the rolling 'x-media' release; return its public URL."""
    repo = os.environ["GITHUB_REPOSITORY"]
    if gh("release", "view", RELEASE_TAG, "-R", repo, check=False).returncode != 0:
        gh("release", "create", RELEASE_TAG, "-R", repo, "--latest=false",
           "--title", "X media (automated)",
           "--notes", "Daily Short MP4s hosted for Buffer/X. Pruned after "
                      f"{KEEP_DAYS} days by post_x_video.py.")
    gh("release", "upload", RELEASE_TAG, str(path), "--clobber", "-R", repo)
    prune_release(repo)
    return f"https://github.com/{repo}/releases/download/{RELEASE_TAG}/{path.name}"


def prune_release(repo: str):
    try:
        r = gh("api", f"repos/{repo}/releases/tags/{RELEASE_TAG}")
        cutoff = datetime.now(timezone.utc) - timedelta(days=KEEP_DAYS)
        for a in json.loads(r.stdout).get("assets", []):
            created = datetime.fromisoformat(a["created_at"].replace("Z", "+00:00"))
            if created < cutoff:
                gh("api", "-X", "DELETE", f"repos/{repo}/releases/assets/{a['id']}",
                   check=False)
                log(f"   pruned old asset {a['name']}")
    except Exception as e:  # pruning is housekeeping; never fail the post
        log(f"   (prune skipped: {e})")


def buffer(query: str) -> dict:
    resp = requests.post(BUFFER_URL, timeout=60, json={"query": query},
                         headers={"Content-Type": "application/json",
                                  "Authorization": f"Bearer {dp.BUFFER_API_KEY}"})
    resp.raise_for_status()
    data = resp.json()
    if data.get("errors"):
        raise RuntimeError(data["errors"])
    return data["data"]


def create_post(text: str, asset: str) -> str:
    """asset is a GraphQL AssetInput literal, e.g. '{ video: { url: "..." } }'."""
    data = buffer(f"""
    mutation {{
      createPost(input: {{
        text: {json.dumps(text)}
        channelId: {json.dumps(dp.BUFFER_X_CHANNEL_ID)}
        schedulingType: automatic
        mode: shareNow
        assets: [{asset}]
      }}) {{
        ... on PostActionSuccess {{ post {{ id }} }}
        ... on MutationError {{ message }}
      }}
    }}""")["createPost"]
    if "message" in data:
        raise RuntimeError(f"Buffer rejected post: {data['message']}")
    return data["post"]["id"]


def wait_for_result(post_id: str, timeout: int = 300) -> tuple[bool, str]:
    """Buffer accepts posts immediately and fails later (media problems show up
    as status=error minutes after). Poll until sent or error."""
    deadline = time.time() + timeout
    status, err = "sending", ""
    while time.time() < deadline:
        p = buffer(f'query {{ post(input: {{ id: {json.dumps(post_id)} }}) '
                   f'{{ status error {{ message }} }} }}')["post"]
        status = p["status"]
        if status == "sent":
            return True, ""
        if status == "error":
            return False, (p.get("error") or {}).get("message", "unknown error")
        time.sleep(15)
    return False, f"still '{status}' after {timeout}s"


def video_text(full_text: str) -> str:
    """Same quote + hashtags as the image post, minus the link: the video is
    the content, and X down-ranks posts with outside links."""
    text = dp.build_x_text(full_text)
    if dp.X_URL and text.endswith(dp.X_URL):
        text = text[: -len(dp.X_URL)].rstrip()
    return text


def main():
    mmdd = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%m-%d")
    txt_path, png_path = dp.get_paths_for(mmdd)
    posted_path = dp.MED_DIR / f"{mmdd}.posted"

    if not posted_path.exists() or "x_twitter: deferred" not in posted_path.read_text():
        log(f"Nothing to do: {posted_path.name} doesn't show X as deferred.")
        return
    full_text = txt_path.read_text(encoding="utf-8")

    result = "failed"
    short = dp.PROJECT_ROOT / "output" / f"{mmdd}-SHORT.mp4"
    if short.exists():
        dur = video_duration(short)
        if dur <= X_VIDEO_MAX_SECONDS:
            try:
                url = host_on_release(short)
                log(f"📤 Hosted Short: {url}")
                pid = create_post(video_text(full_text),
                                  f"{{ video: {{ url: {json.dumps(url)} }} }}")
                ok, err = wait_for_result(pid)
                if ok:
                    result = "ok-video"
                    log("✅ X video post sent")
                else:
                    log(f"⚠️  X video post failed in Buffer ({err}); "
                        f"deleting it and falling back to image")
                    buffer(f'mutation {{ deletePost(input: {{ id: {json.dumps(pid)} }}) '
                           f'{{ ... on DeletePostSuccess {{ id }} }} }}')
            except Exception as e:
                log(f"⚠️  Video path failed ({e}); falling back to image")
        else:
            log(f"ℹ️  Short is {dur:.0f}s (> {X_VIDEO_MAX_SECONDS}s X limit) — image post instead")
    else:
        log("ℹ️  No Short today — image post instead")

    if result == "failed":
        res = dp.post_to_x(dp.build_x_text(full_text),
                           png_path if png_path.exists() else None)
        result = "ok-image" if res else "failed"

    content = posted_path.read_text().replace(
        "x_twitter: deferred", f"x_twitter: {'ok' if result != 'failed' else 'failed'}")
    posted_path.write_text(content.rstrip("\n") + f"\nx_mode: {result}\n")
    log(f"X result: {result}")
    if result == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
