#!/usr/bin/env python3
"""
weekly_stats.py — Weekly reach summary for @StepTwelveSAA, sent by Pushover.

Compares this week's YouTube numbers with last week's snapshot
(stats/youtube-snapshot.json, committed by the workflow), so it reports
views *gained this week*, split into Shorts / English / Spanish videos,
plus subscriber and Bluesky follower changes.

Uses the same YouTube token as the uploader (YT_TOKEN_JSON or
youtube_token.json); costs a handful of API quota units per run.

Usage:
    python weekly_stats.py            # build summary, send Pushover, save snapshot
    python weekly_stats.py --dry-run  # print summary only (no Pushover, no save)
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

DAILY_DIR = Path(__file__).resolve().parent
TOKEN_FILE = DAILY_DIR / "youtube_token.json"
SNAPSHOT = DAILY_DIR / "stats" / "youtube-snapshot.json"
LOOKBACK_DAYS = 120          # videos older than this rarely move; skip them
BSKY_HANDLE = "ocisaa.org"
DRY_RUN = "--dry-run" in sys.argv
SCOPES = ["https://www.googleapis.com/auth/youtube.upload",
          "https://www.googleapis.com/auth/youtube.force-ssl",
          "https://www.googleapis.com/auth/youtube"]


def youtube():
    if not TOKEN_FILE.exists() and os.environ.get("YT_TOKEN_JSON"):
        TOKEN_FILE.write_text(os.environ["YT_TOKEN_JSON"])
    creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        if not os.environ.get("CI"):
            TOKEN_FILE.write_text(creds.to_json())
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def kind(title: str) -> str:
    if "#shorts" in title.lower():
        return "shorts"
    if title.startswith("Voces"):
        return "spanish"
    return "english"


def fetch(yt):
    ch = yt.channels().list(part="statistics,contentDetails", mine=True).execute()["items"][0]
    uploads = ch["contentDetails"]["relatedPlaylists"]["uploads"]
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)

    ids, page = [], None
    while True:
        r = yt.playlistItems().list(part="contentDetails", playlistId=uploads,
                                    maxResults=50, pageToken=page).execute()
        done = False
        for it in r["items"]:
            pub = datetime.fromisoformat(
                it["contentDetails"]["videoPublishedAt"].replace("Z", "+00:00"))
            if pub < cutoff:
                done = True
                break
            ids.append(it["contentDetails"]["videoId"])
        page = r.get("nextPageToken")
        if done or not page:
            break

    videos = {}
    for i in range(0, len(ids), 50):
        r = yt.videos().list(part="snippet,statistics", id=",".join(ids[i:i + 50])).execute()
        for v in r["items"]:
            videos[v["id"]] = {
                "title": v["snippet"]["title"],
                "published": v["snippet"]["publishedAt"],
                "views": int(v["statistics"].get("viewCount", 0)),
                "likes": int(v["statistics"].get("likeCount", 0)),
            }
    return {
        "taken": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "subscribers": int(ch["statistics"].get("subscriberCount", 0)),
        "channel_views": int(ch["statistics"].get("viewCount", 0)),
        "videos": videos,
    }


def bluesky_followers():
    try:
        r = requests.get("https://public.api.bsky.app/xrpc/app.bsky.actor.getProfile",
                         params={"actor": BSKY_HANDLE}, timeout=20)
        return r.json().get("followersCount")
    except Exception:
        return None


def summarize(now, prev):
    pv = (prev or {}).get("videos", {})
    groups = {"shorts": [0, 0], "english": [0, 0], "spanish": [0, 0]}  # [views gained, count]
    best = None
    for vid, v in now["videos"].items():
        gained = v["views"] - pv.get(vid, {}).get("views", 0)
        g = groups[kind(v["title"])]
        g[0] += gained
        g[1] += 1
        if best is None or gained > best[0]:
            best = (gained, v["title"])

    since = (datetime.fromisoformat(prev["taken"]).strftime("%b %-d")
             if prev else "first run")
    lines = [f"Week since {since}" if prev else "First report (totals so far)"]
    labels = {"shorts": "Shorts", "english": "English videos", "spanish": "Spanish videos"}
    for k in ("shorts", "english", "spanish"):
        gained, n = groups[k]
        if n:
            lines.append(f"{labels[k]}: +{gained} views ({n} video{'s' if n != 1 else ''})")
    sub_now, sub_prev = now["subscribers"], (prev or {}).get("subscribers")
    lines.append(f"Subscribers: {sub_now}" +
                 (f" ({sub_now - sub_prev:+d})" if sub_prev is not None else ""))
    bf, bf_prev = now.get("bluesky_followers"), (prev or {}).get("bluesky_followers")
    if bf is not None:
        lines.append(f"Bluesky followers: {bf}" +
                     (f" ({bf - bf_prev:+d})" if bf_prev is not None else ""))
    if best and best[0] > 0:
        lines.append(f"Top: {best[1]} (+{best[0]})")
    return "\n".join(lines)


def pushover(title, msg):
    token, user = os.environ.get("PUSHOVER_APP_TOKEN"), os.environ.get("PUSHOVER_USER_KEY")
    if not (token and user):
        print("(Pushover not configured; message not sent)")
        return
    requests.post("https://api.pushover.net/1/messages.json", timeout=20, data={
        "token": token, "user": user, "title": title, "message": msg, "priority": -1})


def main():
    prev = json.loads(SNAPSHOT.read_text()) if SNAPSHOT.exists() else None
    now = fetch(youtube())
    now["bluesky_followers"] = bluesky_followers()
    msg = summarize(now, prev)
    print(msg)
    if DRY_RUN:
        return
    pushover("VoR weekly reach", msg)
    SNAPSHOT.parent.mkdir(exist_ok=True)
    SNAPSHOT.write_text(json.dumps(now, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
