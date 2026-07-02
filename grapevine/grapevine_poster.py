#!/usr/bin/env python3
"""
================================================================================
grapevine_poster.py
================================================================================
Purpose:
    Fetches the daily AA Grapevine Daily Quote from Gmail and posts it to
    Bluesky (@ocisaa.org) and optionally X (@ocisaa) for the SAA recovery
    community. After successful posting, moves the email to Trash and writes
    a guard file so the reboot script knows today's quote was posted.

Version:    1.4
Created:    2026-06-23
Author:     Jackson Shaw (steptwelve@icloud.com) with Claude (Anthropic)

How it runs:
    Cron job on lsm (ZimaBlade), daily at 9:00 AM ET:
    0 9 * * * /usr/bin/python3 /home/jackson/grapevine/grapevine_poster.py

    Reboot guard (via grapevine_reboot.sh):
    @reboot /home/jackson/grapevine/grapevine_reboot.sh

Usage:
    python3 grapevine_poster.py           # full run — fetch, parse, post, trash
    python3 grapevine_poster.py --dry-run # fetch and parse only, no posting or trashing

Secrets (stored in /home/jackson/.secrets/):
    gmail_token.json        — Gmail OAuth token (jackson.shaw@gmail.com)
    gmail_credentials.json  — Gmail OAuth client credentials (Google Cloud / SARP project)
    grapevine.env           — X and Bluesky credentials
    pushover_grapevine.json — Pushover notification credentials (Daily Meditation app)

Notifications:
    Pushover silent (-1) on success, normal (0) on any failure.
    Notification always shows per-platform success/failure status.

Character limit strategy:
    X:       280 chars
    Bluesky: 300 chars
    Fallback order:
      1. Full attribution: "Quote" — Location, Date, "Article," Publication
      2. Publication only: "Quote" — Publication
      3. Trim quote to fit with publication only

Platform failure policy:
    Each platform (X, Bluesky) is posted independently. A failure on one
    platform does not prevent posting to the other. All errors are reported
    in the Pushover notification and log.

Email retry policy:
    If the Grapevine email is not found, retries every 30 minutes for up
    to 3 hours. If still not found by noon ET, errors out with Pushover.

Email cleanup policy:
    After successful posting to at least one platform, the email is moved
    to Trash. Gmail auto-purges Trash after 30 days.
    If all platforms fail, the email is NOT trashed.

Guard file policy:
    After successful posting, writes /home/jackson/grapevine/.posted_YYYY-MM-DD.
    The reboot script checks for this file to avoid duplicate posts after
    a power failure or reboot.

Platform toggles:
    POST_TO_X       — X/Twitter posting (disabled: API requires paid tier)
    POST_TO_BLUESKY — Bluesky posting (enabled)

Related projects:
    VoR Daily Pipeline — steptwelve/Daily (GitHub Actions, runs 1AM PT)
    Grapevine script   — steptwelve/Daily/grapevine/grapevine_poster.py

Revision history:
    1.0  2026-06-23  Initial version. Gmail → parse → post to X + Bluesky + Pushover.
    1.1  2026-06-24  Each platform posted independently — one failure no longer
                     blocks the other. Success Pushover suppressed if any platform fails.
    1.2  2026-06-24  Added X toggle (POST_TO_X). Retry loop for missing email
                     (every 30 min, up to 3 hours). Improved Pushover notification
                     shows per-platform success/failure status.
    1.3  2026-06-25  Auto-trash email after successful posting to at least one platform.
                     Switch to pushover_grapevine.json (Daily Meditation app token).
    1.4  2026-07-02  Write guard file (.posted_YYYY-MM-DD) after successful post so
                     grapevine_reboot.sh can detect missed runs after power failure.
================================================================================
"""

import sys
import json
import base64
import re
import time
import datetime
import requests
from pathlib import Path

# Google Gmail API
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Bluesky
from atproto import Client as BskyClient

# X / Twitter
import tweepy

# ── Platform toggles ──────────────────────────────────────────────────────────

POST_TO_X       = False  # X API requires paid tier — set True to re-enable
POST_TO_BLUESKY = True

# ── Config ────────────────────────────────────────────────────────────────────

DRY_RUN = "--dry-run" in sys.argv

SECRETS_DIR   = Path("/home/jackson/.secrets")
GMAIL_TOKEN   = SECRETS_DIR / "gmail_token.json"
GMAIL_CREDS   = SECRETS_DIR / "gmail_credentials.json"
PUSHOVER_JSON = SECRETS_DIR / "pushover_grapevine.json"
ENV_FILE      = SECRETS_DIR / "grapevine.env"
GRAPEVINE_DIR = Path("/home/jackson/grapevine")

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
]

# Character limits
X_LIMIT   = 280
BSK_LIMIT = 300

HASHTAGS = "\n\n#Recovery #AAGrapevine #Sobriety #recoveryposse"

# Retry config
RETRY_INTERVAL_SECS = 30 * 60   # 30 minutes
MAX_RETRIES         = 6         # 6 retries = 3 hours total (9am → noon)

# ── Load environment ──────────────────────────────────────────────────────────

def load_env(path):
    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env

# ── Gmail ─────────────────────────────────────────────────────────────────────

def get_gmail_service():
    creds = Credentials.from_authorized_user_file(str(GMAIL_TOKEN), GMAIL_SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(GMAIL_TOKEN, "w") as f:
            f.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)

def fetch_quote_email(service):
    today = datetime.date.today().strftime("%Y/%m/%d")
    query = f'subject:"Grapevine Daily Quote" after:{today}'
    result = service.users().messages().list(userId="me", q=query, maxResults=1).execute()
    messages = result.get("messages", [])
    if not messages:
        return None
    msg = service.users().messages().get(userId="me", id=messages[0]["id"], format="full").execute()
    return msg

def fetch_with_retry(service):
    """Retry fetching the email every 30 minutes for up to 3 hours."""
    for attempt in range(MAX_RETRIES + 1):
        msg = fetch_quote_email(service)
        if msg:
            return msg
        if attempt < MAX_RETRIES:
            next_try = datetime.datetime.now() + datetime.timedelta(seconds=RETRY_INTERVAL_SECS)
            print(f"Email not found. Retrying at {next_try.strftime('%H:%M')}...")
            time.sleep(RETRY_INTERVAL_SECS)
    raise RuntimeError("Grapevine email not found after 3 hours — giving up.")

def extract_text_from_email(msg):
    payload = msg["payload"]
    def get_body(part):
        if part.get("mimeType") == "text/plain":
            data = part.get("body", {}).get("data", "")
            return base64.urlsafe_b64decode(data).decode("utf-8")
        for sub in part.get("parts", []):
            result = get_body(sub)
            if result:
                return result
        return ""
    return get_body(payload)

def trash_email(service, msg_id):
    """Move the email to Trash after successful posting."""
    try:
        service.users().messages().trash(userId="me", id=msg_id).execute()
        print("🗑️  Email moved to Trash")
    except Exception as e:
        print(f"Warning: could not trash email: {e}")

def write_guard_file():
    """Write a guard file so the reboot script knows today's quote was posted."""
    try:
        guard = GRAPEVINE_DIR / f".posted_{datetime.date.today().strftime('%Y-%m-%d')}"
        guard.touch()
        print(f"🗓️  Guard file written: {guard}")
    except Exception as e:
        print(f"Warning: could not write guard file: {e}")

def parse_quote(text):
    """
    Parse quote and attribution from the email body.
    The plain text body arrives as a single long line.
    """
    quote_match = re.search(r'[\u201c\u201d""](.+?)[\u201d""]', text)
    if not quote_match:
        return "", ""
    quote = quote_match.group(1).strip()
    after_quote = text[quote_match.end():].strip()
    attr_match = re.match(r'(.+?)\s{2,}', after_quote)
    attribution = attr_match.group(1).strip() if attr_match else after_quote.split("\n")[0].strip()
    return quote, attribution

# ── Post formatting ───────────────────────────────────────────────────────────

def build_post(quote, attribution, limit):
    """Build post text with character-limit-aware fallback."""
    full = f'"{quote}"\n\u2014 {attribution}{HASHTAGS}'
    if len(full) <= limit:
        return full
    parts = [p.strip() for p in attribution.split(",")]
    publication = parts[-1] if parts else attribution
    fallback = f'"{quote}"\n\u2014 {publication}{HASHTAGS}'
    if len(fallback) <= limit:
        return fallback
    overhead = len(f'"\u2026"\n\u2014 {publication}{HASHTAGS}')
    available = limit - overhead
    trimmed = quote[:available].rsplit(" ", 1)[0] + "\u2026"
    return f'"{trimmed}"\n\u2014 {publication}{HASHTAGS}'

# ── Post to Bluesky ───────────────────────────────────────────────────────────

def post_bluesky(text, env):
    client = BskyClient()
    client.login(env["BSKY_USERNAME"], env["BSKY_APP_PASSWORD"])
    client.send_post(text)
    print("✅ Bluesky: posted")

# ── Post to X ────────────────────────────────────────────────────────────────

def post_x(text, env):
    client = tweepy.Client(
        consumer_key=env["X_API_KEY"],
        consumer_secret=env["X_API_SECRET"],
        access_token=env["X_ACCESS_TOKEN"],
        access_token_secret=env["X_ACCESS_TOKEN_SECRET"]
    )
    client.create_tweet(text=text)
    print("✅ X: posted")

# ── Pushover notification ─────────────────────────────────────────────────────

def notify(title, message, priority=-1):
    try:
        with open(PUSHOVER_JSON) as f:
            po = json.load(f)
        resp = requests.post("https://api.pushover.net/1/messages.json", data={
            "token":    po["PUSHOVER_TOKEN"],
            "user":     po["PUSHOVER_USER"],
            "title":    title,
            "message":  message,
            "priority": priority,
        })
        resp.raise_for_status()
        print("✅ Pushover: sent")
    except Exception as e:
        print(f"Pushover error: {e}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    env = load_env(ENV_FILE)
    today = datetime.date.today().strftime("%B %-d, %Y")

    print(f"Grapevine poster — {today}{' [DRY RUN]' if DRY_RUN else ''}")

    try:
        service = get_gmail_service()
        msg = fetch_with_retry(service)
        msg_id = msg["id"]
        text = extract_text_from_email(msg)
        quote, attribution = parse_quote(text)

        if not quote:
            raise RuntimeError("Could not parse quote from email")

        print(f"\nQuote:       {quote}")
        print(f"Attribution: {attribution}")

        x_post   = build_post(quote, attribution, X_LIMIT)
        bsk_post = build_post(quote, attribution, BSK_LIMIT)

        print(f"\n--- X post ({len(x_post)} chars) ---")
        print(x_post)
        print(f"\n--- Bluesky post ({len(bsk_post)} chars) ---")
        print(bsk_post)

        if DRY_RUN:
            print("\n[DRY RUN] Skipping posting and trashing.")
            return

        # Post to each platform independently
        results = []
        errors  = []

        if POST_TO_X:
            try:
                post_x(x_post, env)
                results.append("✅ X: posted")
            except Exception as e:
                err = f"❌ X: {e}"
                print(err)
                errors.append(err)
        else:
            print("⏭️  X: skipped (POST_TO_X=False)")

        if POST_TO_BLUESKY:
            try:
                post_bluesky(bsk_post, env)
                results.append("✅ Bluesky: posted")
            except Exception as e:
                err = f"❌ Bluesky: {e}"
                print(err)
                errors.append(err)
        else:
            print("⏭️  Bluesky: skipped (POST_TO_BLUESKY=False)")

        # Trash email and write guard file if at least one platform succeeded
        if results:
            trash_email(service, msg_id)
            write_guard_file()
        else:
            print("⚠️  No platforms succeeded — email NOT trashed, guard file NOT written")

        # Build and send notification
        status_lines = results + errors
        status = "\n".join(status_lines)
        has_errors = len(errors) > 0

        notify(
            title=f"Grapevine — {today}{' ⚠️' if has_errors else ''}",
            message=f'"{quote[:80]}..."\n\u2014 {attribution}\n\n{status}',
            priority=0 if has_errors else -1
        )

        print("\nDone!")

    except Exception as e:
        print(f"\nERROR: {e}")
        notify(title="Grapevine FAILED", message=str(e), priority=0)
        raise

if __name__ == "__main__":
    main()
