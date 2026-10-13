"""
get_youtube_token.py - One-time OAuth flow to generate a YouTube token.

    python3 get_youtube_token.py           # @StepTwelveSAA -> youtube_token.json
    python3 get_youtube_token.py scisaa    # SCISAA channel -> youtube_token_scisaa.json

Run locally. A browser opens: pick the Google account (and channel) that
owns the target channel, then authorize. The script confirms which channel
the token belongs to before you copy it into the GitHub Secret.
"""
import json
import sys
import urllib.request
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/youtube",
]
BRAND = sys.argv[1].lower() if len(sys.argv) > 1 else "vor"
TARGETS = {
    "vor":    ("youtube_token.json",        "YT_TOKEN_JSON",        "@StepTwelveSAA"),
    "scisaa": ("youtube_token_scisaa.json", "YT_TOKEN_JSON_SCISAA", "SCISAA (@SCISAA)"),
}
if BRAND not in TARGETS:
    sys.exit(f"Unknown channel '{BRAND}'. Use one of: {', '.join(TARGETS)}")
token_name, secret_name, channel_label = TARGETS[BRAND]

CLIENT_SECRETS = Path(__file__).parent / "youtube_client_secrets.json"
TOKEN_FILE = Path(__file__).parent / token_name

print(f"Authorizing for {channel_label}.")
print("In the browser, choose the Google account/channel that owns it.\n")
flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS), SCOPES)
# select_account: always show the account picker, so the right login is used
creds = flow.run_local_server(port=0, prompt="select_account consent")

# Confirm which channel this token can upload to (plain HTTPS call, so this
# script only needs google_auth_oauthlib)
req = urllib.request.Request(
    "https://www.googleapis.com/youtube/v3/channels?part=snippet&mine=true",
    headers={"Authorization": f"Bearer {creds.token}"})
with urllib.request.urlopen(req, timeout=30) as resp:
    items = json.load(resp).get("items", [])
if not items:
    sys.exit("❌ This Google account has no YouTube channel. Token NOT saved.")
ch = items[0]
handle = ch["snippet"].get("customUrl", "")
print(f"\n🔑 Token is for channel: {ch['snippet']['title']} {handle}  (id {ch['id']})")
if input(f"Is that {channel_label}? Save it? [y/N] ").strip().lower() != "y":
    sys.exit("Not saved. Run again and pick the other account/channel.")

TOKEN_FILE.write_text(creds.to_json())
print(f"\n✅ Token saved to: {TOKEN_FILE}")
print(f"\nPut it in GitHub Secret {secret_name}, e.g.:")
print(f"   gh secret set {secret_name} < {TOKEN_FILE.name}")
