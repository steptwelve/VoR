"""
get_youtube_token.py - One-time OAuth flow to generate youtube_token.json
Run this once locally. It will open a browser for you to authorize.
"""
from google_auth_oauthlib.flow import InstalledAppFlow
from pathlib import Path
import json

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/youtube",
]
CLIENT_SECRETS = Path(__file__).parent / "youtube_client_secrets.json"
TOKEN_FILE = Path(__file__).parent / "youtube_token.json"

flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS), SCOPES)
creds = flow.run_local_server(port=0)

TOKEN_FILE.write_text(creds.to_json())
print(f"\n✅ Token saved to: {TOKEN_FILE}")
print("\nCopy the contents below into GitHub Secret YT_TOKEN_JSON:\n")
print(TOKEN_FILE.read_text())
