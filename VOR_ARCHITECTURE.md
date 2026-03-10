# Voices of Recovery — Daily Meditation Automation
## Architecture, History & Carry-Forward Guide

**Project:** @StepTwelveSAA YouTube Channel  
**Owner:** Jackson Shaw (jackson.shaw@icloud.com)  
**Started:** February 2026  
**Purpose:** Fully automated daily meditation video production and publishing for the SAA recovery community. Built as a proof of concept for potential adoption by the SAA International Service Organization (ISO).

---

## What This System Does

Every day, automatically and without human intervention:

1. Scrapes today's meditation text from the SAA website
2. Posts the meditation image to Bluesky and Twitter/X
3. Downloads a background video from Pexels
4. Generates a narrated video using Microsoft Edge TTS
5. Uploads the video to YouTube with metadata, thumbnail, and playlist placement

Spanish translation and video production is fully plumbed but not yet enabled — see config.yml.

---

## Repository Structure

```
Daily/
├── config.yml                  # ← MASTER CONTROL PANEL — read this first
├── vor_config.py               # Python config reader — all scripts import this
├── backgrounds.txt             # Pexels video ID list for background rotation
│
├── daily_poster.py             # Scrapes SAA site, posts to Bluesky/Twitter
├── build-video.sh              # Builds the MP4 from text + background + TTS
├── upload_youtube.py           # Uploads MP4 + thumbnail to YouTube
├── translate_meditation.py     # Translates English text to Spanish via Anthropic API
├── download_background.py      # Downloads next background video from Pexels
├── get_youtube_token.py        # One-time OAuth flow for YouTube credentials
│
├── meditations/                # Cached meditation text files (MM-DD.txt)
│   └── MM-DD-es.txt            # Spanish translations (when produced)
├── backgrounds/                # Local background videos (NOT in git — too large)
├── output/                     # Built MP4s and PNGs (NOT in git — too large)
├── templates/                  # End card narration text
├── preview_endcard.py          # Quickly preview/iterate on endcard design
├── make_endcard.py             # Legacy static endcard generator (kept for reference)
│
├── .github/workflows/
│   └── daily-poster.yml        # GitHub Actions — runs everything daily at 1AM PT
│
├── requirements.txt            # Python dependencies
└── .env                        # Local secrets (never committed)
```

---

## Master Configuration — config.yml

This is the single file you edit to control what the system does. No code changes needed for on/off decisions.

```yaml
video:
  enabled: true       # false = Bluesky/Twitter only, no video production

english:
  enabled: true       # English video production and YouTube upload

spanish:
  enabled: false      # true = translate, build Spanish video, upload
  same_channel: false # true = same YouTube channel, different playlist
                      # false = separate Spanish channel (needs YT_TOKEN_JSON_ES)
  translation:
    provider: anthropic   # or: deepl
    dialect: neutral-international
```

**To pause video production:** set `video.enabled: false`  
**To enable Spanish:** set `spanish.enabled: true` and fill in playlist/channel IDs

---

## Daily Workflow (GitHub Actions)

Runs at **1:00 AM Pacific** every day via `.github/workflows/daily-poster.yml`

```
Step 1:  Scrape meditation text → post to Bluesky + Twitter
Step 2:  Read config.yml flags
Step 3:  Install ffmpeg + edge-tts (if video enabled)
Step 4:  Download background video from Pexels (if video enabled)
Step 5:  Build English MP4 (if video enabled)
Step 6:  Upload English video to YouTube (if video enabled)
Step 7:  Translate to Spanish (if spanish enabled)
Step 8:  Build Spanish MP4 (if spanish enabled)
Step 9:  Upload Spanish video to YouTube (if spanish enabled)
Step 10: Commit new meditation text files to repo
```

Each step is gated by config flags — steps are skipped cleanly if disabled.

---

## GitHub Secrets Required

Set these in: GitHub → Daily repo → Settings → Secrets and variables → Actions

| Secret | Description | Status |
|--------|-------------|--------|
| `BSKY_USERNAME` | Bluesky username | ✅ Set |
| `BSKY_APP_PASSWORD` | Bluesky app password | ✅ Set |
| `X_API_KEY` | Twitter/X API key | ✅ Set |
| `X_API_SECRET` | Twitter/X API secret | ✅ Set |
| `X_ACCESS_TOKEN` | Twitter/X access token | ✅ Set |
| `X_ACCESS_TOKEN_SECRET` | Twitter/X access token secret | ✅ Set |
| `YT_TOKEN_JSON` | YouTube OAuth token (English channel) | ✅ Set |
| `PEXELS_API_KEY` | Pexels API key for background videos | ⚠️ Needs adding |
| `ANTHROPIC_API_KEY` | Anthropic API key for Spanish translation | ⚠️ Needs adding |
| `YT_TOKEN_JSON_ES` | YouTube OAuth token (Spanish channel) | ⏳ When Spanish enabled |

---

## YouTube Token — Critical 7-Day Expiry

**This is the most important operational concern.**

The YouTube OAuth token expires every 7 days and must be manually refreshed. If not refreshed, video uploads fail with `invalid_grant` error.

**To refresh:**
```bash
cd ~/Documents/GitHub/Daily
python3 get_youtube_token.py
# Browser opens → log in → authorize → token saved to youtube_token.json
```

Then update the GitHub Secret:
```bash
cat youtube_token.json | pbcopy
# Go to GitHub → Secrets → YT_TOKEN_JSON → paste
```

**Set a weekly reminder** (Sunday morning) to do this.

---

## Background Videos — Pexels Rotation

Background videos are NOT stored in the repo (too large — 2.6GB+). Instead:

- `backgrounds.txt` contains Pexels video IDs
- `download_background.py` downloads the next video in rotation at build time
- `.vor-state.json` tracks rotation state (last voice index, last video index)
- Requires `PEXELS_API_KEY` GitHub Secret

**To add a new background video:**
1. Find a video on Pexels (landscape, 1080p, 90+ seconds)
2. Get the numeric video ID from the URL
3. Add a line to `backgrounds.txt`: `12345678 | description`
4. Commit and push

**Personal videos (IMG_0521, IMG_0524, IMG_0505):**
These 3 videos were shot by Jackson and need to be uploaded to Pexels.
Once uploaded, replace `PENDING` lines in `backgrounds.txt` with real IDs.

---

## TTS Voices

**English voices** (rotate daily via `.vor-state.json`):
- en-US-GuyNeural, en-US-JennyNeural, en-US-ChristopherNeural
- en-US-AriaNeural, en-US-EricNeural, en-US-MichelleNeural

**Spanish voices** (for when Spanish is enabled):
- es-MX-JorgeNeural, es-MX-DaliaNeural (Mexico — neutral)
- es-US-AlonsoNeural, es-US-PalomaNeural (US Spanish)
- es-AR-TomasNeural, es-AR-ElenaNeural (Argentina)

Voice rotation is managed by `build-video.sh` via `.vor-state.json`.

---

## Spanish Translation

Translation uses the **Anthropic API** (Claude) to produce neutral international Spanish suitable for a global recovery audience. This was chosen over DeepL for better handling of spiritual/recovery terminology.

The translation prompt instructs Claude to:
- Use neutral Spanish with no regional slang
- Preserve the spiritual tone and first-person voice
- Keep the same document structure (date, quote, source, story, reflection)

**To test translation without enabling Spanish production:**
```bash
python3 translate_meditation.py --date 03-03 --dry-run
```

**Output file:** `meditations/MM-DD-es.txt`

---

## Endcard Design

The endcard is generated dynamically at build time inside `build-video.sh` (Step 6) using Pillow. It does NOT use a static PNG template.

**Background:** `backgrounds/ChatGPT-Image-VoR.png` — the branded forest canopy image. The top ~27% (which contains "Voices of Recovery" in large serif text) is cropped out, leaving only the clean forest background.

**Layout:**
- Semi-transparent dark box over the text area
- "To learn more about SAA, visit:" — font 54, white
- "saa-recovery.org" — font 108, white, bold
- SAA purpose quote — font 40, greenish-white (210, 225, 210)
- Attribution line (Video: Pexels.com | Narration: Microsoft Azure TTS) — font 21, same greenish-white, inside the box

**To preview/iterate on endcard design without rebuilding the full video:**
```bash
python3 preview_endcard.py
```
This generates `templates/endcard-template.png` and opens it in Preview.

---

## Outstanding Work (as of March 2026)

- [ ] Update `build-video.sh` to accept `LANG=es` environment variable and use `-es.txt` source file and Spanish voices
- [ ] Update `upload_youtube.py` to read playlist/channel IDs from `config.yml`
- [ ] Add `PEXELS_API_KEY` and `ANTHROPIC_API_KEY` to GitHub Secrets
- [ ] Upload 3 personal videos (IMG_0521, IMG_0524, IMG_0505) to Pexels, update `backgrounds.txt`
- [ ] Test full end-to-end GitHub Actions run
- [ ] Solve YouTube token 7-day expiry problem for unattended operation
- [ ] When Spanish channel ready: create YouTube channel, get OAuth token, add to GitHub Secrets, fill in `config.yml` playlist/channel IDs, set `spanish.enabled: true`

---

## Local Development

```bash
cd ~/Documents/GitHub/Daily

# Check config
python3 vor_config.py

# Full manual daily run
python3 daily_poster.py
./build-video.sh
python3 upload_youtube.py

# Test Spanish translation (dry run — no file saved)
python3 translate_meditation.py --dry-run

# Refresh YouTube token (do weekly)
python3 get_youtube_token.py
```

---

## Design Decisions & Rationale

**Why GitHub Actions and not a Mac or Pi?**
Fully cloud-hosted — no dependency on any local machine being on. Jackson's Mac and Pi are not suitable for unattended 365-day operation. GitHub Actions is free for public repos and runs reliably on a schedule.

**Why Pexels for backgrounds?**
Free, high quality, commercially licensed. The Pexels API allows programmatic download, making fully automated background rotation possible without storing large video files in the repo.

**Why Anthropic API for Spanish translation?**
Better handling of spiritual and recovery-specific language than word-for-word translation services. Recovery terminology (Higher Power, acting out, surrender) needs sensitivity, not literal translation.

**Why neutral-international Spanish?**
Built with potential ISO adoption in mind. The ISO serves Spanish-speaking SAA members globally. Neutral international Spanish (similar to CNN en Español broadcast standard) is the most broadly understood.

**Why Edge TTS and not ElevenLabs?**
Free, no API key needed, excellent voice quality, wide language support. Voices rotate daily to avoid monotony.

---

## Contact & History

Built by Jackson Shaw with Claude (Anthropic) in February–March 2026.
Conversations archived in Claude.ai (search "VoR YouTube" or "StepTwelveSAA").
For questions about the ISO adoption angle, contact Jackson directly.
