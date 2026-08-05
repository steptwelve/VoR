# VoR / Grapevine Workflow Overview

**Last updated:** 2026-08-05
**Purpose:** A single reference for how the VoR and Grapevine automated posting systems work — what runs where, on what schedule, using what credentials, and how to change or debug any piece of it. Written so someone unfamiliar with the systems (including a future version of Claude) can get oriented quickly.

---

## 1. The Two Systems at a Glance

| | **VoR (Voices of Recovery)** | **Grapevine Daily Quote** |
|---|---|---|
| **What it posts** | Daily SAA meditation (scraped from saa-connect.org) | Daily AA Grapevine quote (from an email) |
| **Where it runs** | GitHub Actions (cloud, ephemeral) | `lsm` server (always-on, cron) |
| **Source repo** | `steptwelve/VoR`, local: `~/Documents/GitHub/Daily/` | Same repo, `grapevine/` subfolder |
| **Schedule** | 1:00 AM Pacific, daily | 9:00 AM Eastern, daily |
| **Platforms** | Bluesky (@ocisaa.org), X (@ocisaa), YouTube (@StepTwelveSAA) | Bluesky (@ocisaa.org), X (@ocisaa) |
| **Human involvement** | None — fully automated | None — fully automated |
| **X posting method** | Buffer API (as of Aug 2026) | Buffer API (as of Aug 2026) |
| **Bluesky posting method** | Direct via atproto | Direct via atproto |

Both systems are fully automated with no manual review step. (An earlier version of the project notes described a manual "check Gmail, compose, review before posting" workflow for Grapevine — that does not reflect what's actually deployed. `grapevine_poster.py` reads Gmail and posts on its own.)

---

## 2. VoR — Daily Meditation Pipeline

### What it does, in order
1. Scrapes today's meditation text from `saa-connect.org`
2. Generates a 1080×1080 branded image (meditation text over a background) with Pillow
3. Posts to **Bluesky** directly via atproto, with the image attached and clickable hashtag/link facets
4. Posts to **X** via **Buffer's GraphQL API** (`createPost` mutation, `mode: shareNow`), with the same image re-hosted on **imgbb** first (Buffer requires a public image URL, not a file upload)
5. If `ENABLE_VIDEO` is on: downloads a Pexels background video, builds a narrated English MP4 (Edge TTS), uploads to YouTube
6. If `ENABLE_SPANISH` is also on: translates the meditation (Anthropic API), builds a Spanish MP4, uploads to a separate YouTube channel
7. Commits new meditation text files and state back to the repo

### Where it runs
**GitHub Actions**, entirely in the cloud — not on Jackson's Mac, not on lsm. A fresh Ubuntu runner spins up nightly, does the work, commits results, and shuts down. This means the pipeline runs whether or not any of Jackson's machines are on.

- Workflow file: `.github/workflows/daily-poster.yml`
- Trigger: `cron: '0 9 * * *'` (9 AM UTC = 1 AM PT) + manual `workflow_dispatch`
- Main script: `daily_poster.py`

### Key files
| File | Purpose |
|---|---|
| `daily_poster.py` | Scrape, build image, post to Bluesky + X |
| `build-video.sh` | Builds MP4 from text + TTS + background video |
| `upload_youtube.py` | Uploads MP4 + thumbnail to YouTube |
| `translate_meditation.py` | English → Spanish translation via Anthropic API |
| `download_background.py` | Downloads next Pexels background video in rotation |
| `.vor-state.json` | Tracks voice/video rotation state, persisted across runs |
| `backgrounds.txt` | Pexels video ID list |
| `config.yml` | Feature flags (mostly superseded by GitHub Actions Variables — see below) |

### Feature toggles
Controlled by **GitHub Actions Variables** (Settings → Secrets and variables → Variables), not `config.yml`:
- `ENABLE_VIDEO` — English video build + YouTube upload
- `ENABLE_SPANISH` — Spanish translation + video (only takes effect if `ENABLE_VIDEO` is also true)

### GitHub Secrets required
| Secret | Used for |
|---|---|
| `BSKY_USERNAME`, `BSKY_APP_PASSWORD` | Bluesky login |
| `BUFFER_API_KEY` | X posting via Buffer |
| `IMGBB_API_KEY` | Image hosting for X posts (Buffer needs a public URL) |
| `PEXELS_API_KEY` | Background video downloads |
| `ANTHROPIC_API_KEY` | Spanish translation |
| `YT_TOKEN_JSON` | YouTube OAuth (English channel) |
| `YT_TOKEN_JSON_ES` | YouTube OAuth (Spanish channel) |
| `PUSHOVER_USER_KEY`, `PUSHOVER_APP_TOKEN` | Failure/success notifications |

### Notifications
Pushover alerts fire on any step failure (priority 0) and once, quietly (priority -1), on a fully successful run. Uses `conclusion` rather than `outcome` for `continue-on-error` steps, since `outcome` always reports success even when the underlying command failed.

---

## 3. Grapevine — Daily Quote Pipeline

### What it does, in order
1. Searches `jackson.shaw@gmail.com` for today's "Grapevine Daily Quote" email (retries every 30 min for up to 3 hours if not yet arrived)
2. Parses the quote and attribution out of the email body
3. Builds platform-specific post text with a no-truncation policy: tries full attribution, falls back to publication name only, and **skips the platform entirely** (rather than truncating the quote) if it still doesn't fit
4. Posts to **Bluesky** directly via atproto
5. Posts to **X** via **Buffer's GraphQL API** (same mechanism as VoR — `createPost`, `mode: shareNow`, no image, this pipeline has always been text-only)
6. On success (at least one platform posted): moves the email to Trash, writes a guard file, sends a quiet Pushover notification
7. On total failure: leaves the email in place (so it can be retried/inspected) and sends a normal-priority Pushover alert

### Where it runs
**`lsm`** (ZimaBlade, Debian home server), via cron — **not** GitHub Actions. This is a genuinely different execution model from VoR: lsm is an always-on physical/local server, not an ephemeral cloud runner.

```
0 9 * * * /usr/bin/python3 /home/jackson/grapevine/grapevine_poster.py >> /home/jackson/grapevine/grapevine.log 2>&1
@reboot /home/jackson/grapevine/grapevine_reboot.sh >> /home/jackson/grapevine/grapevine.log 2>&1
```

The `@reboot` entry handles the case where lsm was powered off at 9 AM: on boot, `grapevine_reboot.sh` checks whether today's guard file already exists, and if not, waits for network connectivity and runs the poster.

### Deployment model (as of 2026-08-05)
Grapevine's code lives in the same `steptwelve/VoR` repo as VoR, under `grapevine/`. On lsm, it's set up as a **git sparse checkout**, not a manual file copy:

- `~/repos/Daily/` — a git clone of `steptwelve/VoR` on lsm, sparse-checked-out to just the `grapevine/` folder
- Authenticated via a **read-only deploy key** (`~/.ssh/deploy_vor_repo`), added to the repo under Settings → Deploy keys, scoped to this repo only, no write access
- `~/grapevine/grapevine_poster.py` and `~/grapevine/grapevine_reboot.sh` are **symlinks** into that clone — cron and the reboot script reference the same paths as always, so no cron changes were needed
- Runtime state (`grapevine.log`, `.posted_YYYY-MM-DD` guard files, `~/.secrets/`) stays outside the git clone entirely — never tracked, never committed

**To deploy an update:** commit + push from the Mac repo as usual, then on lsm:
```bash
cd ~/repos/Daily && git pull
```
That's it — the symlinks mean the running script updates immediately, no service restart needed.

(Prior to 2026-08-05, this was a plain `scp`'d file with no version history on lsm — upgraded to git specifically because that drift caused confusion during debugging.)

### Key files (all under `grapevine/` in the repo, symlinked into `~/grapevine/` on lsm)
| File | Purpose |
|---|---|
| `grapevine_poster.py` | Fetch email, parse, post to Bluesky + X |
| `grapevine_reboot.sh` | Runs at boot; catches up on a missed 9 AM run |

### Secrets (all on lsm, `/home/jackson/.secrets/`, never in git)
| File | Contents |
|---|---|
| `grapevine.env` | `BSKY_USERNAME`, `BSKY_APP_PASSWORD`, `BUFFER_API_KEY` (plus some now-unused X API v1 keys, harmless leftovers) |
| `gmail_token.json` / `gmail_credentials.json` | Gmail OAuth (read + modify scope, for search + trash) |
| `pushover_grapevine.json` | Pushover credentials (Daily Meditation app) |

### Platform toggles
Set directly as constants at the top of `grapevine_poster.py`:
```python
POST_TO_X       = True   # via Buffer, as of v1.6
POST_TO_BLUESKY = True
```

---

## 4. Shared Infrastructure

### The Buffer connection
Both pipelines post to X through the **same Buffer channel** — `ocisaa` (channel ID `678063bf4697c1deff60ae6e`), under the `OCISAA` Buffer organization. Buffer was adopted because X's free-tier API (used previously via `tweepy`) got blocked without a paid API tier; Buffer maintains its own developer relationship with X on the backend, so posting through Buffer doesn't require Jackson's own X API credentials at all — just a Buffer personal API key (`Settings → API` at `publish.buffer.com/settings/api`, distinct from Buffer's "Active Integrations" / OAuth app connections, which is a different thing entirely).

**Important implementation detail:** Buffer's `createPost` mutation requires `mode: shareNow` to publish immediately. `mode: addToQueue` (an easy mistake — it's the more "default-sounding" option) instead parks the post in Buffer's queue for its next scheduled slot, which can be hours later.

### Servers
| Machine | Role |
|---|---|
| **Jackson's Mac** (M1 MacBook Air) | Primary dev machine. Repo lives at `~/Documents/GitHub/Daily/`. Not required for either pipeline to run day-to-day. |
| **GitHub Actions** | Runs VoR nightly. Ephemeral cloud runner. |
| **lsm** (ZimaBlade/Debian) | Runs Grapevine via cron. Always-on home server. Also runs an unrelated Ada Stocks project. SSH: `ssh jackson@lsm` (Tailscale: `lsm.tail10f056.ts.net`), key-based auth already configured. |

### Notifications
Both pipelines use Pushover, but with **separate credential files** — VoR's GitHub Secrets (`PUSHOVER_USER_KEY`/`PUSHOVER_APP_TOKEN`) and Grapevine's `pushover_grapevine.json` on lsm — even though they may point at the same Pushover account/app. Priority convention is consistent across both: `-1` (silent) for success, `0` (normal) for any failure. Priority `1` (wake) is not used anywhere.

---

## 5. Known Quirks & Hard-Won Lessons

- **GitHub Actions `continue-on-error` trap:** steps with `continue-on-error: true` always report `outcome: success`, even when the command actually failed. Only `conclusion` reflects the real result — all failure-notification conditions in `daily-poster.yml` check `conclusion`.
- **`.vor-state.json` / `.posted` file persistence:** these are written mid-run by the Actions runner, but `git checkout -- .` (used to clean the tree before rebasing) will silently discard them if they aren't saved to `/tmp` first and restored after `git pull --rebase`.
- **`build-video.sh` executable bit:** any file-editing tool that writes over this file strips its `+x` bit. Always `chmod +x` (or `git update-index --chmod=+x`) before committing.
- **Buffer `mode` field:** use `shareNow`, not `addToQueue`, for immediate publishing (see §4 above) — a genuine bug hit in production on 2026-08-05, caught and fixed same day.
- **imgbb transient failures:** imgbb's upload API has been observed to return a one-off `500 Internal Server Error` on an otherwise-valid image/key. `upload_to_imgbb()` retries up to 3 times with backoff before falling back to a text-only tweet.
- **imgbb image expiration:** uploads default to *permanent* storage unless an `expiration` parameter is set. VoR sets `expiration: 604800` (7 days) since the URL is only ever needed briefly by Buffer.
- **Grapevine's no-truncation policy:** if a quote doesn't fit even with a shortened attribution, the platform is skipped entirely rather than truncating the quote — this is intentional, not a bug, and shows up in logs as "quote too long — skipping."
- **Bluesky facet byte offsets:** AT Protocol facets (clickable hashtags/links) need UTF-8 *byte* offsets, not Python character offsets — multi-byte characters (curly quotes) shift everything after them if this isn't handled carefully.

---

## 6. Related, Separate Projects (not covered above)
- **SARPodcast** (`steptwelve/SARPodcast`) — monitors a podcast YouTube channel, auto-drafts WordPress posts. Independent of both pipelines above.
- **Ada Stocks** — runs on lsm via its own cron jobs, unrelated to recovery-community posting.
