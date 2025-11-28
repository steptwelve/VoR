"""
daily_poster.py

Version: 2025-11-24.09
Generated: 2025-11-24 07:05:00

Changes in the .09 version:
- Rewrote post_to_x() to use X API v2 (Client) instead of v1.1 (API)
- Free tier X API supports v2 endpoints, fixing 403 Forbidden errors
- Uses tweepy.Client for posting instead of tweepy.API

Version: 2025-11-21.08
Generated: 2025-11-21 14:45:00

Changes in the .08 version:
- Added separate X_URL constant (http://bit.ly/step12-t) for Twitter posts
- Created dedicated build_x_text() function instead of deriving from Bluesky
- Both platforms now use same truncation logic with platform-specific URLs
- Ensures URLs never get cut off on either platform

Changes in the .07 version:
- Rewrote build_bsky_text() truncation logic to preserve all hashtags and URL
- Now truncates quote word-by-word from the end instead of dropping hashtags
- Adds ellipsis (...) to indicate truncated content
- Ensures hashtags and URL always included for discoverability

Version: 2025-11-21.06
Generated: 2025-11-21 9:01am

Changes in the .06 version:
- Added rich text facets to Bluesky posts to make hashtags and URLs clickable
- Hashtags now appear as interactive blue links in Bluesky
- URLs are properly formatted as clickable links

Version: 2025-11-21.05
Generated: 2025-11-21 14:00:00

Changes in the .05 version:
- Enhanced --test mode to generate and display post text previews
- Saves post preview to output/MM-DD-post-preview.txt showing both
  Bluesky and X/Twitter text with character counts
- Displays preview in terminal for immediate review

Version: 2025-11-21.04
Generated: 2025-11-21
Changes in the .04 version:
- Modified build_bsky_text() to use opening quote (parts[1]) instead of 
  page title (parts[0]) for social media posts, providing more meaningful 
  content while maintaining length truncation logic.

Version: 2025-11-21.03
Generated: 2025-11-21
Changes in the .03 version:
- I chose the "b" style (blurred forest with charcoal overlay) as the default
  for the live MM-DD.png image.
Changes in this version:
- Always render a top "Voices of Recovery" title (centered).
- Added spacing between title, date (page h1), and body text.
- Title font set to 36px (off-white #F2F2F2).
- Body font set to 22px (white #FFFFFF).
- Introduced 5 background styles:
    A: Flat charcoal (#111111)
    B: Blurred forest (ChatGPT-Image-VoR.png) with charcoal overlay
    C: Charcoal gradient
    D: Textured charcoal
    E: Semi-transparent charcoal over forest (ChatGPT-Image-VoR.png)
- In --test mode, always regenerate 5 images:
    output/MM-DD-a.png ... output/MM-DD-e.png
- Normal mode behavior (scrape/cache/post + MM-DD.png) preserved,
  currently using style A (flat charcoal) for MM-DD.png.
- Added comments showing where to adjust charcoal transparency for style E
  and how to change which style is used for the live MM-DD.png image.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional, Tuple, List

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

# Optional libraries for posting
try:
    from atproto import Client as BskyClient  # type: ignore
except Exception:
    BskyClient = None

try:
    import tweepy  # type: ignore
except Exception:
    tweepy = None

# -----------------------
# Configuration
# -----------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MED_DIR = PROJECT_ROOT / "meditations"
OUTPUT_DIR = PROJECT_ROOT / "output"
BG_DIR = PROJECT_ROOT / "backgrounds"

MED_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
BG_DIR.mkdir(exist_ok=True)

IMG_SIZE = (1080, 1080)

# Paths
FOREST_BG_PATH = BG_DIR / "ChatGPT-Image-VoR.png"

# Visual tuning
BACKGROUND_CHARCOAL = (17, 17, 17)     # #111111
TITLE_COLOR = (242, 242, 242)          # #F2F2F2
DATE_COLOR = (242, 242, 242)           # same as title for consistency
BODY_COLOR = (255, 255, 255)           # #FFFFFF

TITLE_FONT_SIZE = 36
DATE_FONT_SIZE = 24
BODY_FONT_SIZE = 22

PADDING = 60
LINE_SPACING = 10                       # between lines
PARA_SPACING = 18                       # between paragraphs
TITLE_TO_DATE_SPACING = 20
DATE_TO_BODY_SPACING = 24

QUOTE_CHARS = ['"', '“', '”', '‘', '’']

# Hashtags and URL (for Bluesky + X)
BSKY_HASHTAGS = [
    "#addiction",
    "#saa",
    "#sexaddiction",
    "#recovery",
    "#recoveryposse",
    "#sobriety",
]
BSKY_URL = "https://bit.ly/step12-b"

# X/Twitter uses same hashtags but different URL
X_HASHTAGS = BSKY_HASHTAGS  # Same hashtags for both platforms
X_URL = "http://bit.ly/step12-t"

# -----------------------
# Logging
# -----------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger("daily_poster")

# -----------------------
# Environment / Secrets
# -----------------------

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

BSKY_USERNAME = os.getenv("BSKY_USERNAME")
BSKY_APP_PASSWORD = os.getenv("BSKY_APP_PASSWORD")

X_API_KEY = os.getenv("X_API_KEY")
X_API_SECRET = os.getenv("X_API_SECRET")
X_ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN")
X_ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_TOKEN_SECRET")


# -----------------------
# Utility functions
# -----------------------

def mm_dd_from_arg(arg: str) -> str:
    """Parse MM-DD and return zero-padded 'MM-DD'."""
    try:
        parts = arg.split("-")
        if len(parts) != 2:
            raise ValueError
        mm = int(parts[0])
        dd = int(parts[1])
        return f"{mm:02d}-{dd:02d}"
    except Exception:
        raise ValueError("Date must be in MM-DD format (e.g., 11-19)")


def get_paths_for(mmdd: str) -> Tuple[Path, Path]:
    """Return (text_path, png_path) for this MM-DD."""
    txt = MED_DIR / f"{mmdd}.txt"
    png = OUTPUT_DIR / f"{mmdd}.png"
    return txt, png


# -----------------------
# Scraper
# -----------------------

def scrape_meditation(mmdd: str) -> Tuple[str, str]:
    """
    Scrape SAA daily meditation for given MM-DD.
    Returns (page_title, full_text).

    full_text is constructed as:
      [page_title, opening_quote, body_paragraphs..., closing]
      joined by blank lines.
    """
    mm, dd = mmdd.split("-")
    dt = datetime(datetime.now().year, int(mm), int(dd))
    month_name = dt.strftime("%B")
    day_plain = str(int(dd))
    url = f"https://saa-connect.org/{month_name}-{day_plain}/"
    logger.info("Fetching meditation: %s", url)

    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch {url} (status {resp.status_code})")

    soup = BeautifulSoup(resp.text, "html.parser")
    title_el = soup.find("h1")
    page_title = title_el.get_text(strip=True) if title_el else f"{month_name} {day_plain}"

    main = soup.find("main")
    if not main:
        raise RuntimeError("Could not find <main> on the meditation page")

    p_tags = main.find_all("p")
    paragraphs_raw = [p.get_text(" ", strip=True) for p in p_tags]

    # Find all <em> tags; we only care about the LAST one for closing meditation.
    em_tags = main.find_all("em")
    last_em = em_tags[-1] if em_tags else None

    # Skip first paragraph (often date/author) if more than one
    if len(paragraphs_raw) > 1:
        p_tags = p_tags[1:]
        paragraphs_raw = paragraphs_raw[1:]

    opening: Optional[str] = None
    body_paragraphs: List[str] = []
    closing: Optional[str] = None

    # Identify which paragraph contains the last <em> so we can skip it from body.
    last_em_paragraph_index: Optional[int] = None
    if last_em:
        for idx, p in enumerate(p_tags):
            if last_em in p.descendants:
                last_em_paragraph_index = idx
                closing = last_em.get_text(" ", strip=True)
                break

    for idx, text in enumerate(paragraphs_raw):
        if not text:
            continue

        # First paragraph that starts with a quote is opening
        if opening is None and text and text[0] in QUOTE_CHARS:
            opening = text
            continue

        # Skip the paragraph containing the closing <em>
        if last_em_paragraph_index is not None and idx == last_em_paragraph_index:
            continue

        body_paragraphs.append(text)

    parts: List[str] = [page_title]
    if opening:
        parts.append(opening)
    parts.extend(body_paragraphs)
    if closing:
        parts.append(closing)

    full_text = "\n\n".join([p for p in parts if p])
    return page_title, full_text


# -----------------------
# Fonts / Text
# -----------------------

def load_font(preferred_paths_sizes, fallback_size: int) -> ImageFont.ImageFont:
    """Try several font paths; fallback to default if none available."""
    for path, size in preferred_paths_sizes:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    x: int,
    y: int,
    max_width: int,
    line_spacing: int,
    fill
) -> int:
    """Draw word-wrapped text and return new y after drawing."""
    words = text.split()
    if not words:
        return y

    current: List[str] = []
    for word in words:
        test_line = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        w = bbox[2] - bbox[0]
        if w <= max_width:
            current.append(word)
        else:
            line = " ".join(current)
            draw.text((x, y), line, font=font, fill=fill)
            bbox_line = draw.textbbox((0, 0), line, font=font)
            h = bbox_line[3] - bbox_line[1]
            y += h + line_spacing
            current = [word]

    if current:
        line = " ".join(current)
        draw.text((x, y), line, font=font, fill=fill)
        bbox_line = draw.textbbox((0, 0), line, font=font)
        h = bbox_line[3] - bbox_line[1]
        y += h + line_spacing

    return y


# -----------------------
# Background styles A–E
# -----------------------

def load_and_center_crop_foreground(path: Path) -> Image.Image:
    """Load an image, scale to cover 1080x1080, center-crop."""
    if not path.exists():
        logger.warning("Forest background not found: %s", path)
        return Image.new("RGB", IMG_SIZE, BACKGROUND_CHARCOAL)

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = max(IMG_SIZE[0] / w, IMG_SIZE[1] / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        im_resized = im.resize((new_w, new_h), Image.LANCZOS)

        left = (new_w - IMG_SIZE[0]) // 2
        top = (new_h - IMG_SIZE[1]) // 2
        im_cropped = im_resized.crop((left, top, left + IMG_SIZE[0], top + IMG_SIZE[1]))
        return im_cropped


def make_background_A() -> Image.Image:
    """A: Flat charcoal background."""
    return Image.new("RGB", IMG_SIZE, BACKGROUND_CHARCOAL)


def make_background_B() -> Image.Image:
    """B: Blurred forest with charcoal overlay."""
    base = load_and_center_crop_foreground(FOREST_BG_PATH)
    blurred = base.filter(ImageFilter.GaussianBlur(radius=6))
    # Slight charcoal tint overlay to unify
    overlay = Image.new("RGBA", IMG_SIZE, (0, 0, 0, int(0.45 * 255)))
    return Image.alpha_composite(blurred.convert("RGBA"), overlay).convert("RGB")


def make_background_C() -> Image.Image:
    """C: Charcoal gradient (top darker, bottom lighter)."""
    top_color = (5, 5, 5)
    bottom_color = (34, 34, 34)
    img = Image.new("RGB", IMG_SIZE, top_color)
    draw = ImageDraw.Draw(img)
    for y in range(IMG_SIZE[1]):
        t = y / (IMG_SIZE[1] - 1)
        r = int(top_color[0] * (1 - t) + bottom_color[0] * t)
        g = int(top_color[1] * (1 - t) + bottom_color[1] * t)
        b = int(top_color[2] * (1 - t) + bottom_color[2] * t)
        draw.line([(0, y), (IMG_SIZE[0], y)], fill=(r, g, b))
    return img


def make_background_D() -> Image.Image:
    """D: Textured charcoal using subtle noise."""
    base = Image.new("RGB", IMG_SIZE, BACKGROUND_CHARCOAL)
    # Create noise in grayscale and blend
    noise = Image.effect_noise(IMG_SIZE, 64).convert("L")
    noise_rgb = Image.merge("RGB", (noise, noise, noise))
    # Alpha controls how strong the noise is; 0.07 is subtle
    textured = Image.blend(base, noise_rgb, alpha=0.07)
    return textured


def make_background_E() -> Image.Image:
    """E: Semi-transparent charcoal over forest."""
    base = load_and_center_crop_foreground(FOREST_BG_PATH).convert("RGBA")
    # 70% charcoal overlay (as requested)
    # To change transparency later:
    #   - Modify the 0.70 below: 0.5 = lighter, 0.8 = darker, etc.
    alpha_fraction = 0.70
    alpha_value = int(alpha_fraction * 255)
    overlay = Image.new("RGBA", IMG_SIZE, (17, 17, 17, alpha_value))
    out = Image.alpha_composite(base, overlay).convert("RGB")
    return out


# Map style letters to background generators
BACKGROUND_STYLES = {
    "a": make_background_A,
    "b": make_background_B,
    "c": make_background_C,
    "d": make_background_D,
    "e": make_background_E,
}


# -----------------------
# Text composition on backgrounds
# -----------------------

def compose_text_on_background(
    bg: Image.Image,
    page_title: str,
    full_text: str,
    out_path: Path,
) -> None:
    """
    Draw:
      - "Voices of Recovery" at the top (centered)
      - page_title (e.g., "November 21") centered below it
      - meditation body text below that
    """
    logger.info("Composing image: %s", out_path.name)

    img = bg.copy()
    draw = ImageDraw.Draw(img)

    vof_title = "Voices of Recovery"

    title_font = load_font([
        ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", TITLE_FONT_SIZE),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", TITLE_FONT_SIZE),
    ], TITLE_FONT_SIZE)

    date_font = load_font([
        ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", DATE_FONT_SIZE),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", DATE_FONT_SIZE),
    ], DATE_FONT_SIZE)

    body_font = load_font([
        ("/System/Library/Fonts/Supplemental/Arial.ttf", BODY_FONT_SIZE),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", BODY_FONT_SIZE),
    ], BODY_FONT_SIZE)

    # Break full_text into logical parts (we know first part is page_title)
    parts = [p.strip() for p in full_text.split("\n\n") if p.strip()]
    body_paragraphs = parts[1:] if len(parts) > 1 else []

    # 1) Draw "Voices of Recovery" centered
    y = PADDING
    bbox_title = draw.textbbox((0, 0), vof_title, font=title_font)
    title_w = bbox_title[2] - bbox_title[0]
    title_x = (IMG_SIZE[0] - title_w) // 2
    draw.text((title_x, y), vof_title, font=title_font, fill=TITLE_COLOR)
    y += (bbox_title[3] - bbox_title[1]) + TITLE_TO_DATE_SPACING

    # 2) Draw page_title (date) centered
    if page_title:
        bbox_date = draw.textbbox((0, 0), page_title, font=date_font)
        date_w = bbox_date[2] - bbox_date[0]
        date_x = (IMG_SIZE[0] - date_w) // 2
        draw.text((date_x, y), page_title, font=date_font, fill=DATE_COLOR)
        y += (bbox_date[3] - bbox_date[1]) + DATE_TO_BODY_SPACING

    # 3) Draw body paragraphs, left aligned
    for para in body_paragraphs:
        y = draw_wrapped_text(
            draw,
            para,
            body_font,
            PADDING,
            y,
            IMG_SIZE[0] - 2 * PADDING,
            line_spacing=LINE_SPACING,
            fill=BODY_COLOR,
        )
        y += PARA_SPACING
        if y > IMG_SIZE[1] - PADDING:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG", optimize=True)
    logger.info("Saved image: %s", out_path)


def compose_variant(
    style_letter: str,
    page_title: str,
    full_text: str,
    out_path: Path
) -> None:
    """Generate a single variant image for style_letter ∈ {a,b,c,d,e}."""
    style_letter = style_letter.lower()
    if style_letter not in BACKGROUND_STYLES:
        logger.warning("Unknown style '%s'; defaulting to 'a'.", style_letter)
        style_letter = "a"

    bg_func = BACKGROUND_STYLES[style_letter]
    bg = bg_func()
    compose_text_on_background(bg, page_title, full_text, out_path)


# -----------------------
# Text building for Bluesky / X
# -----------------------

def build_bsky_text(full_text: str) -> str:
    """
    Build Bluesky text:
      - use the opening quote (second paragraph after page_title)
      - append hashtags + URL
      - if total length > 300, truncate quote word-by-word from end
      - add ellipsis (...) to show truncation
      - always preserve all hashtags and URL
    """
    parts = [p.strip() for p in full_text.split("\n\n") if p.strip()]
    
    # Use parts[1] (the opening quote) instead of parts[0] (the date/title)
    # Fall back to parts[0] if there's only one part
    if len(parts) > 1:
        quote = parts[1]
    elif parts:
        quote = parts[0]
    else:
        quote = ""
    
    # Flatten whitespace in quote
    quote = " ".join(quote.split())
    
    # Build the hashtags and URL suffix (always included)
    hashtags_str = " ".join(BSKY_HASHTAGS)
    url = BSKY_URL
    
    # Calculate the suffix (hashtags + URL with spaces)
    suffix_parts = []
    if hashtags_str:
        suffix_parts.append(hashtags_str)
    if url:
        suffix_parts.append(url)
    suffix = " ".join(suffix_parts)
    
    # Reserve space for suffix plus one space between quote and suffix
    reserved = len(suffix) + 1 if suffix else 0
    max_quote_length = 300 - reserved
    
    # If quote fits, return as-is
    if len(quote) <= max_quote_length:
        if suffix:
            return f"{quote} {suffix}"
        return quote
    
    # Quote is too long - truncate word by word from end
    words = quote.split()
    truncated_words = []
    ellipsis = "..."
    
    # Reserve space for ellipsis too
    available = max_quote_length - len(ellipsis) - 1  # -1 for space before ellipsis
    
    current_length = 0
    for word in words:
        # +1 for space between words
        word_length = len(word) + (1 if truncated_words else 0)
        if current_length + word_length <= available:
            truncated_words.append(word)
            current_length += word_length
        else:
            break
    
    # Build final text with ellipsis
    if truncated_words:
        truncated_quote = " ".join(truncated_words) + ellipsis
    else:
        # Edge case: even first word doesn't fit
        truncated_quote = ellipsis
    
    if suffix:
        return f"{truncated_quote} {suffix}"
    return truncated_quote

def build_x_text(full_text: str) -> str:
    """
    Build X/Twitter text:
      - use the opening quote (second paragraph after page_title)
      - append hashtags + X-specific URL
      - if total length > 280, truncate quote word-by-word from end
      - add ellipsis (...) to show truncation
      - always preserve all hashtags and URL
    """
    parts = [p.strip() for p in full_text.split("\n\n") if p.strip()]
    
    # Use parts[1] (the opening quote) instead of parts[0] (the date/title)
    # Fall back to parts[0] if there's only one part
    if len(parts) > 1:
        quote = parts[1]
    elif parts:
        quote = parts[0]
    else:
        quote = ""
    
    # Flatten whitespace in quote
    quote = " ".join(quote.split())
    
    # Build the hashtags and URL suffix (always included)
    hashtags_str = " ".join(X_HASHTAGS)
    url = X_URL
    
    # Calculate the suffix (hashtags + URL with spaces)
    suffix_parts = []
    if hashtags_str:
        suffix_parts.append(hashtags_str)
    if url:
        suffix_parts.append(url)
    suffix = " ".join(suffix_parts)
    
    # Reserve space for suffix plus one space between quote and suffix
    # X/Twitter limit is 280 characters
    reserved = len(suffix) + 1 if suffix else 0
    max_quote_length = 280 - reserved
    
    # If quote fits, return as-is
    if len(quote) <= max_quote_length:
        if suffix:
            return f"{quote} {suffix}"
        return quote
    
    # Quote is too long - truncate word by word from end
    words = quote.split()
    truncated_words = []
    ellipsis = "..."
    
    # Reserve space for ellipsis too
    available = max_quote_length - len(ellipsis) - 1  # -1 for space before ellipsis
    
    current_length = 0
    for word in words:
        # +1 for space between words
        word_length = len(word) + (1 if truncated_words else 0)
        if current_length + word_length <= available:
            truncated_words.append(word)
            current_length += word_length
        else:
            break
    
    # Build final text with ellipsis
    if truncated_words:
        truncated_quote = " ".join(truncated_words) + ellipsis
    else:
        # Edge case: even first word doesn't fit
        truncated_quote = ellipsis
    
    if suffix:
        return f"{truncated_quote} {suffix}"
    return truncated_quote


# -----------------------
# Posting functions
# -----------------------

def post_to_bluesky(text: str, image_path: Optional[Path] = None):
    if BskyClient is None:
        logger.warning("atproto library not installed; skipping Bluesky post.")
        return None
    if not BSKY_USERNAME or not BSKY_APP_PASSWORD:
        logger.warning("Bluesky credentials not configured; skipping Bluesky post.")
        return None

    try:
        client = BskyClient()
        logger.info("Logging into Bluesky as %s", BSKY_USERNAME)
        client.login(BSKY_USERNAME, BSKY_APP_PASSWORD)

        # Build facets for hashtags and URLs to make them clickable
        facets = []
        
        # Find and create facets for hashtags
        import re
        hashtag_pattern = r'#\w+'
        for match in re.finditer(hashtag_pattern, text):
            facets.append({
                "index": {
                    "byteStart": match.start(),
                    "byteEnd": match.end()
                },
                "features": [{
                    "$type": "app.bsky.richtext.facet#tag",
                    "tag": match.group()[1:]  # Remove the # symbol
                }]
            })
        
        # Find and create facets for URLs
        url_pattern = r'https?://[^\s]+'
        for match in re.finditer(url_pattern, text):
            facets.append({
                "index": {
                    "byteStart": match.start(),
                    "byteEnd": match.end()
                },
                "features": [{
                    "$type": "app.bsky.richtext.facet#link",
                    "uri": match.group()
                }]
            })

        # Prepare image embed if provided
        embed = None
        if image_path and image_path.exists():
            with open(image_path, "rb") as f:
                blob = client.upload_blob(f.read())
            embed = {
                "$type": "app.bsky.embed.images",
                "images": [{
                    "image": blob.blob,
                    "alt": "Daily Meditation from Voices of Recovery"
                }]
            }

        logger.info("Sending post to Bluesky with %d facets...", len(facets))
        
        # Create the post with facets
        record = client.send_post(text=text, facets=facets, embed=embed)
        
        logger.info("Bluesky post successful: %s", getattr(record, "uri", "<no-uri>"))
        return record
    except Exception as e:
        logger.exception("Failed to post to Bluesky: %s", e)
        return None

def post_to_x(text: str, image_path: Optional[Path] = None):
    """Post to X/Twitter using API v2"""
    if tweepy is None:
        logger.warning("tweepy not installed; skipping X/Twitter post.")
        return None
    if not all([X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET]):
        logger.warning("X credentials missing; skipping X post.")
        return None

    try:
        # Use API v2 Client (free tier supports this)
        client = tweepy.Client(
            consumer_key=X_API_KEY,
            consumer_secret=X_API_SECRET,
            access_token=X_ACCESS_TOKEN,
            access_token_secret=X_ACCESS_TOKEN_SECRET
        )
        
        media_ids = None
        if image_path and image_path.exists():
            # Media upload still uses v1.1 API (which free tier allows)
            logger.info("Uploading media to X: %s", image_path)
            auth = tweepy.OAuth1UserHandler(
                consumer_key=X_API_KEY,
                consumer_secret=X_API_SECRET,
                access_token=X_ACCESS_TOKEN,
                access_token_secret=X_ACCESS_TOKEN_SECRET,
            )
            api_v1 = tweepy.API(auth)
            media = api_v1.media_upload(str(image_path))
            media_ids = [media.media_id_string]
        
        logger.info("Posting tweet via API v2...")
        
        # Post using v2 Client
        if media_ids:
            response = client.create_tweet(text=text, media_ids=media_ids)
        else:
            response = client.create_tweet(text=text)
        
        tweet_id = response.data['id']
        logger.info("Tweet posted successfully: %s", tweet_id)
        return response
        
    except Exception as e:
        logger.exception("Failed to post to X: %s", e)
        return None


# -----------------------
# Orchestration
# -----------------------

def parse_args() -> tuple[str, bool]:
    """Return (mmdd, test_mode)."""
    args = sys.argv[1:]
    test_mode = False

    if "--test" in args:
        test_mode = True
        args.remove("--test")

    if args:
        mmdd = mm_dd_from_arg(args[0])
    else:
        mmdd = datetime.now().strftime("%m-%d")

    return mmdd, test_mode


def main():
    mmdd, test_mode = parse_args()
    txt_path, png_path = get_paths_for(mmdd)

    # Ensure we have text cached or scraped
    if txt_path.exists():
        full_text = txt_path.read_text(encoding="utf-8")
        logger.info("Loaded text cache: %s", txt_path)
        # Derive page_title from first block of text
        parts = [p.strip() for p in full_text.split("\n\n") if p.strip()]
        page_title = parts[0] if parts else mmdd
    else:
        try:
            page_title, full_text = scrape_meditation(mmdd)
        except Exception as e:
            logger.exception("Scrape error: %s", e)
            sys.exit(1)
        txt_path.write_text(full_text, encoding="utf-8")
        logger.info("Saved text cache: %s", txt_path)

    if test_mode:
        # TEST MODE: generate all 5 image variants, do not post
        logger.info("TEST MODE: Regenerating image variants and skipping posts.")
        for style in ["a", "b", "c", "d", "e"]:
            out_file = OUTPUT_DIR / f"{mmdd}-{style}.png"
            compose_variant(style, page_title, full_text, out_file)
        
        # Generate and save post text previews
        # Build Bsky and X texts
        bsky_text = build_bsky_text(full_text)
        x_text = build_x_text(full_text)
        
        preview_file = OUTPUT_DIR / f"{mmdd}-post-preview.txt"
        with open(preview_file, 'w', encoding='utf-8') as f:
            f.write(f"=== POST TEXT PREVIEW for {mmdd} ===\n\n")
            f.write(f"BLUESKY POST ({len(bsky_text)} chars / 300 max):\n")
            f.write("-" * 70 + "\n")
            f.write(bsky_text + "\n\n")
            f.write(f"X/TWITTER POST ({len(x_text)} chars / 280 max):\n")
            f.write("-" * 70 + "\n")
            f.write(x_text + "\n\n")
            f.write("NOTE: These are the exact texts that would be posted in normal mode.\n")
        
        logger.info("Saved post preview: %s", preview_file)

        print("\n" + "=" * 70)
        print("=== SUMMARY (TEST MODE) ===")
        print("=" * 70)
        print(f"Date: {mmdd}")
        print(f"Text cache: {'EXISTS' if txt_path.exists() else 'MISSING'}")
        print(f"\nGenerated images:")
        for style in ["a", "b", "c", "d", "e"]:
            out_file = OUTPUT_DIR / f"{mmdd}-{style}.png"
            status = "✓" if out_file.exists() else "✗"
            print(f"  {status} Style {style.upper()}: {out_file.name}")
        
        print(f"\n📄 Post preview saved: {preview_file.name}")
        print("\nPost text preview:")
        print("─" * 70)
        print(f"BLUESKY ({len(bsky_text)}/300 chars):")
        print("─" * 70)
        print(bsky_text)
        print("\n" + "─" * 70)
        print(f"X/TWITTER ({len(x_text)}/280 chars):")
        print("─" * 70)
        print(x_text)
        print("─" * 70)
        print("\n⚠️  Posts: SKIPPED (test mode)")
        print("=" * 70 + "\n")
        return

    # NORMAL MODE:
    # Preserve previous behavior: ensure a single MM-DD.png exists using style 'b'
    if not png_path.exists():
        # NOTE: To change the default style used for the live MM-DD.png image,
        # simply change 'a' below to 'b', 'c', 'd', or 'e'.
        # The five styles are:
        #   a: flat charcoal
        #   b: blurred forest + charcoal overlay
        #   c: charcoal gradient
        #   d: textured charcoal
        #   e: semi-transparent charcoal over forest
        compose_variant("b", page_title, full_text, png_path)

    # Build Bsky and X texts
    # Generate and save post text previews
    bsky_text = build_bsky_text(full_text)
    x_text = build_x_text(full_text)

    bsky_res = post_to_bluesky(bsky_text, png_path if png_path.exists() else None)
    x_res = post_to_x(x_text, png_path if png_path.exists() else None)

    # Clear summary
    print("\n=== SUMMARY ===")
    print(f"Date: {mmdd}")
    print(f"Bluesky: {'SUCCESS' if bsky_res else 'FAILED'}")
    print(f"X/Twitter: {'SUCCESS' if x_res else 'FAILED'}\n")

    logger.info(
        "Finished run for %s — Bluesky: %s  X: %s",
        mmdd, bool(bsky_res), bool(x_res)
    )


if __name__ == "__main__":
    main()
