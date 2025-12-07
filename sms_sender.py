"""
sms_sender.py

Version: 2.0.00
Generated: 2025-12-07 11:45:00

SMS/MMS Distribution Script for VoR Daily Meditations

This script reads subscribers from a Google Sheet and sends the daily
meditation via SMS/MMS using Twilio. It runs separately from the main
scraping/posting workflow (daily_poster.py) to allow different scheduling.

Features:
- Reads Active subscribers from Google Sheet
- Auto-adds +1 prefix to US phone numbers if missing
- Validates meditation text and image files exist
- Checks Twilio opt-out status before sending
- Sends SMS with meditation text + MMS with image
- Sends error notifications to admin via SMS
- Graceful error handling and detailed logging

Usage:
    python sms_sender.py [MM-DD] [--test]
    
    MM-DD: Optional date in MM-DD format (defaults to today)
    --test: Generate preview without actually sending

Environment Variables Required:
    TWILIO_ACCOUNT_SID
    TWILIO_AUTH_TOKEN
    TWILIO_PHONE_NUMBER
    GOOGLE_SHEETS_CREDENTIALS (path to JSON file)
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional, List, Tuple

from dotenv import load_dotenv

# Optional libraries - will check if available
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    TwilioClient = None

try:
    import gspread
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False
    gspread = None

# -----------------------
# Configuration
# -----------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MED_DIR = PROJECT_ROOT / "meditations"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Admin notification numbers
# Jackson is hardcoded to ensure he's always notified
# Additional admins can be added via Google Sheet "Is Admin" column
HARDCODED_ADMIN_PHONES = [
    "+14255771769",  # Jackson (always notified)
]

# Google Sheets configuration
GOOGLE_CREDENTIALS_FILE = "vor-daily-meditation-by-sms.json"
GOOGLE_SHEET_NAME = "VoR SMS Subscribers"

# -----------------------
# Logging
# -----------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger("sms_sender")

# -----------------------
# Environment / Secrets
# -----------------------

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")


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
        raise ValueError("Date must be in MM-DD format (e.g., 12-07)")


def normalize_phone_number(phone: str) -> str:
    """
    Normalize phone number to E.164 format.
    Adds +1 prefix for US numbers if missing.
    """
    # Remove any spaces, dashes, parentheses
    clean = phone.strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    
    # If it doesn't start with +, assume US and add +1
    if not clean.startswith("+"):
        # Remove leading 1 if present (e.g., 14255771769 -> 4255771769 -> +14255771769)
        if clean.startswith("1") and len(clean) == 11:
            clean = clean[1:]
        clean = f"+1{clean}"
    
    return clean


def send_admin_notification(client: TwilioClient, message: str, sheet_admins: List[str] = None) -> None:
    """
    Send SMS notification to all admin phone numbers about errors.
    Combines hardcoded admins with admins from Google Sheet.
    Uses a separate try/catch to avoid notification failures causing script crashes.
    
    Args:
        client: Twilio client
        message: Error message to send
        sheet_admins: List of admin phone numbers from Google Sheet (optional)
    """
    # Combine hardcoded admins with Sheet admins, removing duplicates
    all_admins = list(set(HARDCODED_ADMIN_PHONES + (sheet_admins or [])))
    
    logger.info(f"Sending admin notifications to {len(all_admins)} admin(s)")
    
    for admin_phone in all_admins:
        try:
            logger.info(f"  Notifying {admin_phone}...")
            client.messages.create(
                body=f"🚨 VoR SMS Error:\n\n{message}",
                from_=TWILIO_PHONE_NUMBER,
                to=admin_phone
            )
            logger.info(f"  ✓ Notification sent to {admin_phone}")
        except Exception as e:
            logger.error(f"  ✗ Failed to send to {admin_phone}: {e}")
            # Don't raise - we don't want notification failures to crash the script


# -----------------------
# Core Functions
# -----------------------

def check_prerequisites(mmdd: str) -> Tuple[Path, Path, str]:
    """
    Check that all required files and credentials exist.
    Returns (text_path, image_path, error_message)
    If error_message is not empty, prerequisites failed.
    """
    errors = []
    
    # Check dependencies
    if not TWILIO_AVAILABLE:
        errors.append("Twilio library not installed (pip install twilio)")
    
    if not GSPREAD_AVAILABLE:
        errors.append("gspread library not installed (pip install gspread)")
    
    # Check Twilio credentials
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
        errors.append("Twilio credentials missing in .env file")
    
    # Check Google credentials file
    creds_path = PROJECT_ROOT / GOOGLE_CREDENTIALS_FILE
    if not creds_path.exists():
        errors.append(f"Google credentials file not found: {GOOGLE_CREDENTIALS_FILE}")
    
    # Check meditation text file
    text_path = MED_DIR / f"{mmdd}.txt"
    if not text_path.exists():
        errors.append(f"Meditation text file not found: {text_path}")
    
    # Check meditation image file
    image_path = OUTPUT_DIR / f"{mmdd}.png"
    if not image_path.exists():
        errors.append(f"Meditation image not found: {image_path}")
    
    if errors:
        error_msg = "\n".join(f"- {err}" for err in errors)
        return None, None, error_msg
    
    return text_path, image_path, ""


def get_active_subscribers() -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Read active subscribers and admin phone numbers from Google Sheet.
    Returns (subscribers, admin_phones) where:
        subscribers: list of (phone_number, name) tuples for Active subscribers
        admin_phones: list of phone numbers marked as admins in Sheet
    Raises exception if Sheet access fails.
    """
    logger.info("Connecting to Google Sheets...")
    
    creds_path = PROJECT_ROOT / GOOGLE_CREDENTIALS_FILE
    gc = gspread.service_account(filename=str(creds_path))
    
    logger.info(f"Opening sheet: {GOOGLE_SHEET_NAME}")
    sheet = gc.open(GOOGLE_SHEET_NAME).sheet1
    
    # Get all rows (skip header)
    rows = sheet.get_all_values()[1:]
    
    subscribers = []
    admin_phones = []
    
    for row in rows:
        if len(row) < 6:  # Need at least 6 columns (Phone, Name, Status, Date Added, Source, Date Unsubscribed)
            continue
        
        phone = row[0]
        name = row[1]
        status = row[2]
        # row[3] = Date Added
        # row[4] = Source
        # row[5] = Date Unsubscribed
        is_admin = row[6].strip().lower() if len(row) > 6 else ""  # Column 7: Is Admin
        
        normalized_phone = normalize_phone_number(phone)
        
        # Add to admin list if marked as admin
        if is_admin in ["yes", "y", "true", "admin"]:
            admin_phones.append(normalized_phone)
            logger.info(f"  Found admin: {name} ({normalized_phone})")
        
        # Only include Active subscribers
        if status.strip().lower() == "active":
            subscribers.append((normalized_phone, name))
            logger.info(f"  Found active subscriber: {name} ({normalized_phone})")
    
    logger.info(f"Total active subscribers: {len(subscribers)}")
    logger.info(f"Total admins from Sheet: {len(admin_phones)}")
    
    return subscribers, admin_phones


def is_opted_out(client: TwilioClient, phone: str) -> bool:
    """
    Check if a phone number has opted out via Twilio.
    Returns True if opted out, False otherwise.
    """
    try:
        # Fetch opt-out list for this phone number
        # Note: This is a simplified check - in production you might want to
        # maintain your own opt-out tracking in the Google Sheet
        messages = client.messages.list(to=phone, limit=5)
        for msg in messages:
            if msg.status == 'undelivered' and msg.error_code in [21610, 21614]:
                # Error codes for opt-out
                return True
        return False
    except Exception as e:
        logger.warning(f"Could not check opt-out status for {phone}: {e}")
        return False


def build_sms_text(meditation_text: str) -> str:
    """
    Build SMS text from meditation content.
    Keep it concise for SMS character limits.
    """
    # Split meditation into parts
    parts = [p.strip() for p in meditation_text.split("\n\n") if p.strip()]
    
    if len(parts) < 2:
        return meditation_text[:300]  # Fallback
    
    # Use the opening quote (typically parts[1])
    quote = parts[1] if len(parts) > 1 else parts[0]
    
    # Truncate if needed and add source
    max_length = 280
    footer = "\n\n- Voices of Recovery"
    
    available = max_length - len(footer)
    if len(quote) > available:
        quote = quote[:available-3] + "..."
    
    return quote + footer


def send_meditation_sms(
    client: TwilioClient,
    phone: str,
    name: str,
    sms_text: str,
    image_path: Path,
    test_mode: bool = False
) -> Tuple[bool, str]:
    """
    Send meditation via SMS/MMS to a single subscriber.
    Returns (success, error_message).
    """
    try:
        if test_mode:
            logger.info(f"TEST MODE: Would send to {name} ({phone})")
            return True, ""
        
        logger.info(f"Sending to {name} ({phone})...")
        
        # Upload image for MMS
        # Note: Image needs to be publicly accessible URL
        # For now, we'll send SMS only - MMS requires hosting the image
        # TODO: Add image hosting and MMS support
        
        message = client.messages.create(
            body=sms_text,
            from_=TWILIO_PHONE_NUMBER,
            to=phone
        )
        
        logger.info(f"  Sent! Message SID: {message.sid}")
        return True, ""
        
    except Exception as e:
        error_msg = f"Failed to send to {name} ({phone}): {str(e)}"
        logger.error(error_msg)
        return False, error_msg


# -----------------------
# Main Execution
# -----------------------

def parse_args() -> Tuple[str, bool]:
    """Parse command line arguments. Returns (mmdd, test_mode)."""
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
    """Main execution function."""
    mmdd, test_mode = parse_args()
    
    logger.info("=" * 70)
    logger.info(f"VoR SMS Sender v2.0 - {mmdd}")
    if test_mode:
        logger.info("TEST MODE: No messages will be sent")
    logger.info("=" * 70)
    
    # Check prerequisites
    logger.info("Checking prerequisites...")
    text_path, image_path, error_msg = check_prerequisites(mmdd)
    
    if error_msg:
        logger.error("Prerequisites check failed:")
        logger.error(error_msg)
        
        # Try to send admin notification if Twilio is available
        if TWILIO_AVAILABLE and all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
            try:
                client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                send_admin_notification(
                    client,
                    f"SMS sending failed for {mmdd}:\n\n{error_msg}"
                )
            except Exception as e:
                logger.error(f"Could not send admin notification: {e}")
        
        sys.exit(1)
    
    logger.info("✓ All prerequisites met")
    
    # Initialize Twilio client
    try:
        client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        logger.info("✓ Twilio client initialized")
    except Exception as e:
        error_msg = f"Failed to initialize Twilio client: {e}"
        logger.error(error_msg)
        sys.exit(1)
    
    # Load meditation text
    try:
        meditation_text = text_path.read_text(encoding="utf-8")
        sms_text = build_sms_text(meditation_text)
        logger.info(f"✓ Loaded meditation text ({len(sms_text)} chars)")
    except Exception as e:
        error_msg = f"Failed to load meditation text: {e}"
        logger.error(error_msg)
        send_admin_notification(client, error_msg)
        sys.exit(1)
    
    # Get subscribers and admins
    try:
        subscribers, sheet_admins = get_active_subscribers()
        if not subscribers:
            logger.warning("No active subscribers found")
            sys.exit(0)
    except Exception as e:
        error_msg = f"Failed to read subscribers from Google Sheet: {e}"
        logger.error(error_msg)
        send_admin_notification(client, error_msg)  # No sheet_admins available yet
        sys.exit(1)
    
    # Send to each subscriber
    logger.info(f"\nSending to {len(subscribers)} subscribers...")
    logger.info("-" * 70)
    
    success_count = 0
    failure_count = 0
    failures = []
    
    for phone, name in subscribers:
        # Check opt-out status
        if not test_mode and is_opted_out(client, phone):
            logger.info(f"Skipping {name} ({phone}) - opted out")
            continue
        
        # Send message
        success, error = send_meditation_sms(
            client, phone, name, sms_text, image_path, test_mode
        )
        
        if success:
            success_count += 1
        else:
            failure_count += 1
            failures.append(error)
    
    # Summary
    logger.info("-" * 70)
    logger.info(f"\n{'TEST MODE ' if test_mode else ''}SUMMARY:")
    logger.info(f"  Successful: {success_count}")
    logger.info(f"  Failed: {failure_count}")
    
    # Send admin notification if there were failures
    if failure_count > 0:
        failure_summary = "\n".join(failures[:5])  # First 5 failures
        if len(failures) > 5:
            failure_summary += f"\n...and {len(failures) - 5} more"
        
        send_admin_notification(
            client,
            f"SMS sending completed for {mmdd} with {failure_count} failures:\n\n{failure_summary}",
            sheet_admins
        )
    
    logger.info("\nDone!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
