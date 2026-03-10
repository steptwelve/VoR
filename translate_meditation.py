"""
translate_meditation.py

Translates today's English meditation text to Spanish using the Anthropic API.
Reads translation settings from config.yml.

Usage:
  python translate_meditation.py              # today's date
  python translate_meditation.py --date 03-03 # specific date
  python translate_meditation.py --dry-run    # print translation, don't save
  python translate_meditation.py --force      # overwrite existing translation

Output:
  meditations/MM-DD-es.txt

Requires:
  ANTHROPIC_API_KEY in environment or .env file
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from vor_config import config

PROJECT_ROOT = Path(__file__).resolve().parent
MEDITATIONS  = PROJECT_ROOT / "meditations"

load_dotenv(PROJECT_ROOT / ".env")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def parse_args():
    parser = argparse.ArgumentParser(description="Translate daily meditation to Spanish")
    parser.add_argument("--date", default=datetime.now().strftime("%m-%d"), help="Date in MM-DD format")
    parser.add_argument("--dry-run", action="store_true", help="Print translation, do not save")
    parser.add_argument("--force", action="store_true", help="Overwrite existing translation")
    return parser.parse_args()


def load_english_text(mmdd: str) -> str:
    path = MEDITATIONS / f"{mmdd}.txt"
    if not path.exists():
        print(f"❌ English meditation not found: {path}")
        sys.exit(1)
    return path.read_text(encoding="utf-8").strip()


def translate_via_anthropic(english_text: str) -> str:
    try:
        import anthropic
    except ImportError:
        print("❌ Missing: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    system_prompt = config.translation_prompt.strip()

    user_message = f"""Please translate the following recovery meditation text to Spanish:

---
{english_text}
---

Important reminders:
- Use neutral, internationally accessible Spanish understood globally
- Preserve the spiritual tone, humility, and first-person voice
- Keep the same structure: date line, quote, source, story, closing reflection
- Do not add translator notes or any text not in the original
- Output only the translated text
"""

    print(f"[translate] Calling Anthropic API ({config.translation_dialect})...")
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": user_message}],
        system=system_prompt,
    )
    return message.content[0].text.strip()


def translate_via_deepl(english_text: str) -> str:
    try:
        import deepl
    except ImportError:
        print("❌ Missing: pip install deepl")
        sys.exit(1)

    deepl_key = os.getenv("DEEPL_API_KEY")
    if not deepl_key:
        print("❌ DEEPL_API_KEY not set")
        sys.exit(1)

    translator = deepl.Translator(deepl_key)
    result = translator.translate_text(english_text, target_lang="ES", source_lang="EN")
    return result.text.strip()


def translate(english_text: str) -> str:
    provider = config.translation_provider.lower()
    if provider == "anthropic":
        if not ANTHROPIC_API_KEY:
            print("❌ ANTHROPIC_API_KEY not set")
            sys.exit(1)
        return translate_via_anthropic(english_text)
    elif provider == "deepl":
        return translate_via_deepl(english_text)
    else:
        print(f"❌ Unknown translation provider: {provider}")
        sys.exit(1)


def main():
    args = parse_args()
    mmdd = args.date

    if not config.spanish_enabled and not args.dry_run:
        print("⚠️  Spanish is disabled in config.yml (spanish.enabled: false)")
        print("   Use --dry-run to test translation without the flag check.")
        sys.exit(0)

    output_path = MEDITATIONS / f"{mmdd}-es.txt"
    if output_path.exists() and not args.force and not args.dry_run:
        print(f"✅ Spanish translation already exists: {output_path}")
        print("   Use --force to overwrite.")
        sys.exit(0)

    english_text = load_english_text(mmdd)
    print(f"[translate] Source: meditations/{mmdd}.txt ({len(english_text)} chars)")

    spanish_text = translate(english_text)
    print(f"[translate] Translation: {len(spanish_text)} chars")

    if args.dry_run:
        print("\n" + "=" * 60)
        print("ENGLISH:")
        print("=" * 60)
        print(english_text)
        print("\n" + "=" * 60)
        print("SPANISH:")
        print("=" * 60)
        print(spanish_text)
        return

    output_path.write_text(spanish_text, encoding="utf-8")
    print(f"✅ Saved: {output_path}")


if __name__ == "__main__":
    main()
