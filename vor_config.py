"""
vor_config.py

Central config loader for the VoR daily meditation system.
All scripts import this instead of hardcoding values.

Usage:
    from vor_config import config

    if config.video_enabled:
        build_video()

    if config.spanish_enabled:
        build_spanish_video()
"""

import yaml
from pathlib import Path

CONFIG_FILE = Path(__file__).resolve().parent / "config.yml"


class VoRConfig:
    def __init__(self):
        with open(CONFIG_FILE, "r") as f:
            self._raw = yaml.safe_load(f)

    @property
    def video_enabled(self) -> bool:
        return self._raw.get("video", {}).get("enabled", False)

    @property
    def english_enabled(self) -> bool:
        return self._raw.get("english", {}).get("enabled", True)

    @property
    def english_playlist_id(self) -> str:
        return self._raw.get("english", {}).get("youtube_playlist_id", "")

    @property
    def english_channel_id(self) -> str:
        return self._raw.get("english", {}).get("youtube_channel_id", "")

    @property
    def english_voices(self) -> list:
        return self._raw.get("english", {}).get("voices", [
            "en-US-ChristopherNeural",
            "en-US-JennyNeural",
        ])

    @property
    def spanish_enabled(self) -> bool:
        return self._raw.get("spanish", {}).get("enabled", False)

    @property
    def spanish_same_channel(self) -> bool:
        return self._raw.get("spanish", {}).get("same_channel", False)

    @property
    def spanish_playlist_id(self) -> str:
        return self._raw.get("spanish", {}).get("youtube_playlist_id", "")

    @property
    def spanish_channel_id(self) -> str:
        return self._raw.get("spanish", {}).get("youtube_channel_id", "")

    @property
    def spanish_voices(self) -> list:
        return self._raw.get("spanish", {}).get("voices", [
            "es-US-AlonsoNeural",
            "es-US-PalomaNeural",
        ])

    @property
    def translation_provider(self) -> str:
        return self._raw.get("spanish", {}).get("translation", {}).get("provider", "anthropic")

    @property
    def translation_dialect(self) -> str:
        return self._raw.get("spanish", {}).get("translation", {}).get("dialect", "neutral-international")

    @property
    def translation_prompt(self) -> str:
        return self._raw.get("spanish", {}).get("translation", {}).get("prompt", "")

    def summary(self):
        print("=" * 50)
        print("VoR Configuration")
        print("=" * 50)
        print(f"  Video production:  {'✅ ON' if self.video_enabled else '❌ OFF'}")
        print(f"  English channel:   {'✅ ON' if self.english_enabled else '❌ OFF'}")
        print(f"  Spanish channel:   {'✅ ON' if self.spanish_enabled else '❌ OFF'}")
        if self.spanish_enabled:
            print(f"  Spanish dialect:   {self.translation_dialect}")
            print(f"  Same channel:      {self.spanish_same_channel}")
            print(f"  Translation via:   {self.translation_provider}")
        print("=" * 50)


config = VoRConfig()

if __name__ == "__main__":
    config.summary()
