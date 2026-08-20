"""
Voice Package: Clara (Female Customer Service Representative) [voice_clara]
Production Real-Time OmniVoice TTS Engine for Vicidial & AI Voice Bots.
"""

import os
import json
import torch

_VOICE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(_VOICE_DIR, "prompt", "voice_prompt.pt")
CONFIG_PATH = os.path.join(_VOICE_DIR, "config", "config.json")
METADATA_PATH = os.path.join(_VOICE_DIR, "config", "metadata.json")


def load_prompt(device: str = "cpu"):
    """Loads precomputed prompt tensor into memory."""
    return torch.load(PROMPT_PATH, map_location=device, weights_only=False)


def load_config():
    """Loads configuration dictionary."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata():
    """Loads provenance & licensing metadata."""
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# Lazy runtime imports
def get_engine(*args, **kwargs):
    from runtime.engine import get_engine as _get_engine
    return _get_engine(*args, **kwargs)


__all__ = [
    "load_prompt",
    "load_config",
    "load_metadata",
    "get_engine",
    "PROMPT_PATH",
    "CONFIG_PATH",
    "METADATA_PATH",
]
