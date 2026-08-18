"""
Voice Package: Clara (Female Customer Service Representative) [voice_clara]
Warm, clear, natural conversational American female voice package with physiological breathing and telephone prosody.
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
