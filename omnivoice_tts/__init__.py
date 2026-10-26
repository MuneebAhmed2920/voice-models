"""
OmniVoice TTS: Production-Focused Minimal English Speech Synthesis Engine.
"""

from omnivoice_tts.engine import OmniVoiceTTSEngine, get_engine
from omnivoice_tts.text.normalizer import normalize_english_text
from omnivoice_tts.voices.registry import VoiceRegistry

__version__ = "1.0.0"
__all__ = ["OmniVoiceTTSEngine", "get_engine", "normalize_english_text", "VoiceRegistry"]
