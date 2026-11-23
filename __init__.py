"""
Marukh English Voice Model Package
Standalone production English female voice model of Mahrukh.
"""

from .runtime.engine import EnglishVoiceModel
from .runtime.audio_processor import process_telephony_audio, to_wav_bytes
from .runtime.text_normalizer import normalize_english_text

__version__ = "1.0.0"
__all__ = [
    "EnglishVoiceModel",
    "process_telephony_audio",
    "to_wav_bytes",
    "normalize_english_text"
]
