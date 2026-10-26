"""
mahrukh_urdu_model

Standalone Pakistani Urdu Female Voice Model (Mahrukh)
Optimized for AI Call Centers, Telephony (8kHz G.711 / 24kHz HD), and Inbound Bots.
"""

from .runtime import (
    UrduVoiceModel,
    normalize_urdu_text,
    process_telephony_audio,
    apply_telephony_filter,
    to_wav_bytes
)

__version__ = "1.0.0"
__speaker__ = "Mahrukh"
__language__ = "ur"

__all__ = [
    "UrduVoiceModel",
    "normalize_urdu_text",
    "process_telephony_audio",
    "apply_telephony_filter",
    "to_wav_bytes"
]
