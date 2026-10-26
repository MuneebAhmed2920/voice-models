"""Arabic Frontend — Naturalness-focused text preprocessing and prosody planning.

Architecture:
  Input text
  → Normalizer        — numbers, dates, currencies, phone numbers → spoken form
  → Diacritizer       — optional short-vowel vocalization
  → PhraseSegmenter   — linguistically-aware chunk boundaries
  → ProsodyPlanner    — OmniVoice instruct-string generator per intent
  → StyleFormatter    — written → conversational Arabic conversion
"""

from .normalizer import ArabicFrontendNormalizer
from .diacritics import ArabicDiacriticsHandler
from .phrase_segmenter import ArabicPhraseSegmenter
from .prosody_planner import ArabicProsodyPlanner
from .style_formatter import ArabicStyleFormatter

__all__ = [
    "ArabicFrontendNormalizer",
    "ArabicDiacriticsHandler",
    "ArabicPhraseSegmenter",
    "ArabicProsodyPlanner",
    "ArabicStyleFormatter",
]
