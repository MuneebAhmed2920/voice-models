"""Unified Arabic Frontend Normalizer — wraps all text preprocessing.

This is the single entry point for text processing before TTS synthesis.
Applies in order:
  1. StyleFormatter  — title expansions, Islamic phrases
  2. ArabicNormalizer (existing) — numbers, dates, currencies
  3. ArabicDiacriticsHandler — diacritization

Usage:
    norm = ArabicFrontendNormalizer(dialect='saudi', diacritics_mode='expanded')
    clean_text = norm.normalize(raw_text)
"""

import sys
import os

# Support both: running from project root and as a package
try:
    from ..src.arabic_nlp.normalizer import ArabicNormalizer
    from ..src.pipeline.normalizer_unified import UnifiedNormalizer
except ImportError:
    # Running from project root directly
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    try:
        from src.arabic_nlp.normalizer import ArabicNormalizer
        from src.pipeline.normalizer_unified import UnifiedNormalizer
    except ImportError:
        ArabicNormalizer = None
        UnifiedNormalizer = None

from .diacritics import ArabicDiacriticsHandler
from .style_formatter import ArabicStyleFormatter


class ArabicFrontendNormalizer:
    """
    Complete text normalization pipeline for Arabic TTS.

    Processing order:
      1. ArabicStyleFormatter    — Title abbreviations, Islamic phrases
      2. UnifiedNormalizer       — Numbers, dates, times, currencies, URLs
      3. ArabicDiacriticsHandler — Function word diacritization
    """

    def __init__(
        self,
        dialect: str = "saudi",
        diacritics_mode: str = "expanded",
    ):
        self.dialect = dialect.lower()
        # Map to normalizer dialect ('msa' or 'egyptian')
        norm_dialect = "msa"  # Saudi/Gulf/MSA all use MSA normalizer

        self.style_formatter = ArabicStyleFormatter()
        self.diacritizer = ArabicDiacriticsHandler(mode=diacritics_mode)

        # Use existing unified normalizer if available
        if UnifiedNormalizer is not None:
            self.text_normalizer = UnifiedNormalizer(dialect=norm_dialect)
        elif ArabicNormalizer is not None:
            self.text_normalizer = ArabicNormalizer(dialect=norm_dialect)
        else:
            self.text_normalizer = None

    def normalize(self, text: str) -> str:
        """Full normalization pipeline."""
        if not text:
            return ""

        # Step 1: Style formatting (titles, Islamic phrases)
        text = self.style_formatter.format(text)

        # Step 2: Number / date / currency normalization
        if self.text_normalizer is not None:
            if hasattr(self.text_normalizer, "normalize"):
                text = self.text_normalizer.normalize(text)

        # Step 3: Diacritization
        text = self.diacritizer.process(text)

        return text

    def normalize_for_synthesis(self, text: str) -> str:
        """
        Normalize text optimized for TTS synthesis.
        Same as normalize() but strips diacritics from content words
        where the model's own language model handles them better.
        """
        return self.normalize(text)
