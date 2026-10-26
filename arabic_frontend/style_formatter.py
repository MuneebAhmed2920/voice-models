"""Written-to-spoken Arabic style formatter.

Converts written Arabic conventions to their natural spoken equivalents
before TTS synthesis. These are patterns that appear correctly in written
Arabic but would sound unnatural if read verbatim.

Examples:
  Written: "وفقاً للإحصاءات"  → Spoken: "وفقاً للإحصاءات" (no change, correct)
  Written: "أ.د. محمد"        → Spoken: "أستاذ دكتور محمد"
  Written: "سعر الدولار $3.5"  → handled by normalizer
  Written: "الرقم: 05xxxxxxxx" → handled by normalizer
  Written: "ص.ب 1234"         → Spoken: "صندوق بريد ألف ومائتان وأربعة وثلاثون"

This module handles patterns NOT covered by the main ArabicNormalizer:
  - Vocative address patterns
  - Sentence-initial fillers common in Saudi speech
  - Fixed Islamic/religious expressions
  - Honorific and title abbreviations
"""

import re


# ──────────────────────────────────────────────────────────────────
# Religious / Islamic fixed phrases (common in Saudi speech)
# These should be read verbatim as they are already in spoken form
# ──────────────────────────────────────────────────────────────────
ISLAMIC_FIXED: dict = {
    "إن شاء الله": "إن شاء الله",
    "إنشاء الله": "إن شاء الله",
    "ان شاء الله": "إن شاء الله",
    "ماشاء الله": "ما شاء الله",
    "ما شاء الله": "ما شاء الله",
    "الحمد لله": "الحمد لله",
    "سبحان الله": "سبحان الله",
    "بسم الله": "بسم الله",
    "بسم الله الرحمن الرحيم": "بسم الله الرحمن الرحيم",
    "أستغفر الله": "أستغفر الله",
    "جزاك الله خيراً": "جزاك الله خيراً",
    "جزاكم الله خيراً": "جزاكم الله خيراً",
    "الله يعطيك العافية": "الله يعطيك العافية",
    "الله يعافيك": "الله يعافيك",
    "يسلمك": "يسلمك",
    "يسلمكم": "يسلمكم",
}

# ──────────────────────────────────────────────────────────────────
# Title abbreviations (common in formal/business Arabic)
# ──────────────────────────────────────────────────────────────────
TITLE_EXPANSIONS: dict = {
    r"\bأ\.د\.\s*": "أستاذ دكتور ",
    r"\bد\.\s*": "دكتور ",
    r"\bأ\.\s*": "أستاذ ",
    r"\bم\.\s*": "مهندس ",
    r"\bص\.ب\s*": "صندوق بريد ",
    r"\bت\.\s*": "تلفون ",
    r"\bجوال\s*:\s*": "جوال ",
    r"\bهاتف\s*:\s*": "هاتف ",
}


class ArabicStyleFormatter:
    """
    Prepares Arabic text for natural spoken delivery by converting
    written conventions to their spoken equivalents.
    """

    def format(self, text: str) -> str:
        """Apply all style formatting rules."""
        if not text:
            return ""

        # Normalize Islamic phrases (preserve correct forms)
        text = self._normalize_islamic(text)

        # Expand title abbreviations
        text = self._expand_titles(text)

        # Normalize Saudi-specific spoken markers
        text = self._normalize_saudi_markers(text)

        # Clean up double spaces
        text = re.sub(r"[ \t]+", " ", text).strip()
        return text

    def _normalize_islamic(self, text: str) -> str:
        """Normalize common Islamic expressions to standard forms."""
        for written, spoken in ISLAMIC_FIXED.items():
            text = text.replace(written, spoken)
        return text

    def _expand_titles(self, text: str) -> str:
        """Expand Arabic honorific title abbreviations."""
        for pattern, expansion in TITLE_EXPANSIONS.items():
            text = re.sub(pattern, expansion, text)
        return text

    def _normalize_saudi_markers(self, text: str) -> str:
        """
        Normalize conversational Saudi Arabic discourse markers.
        These appear in call-center scripts to convey specific tones.
        """
        # "طيب" as a standalone transition → preserve (common Saudi)
        # "وين" → "أين" (Saudi → MSA, only for MSA/formal scripts)
        # For Saudi dialect, keep dialectal markers as-is
        return text
