"""Language boundary detector for Arabic + English code-switching.

Splits mixed-language text into segments so each segment can be:
  - Normalized using the correct language rules
  - Routed to TTS with the correct language tag
  - Pronounced without cross-language phonetic contamination

Arabic Unicode block: U+0600–U+06FF
English detection: ASCII letters, standard punctuation
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import List


@dataclass
class TextSegment:
    """A continuous run of text in a single detected language."""
    text: str
    lang: str     # 'arabic' | 'english' | 'mixed' | 'numeric'
    index: int    # Position in original segment list


# Arabic Unicode ranges
_AR_UNICODE_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
# Latin / ASCII letter range
_EN_UNICODE_RE = re.compile(r"[A-Za-z]")
# Digits only
_NUM_RE = re.compile(r"^\d[\d,.\s]*$")

# Known English words commonly mixed into Arabic call-center speech
# (kept short — just enough for detection confidence boost)
_EN_KEYWORDS = frozenset([
    "ok", "okay", "yes", "no", "confirmed", "pending", "cancelled", "active",
    "online", "offline", "sms", "otp", "pin", "sim", "wifi", "app", "code",
    "order", "status", "account", "balance", "total", "amount", "invoice",
    "minutes", "hours", "days", "delivery", "shipping", "tracking",
    "number", "id", "email", "password", "login", "reset",
])


def _dominant_script(token: str) -> str:
    """Returns the dominant script of a token."""
    ar_count = len(_AR_UNICODE_RE.findall(token))
    en_count = len(_EN_UNICODE_RE.findall(token))
    digit_count = sum(1 for c in token if c.isdigit())

    total = ar_count + en_count + digit_count
    if total == 0:
        return "neutral"
    if ar_count / max(total, 1) >= 0.5:
        return "arabic"
    if en_count / max(total, 1) >= 0.5:
        return "english"
    if digit_count / max(total, 1) >= 0.8:
        return "numeric"
    return "mixed"


class CodeSwitchSegmenter:
    """
    Segments Arabic+English mixed text at language boundaries.

    Strategy:
    1. Tokenize on whitespace
    2. Assign each token a language (arabic / english / numeric / neutral)
    3. Merge consecutive same-language tokens into segments
    4. Short English words (≤3 chars) adjacent to Arabic are merged into Arabic
       unless they are in the known English keyword list

    This prevents pronouncing English words with Arabic phonetics
    while maintaining the same speaker identity across the whole response.
    """

    def segment(self, text: str) -> List[TextSegment]:
        """
        Split text into language-tagged segments.

        Args:
            text: Raw input text (may contain Arabic and English).

        Returns:
            List of TextSegment objects in order.
        """
        if not text.strip():
            return []

        tokens = re.split(r"(\s+)", text)
        # Label each token
        labelled: List[tuple] = []  # (token_str, lang_label)
        for tok in tokens:
            if re.match(r"^\s+$", tok):
                labelled.append((tok, "space"))
                continue
            lang = _dominant_script(tok)
            # Boost detection for short tokens using keyword list
            if lang in ("english", "mixed") and tok.lower().strip(".,!?;:") in _EN_KEYWORDS:
                lang = "english"
            elif lang == "mixed" and len(tok) <= 2:
                # Very short ambiguous token — default to previous context
                lang = "neutral"
            labelled.append((tok, lang))

        # Merge into runs
        segments: List[TextSegment] = []
        current_text = ""
        current_lang = None

        for tok, lang in labelled:
            if lang == "space":
                current_text += tok
                continue

            effective_lang = lang if lang != "neutral" else (current_lang or "arabic")

            if current_lang is None:
                current_lang = effective_lang
                current_text = tok
            elif effective_lang == current_lang:
                current_text += tok
            else:
                # Language boundary — flush current segment
                if current_text.strip():
                    segments.append(TextSegment(
                        text=current_text.strip(),
                        lang=current_lang,
                        index=len(segments),
                    ))
                current_text = tok
                current_lang = effective_lang

        if current_text.strip():
            segments.append(TextSegment(
                text=current_text.strip(),
                lang=current_lang or "arabic",
                index=len(segments),
            ))

        # Post-process: merge isolated short English tokens (1 word ≤ 4 chars)
        # that aren't keywords back into adjacent Arabic segment
        segments = self._merge_isolated_short_english(segments)

        return segments

    def _merge_isolated_short_english(self, segments: List[TextSegment]) -> List[TextSegment]:
        """Merge single-word English segments that are very short non-keywords into adjacent Arabic."""
        if len(segments) <= 1:
            return segments

        merged = []
        i = 0
        while i < len(segments):
            seg = segments[i]
            # Check if this is a short isolated English word that should stay Arabic
            if (
                seg.lang == "english"
                and len(seg.text.split()) == 1
                and len(seg.text) <= 3
                and seg.text.lower().strip(".,!?") not in _EN_KEYWORDS
            ):
                # Merge into previous if possible
                if merged and merged[-1].lang == "arabic":
                    merged[-1] = TextSegment(
                        text=merged[-1].text + " " + seg.text,
                        lang="arabic",
                        index=merged[-1].index,
                    )
                    i += 1
                    continue
            merged.append(seg)
            i += 1

        return merged

    def to_omnivoice_lang(self, segment: TextSegment) -> str:
        """Maps a segment language to the OmniVoice language string."""
        mapping = {
            "arabic": "Standard Arabic",
            "english": "English",
            "numeric": "Standard Arabic",  # Numbers always read in Arabic
            "mixed": "Standard Arabic",
            "neutral": "Standard Arabic",
        }
        return mapping.get(segment.lang, "Standard Arabic")

    def reconstruct(self, segments: List[TextSegment]) -> str:
        """Reconstructs the original text from segments (for debugging)."""
        return " ".join(s.text for s in segments)
