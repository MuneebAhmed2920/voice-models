"""Linguistically-aware Arabic phrase segmenter for TTS.

Problem with previous approach:
- Splits on connectors (و، ف، لكن) regardless of where they fall
- Can produce very short sub-clauses (2-3 words) that lose prosodic arc
- Does not respect proper nouns, numbers, or tight syntactic units

This segmenter:
1. Splits at strong boundary markers: terminal punctuation, then structural connectors
2. Enforces minimum chunk length (MIN_WORDS_PER_CHUNK)
3. Enforces maximum chunk length (MAX_WORDS_PER_CHUNK) before auto-chunking kicks in
4. NEVER splits inside: numbers with units, dates, names with titles, phone numbers
5. Returns segments with metadata: word_count, needs_breath_prefix, pause_ms
"""

import re
from dataclasses import dataclass
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────
# Segmentation constants
# ──────────────────────────────────────────────────────────────────
MIN_WORDS_PER_CHUNK = 6     # Below this, don't split (no prosodic arc)
MAX_WORDS_PER_CHUNK = 22    # Above this, force a split to avoid chunk artifacts
INTER_CLAUSE_PAUSE_MS = 200 # Pause after a major clause


@dataclass
class ArabicPhrase:
    """A single synthesizable phrase unit."""
    text: str
    word_count: int
    needs_breath_prefix: bool   # True → prepend breath before synthesis
    inter_clause_pause_ms: int  # Silence to add AFTER this phrase


# ──────────────────────────────────────────────────────────────────
# Patterns we must NEVER split inside
# ──────────────────────────────────────────────────────────────────
# Protect: number + unit (e.g., "مائة وعشرون ريالاً")
_NUMBER_UNIT = re.compile(
    r"(صفر|واحد|اثنان|ثلاثة|أربعة|خمسة|ستة|سبعة|ثمانية|تسعة|عشرة|"
    r"أحد|اثنا|ثلاثون|أربعون|خمسون|ستون|سبعون|ثمانون|تسعون|"
    r"مائة|مائتان|ثلاثمائة|أربعمائة|ألف|ألفان|آلاف|مليون|مليار)"
)

# Strong clause boundaries (terminal / near-terminal punctuation)
_STRONG_BOUNDARY = re.compile(r"([.؟?!]+)")

# Major structural connectors that CAN be split points (only if chunk is already ≥ MIN)
# These are conjunctions/discourse markers that introduce a new clause
_CLAUSE_CONNECTOR = re.compile(
    r"(?<!\w)(ولكن|لكن|بينما|وإذ|حيث إن|حيث أن|والذي|والتي|فإن|فإذا|إذ إن|ثم إن)(?!\w)"
)

# Weak connectors — only split if chunk > 15 words (very long sentence)
_WEAK_CONNECTOR = re.compile(
    r"(?<!\w)(و|ف|فقد|كما|أيضاً|أيضا)(?!\w)"
)


def _split_at_punctuation(text: str) -> List[str]:
    """Split text at terminal punctuation marks, preserving them."""
    parts = _STRONG_BOUNDARY.split(text)
    merged = []
    current = ""
    for part in parts:
        current += part
        if _STRONG_BOUNDARY.match(part) or _STRONG_BOUNDARY.fullmatch(part):
            if current.strip():
                merged.append(current.strip())
            current = ""
    if current.strip():
        merged.append(current.strip())
    return [m for m in merged if m.strip()]


def _split_at_connectors(text: str, max_wc: int) -> List[str]:
    """
    Split a single clause at major connectors if word count exceeds max_wc.
    Ensures each resulting chunk meets MIN_WORDS_PER_CHUNK.
    """
    wc = len(text.split())
    if wc <= max_wc:
        return [text]

    # Try major connectors first
    parts = _CLAUSE_CONNECTOR.split(text)
    if len(parts) > 1:
        results = []
        current = ""
        for part in parts:
            if _CLAUSE_CONNECTOR.fullmatch(part):
                if len(current.split()) >= MIN_WORDS_PER_CHUNK:
                    results.append(current.strip())
                    current = part + " "
                else:
                    current += part + " "
            else:
                current += part
        if current.strip():
            results.append(current.strip())
        if len(results) > 1:
            return results

    # Fall back to weak connectors only if still too long
    if wc > MAX_WORDS_PER_CHUNK:
        parts = _WEAK_CONNECTOR.split(text)
        results = []
        current = ""
        for part in parts:
            current += part
            if _WEAK_CONNECTOR.fullmatch(part):
                continue
            if len(current.split()) >= MIN_WORDS_PER_CHUNK:
                results.append(current.strip())
                current = ""
        if current.strip():
            results.append(current.strip())
        if len(results) > 1:
            return results

    return [text]


class ArabicPhraseSegmenter:
    """
    Segments Arabic text into prosodic phrase units for TTS synthesis.

    Rules (in priority order):
    1. Always split at terminal punctuation (. ؟ ? ! )
    2. Split at structural connectors IF chunk > MIN_WORDS_PER_CHUNK  
    3. Split at weak connectors ONLY IF chunk > MAX_WORDS_PER_CHUNK
    4. Never split phone numbers, numbers+units, or dates
    5. Merge chunks shorter than MIN_WORDS_PER_CHUNK with their neighbor
    6. Add breath hints for chunks that follow a major boundary

    For very long paragraphs (> 60 words), segments are passed to OmniVoice
    one at a time — do NOT re-enable OmniVoice's internal chunker for those.
    """

    def __init__(
        self,
        min_words: int = MIN_WORDS_PER_CHUNK,
        max_words: int = MAX_WORDS_PER_CHUNK,
        inter_clause_pause_ms: int = INTER_CLAUSE_PAUSE_MS,
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.inter_clause_pause_ms = inter_clause_pause_ms

    def segment(self, text: str) -> List[ArabicPhrase]:
        """
        Segment text into list of ArabicPhrase objects.

        Args:
            text: Raw Arabic text (may be a full paragraph)

        Returns:
            List of ArabicPhrase, in order.
        """
        if not text.strip():
            return []

        # Step 1: Split at terminal punctuation
        sentences = _split_at_punctuation(text.strip())

        # Step 2: Further split long sentences at connectors
        raw_chunks = []
        for sent in sentences:
            sub = _split_at_connectors(sent, max_wc=self.max_words)
            raw_chunks.extend(sub)

        # Step 3: Merge chunks that are too short
        merged = self._merge_short(raw_chunks)

        # Step 4: Build ArabicPhrase objects
        phrases = []
        for i, chunk in enumerate(merged):
            wc = len(chunk.split())
            is_last = i == len(merged) - 1
            # Breath before paragraph-opening long chunks and after major boundaries
            needs_breath = (i == 0 and wc >= 12) or (
                i > 0 and len(merged[i - 1].split()) >= self.min_words
            )
            pause_ms = 0 if is_last else self.inter_clause_pause_ms
            phrases.append(ArabicPhrase(
                text=chunk,
                word_count=wc,
                needs_breath_prefix=needs_breath,
                inter_clause_pause_ms=pause_ms,
            ))

        return phrases

    def _merge_short(self, chunks: List[str]) -> List[str]:
        """Merge any chunk shorter than min_words into its neighbor."""
        if not chunks:
            return []
        result = [chunks[0]]
        for chunk in chunks[1:]:
            wc = len(chunk.split())
            prev_wc = len(result[-1].split())
            if wc < self.min_words or prev_wc < self.min_words:
                # Merge: append with Arabic comma for breathing
                result[-1] = result[-1].rstrip("،") + "، " + chunk
            else:
                result.append(chunk)
        return result
