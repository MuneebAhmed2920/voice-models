"""Breathing gate — decides whether pre-speech breathing is contextually appropriate.

Rules (in priority order):
  1. NEVER breathe before: short responses (≤ 10 words), numbers/digits,
     names (single word), confirmations, or farewells.
  2. ALWAYS allow breath before: long paragraphs (> 18 words).
  3. CONDITIONALLY allow breath: medium responses (11-18 words) that
     contain a natural discourse opener or multi-clause connector.
  4. DEFAULT: no breath (safe for telephony to avoid sounding artificial).

This module replaces the previous unconditional breath-injection behavior
where breath was added even to 2-word confirmations like "Sure." or "Okay."
"""

import re
from typing import Optional


# ──────────────────────────────────────────────────────────────────
# Pattern: responses that are primarily numeric sequences
# ──────────────────────────────────────────────────────────────────
_NUMERIC_HEAVY_RE = re.compile(
    r"(?:صفر|واحد|اثنان|ثلاثة|أربعة|خمسة|ستة|سبعة|ثمانية|تسعة|عشرة|"
    r"zero|one|two|three|four|five|six|seven|eight|nine)[\s،,]"
    r"(?:صفر|واحد|اثنان|ثلاثة|أربعة|خمسة|ستة|سبعة|ثمانية|تسعة|عشرة|"
    r"zero|one|two|three|four|five|six|seven|eight|nine)",
    re.IGNORECASE,
)

# ──────────────────────────────────────────────────────────────────
# Pattern: single-word or very short confirmations
# ──────────────────────────────────────────────────────────────────
_SHORT_CONFIRM_RE = re.compile(
    r"^(نعم|لا|أكيد|تمام|ماشي|حسناً|حسنا|بالتأكيد|طبعاً|طبعا|"
    r"yes|no|sure|okay|ok|right|great|perfect|got it|understood)[.!،]?\s*$",
    re.IGNORECASE,
)

# ──────────────────────────────────────────────────────────────────
# Pattern: farewell closings (no breath — speech is wrapping up)
# ──────────────────────────────────────────────────────────────────
_FAREWELL_RE = re.compile(
    r"(مع السلامة|وداعاً|وداعا|في أمان الله|إلى اللقاء|"
    r"goodbye|bye|take care|have a nice day)",
    re.IGNORECASE,
)

# ──────────────────────────────────────────────────────────────────
# Discourse openers that legitimize a breath at the start
# ──────────────────────────────────────────────────────────────────
_DISCOURSE_OPENER_RE = re.compile(
    r"^(بالنسبة|فيما يخص|في ما يتعلق|أود أن|يسعدني|يشرفني|"
    r"أريد أن أوضح|دعني أشرح|سأوضح لك|"
    r"regarding|with respect to|i would like to|allow me to|"
    r"let me explain|as for|in terms of)",
    re.IGNORECASE,
)


class BreatheGate:
    """
    Determines whether inserting a pre-speech breath is contextually
    appropriate for a given text and dialog context.

    Call `.should_breathe(text)` before injecting any breath.
    """

    def should_breathe(
        self,
        text: str,
        context_allow_breath: bool = True,
        word_count_override: Optional[int] = None,
    ) -> bool:
        """
        Returns True only when inserting a subtle breath is contextually
        justified and will not make the speech sound artificial.

        Args:
            text: The TTS input text for this utterance.
            context_allow_breath: Hint from ContextProsodyAnalyzer.allow_breath.
            word_count_override: Use this word count instead of computing.

        Returns:
            bool — True means a breath is appropriate, False means skip it.
        """
        if not text:
            return False

        t = text.strip()
        wc = word_count_override if word_count_override is not None else len(t.split())

        # ── Rule 1: Never breathe on single-word or farewell utterances ──
        if wc <= 3:
            return False

        if _SHORT_CONFIRM_RE.match(t):
            return False

        if _FAREWELL_RE.search(t):
            return False

        # ── Rule 2: Never breathe during phone number readouts ──────────
        if _NUMERIC_HEAVY_RE.search(t):
            return False

        # ── Rule 3: Never breathe if prosody analysis says not to ────────
        if not context_allow_breath:
            return False

        # ── Rule 4: Always breathe on long paragraph responses ───────────
        if wc >= 18:
            return True

        # ── Rule 5: Medium responses with discourse opener ───────────────
        if wc >= 12 and _DISCOURSE_OPENER_RE.search(t):
            return True

        # ── Default: no breath (safe for telephony) ──────────────────────
        return False

    def should_breathe_between_clauses(
        self,
        clause_text: str,
        clause_word_count: int,
        is_last_clause: bool,
    ) -> bool:
        """
        Determines if a micro-breath is appropriate between two clauses
        (inter-clause breathing, not pre-speech breathing).

        Only fires when the current clause is long and not the final clause.

        Args:
            clause_text: The text of the current clause.
            clause_word_count: Word count of the clause.
            is_last_clause: True if this is the last clause in the utterance.

        Returns:
            bool
        """
        if is_last_clause:
            return False
        if clause_word_count < 8:
            return False
        if _NUMERIC_HEAVY_RE.search(clause_text):
            return False
        return True
