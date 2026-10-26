"""Context-aware prosody analyzer for real-time Arabic telephone voice agents.

Infers dialog intent from the response text and returns prosody hints
(speaking rate, pitch bias, energy, instruct modifier, sentence-ending style)
to guide TTS synthesis. Zero latency — no LLM required.

Priority order:
  Correct pronunciation > Natural prosody > Conversational rhythm >
  Speaker consistency > Appropriate emotion > Audio quality > Low latency > Breathing
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DialogIntent(Enum):
    """Inferred purpose / communicative function of the utterance."""
    GREETING = "greeting"
    FAREWELL = "farewell"
    CONFIRMATION = "confirmation"
    QUESTION = "question"
    APOLOGY = "apology"
    IMPORTANT_INFO = "important_info"
    GOOD_NEWS = "good_news"
    NORMAL_INFO = "normal_info"
    LONG_EXPLANATION = "long_explanation"


@dataclass
class ProsodyHints:
    """Recommended prosody settings for the TTS engine, derived from dialog intent."""
    intent: DialogIntent
    speed: float              # TTS speed multiplier (0.85 – 1.10)
    pitch_bias: str           # OmniVoice instruct tag component: 'high pitch' / 'low pitch' / ''
    energy: str               # 'loud' / 'whisper' / '' (only loud/whisper are supported)
    guidance_scale: float     # Classifier-free guidance (2.0 – 3.5)
    num_step: int             # Diffusion steps (16 = fast, 32 = best quality)
    style_tag: str            # Human label for logging / debugging
    sentence_ending: str      # 'rising' | 'falling' | 'neutral'
    allow_breath: bool        # Whether a pre-speech breath is appropriate
    instruct_override: Optional[str] = None  # Full instruct string if needed


# ──────────────────────────────────────────────────────────────────
# Arabic greeting / farewell keywords
# ──────────────────────────────────────────────────────────────────
_AR_GREETINGS = re.compile(
    r"(مرحب|أهلاً|أهلا|السلام عليكم|صباح الخير|مساء الخير|كيفك|كيف حالك|"
    r"يا هلا|هلا والله|أهلين|صباح النور|طاب يومك)",
    re.IGNORECASE,
)
_AR_FAREWELLS = re.compile(
    r"(مع السلامة|وداعاً|وداعا|إلى اللقاء|تصبح على خير|في أمان الله|باي|"
    r"شكراً لتواصلك|شكرا لتواصلك|نتشرف بخدمتك دائماً|لا تتردد)",
    re.IGNORECASE,
)
_EN_GREETINGS = re.compile(
    r"\b(hello|hi|good morning|good afternoon|good evening|welcome|hey there)\b",
    re.IGNORECASE,
)

# ──────────────────────────────────────────────────────────────────
# Confirmation / short positive responses
# ──────────────────────────────────────────────────────────────────
_AR_CONFIRMATIONS = re.compile(
    r"^(نعم|بالتأكيد|أكيد|طبعاً|طبعا|تمام|ماشي|حسناً|حسنا|بكل سرور|"
    r"شكراً|شكرا|مفهوم|فهمت|سيتم|سوف نقوم|"
    r"سأتحقق|دعني أتحقق|لحظة من فضلك|انتظر قليلاً)\b",
    re.IGNORECASE,
)
_EN_CONFIRMATIONS = re.compile(
    r"^(yes|sure|okay|ok|absolutely|certainly|of course|"
    r"let me check|one moment|please hold|got it|understood|"
    r"your (total|order|balance|amount))",
    re.IGNORECASE,
)

# ──────────────────────────────────────────────────────────────────
# Questions
# ──────────────────────────────────────────────────────────────────
_AR_QUESTIONS = re.compile(
    r"(هل|هل يمكن|كيف|متى|أين|من|ما|ماذا|لماذا|أي|كم|هل تود|هل تريد|"
    r"هل يمكنني|هل بإمكاني|هل تحتاج).*[؟?]",
    re.IGNORECASE,
)
_EN_QUESTIONS = re.compile(
    r"\b(would you|could you|can you|do you|is there|are there|"
    r"what|when|where|who|how|which|why)\b.+[?]",
    re.IGNORECASE,
)

# ──────────────────────────────────────────────────────────────────
# Apologies
# ──────────────────────────────────────────────────────────────────
_AR_APOLOGIES = re.compile(
    r"(نعتذر|آسف|آسفة|نأسف|عذراً|عذرا|نأسف على|يؤسفنا|أعتذر|"
    r"للأسف|نأمل في تفهمك|لا تقلق|لا يوجد مشكلة)",
    re.IGNORECASE,
)
_EN_APOLOGIES = re.compile(
    r"\b(sorry|i apologize|we apologize|unfortunately|i'm afraid|"
    r"my apologies|please forgive|pardon)\b",
    re.IGNORECASE,
)

# ──────────────────────────────────────────────────────────────────
# Good news / positive announcements
# ──────────────────────────────────────────────────────────────────
_AR_GOOD_NEWS = re.compile(
    r"(تم بنجاح|تمت العملية|يسعدنا إبلاغك|لديك رصيد|تمت الموافقة|"
    r"مبروك|تهانينا|أبشر|بشرى سارة|خبر سار|حظ موفق|وصل طلبك)",
    re.IGNORECASE,
)
_EN_GOOD_NEWS = re.compile(
    r"\b(confirmed|approved|success|great news|congratulations|"
    r"your order|has been processed|has arrived|is ready)\b",
    re.IGNORECASE,
)

# ──────────────────────────────────────────────────────────────────
# Important information markers
# ──────────────────────────────────────────────────────────────────
_AR_IMPORTANT = re.compile(
    r"(يرجى|يجب|من المهم|انتبه|ملاحظة|تنبيه|احترس|"
    r"الرقم السري|كلمة المرور|رمز التحقق|OTP|PIN|رقم الطلب|"
    r"رسوم|غرامة|موعد|آخر موعد|صالح حتى|انتهاء صلاحية)",
    re.IGNORECASE,
)
_EN_IMPORTANT = re.compile(
    r"\b(please note|important|attention|warning|deadline|"
    r"your pin|otp|passcode|verification code|expires|last chance)\b",
    re.IGNORECASE,
)


def _word_count(text: str) -> int:
    return len(text.split())


class ContextProsodyAnalyzer:
    """
    Lightweight dialog-intent classifier that maps text → ProsodyHints.

    Rules are evaluated in priority order so the most specific intent wins.
    No LLM call, no network, ~0 ms latency.
    """

    def analyze(self, text: str) -> ProsodyHints:
        """
        Infer dialog intent and return prosody recommendations.

        Args:
            text: The TTS input sentence or response.

        Returns:
            ProsodyHints dataclass with speed, pitch, energy, guidance, etc.
        """
        t = text.strip()
        wc = _word_count(t)

        # ── 1. Farewell ──────────────────────────────────────────────
        if _AR_FAREWELLS.search(t) or (
            re.search(r"\b(goodbye|bye|take care)\b", t, re.IGNORECASE)
        ):
            return ProsodyHints(
                intent=DialogIntent.FAREWELL,
                speed=0.93,
                pitch_bias="",
                energy="",
                guidance_scale=2.5,
                num_step=24,
                style_tag="farewell",
                sentence_ending="falling",
                allow_breath=False,
            )

        # ── 2. Greeting ──────────────────────────────────────────────
        if _AR_GREETINGS.search(t) or _EN_GREETINGS.search(t):
            return ProsodyHints(
                intent=DialogIntent.GREETING,
                speed=0.95,
                pitch_bias="",
                energy="",
                guidance_scale=2.8,
                num_step=24,
                style_tag="warm greeting",
                sentence_ending="neutral",
                allow_breath=False,
            )

        # ── 3. Apology ───────────────────────────────────────────────
        if _AR_APOLOGIES.search(t) or _EN_APOLOGIES.search(t):
            return ProsodyHints(
                intent=DialogIntent.APOLOGY,
                speed=0.88,
                pitch_bias="",
                energy="",
                guidance_scale=2.5,
                num_step=28,
                style_tag="empathetic apology",
                sentence_ending="falling",
                allow_breath=False,
            )

        # ── 4. Short confirmation (≤ 10 words) ───────────────────────
        if wc <= 10 and (
            _AR_CONFIRMATIONS.search(t) or _EN_CONFIRMATIONS.search(t)
        ):
            return ProsodyHints(
                intent=DialogIntent.CONFIRMATION,
                speed=1.02,
                pitch_bias="",
                energy="",
                guidance_scale=2.3,
                num_step=18,
                style_tag="crisp confirmation",
                sentence_ending="neutral",
                allow_breath=False,
            )

        # ── 5. Question ──────────────────────────────────────────────
        if _AR_QUESTIONS.search(t) or _EN_QUESTIONS.search(t) or t.endswith("?") or t.endswith("؟"):
            return ProsodyHints(
                intent=DialogIntent.QUESTION,
                speed=0.97,
                pitch_bias="",
                energy="",
                guidance_scale=2.6,
                num_step=24,
                style_tag="curious question",
                sentence_ending="rising",
                allow_breath=False,
            )

        # ── 6. Good news ─────────────────────────────────────────────
        if _AR_GOOD_NEWS.search(t) or _EN_GOOD_NEWS.search(t):
            return ProsodyHints(
                intent=DialogIntent.GOOD_NEWS,
                speed=0.97,
                pitch_bias="",
                energy="",
                guidance_scale=2.7,
                num_step=24,
                style_tag="positive good news",
                sentence_ending="neutral",
                allow_breath=wc > 12,
            )

        # ── 7. Long explanation (> 20 words) — checked BEFORE important_info
        # so that long informational sentences are not trapped by keyword matching
        if wc > 20:
            return ProsodyHints(
                intent=DialogIntent.LONG_EXPLANATION,
                speed=0.94,
                pitch_bias="",
                energy="",
                guidance_scale=2.7,
                num_step=28,
                style_tag="measured explanation",
                sentence_ending="falling",
                allow_breath=True,
            )

        # ── 8. Important information (short to medium, ≤ 20 words) ──
        if _AR_IMPORTANT.search(t) or _EN_IMPORTANT.search(t):
            return ProsodyHints(
                intent=DialogIntent.IMPORTANT_INFO,
                speed=0.85,
                pitch_bias="",
                energy="",
                guidance_scale=3.0,
                num_step=30,
                style_tag="clear important info",
                sentence_ending="falling",
                allow_breath=wc > 15,
            )

        # ── 9. Normal information (default) ──────────────────────────
        return ProsodyHints(
            intent=DialogIntent.NORMAL_INFO,
            speed=1.0,
            pitch_bias="",
            energy="",
            guidance_scale=2.5,
            num_step=24,
            style_tag="neutral normal info",
            sentence_ending="neutral",
            allow_breath=wc > 14,
        )
