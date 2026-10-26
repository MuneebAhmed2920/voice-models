"""OmniVoice instruct-string planner for Arabic TTS.

OmniVoice supports a STRICT vocabulary for the instruct field.
Valid English items (as of v0.2.1):
  american accent, australian accent, british accent, canadian accent,
  child, chinese accent, elderly, female, high pitch, indian accent,
  japanese accent, korean accent, low pitch, male, middle-aged,
  moderate pitch, portuguese accent, russian accent, teenager,
  very high pitch, very low pitch, whisper, young adult

This module maps dialog intent + dialect → valid instruct string.

For Arabic naturalness we care about:
  - Pitch profile (high / moderate / low / very low)
  - Age register (middle-aged → most stable, professional)
  - Gender (male for current msa_kareem)
  - Speed is NOT an instruct item → controlled via speed= parameter

All invalid tokens are filtered before passing to generate().
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import re


# ──────────────────────────────────────────────────────────────────
# Validated OmniVoice instruct token set (v0.2.1)
# ──────────────────────────────────────────────────────────────────
VALID_INSTRUCT_TOKENS: frozenset = frozenset([
    "american accent", "australian accent", "british accent",
    "canadian accent", "child", "chinese accent", "elderly",
    "female", "high pitch", "indian accent", "japanese accent",
    "korean accent", "low pitch", "male", "middle-aged",
    "moderate pitch", "portuguese accent", "russian accent",
    "teenager", "very high pitch", "very low pitch", "whisper",
    "young adult",
])


class DialogStyle(Enum):
    """Named speaking style for a TTS turn."""
    NEUTRAL = "neutral"
    GREETING = "greeting"
    FAREWELL = "farewell"
    QUESTION = "question"
    CONFIRMATION = "confirmation"
    APOLOGY = "apology"
    GOOD_NEWS = "good_news"
    IMPORTANT = "important"
    EMOTIONAL = "emotional"
    LONG_INFO = "long_info"
    WHISPER = "whisper"


@dataclass
class ProsodyPlan:
    """Complete set of OmniVoice parameters for one utterance."""
    instruct: Optional[str]       # validated instruct string or None
    speed: float                  # 0.85 – 1.05
    guidance_scale: float         # 2.0 – 3.5
    num_step: int                 # 24 – 36
    class_temperature: float      # 0.05 – 0.45 (0 = greedy/robotic)
    t_shift: float                # 0.05 – 0.20
    postprocess_output: bool      # True strips internal silences
    language: str                 # OmniVoice language string
    style: DialogStyle


# ──────────────────────────────────────────────────────────────────
# Intent detection keywords (lightweight, no LLM)
# ──────────────────────────────────────────────────────────────────
_GREET = re.compile(
    r"(مرحب|أهل|السلام عليكم|صباح الخير|مساء الخير|كيفك|كيف حالك|يا هلا|هلا والله|أهلين|طاب يومك)",
    re.IGNORECASE,
)
_FAREWELL = re.compile(
    r"(مع السلامة|وداعاً|وداعا|إلى اللقاء|تصبح على خير|في أمان الله|شكراً لتواصلك|لا تتردد)",
    re.IGNORECASE,
)
_APOLOGY = re.compile(
    r"(نعتذر|آسف|آسفة|نأسف|عذراً|عذرا|للأسف|نأمل في تفهمك|أعتذر|يؤسفنا)",
    re.IGNORECASE,
)
_QUESTION = re.compile(
    r"(هل|كيف|متى|أين|من|ما|ماذا|لماذا|أي|كم|هل يمكن).+[؟?]",
    re.IGNORECASE,
)
_CONFIRM = re.compile(
    r"^(نعم|بالتأكيد|أكيد|طبعاً|طبعا|تمام|ماشي|حسناً|حسنا|سيتم|شكراً)[.!،]?\s*$",
    re.IGNORECASE,
)
_GOOD_NEWS = re.compile(
    r"(تم بنجاح|تمت العملية|يسعدنا|مبروك|تهانينا|أبشر|بشرى سارة|خبر سار|وصل طلبك)",
    re.IGNORECASE,
)
_IMPORTANT = re.compile(
    r"(يرجى|يجب|من المهم|انتبه|ملاحظة|تنبيه|الرقم السري|كلمة المرور|رمز التحقق|OTP|PIN|موعد)",
    re.IGNORECASE,
)


def _build_instruct(tokens: list) -> Optional[str]:
    """Build a valid instruct string, filtering unsupported tokens."""
    valid = [t for t in tokens if t in VALID_INSTRUCT_TOKENS]
    if not valid:
        return None
    return ", ".join(valid)


class ArabicProsodyPlanner:
    """
    Maps Arabic text + dialect → OmniVoice generation parameters.

    Respects the strict OmniVoice instruct vocabulary.
    Uses class_temperature > 0 to enable non-greedy (natural) sampling.
    """

    # Default speaker gender — can be changed per deployment
    DEFAULT_GENDER = "male"
    DEFAULT_AGE = "middle-aged"

    def __init__(
        self,
        dialect: str = "saudi",
        gender: str = "male",
        age: str = "middle-aged",
    ):
        self.dialect = dialect.lower()
        self.gender = gender if gender in VALID_INSTRUCT_TOKENS else "male"
        self.age = age if age in VALID_INSTRUCT_TOKENS else "middle-aged"
        self.language = self._resolve_language(dialect)

    def _resolve_language(self, dialect: str) -> str:
        """Map dialect name to OmniVoice language string."""
        mapping = {
            "saudi": "najdi arabic",       # Saudi — Najdi (Riyadh/Najd area)
            "hijazi": "hijazi arabic",      # Saudi — Hijazi (Jeddah/Makkah)
            "najdi": "najdi arabic",
            "gulf": "gulf arabic",
            "uae": "gulf arabic",
            "kuwait": "gulf arabic",
            "bahrain": "bahrain arabic",
            "egyptian": "egyptian arabic",
            "egypt": "egyptian arabic",
            "levantine": "levantine arabic",
            "syrian": "levantine arabic",
            "jordanian": "levantine arabic",
            "moroccan": "moroccan arabic",
            "msa": "standard arabic",
            "standard": "standard arabic",
            "fusha": "standard arabic",
        }
        return mapping.get(dialect, "najdi arabic")

    def analyze_and_plan(self, text: str) -> ProsodyPlan:
        """
        Analyze text intent and return complete ProsodyPlan.

        Args:
            text: Arabic (or mixed) TTS input.

        Returns:
            ProsodyPlan with all OmniVoice parameters set.
        """
        t = text.strip()
        wc = len(t.split())

        # ── Intent detection (priority order) ──────────────────────
        if _FAREWELL.search(t):
            return self._make_plan(
                style=DialogStyle.FAREWELL,
                pitch="low pitch",
                speed=0.94,
                guidance=2.6,
                steps=28,
                temp=0.15,
                postprocess=True,
            )

        if _GREET.search(t):
            return self._make_plan(
                style=DialogStyle.GREETING,
                pitch="moderate pitch",
                speed=0.97,
                guidance=2.8,
                steps=30,
                temp=0.20,
                postprocess=True,
            )

        if _APOLOGY.search(t):
            return self._make_plan(
                style=DialogStyle.APOLOGY,
                pitch="low pitch",
                speed=0.90,
                guidance=2.6,
                steps=30,
                temp=0.15,
                postprocess=True,
            )

        if wc <= 8 and _CONFIRM.match(t):
            return self._make_plan(
                style=DialogStyle.CONFIRMATION,
                pitch="moderate pitch",
                speed=1.00,
                guidance=2.4,
                steps=24,
                temp=0.10,
                postprocess=True,
            )

        if _QUESTION.search(t) or t.rstrip().endswith("؟") or t.rstrip().endswith("?"):
            return self._make_plan(
                style=DialogStyle.QUESTION,
                pitch="moderate pitch",
                speed=0.97,
                guidance=2.6,
                steps=28,
                temp=0.20,
                postprocess=True,
            )

        if _GOOD_NEWS.search(t):
            return self._make_plan(
                style=DialogStyle.GOOD_NEWS,
                pitch="moderate pitch",
                speed=0.97,
                guidance=2.7,
                steps=28,
                temp=0.20,
                postprocess=True,
            )

        # Long explanations → more steps, slightly slower, preserve pauses
        if wc > 20:
            return self._make_plan(
                style=DialogStyle.LONG_INFO,
                pitch="moderate pitch",
                speed=0.95,
                guidance=2.7,
                steps=32,
                temp=0.20,
                # Keep internal pauses for long phrases
                postprocess=False,
            )

        if _IMPORTANT.search(t):
            return self._make_plan(
                style=DialogStyle.IMPORTANT,
                pitch="low pitch",
                speed=0.88,
                guidance=3.0,
                steps=32,
                temp=0.10,
                postprocess=True,
            )

        # Default: neutral conversational
        return self._make_plan(
            style=DialogStyle.NEUTRAL,
            pitch="moderate pitch",
            speed=1.00,
            guidance=2.5,
            steps=28,
            temp=0.20,
            postprocess=True,
        )

    def _make_plan(
        self,
        style: DialogStyle,
        pitch: str,
        speed: float,
        guidance: float,
        steps: int,
        temp: float,
        postprocess: bool,
    ) -> ProsodyPlan:
        """Construct a ProsodyPlan with validated instruct string."""
        instruct_tokens = [self.gender, self.age]
        if pitch and pitch in VALID_INSTRUCT_TOKENS:
            instruct_tokens.append(pitch)
        instruct = _build_instruct(instruct_tokens)

        return ProsodyPlan(
            instruct=instruct,
            speed=speed,
            guidance_scale=guidance,
            num_step=steps,
            class_temperature=temp,
            t_shift=0.10,
            postprocess_output=postprocess,
            language=self.language,
            style=style,
        )

    def validate_instruct(self, instruct_str: str) -> str:
        """
        Validate and clean an instruct string.
        Removes any tokens not in the OmniVoice vocabulary.
        Returns cleaned string or None if nothing valid remains.
        """
        if not instruct_str:
            return None
        tokens = [t.strip() for t in instruct_str.split(",")]
        valid = [t for t in tokens if t in VALID_INSTRUCT_TOKENS]
        return ", ".join(valid) if valid else None
