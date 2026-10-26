"""Arabic Voice Presets and Voice Cloning Profile Manager.

Provides:
- 15+ built-in studio voices (Male & Female)
- Multi-dialect profiles (MSA, Egyptian, Gulf, Levantine)
- Style attributes (Formal News, Storyteller, Conversational, Documentary)
- Reference audio manager for zero-shot voice cloning
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class VoiceProfile:
    """Represents a speaker profile in the Arabic TTS system."""
    id: str
    name_ar: str
    name_en: str
    gender: str          # 'male' or 'female'
    dialect: str         # 'msa', 'egyptian', 'gulf', 'levantine'
    style: str           # 'formal', 'conversational', 'news', 'storyteller', 'warm'
    engine: str          # 'voicetut' or 'omnivoice'
    speaker_key: Optional[str] = None      # VoiceTut speaker name if applicable
    instruct_prompt: Optional[str] = None  # OmniVoice natural language prompt
    sample_text: str = "مرحباً بكم في نظام أومني فويس لتوليد الصوت العربي الطبيعي."
    ref_audio_path: Optional[str] = None   # Path to reference audio if cloned
    ref_transcript: Optional[str] = None


# Built-in Studio Voices
BUILTIN_VOICES: List[VoiceProfile] = [
    # VoiceTut Egyptian Studio Voices
    VoiceProfile(
        id="egy_mohamed",
        name_ar="محمد (مصري - محادثة)",
        name_en="Mohamed (Egyptian - Conversational)",
        gender="male",
        dialect="egyptian",
        style="conversational",
        engine="voicetut",
        speaker_key="Mohamed",
        sample_text="ازيك عامل ايه؟ مرحب بيك في النظام الصوتي الجديد، هنقدر نتكلم مع بعض بكل سهولة."
    ),
    VoiceProfile(
        id="egy_asmaa",
        name_ar="أسماء (مصرية - ودودة وواضحة)",
        name_en="Asmaa (Egyptian - Friendly & Clear)",
        gender="female",
        dialect="egyptian",
        style="warm",
        engine="voicetut",
        speaker_key="Asmaa",
        sample_text="أهلاً بيك! النهاردة هنقدم تجربة صوتية متميزة تدعم العربي والإنجليزي سوا."
    ),
    VoiceProfile(
        id="egy_sayed",
        name_ar="سيد (مصري - وقور وهادئ)",
        name_en="Sayed (Egyptian - Calm & Authoritative)",
        gender="male",
        dialect="egyptian",
        style="formal",
        engine="voicetut",
        speaker_key="Sayed",
        sample_text="مساء الخير، يسعدنا وجودكم معنا في هذه الحلقة الخاصة والحديث عن التقنيات الحديثة."
    ),
    VoiceProfile(
        id="egy_nour",
        name_ar="نور (مصرية - شابة وإعلامية)",
        name_en="Nour (Egyptian - Dynamic & Broadcast)",
        gender="female",
        dialect="egyptian",
        style="news",
        engine="voicetut",
        speaker_key="Nour",
        sample_text="أهلاً بكم في موجز الأخبار السريع، نستعرض فيه أبرز التطورات والمستجدات لليوم."
    ),
    VoiceProfile(
        id="egy_amr",
        name_ar="عمرو (مصري - إذاعي وحيوي)",
        name_en="Amr (Egyptian - Radio & Energetic)",
        gender="male",
        dialect="egyptian",
        style="conversational",
        engine="voicetut",
        speaker_key="Amr",
        sample_text="صباح الفل والجمال على كل السامعين، يومكم سعيد ومليان طاقة ونجاح بإذن الله."
    ),
    VoiceProfile(
        id="egy_farida",
        name_ar="فريدة (مصرية - ناعمة وقصصية)",
        name_en="Farida (Egyptian - Soft & Storyteller)",
        gender="female",
        dialect="egyptian",
        style="storyteller",
        engine="voicetut",
        speaker_key="Farida",
        sample_text="في قديم الزمان، كانت هناك مدينة جميلة تقع بين ضفاف النيل العظيم."
    ),

    # Modern Standard Arabic (Fusha) OmniVoice Voices
    VoiceProfile(
        id="msa_kareem",
        name_ar="كريم (فصحى - مذيع إخباري وقور)",
        name_en="Kareem (MSA - Formal News Broadcaster)",
        gender="male",
        dialect="msa",
        style="news",
        engine="omnivoice",
        instruct_prompt="male, middle-aged, low pitch",
        sample_text="سَيِّدَاتِي وَآنِسَاتِي، طَابَتْ أَوْقَاتُكُمْ بِكُلِّ خَيْر، نَسْتَعْرِضُ مَعَكُمْ آخِرَ التَّطَوُّرَاتِ العَالَمِيَّة."
    ),
    VoiceProfile(
        id="msa_marium",
        name_ar="مريم (فصحى - وثائقية فصيحة)",
        name_en="Mariam (MSA - Documentary & Narration)",
        gender="female",
        dialect="msa",
        style="storyteller",
        engine="omnivoice",
        instruct_prompt="female, young adult, moderate pitch",
        sample_text="فِي أَعْمَاقِ التَّارِيخِ، نَشَأَتْ حَضَارَاتٌ رَسَّخَتْ قِيَمَ المَعْرِفَةِ وَالعِلْمِ وَالبِنَاء."
    ),
    VoiceProfile(
        id="msa_tariq",
        name_ar="طارق (فصحى - تعليمي وواضح)",
        name_en="Tariq (MSA - Educational & Articulate)",
        gender="male",
        dialect="msa",
        style="formal",
        engine="omnivoice",
        instruct_prompt="male, young adult, moderate pitch",
        sample_text="أَهْلًا بِكُمْ فِي هَذَا الدَّرْسِ التَّعْلِيمِيِّ حَوْلَ أَسَاسِيَّاتِ الذَّكَاءِ الاصْطِنَاعِيِّ."
    ),
    VoiceProfile(
        id="msa_layla",
        name_ar="ليلى (فصحى - مساعد صوتي وذكي)",
        name_en="Layla (MSA - Smart Assistant)",
        gender="female",
        dialect="msa",
        style="warm",
        engine="omnivoice",
        instruct_prompt="female, young adult, high pitch",
        sample_text="مَرْحَبًا! أَنَا مُسَاعِدُكَ الذَّكِيُّ، كَيْفَ يُمْكِنُنِي مُسَاعَدَتُكَ اليَوْمَ؟"
    ),

    # Gulf & Regional Dialect Profiles
    VoiceProfile(
        id="gulf_sultan",
        name_ar="سلطان (لهجة خليجية - محادثة هادئة)",
        name_en="Sultan (Gulf / Khaleeji - Calm & Warm)",
        gender="male",
        dialect="gulf",
        style="conversational",
        engine="omnivoice",
        instruct_prompt="male, middle-aged, low pitch",
        sample_text="يا هلا ومسهلا فيكم، نورتونا وشرفتونا اليوم، عسى ربي يحفظكم ويسعدكم."
    ),
    VoiceProfile(
        id="levant_reem",
        name_ar="ريم (لهجة شامية - لطيفة وحيوية)",
        name_en="Reem (Levantine - Bright & Friendly)",
        gender="female",
        dialect="levantine",
        style="warm",
        engine="omnivoice",
        instruct_prompt="female, young adult, high pitch",
        sample_text="يسعدلي صباحكن ومساكن جميعاً، انشالله تكونو بألف خير وصحة وسلامة."
    ),
    VoiceProfile(
        id="high_laugh_host",
        name_ar="ماجد (نبرة عالية، ضحك وبهجة، استعراض بشري)",
        name_en="Majed (High Pitch, Laughter & Upbeat Host)",
        gender="male",
        dialect="msa",
        style="conversational",
        engine="omnivoice",
        ref_audio_path="test_high_laugh.wav",
        ref_transcript="يا هلا والله! [laughter] نورتونا النهاردة!",
        instruct_prompt="male, high pitch",
        sample_text="يا هلا والله بكل الحبايب! [laughter] نورتونا ومش قادر أوصفلكم قد إيه أنا مبسوط بوجودكم النهاردة!"
    ),
]


def resolve_instruct_prompt(prompt: Optional[str]) -> Optional[str]:
    """
    Parses Arabic or English descriptions into valid OmniVoice instruct tags.
    Valid items: male, female, child, teenager, young adult, middle-aged, elderly,
    low pitch, moderate pitch, high pitch, very low pitch, very high pitch, whisper.
    """
    if not prompt or not prompt.strip():
        return None

    # Check if user already provided valid comma-separated tags
    valid_tokens = {
        "male", "female", "child", "teenager", "young adult", "middle-aged", "elderly",
        "low pitch", "moderate pitch", "high pitch", "very low pitch", "very high pitch", "whisper"
    }
    raw_tokens = [t.strip().lower() for t in prompt.split(",")]
    matching = [t for t in raw_tokens if t in valid_tokens]
    if matching:
        return ", ".join(matching)

    # Keyword search across Arabic and English terms
    p_lower = prompt.lower()
    items = []

    # Gender
    if any(w in p_lower for w in ["رجل", "ذكر", "شاب", "مذكر", "male", "man", "boy"]):
        items.append("male")
    elif any(w in p_lower for w in ["امرأة", "أنثى", "سيدة", "فتاة", "مؤنث", "female", "woman", "girl"]):
        items.append("female")

    # Age
    if any(w in p_lower for w in ["طفل", "صغير", "child", "kid"]):
        items.append("child")
    elif any(w in p_lower for w in ["مراهق", "فتى", "teenager", "teen"]):
        items.append("teenager")
    elif any(w in p_lower for w in ["كبير", "عجوز", "مسن", "شيخ", "elderly", "old"]):
        items.append("elderly")
    elif any(w in p_lower for w in ["متوسط", "middle-aged"]):
        items.append("middle-aged")
    elif any(w in p_lower for w in ["شاب", "شابة", "young"]):
        items.append("young adult")

    # Pitch
    if any(w in p_lower for w in ["عميق", "منخفض", "وقور", "رخيم", "deep", "low pitch", "low"]):
        items.append("low pitch")
    elif any(w in p_lower for w in ["عالي", "رفيع", "حرج", "high pitch", "high"]):
        items.append("high pitch")
    elif any(w in p_lower for w in ["همس", "خافت", "whisper"]):
        items.append("whisper")
    elif any(w in p_lower for w in ["معتدل", "متوسط النبرة", "moderate"]):
        items.append("moderate pitch")

    return ", ".join(items) if items else None


class VoiceManager:
    """Manages built-in and user-uploaded custom cloned voices."""

    def __init__(self, custom_voices_dir: str = "custom_voices"):
        self.custom_voices_dir = Path(custom_voices_dir)
        self.custom_voices_dir.mkdir(parents=True, exist_ok=True)
        self._voices: Dict[str, VoiceProfile] = {v.id: v for v in BUILTIN_VOICES}
        self._load_custom_voices()

    def _load_custom_voices(self):
        """Loads cloned voices saved in custom directory."""
        metadata_file = self.custom_voices_dir / "voices.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        profile = VoiceProfile(**item)
                        self._voices[profile.id] = profile
            except Exception as e:
                print(f"[VoiceManager] Warning: could not load custom voices: {e}")

    def _save_custom_voices(self):
        """Saves custom voices registry to disk."""
        metadata_file = self.custom_voices_dir / "voices.json"
        custom_list = [
            profile.__dict__
            for profile in self._voices.values()
            if profile.id.startswith("custom_")
        ]
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(custom_list, f, ensure_ascii=False, indent=2)

    def list_voices(self, dialect: Optional[str] = None, gender: Optional[str] = None) -> List[VoiceProfile]:
        """Returns list of voices filtered by dialect and/or gender."""
        voices = list(self._voices.values())
        if dialect:
            voices = [v for v in voices if v.dialect.lower() == dialect.lower()]
        if gender:
            voices = [v for v in voices if v.gender.lower() == gender.lower()]
        return voices

    def get_voice(self, voice_id: str) -> Optional[VoiceProfile]:
        """Gets voice profile by ID."""
        return self._voices.get(voice_id)

    def register_custom_voice(
        self,
        name: str,
        audio_file_path: str,
        transcript: Optional[str] = None,
        dialect: str = "custom",
        gender: str = "neutral"
    ) -> VoiceProfile:
        """Registers a newly cloned voice from reference audio."""
        import uuid
        voice_id = f"custom_{uuid.uuid4().hex[:8]}"
        profile = VoiceProfile(
            id=voice_id,
            name_ar=f"صوت مخصص: {name}",
            name_en=f"Custom Voice: {name}",
            gender=gender,
            dialect=dialect,
            style="cloned",
            engine="omnivoice",
            ref_audio_path=audio_file_path,
            ref_transcript=transcript
        )
        self._voices[voice_id] = profile
        self._save_custom_voices()
        return profile
