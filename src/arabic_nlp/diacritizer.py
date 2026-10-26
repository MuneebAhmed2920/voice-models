"""Arabic Diacritizer (Tashkeel / Harakat) for OmniVoice TTS.

Ensures proper vocalization of Arabic words so that the neural TTS engine
pronounces short vowels (fatha, damma, kasra, sukun, shadda, tanween) accurately.
"""

import re
from typing import Optional


# Arabic Unicode diacritic marks
FATHA = "\u064E"
DAMMA = "\u064F"
KASRA = "\u0650"
SUKUN = "\u0652"
SHADDA = "\u0651"
FATHATAN = "\u064B"
DAMMATAN = "\u064C"
KASRATAN = "\u064D"

ALL_DIACRITICS = [FATHA, DAMMA, KASRA, SUKUN, SHADDA, FATHATAN, DAMMATAN, KASRATAN]
DIACRITICS_REGEX = re.compile(r"[\u064B-\u0652]")

# Common Arabic functional words with standard Tashkeel
COMMON_VOCALIZED_WORDS = {
    "في": "فِي",
    "من": "مِنْ",
    "الى": "إِلَى",
    "إلى": "إِلَى",
    "على": "عَلَى",
    "عن": "عَنْ",
    "مع": "مَعَ",
    "هذا": "هَٰذَا",
    "هذه": "هَٰذِهِ",
    "ذلك": "ذَٰلِكَ",
    "تلك": "تِلْكَ",
    "الذي": "الَّذِي",
    "التي": "الَّتِي",
    "الذين": "الَّذِينَ",
    "اللواتي": "اللَّوَاتِي",
    "ما": "مَا",
    "لا": "لَا",
    "لم": "لَمْ",
    "لن": "لَنْ",
    "لو": "لَوْ",
    "إن": "إِنَّ",
    "ان": "أَنْ",
    "أن": "أَنَّ",
    "إنما": "إِنَّمَا",
    "أنما": "أَنَّمَا",
    "كأن": "كَأَنَّ",
    "لكن": "لَٰكِنَّ",
    "ليت": "لَيْتَ",
    "لعل": "لَعَلَّ",
    "ثم": "ثُمَّ",
    "أو": "أَوْ",
    "أم": "أَمْ",
    "بل": "بَلْ",
    "حتى": "حَتَّى",
    "هو": "هُوَ",
    "هي": "هِيَ",
    "هم": "هُمْ",
    "هن": "هُنَّ",
    "أنا": "أَنَا",
    "نحن": "نَحْنُ",
    "أنت": "أَنْتَ",
    "أنتِ": "أَنْتِ",
    "أنتم": "أَنْتُمْ",
    "أنتن": "أَنْتُنَّ",
    "كل": "كُلّ",
    "غير": "غَيْر",
    "سوف": "سَوْفَ",
    "قد": "قَدْ",
    "عند": "عِنْدَ",
    "بين": "بَيْنَ",
    "قبل": "قَبْلَ",
    "بعد": "بَعْدَ",
    "تحت": "تَحْتَ",
    "فوق": "فَوْقَ",
    "أمام": "أَمَامَ",
    "خلف": "خَلْفَ",
    "وراء": "وَرَاءَ",
    "حول": "حَوْلَ",
    "جدا": "جِدًّا",
    "جداً": "جِدًّا",
    "أيضا": "أَيْضًا",
    "أيضاً": "أَيْضًا",
    "دائما": "دَائِمًا",
    "دائماً": "دَائِمًا",
    "أبدا": "أَبَدًا",
    "أبداً": "أَبَدًا",
    "فقط": "فَقَطْ",
    "نعم": "نَعَمْ",
    "كلا": "كَلَّا",
    "بلى": "بَلَى",
    "مرحبا": "مَرْحَبًا",
    "مرحباً": "مَرْحَبًا",
    "أهلا": "أَهْلًا",
    "أهلاً": "أَهْلًا",
    "وسهلا": "وَسَهْلًا",
    "وسهلاً": "وَسَهْلًا",
    "السلام": "السَّلَام",
    "عليكم": "عَلَيْكُمْ",
    "ورحمة": "وَرَحْمَة",
    "الله": "اللَّه",
    "وبركاته": "وَبَرَكَاتُه",
    "صباح": "صَبَاح",
    "الخير": "الْخَيْر",
    "مساء": "مَسَاء",
    "شكرا": "شُكْرًا",
    "شكراً": "شُكْرًا",
    "عفوا": "عَفْوًا",
    "عفواً": "عَفْوًا",
    "من فضلك": "مِنْ فَضْلِكَ",
    "لو سمحت": "لَوْ سَمَحْتَ",
}


class ArabicDiacritizer:
    """Provides diacritization tools for Arabic text."""

    def __init__(self, mode: str = "rule_based"):
        """
        Initialize the diacritizer.
        
        Args:
            mode: 'rule_based' (fast lexical matching) or 'preserve' (only keep user diacritics)
        """
        self.mode = mode

    @staticmethod
    def strip_diacritics(text: str) -> str:
        """Removes all Arabic diacritical marks."""
        return DIACRITICS_REGEX.sub("", text)

    @staticmethod
    def diacritic_density(text: str) -> float:
        """Calculates ratio of diacritics to Arabic letters (0.0 = undiacritized, >0.5 = heavily vocalized)."""
        arabic_letters = re.findall(r"[\u0621-\u064A]", text)
        if not arabic_letters:
            return 0.0
        diacritics = re.findall(r"[\u064B-\u0652]", text)
        return len(diacritics) / len(arabic_letters)

    def is_vocalized(self, text: str, threshold: float = 0.3) -> bool:
        """Determines if the text already contains sufficient diacritics."""
        return self.diacritic_density(text) >= threshold

    def diacritize(self, text: str) -> str:
        """
        Vocalizes text. If the user already provided diacritics (density >= 0.3),
        we preserve their vocalization. Otherwise, we enrich common functional words
        and particle markers to assist the TTS engine.
        """
        if not text:
            return ""

        # If user has already provided rich tashkeel, respect their input
        if self.is_vocalized(text):
            return text

        # Protect expressive tags like [laughter], [sigh], etc.
        tags = []
        def save_tag(m):
            tags.append(m.group(0))
            return f"__TAG_{len(tags)-1}__"

        text_protected = re.sub(r"\[[a-zA-Z0-9_\-]+\]", save_tag, text)

        # Tokenize words and punctuation
        tokens = re.split(r"(\s+|[،.؟!:;\"\'])", text_protected)
        diacritized_tokens = []

        for token in tokens:
            bare = self.strip_diacritics(token)
            if bare in COMMON_VOCALIZED_WORDS:
                diacritized_tokens.append(COMMON_VOCALIZED_WORDS[bare])
            elif bare.startswith("و") and bare[1:] in COMMON_VOCALIZED_WORDS:
                # Conjunction 'wa' + word
                sub_bare = bare[1:]
                vocalized_sub = COMMON_VOCALIZED_WORDS[sub_bare]
                diacritized_tokens.append(f"وَ{vocalized_sub}")
            elif bare.startswith("ف") and bare[1:] in COMMON_VOCALIZED_WORDS:
                # Particle 'fa' + word
                sub_bare = bare[1:]
                vocalized_sub = COMMON_VOCALIZED_WORDS[sub_bare]
                diacritized_tokens.append(f"فَ{vocalized_sub}")
            elif bare.startswith("ب") and bare[1:] in COMMON_VOCALIZED_WORDS:
                # Preposition 'bi' + word
                sub_bare = bare[1:]
                vocalized_sub = COMMON_VOCALIZED_WORDS[sub_bare]
                diacritized_tokens.append(f"بِ{vocalized_sub}")
            elif bare.startswith("ل") and bare[1:] in COMMON_VOCALIZED_WORDS:
                # Preposition 'li' + word
                sub_bare = bare[1:]
                vocalized_sub = COMMON_VOCALIZED_WORDS[sub_bare]
                diacritized_tokens.append(f"لِ{vocalized_sub}")
            else:
                diacritized_tokens.append(token)

        result = "".join(diacritized_tokens)

        # Restore expressive tags
        for idx, tag in enumerate(tags):
            result = result.replace(f"__TAG_{idx}__", tag)

        return result
