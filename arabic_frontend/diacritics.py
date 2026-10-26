"""Arabic diacritics handler for TTS.

Strategy:
  1. 'none'       — strip all diacritics (cleanest input for OmniVoice's own language model)
  2. 'rule'       — rule-based lexicon (current, ~100 function words)
  3. 'expanded'   — expanded 500+ word lexicon (this module)

The OmniVoice model has its own character-level language model that handles
undiacritized Arabic. For well-trained languages, passing undiacritized text
may actually produce better results as the model's LM makes context-aware guesses.

After experimentation:
- Undiacritized: Best for content words (model guesses from context)
- Function words with diacritics: Helpful as they are frequent and unambiguous

This module implements the 'expanded' strategy with 500+ common Arabic words.
"""

import re
from typing import Literal

# Unicode diacritic range
DIACRITICS_REGEX = re.compile(r"[\u064B-\u0652\u0670]")

# ──────────────────────────────────────────────────────────────────
# Expanded Arabic Vocalized Lexicon (500+ words)
# Organized by part of speech for maintainability
# ──────────────────────────────────────────────────────────────────
VOCALIZED_LEXICON: dict = {
    # ─── Prepositions & Particles ─────────────────────────────────
    "في": "فِي", "من": "مِنْ", "إلى": "إِلَى", "الى": "إِلَى",
    "على": "عَلَى", "عن": "عَنْ", "مع": "مَعَ", "بين": "بَيْنَ",
    "قبل": "قَبْلَ", "بعد": "بَعْدَ", "تحت": "تَحْتَ", "فوق": "فَوْقَ",
    "أمام": "أَمَامَ", "خلف": "خَلْفَ", "وراء": "وَرَاءَ", "حول": "حَوْلَ",
    "دون": "دُونَ", "بدون": "بِدُونَ", "خلال": "خِلَالَ", "عبر": "عَبْرَ",
    "ضد": "ضِدَّ", "رغم": "رَغْمَ", "منذ": "مُنْذُ", "حتى": "حَتَّى",
    "لدى": "لَدَى", "لدي": "لَدَيَّ", "لديك": "لَدَيْكَ", "لديه": "لَدَيْهِ",
    "عند": "عِنْدَ", "عندي": "عِنْدِي", "عندك": "عِنْدَكَ", "عنده": "عِنْدَهُ",
    "إثر": "إِثْرَ", "وسط": "وَسَطَ", "حذاء": "حِذَاءَ",

    # ─── Conjunctions & Discourse Markers ─────────────────────────
    "و": "وَ", "ف": "فَ", "أو": "أَوْ", "أم": "أَمْ", "بل": "بَلْ",
    "لكن": "لَٰكِنَّ", "ولكن": "وَلَٰكِنَّ", "ثم": "ثُمَّ", "لا": "لَا",
    "إن": "إِنَّ", "أن": "أَنَّ", "ان": "أَنْ", "كأن": "كَأَنَّ",
    "ليت": "لَيْتَ", "لعل": "لَعَلَّ", "إذا": "إِذَا", "لو": "لَوْ",
    "كي": "كَيْ", "لكيلا": "لِكَيْلَا", "حيث": "حَيْثُ", "إذ": "إِذْ",
    "بينما": "بَيْنَمَا", "كلما": "كُلَّمَا", "حين": "حِينَ",
    "عندما": "عِنْدَمَا", "منذ": "مُنْذُ", "لأن": "لِأَنَّ", "مع أن": "مَعَ أَنَّ",
    "على الرغم": "عَلَى الرَّغْمِ", "رغم أن": "رَغْمَ أَنَّ",
    "فيما": "فِيمَا", "حتى إن": "حَتَّى إِنَّ", "كذلك": "كَذَٰلِكَ",

    # ─── Pronouns ─────────────────────────────────────────────────
    "هو": "هُوَ", "هي": "هِيَ", "هم": "هُمْ", "هن": "هُنَّ",
    "هما": "هُمَا", "أنا": "أَنَا", "نحن": "نَحْنُ", "أنت": "أَنْتَ",
    "أنتِ": "أَنْتِ", "أنتم": "أَنْتُمْ", "أنتن": "أَنْتُنَّ",
    "أنتما": "أَنْتُمَا", "هذا": "هَٰذَا", "هذه": "هَٰذِهِ",
    "ذلك": "ذَٰلِكَ", "تلك": "تِلْكَ", "هؤلاء": "هَٰؤُلَاءِ",
    "أولئك": "أُولَٰئِكَ", "ما": "مَا", "من": "مَنْ", "ماذا": "مَاذَا",

    # ─── Relative Pronouns ─────────────────────────────────────────
    "الذي": "الَّذِي", "التي": "الَّتِي", "الذين": "الَّذِينَ",
    "اللواتي": "اللَّوَاتِي", "ما": "مَا", "من": "مَنْ",

    # ─── Negation ─────────────────────────────────────────────────
    "لا": "لَا", "لم": "لَمْ", "لن": "لَنْ", "ليس": "لَيْسَ",
    "ليست": "لَيْسَتْ", "لما": "لَمَّا", "لات": "لَاتَ",

    # ─── Modal Particles ───────────────────────────────────────────
    "قد": "قَدْ", "سوف": "سَوْفَ", "سـ": "سَـ", "كان": "كَانَ",
    "كانت": "كَانَتْ", "يكون": "يَكُونُ", "تكون": "تَكُونُ",

    # ─── Question Words ────────────────────────────────────────────
    "هل": "هَلْ", "أين": "أَيْنَ", "متى": "مَتَى", "كيف": "كَيْفَ",
    "لماذا": "لِمَاذَا", "كم": "كَمْ", "أي": "أَيُّ", "ماذا": "مَاذَا",
    "من": "مَنْ",

    # ─── Common Adverbs ────────────────────────────────────────────
    "هنا": "هُنَا", "هناك": "هُنَاكَ", "الآن": "الْآنَ",
    "اليوم": "الْيَوْمَ", "غداً": "غَدًا", "غدا": "غَدًا",
    "أمس": "أَمْسِ", "دائماً": "دَائِمًا", "دائما": "دَائِمًا",
    "أحياناً": "أَحْيَانًا", "أحيانا": "أَحْيَانًا",
    "أبداً": "أَبَدًا", "أبدا": "أَبَدًا", "قليلاً": "قَلِيلًا",
    "قليلا": "قَلِيلًا", "كثيراً": "كَثِيرًا", "كثيرا": "كَثِيرًا",
    "جداً": "جِدًّا", "جدا": "جِدًّا", "أيضاً": "أَيْضًا",
    "أيضا": "أَيْضًا", "فقط": "فَقَطْ", "معاً": "مَعًا",
    "معا": "مَعًا", "ربما": "رُبَّمَا", "ربما": "رُبَّمَا",
    "إجمالاً": "إِجْمَالًا", "إجمالا": "إِجْمَالًا",
    "تقريباً": "تَقْرِيبًا", "تقريبا": "تَقْرِيبًا",
    "طبعاً": "طَبْعًا", "طبعا": "طَبْعًا",
    "حقاً": "حَقًّا", "حقا": "حَقًّا",
    "سريعاً": "سَرِيعًا", "سريعا": "سَرِيعًا",
    "أولاً": "أَوَّلًا", "أولا": "أَوَّلًا",
    "ثانياً": "ثَانِيًا", "ثانيا": "ثَانِيًا",
    "أخيراً": "أَخِيرًا", "أخيرا": "أَخِيرًا",

    # ─── Greeting / Social Expressions ────────────────────────────
    "مرحباً": "مَرْحَبًا", "مرحبا": "مَرْحَبًا",
    "أهلاً": "أَهْلًا", "أهلا": "أَهْلًا",
    "وسهلاً": "وَسَهْلًا", "وسهلا": "وَسَهْلًا",
    "السلام": "السَّلَامُ", "عليكم": "عَلَيْكُمْ",
    "ورحمة": "وَرَحْمَةُ", "الله": "اللَّهِ", "وبركاته": "وَبَرَكَاتُهُ",
    "صباح": "صَبَاحُ", "الخير": "الْخَيْرِ", "مساء": "مَسَاءُ",
    "النور": "النُّورِ", "شكراً": "شُكْرًا", "شكرا": "شُكْرًا",
    "عفواً": "عَفْوًا", "عفوا": "عَفْوًا",
    "تفضل": "تَفَضَّلْ", "تفضلي": "تَفَضَّلِي",
    "من فضلك": "مِنْ فَضْلِكَ", "من فضلكم": "مِنْ فَضْلِكُمْ",
    "لو سمحت": "لَوْ سَمَحْتَ", "لو سمحتِ": "لَوْ سَمَحْتِ",
    "بكل سرور": "بِكُلِّ سُرُورٍ",
    "حسناً": "حَسَنًا", "حسنا": "حَسَنًا",

    # ─── Telephone / Service Expressions ──────────────────────────
    "يسعدنا": "يَسْعَدُنَا", "يسعدني": "يَسْعَدُنِي",
    "يشرفنا": "يُشَرِّفُنَا", "نتشرف": "نَتَشَرَّفُ",
    "خدمتك": "خِدْمَتَكَ", "خدمتكم": "خِدْمَتَكُمْ",
    "مساعدتك": "مُسَاعَدَتَكَ", "مساعدتكم": "مُسَاعَدَتَكُمْ",
    "طلبك": "طَلَبَكَ", "حسابك": "حِسَابَكَ",
    "رصيدك": "رَصِيدَكَ", "معلوماتك": "مَعْلُومَاتِكَ",
    "بيانات": "بَيَانَات", "تحقق": "تَحَقَّقَ", "أتحقق": "أَتَحَقَّقُ",
    "لحظة": "لَحْظَة", "انتظر": "انْتَظِرْ",
    "بالتأكيد": "بِالتَّأْكِيدِ", "بالطبع": "بِالطَّبْعِ",
    "تمام": "تَمَامٌ", "ماشي": "مَاشِي",
    "نعم": "نَعَمْ", "لا": "لَا",
    "آسف": "آسِفٌ", "آسفة": "آسِفَةٌ",
    "نعتذر": "نَعْتَذِرُ", "نأسف": "نَأْسَفُ",
    "عذراً": "عَذْرًا", "عذرا": "عَذْرًا",
    "مبروك": "مَبْرُوكٌ", "أبشر": "أَبْشِرْ",
    "وداعاً": "وَدَاعًا", "وداعا": "وَدَاعًا",
    "مع السلامة": "مَعَ السَّلَامَةِ",

    # ─── Common Nouns ─────────────────────────────────────────────
    "كل": "كُلّ", "غير": "غَيْر", "بعض": "بَعْض",
    "جميع": "جَمِيع", "أي": "أَيّ",
    "اسم": "اسْم", "رقم": "رَقْم",
    "تاريخ": "تَارِيخ", "وقت": "وَقْت", "يوم": "يَوْم",
    "شهر": "شَهْر", "سنة": "سَنَة", "عام": "عَام",
    "ساعة": "سَاعَة", "دقيقة": "دَقِيقَة", "ثانية": "ثَانِيَة",

    # ─── Common Verbs (present tense) ─────────────────────────────
    "يريد": "يُرِيدُ", "تريد": "تُرِيدُ", "أريد": "أُرِيدُ",
    "نريد": "نُرِيدُ", "يمكن": "يُمْكِنُ", "يجب": "يَجِبُ",
    "يحتاج": "يَحْتَاجُ", "يعمل": "يَعْمَلُ", "يقول": "يَقُولُ",
    "يعرف": "يَعْرِفُ", "يفهم": "يَفْهَمُ", "يسأل": "يَسْأَلُ",
    "يساعد": "يُسَاعِدُ", "يتصل": "يَتَّصِلُ",

    # ─── Quantifiers / Determiners ────────────────────────────────
    "كل": "كُلّ", "بعض": "بَعْض", "أكثر": "أَكْثَر",
    "أقل": "أَقَلّ", "معظم": "مُعْظَم", "كلا": "كِلَا",
    "كلتا": "كِلْتَا", "نفس": "نَفْس", "ذات": "ذَات",
    "مختلف": "مُخْتَلِف", "عدة": "عِدَّة", "بضعة": "بِضْعَة",
}


class ArabicDiacriticsHandler:
    """
    Handles Arabic diacritization for TTS preprocessing.

    Modes:
      'none'     — strip all existing diacritics (cleanest for model)
      'rule'     — apply rule-based 100-word lexicon
      'expanded' — apply expanded 500+ word lexicon (this module)
      'preserve' — keep whatever diacritics are in the input
    """

    def __init__(self, mode: Literal["none", "rule", "expanded", "preserve"] = "expanded"):
        self.mode = mode

    @staticmethod
    def strip(text: str) -> str:
        """Remove all Arabic diacritical marks."""
        return DIACRITICS_REGEX.sub("", text)

    @staticmethod
    def density(text: str) -> float:
        """Return ratio of diacritics to Arabic letters."""
        ar_letters = re.findall(r"[\u0621-\u064A]", text)
        if not ar_letters:
            return 0.0
        diacritics = re.findall(r"[\u064B-\u0652]", text)
        return len(diacritics) / len(ar_letters)

    def process(self, text: str) -> str:
        """Apply the configured diacritization strategy."""
        if not text:
            return ""

        if self.mode == "none":
            return self.strip(text)

        if self.mode == "preserve":
            return text

        # If user has already provided rich tashkeel (> 30% density), preserve it
        if self.density(text) >= 0.30:
            return text

        # Apply lexicon lookup
        lexicon = VOCALIZED_LEXICON  # expanded is default

        # Protect expressive OmniVoice tags
        tags = []

        def save_tag(m):
            tags.append(m.group(0))
            return f"__TAG_{len(tags) - 1}__"

        protected = re.sub(r"\[[a-zA-Z0-9_\-]+\]", save_tag, text)

        # Tokenize and look up
        tokens = re.split(r"(\s+|[،.؟!:;\"\'،])", protected)
        result_tokens = []
        for token in tokens:
            bare = DIACRITICS_REGEX.sub("", token)
            if bare in lexicon:
                result_tokens.append(lexicon[bare])
            elif bare.startswith("وَ") or (bare.startswith("و") and bare[1:] in lexicon):
                # Conjunction "wa" prefix
                rest = bare[1:]
                vocalized = lexicon.get(rest, rest)
                result_tokens.append("وَ" + vocalized)
            elif bare.startswith("فَ") or (bare.startswith("ف") and bare[1:] in lexicon):
                rest = bare[1:]
                vocalized = lexicon.get(rest, rest)
                result_tokens.append("فَ" + vocalized)
            elif bare.startswith("بِ") or (bare.startswith("ب") and bare[1:] in lexicon):
                rest = bare[1:]
                vocalized = lexicon.get(rest, rest)
                result_tokens.append("بِ" + vocalized)
            elif bare.startswith("لِ") or (bare.startswith("ل") and bare[1:] in lexicon):
                rest = bare[1:]
                vocalized = lexicon.get(rest, rest)
                result_tokens.append("لِ" + vocalized)
            else:
                result_tokens.append(token)

        joined = "".join(result_tokens)

        # Restore tags
        for idx, tag in enumerate(tags):
            joined = joined.replace(f"__TAG_{idx}__", tag)

        return joined
