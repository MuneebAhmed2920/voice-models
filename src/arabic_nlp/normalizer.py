"""Arabic Text Normalizer for Speech Synthesis (OmniVoice).

Handles:
- Western (0-9) and Eastern Arabic (٠-٩) numerals
- Compound numbers (units, tens, hundreds, thousands, millions, billions)
- Dialect modes: Modern Standard Arabic (MSA / Fusha) and Egyptian (Masri)
- Time expressions (e.g., 3:30 -> الثالثة والنصف / تلاتة ونص)
- Dates (e.g., 15/09/2026)
- Currencies ($, €, £, EGP, SAR, AED, KWD)
- Percentages (e.g., 50% -> خمسون بالمائة)
- Fractions (e.g., 1/2 -> نصف, 1/4 -> ربع)
- Abbreviations expansion (د. -> دكتور, إلخ -> إلى آخره)
- Tatweel / Kashida stripping and punctuation cleanup
"""

import re
from typing import Dict, List, Optional, Tuple, Union


# Digit mappings
EASTERN_TO_WESTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Arabic Number Words - Modern Standard Arabic
ONES_MSA = ["", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة"]
ONES_FEMININE_MSA = ["", "واحدة", "اثنتان", "ثلاث", "أربع", "خمس", "ست", "سبع", "ثمان", "تسع"]
TEENS_MSA = [
    "عشرة", "أحد عشر", "اثنا عشر", "ثلاثة عشر", "أربعة عشر", "خمسة عشر",
    "ستة عشر", "سبعة عشر", "ثمانية عشر", "تسعة عشر"
]
TENS_MSA = ["", "عشرة", "عشرون", "ثلاثون", "أربعون", "خمسون", "ستون", "سبعون", "ثمانون", "تسعون"]
HUNDREDS_MSA = ["", "مائة", "مائتان", "ثلاثمائة", "أربعمائة", "خمسمائة", "ستمائة", "سبعمائة", "ثمانمائة", "تسعمائة"]

# Egyptian Dialect Nuances
ONES_EGY = ["", "واحد", "اتنين", "تلاتة", "أربعة", "خمسة", "ستة", "سبعة", "تمانية", "تسعة"]
TEENS_EGY = [
    "عشرة", "حدعشر", "اتناعشر", "تلاتاشر", "أربعتاشر", "خمستاشر",
    "ستاشر", "سبعتاشر", "تمنتاشر", "تسعتاشر"
]
TENS_EGY = ["", "عشرة", "عشرين", "تلاتين", "أربعين", "خمسين", "ستين", "سبعين", "تمانين", "تسعين"]
HUNDREDS_EGY = ["", "مية", "ميتين", "تلتومية", "ربعمية", "خمسمية", "ستمية", "سبعمية", "تمنمية", "تسعمية"]

MONTHS_ARABIC = {
    1: "يناير", 2: "فبراير", 3: "مارس", 4: "أبريل", 5: "مايو", 6: "يونيو",
    7: "يوليو", 8: "أغسطس", 9: "سبتمبر", 10: "أكتوبر", 11: "نوفمبر", 12: "ديسمبر"
}

ORDINAL_NUMBERS = {
    1: "الأول", 2: "الثاني", 3: "الثالث", 4: "الرابع", 5: "الخامس",
    6: "السادس", 7: "السابع", 8: "الثامن", 9: "التاسع", 10: "العاشر",
    11: "الحادي عشر", 12: "الثاني عشر", 13: "الثالث عشر", 14: "الرابع عشر",
    15: "الخامس عشر", 16: "السادس عشر", 17: "السابع عشر", 18: "الثامن عشر",
    19: "التاسع عشر", 20: "العشرون", 21: "الواحد والعشرون", 22: "الثاني والعشرون",
    23: "الثالث والعشرون", 24: "الرابع والعشرون", 25: "الخامس والعشرون",
    26: "السادس والعشرون", 27: "السابع والعشرون", 28: "الثامن والعشرون",
    29: "التاسع والعشرون", 30: "الثلاثون", 31: "الحادي والثلاثون"
}

HOUR_ORDINALS_MSA = {
    1: "الواحدة", 2: "الثانية", 3: "الثالثة", 4: "الرابعة", 5: "الخامسة",
    6: "السادسة", 7: "السابعة", 8: "الثامنة", 9: "التاسعة", 10: "العاشرة",
    11: "الحادية عشرة", 12: "الثانية عشرة"
}

HOUR_ORDINALS_EGY = {
    1: "واحدة", 2: "اتنين", 3: "تلاتة", 4: "أربعة", 5: "خمسة",
    6: "ستة", 7: "سبعة", 8: "تمانية", 9: "تسعة", 10: "عشرة",
    11: "حدعشر", 12: "اتناعشر"
}

ABBREVIATIONS = {
    r"\bد\.": "دكتور ",
    r"\bأ\.د\.": "أستاذ دكتور ",
    r"\bأ\.": "أستاذ ",
    r"\bم\.": "مهندس ",
    r"\bص\.ب\.": "صندوق بريد ",
    r"\bإلخ\b": "إلى آخره",
    r"\bق\.م\b": "قبل الميلاد",
    r"\bم\b(?=\s*\d)": "ميلادي ",
    r"\bهـ\b(?=\s*\d)": "هجري ",
    r"\bج\.م\b": "جنيه مصري",
    r"\bر\.س\b": "ريال سعودي",
    r"\bد\.إ\b": "درهم إماراتي",
    r"\bد\.ك\b": "دينار كويتي",
}


class ArabicNormalizer:
    """Normalizes raw Arabic text for optimal neural TTS synthesis."""

    def __init__(self, dialect: str = "msa"):
        """
        Initialize normalizer.
        
        Args:
            dialect: 'msa' (Modern Standard Arabic) or 'egyptian' (Masri)
        """
        self.dialect = dialect.lower()

    def normalize(self, text: str) -> str:
        """Complete normalization pipeline."""
        if not text:
            return ""

        # Convert Eastern numerals to Western
        text = text.translate(EASTERN_TO_WESTERN)

        # Remove tatweel (kashida ـ)
        text = re.sub(r"ـ+", "", text)

        # Normalize abbreviations
        for pattern, replacement in ABBREVIATIONS.items():
            text = re.sub(pattern, replacement, text)

        # Normalize dates: DD/MM/YYYY or YYYY/MM/DD
        text = self._normalize_dates(text)

        # Normalize time: HH:MM
        text = self._normalize_time(text)

        # Normalize percentages: 25% -> خمسة وعشرون بالمائة
        text = self._normalize_percentages(text)

        # Normalize currency amounts: $100, 50$, 100 ج.م
        text = self._normalize_currencies(text)

        # Normalize fractions: 1/2, 1/4, 3/4
        text = self._normalize_fractions(text)

        # Normalize standalone numbers
        text = self._normalize_numbers(text)

        # Clean spaces and redundant punctuation
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n", "\n", text)
        text = text.strip()

        return text

    def _convert_below_thousand(self, n: int, is_egyptian: bool = False) -> str:
        """Converts integer 0-999 into Arabic words."""
        if n == 0:
            return ""

        ones = ONES_EGY if is_egyptian else ONES_MSA
        teens = TEENS_EGY if is_egyptian else TEENS_MSA
        tens = TENS_EGY if is_egyptian else TENS_MSA
        hundreds = HUNDREDS_EGY if is_egyptian else HUNDREDS_MSA

        parts = []
        c = n // 100
        r = n % 100

        if c > 0:
            parts.append(hundreds[c])

        if r > 0:
            if r < 10:
                parts.append(ones[r])
            elif r < 20:
                parts.append(teens[r - 10])
            else:
                unit = r % 10
                ten = r // 10
                if unit > 0:
                    parts.append(f"{ones[unit]} و{tens[ten]}")
                else:
                    parts.append(tens[ten])

        connector = " و" if is_egyptian else " و"
        return connector.join([p for p in parts if p])

    def number_to_arabic(self, n: int) -> str:
        """Converts integer into spoken Arabic words."""
        if n == 0:
            return "صفر"

        is_egyptian = (self.dialect == "egyptian")
        prefix = ""
        if n < 0:
            prefix = "سالب "
            n = abs(n)

        scales = [
            (10**9, "مليار", "ملياران", "مليارات" if not is_egyptian else "مليار"),
            (10**6, "مليون", "مليونان", "ملايين" if not is_egyptian else "مليون"),
            (10**3, "ألف", "ألفان" if not is_egyptian else "ألفين", "آلاف" if not is_egyptian else "آلاف"),
        ]

        chunks = []
        rem = n

        for scale, singular, dual, plural in scales:
            count = rem // scale
            if count > 0:
                if count == 1:
                    chunks.append(singular)
                elif count == 2:
                    chunks.append(dual)
                elif 3 <= count <= 10:
                    sub = self._convert_below_thousand(count, is_egyptian)
                    chunks.append(f"{sub} {plural}")
                else:
                    sub = self._convert_below_thousand(count, is_egyptian)
                    chunks.append(f"{sub} {singular}")
                rem %= scale

        if rem > 0:
            chunks.append(self._convert_below_thousand(rem, is_egyptian))

        return prefix + " و".join(chunks)

    def _normalize_dates(self, text: str) -> str:
        """Expands dates like 15/09/2026 or 2026-09-15."""
        def date_repl(match):
            d_str, m_str, y_str = match.group(1), match.group(2), match.group(3)
            # Detect year order
            if len(d_str) == 4:  # YYYY/MM/DD
                y, m, d = int(d_str), int(m_str), int(y_str)
            else:  # DD/MM/YYYY
                d, m, y = int(d_str), int(m_str), int(y_str)

            d_word = ORDINAL_NUMBERS.get(d, str(d))
            m_word = MONTHS_ARABIC.get(m, str(m))
            y_word = self.number_to_arabic(y)

            if self.dialect == "egyptian":
                return f"{d_word} من {m_word} سنة {y_word}"
            return f"{d_word} من {m_word} عام {y_word}"

        pattern = r"\b(\d{1,4})[/\-.](\d{1,2})[/\-.](\d{1,4})\b"
        return re.sub(pattern, date_repl, text)

    def _normalize_time(self, text: str) -> str:
        """Expands time expressions like 3:30, 07:15, 12:45."""
        def time_repl(match):
            has_saa_prefix = bool(match.group(1))
            h = int(match.group(2))
            m = int(match.group(3))
            period = match.group(4) or ""

            # 12-hour normalize
            h12 = h if h <= 12 else h - 12
            if h12 == 0:
                h12 = 12

            prefix = "" if has_saa_prefix else "الساعة "

            if self.dialect == "egyptian":
                h_word = HOUR_ORDINALS_EGY.get(h12, str(h12))
                if m == 0:
                    res = f"{prefix}{h_word}"
                elif m == 15:
                    res = f"{prefix}{h_word} وربع"
                elif m == 20:
                    res = f"{prefix}{h_word} وتلت"
                elif m == 30:
                    res = f"{prefix}{h_word} ونص"
                elif m == 40:
                    next_h = 1 if h12 == 12 else h12 + 1
                    next_w = HOUR_ORDINALS_EGY.get(next_h, str(next_h))
                    res = f"{prefix}{next_w} إلا تلت"
                elif m == 45:
                    next_h = 1 if h12 == 12 else h12 + 1
                    next_w = HOUR_ORDINALS_EGY.get(next_h, str(next_h))
                    res = f"{prefix}{next_w} إلا ربع"
                else:
                    m_word = self.number_to_arabic(m)
                    res = f"{prefix}{h_word} و{m_word} دقيقة"
            else:  # MSA
                h_word = HOUR_ORDINALS_MSA.get(h12, str(h12))
                if m == 0:
                    res = f"{prefix}{h_word} تماماً"
                elif m == 15:
                    res = f"{prefix}{h_word} والربع"
                elif m == 20:
                    res = f"{prefix}{h_word} والثلث"
                elif m == 30:
                    res = f"{prefix}{h_word} والنصف"
                elif m == 45:
                    next_h = 1 if h12 == 12 else h12 + 1
                    next_w = HOUR_ORDINALS_MSA.get(next_h, str(next_h))
                    res = f"{prefix}{next_w} إلا ربعاً"
                else:
                    m_word = self.number_to_arabic(m)
                    res = f"{prefix}{h_word} و{m_word} دقيقة"

            if "am" in period.lower() or "صباحا" in period:
                res += " صباحاً"
            elif "pm" in period.lower() or "مساء" in period:
                res += " مساءً"

            return ("الساعة " if has_saa_prefix else "") + res

        pattern = r"(الساعة\s+)?\b([0-1]?[0-9]|2[0-3]):([0-5][0-9])(?:\s*(صباحاً|مساءً|am|pm|AM|PM))?\b"
        return re.sub(pattern, time_repl, text)

    def _normalize_percentages(self, text: str) -> str:
        """Converts percentages like 50% or 12.5% into spoken Arabic."""
        def pct_repl(match):
            num_str = match.group(1)
            if "." in num_str:
                int_part, dec_part = num_str.split(".", 1)
                w_int = self.number_to_arabic(int(int_part))
                w_dec = self.number_to_arabic(int(dec_part))
                return f"{w_int} فاصل {w_dec} بالمائة"
            else:
                w = self.number_to_arabic(int(num_str))
                return f"{w} بالمائة"

        return re.sub(r"(\d+(?:\.\d+)?)\s*%", pct_repl, text)

    def _normalize_currencies(self, text: str) -> str:
        """Converts currency symbols with quantities."""
        currencies = [
            (r"\$\s*(\d+(?:\.\d+)?)", "دولار"),
            (r"(\d+(?:\.\d+)?)\s*\$", "دولار"),
            (r"€\s*(\d+(?:\.\d+)?)", "يورو"),
            (r"(\d+(?:\.\d+)?)\s*€", "يورو"),
            (r"£\s*(\d+(?:\.\d+)?)", "جنيه إسترليني"),
            (r"(\d+(?:\.\d+)?)\s*ج\.م", "جنيه مصري" if self.dialect != "egyptian" else "جنيه"),
            (r"(\d+(?:\.\d+)?)\s*ر\.س", "ريال سعودي"),
            (r"(\d+(?:\.\d+)?)\s*د\.إ", "درهم إماراتي"),
            (r"(\d+(?:\.\d+)?)\s*د\.ك", "دينار كويتي"),
        ]

        for pattern, name in currencies:
            def curr_repl(match, currency_name=name):
                num = int(float(match.group(1)))
                words = self.number_to_arabic(num)
                return f"{words} {currency_name}"
            text = re.sub(pattern, curr_repl, text)

        return text

    def _normalize_fractions(self, text: str) -> str:
        """Expands common fractions."""
        fractions = {
            r"\b1/2\b": "نصف",
            r"\b1/4\b": "ربع",
            r"\b3/4\b": "ثلاثة أرباع",
            r"\b1/3\b": "ثلث",
            r"\b2/3\b": "ثلثان",
            r"\b1/5\b": "خمس",
            r"\b1/8\b": "ثمن",
        }
        for pattern, replacement in fractions.items():
            text = re.sub(pattern, replacement, text)
        return text

    def _normalize_numbers(self, text: str) -> str:
        """Expands all remaining standalone numbers."""
        def num_repl(match):
            num_str = match.group(0)
            if len(num_str) >= 7 and not num_str.startswith("0"):
                # Could be a phone or id number if very long, or a huge number
                try:
                    return self.number_to_arabic(int(num_str))
                except Exception:
                    return " ".join([ONES_MSA[int(d)] for d in num_str])
            try:
                val = int(num_str)
                return self.number_to_arabic(val)
            except ValueError:
                return num_str

        return re.sub(r"\b\d+\b", num_repl, text)

    def extract_gestures(self, text: str) -> Tuple[str, str, bool, float]:
        """
        Parses in-line gesture tags, emotions, and pauses from Arabic and English tags.
        
        Tags:
        - Arabic: [تنفس], [متحمس], [متعاطف], [مؤثر], [وقفة]
        - English: [breath], [excited], [sympathetic], [emotional], [pause], <break time="...ms"/>
        
        Returns:
            Tuple of (clean_text, emotion, include_breath, pause_override_sec)
        """
        emotion = "neutral"
        include_breath = True
        custom_pause = 0.0

        if re.search(r"\[(متحمس|excited|happy|بهجة)\]", text, re.IGNORECASE):
            emotion = "excited"
        elif re.search(r"\[(متعاطف|sympathetic|apologetic|تعاطف)\]", text, re.IGNORECASE):
            emotion = "sympathetic"
        elif re.search(r"\[(مؤثر|emotional|warm|حميمي)\]", text, re.IGNORECASE):
            emotion = "emotional"

        if re.search(r"\[(تنفس|breath|inhalation)\]", text, re.IGNORECASE):
            include_breath = True
        elif re.search(r"\[(بدون_تنفس|no-breath|nobreath)\]", text, re.IGNORECASE):
            include_breath = False

        break_match = re.search(r'<break\s+time=["\'](\d+)ms["\']\s*/>', text, re.IGNORECASE)
        if break_match:
            custom_pause = float(break_match.group(1)) / 1000.0
        elif re.search(r"\[(وقفة|pause)\]", text, re.IGNORECASE):
            custom_pause = 0.45

        clean = re.sub(
            r"\[(متحمس|متعاطف|مؤثر|تنفس|بدون_تنفس|وقفة|excited|sympathetic|emotional|breath|no-breath|pause|happy|warm)\]",
            "",
            text,
            flags=re.IGNORECASE
        )
        clean = re.sub(r'<break\s+time=["\']\d+ms["\']\s*/>', "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"[ \t]+", " ", clean).strip()

        return clean, emotion, include_breath, custom_pause

    def split_conversational_clauses(
        self,
        text: str,
        max_words_per_clause: int = 8
    ) -> List[Dict[str, Union[str, bool]]]:
        """
        Splits conversational Arabic sentences at major syntactic connectors
        when clause length exceeds max_words_per_clause.
        
        Connectors: و، فـ، لكن/ولكن، يعني، طبعاً/أكيد، إذا/لو
        """
        words = text.split()
        if len(words) <= max_words_per_clause:
            return [{"text": text, "needs_breath": False, "pause_ms": 0}]

        connector_pattern = re.compile(
            r"(?<=\s)(و|ف|فقد|ولكن|لكن|يعني|طبعاً|أكيد|إذا|لو|ثم|بينما|حيث\s+أن)\s+"
        )

        clauses = []
        raw_splits = connector_pattern.split(text)

        current_text = ""
        for part in raw_splits:
            if not part:
                continue
            if part in ["و", "ف", "فقد", "ولكن", "لكن", "يعني", "طبعاً", "أكيد", "إذا", "لو", "ثم", "بينما"] or part.startswith("حيث"):
                if current_text and len(current_text.split()) >= max_words_per_clause:
                    clauses.append({"text": current_text.strip(), "needs_breath": True, "pause_ms": 180})
                    current_text = part + " "
                else:
                    current_text += part + " "
            else:
                current_text += part

        if current_text.strip():
            clauses.append({"text": current_text.strip(), "needs_breath": False, "pause_ms": 0})

        return clauses if len(clauses) > 1 else [{"text": text, "needs_breath": False, "pause_ms": 0}]

