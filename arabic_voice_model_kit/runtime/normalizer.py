"""
Arabic Telephony Text Normalizer (Pillar 7)
Deterministic normalization for dates, times, currencies, phone numbers, percentages, and numerals.
"""

import re
from typing import Tuple, Optional, List


class ArabicTelephonyNormalizer:
    """
    Normalizes conversational Arabic text for telephony voice agents.
    Expands numerals, currencies, times, dates, phone numbers, percentages,
    and extracts emotion control tags.
    """

    EASTERN_TO_WESTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

    DIGITS_AR = {
        0: "صفر", 1: "واحد", 2: "اثنين", 3: "ثلاثة", 4: "أربعة",
        5: "خمسة", 6: "ستة", 7: "سبعة", 8: "ثمانية", 9: "تسعة",
        10: "عشرة", 11: "أحد عشر", 12: "اثنا عشر"
    }

    TENS_AR = {
        2: "عشرين", 3: "ثلاثين", 4: "أربعين", 5: "خمسين",
        6: "ستين", 7: "سبعين", 8: "ثمانين", 9: "تسعين"
    }

    HUNDREDS_AR = {
        1: "مئة", 2: "مئتان", 3: "ثلاثمئة", 4: "أربعمئة", 5: "خمسمئة",
        6: "ستمئة", 7: "سبعمئة", 8: "ثمانمئة", 9: "تسعمئة"
    }

    MONTH_NAMES_AR = {
        1: "يناير", 2: "فبراير", 3: "مارس", 4: "أبريل",
        5: "مايو", 6: "يونيو", 7: "يوليو", 8: "أغسطس",
        9: "سبتمبر", 10: "أكتوبر", 11: "نوفمبر", 12: "ديسمبر"
    }

    HOUR_NAMES_AR = {
        1: "الواحدة", 2: "الثانية", 3: "الثالثة", 4: "الرابعة",
        5: "الخامسة", 6: "السادسة", 7: "السابعة", 8: "الثامنة",
        9: "التاسعة", 10: "العاشرة", 11: "الحادية عشرة", 12: "الثانية عشرة"
    }

    ORDINALS_AR = {
        1: "الأول", 2: "الثاني", 3: "الثالث", 4: "الرابع", 5: "الخامس",
        6: "السادس", 7: "السابع", 8: "الثامن", 9: "التاسع", 10: "العاشر",
        11: "الحادي عشر", 12: "الثاني عشر", 13: "الثالث عشر", 14: "الرابع عشر",
        15: "الخامس عشر", 16: "السادس عشر", 17: "السابع عشر", 18: "الثامن عشر",
        19: "التاسع عشر", 20: "العشرون", 21: "الحادي والعشرون", 22: "الثاني والعشرون",
        23: "الثالث والعشرون", 24: "الرابع والعشرون", 25: "الخامس والعشرون",
        26: "السادس والعشرون", 27: "السابع والعشرون", 28: "الثامن والعشرون",
        29: "التاسع والعشرون", 30: "الثلاثون", 31: "الحادي والثلاثون"
    }

    def __init__(self):
        pass

    def extract_emotion_and_tags(self, text: str) -> Tuple[str, Optional[str], bool]:
        """
        Extracts emotion tag and breath directive, returning sanitized clean text.
        """
        emotion = "neutral"
        has_breath = True

        if re.search(r"\[(no_breath|بدون_تنفس|بلا_تنفس)\]", text, re.IGNORECASE):
            has_breath = False

        if re.search(r"\[(متحمس|excited|happy|فرحان)\]", text, re.IGNORECASE):
            emotion = "excited"
        elif re.search(r"\[(متعاطف|sympathetic|apologetic|حزين|اعتذار)\]", text, re.IGNORECASE):
            emotion = "sympathetic"
        elif re.search(r"\[(مؤثر|emotional|sincere|صادق|دافئ)\]", text, re.IGNORECASE):
            emotion = "emotional"
        elif re.search(r"\[(طبيعي|neutral|هادئ)\]", text, re.IGNORECASE):
            emotion = "neutral"

        if re.search(r"\[(تنفس|breath|inhalation)\]", text, re.IGNORECASE):
            has_breath = True

        clean = re.sub(
            r"\[(متحمس|متعاطف|مؤثر|تنفس|طبيعي|بدون_تنفس|بلا_تنفس|excited|sympathetic|emotional|breath|no_breath|pause)\]",
            "",
            text,
            flags=re.IGNORECASE
        )
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean, emotion, has_breath

    def to_western_digits(self, text: str) -> str:
        """Converts Eastern Arabic numerals (٠-٩) to Western (0-9)."""
        return text.translate(self.EASTERN_TO_WESTERN)

    def number_to_words(self, n: int) -> str:
        """Converts integer (0 to 999,999) to spoken Arabic words."""
        if n < 0:
            return "سالب " + self.number_to_words(-n)
        if n <= 12:
            return self.DIGITS_AR[n]
        if n < 20:
            return f"{self.DIGITS_AR[n - 10]} عشر"
        if n < 100:
            tens_val = n // 10
            rem = n % 10
            tens_str = self.TENS_AR.get(tens_val, "")
            if rem == 0:
                return tens_str
            return f"{self.DIGITS_AR[rem]} و{tens_str}"
        if n < 1000:
            hundreds_val = n // 100
            rem = n % 100
            h_str = self.HUNDREDS_AR.get(hundreds_val, "")
            if rem == 0:
                return h_str
            return f"{h_str} و{self.number_to_words(rem)}"
        if n < 1000000:
            thousands_val = n // 1000
            rem = n % 1000
            if thousands_val == 1:
                t_str = "ألف"
            elif thousands_val == 2:
                t_str = "ألفان"
            elif 3 <= thousands_val <= 10:
                t_str = f"{self.DIGITS_AR[thousands_val]} آلاف"
            else:
                t_str = f"{self.number_to_words(thousands_val)} ألفاً"

            if rem == 0:
                return t_str
            return f"{t_str} و{self.number_to_words(rem)}"

        return str(n)

    def digits_to_words(self, digits: str) -> str:
        """Converts sequence of digits to individual spoken Arabic words (for phone numbers)."""
        res = []
        for ch in digits:
            if ch.isdigit():
                res.append(self.DIGITS_AR[int(ch)])
        return " ".join(res)

    def _normalize_currencies(self, text: str) -> str:
        """Normalizes currencies (SAR, EGP, USD, EUR, AED)."""
        # Form: ٤٥ ريال, 120 جنيه, $1,250, 50 SAR, 100 EUR
        # Convert Eastern digits first for pattern matching
        t = self.to_western_digits(text)

        # Dollar: $1,250 or $45 or 45$
        def repl_dollar(m):
            amt = int(m.group(1).replace(",", ""))
            w = self.number_to_words(amt)
            return f"{w} دولار"

        t = re.sub(r"\$\s*([0-9]+(?:,[0-9]+)*)", repl_dollar, t)
        t = re.sub(r"([0-9]+(?:,[0-9]+)*)\s*\$", repl_dollar, t)
        t = re.sub(r"([0-9]+(?:,[0-9]+)*)\s*(?:USD|دولار)", repl_dollar, t)

        # Saudi Riyal: 45 ريال or 45 SAR
        def repl_riyal(m):
            amt = int(m.group(1).replace(",", ""))
            w = self.number_to_words(amt)
            if amt == 1:
                return "ريال واحد"
            elif amt == 2:
                return "ريالان"
            elif 3 <= amt <= 10:
                return f"{w} ريالات"
            return f"{w} ريال"

        t = re.sub(r"([0-9]+(?:,[0-9]+)*)\s*(?:ريالاً|ريال|SAR)", repl_riyal, t)

        # Egyptian Pound: 120 جنيه or 120 EGP
        def repl_pound(m):
            amt = int(m.group(1).replace(",", ""))
            w = self.number_to_words(amt)
            return f"{w} جنيه"

        t = re.sub(r"([0-9]+(?:,[0-9]+)*)\s*(?:جنيه|جنيهات|EGP)", repl_pound, t)

        # UAE Dirham: 100 درهم or 100 AED
        def repl_dirham(m):
            amt = int(m.group(1).replace(",", ""))
            w = self.number_to_words(amt)
            return f"{w} درهم"

        t = re.sub(r"([0-9]+(?:,[0-9]+)*)\s*(?:درهم|درهماً|AED)", repl_dirham, t)

        # Euro: 100 يورو or 100 EUR or €100
        def repl_euro(m):
            amt = int(m.group(1).replace(",", ""))
            w = self.number_to_words(amt)
            return f"{w} يورو"

        t = re.sub(r"€\s*([0-9]+(?:,[0-9]+)*)", repl_euro, t)
        t = re.sub(r"([0-9]+(?:,[0-9]+)*)\s*(?:يورو|EUR)", repl_euro, t)

        return t

    def _normalize_times(self, text: str) -> str:
        """
        Normalizes times e.g. ٤:٣٠ مساءً or 4:30 or 10:15 صباحاً.
        """
        t = self.to_western_digits(text)

        def repl_time(m):
            hour = int(m.group(1))
            minute = int(m.group(2))
            period = m.group(3) or ""

            h_12 = hour % 12
            if h_12 == 0:
                h_12 = 12
            h_str = self.HOUR_NAMES_AR.get(h_12, self.number_to_words(h_12))

            if minute == 0:
                m_str = "تماماً"
            elif minute == 15:
                m_str = "والربع"
            elif minute == 30:
                m_str = "والنصف"
            elif minute == 45:
                # e.g. "إلا ربع" of next hour
                next_h = (h_12 % 12) + 1
                next_h_str = self.HOUR_NAMES_AR.get(next_h, self.number_to_words(next_h))
                return f"{next_h_str} إلا ربع {period}".strip()
            elif minute == 20:
                m_str = "والثلث"
            else:
                m_str = f"و{self.number_to_words(minute)} دقيقة"

            res = f"{h_str} {m_str}"
            if period:
                res += f" {period}"
            return res.strip()

        # Match (الساعة )? H:MM (صباحاً|مساءً|م|ص)?
        t = re.sub(
            r"(?:الساعة\s+)?([0-1]?[0-9]|2[0-3]):([0-5][0-9])\s*(صباحاً|مساءً|م|ص)?",
            lambda m: ("الساعة " if "الساعة" in m.group(0) else "") + repl_time(m),
            t
        )
        return t

    def _normalize_dates(self, text: str) -> str:
        """
        Normalizes dates e.g. 15/09/2026 or 15-09-2026 or 15/9.
        """
        t = self.to_western_digits(text)

        def repl_full_date(m):
            day = int(m.group(1))
            month = int(m.group(2))
            year = int(m.group(3))

            d_str = self.ORDINALS_AR.get(day, self.number_to_words(day))
            m_str = self.MONTH_NAMES_AR.get(month, f"الشهر {self.number_to_words(month)}")
            y_str = self.number_to_words(year)

            return f"{d_str} من {m_str} عام {y_str}"

        t = re.sub(r"\b([0-3]?[0-9])[\/\-]([0-1]?[0-9])[\/\-](20[2-3][0-9]|19[0-9]{2})\b", repl_full_date, t)

        def repl_short_date(m):
            day = int(m.group(1))
            month = int(m.group(2))
            d_str = self.ORDINALS_AR.get(day, self.number_to_words(day))
            m_str = self.MONTH_NAMES_AR.get(month, f"الشهر {self.number_to_words(month)}")
            return f"{d_str} من {m_str}"

        t = re.sub(r"\b([0-3]?[0-9])[\/\-]([0-1]?[0-9])\b", repl_short_date, t)
        return t

    def _normalize_percentages(self, text: str) -> str:
        """
        Normalizes percentages e.g. 95% or ٩٥٪ or 50 بالمئة.
        """
        t = self.to_western_digits(text)

        def repl_percent(m):
            val = int(m.group(1))
            return f"{self.number_to_words(val)} بالمئة"

        t = re.sub(r"([0-9]+)\s*(?:%|٪)", repl_percent, t)
        return t

    def _normalize_phone_numbers(self, text: str) -> str:
        """
        Normalizes phone numbers by pacing digits (e.g. +966 50 123 4567 or 0501234567).
        """
        t = self.to_western_digits(text)

        def repl_phone(m):
            num = m.group(0).replace(" ", "").replace("-", "")
            words = []
            if num.startswith("+"):
                words.append("مفتاح دولة")
                num = num[1:]
            for d in num:
                if d.isdigit():
                    words.append(self.DIGITS_AR[int(d)])
            return "، ".join(words)

        # Phone numbers usually 7 to 15 digits
        t = re.sub(r"(?:\+966|05|\b00966)[0-9\s\-]{7,13}\b", repl_phone, t)
        return t

    def _normalize_remaining_numbers(self, text: str) -> str:
        """
        Expands any remaining isolated digits to Arabic words.
        """
        t = self.to_western_digits(text)

        def repl_num(m):
            val = int(m.group(0))
            return self.number_to_words(val)

        return re.sub(r"\b[0-9]+\b", repl_num, t)

    def normalize(self, text: str) -> str:
        """
        Executes full deterministic normalization pipeline on Arabic text.
        """
        if not text:
            return ""

        t = text
        t = self._normalize_currencies(t)
        t = self._normalize_times(t)
        t = self._normalize_dates(t)
        t = self._normalize_percentages(t)
        t = self._normalize_phone_numbers(t)
        t = self._normalize_remaining_numbers(t)

        # Clean spaces
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def process(self, text: str) -> Tuple[str, str, bool]:
        """
        Extracts emotion/breath tags, then normalizes spoken Arabic text.
        Returns: (normalized_text, emotion, has_breath)
        """
        clean_text, emotion, has_breath = self.extract_emotion_and_tags(text)
        normalized = self.normalize(clean_text)
        return normalized, emotion, has_breath
