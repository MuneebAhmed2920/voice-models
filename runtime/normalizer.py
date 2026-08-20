"""
Telephony Text Normalizer & Emotion Tag Extractor
Optimized for Call Center Telephony, Vicidial Agents, and AI Voice Bots.
"""

import re
import calendar
from typing import Tuple, Optional, List

try:
    import num2words
except ImportError:
    num2words = None


class TelephonyTextNormalizer:
    """
    Robust text normalizer tailored for conversational telephony and AI voice agents.
    Expands currency, dates, times, phone numbers, percentages, addresses, abbreviations,
    and extracts emotion control tags like [excited], [sympathetic], [emotional], [breath].
    """

    MONTH_NAMES = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }

    ORDINAL_SUFFIXES = {
        1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
        6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
        11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth", 15: "fifteenth",
        16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth", 20: "twentieth",
        21: "twenty-first", 22: "twenty-second", 23: "twenty-third", 24: "twenty-fourth",
        25: "twenty-fifth", 26: "twenty-sixth", 27: "twenty-seventh", 28: "twenty-eighth",
        29: "twenty-ninth", 30: "thirtieth", 31: "thirty-first"
    }

    DIGIT_WORDS = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"
    }

    ABBREVIATIONS = [
        (re.compile(r"\bDr\.(?=\s+[A-Z])", re.IGNORECASE), "Doctor"),
        (re.compile(r"\bMr\.(?=\s+[A-Z])", re.IGNORECASE), "Mister"),
        (re.compile(r"\bMrs\.(?=\s+[A-Z])", re.IGNORECASE), "Missus"),
        (re.compile(r"\bMs\.(?=\s+[A-Z])", re.IGNORECASE), "Mizz"),
        (re.compile(r"\bProf\.(?=\s+[A-Z])", re.IGNORECASE), "Professor"),
        (re.compile(r"\b([A-Za-z]+)\s+St\.?\b", re.IGNORECASE), r"\1 Street"),
        (re.compile(r"\b([A-Za-z]+)\s+Ave\.?\b", re.IGNORECASE), r"\1 Avenue"),
        (re.compile(r"\b([A-Za-z]+)\s+Blvd\.?\b", re.IGNORECASE), r"\1 Boulevard"),
        (re.compile(r"\b([A-Za-z]+)\s+Rd\.?\b", re.IGNORECASE), r"\1 Road"),
        (re.compile(r"\bSt\.(?=\s+[A-Z])", re.IGNORECASE), "Saint"),
        (re.compile(r"\bApt\.?\s*#?(\w+)", re.IGNORECASE), r"Apartment \1"),
        (re.compile(r"\bSte\.?\s*#?(\w+)", re.IGNORECASE), r"Suite \1"),
        (re.compile(r"\bDept\.?\b", re.IGNORECASE), "department"),
        (re.compile(r"\bExt\.?\s*(\d+)", re.IGNORECASE), r"extension \1"),
        (re.compile(r"\betc\.", re.IGNORECASE), "etcetera"),
        (re.compile(r"\bvs\.?\b", re.IGNORECASE), "versus"),
        (re.compile(r"\bapprox\.?\b", re.IGNORECASE), "approximately"),
        (re.compile(r"\best\.?\b", re.IGNORECASE), "estimated"),
        (re.compile(r"\bmin\.?\b", re.IGNORECASE), "minutes"),
        (re.compile(r"\bsec\.?\b", re.IGNORECASE), "seconds"),
        (re.compile(r"\bhr\.?\b", re.IGNORECASE), "hour"),
        (re.compile(r"\bhrs\.?\b", re.IGNORECASE), "hours"),
        (re.compile(r"(?<=\s)w/(?=\s)", re.IGNORECASE), "with"),
        (re.compile(r"(?<=\s)w/o(?=\s)", re.IGNORECASE), "without"),
        (re.compile(r"\bASAP\b", re.IGNORECASE), "as soon as possible"),
        (re.compile(r"\bETA\b", re.IGNORECASE), "E T A"),
        (re.compile(r"\bFAQ\b", re.IGNORECASE), "F A Q"),
    ]

    EMOTION_TAG_PATTERN = re.compile(
        r"\[(excited|sympathetic|emotional|neutral|happy|calm|urgent|warm|apologetic|breath)\]",
        re.IGNORECASE
    )

    def __init__(self):
        pass

    def extract_emotion_and_tags(self, text: str) -> Tuple[str, Optional[str], bool]:
        """
        Extracts emotion style tag and breath flags from the input text,
        and returns (cleaned_text, emotion, has_breath_tag).
        """
        emotion = None
        has_breath = False

        tags = self.EMOTION_TAG_PATTERN.findall(text)
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower == "breath":
                has_breath = True
            elif tag_lower in ["excited", "sympathetic", "emotional", "neutral", "happy", "calm", "urgent", "warm", "apologetic"]:
                emotion = tag_lower

        cleaned_text = self.EMOTION_TAG_PATTERN.sub("", text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        return cleaned_text, emotion, has_breath

    def _num_to_words(self, number: int) -> str:
        """Converts an integer to English words."""
        if num2words is not None:
            try:
                return num2words.num2words(number)
            except Exception:
                pass
        
        # Fallback simple converter
        if number == 0:
            return "zero"
        units = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                 "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                 "seventeen", "eighteen", "nineteen"]
        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        
        def _helper(n):
            if n < 20:
                return units[n]
            elif n < 100:
                return tens[n // 10] + ("-" + units[n % 10] if n % 10 != 0 else "")
            elif n < 1000:
                return units[n // 100] + " hundred" + (" and " + _helper(n % 100) if n % 100 != 0 else "")
            elif n < 1000000:
                return _helper(n // 1000) + " thousand" + (" " + _helper(n % 1000) if n % 1000 != 0 else "")
            else:
                return str(n)
        
        return _helper(number)

    def _digits_to_words(self, digits_str: str) -> str:
        """Converts string of digits to spoken digit sequence (e.g. '800' -> 'eight zero zero')."""
        return " ".join(self.DIGIT_WORDS.get(d, d) for d in digits_str if d.isdigit())

    def _normalize_currency(self, text: str) -> str:
        """Expands currency notations like $45.50, $1,500, £20, €50."""
        def _usd_sub(match):
            amount_str = match.group(1).replace(",", "")
            if "." in amount_str:
                parts = amount_str.split(".")
                dollars = int(parts[0]) if parts[0] else 0
                cents_str = parts[1][:2].ljust(2, "0")
                cents = int(cents_str)
                d_str = f"{self._num_to_words(dollars)} dollar" if dollars == 1 else f"{self._num_to_words(dollars)} dollars"
                if cents > 0:
                    c_str = f"{self._num_to_words(cents)} cent" if cents == 1 else f"{self._num_to_words(cents)} cents"
                    return f"{d_str} and {c_str}"
                return d_str
            else:
                dollars = int(amount_str)
                return f"{self._num_to_words(dollars)} dollar" if dollars == 1 else f"{self._num_to_words(dollars)} dollars"

        text = re.sub(r"\$(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)", _usd_sub, text)

        def _gbp_sub(match):
            amount_str = match.group(1).replace(",", "")
            if "." in amount_str:
                parts = amount_str.split(".")
                pounds = int(parts[0]) if parts[0] else 0
                pence = int(parts[1][:2].ljust(2, "0"))
                p_str = f"{self._num_to_words(pounds)} pound" if pounds == 1 else f"{self._num_to_words(pounds)} pounds"
                if pence > 0:
                    pence_str = f"{self._num_to_words(pence)} pence"
                    return f"{p_str} and {pence_str}"
                return p_str
            else:
                pounds = int(amount_str)
                return f"{self._num_to_words(pounds)} pound" if pounds == 1 else f"{self._num_to_words(pounds)} pounds"

        text = re.sub(r"£(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)", _gbp_sub, text)

        def _eur_sub(match):
            amount_str = match.group(1).replace(",", "")
            if "." in amount_str:
                parts = amount_str.split(".")
                euros = int(parts[0]) if parts[0] else 0
                cents = int(parts[1][:2].ljust(2, "0"))
                e_str = f"{self._num_to_words(euros)} euro" if euros == 1 else f"{self._num_to_words(euros)} euros"
                if cents > 0:
                    c_str = f"{self._num_to_words(cents)} cent" if cents == 1 else f"{self._num_to_words(cents)} cents"
                    return f"{e_str} and {c_str}"
                return e_str
            else:
                euros = int(amount_str)
                return f"{self._num_to_words(euros)} euro" if euros == 1 else f"{self._num_to_words(euros)} euros"

        text = re.sub(r"€(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)", _eur_sub, text)
        return text

    def _normalize_phone_numbers(self, text: str) -> str:
        """Expands US phone numbers formatted with dashes/parens/spaces to digit-by-digit speech."""
        def _phone_sub(match):
            country_code = match.group(1)
            area_code = match.group(2)
            prefix = match.group(3)
            line = match.group(4)
            
            result = []
            if country_code:
                result.append("plus " + self._digits_to_words(country_code.strip("+- ")))
            result.append(self._digits_to_words(area_code))
            result.append(self._digits_to_words(prefix))
            result.append(self._digits_to_words(line))
            return ", ".join(result)

        phone_pattern = re.compile(
            r"(?:\b(\+?1)[-.\s]?)?\(?([2-9]\d{2})\)?[-.\s]([2-9]\d{2})[-.\s](\d{4})\b"
        )
        return phone_pattern.sub(_phone_sub, text)

    def _normalize_dates(self, text: str) -> str:
        """Expands dates like MM/DD/YYYY, YYYY-MM-DD, Month DD, YYYY."""
        def _date_slash_sub(match):
            m, d, y = int(match.group(1)), int(match.group(2)), int(match.group(3))
            if 1 <= m <= 12 and 1 <= d <= 31:
                month_name = self.MONTH_NAMES.get(m, "")
                day_ordinal = self.ORDINAL_SUFFIXES.get(d, f"{d}th")
                year_words = self._normalize_year(y)
                return f"{month_name} {day_ordinal}, {year_words}"
            return match.group(0)

        text = re.sub(r"\b(0?[1-9]|1[0-2])[/\-](0?[1-9]|[12]\d|3[01])[/\-](\d{4})\b", _date_slash_sub, text)

        def _date_iso_sub(match):
            y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
            if 1 <= m <= 12 and 1 <= d <= 31:
                month_name = self.MONTH_NAMES.get(m, "")
                day_ordinal = self.ORDINAL_SUFFIXES.get(d, f"{d}th")
                year_words = self._normalize_year(y)
                return f"{month_name} {day_ordinal}, {year_words}"
            return match.group(0)

        text = re.sub(r"\b(\d{4})-(0?[1-9]|1[0-2])-(0?[1-9]|[12]\d|3[01])\b", _date_iso_sub, text)

        month_names_pattern = "|".join(list(self.MONTH_NAMES.values()) + [k[:3] for k in self.MONTH_NAMES.values()])
        def _month_day_year_sub(match):
            month_str = match.group(1)
            day_val = int(match.group(2))
            year_val = int(match.group(3)) if len(match.groups()) >= 3 and match.group(3) else None
            
            day_ordinal = self.ORDINAL_SUFFIXES.get(day_val, f"{day_val}th")
            if year_val:
                year_words = self._normalize_year(year_val)
                return f"{month_str} {day_ordinal}, {year_words}"
            return f"{month_str} {day_ordinal}"

        text = re.sub(
            rf"\b({month_names_pattern})\.?\s+([1-9]|[12]\d|3[01])(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b",
            _month_day_year_sub,
            text,
            flags=re.IGNORECASE
        )
        return text

    def _normalize_year(self, year: int) -> str:
        """Converts year like 2026 -> 'twenty twenty-six', 1995 -> 'nineteen ninety-five', 2000 -> 'two thousand'."""
        if 2000 <= year <= 2009:
            return f"two thousand {self._num_to_words(year % 2000)}" if year > 2000 else "two thousand"
        elif 1000 <= year <= 9999:
            century = year // 100
            decade = year % 100
            if decade == 0:
                return f"{self._num_to_words(century)} hundred"
            elif decade < 10:
                return f"{self._num_to_words(century)} oh {self._num_to_words(decade)}"
            else:
                return f"{self._num_to_words(century)} {self._num_to_words(decade)}"
        return self._num_to_words(year)

    def _normalize_time(self, text: str) -> str:
        """Expands times like 3:30 PM, 10:15am, 12:00."""
        def _time_sub(match):
            hour = int(match.group(1))
            minute = int(match.group(2))
            meridiem = match.group(3)

            hour_words = self._num_to_words(hour)
            if minute == 0:
                if meridiem:
                    return f"{hour_words} {meridiem.upper().strip()}"
                return f"{hour_words} o'clock"
            elif minute < 10:
                minute_words = f"oh {self._num_to_words(minute)}"
            else:
                minute_words = self._num_to_words(minute)

            if meridiem:
                return f"{hour_words} {minute_words} {meridiem.upper().strip()}"
            return f"{hour_words} {minute_words}"

        time_pattern = re.compile(r"\b(1[0-2]|0?[1-9]):([0-5]\d)(?:\s*([ap]\.?m\.?))?\b", re.IGNORECASE)
        return time_pattern.sub(_time_sub, text)

    def _normalize_percentages(self, text: str) -> str:
        """Expands percentages like 25%, 4.5%."""
        def _pct_sub(match):
            val_str = match.group(1)
            if "." in val_str:
                parts = val_str.split(".")
                integer_part = self._num_to_words(int(parts[0]))
                dec_part = " ".join(self.DIGIT_WORDS.get(d, d) for d in parts[1])
                return f"{integer_part} point {dec_part} percent"
            else:
                return f"{self._num_to_words(int(val_str))} percent"

        return re.sub(r"\b(\d+(?:\.\d+)?)%", _pct_sub, text)

    def _normalize_isolated_numbers(self, text: str) -> str:
        """Expands comma-formatted numbers and standalone integers."""
        def _hash_num_sub(match):
            return "number " + self._digits_to_words(match.group(1))
        text = re.sub(r"#(\d+)\b", _hash_num_sub, text)

        def _comma_num_sub(match):
            num = int(match.group(0).replace(",", ""))
            return self._num_to_words(num)

        text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", _comma_num_sub, text)

        def _standalone_num_sub(match):
            num_str = match.group(0)
            if len(num_str) > 1 and num_str.startswith("0"):
                return self._digits_to_words(num_str)
            num = int(num_str)
            return self._num_to_words(num)

        text = re.sub(r"\b\d+\b", _standalone_num_sub, text)
        return text

    def normalize(self, text: str) -> str:
        """
        Runs the full normalization pipeline for telephony speech generation.
        """
        if not text:
            return ""

        # 1. Clean whitespace and non-standard dashes/quotes
        t = " " + text.replace("—", ", ").replace("–", ", ").replace("…", "...") + " "
        t = t.replace('"', '').replace('“', '').replace('”', '')
        t = t.replace("’", "'")

        # 2. Apply common abbreviation substitutions
        for pattern, repl in self.ABBREVIATIONS:
            t = pattern.sub(repl, t)

        # 3. Currency
        t = self._normalize_currency(t)

        # 4. Dates
        t = self._normalize_dates(t)

        # 5. Times
        t = self._normalize_time(t)

        # 6. Phone Numbers
        t = self._normalize_phone_numbers(t)

        # 7. Percentages
        t = self._normalize_percentages(t)

        # 8. Remaining numbers
        t = self._normalize_isolated_numbers(t)

        # 9. Clean up extra punctuation and whitespace
        t = re.sub(r"\s+", " ", t).strip()
        t = re.sub(r"\s+([,.?!])", r"\1", t)
        return t

    def process(self, text: str) -> Tuple[str, Optional[str], bool]:
        """
        Main entrypoint: extracts emotion and breath tags, then normalizes text.
        Returns: (normalized_text, emotion, has_breath)
        """
        clean_text, emotion, has_breath = self.extract_emotion_and_tags(text)
        normalized_text = self.normalize(clean_text)
        return normalized_text, emotion, has_breath
