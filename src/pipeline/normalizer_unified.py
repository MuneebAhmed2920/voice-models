"""Unified text normalizer for telephony TTS.

Wraps ArabicNormalizer and adds missing telephony-specific formats:
  - Phone numbers (Arabic + international formats)
  - Email addresses
  - URLs / domain names
  - Decimal numbers (e.g., 3.14 → ثلاثة فاصل واحد أربعة)
  - Common acronyms (SMS, OTP, PIN, VAT, etc.)
  - English contractions and connected-speech reductions
  - Ordinal numbers in English (1st, 2nd, 3rd)
  - IP addresses
  - Arabic-specific: large numbers, model numbers, version strings

All methods are stateless and return the transformed string.
"""

import re
from typing import Optional

from ..arabic_nlp.normalizer import ArabicNormalizer


# ──────────────────────────────────────────────────────────────────
# Arabic digit lookup (for phone number spelling)
# ──────────────────────────────────────────────────────────────────
_AR_DIGIT = {
    "0": "صفر", "1": "واحد", "2": "اثنان", "3": "ثلاثة", "4": "أربعة",
    "5": "خمسة", "6": "ستة", "7": "سبعة", "8": "ثمانية", "9": "تسعة",
}

_EN_DIGIT = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}

# ──────────────────────────────────────────────────────────────────
# Acronym expansions — Arabic TTS-safe
# ──────────────────────────────────────────────────────────────────
_ACRONYM_AR = {
    "OTP": "أو تي بي",
    "PIN": "بين",
    "SMS": "إس إم إس",
    "MMS": "إم إم إس",
    "SIM": "سيم",
    "VAT": "ضريبة القيمة المضافة",
    "ATM": "أيه تي إم",
    "ID": "أي دي",
    "URL": "يو آر إل",
    "PDF": "بي دي إف",
    "GPS": "جي بي إس",
    "VIP": "في آي بي",
    "CEO": "سي إي أو",
    "PO": "صندوق البريد",
    "KSA": "المملكة العربية السعودية",
    "UAE": "الإمارات العربية المتحدة",
    "KWD": "دينار كويتي",
    "SAR": "ريال سعودي",
    "AED": "درهم إماراتي",
    "EGP": "جنيه مصري",
    "USD": "دولار أمريكي",
    "EUR": "يورو",
    "GBP": "جنيه إسترليني",
    "CV": "سيرة ذاتية",
    "HR": "إدارة الموارد البشرية",
    "IT": "تقنية المعلومات",
}

# ──────────────────────────────────────────────────────────────────
# English contraction expansions
# ──────────────────────────────────────────────────────────────────
_EN_CONTRACTIONS = {
    r"\bwon't\b": "will not",
    r"\bcan't\b": "cannot",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bisn't\b": "is not",
    r"\baren't\b": "are not",
    r"\bwasn't\b": "was not",
    r"\bweren't\b": "were not",
    r"\bhadn't\b": "had not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bshouldn't\b": "should not",
    r"\bwouldn't\b": "would not",
    r"\bcouldn't\b": "could not",
    r"\bI'm\b": "I am",
    r"\bI'll\b": "I will",
    r"\bI've\b": "I have",
    r"\bI'd\b": "I would",
    r"\byou're\b": "you are",
    r"\byou'll\b": "you will",
    r"\byou've\b": "you have",
    r"\bthey're\b": "they are",
    r"\bthey'll\b": "they will",
    r"\bwe're\b": "we are",
    r"\bwe'll\b": "we will",
    r"\bwe've\b": "we have",
    r"\bhe's\b": "he is",
    r"\bshe's\b": "she is",
    r"\bit's\b": "it is",
    r"\bthat's\b": "that is",
    r"\bthere's\b": "there is",
    r"\blet's\b": "let us",
}

# English ordinals
_EN_ORDINALS = {
    "1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth",
    "5th": "fifth", "6th": "sixth", "7th": "seventh", "8th": "eighth",
    "9th": "ninth", "10th": "tenth", "11th": "eleventh", "12th": "twelfth",
    "13th": "thirteenth", "14th": "fourteenth", "15th": "fifteenth",
    "20th": "twentieth", "21st": "twenty-first", "30th": "thirtieth",
    "31st": "thirty-first",
}


def _digits_to_arabic_words(digits: str) -> str:
    """Converts a string of digits to Arabic words with short pauses (commas)."""
    return " ".join(_AR_DIGIT.get(d, d) for d in digits)


def _digits_to_english_words(digits: str) -> str:
    """Converts a string of digits to English words."""
    return " ".join(_EN_DIGIT.get(d, d) for d in digits)


class UnifiedNormalizer:
    """
    Full telephony-grade text normalizer for Arabic + English TTS.

    Usage:
        norm = UnifiedNormalizer(dialect='msa')
        clean_text = norm.normalize(raw_text)
    """

    def __init__(self, dialect: str = "msa"):
        self.dialect = dialect.lower()
        self._ar_norm = ArabicNormalizer(dialect=dialect)

    def normalize(self, text: str) -> str:
        """Complete normalization pipeline — Arabic + English."""
        if not text:
            return ""

        # 1. Normalize acronyms first (before case-folding destroys them)
        text = self._normalize_acronyms(text)

        # 2. Expand English contractions
        text = self._expand_english_contractions(text)

        # 3. Normalize English ordinals
        text = self._normalize_english_ordinals(text)

        # 4. Normalize phone numbers (before generic number normalization)
        text = self._normalize_phone_numbers(text)

        # 5. Normalize email addresses
        text = self._normalize_emails(text)

        # 6. Normalize URLs / domain names
        text = self._normalize_urls(text)

        # 7. Normalize decimal numbers (before integer normalization)
        text = self._normalize_decimals(text)

        # 8. Normalize IP addresses
        text = self._normalize_ip_addresses(text)

        # 9. Run the base Arabic normalizer (handles dates, times, currency, %, fractions, integers)
        text = self._ar_norm.normalize(text)

        return text

    # ──────────────────────────────────────────────────────────────
    # Phone numbers
    # ──────────────────────────────────────────────────────────────
    def _normalize_phone_numbers(self, text: str) -> str:
        """
        Converts phone numbers into Arabic digit-word sequences.

        Handles:
          +966501234567, 00966501234567, 0501234567, 05012-34567
          Saudi: 05xx-xxxxxxx  Gulf: 9714, 9715  Egypt: 01x
        """
        def phone_repl(match: re.Match) -> str:
            raw = match.group(0)
            # Strip all non-digit characters
            digits = re.sub(r"[^\d]", "", raw)
            if len(digits) < 7:
                return raw  # Not a phone number — leave alone
            return _digits_to_arabic_words(digits)

        # Pattern 1: International format +966..., 00966...
        intl_pattern = re.compile(
            r"(?<![\d])(?:\+|00)\d{10,14}(?![\d])"
        )
        text = intl_pattern.sub(phone_repl, text)

        # Pattern 2: Local mobile numbers starting with 0 followed by 9 more digits
        # Matches 0XXXXXXXXX (10 digits) — Saudi/Gulf/Egypt local format
        local_pattern = re.compile(
            r"(?<![\d])(0[5-9]\d{8})(?![\d])"
        )
        text = local_pattern.sub(phone_repl, text)

        # Pattern 3: 9-digit numbers that look like service numbers (9XXXXXXXX)
        service_pattern = re.compile(
            r"(?<![\d])(9[012]\d{7})(?![\d])"
        )
        text = service_pattern.sub(phone_repl, text)

        return text

    # ──────────────────────────────────────────────────────────────
    # Email addresses
    # ──────────────────────────────────────────────────────────────
    def _normalize_emails(self, text: str) -> str:
        """
        Reads email addresses character by character in a natural way.
        e.g., info@company.com → إنفو آت كومباني دوت كوم
        """
        def email_repl(match: re.Match) -> str:
            addr = match.group(0)
            # Split into user + domain
            parts = addr.split("@", 1)
            if len(parts) != 2:
                return addr
            user, domain = parts
            # Speak user part letter by letter if short, or as a word
            user_spoken = user if len(user) <= 8 else " ".join(user)
            # Speak @ as "آت"
            at_spoken = "آت"
            # Speak domain: replace . with "دوت"
            domain_spoken = domain.replace(".", " دوت ")
            return f"{user_spoken} {at_spoken} {domain_spoken}"

        email_pattern = re.compile(
            r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
        )
        return email_pattern.sub(email_repl, text)

    # ──────────────────────────────────────────────────────────────
    # URLs
    # ──────────────────────────────────────────────────────────────
    def _normalize_urls(self, text: str) -> str:
        """
        Converts URLs to speakable form.
        www.example.com → ويب سايت إكزامبل دوت كوم
        https://example.com/page → موقع إكزامبل دوت كوم
        """
        def url_repl(match: re.Match) -> str:
            url = match.group(0)
            # Strip protocol
            cleaned = re.sub(r"https?://", "", url)
            cleaned = re.sub(r"^www\.", "", cleaned)
            # Strip trailing path for brevity in speech
            domain = cleaned.split("/")[0]
            spoken = domain.replace(".", " دوت ").replace("-", " ")
            prefix = "موقع " if "http" in url else "ويب سايت "
            return prefix + spoken

        url_pattern = re.compile(
            r"https?://[^\s,،]+|www\.[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}[^\s,،]*"
        )
        return url_pattern.sub(url_repl, text)

    # ──────────────────────────────────────────────────────────────
    # Decimal numbers
    # ──────────────────────────────────────────────────────────────
    def _normalize_decimals(self, text: str) -> str:
        """
        Converts decimal numbers to Arabic spoken form.
        3.14 → ثلاثة فاصل واحد أربعة
        12.5 → اثنا عشر فاصل خمسة
        """
        def decimal_repl(match: re.Match) -> str:
            int_part = int(match.group(1))
            dec_str = match.group(2)
            int_word = self._ar_norm.number_to_arabic(int_part)
            # Speak decimal digits individually
            dec_word = " ".join(_AR_DIGIT.get(d, d) for d in dec_str)
            return f"{int_word} فاصل {dec_word}"

        # Match decimals that are NOT already part of a time or date pattern
        decimal_pattern = re.compile(r"(?<!\d)(\d{1,6})\.(\d{1,4})(?!\d)(?![/\-])")
        return decimal_pattern.sub(decimal_repl, text)

    # ──────────────────────────────────────────────────────────────
    # IP addresses
    # ──────────────────────────────────────────────────────────────
    def _normalize_ip_addresses(self, text: str) -> str:
        """192.168.1.1 → مائة وتسعة واثنتان وستون نقطة ..."""
        def ip_repl(match: re.Match) -> str:
            parts = match.group(0).split(".")
            spoken = " نقطة ".join(self._ar_norm.number_to_arabic(int(p)) for p in parts)
            return spoken

        ip_pattern = re.compile(
            r"\b(25[0-5]|2[0-4]\d|[01]?\d\d?)\."
            r"(25[0-5]|2[0-4]\d|[01]?\d\d?)\."
            r"(25[0-5]|2[0-4]\d|[01]?\d\d?)\."
            r"(25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        )
        return ip_pattern.sub(ip_repl, text)

    # ──────────────────────────────────────────────────────────────
    # Acronyms
    # ──────────────────────────────────────────────────────────────
    def _normalize_acronyms(self, text: str) -> str:
        """Expands known acronyms to their full Arabic spoken form."""
        for acronym, expansion in _ACRONYM_AR.items():
            # Match whole word, case-insensitive for English acronyms
            pattern = re.compile(r"\b" + re.escape(acronym) + r"\b", re.IGNORECASE)
            text = pattern.sub(expansion, text)
        return text

    # ──────────────────────────────────────────────────────────────
    # English contractions
    # ──────────────────────────────────────────────────────────────
    def _expand_english_contractions(self, text: str) -> str:
        """Expands English contractions for natural connected speech."""
        for pattern, expansion in _EN_CONTRACTIONS.items():
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        return text

    # ──────────────────────────────────────────────────────────────
    # English ordinals
    # ──────────────────────────────────────────────────────────────
    def _normalize_english_ordinals(self, text: str) -> str:
        """Converts English ordinal numbers to words."""
        for ordinal, word in _EN_ORDINALS.items():
            text = re.sub(r"\b" + re.escape(ordinal) + r"\b", word, text, flags=re.IGNORECASE)
        return text
