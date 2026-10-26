"""
Production English Text Normalizer for Speech Synthesis.
100% Deterministic expansion for phone numbers, currency, dates, times,
addresses, percentages, emails, and acronyms.
"""

import re
from typing import Optional

ABBREVIATIONS = {
    r"\bAve\b\.?": "Avenue",
    r"\bSt\b\.?": "Street",
    r"\bRd\b\.?": "Road",
    r"\bBlvd\b\.?": "Boulevard",
    r"\bDr\b\.?": "Drive",
    r"\bLn\b\.?": "Lane",
    r"\bSte\b\.?": "Suite",
    r"\bApt\b\.?": "Apartment",
    r"\bDept\b\.?": "Department",
    r"\bCorp\b\.?": "Corporation",
    r"\bInc\b\.?": "Incorporated",
    r"\bLLC\b": "L L C",
    r"\bLtd\b\.?": "Limited",
    r"\bCo\b\.?": "Company",
    r"\bGov\b\.?": "Government",
    r"\bAdmin\b\.?": "Administration",
    r"\bID\b": "I D",
    r"\bPIN\b": "Pin",
    r"\bSSN\b": "S S N",
    r"\bDOB\b": "Date of birth",
    r"\bFAQ\b": "F A Q",
    r"\bETA\b": "E T A",
    r"\bVIP\b": "V I P",
    r"\bASAP\b": "A S A P",
}

MONTHS = {
    "Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April",
    "May": "May", "Jun": "June", "Jul": "July", "Aug": "August",
    "Sep": "September", "Sept": "September", "Oct": "October",
    "Nov": "November", "Dec": "December"
}

DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"]

TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

ORDINALS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
    11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth", 15: "fifteenth",
    16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth", 20: "twentieth",
    21: "twenty-first", 22: "twenty-second", 23: "twenty-third", 24: "twenty-fourth", 25: "twenty-fifth",
    26: "twenty-sixth", 27: "twenty-seventh", 28: "twenty-eighth", 29: "twenty-ninth", 30: "thirtieth",
    31: "thirty-first"
}


def number_to_words(n: int) -> str:
    """Converts integer into spoken English words."""
    if n == 0:
        return "zero"
    if n < 0:
        return "negative " + number_to_words(-n)
    
    parts = []
    if n >= 1_000_000_000:
        b = n // 1_000_000_000
        parts.append(f"{number_to_words(b)} billion")
        n %= 1_000_000_000
    if n >= 1_000_000:
        m = n // 1_000_000
        parts.append(f"{number_to_words(m)} million")
        n %= 1_000_000
    if n >= 1_000:
        k = n // 1_000
        parts.append(f"{number_to_words(k)} thousand")
        n %= 1_000
    if n >= 100:
        h = n // 100
        parts.append(f"{ONES[h]} hundred")
        n %= 100
    if n >= 20:
        t = n // 10
        rem = n % 10
        if rem:
            parts.append(f"{TENS[t]}-{ONES[rem]}")
        else:
            parts.append(TENS[t])
    elif n > 0:
        parts.append(ONES[n])
        
    return " ".join(parts)


def normalize_phone_number(match: re.Match) -> str:
    raw = re.sub(r"[^\d]", "", match.group(0))
    if len(raw) == 11 and raw.startswith("1"):
        digits = list(raw)
        return (f"one, {digits[1]}{digits[2]}{digits[3]}, "
                f"{digits[4]}{digits[5]}{digits[6]}, "
                f"{digits[7]}{digits[8]}{digits[9]}{digits[10]}")
    elif len(raw) == 10:
        digits = list(raw)
        return (f"{digits[0]}{digits[1]}{digits[2]}, "
                f"{digits[3]}{digits[4]}{digits[5]}, "
                f"{digits[6]}{digits[7]}{digits[8]}{digits[9]}")
    return " ".join(DIGIT_WORDS[int(d)] for d in raw)


def normalize_currency(match: re.Match) -> str:
    amount_str = match.group(1).replace(",", "")
    try:
        if "." in amount_str:
            dollars, cents = amount_str.split(".", 1)
            dollars_num = int(dollars) if dollars else 0
            cents_num = int(cents[:2]) if cents else 0
            
            d_words = number_to_words(dollars_num)
            c_words = number_to_words(cents_num)
            
            d_label = "dollar" if dollars_num == 1 else "dollars"
            c_label = "cent" if cents_num == 1 else "cents"
            
            if dollars_num > 0 and cents_num > 0:
                return f"{d_words} {d_label} and {c_words} {c_label}"
            elif dollars_num > 0:
                return f"{d_words} {d_label}"
            else:
                return f"{c_words} {c_label}"
        else:
            dollars_num = int(amount_str)
            d_words = number_to_words(dollars_num)
            d_label = "dollar" if dollars_num == 1 else "dollars"
            return f"{d_words} {d_label}"
    except Exception:
        return match.group(0)


def normalize_dates(text: str) -> str:
    pattern = r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s*,?\s*(\d{4}))?\b"
    
    def _repl(m: re.Match) -> str:
        mon = MONTHS.get(m.group(1)[:3], m.group(1))
        day = int(m.group(2))
        day_str = ORDINALS.get(day, f"{day}th")
        year = m.group(3)
        if year:
            year_num = int(year)
            if 2000 <= year_num <= 2009:
                year_str = f"two thousand {number_to_words(year_num - 2000)}" if year_num > 2000 else "two thousand"
            elif 2010 <= year_num <= 2099:
                year_str = f"twenty {number_to_words(year_num - 2000)}"
            else:
                year_str = number_to_words(year_num)
            return f"{mon} {day_str}, {year_str}"
        return f"{mon} {day_str}"
        
    return re.sub(pattern, _repl, text, flags=re.IGNORECASE)


def normalize_times(text: str) -> str:
    pattern = r"\b(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?\b"
    
    def _repl(m: re.Match) -> str:
        hours = int(m.group(1))
        minutes = int(m.group(2))
        period = m.group(3)
        
        h_words = number_to_words(hours)
        if minutes == 0:
            m_words = "o'clock" if not period else ""
        elif minutes < 10:
            m_words = f"oh {ONES[minutes]}"
        else:
            m_words = number_to_words(minutes)
            
        period_words = f"{period[0].upper()} {period[1].upper()}" if period else ""
        return f"{h_words} {m_words} {period_words}".strip()
        
    return re.sub(pattern, _repl, text)


def normalize_emails(text: str) -> str:
    pattern = r"\b([a-zA-Z0-9_.+-]+)@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\b"
    
    def _repl(m: re.Match) -> str:
        user = m.group(1).replace(".", " dot ").replace("_", " underscore ").replace("-", " dash ")
        domain = m.group(2).replace(".", " dot ").replace("-", " dash ")
        return f"{user} at {domain}"
        
    return re.sub(pattern, _repl, text)


def normalize_english_text(text: str) -> str:
    """Main English text normalizer entrypoint."""
    if not text:
        return ""
        
    # 1. Protect Gesture/XML Tags
    tag_matches = []
    def _save_tags(m: re.Match) -> str:
        tag_matches.append(m.group(0))
        return f"__GESTURE_TAG_{len(tag_matches)-1}__"
        
    text = re.sub(r"<[^>]+>|\[[^\]]+\]", _save_tags, text)
    
    # 2. Currency
    text = re.sub(r"\$([0-9]+(?:,[0-9]{3})*(?:\.[0-9]{1,2})?)", normalize_currency, text)
    
    # 3. Percentages
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"\1 percent", text)
    
    # 4. Dates & Times
    text = normalize_dates(text)
    text = normalize_times(text)
    
    # 5. Emails
    text = normalize_emails(text)
    
    # 6. Phone Numbers
    phone_pattern = r"(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    text = re.sub(phone_pattern, normalize_phone_number, text)
    
    # 7. Common Abbreviations
    for abbrev, repl in ABBREVIATIONS.items():
        text = re.sub(abbrev, repl, text)
        
    # 8. Remaining standalone integers
    text = re.sub(r"\b(\d+)\b", lambda m: number_to_words(int(m.group(1))), text)
    
    # 9. Clean whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # 10. Restore Tags
    for i, tag in enumerate(tag_matches):
        text = text.replace(f"__GESTURE_TAG_{i}__", tag)
        
    return text
