#!/usr/bin/env python3
"""
runtime/urdu_normalizer.py

Dedicated Pakistani Urdu Preprocessor & Phonetic Normalizer for Mahrukh Voice Model.
Ensures zero-lag pronunciation for English loanwords, proper female grammatical agreement,
and telephony identifier verbalization.
"""

import re
from typing import Tuple, List, Dict

# Phonetically optimized loanword dictionary for Pakistani Urdu speech
# Loanword dictionary: preserves authentic English pronunciation in Latin script for telecom/technical terms
LOANWORD_DICTIONARY: Dict[str, str] = {
    "asalamualikum": "اَلسَّلَامُ عَلَیْکُم",
    "assalamualikum": "اَلسَّلَامُ عَلَیْکُم",
    "asalam-o-alaikum": "اَلسَّلَامُ عَلَیْکُم",
    "assalam-o-alaikum": "اَلسَّلَامُ عَلَیْکُم",
    "office": "office",
    "complete": "complete",
    "address": "address",
    "delivery": "delivery",
    "details": "details",
    "update": "update",
    "confirm": "confirm",
    "order": "order",
    "payment": "payment",
    "account": "Account",
    "check": "check",
    "package": "Package",
    "packages": "Packages",
    "reference": "reference",
    "number": "نمبر",
    "service": "Service",
    "customer": "Customer",
    "process": "process",
    "active": "active",
    "request": "request",
    "dispatch": "dispatch",
    "receive": "receive",
    "internet": "Internet",
    "mobile": "Mobile",
    "whatsapp": "WhatsApp",
    "email": "Email",
    "appointment": "appointment",
    "cancel": "cancel",
    "balance": "Balance",
    "bill": "Bill",
    "billing": "Billing",
    "sms": "SMS",
    "tracking": "tracking",
    "nayatel": "Nayatel",
    "naya-tel": "Nayatel",
    "nayatal": "Nayatel",
    "fiber": "Fiber",
    "fiber optic": "Fiber Optic",
    "broadband": "Broadband",
    "connection": "Connection",
    "connections": "Connections",
    "router": "Router",
    "wifi": "WiFi",
    "wi-fi": "WiFi",
    "support": "Support",
    "technical": "Technical",
    "speed": "Speed",
    "helpline": "Helpline",
    "representative": "representative",
    "call": "Call",
    "portal": "Portal",
    "network": "Network",
    "status": "Status",
    "restart": "Restart",
    "reboot": "Reboot",
    "hd": "HD",
    "green": "Green",
    "red": "Red",
    "light": "Light",
    "blink": "Blink",
    "blinking": "Blinking",
    "refresh": "Refresh",
    "browse": "Browse",
    "upgrade": "Upgrade",
    "care": "Support",
    "customer care": "Customer Support",
    "customercare": "Customer Support",
    "customer support": "Customer Support",
    "customersupport": "Customer Support",
    "ont": "ONT",
    "onu": "ONU",
    "pon": "PON",
    "los": "LOS",
    "patch cord": "patch cord",
    "patchcord": "patch cord",
    "optical power": "optical power",
    "optical": "optical",
    "power": "power",
    "attenuation": "attenuation",
    "dbm": "dBm",
    "dual band": "dual-band",
    "dual-band": "dual-band",
    "ssid": "SSID",
    "frequency": "frequency",
    "channel": "channel",
    "interference": "interference",
    "latency": "latency",
    "ping": "ping",
    "ping spikes": "ping spikes",
    "jitter": "jitter",
    "packet loss": "packet loss",
    "throttle": "throttle",
    "throttling": "throttling",
    "bandwidth": "bandwidth",
    "static ip": "static IP",
    "public ip": "public IP",
    "ip": "IP",
    "ip address": "IP address",
    "subnet mask": "subnet mask",
    "gateway": "gateway",
    "default gateway": "default gateway",
    "dns": "DNS",
    "port forwarding": "port forwarding",
    "bridge mode": "bridge mode",
    "cctv": "CCTV",
    "dvr": "DVR",
    "nvr": "NVR",
    "nwatch": "NWatch",
    "n-watch": "NWatch",
    "joybox": "JoyBox",
    "joy box": "JoyBox",
    "smart box": "smart box",
    "smartbox": "smart box",
    "live tv": "Live TV",
    "streams": "streams",
    "stream": "stream",
    "firmware": "firmware",
    "cache": "cache",
    "motion detection": "motion detection",
    "cloud": "cloud",
    "cloud recording": "cloud recording",
    "cloud storage": "cloud storage",
    "hdmi": "HDMI",
    "otdr": "OTDR",
    "splicing": "splicing",
    "jointing": "jointing",
    "outage": "outage",
    "backbone": "backbone",
    "sla": "SLA",
    "kuickpay": "Kuickpay",
    "1link": "1Link",
    "fbr": "FBR",
    "pra": "PRA",
    "withholding tax": "withholding tax",
    "exemption certificate": "exemption certificate",
    "customer portal": "Customer Portal",
    "corporate invoice": "corporate invoice",
    "invoice": "invoice",
    "billing cycle": "billing cycle",
    "speed test": "speed test",
    "speedtest": "speed test",
    "download": "download",
    "upload": "upload",
    "backend": "backend",
    "optimize": "optimize",
    "field team": "field team",
    "field tech": "field tech",
    "support team": "support team",
    "enterprise": "Enterprise",
    "power cycle": "power cycle",
    "fusion splicing": "fusion splicing",
    "unplanned": "unplanned",
    "cell phone": "cell phone",
    "timeline": "timeline",
    "video-on-demand": "video-on-demand",
    "consumer id": "Consumer ID",
    "ookla": "Ookla",
    "auto channel": "auto channel",
}

# Phonetic corrections for native/transcribed Urdu text:
# Routes Urdu telecom transliterations to authentic English words for flawless native acoustics
URDU_PHONETIC_CORRECTIONS: Dict[str, str] = {
    r"\b(اَلسَّلَامُ|السلام|اسلام)\s*(عَلَیْکُم|علیکم|و\s*علیکم)\s*([!۔،])?": "اَلسَّلَامُ عَلَیْکُم، ",
    r"\b(نیاٹیل|نایاتل|نیا\s*ٹیل)\b": "Nayatel",
    r"\b(کسٹمر\s*کیئر|کسٹمر\s*سپورٹ|کسٹمر)\b": "Customer Support",
    r"\b(وائی\s*فائی)\b": "WiFi",
    r"\b(راؤٹر)\b": "Router",
    r"\b(انٹرنیٹ)\b": "Internet",
    r"\b(سپیڈ)\b": "Speed",
    r"\b(فائبر\s*آپٹک|فائبر)\b": "Fiber",
    r"\b(کنکشن)\b": "Connection",
    r"\b(ریفریش)\b": "Refresh",
    r"\b(براؤز)\b": "Browse",
    r"\b(بلنک)\b": "Blink",
    r"\b(اپ\s*گریڈ)\b": "Upgrade",
    r"\bٹیکنیکل\b": "Technical",
    r"\bنیٹورک\b": "Network",
    r"\bریسیو\b": "receive",
    r"\b(پیکیج|پیکج)\b": "Package",
    r"\b(ڈسپَیچ|ڈسپیچ|ڈِسپیچ|ڈِس\s*پَیچ)\b": "dispatch",
    r"\bکن\s*فرم\b": "confirm",
    r"\b(پیمنٹ|پے\s*منٹ)\b": "payment",
    r"\b(ڈی\s*لیوری|ڈیلِوری|ڈلیوری)\b": "delivery",
    r"\bپراسیس\b": "process",
}

# Feminine grammatical agreement replacements for female speaker
FEMININE_REPLACEMENTS: Dict[str, str] = {
    r"\bکرتا\s+ہوں\b": "کرتی ہوں",
    r"\bکر\s+سکتا\s+ہوں\b": "کر سکتی ہوں",
    r"\bکر\s+دیتا\s+ہوں\b": "کر دیتی ہوں",
    r"\bکر\s+لیتا\s+ہوں\b": "کر لیتی ہوں",
    r"\bسکتا\s+ہوں\b": "سکتی ہوں",
    r"\bدیتا\s+ہوں\b": "دیتی ہوں",
    r"\bلیتا\s+ہوں\b": "لیتی ہوں",
}

# Urdu digit words for digit-by-digit verbalization
URDU_DIGITS = {
    "0": "صفر",
    "1": "ایک",
    "2": "دو",
    "3": "تین",
    "4": "چار",
    "5": "پانچ",
    "6": "چھ",
    "7": "سات",
    "8": "آٹھ",
    "9": "نو",
}

# Small cardinals for quantities
URDU_CARDINALS = {
    "0": "صفر",
    "1": "ایک",
    "2": "دو",
    "3": "تین",
    "4": "چار",
    "5": "پانچ",
    "6": "چھ",
    "7": "سات",
    "8": "آٹھ",
    "9": "نو",
    "10": "دس",
}

def normalize_urdu_text(text: str, female_mode: bool = True) -> str:
    """
    Normalizes Urdu text containing English loanwords, numbers, and gender inflection
    for the Mahrukh Pakistani Urdu Voice Model.
    """
    normalized = text
    
    # 1. Feminine Grammatical Agreement
    if female_mode:
        for pattern, repl in FEMININE_REPLACEMENTS.items():
            normalized = re.sub(pattern, repl, normalized)
            
    # 2. Number Normalization
    def replace_identifier(match):
        prefix = match.group(1)
        digits = match.group(2)
        spoken_digits = "، ".join(URDU_DIGITS.get(d, d) for d in digits)
        return f"{prefix}{spoken_digits}"

    id_pattern = r"(\b(?:reference\s+number|number|reference|ریفرنس\s+نمبر|نمبر|ریفرنس|کوڈ|code|id)\s*(?:ہے\s*[:\s]*)?)(\d+)"
    normalized = re.sub(id_pattern, replace_identifier, normalized, flags=re.IGNORECASE)
    
    def replace_standalone_multidigit(match):
        digits = match.group(0)
        return "، ".join(URDU_DIGITS.get(d, d) for d in digits)
        
    normalized = re.sub(r"\b\d{4,}\b", replace_standalone_multidigit, normalized)
    
    def replace_cardinal(match):
        num_str = match.group(0)
        return URDU_CARDINALS.get(num_str, num_str)
        
    normalized = re.sub(r"\b\d{1,2}\b", replace_cardinal, normalized)

    # 3. Apply approved loanword mappings (longer multi-word phrases first)
    sorted_loanwords = sorted(LOANWORD_DICTIONARY.keys(), key=len, reverse=True)
    for phrase in sorted_loanwords:
        replacement = LOANWORD_DICTIONARY[phrase]
        pattern = rf"(?i)\b{re.escape(phrase)}\b"
        normalized = re.sub(pattern, replacement, normalized)

    # 4. Apply Urdu phonetic corrections
    for pattern, repl in URDU_PHONETIC_CORRECTIONS.items():
        normalized = re.sub(pattern, repl, normalized)
                    
    # 5. Script Boundary Spacing & Prosodic Punctuation Cleanup
    # Ensure natural phonetic separation between Urdu Perso-Arabic and English Latin scripts
    normalized = re.sub(r"([\u0600-\u06FF])([a-zA-Z])", r"\1 \2", normalized)
    normalized = re.sub(r"([a-zA-Z])([\u0600-\u06FF])", r"\1 \2", normalized)
    
    # Prosodic commas & periods for natural human breath pauses
    normalized = re.sub(r"\s*،\s*", "، ", normalized)
    normalized = re.sub(r"\s*,\s*", ", ", normalized)
    normalized = re.sub(r"\s*۔\s*", "۔ ", normalized)
    normalized = re.sub(r"\s*\.\s*", ". ", normalized)
    normalized = re.sub(r"\s*؟\s*", "؟ ", normalized)
    normalized = re.sub(r"\s*\?\s*", "? ", normalized)
    normalized = re.sub(r"\s*!\s*", "! ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    
    return normalized

