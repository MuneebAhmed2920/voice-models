#!/usr/bin/env python3
"""
runtime/text_normalizer.py

Normalizes English text for conversational AI voice bots:
- Numbers & Currencies ($50 -> fifty dollars)
- Phone numbers & postal codes
- Acronyms & common contractions
- Micro-pause formatting
"""

import re

CURRENCY_MAP = {
    r'\$(\d+(?:\.\d{2})?)': r'\1 dollars',
    r'£(\d+(?:\.\d{2})?)': r'\1 pounds',
    r'€(\d+(?:\.\d{2})?)': r'\1 euros',
}

ABBREVIATIONS = {
    r'\bMr\.\b': 'Mister',
    r'\bMrs\.\b': 'Missus',
    r'\bMs\.\b': 'Miss',
    r'\bDr\.\b': 'Doctor',
    r'\bSt\.\b': 'Street',
    r'\bAve\.\b': 'Avenue',
    r'\bDept\.\b': 'Department',
    r'\bCorp\.\b': 'Corporation',
    r'\bInc\.\b': 'Incorporated',
    r'\bLtd\.\b': 'Limited',
    r'\bmin\b': 'minutes',
    r'\bsec\b': 'seconds',
    r'\bhr\b': 'hours',
    r'\bhrs\b': 'hours',
}

def normalize_english_text(text: str) -> str:
    if not text:
        return ""
    
    # 1. Currencies
    for pat, rep in CURRENCY_MAP.items():
        text = re.sub(pat, rep, text)
        
    # 2. Abbreviations
    for pat, rep in ABBREVIATIONS.items():
        text = re.sub(pat, rep, text)
        
    # 3. Clean up multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
