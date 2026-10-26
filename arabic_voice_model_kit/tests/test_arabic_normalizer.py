"""
Unit Tests for Arabic Telephony Text Normalizer (Pillar 7)
"""

import pytest
from arabic_voice_model_kit.runtime.normalizer import ArabicTelephonyNormalizer


@pytest.fixture
def normalizer():
    return ArabicTelephonyNormalizer()


def test_eastern_to_western_conversion(normalizer):
    res = normalizer.to_western_digits("رقم ٠١٢٣٤٥٦٧٨٩")
    assert res == "رقم 0123456789"


def test_number_to_words(normalizer):
    assert normalizer.number_to_words(0) == "صفر"
    assert normalizer.number_to_words(5) == "خمسة"
    assert normalizer.number_to_words(15) == "خمسة عشر"
    assert normalizer.number_to_words(45) == "خمسة وأربعين"
    assert normalizer.number_to_words(120) == "مئة وعشرين"
    assert normalizer.number_to_words(1250) == "ألف ومئتان وخمسين"


def test_currency_normalization(normalizer):
    # Saudi Riyal
    out_sar = normalizer.normalize("المبلغ هو 45 ريال")
    assert "خمسة وأربعين ريال" in out_sar

    # Eastern numerals with riyal
    out_sar_ar = normalizer.normalize("المبلغ ٤٥ ريال")
    assert "خمسة وأربعين ريال" in out_sar_ar

    # Egyptian pound
    out_egp = normalizer.normalize("سعر الكشف 120 جنيه")
    assert "مئة وعشرين جنيه" in out_egp

    # Dollar
    out_usd = normalizer.normalize("الإجمالي $1,250")
    assert "دولار" in out_usd
    assert "ألف" in out_usd


def test_time_normalization(normalizer):
    # ٤:٣٠ مساءً
    out1 = normalizer.normalize("الساعة ٤:٣٠ مساءً")
    assert "الرابعة والنصف مساءً" in out1

    # 10:15 صباحاً
    out2 = normalizer.normalize("الساعة 10:15 صباحاً")
    assert "العاشرة والربع صباحاً" in out2

    # 10:45
    out3 = normalizer.normalize("موعدك في تمام 10:45")
    assert "الحادية عشرة إلا ربع" in out3


def test_date_normalization(normalizer):
    out = normalizer.normalize("الموعد يوم 15/09/2026")
    assert "الخامس عشر من سبتمبر" in out
    assert "عشرين" in out or "ألفين" in out or "عام" in out


def test_percentage_normalization(normalizer):
    out1 = normalizer.normalize("نسبة الخصم 95%")
    assert "خمسة وتسعين بالمئة" in out1

    out2 = normalizer.normalize("النسبة ٢٠٪")
    assert "عشرين بالمئة" in out2


def test_process_tag_extraction(normalizer):
    text = "[متحمس] ألف مبروك! موعدك الساعة ٤:٣٠ مساءً."
    norm, emo, breath = normalizer.process(text)
    assert emo == "excited"
    assert breath is True
    assert "[" not in norm
    assert "الرابعة والنصف مساءً" in norm
