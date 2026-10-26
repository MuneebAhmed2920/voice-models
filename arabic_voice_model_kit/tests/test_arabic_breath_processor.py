"""
Unit Tests for Arabic Physiological Breath Processor (Pillar 1 & 6)
"""

import os
import numpy as np
import pytest
from arabic_voice_model_kit.runtime.arabic_breath_processor import ArabicBreathProcessor


@pytest.fixture
def breath_proc():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_dir = os.path.join(base_dir, "prompt")
    return ArabicBreathProcessor(prompt_dir=prompt_dir, sample_rate=24000)


def test_tag_parsing(breath_proc):
    text = "[متحمس] أهلاً بك معنا اليوم!"
    clean, emo, breath = breath_proc.parse_arabic_tags(text)
    assert emo == "excited"
    assert breath is True
    assert "[" not in clean and "]" not in clean
    assert "أهلاً بك معنا اليوم!" in clean

    text_symp = "[متعاطف] أعتذر جداً عن التأخير."
    clean, emo, breath = breath_proc.parse_arabic_tags(text_symp)
    assert emo == "sympathetic"
    assert breath is True

    text_emo = "[مؤثر] شكراً جزيلاً لثقتكم الغالية."
    clean, emo, breath = breath_proc.parse_arabic_tags(text_emo)
    assert emo == "emotional"
    assert breath is True

    text_no_breath = "[طبيعي][no_breath] مرحباً بك."
    clean, emo, breath = breath_proc.parse_arabic_tags(text_no_breath)
    assert emo == "neutral"
    assert breath is False


def test_emotion_tailored_breath_shapes(breath_proc):
    sr = 24000
    for emo, dur_target, gap_target in [
        ("neutral", 140, 50),
        ("excited", 95, 25),
        ("sympathetic", 210, 75),
        ("emotional", 165, 60)
    ]:
        breath_audio = breath_proc.create_tailored_breath(emotion=emo)
        total_ms = dur_target + gap_target
        expected_samples = int((total_ms / 1000.0) * sr)
        assert abs(len(breath_audio) - expected_samples) <= 2

        # Check silence gap at end
        gap_samples = int((gap_target / 1000.0) * sr)
        assert np.all(breath_audio[-gap_samples:] == 0.0)

        # Check peak amplitude is controlled
        peak = np.max(np.abs(breath_audio))
        assert 0.005 < peak < 0.10


def test_inject_pre_speech_breath(breath_proc):
    sr = 24000
    dummy_speech = np.ones(int(sr * 1.0), dtype=np.float32) * 0.5
    combined = breath_proc.inject_pre_speech_breath(dummy_speech, emotion="neutral", include_breath=True)

    neutral_samples = int((190 / 1000.0) * sr)
    assert abs(len(combined) - (len(dummy_speech) + neutral_samples)) <= 2

    # Without breath
    raw = breath_proc.inject_pre_speech_breath(dummy_speech, emotion="neutral", include_breath=False)
    assert np.array_equal(raw, dummy_speech)


def test_conjunction_micro_breaths(breath_proc):
    long_clause = "يسعدنا دائماً تقديم أفضل الخدمات لجميع عملائنا الكرام ولكن نحرص أيضاً على المتابعة الدقيقة"
    out = breath_proc.inject_conjunction_micro_breaths(long_clause, min_clause_words=5)
    assert "، ولكن" in out or "، لكن" in out
