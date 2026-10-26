"""
Unit Tests for Arabic Voice Manager & Streamer (Pillars 4 & 6)
"""

import os
import pytest
from arabic_voice_model_kit.runtime.voice_manager import ArabicVoiceManager
from arabic_voice_model_kit.runtime.streamer import ArabicClauseStreamer


@pytest.fixture
def voice_mgr():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return ArabicVoiceManager(base_dir=base_dir)


@pytest.fixture
def streamer():
    return ArabicClauseStreamer(min_clause_words=2, max_clause_words=10)


def test_voice_manager_config(voice_mgr):
    cfg = voice_mgr.get_config("voice_arabic_clara")
    assert cfg["voice_id"] == "voice_arabic_clara"
    assert cfg["default_speed"] == 0.92
    assert cfg["default_guidance_scale"] == 2.2
    assert cfg["class_temperature"] == 0.20


def test_voice_manager_prompt_caching(voice_mgr):
    for emo in ["neutral", "excited", "sympathetic", "emotional"]:
        prompt = voice_mgr.get_prompt("voice_arabic_clara", emotion=emo)
        assert prompt is not None, f"Prompt for {emo} should be loaded"

    # Alias testing
    prompt_ar = voice_mgr.get_prompt("arabic", emotion="متحمس")
    assert prompt_ar is not None


def test_streamer_clause_chunking(streamer):
    text = "أهلاً وسهلاً بك في خدمة العملاء. كيف يمكنني مساعدتك اليوم؟"
    clauses = streamer.feed_token(text)
    assert len(clauses) >= 2
    # First clause should end with period and have ~280ms pause
    assert "أهلاً وسهلاً بك في خدمة العملاء." in clauses[0][0]
    assert clauses[0][1] == 280

    # Second clause
    assert "كيف يمكنني مساعدتك اليوم؟" in clauses[1][0]
    assert clauses[1][1] == 280


def test_streamer_comma_pauses(streamer):
    text = "أهلاً بك، تفضل بالسؤال"
    clauses = streamer.feed_token(text)
    # Comma clause
    assert len(clauses) >= 1
    assert clauses[0][1] == 120
