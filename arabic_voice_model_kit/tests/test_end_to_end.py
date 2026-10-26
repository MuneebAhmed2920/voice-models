"""
End-to-End Integration Tests for Arabic Voice Model Kit
Verifies complete pipeline from text input with Arabic emotion tags to encoded audio output.
"""

import os
import pytest
import soundfile as sf
from arabic_voice_model_kit.runtime.engine import ArabicVoiceEngine


@pytest.fixture(scope="module")
def engine():
    # Model is loaded from local HuggingFace cache
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return ArabicVoiceEngine(warmup=False)


def test_e2e_synthesize_greeting(engine):
    text = "أهلاً وسهلاً بحضرتك، تفضل كيف أقدر أساعدك اليوم؟"
    audio_bytes, metrics = engine.synthesize(
        text=text,
        voice="voice_arabic_clara",
        sample_rate=24000,
        format_type="wav",
        ode_steps=12,
        speed=0.92,
        add_breath=True,
        apply_eq=True
    )
    assert len(audio_bytes) > 1000
    assert metrics["audio_duration_sec"] > 2.0
    assert metrics["has_breath"] is True
    assert metrics["sample_rate"] == 24000
    assert metrics["format"] == "wav"


def test_e2e_synthesize_telephony_8k_pcm16(engine):
    text = "[متحمس] تم تأكيد حجزك بنجاح!"
    audio_bytes, metrics = engine.synthesize(
        text=text,
        voice="voice_arabic_clara",
        sample_rate=8000,
        format_type="pcm16",
        ode_steps=10,
        speed=0.92
    )
    assert len(audio_bytes) > 1000
    assert metrics["sample_rate"] == 8000
    assert metrics["format"] == "pcm16"
    assert metrics["emotion"] == "excited"
