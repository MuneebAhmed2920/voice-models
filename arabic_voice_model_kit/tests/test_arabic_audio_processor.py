"""
Unit Tests for Arabic Audio Processor & Telephony Codecs (Pillar 5)
"""

import numpy as np
import pytest
from arabic_voice_model_kit.runtime.audio_processor import ArabicAudioProcessor


@pytest.fixture
def audio_proc():
    return ArabicAudioProcessor(default_sample_rate=8000, apply_mastering_eq=True)


def test_hpf_dc_cleanup(audio_proc):
    sr = 24000
    # Signal with strong DC offset + 10 Hz rumble + 500 Hz tone
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    sig = 0.5 + 0.3 * np.sin(2 * np.pi * 10 * t) + 0.2 * np.sin(2 * np.pi * 500 * t)
    sig = sig.astype(np.float32)

    filtered = audio_proc.apply_high_pass_filter(sig, sample_rate=sr, cutoff_hz=75.0)
    # Mean of filtered should be near 0 (DC removed)
    assert abs(np.mean(filtered)) < 0.05


def test_consonant_presence_boost(audio_proc):
    sr = 24000
    # 4kHz sine wave
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    tone_4k = np.sin(2 * np.pi * 4000 * t).astype(np.float32) * 0.2

    boosted = audio_proc.apply_consonant_presence(tone_4k, sample_rate=sr, center_hz=4000.0, gain_db=2.0)
    rms_orig = np.sqrt(np.mean(tone_4k ** 2))
    rms_boosted = np.sqrt(np.mean(boosted ** 2))
    assert rms_boosted > rms_orig


def test_soft_knee_limiter(audio_proc):
    # Oversaturated signal exceeding 1.0
    sig = np.array([-1.5, -0.9, 0.0, 0.9, 1.5], dtype=np.float32)
    limited = audio_proc.apply_soft_knee_limiting(sig, threshold=0.85, ceiling=0.95)
    assert np.all(np.abs(limited) <= 0.95)
    # Signs should be preserved
    assert np.all(np.sign(limited) == np.sign(sig))


def test_resampling(audio_proc):
    sr_orig = 24000
    sr_target = 8000
    sig = np.sin(2 * np.pi * 300 * np.linspace(0, 1.0, sr_orig)).astype(np.float32)
    resampled = audio_proc.resample(sig, sr_orig, sr_target)
    assert len(resampled) == sr_target


def test_telephony_codecs(audio_proc):
    sig = np.linspace(-0.8, 0.8, 800, dtype=np.float32)

    # PCM16
    pcm_bytes = audio_proc.float_to_pcm16(sig)
    assert len(pcm_bytes) == len(sig) * 2
    decoded_pcm = audio_proc.pcm16_to_float(pcm_bytes)
    assert np.allclose(sig, decoded_pcm, atol=1e-3)

    # μ-law
    mulaw_bytes = audio_proc.linear_to_mulaw(sig)
    assert len(mulaw_bytes) == len(sig)

    # A-law
    alaw_bytes = audio_proc.linear_to_alaw(sig)
    assert len(alaw_bytes) == len(sig)

    # WAV container
    wav_bytes = audio_proc.to_wav_bytes(sig, sample_rate=8000)
    assert wav_bytes.startswith(b"RIFF")
    assert b"WAVE" in wav_bytes


def test_20ms_framing(audio_proc):
    sr = 8000
    # 1 second of 8kHz 16-bit PCM = 16000 bytes
    # 20ms frame = 160 samples = 320 bytes
    dummy_bytes = b"\x00" * 16000
    frames = list(audio_proc.frame_audio(dummy_bytes, sample_rate=sr, frame_duration_ms=20, bytes_per_sample=2))
    assert len(frames) == 50  # 50 frames per second
    assert all(len(f) == 320 for f in frames)
