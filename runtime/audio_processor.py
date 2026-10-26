#!/usr/bin/env python3
"""
runtime/audio_processor.py

Telephony audio processor for Mahrukh Urdu Voice Model.
Supports 8kHz G.711 PSTN filtering, 16kHz HD Voice, and 24kHz studio master formats.
"""

import io
import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, lfilter
from typing import Tuple, Optional

def apply_telephony_filter(audio: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
    """
    Applies standard telephony bandpass filter (300 Hz - 3400 Hz).
    Simulates authentic PSTN / landline phone receiver frequency response.
    """
    nyquist = 0.5 * sample_rate
    low = 300.0 / nyquist
    high = min(3400.0 / nyquist, 0.99)
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, audio).astype(np.float32)

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resamples audio tensor from orig_sr to target_sr."""
    if orig_sr == target_sr:
        return audio
    import torchaudio.functional as F
    tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    resampled = F.resample(tensor, orig_freq=orig_sr, new_freq=target_sr).squeeze(0).numpy()
    return resampled

def process_telephony_audio(
    audio: np.ndarray,
    orig_sr: int = 24000,
    target_sr: int = 8000,
    enable_filter: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Complete telephony & studio audio processing pipeline:
    1. Bandpass filter (300-3400 Hz) if enabled
    2. Resample to target rate (e.g. 8000 Hz or 24000 Hz)
    3. Smooth envelope fade-in/fade-out to eliminate edge clicks
    4. Safe peak normalization to 0.90 (-0.9 dBFS)
    """
    processed = audio
    if enable_filter:
        processed = apply_telephony_filter(processed, orig_sr)
        
    if target_sr != orig_sr:
        processed = resample_audio(processed, orig_sr, target_sr)
        
    # Smooth envelope fade-in (10ms) and fade-out (30ms) to ensure organic acoustic boundaries
    fade_in_len = int(0.010 * target_sr)
    fade_out_len = int(0.030 * target_sr)
    if len(processed) > fade_in_len + fade_out_len:
        fade_in = np.linspace(0.0, 1.0, fade_in_len, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_out_len, dtype=np.float32)
        processed[:fade_in_len] *= fade_in
        processed[-fade_out_len:] *= fade_out

    peak = np.max(np.abs(processed))
    if peak > 0:
        processed = (processed / peak) * 0.90
        
    return processed, target_sr

def to_wav_bytes(audio: np.ndarray, sample_rate: int, subtype: str = "PCM_16") -> bytes:
    """Encodes float32 numpy audio into WAV byte stream."""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype=subtype)
    return buf.getvalue()
