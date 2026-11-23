#!/usr/bin/env python3
"""
runtime/audio_processor.py

Processes 24kHz studio HD audio and converts it to G.711 standard
8kHz telephony PSTN bandpass filtered audio (300Hz-3400Hz).
"""

import io
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
import librosa

def butter_bandpass_filter(data: np.ndarray, lowcut: float = 300.0, highcut: float = 3400.0, fs: int = 8000, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    filtered = sosfilt(sos, data)
    return filtered.astype(np.float32)

def process_telephony_audio(audio_24k: np.ndarray, orig_sr: int = 24000, target_sr: int = 8000, apply_filter: bool = True) -> np.ndarray:
    # 1. Resample to 8kHz
    if orig_sr != target_sr:
        audio_8k = librosa.resample(audio_24k, orig_sr=orig_sr, target_sr=target_sr)
    else:
        audio_8k = audio_24k.copy()

    # 2. Apply G.711 telephone line bandpass filter (300Hz - 3400Hz)
    if apply_filter:
        audio_8k = butter_bandpass_filter(audio_8k, lowcut=300.0, highcut=3400.0, fs=target_sr, order=4)

    # 3. Peak normalization to -1.0 dBFS (0.891)
    peak = np.max(np.abs(audio_8k))
    if peak > 0:
        audio_8k = audio_8k * (0.891 / peak)

    return audio_8k

def to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format='WAV', subtype='PCM_16')
    return buf.getvalue()
