"""
Fast Audio Codecs & Resamplers for omnivoice_tts.
"""

import io
import numpy as np
import torch
import torchaudio
import scipy.signal as signal
import soundfile as sf
from typing import Literal

AudioFormat = Literal["wav", "pcm16", "mulaw"]

def resample_audio(audio_np: np.ndarray, orig_sr: int = 24000, target_sr: int = 24000) -> np.ndarray:
    """Fast polyphase resampler with band-limiting."""
    if orig_sr == target_sr or len(audio_np) == 0:
        return audio_np
        
    if target_sr == 8000:
        # ITU-T G.712 telephone bandpass filter
        try:
            b, a = signal.butter(4, [300.0 / (orig_sr / 2.0), 3400.0 / (orig_sr / 2.0)], btype='band')
            audio_np = signal.lfilter(b, a, audio_np)
        except Exception:
            pass
            
    t_audio = torch.from_numpy(audio_np).unsqueeze(0)
    resampled = torchaudio.functional.resample(t_audio, orig_sr, target_sr).squeeze(0).numpy()
    
    # Peak normalization with 0.95 headroom
    peak = np.max(np.abs(resampled)) if resampled.size > 0 else 0.0
    if peak > 0.98:
        resampled = resampled * (0.95 / peak)
        
    return resampled.astype(np.float32)

def encode_audio(
    audio_pcm: np.ndarray,
    sample_rate: int = 24000,
    output_format: AudioFormat = "wav"
) -> bytes:
    """Encodes float32 PCM [-1.0, 1.0] into target audio format bytes."""
    if output_format == "pcm16":
        int16_pcm = np.clip(audio_pcm * 32767.0, -32768, 32767).astype(np.int16)
        return int16_pcm.tobytes()
        
    elif output_format == "mulaw":
        mu = 255
        x = np.clip(audio_pcm, -1.0, 1.0)
        out = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
        quantized = np.round((out + 1) / 2 * 255).astype(np.uint8)
        return quantized.tobytes()
        
    else:  # WAV
        buf = io.BytesIO()
        sf.write(buf, audio_pcm, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()
