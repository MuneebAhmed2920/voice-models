"""
Unified Conversational Gesture and Non-Verbal Tag Parser for omnivoice_tts.
"""

import re
import numpy as np
import scipy.signal as signal
from typing import Tuple, Optional

TAG_MAP = {
    "<laugh>": "[laughter]",
    "<chuckle>": "[laughter]",
    "<sigh>": "[sigh]",
    "<cough>": "[cough]",
    "<whisper>": "[whisper]",
    "<gasp>": "[gasp]",
    "<throat-clear>": "[clear-throat]",
    "<hmm>": "Hmm, ",
    "<uh>": "Uh, ",
    "<um>": "Um, ",
}

EMOTION_MAP = {
    "<excited>": "excited, upbeat tone",
    "<calm>": "calm, soothing, patient tone",
    "<concerned>": "empathetic, concerned, gentle tone",
    "<friendly>": "warm, friendly, engaging tone",
}


class GestureParser:
    def __init__(self, sample_rate: int = 24000):
        self.sr = sample_rate

    def parse(self, text: str) -> Tuple[str, Optional[str]]:
        """Parses text and extracts style modifiers and native gesture tokens."""
        emotion_mod = None
        for emo_tag, emo_style in EMOTION_MAP.items():
            if emo_tag in text:
                emotion_mod = emo_style
                text = text.replace(emo_tag, "")

        def _pause_repl(m: re.Match) -> str:
            return ", "
        text = re.sub(r"<pause(?::[\d.]+)?\/?>", _pause_repl, text)
        text = re.sub(r"<hesitate\/?>", "Um, ", text)

        for tag, ov_token in TAG_MAP.items():
            text = text.replace(tag, ov_token)

        text = re.sub(r"\s+", " ", text).strip()
        return text, emotion_mod

    def generate_breath(self, duration_sec: float = 0.28, intensity: float = 0.12, is_male: bool = False) -> np.ndarray:
        """Synthesizes natural soft airway inhale breath."""
        n_samples = int(self.sr * duration_sec)
        noise = np.random.randn(n_samples).astype(np.float32)
        
        t = np.linspace(0, 1, n_samples, dtype=np.float32)
        envelope = (np.sin(t * np.pi) ** 2.0) * intensity
        
        low = 450 if is_male else 750
        high = 1700 if is_male else 2300
        try:
            b, a = signal.butter(2, [low / (self.sr / 2), high / (self.sr / 2)], btype='band')
            breath = signal.lfilter(b, a, noise)
        except Exception:
            breath = noise
            
        breath = breath / (np.max(np.abs(breath)) + 1e-6)
        return (breath * envelope).astype(np.float32)

    def crossfade(self, audio1: np.ndarray, audio2: np.ndarray, fade_sec: float = 0.04) -> np.ndarray:
        """Crossfades two audio buffers to eliminate clicks."""
        fade_len = int(self.sr * fade_sec)
        if len(audio1) < fade_len or len(audio2) < fade_len:
            return np.concatenate([audio1, audio2])
            
        t = np.linspace(0, 1, fade_len, dtype=np.float32)
        fade_out = np.cos(t * 0.5 * np.pi)
        fade_in = np.sin(t * 0.5 * np.pi)
        
        overlap = audio1[-fade_len:] * fade_out + audio2[:fade_len] * fade_in
        return np.concatenate([audio1[:-fade_len], overlap, audio2[fade_len:]])
