"""
Arabic Physiological Breath Processor (Pillars 1 & 6)
Implements natural airway inhalation before turn-initial speech and conjunction-based respiratory pauses.
"""

import os
import re
from typing import Tuple, Optional, Dict, Any
import numpy as np
import scipy.signal
import soundfile as sf


class ArabicBreathProcessor:
    """
    Manages emotion-tailored physiological breathing audio, respiratory envelopes,
    pre-speech airway gaps, and tag extraction for conversational Arabic speech agents.
    """

    EMOTION_PARAMS = {
        "neutral": {
            "duration_ms": 140,
            "gain": 0.032,       # -28 dBFS
            "pre_gap_ms": 50,
            "audio_file": "breath_organic.wav"
        },
        "excited": {
            "duration_ms": 95,
            "gain": 0.024,       # -30 dBFS
            "pre_gap_ms": 25,
            "audio_file": "breath_excited.wav"
        },
        "sympathetic": {
            "duration_ms": 210,
            "gain": 0.045,       # -26 dBFS
            "pre_gap_ms": 75,
            "audio_file": "breath_sympathetic.wav"
        },
        "emotional": {
            "duration_ms": 165,
            "gain": 0.038,       # -27 dBFS
            "pre_gap_ms": 60,
            "audio_file": "breath_emotional.wav"
        }
    }

    # Arabic coordinating conjunctions that trigger natural micro-respiration in long clauses
    CONJUNCTION_PATTERN = re.compile(r"(\s+(?:ولكن|لكن|و|فـ|ف|يعني|طبعاً|بالتالي)\s+)", re.UNICODE)

    def __init__(self, prompt_dir: Optional[str] = None, sample_rate: int = 24000):
        self.sample_rate = sample_rate

        if prompt_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.prompt_dir = os.path.join(base, "prompt")
        else:
            self.prompt_dir = prompt_dir

        self.breath_audio_cache: Dict[str, np.ndarray] = {}
        self._load_all_breath_assets()

    def _load_all_breath_assets(self):
        """Loads and caches all emotion breath audio assets with resampling."""
        for emo, cfg in self.EMOTION_PARAMS.items():
            fname = cfg["audio_file"]
            path = os.path.join(self.prompt_dir, fname)

            # Fallback paths
            if not os.path.exists(path):
                alt_paths = [
                    os.path.join(self.prompt_dir, "breath_organic.wav"),
                    f"/Users/haseebahmed/agent/arabic_breath_assets/{fname}",
                    "/Users/haseebahmed/agent/arabic_breath_assets/breath_organic.wav",
                    "/Users/haseebahmed/agent/clara/clara-voice-model/prompt/breath_organic.wav"
                ]
                for alt in alt_paths:
                    if os.path.exists(alt):
                        path = alt
                        break

            if os.path.exists(path):
                try:
                    data, sr = sf.read(path, dtype="float32")
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    if sr != self.sample_rate:
                        target_len = int(len(data) * self.sample_rate / sr)
                        data = scipy.signal.resample(data, target_len).astype(np.float32)
                    
                    # Normalize peak to 1.0
                    peak = np.max(np.abs(data))
                    if peak > 1e-6:
                        data = data / peak
                    self.breath_audio_cache[emo] = data
                except Exception as e:
                    print(f"Warning: Failed loading breath file {path}: {e}")

        # Ensure base organic breath is loaded
        if "neutral" not in self.breath_audio_cache:
            # Synthesize gentle pink-noise filtered inhalation fallback
            num_samples = int(0.15 * self.sample_rate)
            noise = np.random.randn(num_samples).astype(np.float32)
            # Low pass filter at 1200 Hz
            b, a = scipy.signal.butter(2, 1200 / (self.sample_rate / 2), btype="low")
            filtered = scipy.signal.lfilter(b, a, noise).astype(np.float32)
            self.breath_audio_cache["neutral"] = filtered / (np.max(np.abs(filtered)) + 1e-8)

    def parse_arabic_tags(self, text: str) -> Tuple[str, str, bool]:
        """
        Extracts emotion and breath tags while returning clean Arabic text.
        
        Supported tags:
            - Emotion: [متحمس], [excited], [happy], [متعاطف], [sympathetic], [apologetic],
                       [مؤثر], [emotional], [sincere], [طبيعي], [neutral]
            - Respiration: [تنفس], [breath], [inhalation], [no_breath], [بدون_تنفس]
        """
        emotion = "neutral"
        include_breath = True

        # Check for explicit disable breath
        if re.search(r"\[(no_breath|بدون_تنفس|بلا_تنفس)\]", text, re.IGNORECASE):
            include_breath = False

        if re.search(r"\[(متحمس|excited|happy|فرحان)\]", text, re.IGNORECASE):
            emotion = "excited"
        elif re.search(r"\[(متعاطف|sympathetic|apologetic|حزين|اعتذار)\]", text, re.IGNORECASE):
            emotion = "sympathetic"
        elif re.search(r"\[(مؤثر|emotional|sincere|صادق|دافئ)\]", text, re.IGNORECASE):
            emotion = "emotional"
        elif re.search(r"\[(طبيعي|neutral|هادئ)\]", text, re.IGNORECASE):
            emotion = "neutral"

        if re.search(r"\[(تنفس|breath|inhalation)\]", text, re.IGNORECASE):
            include_breath = True

        # Strip tags from spoken text
        clean_text = re.sub(
            r"\[(متحمس|متعاطف|مؤثر|تنفس|طبيعي|بدون_تنفس|بلا_تنفس|excited|sympathetic|emotional|breath|no_breath|pause)\]",
            "",
            text,
            flags=re.IGNORECASE
        )
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        return clean_text, emotion, include_breath

    def create_tailored_breath(self, emotion: str = "neutral", custom_duration_ms: Optional[float] = None) -> np.ndarray:
        """
        Synthesizes / sculpts an emotion-tailored airway breath with raised-cosine
        fade-in, fade-out, and pre-speech gap.
        """
        cfg = self.EMOTION_PARAMS.get(emotion, self.EMOTION_PARAMS["neutral"])
        dur_ms = custom_duration_ms if custom_duration_ms is not None else cfg["duration_ms"]
        gain = cfg["gain"]
        pre_gap_ms = cfg["pre_gap_ms"]

        raw = self.breath_audio_cache.get(emotion, self.breath_audio_cache.get("neutral"))
        target_len = max(int((dur_ms / 1000.0) * self.sample_rate), 10)

        # Resample to precise duration
        breath = scipy.signal.resample(raw, target_len).astype(np.float32) * gain

        # Smooth raised-cosine windowing: 15ms fade-in, 25ms fade-out
        fade_in = min(int(0.015 * self.sample_rate), len(breath) // 3)
        fade_out = min(int(0.025 * self.sample_rate), len(breath) // 3)
        if fade_in > 0:
            breath[:fade_in] *= 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_in)))
        if fade_out > 0:
            breath[-fade_out:] *= 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_out)))

        # Silent pre-speech airway gap
        gap_len = int((pre_gap_ms / 1000.0) * self.sample_rate)
        gap = np.zeros(gap_len, dtype=np.float32)

        return np.concatenate([breath, gap]).astype(np.float32)

    def inject_pre_speech_breath(
        self,
        speech_waveform: np.ndarray,
        emotion: str = "neutral",
        include_breath: bool = True
    ) -> np.ndarray:
        """
        Prepends physiological airway breath to speech waveform with click-free crossfade.
        """
        if not include_breath or len(speech_waveform) == 0:
            return speech_waveform

        breath_audio = self.create_tailored_breath(emotion)

        # Apply a subtle 10ms raised-cosine fade-in to speech onset to eliminate any edge discontinuity
        speech = speech_waveform.copy()
        speech_fade = min(int(0.010 * self.sample_rate), len(speech))
        if speech_fade > 0:
            fade_curve = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, speech_fade)))
            speech[:speech_fade] *= fade_curve

        return np.concatenate([breath_audio, speech]).astype(np.float32)

    def inject_conjunction_micro_breaths(self, text: str, min_clause_words: int = 7) -> str:
        """
        Detects long clauses connected by coordinating conjunctions (و, فـ, لكن, يعني, طبعاً)
        and inserts pause/respiratory boundaries for the streamer.
        """
        words = text.split()
        if len(words) < min_clause_words:
            return text

        # Replace conjunction occurrences with natural pause marks if clause is substantial
        parts = self.CONJUNCTION_PATTERN.split(text)
        reconstructed = []
        clause_len = 0

        for part in parts:
            p_words = len(part.strip().split())
            if self.CONJUNCTION_PATTERN.match(part):
                if clause_len >= min_clause_words:
                    reconstructed.append(f"، {part.strip()} ")
                    clause_len = 0
                else:
                    reconstructed.append(part)
            else:
                reconstructed.append(part)
                clause_len += p_words

        return "".join(reconstructed)
