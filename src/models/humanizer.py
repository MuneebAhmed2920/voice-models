"""Humanizer Audio Engine for Arabic Speech Synthesis.

Adds:
1. Natural human breathing gestures (subtle inhalations, micro-breaths, exhalations).
2. Human non-verbal acoustic tokens (sighs, reflective pauses).
3. Organic inter-sentence pacing and phrase cadences.
4. Studio warmth and proximity effect, eliminating sterile digital silences.
"""

import numpy as np
import scipy.signal as signal
from typing import List, Optional, Tuple


class VoiceHumanizer:
    """Provides acoustic humanization and respiratory gesture modeling for TTS speech."""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate

    def generate_breath(
        self,
        duration: float = 0.32,
        intensity: float = 0.032,
        breath_type: str = "inhale"
    ) -> np.ndarray:
        """
        Synthesizes a realistic human respiratory breath gesture.
        
        Args:
            duration: Duration in seconds (0.18s to 0.45s)
            intensity: Amplitude gain (0.015 to 0.05, subtle and organic)
            breath_type: 'inhale' (pre-speech intake), 'micro_catch' (quick mid-sentence),
                         or 'exhale' (post-speech release)
        """
        num_samples = int(self.sample_rate * duration)
        if num_samples <= 0:
            return np.zeros(0, dtype=np.float32)

        # Base turbulence noise
        noise = np.random.randn(num_samples).astype(np.float32)

        # Dual-band pharyngeal & tracheal resonance filter (1000 Hz to 3600 Hz)
        sos = signal.butter(3, [900, 3600], btype="bandpass", fs=self.sample_rate, output="sos")
        filtered = signal.sosfilt(sos, noise)

        t = np.linspace(0, 1, num_samples, endpoint=False)

        if breath_type == "inhale":
            # Natural diaphragmatic intake: gradual rise, peaking near 75%, soft cutoff before speech
            envelope = (t ** 1.6) * np.exp(-1.5 * (t - 0.78) ** 2)
        elif breath_type == "micro_catch":
            # Quick 0.18s catch of breath between rapid clauses
            envelope = np.sin(np.pi * t) ** 1.5
        else:  # exhale / release
            # Soft exponential decay of breath release
            envelope = np.exp(-3.5 * t) * (1 - np.exp(-8 * t))

        peak = np.max(np.abs(envelope)) + 1e-8
        envelope = envelope / peak

        breath = filtered * envelope * intensity
        return breath.astype(np.float32)

    def apply_studio_warmth(self, audio: np.ndarray, db_gain: float = 1.0) -> np.ndarray:
        """
        Enhances vocal proximity warmth (chest resonance ~140Hz) and smooth presence (~3.5kHz)
        to make synthetic speech feel intimately recorded in a high-end broadcast studio.
        """
        if len(audio) == 0:
            return audio

        # Warm chest resonance low-shelf filter (~140 Hz)
        b_low, a_low = signal.butter(1, 180 / (self.sample_rate / 2), btype="low")
        low_band = signal.lfilter(b_low, a_low, audio)

        # Gentle broadcast presence boost (~3200 Hz)
        sos_mid = signal.butter(2, [2800, 4200], btype="bandpass", fs=self.sample_rate, output="sos")
        mid_band = signal.sosfilt(sos_mid, audio)

        warmth = audio + 0.18 * low_band + 0.12 * mid_band
        # Normalize to prevent clipping while maintaining dynamic range
        peak = np.max(np.abs(warmth))
        if peak > 0.95:
            warmth = warmth * (0.95 / peak)

        return warmth.astype(np.float32)

    def assemble_human_speech(
        self,
        sentence_chunks: List[np.ndarray],
        breaths_before: Optional[List[bool]] = None,
        pause_durations: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Assembles speech chunks with organic breathing gestures, reflective pauses,
        and studio room warmth so there are zero sterile digital dropouts.
        """
        if not sentence_chunks:
            return np.zeros(0, dtype=np.float32)

        n = len(sentence_chunks)
        if breaths_before is None:
            # Default: subtle breath before opening, and before every 2nd sentence
            breaths_before = [True] + [True if i % 2 == 0 else False for i in range(1, n)]

        if pause_durations is None:
            pause_durations = [1.1] * (n - 1)

        assembled_parts = []

        # 1. Subtle opening ambient tone + pre-speech breath
        opening_room = np.random.randn(int(self.sample_rate * 0.4)).astype(np.float32) * 0.0004
        assembled_parts.append(opening_room)

        opening_breath = self.generate_breath(duration=0.36, intensity=0.035, breath_type="inhale")
        assembled_parts.append(opening_breath)

        for i, chunk in enumerate(sentence_chunks):
            # Apply smooth 15ms onset/offset window to speech chunk to eliminate edge clicks
            window_len = int(self.sample_rate * 0.015)
            if len(chunk) > 2 * window_len:
                chunk = chunk.copy()
                chunk[:window_len] *= np.linspace(0, 1, window_len)
                chunk[-window_len:] *= np.linspace(1, 0, window_len)

            assembled_parts.append(chunk)

            # Insert pause + breath between sentences
            if i < n - 1:
                p_dur = pause_durations[i] if i < len(pause_durations) else 1.0
                need_breath = breaths_before[i + 1] if (i + 1) < len(breaths_before) else False

                if need_breath:
                    # Pause is divided into pre-pause silence, breath intake, and micro-hold
                    breath_dur = 0.32
                    silence_dur = max(0.2, p_dur - breath_dur)
                    pre_silence = np.random.randn(int(self.sample_rate * silence_dur)).astype(np.float32) * 0.0004
                    breath = self.generate_breath(duration=breath_dur, intensity=0.032, breath_type="inhale")
                    assembled_parts.append(pre_silence)
                    assembled_parts.append(breath)
                else:
                    # Soft pause with ambient studio warmth
                    pause_audio = np.random.randn(int(self.sample_rate * p_dur)).astype(np.float32) * 0.0004
                    assembled_parts.append(pause_audio)

        # 2. Concluding breath release / room decay
        final_exhale = self.generate_breath(duration=0.45, intensity=0.022, breath_type="exhale")
        closing_room = np.random.randn(int(self.sample_rate * 0.6)).astype(np.float32) * 0.0003
        assembled_parts.append(final_exhale)
        assembled_parts.append(closing_room)

        full_speech = np.concatenate(assembled_parts)
        full_speech = self.apply_studio_warmth(full_speech)
        return full_speech
