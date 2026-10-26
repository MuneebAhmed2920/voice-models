"""
Arabic Telephony Audio Processor & Mastering Chain (Pillar 5)
Implements proximity EQ (75Hz HPF, 280Hz low-shelf warmth, 4kHz consonant presence),
soft-knee limiting, resampling, codecs (PCM16, μ-law, A-law, WAV), and 20ms framing.
"""

import io
import struct
import numpy as np
import scipy.signal as signal
from typing import Generator, Tuple, Optional, Union


class ArabicAudioProcessor:
    """
    Studio-grade mastering and telephony audio pipeline for Arabic conversational voice agents.
    Provides proximity EQ coloration, consonant enunciation boost, soft-knee limiting,
    and telephony codec encoders (PCM16, G.711 μ-law, G.711 A-law, WAV).
    """

    SUPPORTED_SAMPLE_RATES = [8000, 16000, 24000, 48000]
    SUPPORTED_FORMATS = ["pcm16", "mulaw", "alaw", "wav", "raw"]

    def __init__(
        self,
        default_sample_rate: int = 8000,
        apply_mastering_eq: bool = True,
        target_rms: float = 0.05
    ):
        self.default_sample_rate = default_sample_rate
        self.apply_mastering_eq = apply_mastering_eq
        self.target_rms = target_rms

    @staticmethod
    def apply_high_pass_filter(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 75.0) -> np.ndarray:
        """
        2nd-order Butterworth High-Pass Filter (12 dB/octave) to eliminate DC offset and sub-audible rumble.
        """
        if len(audio) == 0 or sample_rate <= cutoff_hz * 2:
            return audio
        nyquist = 0.5 * sample_rate
        norm_cutoff = min(cutoff_hz / nyquist, 0.99)
        b, a = signal.butter(2, norm_cutoff, btype="highpass")
        return signal.filtfilt(b, a, audio).astype(np.float32)

    @staticmethod
    def apply_warmth_low_shelf(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 280.0, gain_db: float = 1.5) -> np.ndarray:
        """
        Simulates close-mic proximity effect by applying a gentle low-shelf warmth boost.
        """
        if len(audio) == 0 or sample_rate <= cutoff_hz * 2 or abs(gain_db) < 0.01:
            return audio

        # 2nd-order low-shelf biquad filter design
        nyquist = 0.5 * sample_rate
        w0 = 2.0 * np.pi * (cutoff_hz / sample_rate)
        A = 10.0 ** (gain_db / 40.0)
        alpha = np.sin(w0) / 2.0 * np.sqrt(2.0)
        cos_w0 = np.cos(w0)

        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha

        b = np.array([b0, b1, b2], dtype=np.float64) / a0
        a = np.array([a0, a1, a2], dtype=np.float64) / a0

        return signal.lfilter(b, a, audio).astype(np.float32)

    @staticmethod
    def apply_consonant_presence(
        audio: np.ndarray,
        sample_rate: int,
        center_hz: float = 4000.0,
        gain_db: float = 2.0,
        q: float = 1.2
    ) -> np.ndarray:
        """
        Peaking parametric EQ filter to enhance Arabic pharyngeal and emphatic consonants (ح, ع, ص, ض, ط, ظ, ق).
        """
        if len(audio) == 0 or sample_rate <= center_hz * 2.1 or abs(gain_db) < 0.01:
            return audio

        w0 = 2.0 * np.pi * (center_hz / sample_rate)
        A = 10.0 ** (gain_db / 40.0)
        alpha = np.sin(w0) / (2.0 * q)
        cos_w0 = np.cos(w0)

        b0 = 1.0 + alpha * A
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / A

        b = np.array([b0, b1, b2], dtype=np.float64) / a0
        a = np.array([a0, a1, a2], dtype=np.float64) / a0

        return signal.lfilter(b, a, audio).astype(np.float32)

    @staticmethod
    def apply_soft_knee_limiting(audio: np.ndarray, threshold: float = 0.85, ceiling: float = 0.95) -> np.ndarray:
        """
        Transparent soft-knee peak limiter to prevent digital clipping on telephone lines.
        """
        if len(audio) == 0:
            return audio

        peak = np.max(np.abs(audio))
        if peak <= threshold:
            return audio

        # Tanh soft-saturation above threshold
        x = audio.copy()
        mask = np.abs(x) > threshold
        excess = np.abs(x[mask]) - threshold
        headroom = ceiling - threshold
        compressed_excess = headroom * np.tanh(excess / max(1e-5, headroom))
        x[mask] = np.sign(x[mask]) * (threshold + compressed_excess)

        # Final safety ceiling clamp
        return np.clip(x, -ceiling, ceiling).astype(np.float32)

    def apply_mastering_chain(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Executes full Pillar 5 studio mastering chain:
        1. 75Hz High-Pass Filter (DC rumble cleanup)
        2. 280Hz Low-Shelf Warmth (+1.5 dB proximity effect)
        3. 4.0kHz Consonant Presence Peaking (+2.0 dB Arabic clarity)
        4. Soft-Knee Peak Limiting
        """
        if not self.apply_mastering_eq or len(audio) == 0:
            return audio

        out = self.apply_high_pass_filter(audio, sample_rate, cutoff_hz=75.0)
        out = self.apply_warmth_low_shelf(out, sample_rate, cutoff_hz=280.0, gain_db=1.5)
        if sample_rate >= 12000:
            out = self.apply_consonant_presence(out, sample_rate, center_hz=4000.0, gain_db=2.0, q=1.2)
        out = self.apply_soft_knee_limiting(out, threshold=0.85, ceiling=0.95)
        return out

    @staticmethod
    def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Polyphase resampling from orig_sr to target_sr.
        """
        if orig_sr == target_sr:
            return audio.astype(np.float32, copy=False)
        if len(audio) == 0:
            return np.array([], dtype=np.float32)

        gcd = np.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        return signal.resample_poly(audio, up, down).astype(np.float32)

    @staticmethod
    def normalize_rms(audio: np.ndarray, target_rms: float = 0.05, max_gain: float = 4.0) -> np.ndarray:
        """
        Normalizes audio volume to target RMS level with clipping prevention.
        """
        if len(audio) == 0:
            return audio

        current_rms = np.sqrt(np.mean(np.square(audio)))
        if current_rms > 1e-5:
            gain = min(target_rms / current_rms, max_gain)
            normalized = audio * gain
            max_val = np.max(np.abs(normalized))
            if max_val > 0.95:
                normalized = normalized / max_val * 0.95
            return normalized.astype(np.float32)
        return audio

    @staticmethod
    def float_to_pcm16(audio: np.ndarray) -> bytes:
        """
        Converts float32 audio [-1.0, 1.0] to signed 16-bit Little-Endian PCM bytes.
        """
        clamped = np.clip(audio, -1.0, 1.0)
        pcm16_data = (clamped * 32767.0).astype(np.int16)
        return pcm16_data.tobytes()

    @staticmethod
    def pcm16_to_float(pcm_bytes: bytes) -> np.ndarray:
        """
        Converts signed 16-bit Little-Endian PCM bytes to float32 numpy array.
        """
        pcm16_data = np.frombuffer(pcm_bytes, dtype=np.int16)
        return (pcm16_data.astype(np.float32) / 32767.0)

    @staticmethod
    def linear_to_mulaw(audio: np.ndarray) -> bytes:
        """
        Converts float32 audio [-1.0, 1.0] to G.711 μ-law 8-bit bytes (ITU-T G.711).
        """
        clamped = np.clip(audio, -1.0, 1.0)
        pcm = (clamped * 32767.0).astype(np.int16)

        mu = 255
        sign = np.sign(pcm)
        pcm_abs = np.abs(pcm).astype(np.float32)

        mag = np.log1p(mu * (pcm_abs / 32768.0)) / np.log1p(mu)
        quantized = (mag * 127.5).astype(np.uint8)

        sign_bit = np.where(sign >= 0, 0x80, 0x00).astype(np.uint8)
        encoded = ~(sign_bit | np.clip(quantized, 0, 127).astype(np.uint8))
        return encoded.tobytes()

    @staticmethod
    def linear_to_alaw(audio: np.ndarray) -> bytes:
        """
        Converts float32 audio [-1.0, 1.0] to G.711 A-law 8-bit bytes (ITU-T G.711).
        """
        clamped = np.clip(audio, -1.0, 1.0)
        pcm = (clamped * 32767.0).astype(np.int16)

        A = 87.6
        sign = np.sign(pcm)
        pcm_abs = (np.abs(pcm).astype(np.float32) / 32768.0)
        pcm_safe = np.maximum(pcm_abs, 1e-12)

        part1 = (A * pcm_abs) / (1.0 + np.log(A))
        part2 = (1.0 + np.log(A * pcm_safe)) / (1.0 + np.log(A))
        mag = np.where(pcm_abs < (1.0 / A), part1, part2)
        quantized = (mag * 127.5).astype(np.uint8)

        sign_bit = np.where(sign >= 0, 0x80, 0x00).astype(np.uint8)
        alaw_val = sign_bit | np.clip(quantized, 0, 127).astype(np.uint8)
        encoded = alaw_val ^ 0x55
        return encoded.tobytes()

    @staticmethod
    def create_wav_header(sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16, data_size: int = 0) -> bytes:
        """
        Creates standard 44-byte RIFF WAV header.
        """
        byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
        block_align = num_channels * (bits_per_sample // 8)
        file_size = 36 + data_size

        return struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            file_size,
            b"WAVE",
            b"fmt ",
            16,
            1,
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            data_size
        )

    def to_wav_bytes(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """
        Encodes float32 audio into complete WAV container bytes.
        """
        pcm_bytes = self.float_to_pcm16(audio)
        header = self.create_wav_header(sample_rate=sample_rate, data_size=len(pcm_bytes))
        return header + pcm_bytes

    def frame_audio(
        self,
        audio_bytes: bytes,
        sample_rate: int = 8000,
        frame_duration_ms: int = 20,
        bytes_per_sample: int = 2
    ) -> Generator[bytes, None, None]:
        """
        Generates fixed-size telephony frames (e.g. 20ms frames: 320 bytes @ 8kHz PCM16).
        """
        samples_per_frame = int(sample_rate * (frame_duration_ms / 1000.0))
        frame_size = samples_per_frame * bytes_per_sample

        for i in range(0, len(audio_bytes), frame_size):
            chunk = audio_bytes[i:i + frame_size]
            if len(chunk) < frame_size:
                chunk = chunk + b"\x00" * (frame_size - len(chunk))
            yield chunk

    def process_and_encode(
        self,
        audio_native: np.ndarray,
        orig_sr: int = 24000,
        target_sr: int = 8000,
        format_type: str = "pcm16",
        apply_eq: bool = True,
        normalize_volume: bool = True
    ) -> Tuple[bytes, float]:
        """
        Complete telephony audio pipeline:
        1. Studio Mastering EQ chain (75Hz HPF, 280Hz low-shelf, 4kHz consonant presence)
        2. RMS Volume normalization
        3. Polyphase resampling to target sample rate
        4. Codec encoding (PCM16, μ-law, A-law, WAV)
        
        Returns: (encoded_bytes, duration_seconds)
        """
        audio = audio_native

        if apply_eq:
            audio = self.apply_mastering_chain(audio, orig_sr)

        if normalize_volume:
            audio = self.normalize_rms(audio, target_rms=self.target_rms)

        if orig_sr != target_sr:
            audio = self.resample(audio, orig_sr, target_sr)

        duration_sec = float(len(audio) / max(1, target_sr))
        fmt = format_type.lower()

        if fmt in ["pcm16", "raw", "pcm"]:
            return self.float_to_pcm16(audio), duration_sec
        elif fmt in ["mulaw", "ulaw", "pcmu"]:
            return self.linear_to_mulaw(audio), duration_sec
        elif fmt in ["alaw", "pcma"]:
            return self.linear_to_alaw(audio), duration_sec
        elif fmt == "wav":
            return self.to_wav_bytes(audio, target_sr), duration_sec
        else:
            return self.float_to_pcm16(audio), duration_sec
