"""
Telephony Audio Processor: Resampling, Codec Encoders (PCM16, μ-law, a-law, WAV),
Framing (20ms/50ms), RMS Volume Normalization, and Organic Breath Injection.
"""

import io
import os
import struct
import wave
import numpy as np
from typing import Generator, Tuple, Optional, Union

try:
    import scipy.signal as signal
except ImportError:
    signal = None

try:
    import soundfile as sf
except ImportError:
    sf = None


class AudioProcessor:
    """
    High-performance audio processing pipeline designed for telephony real-time streaming,
    Vicidial / Asterisk PBX, and AI Voice Bots.
    """

    SUPPORTED_SAMPLE_RATES = [8000, 16000, 24000, 48000]
    SUPPORTED_FORMATS = ["pcm16", "mulaw", "alaw", "wav", "raw"]

    def __init__(self, default_sample_rate: int = 8000, breath_audio_path: Optional[str] = None):
        self.default_sample_rate = default_sample_rate
        self.breath_audio_cache = {}  # {sample_rate: np.ndarray (float32)}

        if breath_audio_path and os.path.exists(breath_audio_path):
            self._load_breath_audio(breath_audio_path)
        else:
            # Check default path relative to this file
            default_breath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompt", "breath_organic.wav")
            if os.path.exists(default_breath):
                self._load_breath_audio(default_breath)

    def _load_breath_audio(self, path: str):
        """Loads and caches reference breath audio."""
        try:
            if sf is not None:
                audio, sr = sf.read(path, dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                self.breath_audio_cache[sr] = audio
                # Pre-resample to standard telephony rates
                for target_sr in [8000, 16000, 24000]:
                    if target_sr != sr:
                        self.breath_audio_cache[target_sr] = self.resample(audio, sr, target_sr)
        except Exception as e:
            print(f"Warning: Could not preload breath audio from {path}: {e}")

    @staticmethod
    def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Ultra-fast polyphase / linear / FFT resampling from orig_sr to target_sr.
        Handles float32 numpy arrays [-1.0, 1.0].
        """
        if orig_sr == target_sr:
            return audio.astype(np.float32, copy=False)
        
        if len(audio) == 0:
            return np.array([], dtype=np.float32)

        # Use scipy.signal.resample_poly if available for optimal speed and anti-aliasing
        if signal is not None:
            gcd = np.gcd(orig_sr, target_sr)
            up = target_sr // gcd
            down = orig_sr // gcd
            resampled = signal.resample_poly(audio, up, down).astype(np.float32)
            return resampled
        
        # Fallback to linear interpolation
        duration = len(audio) / orig_sr
        target_len = int(round(duration * target_sr))
        indices = np.linspace(0, len(audio) - 1, target_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

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
            # Soft clip if peaks exceed 0.98
            max_val = np.max(np.abs(normalized))
            if max_val > 0.98:
                normalized = normalized / max_val * 0.98
            return normalized.astype(np.float32)
        return audio

    def prepend_breath(self, audio: np.ndarray, sample_rate: int, breath_gain: float = 0.7) -> np.ndarray:
        """
        Prepends organic airway inhalation breath with a 15ms smooth crossfade.
        """
        breath = self.breath_audio_cache.get(sample_rate)
        if breath is None:
            # Try to resample from any available cached breath
            for cached_sr, cached_breath in self.breath_audio_cache.items():
                breath = self.resample(cached_breath, cached_sr, sample_rate)
                self.breath_audio_cache[sample_rate] = breath
                break

        if breath is None or len(breath) == 0:
            return audio

        scaled_breath = (breath * breath_gain).astype(np.float32)
        
        # Smooth crossfade
        fade_samples = min(int(0.015 * sample_rate), len(scaled_breath), len(audio))
        if fade_samples > 0:
            fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
            scaled_breath[-fade_samples:] *= fade_out

        return np.concatenate([scaled_breath, audio]).astype(np.float32)

    @staticmethod
    def float_to_pcm16(audio: np.ndarray) -> bytes:
        """
        Converts float32 audio array in [-1.0, 1.0] to signed 16-bit Little-Endian PCM bytes.
        """
        # Clamp to [-1.0, 1.0]
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
        Converts float32 audio [-1.0, 1.0] to G.711 μ-law (PCMU) 8-bit bytes (ITU-T G.711).
        Standard for North American PSTN / Vicidial / Asterisk.
        """
        # Linear PCM16 integers
        clamped = np.clip(audio, -1.0, 1.0)
        pcm = (clamped * 32767.0).astype(np.int16)

        # μ-law conversion table / bit logic
        mu = 255
        sign = np.sign(pcm)
        pcm_abs = np.abs(pcm).astype(np.float32)
        
        # Quantize according to mu-law formula: sign(x) * ln(1 + mu*|x|) / ln(1 + mu)
        mag = np.log1p(mu * (pcm_abs / 32768.0)) / np.log1p(mu)
        quantized = (mag * 127.5).astype(np.uint8)

        # Standard ITU-T bit inversion and polarity representation
        sign_bit = np.where(sign >= 0, 0x80, 0x00).astype(np.uint8)
        encoded = ~(sign_bit | np.clip(quantized, 0, 127).astype(np.uint8))
        return encoded.tobytes()

    @staticmethod
    def linear_to_alaw(audio: np.ndarray) -> bytes:
        """
        Converts float32 audio [-1.0, 1.0] to G.711 A-law (PCMA) 8-bit bytes (ITU-T G.711).
        Standard for European / International PSTN.
        """
        clamped = np.clip(audio, -1.0, 1.0)
        pcm = (clamped * 32767.0).astype(np.int16)

        A = 87.6
        sign = np.sign(pcm)
        pcm_abs = (np.abs(pcm).astype(np.float32) / 32768.0)
        pcm_safe = np.maximum(pcm_abs, 1e-12)
        
        # A-law formula
        part1 = (A * pcm_abs) / (1.0 + np.log(A))
        part2 = (1.0 + np.log(A * pcm_safe)) / (1.0 + np.log(A))
        mag = np.where(pcm_abs < (1.0 / A), part1, part2)
        quantized = (mag * 127.5).astype(np.uint8)

        sign_bit = np.where(sign >= 0, 0x80, 0x00).astype(np.uint8)
        # Even bits inverted in standard A-law
        alaw_val = sign_bit | np.clip(quantized, 0, 127).astype(np.uint8)
        encoded = alaw_val ^ 0x55
        return encoded.tobytes()

    @staticmethod
    def create_wav_header(sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16, data_size: int = 0) -> bytes:
        """
        Creates a 44-byte RIFF WAV header for standard PCM audio.
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
            16,              # Subchunk1Size (16 for PCM)
            1,               # AudioFormat (1 for PCM)
            num_channels,    # NumChannels
            sample_rate,     # SampleRate
            byte_rate,       # ByteRate
            block_align,     # BlockAlign
            bits_per_sample, # BitsPerSample
            b"data",         # Subchunk2ID
            data_size        # Subchunk2Size
        )

    def to_wav_bytes(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """
        Encodes float32 audio into standard WAV container bytes.
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
        Splits raw audio bytes into fixed-size frames (e.g. 20ms frames for telephony transmission).
        
        Examples:
            - 8000 Hz, 20ms, 16-bit PCM (2 bytes/sample) -> 160 samples -> 320 bytes per frame.
            - 8000 Hz, 20ms, 8-bit μ-law (1 byte/sample)   -> 160 samples -> 160 bytes per frame.
            - 16000 Hz, 20ms, 16-bit PCM (2 bytes/sample) -> 320 samples -> 640 bytes per frame.
        """
        samples_per_frame = int(sample_rate * (frame_duration_ms / 1000.0))
        frame_size = samples_per_frame * bytes_per_sample

        for i in range(0, len(audio_bytes), frame_size):
            chunk = audio_bytes[i:i + frame_size]
            if len(chunk) < frame_size:
                # Zero pad the final frame to maintain precise 20ms alignment
                chunk = chunk + b"\x00" * (frame_size - len(chunk))
            yield chunk

    def process_and_encode(
        self,
        audio_24k: np.ndarray,
        orig_sr: int = 24000,
        target_sr: int = 8000,
        format_type: str = "pcm16",
        has_breath: bool = False,
        normalize_volume: bool = True,
        target_rms: float = 0.05
    ) -> Tuple[bytes, float]:
        """
        Full processing pipeline:
        1. Volume normalization
        2. Resampling to target_sr
        3. Optional organic breath injection
        4. Encoding into requested format (pcm16, mulaw, alaw, wav)
        
        Returns: (encoded_bytes, duration_seconds)
        """
        audio = audio_24k
        if normalize_volume:
            audio = self.normalize_rms(audio, target_rms=target_rms)

        if orig_sr != target_sr:
            audio = self.resample(audio, orig_sr, target_sr)

        if has_breath:
            audio = self.prepend_breath(audio, target_sr)

        duration_sec = float(len(audio) / target_sr)

        fmt = format_type.lower()
        if fmt == "pcm16" or fmt == "raw" or fmt == "pcm":
            return self.float_to_pcm16(audio), duration_sec
        elif fmt == "mulaw" or fmt == "pcmu" or fmt == "ulaw":
            return self.linear_to_mulaw(audio), duration_sec
        elif fmt == "alaw" or fmt == "pcma":
            return self.linear_to_alaw(audio), duration_sec
        elif fmt == "wav":
            return self.to_wav_bytes(audio, target_sr), duration_sec
        else:
            # Fallback to wav
            return self.to_wav_bytes(audio, target_sr), duration_sec
