"""Telephony audio processor for real-time Arabic TTS.

Handles:
  - LUFS/RMS loudness normalization to telephony target (-18 dBFS)
  - Anti-aliasing low-pass filter before downsampling
  - Resampling: 24 kHz → 8 kHz (G.711 narrowband) or 16 kHz (G.722 wideband)
  - G.711 μ-law codec simulation for quality verification
  - Clipping prevention and output safety limiter
  - DC offset removal

All functions operate on float32 numpy arrays and return (audio, sample_rate).
"""

import logging
from typing import Tuple

import numpy as np
import scipy.signal as signal

logger = logging.getLogger("TelephonyAudio")


# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────
TELEPHONY_TARGET_LUFS = -18.0    # Target loudness for telephony
TELEPHONY_PEAK_LIMIT = -1.0      # Peak ceiling in dBFS (prevents digital clipping)
G711_MU = 255                     # G.711 μ-law compression constant


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def linear_to_db(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-9))


class TelephonyAudioProcessor:
    """
    Prepares TTS audio for telephony delivery.

    Workflow:
        raw_audio (24kHz float32)
        → remove_dc_offset()
        → normalize_loudness()
        → resample_for_telephony()
        → verify_telephony_quality()   (optional, for testing)
        → apply_codec_simulation()     (optional, for verification)
    """

    def __init__(self, source_sample_rate: int = 24000):
        self.source_sr = source_sample_rate

    # ──────────────────────────────────────────────────────────────
    # DC offset removal
    # ──────────────────────────────────────────────────────────────
    def remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """Removes any DC bias that can cause distortion after resampling."""
        if len(audio) == 0:
            return audio
        return (audio - np.mean(audio)).astype(np.float32)

    # ──────────────────────────────────────────────────────────────
    # Loudness normalization
    # ──────────────────────────────────────────────────────────────
    def normalize_loudness(
        self,
        audio: np.ndarray,
        target_rms_db: float = TELEPHONY_TARGET_LUFS,
        peak_limit_db: float = TELEPHONY_PEAK_LIMIT,
    ) -> np.ndarray:
        """
        Normalizes audio to a consistent RMS loudness level suitable
        for telephony transmission.

        Args:
            audio: float32 audio array
            target_rms_db: Target RMS in dBFS (default: -18 dBFS for telephony)
            peak_limit_db: Hard peak limit to prevent clipping (default: -1 dBFS)

        Returns:
            Loudness-normalized float32 array
        """
        if len(audio) == 0:
            return audio

        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-8:
            return audio

        current_db = linear_to_db(rms)
        gain_db = target_rms_db - current_db
        gain = db_to_linear(gain_db)
        normalized = audio * gain

        # Hard peak limiter
        peak_limit_linear = db_to_linear(peak_limit_db)
        peak = np.max(np.abs(normalized))
        if peak > peak_limit_linear:
            normalized = normalized * (peak_limit_linear / peak)
            logger.debug(f"[TelephonyAudio] Peak limiter applied: {linear_to_db(peak):.1f} dBFS → {peak_limit_db:.1f} dBFS")

        return normalized.astype(np.float32)

    # ──────────────────────────────────────────────────────────────
    # Anti-aliased resampling
    # ──────────────────────────────────────────────────────────────
    def resample_for_telephony(
        self,
        audio: np.ndarray,
        target_sr: int = 8000,
    ) -> Tuple[np.ndarray, int]:
        """
        Resamples audio from source_sr to target_sr with anti-aliasing.

        Applies a Butterworth low-pass filter at (target_sr / 2 * 0.9) Hz
        before downsampling to prevent aliasing artifacts.

        Args:
            audio: float32 audio at self.source_sr
            target_sr: Target sample rate (8000 for G.711, 16000 for G.722)

        Returns:
            Tuple of (resampled_audio, target_sr)
        """
        if len(audio) == 0:
            return audio, target_sr
        if target_sr == self.source_sr:
            return audio, target_sr
        if target_sr > self.source_sr:
            # Upsampling — just use scipy resample
            n_target = int(len(audio) * target_sr / self.source_sr)
            resampled = signal.resample(audio, n_target)
            return resampled.astype(np.float32), target_sr

        # Downsampling — apply anti-alias LPF first
        nyquist_target = target_sr / 2.0
        # Set cutoff at 90% of target Nyquist to prevent aliasing while keeping warmth
        cutoff = (nyquist_target * 0.90) / (self.source_sr / 2.0)
        cutoff = min(cutoff, 0.99)

        # 8th-order Butterworth low-pass for clean rolloff
        sos = signal.butter(8, cutoff, btype="low", output="sos")
        filtered = signal.sosfilt(sos, audio)

        # Resample
        n_target = int(len(audio) * target_sr / self.source_sr)
        resampled = signal.resample(filtered, n_target)

        logger.debug(
            f"[TelephonyAudio] Resampled {self.source_sr}Hz → {target_sr}Hz "
            f"(cutoff: {nyquist_target * 0.90:.0f} Hz, samples: {len(audio)} → {n_target})"
        )
        return resampled.astype(np.float32), target_sr

    # ──────────────────────────────────────────────────────────────
    # Telephony quality verification
    # ──────────────────────────────────────────────────────────────
    def verify_telephony_quality(self, audio: np.ndarray, sr: int) -> dict:
        """
        Runs quality checks on telephony audio and returns a report dict.

        Checks:
          - Peak level (must not clip)
          - RMS level (must be above silence threshold)
          - DC offset (should be near zero)
          - High-frequency energy ratio (for metallic/aliasing detection)

        Args:
            audio: Audio at telephony sample rate
            sr: Sample rate of the audio

        Returns:
            Dict with 'pass' (bool) and per-check details.
        """
        if len(audio) == 0:
            return {"pass": False, "reason": "empty audio"}

        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio ** 2)))
        dc_offset = float(np.mean(audio))

        # Frequency analysis: check for metallic high-freq ringing
        # Compare energy above 80% of Nyquist vs total
        nyq = sr / 2
        hf_cutoff = 0.80  # normalized
        try:
            sos_hf = signal.butter(4, hf_cutoff, btype="high", output="sos")
            hf_signal = signal.sosfilt(sos_hf, audio)
            hf_energy_ratio = float(np.sqrt(np.mean(hf_signal ** 2)) / (rms + 1e-8))
        except Exception:
            hf_energy_ratio = 0.0

        results = {
            "sample_rate": sr,
            "peak_dbfs": round(linear_to_db(peak), 2),
            "rms_dbfs": round(linear_to_db(rms), 2),
            "dc_offset": round(dc_offset, 6),
            "hf_energy_ratio": round(hf_energy_ratio, 4),
            "duration_sec": round(len(audio) / sr, 3),
            "clipping": peak > 1.0,
            "silence": rms < 0.001,
            "hf_ringing": hf_energy_ratio > 0.35,  # Possible aliasing if > 35%
            "dc_issue": abs(dc_offset) > 0.02,
        }

        issues = [k for k in ("clipping", "silence", "hf_ringing", "dc_issue") if results[k]]
        results["pass"] = len(issues) == 0
        results["issues"] = issues

        return results

    # ──────────────────────────────────────────────────────────────
    # G.711 μ-law codec simulation
    # ──────────────────────────────────────────────────────────────
    def apply_g711_codec_simulation(self, audio: np.ndarray) -> np.ndarray:
        """
        Simulates G.711 μ-law encode → decode cycle to verify the audio
        sounds clean after telephone codec compression.

        This is for testing only — do not use this in the actual streaming path.

        Args:
            audio: float32 audio (normalized to [-1.0, 1.0])

        Returns:
            Audio after simulated G.711 encode/decode (float32)
        """
        mu = float(G711_MU)
        # μ-law encode
        audio_clipped = np.clip(audio, -1.0, 1.0)
        encoded = np.sign(audio_clipped) * np.log1p(mu * np.abs(audio_clipped)) / np.log1p(mu)
        # Quantize to 8-bit (256 levels)
        quantized = np.round(encoded * 127.0) / 127.0
        # μ-law decode
        decoded = np.sign(quantized) * (np.exp(np.abs(quantized) * np.log1p(mu)) - 1.0) / mu
        return decoded.astype(np.float32)

    # ──────────────────────────────────────────────────────────────
    # Full telephony pipeline
    # ──────────────────────────────────────────────────────────────
    def prepare_for_telephony(
        self,
        audio: np.ndarray,
        target_sr: int = 8000,
        simulate_codec: bool = False,
    ) -> Tuple[np.ndarray, int, dict]:
        """
        Complete telephony preparation pipeline.

        Steps:
          1. Remove DC offset
          2. Normalize loudness to -18 dBFS
          3. Anti-aliased resample to target_sr
          4. Optional: G.711 codec simulation
          5. Quality verification

        Args:
            audio: Raw TTS output (float32, self.source_sr)
            target_sr: 8000 (G.711) or 16000 (G.722)
            simulate_codec: If True, runs G.711 simulation for testing

        Returns:
            (prepared_audio, target_sr, quality_report)
        """
        audio = self.remove_dc_offset(audio)
        audio = self.normalize_loudness(audio)
        audio, sr = self.resample_for_telephony(audio, target_sr=target_sr)

        if simulate_codec and target_sr == 8000:
            audio = self.apply_g711_codec_simulation(audio)

        quality = self.verify_telephony_quality(audio, sr)

        if not quality["pass"]:
            logger.warning(f"[TelephonyAudio] Quality issues: {quality['issues']}")

        return audio, sr, quality
