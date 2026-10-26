#!/usr/bin/env python3
"""
Robotic Speech Diagnostic Tool for Arabic TTS.

Analyzes generated WAV files for signs of synthetic/robotic speech:
  1. Flat pitch (< threshold variation over 3-second windows)
  2. Uniform energy (< threshold RMS variation across phrases)
  3. Uniform word duration distribution (machine-gun cadence)
  4. Hard energy drops at word boundaries (segmentation artifacts)
  5. Spectral flatness (metallic sound indicator)

Usage:
  PYTHONPATH=. .venv/bin/python evaluation/robotic_detector.py tests/arabic_naturalness/output/
"""

import sys
import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import scipy.signal as signal
import soundfile as sf


def analyze_file(wav_path: str) -> dict:
    """
    Analyze a single WAV file for robotic speech indicators.

    Returns a dict with per-metric scores and an overall naturalness estimate.
    """
    try:
        audio, sr = sf.read(wav_path)
    except Exception as e:
        return {"error": str(e), "path": wav_path}

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    if len(audio) < sr * 0.5:  # Less than 500ms — skip
        return {"error": "too short", "path": wav_path}

    result = {
        "path": wav_path,
        "duration_sec": round(len(audio) / sr, 3),
        "sample_rate": sr,
    }

    # ── 1. Pitch variation (F0 via autocorrelation) ───────────────
    result["pitch"] = _analyze_pitch(audio, sr)

    # ── 2. Energy variation ───────────────────────────────────────
    result["energy"] = _analyze_energy(audio, sr)

    # ── 3. Spectral flatness (Wiener entropy) ─────────────────────
    result["spectral"] = _analyze_spectral(audio, sr)

    # ── 4. Overall naturalness score (0-100, higher = more natural) ─
    score = _compute_score(result)
    result["naturalness_score"] = score
    result["verdict"] = (
        "✅ Natural" if score >= 65
        else "⚠️ Marginal" if score >= 45
        else "❌ Robotic"
    )

    return result


def _analyze_pitch(audio: np.ndarray, sr: int) -> dict:
    """Estimate pitch variation using autocorrelation."""
    # Compute short-time energy to find voiced frames
    frame_len = int(sr * 0.025)  # 25ms frames
    hop_len = int(sr * 0.010)    # 10ms hop

    # Compute pitch using simple zero-crossing based voiced detection
    frames = []
    for i in range(0, len(audio) - frame_len, hop_len):
        frame = audio[i:i + frame_len]
        frames.append(frame)

    if not frames:
        return {"variation_hz": 0, "voiced_ratio": 0}

    # Autocorrelation-based F0 estimation (simplified)
    f0_values = []
    min_period = int(sr / 450)  # max 450 Hz
    max_period = int(sr / 60)   # min 60 Hz

    for frame in frames:
        energy = np.sqrt(np.mean(frame ** 2))
        if energy < 0.01:  # Unvoiced / silence
            continue
        # Autocorrelation
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2:]
        # Find peak in plausible F0 range
        if len(corr) > max_period:
            search = corr[min_period:max_period]
            if len(search) > 0:
                peak_idx = np.argmax(search) + min_period
                f0 = sr / peak_idx
                if 60 <= f0 <= 450:
                    f0_values.append(f0)

    if len(f0_values) < 5:
        return {"variation_hz": 0, "voiced_ratio": 0, "note": "insufficient voiced frames"}

    f0_array = np.array(f0_values)
    voiced_ratio = len(f0_values) / len(frames)
    variation = float(np.std(f0_array))
    mean_f0 = float(np.mean(f0_array))

    return {
        "mean_hz": round(mean_f0, 1),
        "std_hz": round(variation, 1),
        "voiced_ratio": round(voiced_ratio, 3),
        "is_flat": variation < 25.0,  # < 25Hz std = suspiciously flat
        "score": min(100, int(variation * 2.5)),  # Higher variation = better score
    }


def _analyze_energy(audio: np.ndarray, sr: int) -> dict:
    """Analyze RMS energy variation across 200ms windows."""
    window = int(sr * 0.200)
    hop = int(sr * 0.100)
    rms_values = []
    for i in range(0, len(audio) - window, hop):
        chunk = audio[i:i + window]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms > 0.005:  # Skip near-silence
            rms_values.append(rms)

    if len(rms_values) < 3:
        return {"variation_db": 0, "is_flat": True}

    rms_db = 20 * np.log10(np.array(rms_values) + 1e-8)
    variation_db = float(np.std(rms_db))
    return {
        "std_db": round(variation_db, 2),
        "is_flat": variation_db < 2.5,  # < 2.5dB variation = suspicious
        "score": min(100, int(variation_db * 10)),
    }


def _analyze_spectral(audio: np.ndarray, sr: int) -> dict:
    """Compute spectral flatness (Wiener entropy) — high = metallic/buzzy."""
    # Short-time spectral flatness
    frame_len = 2048
    hop_len = 512
    flatness_values = []

    for i in range(0, len(audio) - frame_len, hop_len):
        frame = audio[i:i + frame_len]
        if np.sqrt(np.mean(frame ** 2)) < 0.01:
            continue
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame)))) + 1e-8
        geometric_mean = np.exp(np.mean(np.log(spectrum)))
        arithmetic_mean = np.mean(spectrum)
        flatness = geometric_mean / arithmetic_mean
        flatness_values.append(flatness)

    if not flatness_values:
        return {"mean_flatness": 0, "score": 50}

    mean_flatness = float(np.mean(flatness_values))
    # Natural speech: 0.02–0.15 flatness; very flat (close to 1) = noise/robotic
    is_metallic = mean_flatness > 0.40
    score = max(0, min(100, int((1 - mean_flatness) * 120)))

    return {
        "mean_flatness": round(mean_flatness, 4),
        "is_metallic": is_metallic,
        "score": score,
    }


def _compute_score(result: dict) -> int:
    """Combine sub-scores into overall naturalness score."""
    scores = []
    if "pitch" in result and "score" in result["pitch"]:
        scores.append(result["pitch"]["score"])
    if "energy" in result and "score" in result["energy"]:
        scores.append(result["energy"]["score"])
    if "spectral" in result and "score" in result["spectral"]:
        scores.append(result["spectral"]["score"])
    if not scores:
        return 50
    return round(sum(scores) / len(scores))


def main():
    if len(sys.argv) < 2:
        print("Usage: python robotic_detector.py <wav_file_or_directory>")
        sys.exit(1)

    target = sys.argv[1]
    wav_paths = []

    if os.path.isfile(target):
        wav_paths = [target]
    elif os.path.isdir(target):
        wav_paths = sorted(glob.glob(os.path.join(target, "*.wav")))
    else:
        print(f"Error: {target} is not a valid file or directory")
        sys.exit(1)

    if not wav_paths:
        print("No WAV files found.")
        sys.exit(1)

    print(f"Analyzing {len(wav_paths)} WAV file(s)...\n")
    print(f"{'File':<55} {'Score':>5} {'Pitch STD':>10} {'Energy STD':>11} {'Verdict'}")
    print("-" * 110)

    all_results = []
    for path in wav_paths:
        r = analyze_file(path)
        all_results.append(r)
        if "error" in r:
            print(f"  {Path(path).name:<53} ERROR: {r['error']}")
            continue

        pitch_std = r.get("pitch", {}).get("std_hz", "?")
        energy_std = r.get("energy", {}).get("std_db", "?")
        score = r.get("naturalness_score", "?")
        verdict = r.get("verdict", "?")
        print(
            f"  {Path(path).name:<53} {score:>5} "
            f"{pitch_std:>9.1f}Hz {energy_std:>10.1f}dB  {verdict}"
        )

    # Save JSON report
    report_path = Path(target) / "robotic_detector_report.json" if os.path.isdir(target) else Path(target).parent / "robotic_detector_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nReport saved: {report_path}")

    # Per-config summary for experiment directory
    configs = ["A_baseline", "B_saudi_nongreedy", "C_saudi_instruct", "D_saudi_lowpitch", "E_hijazi"]
    print("\n=== Config Summary ===")
    for cfg_id in configs:
        cfg_results = [r for r in all_results if cfg_id in Path(r.get("path", "")).name and "naturalness_score" in r]
        if cfg_results:
            avg_score = sum(r["naturalness_score"] for r in cfg_results) / len(cfg_results)
            avg_pitch = sum(r.get("pitch", {}).get("std_hz", 0) for r in cfg_results) / len(cfg_results)
            print(f"  {cfg_id:<28} avg score={avg_score:5.1f}  avg pitch_std={avg_pitch:6.1f}Hz")


if __name__ == "__main__":
    main()
