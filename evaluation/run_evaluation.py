"""Evaluation runner — generates before/after audio samples for the 18 test categories.

Usage:
    cd "/Users/haseebahmed/agent/omni voice Arabic"
    .venv/bin/python evaluation/run_evaluation.py

Outputs:
    evaluation/audio/before/<category>_<id>.wav  — legacy pipeline
    evaluation/audio/after/<category>_<id>.wav   — new humanized 12-stage pipeline

Each output line prints: category, text, TTFA (ms), generation time (ms), duration (s), RTF.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.omnivoice_engine import ArabicOmniVoiceEngine

# ──────────────────────────────────────────────────────────────────
EVAL_DIR = Path(__file__).parent
AUDIO_DIR = EVAL_DIR / "audio"
BEFORE_DIR = AUDIO_DIR / "before"
AFTER_DIR = AUDIO_DIR / "after"
TEST_CASES_FILE = EVAL_DIR / "test_cases.json"

BEFORE_DIR.mkdir(parents=True, exist_ok=True)
AFTER_DIR.mkdir(parents=True, exist_ok=True)

# Voice to use for all evaluations (MSA Kareem — formal clear voice)
DEFAULT_VOICE_ID = "msa_kareem"
DEFAULT_DIALECT = "msa"
NUM_STEP_BEFORE = 16  # Faster for before samples
NUM_STEP_AFTER = 20   # Slightly higher quality for after samples
# ──────────────────────────────────────────────────────────────────


def measure_synthesis(engine: ArabicOmniVoiceEngine, text: str, pipeline: str, outfile: str) -> dict:
    """
    Synthesizes one test sentence and returns latency metrics.

    Args:
        engine: Initialized ArabicOmniVoiceEngine
        text: Input text
        pipeline: 'before' or 'after'
        outfile: Path to save WAV

    Returns:
        dict with timing and quality metrics
    """
    t0 = time.perf_counter()

    if pipeline == "before":
        # Legacy pipeline: basic synthesize with default normalization
        audio, sr = engine.synthesize(
            text=text,
            voice_id=DEFAULT_VOICE_ID,
            dialect=DEFAULT_DIALECT,
            num_step=NUM_STEP_BEFORE,
            auto_preprocess=True,
        )
    else:
        # New 12-stage humanized pipeline
        audio, sr = engine.synthesize_pipeline(
            text=text,
            voice_id=DEFAULT_VOICE_ID,
            dialect=DEFAULT_DIALECT,
            num_step=NUM_STEP_AFTER,
        )

    t_end = time.perf_counter()
    gen_ms = (t_end - t0) * 1000.0
    duration_s = len(audio) / sr if sr > 0 else 0.0
    rtf = gen_ms / max(duration_s * 1000, 1.0)

    # Save WAV
    sf.write(str(outfile), audio, sr)

    return {
        "gen_ms": round(gen_ms, 1),
        "duration_s": round(duration_s, 2),
        "rtf": round(rtf, 3),
        "sample_rate": sr,
        "num_samples": len(audio),
        "peak": round(float(np.max(np.abs(audio))), 4) if len(audio) > 0 else 0.0,
        "rms": round(float(np.sqrt(np.mean(audio ** 2))), 4) if len(audio) > 0 else 0.0,
    }


def run_evaluation():
    """Main evaluation loop."""
    print("=" * 72)
    print("  Arabic OmniVoice TTS — Before/After Evaluation")
    print(f"  Output: {AUDIO_DIR}")
    print("=" * 72)
    print()

    # Load test cases
    with open(TEST_CASES_FILE, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # Initialize engine
    print("[*] Loading engine...")
    engine = ArabicOmniVoiceEngine(preload_models=False)
    print(f"[*] Engine initialized on device: {engine.device}\n")

    results = []
    total_before_ms = 0.0
    total_after_ms = 0.0
    processed = 0

    for case in test_cases:
        cat_id = case["id"]
        category = case["category"]
        description = case["description"]
        texts = case["texts"]

        print(f"[Category {cat_id:02d}] {category} — {description}")
        print("-" * 60)

        for i, text_pair in enumerate(texts):
            # Use Arabic text if available, otherwise English
            text = text_pair.get("ar") or text_pair.get("en")
            if not text:
                continue

            item_id = f"{cat_id:02d}_{category}_{i+1:02d}"
            before_file = BEFORE_DIR / f"{item_id}.wav"
            after_file = AFTER_DIR / f"{item_id}.wav"

            # Skip if already generated (use --force to regenerate)
            if "--force" not in sys.argv and before_file.exists() and after_file.exists():
                print(f"  [skip] {item_id} (already exists)")
                continue

            # Truncate for display
            display = text[:65] + "..." if len(text) > 65 else text
            print(f"  [{i+1}] {display}")

            try:
                # BEFORE: legacy pipeline
                b = measure_synthesis(engine, text, "before", before_file)
                # AFTER: new humanized pipeline
                a = measure_synthesis(engine, text, "after", after_file)

                total_before_ms += b["gen_ms"]
                total_after_ms += a["gen_ms"]
                processed += 1

                print(
                    f"       BEFORE: {b['gen_ms']:.0f}ms gen | {b['duration_s']:.2f}s audio | RTF={b['rtf']:.2f} | "
                    f"peak={b['peak']:.3f}"
                )
                print(
                    f"       AFTER:  {a['gen_ms']:.0f}ms gen | {a['duration_s']:.2f}s audio | RTF={a['rtf']:.2f} | "
                    f"peak={a['peak']:.3f}"
                )

                results.append({
                    "id": item_id,
                    "category": category,
                    "text": text,
                    "before": b,
                    "after": a,
                    "before_file": str(before_file),
                    "after_file": str(after_file),
                })

            except Exception as e:
                print(f"       ERROR: {e}")
                results.append({
                    "id": item_id,
                    "category": category,
                    "text": text,
                    "error": str(e),
                })

        print()

    # Summary
    print("=" * 72)
    print(f"  Processed: {processed} test cases")
    if processed > 0:
        print(f"  Avg BEFORE gen time: {total_before_ms/processed:.0f}ms")
        print(f"  Avg AFTER  gen time: {total_after_ms/processed:.0f}ms")
        delta = total_after_ms - total_before_ms
        sign = "+" if delta > 0 else ""
        print(f"  Latency delta (AFTER - BEFORE): {sign}{delta/processed:.0f}ms per sample")
    print(f"  Audio files: {AUDIO_DIR}")
    print("=" * 72)

    # Save results JSON
    results_file = EVAL_DIR / "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[*] Results saved to {results_file}")


if __name__ == "__main__":
    run_evaluation()
