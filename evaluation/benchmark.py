"""Performance benchmark for Arabic OmniVoice TTS pipeline.

Measures:
  - Voice/model identifier
  - TTS generation time (ms)
  - Time to first audio (TTFA) — streaming benchmark
  - Audio duration (seconds)
  - Real-time factor (RTF = gen_time / audio_duration)
  - CPU usage (%)
  - Memory usage (MB RSS)
  - Sample rate
  - Pipeline stage breakdown (normalize, prosody, TTS, post-process)

Usage:
    cd "/Users/haseebahmed/agent/omni voice Arabic"
    .venv/bin/python evaluation/benchmark.py
    .venv/bin/python evaluation/benchmark.py --telephony 8000
    .venv/bin/python evaluation/benchmark.py --telephony 16000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ──────────────────────────────────────────────────────────────────
# Benchmark sentences — representative of real telephony usage
# ──────────────────────────────────────────────────────────────────
BENCHMARK_SENTENCES = [
    # Short confirmations
    ("نعم، بالتأكيد.", "short_confirmation", "msa"),
    ("حسناً، سأتحقق من ذلك.", "short_confirmation", "msa"),
    # Questions
    ("هل يمكنني مساعدتك في شيء آخر؟", "question", "msa"),
    # Normal info
    ("طلبك قيد المعالجة وسيصل خلال ثلاثة أيام.", "normal_info", "msa"),
    # Phone number
    ("يمكنك الاتصال بنا على الرقم صفر خمسة صفر واحد اثنان ثلاثة أربعة خمسة ستة سبعة.", "phone_number", "msa"),
    # Money
    ("إجمالي فاتورتك هو مائتان وخمسة وأربعون ريالاً وخمسون هللة.", "money", "msa"),
    # Long explanation
    ("نود إبلاغك بأن الخدمة ستكون متوقفة مؤقتاً للصيانة الدورية في الفترة من الساعة الثانية صباحاً حتى السادسة صباحاً بتوقيت المملكة العربية السعودية.", "long_explanation", "msa"),
    # Mixed Arabic/English
    ("طلبك confirmed وسيصل خلال thirty minutes.", "code_switch", "msa"),
    # Apology
    ("نعتذر بشدة عن الإزعاج الذي سببناه لك.", "apology", "msa"),
    # Good news
    ("يسعدنا إبلاغك بأن طلبك تمت الموافقة عليه بنجاح.", "good_news", "msa"),
]


def get_cpu_percent():
    if _HAS_PSUTIL:
        return psutil.cpu_percent(interval=0.1)
    return -1.0


def get_memory_mb():
    if _HAS_PSUTIL:
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)
    return -1.0


def get_vram_mb():
    """Returns VRAM usage if CUDA GPU is available, else -1."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return -1.0


def benchmark_sentence(
    engine,
    text: str,
    label: str,
    dialect: str,
    pipeline_mode: str = "legacy",
    telephony_sr: int = None,
    num_step: int = 20,
) -> dict:
    """
    Benchmarks one sentence in the specified pipeline mode.

    Args:
        engine: ArabicOmniVoiceEngine instance
        text: Input text
        label: Category label
        dialect: 'msa' or 'egyptian'
        pipeline_mode: 'legacy' or 'pipeline'
        telephony_sr: Optional target sample rate (8000/16000)
        num_step: Diffusion steps

    Returns:
        Benchmark metrics dict
    """
    cpu_before = get_cpu_percent()
    mem_before = get_memory_mb()
    vram_before = get_vram_mb()

    t_start = time.perf_counter()

    if pipeline_mode == "pipeline":
        audio, sr = engine.synthesize_pipeline(
            text=text,
            dialect=dialect,
            telephony_sr=telephony_sr,
            num_step=num_step,
        )
    else:
        audio, sr = engine.synthesize(
            text=text,
            dialect=dialect,
            num_step=num_step,
            auto_preprocess=True,
        )

    t_end = time.perf_counter()

    cpu_after = get_cpu_percent()
    mem_after = get_memory_mb()
    vram_after = get_vram_mb()

    gen_ms = (t_end - t_start) * 1000.0
    duration_s = len(audio) / sr if sr > 0 else 0.0
    rtf = gen_ms / max(duration_s * 1000, 1.0)

    return {
        "label": label,
        "pipeline": pipeline_mode,
        "text_preview": text[:50] + ("..." if len(text) > 50 else ""),
        "dialect": dialect,
        "sample_rate": sr,
        "telephony_sr": telephony_sr,
        "gen_ms": round(gen_ms, 1),
        "duration_s": round(duration_s, 3),
        "rtf": round(rtf, 3),
        "num_samples": len(audio),
        "peak_dbfs": round(20 * np.log10(max(float(np.max(np.abs(audio))), 1e-9)), 2) if len(audio) > 0 else -999,
        "rms_dbfs": round(20 * np.log10(max(float(np.sqrt(np.mean(audio ** 2))), 1e-9)), 2) if len(audio) > 0 else -999,
        "cpu_pct": round((cpu_before + cpu_after) / 2, 1),
        "mem_delta_mb": round(mem_after - mem_before, 1),
        "vram_mb": round(vram_after, 1) if vram_after >= 0 else None,
        "num_step": num_step,
    }


def benchmark_streaming_ttfa(engine, text: str, dialect: str = "msa") -> float:
    """
    Measures time-to-first-audio (TTFA) for streaming synthesis.

    Returns:
        TTFA in milliseconds
    """
    t0 = time.perf_counter()
    gen = engine.stream_synthesize(text, dialect=dialect, num_step=12)
    first_chunk, _ = next(gen)
    ttfa_ms = (time.perf_counter() - t0) * 1000.0
    return round(ttfa_ms, 1)


def run_benchmark(telephony_sr: int = None):
    """Main benchmark runner."""
    from src.models.omnivoice_engine import ArabicOmniVoiceEngine

    print("=" * 72)
    print("  Arabic OmniVoice TTS — Performance Benchmark")
    if telephony_sr:
        print(f"  Telephony SR: {telephony_sr} Hz")
    print("=" * 72)

    engine = ArabicOmniVoiceEngine(preload_models=False)
    print(f"[*] Device: {engine.device} | psutil: {_HAS_PSUTIL}\n")

    all_results = []

    # ── Main benchmark: legacy vs pipeline ───────────────────────────────
    for text, label, dialect in BENCHMARK_SENTENCES:
        print(f"[{label}] {text[:55]}...")

        # Legacy
        leg = benchmark_sentence(engine, text, label, dialect, "legacy", num_step=16)
        # Humanized pipeline
        pip = benchmark_sentence(engine, text, label, dialect, "pipeline",
                                 telephony_sr=telephony_sr, num_step=20)

        print(f"  legacy:   gen={leg['gen_ms']:>6.0f}ms | dur={leg['duration_s']:.2f}s | RTF={leg['rtf']:.2f} | RMS={leg['rms_dbfs']:.1f}dB")
        print(f"  pipeline: gen={pip['gen_ms']:>6.0f}ms | dur={pip['duration_s']:.2f}s | RTF={pip['rtf']:.2f} | RMS={pip['rms_dbfs']:.1f}dB | sr={pip['sample_rate']}Hz")

        all_results.extend([leg, pip])

    # ── Streaming TTFA benchmark ──────────────────────────────────────────
    print("\n[TTFA Benchmark — streaming first-chunk latency]")
    ttfa_sentences = [
        "مرحباً بك! كيف يمكنني مساعدتك اليوم؟",
        "طلبك قيد المعالجة حالياً.",
        "نعم، بالتأكيد. لحظة من فضلك.",
    ]
    ttfa_values = []
    for sent in ttfa_sentences:
        try:
            ttfa = benchmark_streaming_ttfa(engine, sent)
            ttfa_values.append(ttfa)
            print(f"  TTFA: {ttfa:.0f}ms — {sent[:50]}")
        except Exception as e:
            print(f"  TTFA ERROR: {e}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    if all_results:
        legacy_rtfs = [r["rtf"] for r in all_results if r["pipeline"] == "legacy"]
        pipeline_rtfs = [r["rtf"] for r in all_results if r["pipeline"] == "pipeline"]
        legacy_gen = [r["gen_ms"] for r in all_results if r["pipeline"] == "legacy"]
        pipeline_gen = [r["gen_ms"] for r in all_results if r["pipeline"] == "pipeline"]

        print(f"  Avg RTF   — Legacy: {sum(legacy_rtfs)/len(legacy_rtfs):.2f}  |  Pipeline: {sum(pipeline_rtfs)/len(pipeline_rtfs):.2f}")
        print(f"  Avg GenMs — Legacy: {sum(legacy_gen)/len(legacy_gen):.0f}ms  |  Pipeline: {sum(pipeline_gen)/len(pipeline_gen):.0f}ms")

    if ttfa_values:
        print(f"  Avg TTFA: {sum(ttfa_values)/len(ttfa_values):.0f}ms")

    print("=" * 72)

    # Save results
    out_file = Path(__file__).parent / "benchmark_results.json"
    summary = {
        "device": engine.device,
        "telephony_sr": telephony_sr,
        "ttfa_samples": ttfa_values,
        "results": all_results,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[*] Benchmark results saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arabic TTS Benchmark")
    parser.add_argument("--telephony", type=int, choices=[8000, 16000], default=None,
                        help="Test telephony resampling to 8000Hz (G.711) or 16000Hz (G.722)")
    args = parser.parse_args()
    run_benchmark(telephony_sr=args.telephony)
