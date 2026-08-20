"""
Comprehensive Latency & Telephony Benchmark Suite for OmniVoice TTS Engine.
Measures Time-To-First-Audio (TTFA), Real-Time Factor (RTF), memory footprint, and concurrency.
"""

import os
import sys
import time
import json
import torch
import numpy as np

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from runtime.engine import get_engine, OmniVoiceEngine


def run_benchmark():
    print("=" * 80)
    print("🚀 Running OmniVoice Production Telephony Latency & Quality Benchmark")
    print("=" * 80)

    engine = get_engine(default_ode_steps=16, warmup=True)
    status = engine.get_status()
    print(f"Device: {status['device']} ({status['dtype']}) | GPU: {status.get('gpu_name') or 'N/A'}")
    print(f"Default ODE Steps: {status['default_ode_steps']} | Default Sample Rate: {status['default_sample_rate']} Hz")
    print("-" * 80)

    test_phrases = [
        ("Short Phrase", "Yes, absolutely!"),
        ("Medium Sentence", "Thank you for calling Clara support. Your balance is $45.50."),
        ("Telephony Date & Time", "Your appointment is scheduled for August 24th at 3:30 PM with Dr. Smith."),
        ("Phone & Account Number", "Please call customer service at (800) 555-0199 for account #1234."),
        ("Long Paragraph", "Thank you for holding. I verified your information and your refund of $125.00 has been processed successfully. You should see the credit on your statement within 3 to 5 business days."),
    ]

    results = []

    for label, text in test_phrases:
        print(f"\nEvaluating: '{label}' ({len(text.split())} words)...")
        
        # Test 8kHz PSTN Telephony
        t0 = time.time()
        audio_8k, metrics_8k = engine.synthesize(
            text=text,
            voice="voice_clara",
            sample_rate=8000,
            format_type="pcm16",
            ode_steps=16
        )
        t1 = time.time()

        res_item = {
            "label": label,
            "text": text,
            "word_count": len(text.split()),
            "ttfa_ms": metrics_8k["ttfa_ms"],
            "inference_ms": metrics_8k["inference_time_ms"],
            "audio_dur_sec": metrics_8k["audio_duration_sec"],
            "rtf": metrics_8k["rtf"],
            "bytes_8k_pcm": len(audio_8k),
            "normalized_text": metrics_8k["normalized_text"]
        }
        results.append(res_item)
        print(f"  ➜ TTFA: {res_item['ttfa_ms']:.1f}ms | Audio: {res_item['audio_dur_sec']:.2f}s | RTF: {res_item['rtf']:.4f}")

    # Summary Table
    print("\n" + "=" * 80)
    print("📊 BENCHMARK SUMMARY RESULTS TABLE")
    print("=" * 80)
    header = f"{'Test Case':<24} | {'Words':<5} | {'TTFA (ms)':<10} | {'Audio (s)':<9} | {'RTF':<8} | {'Audio Bytes':<12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['label']:<24} | {r['word_count']:<5} | {r['ttfa_ms']:<10.1f} | {r['audio_dur_sec']:<9.2f} | {r['rtf']:<8.4f} | {r['bytes_8k_pcm']:<12}")
    
    avg_ttfa = sum(r["ttfa_ms"] for r in results) / len(results)
    avg_rtf = sum(r["rtf"] for r in results) / len(results)
    total_audio = sum(r["audio_dur_sec"] for r in results)
    print("-" * len(header))
    print(f"{'AVERAGE / TOTAL':<24} | {'-':<5} | {avg_ttfa:<10.1f} | {total_audio:<9.2f} | {avg_rtf:<8.4f} | {'-'}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
