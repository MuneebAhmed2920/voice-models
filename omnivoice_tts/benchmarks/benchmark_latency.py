#!/usr/bin/env python3
"""
Comprehensive Latency & Resource Benchmark Suite for omnivoice_tts.
Reports:
- Cold Startup Time & Warm TTFA
- RTF (Real-Time Factor)
- Memory (RAM MB) & CPU Utilization
- p50, p95, and Mean metrics across all 4 voices
"""

import os
import sys
import time
import psutil
import json
import numpy as np

# Ensure project root in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print("=" * 85)
print("📊 BENCHMARKING PRODUCTION OMNIVOICE TTS ENGINE")
print("=" * 85)

process = psutil.Process(os.getpid())
ram_start_mb = process.memory_info().rss / (1024 * 1024)

# 1. Cold Startup Measurement
t_boot_0 = time.perf_counter()
from omnivoice_tts.engine import get_engine
engine = get_engine(device="cpu")
t_boot = time.perf_counter() - t_boot_0
ram_booted_mb = process.memory_info().rss / (1024 * 1024)

print(f"✅ Cold Boot & Warmup Completed in {t_boot:.2f}s | Model RAM: {ram_booted_mb - ram_start_mb:.1f} MB (Total: {ram_booted_mb:.1f} MB)")

# Benchmark Test Sentences
test_sentences = [
    ("1-Word Ack", "Hello.", "female_1"),
    ("Short Ack", "Absolutely.", "male_1"),
    ("Short Phrase", "I can help you with that.", "female_2"),
    ("Question Chunk", "What date would work best for you?", "male_2"),
    ("Call-Center Balance", "Your total balance of $129.50 is due on Monday at 3:30 PM.", "female_1"),
    ("Appointment Reminder", "Your appointment is scheduled for August 24th at 3:30 PM. Please call 310-555-0182.", "male_1"),
    ("Gesture & Pacing", "<breath> Thank you for calling! <chuckle> Let me look up your records right now.", "female_2")
]

results = []
all_ttfa = []
all_rtf = []

NUM_REPEATS = 3

print("\n" + "=" * 85)
print(f"⚡ RUNNING MULTI-PASS BENCHMARKS ({NUM_REPEATS} ITERATIONS PER CASE)")
print("=" * 85)

for label, text, voice_id in test_sentences:
    ttfa_runs = []
    rtf_runs = []
    
    print(f"\n🧪 Testing [{label}] (Voice: {voice_id})")
    print(f"   Text: \"{text}\"")
    
    for r in range(NUM_REPEATS):
        cpu_0 = psutil.cpu_percent(interval=None)
        audio_bytes, metrics = engine.generate(
            text=text,
            voice_id=voice_id,
            quality="fast",
            sample_rate=24000,
            output_format="wav"
        )
        ttfa_runs.append(metrics["ttfa_sec"])
        rtf_runs.append(metrics["rtf"])
        all_ttfa.append(metrics["ttfa_sec"])
        all_rtf.append(metrics["rtf"])
        
    p50_ttfa = float(np.percentile(ttfa_runs, 50))
    p95_ttfa = float(np.percentile(ttfa_runs, 95))
    mean_ttfa = float(np.mean(ttfa_runs))
    mean_rtf = float(np.mean(rtf_runs))
    
    print(f"   ⏱️ Mean TTFA: {mean_ttfa:.2f}s | p50: {p50_ttfa:.2f}s | p95: {p95_ttfa:.2f}s | RTF: {mean_rtf:.2f}x (Speed: {1/mean_rtf:.2f}x)")
    
    results.append({
        "label": label,
        "text": text,
        "voice_id": voice_id,
        "steps_used": metrics["steps_used"],
        "audio_sec": metrics["audio_duration_sec"],
        "mean_ttfa_sec": mean_ttfa,
        "p50_ttfa_sec": p50_ttfa,
        "p95_ttfa_sec": p95_ttfa,
        "mean_rtf": mean_rtf,
        "speed_x": 1.0 / mean_rtf if mean_rtf > 0 else 0
    })

overall_p50 = float(np.percentile(all_ttfa, 50))
overall_p95 = float(np.percentile(all_ttfa, 95))
overall_mean_rtf = float(np.mean(all_rtf))

print("\n" + "=" * 85)
print("📊 BENCHMARK SUMMARY TABLE")
print("=" * 85)
print(f"{'Test Case':<22} | {'Voice':<9} | {'Steps':<5} | {'Audio(s)':<8} | {'p50 TTFA':<9} | {'p95 TTFA':<9} | {'RTF':<6} | {'Speed':<8}")
print("-" * 85)
for r in results:
    print(f"{r['label']:<22} | {r['voice_id']:<9} | {r['steps_used']:<5} | {r['audio_sec']:<8.2f} | {r['p50_ttfa_sec']:<9.2f} | {r['p95_ttfa_sec']:<9.2f} | {r['mean_rtf']:<6.2f} | {r['speed_x']:<7.2f}x")

print("\n" + "=" * 85)
print(f"🌟 OVERALL ENGINE METRICS (CPU):")
print(f" • Cold Boot Time      : {t_boot:.2f} seconds")
print(f" • Model RAM Footprint : {ram_booted_mb - ram_start_mb:.1f} MB (Total: {ram_booted_mb:.1f} MB)")
print(f" • Overall Median TTFA : {overall_p50:.2f} seconds")
print(f" • Overall p95 TTFA    : {overall_p95:.2f} seconds")
print(f" • Overall Mean RTF    : {overall_mean_rtf:.2f}x ({1/overall_mean_rtf:.2f}x real-time)")
print("=" * 85)

out_json = os.path.join(BASE_DIR, "omnivoice_tts", "benchmarks", "benchmark_report.json")
with open(out_json, "w") as f:
    json.dump({
        "boot_time_sec": t_boot,
        "ram_model_mb": ram_booted_mb - ram_start_mb,
        "ram_total_mb": ram_booted_mb,
        "overall_p50_ttfa": overall_p50,
        "overall_p95_ttfa": overall_p95,
        "overall_mean_rtf": overall_mean_rtf,
        "cases": results
    }, f, indent=2)

print(f"Report written to {out_json}")
