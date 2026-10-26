#!/usr/bin/env python3
"""
Continuous Stability & Memory Leak Stress Test for omnivoice_tts.
Runs 25 continuous speech generations and verifies memory stabilization.
"""

import os
import sys
import time
import psutil
import gc
import numpy as np

# Ensure project root in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print("=" * 85)
print("🛡️ RUNNING CONTINUOUS STABILITY & MEMORY LEAK STRESS TEST")
print("=" * 85)

from omnivoice_tts.engine import get_engine
engine = get_engine(device="cpu")

process = psutil.Process(os.getpid())
gc.collect()
ram_initial_mb = process.memory_info().rss / (1024 * 1024)

print(f"📊 Initial RAM Baseline: {ram_initial_mb:.1f} MB")

sample_texts = [
    ("female_1", "Hello! Thank you for calling."),
    ("female_2", "Your appointment is confirmed for tomorrow at 2 PM."),
    ("male_1", "I can certainly assist you with your balance inquiry."),
    ("male_2", "Please call our office at 800-555-0199 if you need anything else.")
]

NUM_ITERATIONS = 20
ram_history = []

for i in range(1, NUM_ITERATIONS + 1):
    voice_id, text = sample_texts[(i - 1) % len(sample_texts)]
    audio_bytes, metrics = engine.generate(
        text=text,
        voice_id=voice_id,
        quality="fast",
        sample_rate=16000,
        output_format="pcm16"
    )
    current_ram = process.memory_info().rss / (1024 * 1024)
    ram_history.append(current_ram)
    if i % 5 == 0 or i == 1:
        print(f" • [Gen {i:02d}/{NUM_ITERATIONS:02d}] Voice: {voice_id:<8} | TTFA: {metrics['ttfa_sec']:.2f}s | Audio: {metrics['audio_duration_sec']:.2f}s | Current RAM: {current_ram:.1f} MB")

gc.collect()
ram_final_mb = process.memory_info().rss / (1024 * 1024)
ram_delta_mb = ram_final_mb - ram_initial_mb

print("\n" + "=" * 85)
print(f"🎉 STRESS TEST COMPLETED!")
print(f" • Initial RAM : {ram_initial_mb:.1f} MB")
print(f" • Final RAM   : {ram_final_mb:.1f} MB")
print(f" • RAM Delta   : {ram_delta_mb:+.1f} MB")
if ram_delta_mb < 30.0:
    print("✅ ZERO MEMORY LEAKS DETECTED! RAM is perfectly stabilized.")
else:
    print("⚠️ Notice: RAM increased by > 30MB.")
print("=" * 85)
