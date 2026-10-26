"""
Arabic Human Realism Benchmark Suite (Section 5, PROMPT_FOR_ANTIGRAVITY.md)
Synthesizes all 4 official benchmark sentences and generates 24kHz studio and 8kHz telephony samples.
"""

import os
import sys
import time
import json
import numpy as np
import soundfile as sf

# Set offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from arabic_voice_model_kit.runtime.engine import ArabicVoiceEngine

BENCHMARK_PROMPTS = [
    {
        "id": "1_conversational_greeting",
        "name": "Conversational Greeting",
        "text": "أهلاً وسهلاً بحضرتك، تفضل كيف أقدر أساعدك اليوم؟",
        "expected_emotion": "neutral",
        "description": "Base customer care welcome with relaxed airway breath"
    },
    {
        "id": "2_excited_confirmation",
        "name": "Excited Confirmation",
        "text": "[متحمس] ألف مبروك! تم تأكيد موعدك بنجاح ليوم الثلاثاء الساعة ٤:٣٠ مساءً.",
        "expected_emotion": "excited",
        "description": "High-energy joyful confirmation with 95ms crisp breath and number normalization"
    },
    {
        "id": "3_sympathetic_customer_care",
        "name": "Sympathetic Customer Care",
        "text": "[متعاطف] أعتذر جداً لحضرتك عن الإزعاج، وأنا هتابع الموضوع بنفسي حتى يتم حله بالكامل.",
        "expected_emotion": "sympathetic",
        "description": "Warm empathetic customer service with gentle 210ms breath and soft presence"
    },
    {
        "id": "4_emotional_reassurance",
        "name": "Emotional Reassurance",
        "text": "[مؤثر] شكراً جزيلاً لثقتك الغالية فينا وشراكتنا الممتدة على مدار السنوات.",
        "expected_emotion": "emotional",
        "description": "Heartfelt gratitude with chest resonance and resonant inhalation"
    }
]


def run_benchmarks(output_dir: str = None):
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "samples")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("🎙️ ARABIC HUMAN REALISM BENCHMARK SUITE (PROMPT_FOR_ANTIGRAVITY.MD)")
    print("   Testing 7-Pillar Humanization: Physiological Breath, 0.92x Cadence, 2.2 Guidance, Proximity EQ")
    print(f"   Output Directory: {output_dir}")
    print("=" * 80)

    print("\n⏳ Initializing ArabicVoiceEngine (OmniVoice + VoiceClonePrompt cache)...")
    t0 = time.time()
    engine = ArabicVoiceEngine(warmup=False)
    print(f"✅ Engine initialized in {time.time() - t0:.2f}s.\n")

    results = []

    for idx, item in enumerate(BENCHMARK_PROMPTS, 1):
        print("-" * 80)
        print(f"[{idx}/4] Benchmark: {item['name']}")
        print(f"     Text: \"{item['text']}\"")
        print(f"     Trait: {item['description']}")

        # 1. Synthesize Studio 24kHz WAV
        wav_bytes_24k, metrics_24k = engine.synthesize(
            text=item["text"],
            voice="voice_arabic_clara",
            language="najdi arabic",
            sample_rate=24000,
            format_type="wav",
            ode_steps=24,  # Studio quality
            guidance_scale=2.2,
            speed=0.92,
            add_breath=True,
            apply_eq=True
        )

        out_24k_path = os.path.join(output_dir, f"benchmark_{item['id']}_24k.wav")
        with open(out_24k_path, "wb") as f:
            f.write(wav_bytes_24k)

        # 2. Synthesize Telephony 8kHz WAV (PSTN / Asterisk PBX)
        wav_bytes_8k, metrics_8k = engine.synthesize(
            text=item["text"],
            voice="voice_arabic_clara",
            language="najdi arabic",
            sample_rate=8000,
            format_type="wav",
            ode_steps=16,  # Ultra-low latency telephony
            guidance_scale=2.2,
            speed=0.92,
            add_breath=True,
            apply_eq=True
        )

        out_8k_path = os.path.join(output_dir, f"benchmark_{item['id']}_8k.wav")
        with open(out_8k_path, "wb") as f:
            f.write(wav_bytes_8k)

        # Inspect 24k audio metrics
        audio_24k, _ = sf.read(out_24k_path)
        rms = np.sqrt(np.mean(audio_24k ** 2))
        rms_db = 20 * np.log10(rms + 1e-9)
        peak_db = 20 * np.log10(np.max(np.abs(audio_24k)) + 1e-9)

        print(f"     ✅ Studio (24kHz):   {out_24k_path}")
        print(f"     ✅ Telephony (8kHz): {out_8k_path}")
        print(f"     ⏱️ Audio Dur:        {metrics_24k['audio_duration_sec']:.2f}s")
        print(f"     ⚡ Inference:        {metrics_24k['inference_time_ms']:.0f}ms (RTF: {metrics_24k['rtf']:.2f})")
        print(f"     🔊 Loudness:         RMS {rms_db:.1f} dBFS | Peak {peak_db:.1f} dBFS")
        print(f"     🏷️ Normalized Text: \"{metrics_24k['normalized_text']}\"")

        results.append({
            "id": item["id"],
            "name": item["name"],
            "input_text": item["text"],
            "normalized_text": metrics_24k["normalized_text"],
            "emotion": metrics_24k["emotion"],
            "has_breath": metrics_24k["has_breath"],
            "studio_file_24k": out_24k_path,
            "telephony_file_8k": out_8k_path,
            "duration_sec": metrics_24k["audio_duration_sec"],
            "inference_time_ms": metrics_24k["inference_time_ms"],
            "rtf": metrics_24k["rtf"],
            "rms_dbfs": round(float(rms_db), 2),
            "peak_dbfs": round(float(peak_db), 2)
        })

    # Save benchmark report
    report_path = os.path.join(output_dir, "benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("✨ ALL 4 ARABIC BENCHMARK RECORDINGS GENERATED SUCCESSFULLY!")
    print(f"   Report: {report_path}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_benchmarks()
