#!/usr/bin/env python3
"""
Arabic TTS A/B/C/D Naturalness Experiment Runner.

Generates 4 configuration variants of the benchmark texts and saves them
for listening comparison. All outputs go to tests/arabic_naturalness/output/.

Configs:
  A — Baseline (original engine, Standard Arabic, greedy=class_temp=0)
  B — Saudi dialect + non-greedy (class_temp=0.20, najdi arabic)
  C — Saudi + prosody planner instruct (male, middle-aged, moderate pitch)
  D — Saudi + low pitch + more steps (male, middle-aged, low pitch, 36 steps)
  E — Standard Arabic, no postprocess (preserve internal pauses)

Usage:
  PYTHONPATH=. .venv/bin/python tests/arabic_naturalness/run_experiments.py
"""

import os
import sys
import time
import json
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf

# ──────────────────────────────────────────────────────────────────
# Benchmark texts — representative Saudi call-center speech
# ──────────────────────────────────────────────────────────────────
BENCHMARK_TEXTS = [
    {
        "id": "01_greeting",
        "label": "Greeting",
        "text": "أهلاً وسهلاً بكم في خدمة العملاء. كيف يمكنني مساعدتك اليوم؟",
    },
    {
        "id": "02_question",
        "label": "Question",
        "text": "هل يمكنك تزويدنا برقم الهوية الوطنية للتحقق من معلومات حسابك؟",
    },
    {
        "id": "03_confirmation",
        "label": "Confirmation",
        "text": "تمام، لحظة من فضلك بينما أتحقق من معلومات طلبك.",
    },
    {
        "id": "04_apology",
        "label": "Apology",
        "text": "نعتذر منك على هذا الإزعاج. سنبذل قصارى جهدنا لحل المشكلة في أقرب وقت ممكن.",
    },
    {
        "id": "05_important_info",
        "label": "Important Info",
        "text": "يرجى الانتباه: رمز التحقق المكوّن من ستة أرقام صالح لمدة خمس دقائق فقط، ولا تشاركه مع أي شخص.",
    },
    {
        "id": "06_good_news",
        "label": "Good News",
        "text": "يسعدنا إبلاغك بأن طلبك قد تمت الموافقة عليه بنجاح، وسيتم تسليمه خلال يومين إلى ثلاثة أيام عمل.",
    },
    {
        "id": "07_long_explanation",
        "label": "Long Explanation",
        "text": "يمكنك تتبع حالة طلبك من خلال تطبيق الجوال أو الموقع الإلكتروني الرسمي للشركة، حيث ستجد تفاصيل الشحن ورقم التتبع وموعد التسليم المتوقع.",
    },
    {
        "id": "08_farewell",
        "label": "Farewell",
        "text": "شكراً جزيلاً على تواصلك معنا. نتمنى لك يوماً سعيداً، مع السلامة.",
    },
    {
        "id": "09_benchmark_paragraph",
        "label": "Benchmark Paragraph",
        "text": (
            "نرحب بكم في مركز خدمة العملاء المتميزة. "
            "نحن هنا للإجابة على جميع استفساراتكم وحل مشكلاتكم بكل سرعة ودقة. "
            "يمكنكم التواصل معنا على مدار الساعة، طوال أيام الأسبوع. "
            "نسعى دائماً إلى تقديم أفضل تجربة خدمة ممكنة لكم. "
            "شكراً لثقتكم بنا."
        ),
    },
]

# ──────────────────────────────────────────────────────────────────
# Experiment Configurations
# ──────────────────────────────────────────────────────────────────
CONFIGS = {
    "A_baseline": {
        "label": "Baseline (Standard Arabic, greedy, original)",
        "language": "standard arabic",
        "instruct": None,
        "num_step": 20,
        "guidance_scale": 2.5,
        "class_temperature": 0.0,   # greedy — original behavior
        "postprocess_output": True,
        "speed": 1.0,
    },
    "B_saudi_nongreedy": {
        "label": "Saudi Arabic (Najdi), non-greedy sampling",
        "language": "najdi arabic",
        "instruct": None,
        "num_step": 28,
        "guidance_scale": 2.6,
        "class_temperature": 0.20,  # non-greedy → natural prosody variation
        "postprocess_output": True,
        "speed": 1.0,
    },
    "C_saudi_instruct": {
        "label": "Saudi + valid instruct (male, middle-aged, moderate pitch)",
        "language": "najdi arabic",
        "instruct": "male, middle-aged, moderate pitch",
        "num_step": 30,
        "guidance_scale": 2.8,
        "class_temperature": 0.20,
        "postprocess_output": True,
        "speed": 0.97,
    },
    "D_saudi_lowpitch": {
        "label": "Saudi + low pitch, more steps",
        "language": "najdi arabic",
        "instruct": "male, middle-aged, low pitch",
        "num_step": 36,
        "guidance_scale": 2.8,
        "class_temperature": 0.20,
        "postprocess_output": True,
        "speed": 0.95,
    },
    "E_hijazi": {
        "label": "Hijazi Arabic (Jeddah/Makkah) + moderate pitch",
        "language": "hijazi arabic",
        "instruct": "male, middle-aged, moderate pitch",
        "num_step": 30,
        "guidance_scale": 2.7,
        "class_temperature": 0.20,
        "postprocess_output": True,
        "speed": 0.97,
    },
}


def load_model():
    """Load OmniVoice model once — uses local cache, no network needed."""
    from omnivoice import OmniVoice
    import os
    # Use cached model — avoids HuggingFace network call at startup
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    print("Loading k2-fsa/OmniVoice from local cache...")
    m = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cpu",
        dtype="float32",
        local_files_only=True,
    )
    print(f"Model loaded. SR={m.sampling_rate}Hz")
    return m


def run_single(model, text: str, cfg: dict, output_path: str) -> dict:
    """Run a single synthesis with given config, save WAV, return stats."""
    t0 = time.time()
    kwargs = {
        "text": text,
        "language": cfg["language"],
        "num_step": cfg["num_step"],
        "guidance_scale": cfg["guidance_scale"],
        "class_temperature": cfg["class_temperature"],
        "postprocess_output": cfg["postprocess_output"],
        "speed": cfg.get("speed", 1.0),
    }
    if cfg.get("instruct"):
        kwargs["instruct"] = cfg["instruct"]

    results = model.generate(**kwargs)
    elapsed = time.time() - t0

    audio = results[0]
    sr = model.sampling_rate
    dur = len(audio) / sr
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))

    sf.write(output_path, audio, sr)
    return {
        "output_path": output_path,
        "duration_sec": round(dur, 3),
        "gen_time_sec": round(elapsed, 2),
        "rtf": round(elapsed / max(dur, 1e-6), 3),
        "rms_dbfs": round(20 * np.log10(rms + 1e-8), 1),
        "peak_dbfs": round(20 * np.log10(peak + 1e-8), 1),
        "sr": sr,
    }


def main():
    out_dir = PROJECT_ROOT / "tests" / "arabic_naturalness" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model()

    all_results = {}

    for btext in BENCHMARK_TEXTS:
        tid = btext["id"]
        text = btext["text"]
        print(f"\n{'='*60}")
        print(f"TEXT: {tid} — {btext['label']}")
        print(f"  {text[:80]}...")
        all_results[tid] = {"text": text, "configs": {}}

        for cfg_id, cfg in CONFIGS.items():
            out_path = str(out_dir / f"{tid}_{cfg_id}.wav")
            if Path(out_path).exists():
                print(f"  [{cfg_id}] Already exists, skipping.")
                continue
            print(f"  [{cfg_id}] {cfg['label']} ...", end="", flush=True)
            try:
                stats = run_single(model, text, cfg, out_path)
                print(f" ✓ {stats['duration_sec']:.2f}s audio / {stats['gen_time_sec']:.1f}s gen (RTF {stats['rtf']:.2f})")
                all_results[tid]["configs"][cfg_id] = stats
            except Exception as e:
                print(f" ✗ ERROR: {e}")
                all_results[tid]["configs"][cfg_id] = {"error": str(e)}

    # Save results JSON
    report_path = out_dir / "experiment_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"All experiments done. Results saved to:")
    print(f"  {out_dir}")
    print(f"  {report_path}")

    # Summary table
    print("\n=== RTF Summary (lower = faster) ===")
    print(f"{'Config':<25} {'Avg RTF':>8} {'Avg Dur':>8}")
    for cfg_id in CONFIGS:
        rtfs = []
        durs = []
        for tid in all_results:
            r = all_results[tid]["configs"].get(cfg_id, {})
            if "rtf" in r:
                rtfs.append(r["rtf"])
                durs.append(r["duration_sec"])
        if rtfs:
            print(f"  {cfg_id:<23} {sum(rtfs)/len(rtfs):>8.2f} {sum(durs)/len(durs):>8.2f}s")


if __name__ == "__main__":
    main()
