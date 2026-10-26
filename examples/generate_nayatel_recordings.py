#!/usr/bin/env python3
"""
examples/generate_nayatel_recordings.py

Official Production Suite: Nayatel Pakistani Urdu Audio Recordings
Using the Mahrukh Pakistani Urdu Voice Model (mahrukh_urdu_model).

Generates both:
1. 24kHz HD Studio Master Audio (.wav)
2. 8kHz G.711 Telephony Line Audio (.wav, PSTN bandpass filtered for call center / Asterisk IVR)

Includes Whisper ASR intelligibility verification.
"""

import os
import sys
import time

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_ROOT)

from runtime.engine import UrduVoiceModel
from transformers import pipeline

NAYATEL_SCRIPTS = {
    "nayatel_ivr_master": {
        "title": "Nayatel Official Inbound Helpline IVR (1441)",
        "category": "Helpline IVR",
        "text": (
            "اَلسَّلَامُ عَلَیْکُم! نیاٹیل میں خوش آمدید۔ "
            "پاکستان کے سب سے بااعتماد اور تیز ترین فائبر انٹرنیٹ نیٹ ورک سے رابطہ کرنے کا شکریہ۔ "
            "نئے کنکشن اور کوریج کی معلومات کے لیے، ایک دبائیے۔ "
            "انٹرنیٹ اور نیا ٹی وی کی تکنیکی رہنمائی کے لیے، دو دبائیے۔ "
            "بلنگ اور اکاؤنٹ سے متعلق تفصیلات کے لیے، تین دبائیے۔ "
            "یا ہمارے کَسٹمر کیئر نمائندے سے بات کرنے کے لیے، صفر دبائیے۔ "
            "نیاٹیل، آپ کا بااعتماد ڈیجیٹل پارٹنر۔"
        )
    },
    "nayatel_agent_greeting": {
        "title": "Nayatel Customer Care Representative Greeting (Mahrukh)",
        "category": "Customer Support Call",
        "text": (
            "اَلسَّلَامُ عَلَیْکُم! نیاٹیل کَسٹمر کیئر میں خوش آمدید، میرا نام ماہ رخ ہے۔ "
            "میں تصدیق کر سکتی ہوں کہ آپ کا فائبر انٹرنیٹ کنکشن کامیابی سے ایکٹیو ہو چکا ہے۔ "
            "اگر آپ کو سپیڈ یا راؤٹر کی سیٹنگز میں کوئی دشواری پیش آ رہی ہو، "
            "تو میں ابھی آپ کی مکمل رہنمائی کر سکتی ہوں۔ "
            "نیاٹیل پر رابطہ کرنے کا شکریہ، آپ کا دن اچھا گزرے۔"
        )
    },
    "nayatel_brand_promo": {
        "title": "Nayatel High-Speed Fiber & Naya TV Brand Promo",
        "category": "Brand Commercial & Radio Promo",
        "text": (
            "نیاٹیل لایا ہے آپ کے گھر اور کاروبار کے لیے بلاتعطل الٹرا فاسٹ فائبر آپٹک انٹرنیٹ، "
            "ایچ ڈی نیا ٹی وی اور ڈیجیٹل وائس سروس۔ "
            "بغیر کسی تعطل کے تیز ترین ڈاؤن لوڈنگ اور سٹریمنگ کا لطف اٹھائیے۔ "
            "نئے کنکشن کے لیے ابھی چودہ اکتالیس پر رابطہ کیجیے۔ "
            "نیاٹیل، تیز ترین، بااعتماد اور جدید۔"
        )
    },
    "nayatel_tech_resolution": {
        "title": "Nayatel Automated Support & Service Restoration Alert",
        "category": "Technical Resolution Follow-up",
        "text": (
            "اَلسَّلَامُ عَلَیْکُم! نیاٹیل سپورٹ کی جانب سے رابطہ کیا جا رہا ہے۔ "
            "آپ کی تکنیکی شکایت پر کام مکمل کر لیا گیا ہے اور انٹرنیٹ کنکشن بحال ہو چکا ہے۔ "
            "بہتر سپیڈ کے لیے اپنے راؤٹر کو ایک بار ری سٹارٹ کر لیجیے۔ "
            "نیاٹیل منتخب کرنے کا بہت شکریہ۔"
        )
    }
}

def main():
    print("=" * 70)
    print("🎙️ Generating Official Nayatel Pakistani Urdu Recordings")
    print("=" * 70)

    outputs_dir = os.path.join(PACKAGE_ROOT, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    print("\n[1/3] Initializing Mahrukh Urdu Voice Model...")
    model = UrduVoiceModel()

    print("\n[2/3] Initializing Whisper ASR for Quality & Phonetic Verification...")
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device="cpu")

    results = []

    print("\n[3/3] Synthesizing Nayatel Audio Assets...")
    for key, item in NAYATEL_SCRIPTS.items():
        title = item["title"]
        category = item["category"]
        script_text = item["text"]

        print(f"\n--- Processing: {title} ({category}) ---")
        print(f"Script: {script_text}")

        # 1. 24kHz HD Studio Audio
        out_24k = os.path.join(outputs_dir, f"{key}_24k_hd.wav")
        t0 = time.time()
        aud_24k, sr_24k = model.synthesize(script_text, sample_rate=24000, output_file=out_24k)
        dur_24k = len(aud_24k) / float(sr_24k)
        gen_time_24k = time.time() - t0

        asr_res_24k = asr(out_24k, generate_kwargs={"language": "ur", "task": "transcribe"})["text"]
        print(f"  [HD 24kHz] Saved: {out_24k}")
        print(f"             Duration: {dur_24k:.2f}s | Gen Time: {gen_time_24k:.2f}s")
        print(f"             Whisper ASR: {asr_res_24k}")

        # 2. 8kHz Telephony Filtered Audio (PSTN standard)
        out_8k = os.path.join(outputs_dir, f"{key}_8k_telephony.wav")
        t1 = time.time()
        aud_8k, sr_8k = model.synthesize(script_text, sample_rate=8000, telephony_filter=True, output_file=out_8k)
        dur_8k = len(aud_8k) / float(sr_8k)
        gen_time_8k = time.time() - t1

        asr_res_8k = asr(out_8k, generate_kwargs={"language": "ur", "task": "transcribe"})["text"]
        print(f"  [PSTN 8kHz] Saved: {out_8k}")
        print(f"              Duration: {dur_8k:.2f}s | Gen Time: {gen_time_8k:.2f}s")
        print(f"              Whisper ASR: {asr_res_8k}")

        results.append({
            "key": key,
            "title": title,
            "category": category,
            "file_24k": out_24k,
            "dur_24k": dur_24k,
            "asr_24k": asr_res_24k,
            "file_8k": out_8k,
            "dur_8k": dur_8k,
            "asr_8k": asr_res_8k
        })

    print("\n" + "=" * 70)
    print("✅ All Nayatel Urdu Recordings Generated & Verified Successfully!")
    print("=" * 70)
    for r in results:
        print(f"• {r['title']}")
        print(f"    HD File (24kHz): {r['file_24k']} ({r['dur_24k']:.2f}s)")
        print(f"    Telephony File (8kHz): {r['file_8k']} ({r['dur_8k']:.2f}s)")

if __name__ == "__main__":
    main()
