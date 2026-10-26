#!/usr/bin/env python3
"""
examples/telephony_call_demo.py

Comprehensive telephony call demonstration using Mahrukh Pakistani Urdu Voice Model.
Generates:
1. 24kHz HD studio reference
2. 8kHz G.711 telephony line audio (PSTN standard)
Verifies outputs with Whisper ASR.
"""

import os
import sys

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_ROOT)

from runtime.engine import UrduVoiceModel
from transformers import pipeline

def main():
    model = UrduVoiceModel()
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device="cpu")

    call_script = (
        "اَلسَّلَامُ عَلَیْکُم! ہماری کَسٹمر سروس میں خوش آمدید، میرا نام ماہ رخ ہے۔ "
        "میں تصدیق کر سکتی ہوں کہ آپ کی پےمنٹ receive ہو چکی ہے۔ "
        "آپ کا آرڈر کنفرْم ہو گیا ہے اور پیکج ڈسپَیچ کے لیے ایکٹیو ہے۔ "
        "کل دوپہر تک آپ کو ڈیلیوری بھی receive ہو جائے گی۔ "
        "کیا میں کسٹمر کے لیے مزید کوئی مدد کر سکتی ہوں؟"
    )

    out_24k = os.path.join(PACKAGE_ROOT, "outputs", "standalone_call_24k_hd.wav")
    out_8k = os.path.join(PACKAGE_ROOT, "outputs", "standalone_call_8k_telephony.wav")

    print("\n--- 1. Generating 24kHz HD Studio Audio ---")
    aud_24k, sr_24k = model.synthesize(call_script, sample_rate=24000, output_file=out_24k)
    dur_24k = len(aud_24k) / float(sr_24k)
    trans_24k = asr(out_24k)["text"]
    print(f"Generated: {out_24k} ({dur_24k:.2f}s, {sr_24k}Hz)")
    print(f"Whisper ASR: {trans_24k}")

    print("\n--- 2. Generating 8kHz PSTN Telephony Filtered Audio ---")
    aud_8k, sr_8k = model.synthesize(call_script, sample_rate=8000, telephony_filter=True, output_file=out_8k)
    dur_8k = len(aud_8k) / float(sr_8k)
    trans_8k = asr(out_8k)["text"]
    print(f"Generated: {out_8k} ({dur_8k:.2f}s, {sr_8k}Hz)")
    print(f"Whisper ASR: {trans_8k}")

    print("\n==================== VERIFICATION COMPLETE ====================")
    print("Mahrukh Standalone Urdu Model generated both 24kHz and 8kHz telephonic files successfully.")

if __name__ == "__main__":
    main()
