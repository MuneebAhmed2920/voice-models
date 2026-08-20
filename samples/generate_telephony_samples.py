"""
Telephony Sample Audio Generator.
Generates reference WAV and PSTN audio benchmark samples for Vicidial and AI Bot verification.
"""

import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from runtime.engine import get_engine

SAMPLE_SCRIPTS = [
    ("clara_01_conversation.wav", "Hello! Thank you for calling Clara customer support. How can I help you today?", 24000, "wav", None),
    ("clara_02_question.wav", "Would you like me to look up your recent order history or help you with a return?", 24000, "wav", None),
    ("clara_03_numbers.wav", "Your confirmation code is 849201.", 24000, "wav", None),
    ("clara_04_money.wav", "The total balance on your account is $149.99, which includes your $15.50 discount.", 24000, "wav", None),
    ("clara_05_phone_number.wav", "You can reach our dedicated support line directly at (800) 555-0199 extension 204.", 24000, "wav", None),
    ("clara_06_date_time.wav", "Your appointment is confirmed for August 24th at 3:30 PM.", 24000, "wav", None),
    ("clara_07_long_sentence.wav", "I have successfully processed your request, and a confirmation email along with your tracking details has been sent to your primary address on file.", 24000, "wav", None),
    ("clara_08_breathing.wav", "[breath] I understand your concern, and I will be more than happy to resolve this for you right away.", 24000, "wav", None),
    ("clara_09_pstn_8k.wav", "Thank you for calling. This is an 8 kilohertz PSTN telephony audio sample for Vicidial call center testing.", 8000, "wav", None),
]


def generate_all_samples():
    output_dir = os.path.join(_PROJECT_ROOT, "samples")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(f"🎙️ Generating Telephony Verification Audio Samples into: {output_dir}")
    print("=" * 80)

    engine = get_engine(default_ode_steps=16, warmup=True)

    for filename, text, sample_rate, fmt, emotion in SAMPLE_SCRIPTS:
        out_path = os.path.join(output_dir, filename)
        print(f"\nGenerating: {filename} ({sample_rate}Hz)...")
        print(f"  Script: '{text}'")
        
        t0 = time.time()
        audio_bytes, metrics = engine.synthesize(
            text=text,
            voice="voice_clara",
            emotion=emotion,
            sample_rate=sample_rate,
            format_type=fmt,
            ode_steps=16
        )
        
        with open(out_path, "wb") as f:
            f.write(audio_bytes)
            
        print(f"  ✅ Saved {len(audio_bytes)} bytes to {filename} (Duration: {metrics['audio_duration_sec']}s, TTFA: {metrics['ttfa_ms']}ms)")

    print("\n" + "=" * 80)
    print("🎉 All telephony benchmark audio samples generated successfully.")
    print("=" * 80)


if __name__ == "__main__":
    generate_all_samples()
