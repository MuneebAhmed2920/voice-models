#!/usr/bin/env python3
"""
examples/quickstart.py

Quick demonstration of the Marukh English Voice Model.
"""

from marukh_english_model import EnglishVoiceModel

def main():
    print("Initializing Marukh English Voice Model...")
    model = EnglishVoiceModel()

    # 1. Generate 24kHz Studio HD Audio
    print("Generating 24kHz Studio HD audio...")
    audio_hd, sr_hd = model.synthesize(
        text="Hello, thank you for calling. My name is Mahrukh, how may I assist you today?",
        sample_rate=24000,
        output_file="greeting_hd.wav"
    )
    print(f"Saved: greeting_hd.wav ({len(audio_hd)/sr_hd:.2f}s at {sr_hd}Hz)")

    # 2. Generate 8kHz Telephony Line Audio (for Asterisk PBX / Vicidial)
    print("Generating 8kHz Telephony line audio...")
    audio_8k, sr_8k = model.synthesize(
        text="Please give me a moment while I check your account details and update your delivery status.",
        sample_rate=8000,
        telephony_filter=True,
        output_file="account_check_telephony.wav"
    )
    print(f"Saved: account_check_telephony.wav ({len(audio_8k)/sr_8k:.2f}s at {sr_8k}Hz)")

if __name__ == "__main__":
    main()
