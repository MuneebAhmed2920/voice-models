#!/usr/bin/env python3
"""
examples/telephony_call_demo.py

Full customer service dialogue simulation using Marukh English Voice Model.
Generates both 24kHz HD and 8kHz PSTN audio.
"""

import os
from marukh_english_model import EnglishVoiceModel

DIALOGUE = [
    ("step1_intro", "Hello, thank you for calling customer care. My name is Mahrukh, how may I help you today?"),
    ("step2_verify", "Could you please confirm your account number or registered phone number for verification?"),
    ("step3_processing", "Thank you. Please hold on for just a moment while I pull up your latest order status."),
    ("step4_confirmed", "Great news! Your payment of $49.99 has been confirmed, and your order will be delivered tomorrow."),
    ("step5_closing", "Is there anything else I can assist you with today? Thank you for calling, and have a wonderful day!")
]

def main():
    out_dir = "telephony_demo_outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    model = EnglishVoiceModel()
    
    for tag, text in DIALOGUE:
        # 24kHz HD
        hd_path = os.path.join(out_dir, f"{tag}_24k.wav")
        model.synthesize(text=text, sample_rate=24000, output_file=hd_path)
        
        # 8kHz Telephony
        tel_path = os.path.join(out_dir, f"{tag}_8k_telephony.wav")
        model.synthesize(text=text, sample_rate=8000, telephony_filter=True, output_file=tel_path)
        
        print(f"Synthesized: {tag}")

    print(f"\nAll telephony demonstration files generated in '{out_dir}/'!")

if __name__ == "__main__":
    main()
