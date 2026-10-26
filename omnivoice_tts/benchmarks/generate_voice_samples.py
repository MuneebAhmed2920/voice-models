#!/usr/bin/env python3
"""
Generate demonstration WAV audio samples for all 4 Fixed Production Voices:
- female_1 (Dr. Sara)
- female_2 (Emma)
- male_1   (Dr. Alex)
- male_2   (Marcus)
"""

import os
import sys
import soundfile as sf

# Ensure project root in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print("=" * 85)
print("🎧 GENERATING PRODUCTION SAMPLES FOR ALL 4 FIXED VOICES")
print("=" * 85)

from omnivoice_tts.engine import get_engine
engine = get_engine(device="cpu")

test_text = "Your appointment is scheduled for August 24th at 3:30 PM. Your total is $129.50. Please call us at 310-555-0182."

voices = [
    ("female_1", "Dr. Sara - Articulate & Friendly"),
    ("female_2", "Emma - Calm & Empathetic"),
    ("male_1", "Dr. Alex - Resonant Baritone"),
    ("male_2", "Marcus - Conversational American")
]

out_dir = os.path.join(BASE_DIR, "omnivoice_tts", "samples")
os.makedirs(out_dir, exist_ok=True)

for voice_id, persona in voices:
    print(f"\n🎙️ Generating audio for [{voice_id}] ({persona})...")
    audio_bytes, metrics = engine.generate(
        text=test_text,
        voice_id=voice_id,
        quality="fast",
        sample_rate=24000,
        output_format="wav",
        include_breath=True
    )
    
    out_file = os.path.join(out_dir, f"sample_{voice_id}.wav")
    with open(out_file, "wb") as f:
        f.write(audio_bytes)
        
    print(f"✅ Saved: {out_file}")
    print(f"   ⏱️ TTFA: {metrics['ttfa_sec']:.2f}s | Audio Duration: {metrics['audio_duration_sec']:.2f}s | Speed: {metrics['speed_x']:.2f}x real-time")

print("\n" + "=" * 85)
print(f"🎉 ALL 4 VOICE SAMPLES GENERATED IN: {out_dir}")
print("=" * 85)
