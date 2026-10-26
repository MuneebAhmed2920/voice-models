"""
Precompute and save the 4 Production Fixed Voices for omnivoice_tts.
Voices:
1. female_1 (Dr. Sara - Articulate & Friendly)
2. female_2 (Emma - Calm & Empathetic)
3. male_1   (Dr. Alex - Resonant Baritone)
4. male_2   (Marcus - Conversational American)
"""

import os
import sys
import torch
from omnivoice import OmniVoice

VOICES_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(VOICES_DIR, exist_ok=True)

print("⚡ Initializing OmniVoice for Fixed Voice Pre-computation...")
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cpu")

# Voice 1: female_1 (Synthesized prompt from reference audio or high-fidelity style instruction)
print("🎙️ Precomputing [female_1] (Dr. Sara)...")
prompt_f1 = {
    "voice_id": "female_1",
    "persona": "Dr. Sara (Articulate & Friendly)",
    "instruct": "female, american accent, young adult, moderate pitch",
    "speed": 1.0
}
torch.save(prompt_f1, os.path.join(VOICES_DIR, "female_1.pt"))

# Voice 2: female_2 (Emma - Calm & Empathetic)
print("🎙️ Precomputing [female_2] (Emma)...")
prompt_f2 = {
    "voice_id": "female_2",
    "persona": "Emma (Calm & Empathetic)",
    "instruct": "female, american accent, middle-aged, moderate pitch",
    "speed": 0.96
}
torch.save(prompt_f2, os.path.join(VOICES_DIR, "female_2.pt"))

# Voice 3: male_1 (Dr. Alex - Cloned Reference Prompt)
print("🎙️ Precomputing [male_1] (Dr. Alex)...")
ref_clean = "/Users/haseebahmed/agent/VICIDial/bot/cursor_ref_clean.wav"
if os.path.exists(ref_clean):
    clone_prompt_m1 = model.create_voice_clone_prompt(
        ref_audio=ref_clean,
        ref_text="Before we gave cursor agents their own computers, it felt like the world was kind of upside down."
    )
    torch.save({
        "voice_id": "male_1",
        "persona": "Dr. Alex (Resonant Baritone)",
        "prompt_obj": clone_prompt_m1,
        "speed": 1.0
    }, os.path.join(VOICES_DIR, "male_1.pt"))
else:
    torch.save({
        "voice_id": "male_1",
        "persona": "Dr. Alex (Resonant Baritone)",
        "instruct": "male, american accent, low pitch, middle-aged",
        "speed": 1.0
    }, os.path.join(VOICES_DIR, "male_1.pt"))

# Voice 4: male_2 (Marcus - Conversational American)
print("🎙️ Precomputing [male_2] (Marcus)...")
prompt_m2 = {
    "voice_id": "male_2",
    "persona": "Marcus (Conversational American)",
    "instruct": "male, american accent, young adult, moderate pitch",
    "speed": 1.02
}
torch.save(prompt_m2, os.path.join(VOICES_DIR, "male_2.pt"))

print("\n✅ All 4 Production Fixed Voices precomputed and saved in:", VOICES_DIR)
