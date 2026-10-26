#!/usr/bin/env python3
"""
Custom Voice Generator for Saved Female 1 Voice.
Synthesizes ANY text using your exact saved female voice profile.

Usage:
  1. Direct command line text:
     python3 generate_speech.py "Hello! Your total balance is $450.00."
  
  2. Custom output filename:
     python3 generate_speech.py "Welcome to our store!" -o welcome.wav
     
  3. Interactive Live Mode:
     python3 generate_speech.py --interactive
"""

import os
import sys
import time
import argparse
import subprocess
import tempfile
import soundfile as sf
import torch
import numpy as np

# Ensure project root in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

VOICE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(VOICE_DIR, "female_1_prompt.pt")

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice_tts.text.normalizer import normalize_english_text
from omnivoice_tts.text.gestures import GestureParser
from omnivoice_tts.audio.codecs import resample_audio, encode_audio

_MODEL = None
_VOICE_PROMPT = None
_GESTURE_PARSER = None

def get_voice_engine():
    global _MODEL, _VOICE_PROMPT, _GESTURE_PARSER
    if _MODEL is None:
        print("⚡ Loading OmniVoice Speech Engine...")
        t0 = time.perf_counter()
        _MODEL = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cpu")
        _VOICE_PROMPT = torch.load(PROMPT_PATH, map_location="cpu", weights_only=False)
        _GESTURE_PARSER = GestureParser(sample_rate=24000)
        print(f"✅ Loaded in {(time.perf_counter() - t0):.2f}s | Voice Profile: Exact Female 1 (Dr. Sara)")
    return _MODEL, _VOICE_PROMPT, _GESTURE_PARSER

def synthesize_text(
    text: str,
    output_path: str = "output_female_1.wav",
    sample_rate: int = 24000,
    speed: float = 1.0,
    play_audio: bool = False,
    include_breath: bool = True
):
    model, voice_prompt, gesture_parser = get_voice_engine()
    
    t0 = time.perf_counter()
    # 1. English Normalization
    norm_text = normalize_english_text(text)
    
    # 2. Gesture Parsing
    clean_text, emotion_mod = gesture_parser.parse(norm_text)
    
    gen_cfg = OmniVoiceGenerationConfig(
        num_step=16,          # Exact high-fidelity 16-step flow matching
        guidance_scale=2.2,    # Exact prompt stability
        denoise=True,
        postprocess_output=True,
        pad_duration=0.08,
        fade_duration=0.08
    )
    
    audio_list = model.generate(
        text=clean_text,
        language="en",
        voice_clone_prompt=voice_prompt,
        generation_config=gen_cfg,
        speed=speed
    )
    
    raw_speech = np.asarray(audio_list[0], dtype=np.float32).reshape(-1)
    
    if include_breath:
        breath = gesture_parser.generate_breath(duration_sec=0.25, intensity=0.12, is_male=False)
        combined = np.concatenate([breath, raw_speech])
    else:
        combined = raw_speech
        
    resampled = resample_audio(combined, orig_sr=24000, target_sr=sample_rate)
    sf.write(output_path, resampled, sample_rate, subtype="PCM_16")
    
    t_gen = time.perf_counter() - t0
    dur = len(resampled) / float(sample_rate)
    
    print(f"\n✅ Audio generated successfully!")
    print(f"📁 Saved to       : {output_path}")
    print(f"⏱️ Audio Duration : {dur:.2f} seconds")
    print(f"⚡ Generation Time: {t_gen:.2f} seconds ({dur/t_gen:.2f}x real-time)")
    print(f"📝 Spoken Text    : \"{norm_text}\"")
    
    if play_audio:
        try:
            print("🔊 Playing audio through Mac speaker...")
            subprocess.run(["afplay", output_path], check=True)
        except Exception as e:
            print(f"Notice: {e}")

def run_interactive():
    print("=" * 80)
    print("🎙️ INTERACTIVE CONSOLE FOR EXACT FEMALE 1 VOICE")
    print("=" * 80)
    print("Type any text to synthesize and hear it speak aloud through your Mac speakers.")
    print("Type 'exit' or 'quit' to close.\n")
    
    while True:
        try:
            text = input("\n🗣️ Enter text >> ").strip()
            if not text:
                continue
            if text.lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break
                
            synthesize_text(text, output_path="temp_voice.wav", play_audio=True)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

def main():
    parser = argparse.ArgumentParser(description="Generate speech using exact Female 1 voice profile.")
    parser.add_argument("text", nargs="?", default="", help="Text to speak")
    parser.add_argument("-o", "--output", default="output_female_1.wav", help="Output WAV path (default: output_female_1.wav)")
    parser.add_argument("-sr", "--sample-rate", type=int, default=24000, choices=[8000, 16000, 24000], help="Audio sample rate (default: 24000)")
    parser.add_argument("-s", "--speed", type=float, default=1.0, help="Speaking speed factor (default: 1.0)")
    parser.add_argument("-p", "--play", action="store_true", help="Play audio immediately on Mac speakers")
    parser.add_argument("-i", "--interactive", action="store_true", help="Launch interactive CLI console")
    
    args = parser.parse_args()
    
    if args.interactive or not args.text:
        run_interactive()
    else:
        synthesize_text(
            text=args.text,
            output_path=args.output,
            sample_rate=args.sample_rate,
            speed=args.speed,
            play_audio=args.play
        )

if __name__ == "__main__":
    main()
