#!/usr/bin/env python3
"""
OmniVoice Production CLI Tool.
Synthesizes high-fidelity speech from the command line using any of the 4 production voices.

Usage:
  python3 generate.py "Hello! Your appointment is confirmed for August 24th at 3:30 PM." --voice female_1 -o greeting.wav
  python3 generate.py --interactive
"""

import os
import sys
import argparse
import subprocess

# Ensure project root in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from omnivoice_tts.engine import get_engine

def run_interactive():
    engine = get_engine(device="cpu")
    print("=" * 80)
    print("🎙️ OMNIVOICE PRODUCTION INTERACTIVE CONSOLE")
    print("=" * 80)
    print("Available Voices: female_1 (Dr. Sara), female_2 (Emma), male_1 (Dr. Alex), male_2 (Marcus)")
    print("Type any text to synthesize and play through your speakers.")
    print("Type 'exit' to quit.\n")
    
    current_voice = "female_1"
    while True:
        try:
            cmd = input(f"\n🗣️ [{current_voice}] Enter text >> ").strip()
            if not cmd:
                continue
            if cmd.lower() in ["exit", "quit", "q"]:
                print("👋 Exiting console. Goodbye!")
                break
            if cmd.startswith("/voice"):
                parts = cmd.split()
                if len(parts) > 1 and parts[1] in engine.registry.list_voices():
                    current_voice = parts[1]
                    print(f"✅ Switched voice to: {current_voice}")
                else:
                    print(f"⚠️ Supported voices: {engine.registry.list_voices()}")
                continue
                
            audio_bytes, metrics = engine.generate(
                text=cmd,
                voice_id=current_voice,
                quality="fast",
                sample_rate=24000,
                output_format="wav",
                include_breath=True
            )
            
            temp_file = "temp_console_output.wav"
            with open(temp_file, "wb") as f:
                f.write(audio_bytes)
                
            print(f"✅ Generated in {metrics['ttfa_sec']:.2f}s | Speed: {metrics['speed_x']:.2f}x real-time")
            try:
                subprocess.run(["afplay", temp_file], check=True)
            except Exception:
                pass
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        except KeyboardInterrupt:
            print("\n👋 Exiting console. Goodbye!")
            break

def main():
    parser = argparse.ArgumentParser(description="OmniVoice Production Speech Generator")
    parser.add_argument("text", nargs="?", default="", help="Text to speak")
    parser.add_argument("-v", "--voice", default="female_1", choices=["female_1", "female_2", "male_1", "male_2"], help="Voice ID (default: female_1)")
    parser.add_argument("-o", "--output", default="output.wav", help="Output WAV file path (default: output.wav)")
    parser.add_argument("-sr", "--sample-rate", type=int, default=24000, choices=[8000, 16000, 24000], help="Audio sample rate (default: 24000)")
    parser.add_argument("-f", "--format", default="wav", choices=["wav", "pcm16", "mulaw"], help="Audio format (default: wav)")
    parser.add_argument("-s", "--speed", type=float, default=1.0, help="Speaking speed (default: 1.0)")
    parser.add_argument("-p", "--play", action="store_true", help="Play audio immediately on speakers")
    parser.add_argument("-i", "--interactive", action="store_true", help="Launch interactive CLI console")
    
    args = parser.parse_args()
    
    if args.interactive or not args.text:
        run_interactive()
    else:
        engine = get_engine(device="cpu")
        audio_bytes, metrics = engine.generate(
            text=args.text,
            voice_id=args.voice,
            speed=args.speed,
            quality="fast",
            sample_rate=args.sample_rate,
            output_format=args.format,
            include_breath=True
        )
        
        with open(args.output, "wb") as f:
            f.write(audio_bytes)
            
        print(f"\n✅ Generated audio successfully!")
        print(f"📁 Saved to       : {args.output}")
        print(f"⏱️ Audio Duration : {metrics['audio_duration_sec']:.2f}s")
        print(f"⚡ Synthesis Time : {metrics['ttfa_sec']:.2f}s ({metrics['speed_x']:.2f}x real-time)")
        print(f"🎭 Voice ID       : {args.voice} ({metrics['persona']})")
        print(f"📝 Spoken Text    : \"{metrics['text_normalized']}\"")
        
        if args.play:
            try:
                subprocess.run(["afplay", args.output], check=True)
            except Exception:
                pass

if __name__ == "__main__":
    main()
