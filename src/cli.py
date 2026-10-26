"""Command-Line Interface for Arabic OmniVoice TTS."""

import argparse
import os
import sys

from src.arabic_nlp.diacritizer import ArabicDiacritizer
from src.arabic_nlp.normalizer import ArabicNormalizer
from src.models.omnivoice_engine import ArabicOmniVoiceEngine
from src.models.voice_manager import BUILTIN_VOICES, VoiceManager


def main():
    parser = argparse.ArgumentParser(
        description="🎙️ Arabic OmniVoice TTS CLI — Speech Synthesis & Voice Cloning"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: synthesize
    synth_parser = subparsers.add_parser("synthesize", help="Synthesize Arabic text to audio")
    synth_parser.add_argument("text", type=str, help="Arabic text to speak")
    synth_parser.add_argument("--voice", "-v", type=str, default="egy_mohamed", help="Voice ID")
    synth_parser.add_argument("--dialect", "-d", type=str, choices=["msa", "egyptian"], default=None, help="Dialect")
    synth_parser.add_argument("--speed", "-s", type=float, default=1.0, help="Speech speed multiplier")
    synth_parser.add_argument("--steps", type=int, default=24, help="Diffusion steps (16-32)")
    synth_parser.add_argument("--output", "-o", type=str, default="output.wav", help="Output WAV path")
    synth_parser.add_argument("--instruct", type=str, default=None, help="Voice design prompt")

    # Command: conversational
    conv_parser = subparsers.add_parser("conversational", help="Synthesize with human gestures, breathing & emotion")
    conv_parser.add_argument("text", type=str, help="Arabic text (supports [متحمس], [متعاطف], [مؤثر], [تنفس])")
    conv_parser.add_argument("--voice", "-v", type=str, default="msa_kareem", help="Voice ID")
    conv_parser.add_argument("--dialect", "-d", type=str, choices=["msa", "egyptian"], default=None, help="Dialect")
    conv_parser.add_argument("--emotion", "-e", type=str, choices=["neutral", "excited", "sympathetic", "emotional"], default=None, help="Emotion override")
    conv_parser.add_argument("--no-breath", action="store_true", help="Disable turn-initial breath")
    conv_parser.add_argument("--steps", type=int, default=20, help="Diffusion steps")
    conv_parser.add_argument("--output", "-o", type=str, default="conversational_output.wav", help="Output WAV path")

    # Command: clone
    clone_parser = subparsers.add_parser("clone", help="Zero-shot clone voice from reference audio")
    clone_parser.add_argument("--ref", "-r", type=str, required=True, help="Path to reference audio file")
    clone_parser.add_argument("--text", "-t", type=str, required=True, help="Target Arabic text to speak")
    clone_parser.add_argument("--ref-transcript", type=str, default=None, help="Transcript of reference audio")
    clone_parser.add_argument("--output", "-o", type=str, default="cloned_output.wav", help="Output WAV path")

    # Command: list-voices
    list_parser = subparsers.add_parser("list-voices", help="List all available Arabic voices")
    list_parser.add_argument("--dialect", type=str, default=None, help="Filter by dialect")
    list_parser.add_argument("--gender", type=str, default=None, help="Filter by gender (male/female)")

    # Command: normalize
    norm_parser = subparsers.add_parser("normalize", help="Normalize Arabic text and add Tashkeel")
    norm_parser.add_argument("text", type=str, help="Raw Arabic text")
    norm_parser.add_argument("--dialect", type=str, default="msa", choices=["msa", "egyptian"])

    # Command: web
    web_parser = subparsers.add_parser("web", help="Launch interactive Gradio Web Studio")
    web_parser.add_argument("--port", "-p", type=int, default=7860, help="Server port")
    web_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    web_parser.add_argument("--share", action="store_true", help="Create public Gradio link")

    # Command: api
    api_parser = subparsers.add_parser("api", help="Launch FastAPI REST server")
    api_parser.add_argument("--port", "-p", type=int, default=8000, help="Server port")
    api_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "synthesize":
        print(f"🎙️ Synthesizing text: '{args.text}'")
        engine = ArabicOmniVoiceEngine()
        audio, sr = engine.synthesize(
            text=args.text,
            voice_id=args.voice,
            dialect=args.dialect,
            speed=args.speed,
            num_step=args.steps,
            instruct=args.instruct,
            output_path=args.output
        )
        print(f"✅ Audio generated successfully: {args.output} ({sr} Hz, {len(audio)/sr:.2f}s)")

    elif args.command == "conversational":
        print(f"🎙️ Conversational Synthesis: '{args.text}'")
        engine = ArabicOmniVoiceEngine()
        audio, sr = engine.synthesize_conversational(
            text=args.text,
            voice_id=args.voice,
            dialect=args.dialect,
            emotion_override=args.emotion,
            include_breath=not args.no_breath,
            num_step=args.steps,
            output_path=args.output
        )
        print(f"✅ Conversational audio saved: {args.output} ({sr} Hz, {len(audio)/sr:.2f}s)")

    elif args.command == "clone":
        if not os.path.exists(args.ref):
            print(f"❌ Error: Reference audio file '{args.ref}' not found.")
            sys.exit(1)
        print(f"🧬 Cloning voice from '{args.ref}' to speak: '{args.text}'")
        engine = ArabicOmniVoiceEngine()
        audio, sr = engine.synthesize(
            text=args.text,
            ref_audio=args.ref,
            ref_text=args.ref_transcript,
            output_path=args.output
        )
        print(f"✅ Cloned audio saved: {args.output} ({sr} Hz)")

    elif args.command == "list-voices":
        vm = VoiceManager()
        voices = vm.list_voices(dialect=args.dialect, gender=args.gender)
        print(f"\n📢 Available Arabic Voices ({len(voices)}):")
        print("=" * 70)
        for v in voices:
            print(f"  • ID: {v.id:16} | Name: {v.name_ar}")
            print(f"    Dialect: {v.dialect.upper()} | Gender: {v.gender} | Style: {v.style} | Engine: {v.engine}")
            print("-" * 70)

    elif args.command == "normalize":
        norm = ArabicNormalizer(dialect=args.dialect)
        diac = ArabicDiacritizer()
        normalized = norm.normalize(args.text)
        vocalized = diac.diacritize(normalized)
        print("\n📝 Arabic Text Normalization:")
        print(f"  Original:   {args.text}")
        print(f"  Normalized: {normalized}")
        print(f"  Vocalized:  {vocalized}\n")

    elif args.command == "web":
        from src.app.web_studio import launch_studio
        print(f"🌐 Launching Arabic Web Studio at http://{args.host}:{args.port}")
        launch_studio(host=args.host, port=args.port, share=args.share)

    elif args.command == "api":
        import uvicorn
        print(f"🚀 Launching FastAPI server on http://{args.host}:{args.port}")
        uvicorn.run("src.app.api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
