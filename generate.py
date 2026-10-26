#!/usr/bin/env python3
"""Simple CLI for High-Quality Spanish Speech Synthesis.

Usage:
    python generate.py --text "Hola, ¿cómo estás?" --voice female --output output.wav
    python generate.py --text "Buenos días." --voice male --output output.wav
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure package directory is importable
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from spanish_voice import SpanishVoice, normalize_spanish


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate high-quality 24 kHz Spanish speech using OmniVoice."
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input Spanish text to synthesize.",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="female",
        choices=["female", "male", "spanish_female", "spanish_male"],
        help="Voice persona to use (default: female).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.wav",
        help="Output WAV file path (default: output.wav).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=24,
        help="Diffusion steps (default: 24 for highest quality).",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=2.2,
        help="Guidance scale (default: 2.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation (default: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on (default: cpu).",
    )

    args = parser.parse_args()

    print(f"[*] Initializing Spanish Voice ({args.voice})...")
    tts = SpanishVoice(
        voice=args.voice,
        device=args.device,
        diffusion_steps=args.steps,
        guidance_scale=args.guidance,
    )

    print(f"[*] Raw text:    \"{args.text}\"")
    normalized = normalize_spanish(args.text)
    print(f"[*] Spoken text: \"{normalized}\"")

    audio = tts.generate(
        text=args.text,
        output=args.output,
        seed=args.seed,
    )

    duration_s = len(audio) / float(tts.sample_rate)
    print(f"[+] Output saved: {args.output} ({duration_s:.2f}s, {tts.sample_rate} Hz, mono PCM_16 WAV)")


if __name__ == "__main__":
    main()
