#!/usr/bin/env python3
"""
Spanish Whisper Transcription & Evaluation Tool.

Standardized blind ASR transcription pipeline using faster-whisper large-v3
for Spanish TTS intelligibility benchmarking and audio transcription.

Features:
- Uses OpenAI's Whisper large-v3 via faster-whisper (CTranslate2).
- Automatic hardware acceleration (CUDA float16, Apple Silicon / CPU int8).
- Enforces strict blind Spanish evaluation settings (language='es', task='transcribe', beam_size=5).
- Word-level timestamps, segment confidence logprobs, and no-speech probabilities.
- Outputs detailed JSON and clean TXT transcripts.
- Supports pre-downloading model weights for offline execution.
"""

import argparse
import json
import os
import sys
import time
from typing import Optional, List, Dict, Any


def get_optimal_device() -> tuple[str, str]:
    """Determine the optimal device and compute type for faster-whisper."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    
    # On CPU and Apple Silicon, int8 provides optimal latency and low memory footprint
    return "cpu", "int8"


def get_whisper_model(
    model_size: str = "large-v3",
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    download_root: Optional[str] = None
):
    """Load or download the faster-whisper model."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Error: 'faster-whisper' is not installed.", file=sys.stderr)
        print("Install it with: pip install faster-whisper", file=sys.stderr)
        sys.exit(1)

    opt_device, opt_compute = get_optimal_device()
    active_device = device or opt_device
    active_compute = compute_type or opt_compute

    print(f"Loading Whisper model '{model_size}' on {active_device.upper()} (compute: {active_compute})...")
    model = WhisperModel(
        model_size_or_path=model_size,
        device=active_device,
        compute_type=active_compute,
        download_root=download_root
    )
    return model


def transcribe_audio(
    model,
    audio_path: str,
    language: str = "es",
    beam_size: int = 5,
    word_timestamps: bool = True
) -> Dict[str, Any]:
    """Transcribe a Spanish audio file using standardized blind evaluation settings."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    t_start = time.time()
    segments, info = model.transcribe(
        audio_path,
        language=language,
        task="transcribe",
        beam_size=beam_size,
        word_timestamps=word_timestamps,
        temperature=0.0
    )

    segment_list = []
    full_text_parts = []

    for seg in segments:
        words_data = []
        if word_timestamps and seg.words:
            for w in seg.words:
                words_data.append({
                    "word": w.word,
                    "start": round(w.start, 2),
                    "end": round(w.end, 2),
                    "probability": round(w.probability, 4)
                })

        seg_dict = {
            "id": seg.id,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
            "avg_logprob": round(seg.avg_logprob, 4),
            "no_speech_prob": round(seg.no_speech_prob, 4),
            "words": words_data
        }
        segment_list.append(seg_dict)
        full_text_parts.append(seg.text.strip())

    elapsed = round(time.time() - t_start, 2)
    full_transcript = " ".join(full_text_parts)

    return {
        "audio": os.path.basename(audio_path),
        "model": "large-v3",
        "language": language,
        "duration": round(info.duration, 2),
        "transcription_time_seconds": elapsed,
        "text": full_transcript,
        "segments": segment_list
    }


def main():
    parser = argparse.ArgumentParser(
        description="Spanish Whisper ASR Transcription & Model Setup Tool"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to an input audio file (.wav, .mp3, .flac) or directory of files."
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download and cache the Whisper large-v3 model weights, then exit."
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model name (default: large-v3)."
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default=None,
        help="Execution device (default: auto-detected)."
    )
    parser.add_argument(
        "--compute-type",
        choices=["float16", "int8", "float32"],
        default=None,
        help="Quantization / compute type (default: float16 for CUDA, int8 for CPU)."
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save JSON transcript with timestamps."
    )
    parser.add_argument(
        "--output-txt",
        default=None,
        help="Optional path to save clean plain text transcript."
    )

    args = parser.parse_args()

    if args.download_only:
        print("Pre-downloading Spanish Whisper model weights...")
        _ = get_whisper_model(model_size=args.model, device=args.device, compute_type=args.compute_type)
        print("Model downloaded and cached successfully.")
        return

    if not args.input:
        parser.print_help()
        sys.exit(1)

    # Determine files to process
    if os.path.isdir(args.input):
        supported_exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
        files = [
            os.path.join(args.input, f) for f in sorted(os.listdir(args.input))
            if f.lower().endswith(supported_exts)
        ]
        if not files:
            print(f"No audio files found in directory: {args.input}")
            sys.exit(1)
    else:
        files = [args.input]

    model = get_whisper_model(model_size=args.model, device=args.device, compute_type=args.compute_type)

    for audio_file in files:
        print(f"\nTranscribing: {audio_file}...")
        result = transcribe_audio(model, audio_file)

        print("\n--- TRANSCRIPTION RESULT ---")
        print(result["text"])
        print(f"Duration: {result['duration']}s | Processing time: {result['transcription_time_seconds']}s")

        # Save files
        base_name, _ = os.path.splitext(audio_file)
        json_path = args.output_json or f"{base_name}_whisper.json"
        txt_path = args.output_txt or f"{base_name}_whisper.txt"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["text"] + "\n")

        print(f"Saved JSON: {json_path}")
        print(f"Saved TXT:  {txt_path}")


if __name__ == "__main__":
    main()
