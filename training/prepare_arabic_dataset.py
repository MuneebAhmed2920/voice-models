"""Arabic Audio & Transcript Dataset Preparation for OmniVoice Fine-Tuning.

Tasks:
1. Validates and resamples raw Arabic audio files to 24000 Hz mono WAV.
2. Normalizes Arabic transcripts (numbers, currencies, dates) and adds Tashkeel.
3. Generates JSONL manifests for training and validation.
4. Optional: extracts discrete acoustic tokens using OmniVoice discrete tokenizer.
"""

import argparse
import json
import os
from pathlib import Path
import random
from typing import Dict, List

import numpy as np
import soundfile as sf

from src.arabic_nlp.diacritizer import ArabicDiacritizer
from src.arabic_nlp.normalizer import ArabicNormalizer


def process_audio_file(src_path: Path, dest_path: Path, target_sr: int = 24000) -> float:
    """Reads audio, converts to mono, resamples to target_sr, and saves as 16-bit WAV."""
    import librosa
    audio, sr = librosa.load(str(src_path), sr=target_sr, mono=True)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest_path), audio, target_sr, subtype="PCM_16")
    duration = len(audio) / target_sr
    return duration


def prepare_dataset(
    raw_dir: str,
    transcripts_file: str,
    output_dir: str,
    val_split: float = 0.05,
    dialect: str = "msa",
    add_tashkeel: bool = True
):
    """
    Prepares an Arabic speech dataset for OmniVoice training.
    
    Args:
        raw_dir: Directory containing input audio files (.wav, .mp3, .flac)
        transcripts_file: JSON or TSV file mapping filename to Arabic transcript
        output_dir: Destination folder for normalized dataset & manifests
        val_split: Fraction of dataset reserved for validation (default: 5%)
        dialect: 'msa' or 'egyptian'
        add_tashkeel: Whether to automatically diacritize transcripts
    """
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    wavs_out = out_path / "wavs_24k"
    wavs_out.mkdir(parents=True, exist_ok=True)

    normalizer = ArabicNormalizer(dialect=dialect)
    diacritizer = ArabicDiacritizer()

    # Load transcripts
    transcripts: Dict[str, str] = {}
    if transcripts_file.endswith(".json"):
        with open(transcripts_file, "r", encoding="utf-8") as f:
            transcripts = json.load(f)
    elif transcripts_file.endswith((".tsv", ".csv", ".txt")):
        delimiter = "\t" if transcripts_file.endswith(".tsv") else ("|" if "|" in open(transcripts_file, "r", encoding="utf-8").readline() else ",")
        with open(transcripts_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(delimiter)
                if len(parts) >= 2:
                    transcripts[parts[0].strip()] = parts[1].strip()

    print(f"[PrepareDataset] Loaded {len(transcripts)} transcripts.")

    records = []
    total_duration = 0.0

    for fname, raw_text in transcripts.items():
        # Find audio file
        audio_src = raw_path / fname
        if not audio_src.exists():
            for ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
                candidate = raw_path / f"{Path(fname).stem}{ext}"
                if candidate.exists():
                    audio_src = candidate
                    break

        if not audio_src.exists():
            print(f"  ⚠️ Warning: Audio file not found for {fname}, skipping.")
            continue

        # Normalize text
        norm_text = normalizer.normalize(raw_text)
        if add_tashkeel and dialect == "msa":
            final_text = diacritizer.diacritize(norm_text)
        else:
            final_text = norm_text

        # Resample & save audio
        dest_wav = wavs_out / f"{audio_src.stem}.wav"
        try:
            dur = process_audio_file(audio_src, dest_wav, target_sr=24000)
            if dur < 0.5 or dur > 30.0:  # Skip degenerate or overly long files
                continue
            total_duration += dur

            records.append({
                "audio_file": str(dest_wav.relative_to(out_path)),
                "raw_text": raw_text,
                "text": final_text,
                "duration": round(dur, 2),
                "dialect": dialect
            })
        except Exception as e:
            print(f"  ❌ Error processing {audio_src.name}: {e}")

    # Shuffle and split train / val
    random.seed(42)
    random.shuffle(records)
    n_val = max(1, int(len(records) * val_split))
    val_records = records[:n_val]
    train_records = records[n_val:]

    train_manifest = out_path / "train_manifest.jsonl"
    val_manifest = out_path / "val_manifest.jsonl"

    with open(train_manifest, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_manifest, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ Dataset Preparation Complete!")
    print(f"  • Total Audio Duration: {total_duration / 3600:.2f} hours ({len(records)} utterances)")
    print(f"  • Training Samples:     {len(train_records)} -> {train_manifest}")
    print(f"  • Validation Samples:   {len(val_records)} -> {val_manifest}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Arabic Audio Dataset for OmniVoice Training")
    parser.add_argument("--raw-dir", "-i", type=str, required=True, help="Path to raw audio folder")
    parser.add_argument("--transcripts", "-t", type=str, required=True, help="Path to transcript file (JSON or TSV)")
    parser.add_argument("--output-dir", "-o", type=str, default="data/processed_arabic", help="Output directory")
    parser.add_argument("--dialect", "-d", type=str, default="msa", choices=["msa", "egyptian"], help="Dialect")
    parser.add_argument("--val-split", type=float, default=0.05, help="Validation ratio (default: 0.05)")

    args = parser.parse_args()
    prepare_dataset(
        raw_dir=args.raw_dir,
        transcripts_file=args.transcripts,
        output_dir=args.output_dir,
        val_split=args.val_split,
        dialect=args.dialect
    )


if __name__ == "__main__":
    main()
