#!/usr/bin/env python3
"""
generate_spanish_recording.py

Generates the perfect, humanized Spanish customer service recording for the website suite:
- Voice: Mahrukh Dual-Native Voice Identity adapted for Spanish
- Language: Spanish (es)
- Pacing: Humanized sentence-by-sentence segmented synthesis with natural breathing pauses (260-280ms)
- Audio Engineering:
  * 24kHz Studio HD Master (WAV)
  * 8kHz PSTN Telephony Filter (WAV)
  * 192kbps Web-Optimized High-Bitrate (MP3)
  * 128kbps Telephony (MP3)
- Validation: Whisper ASR Spanish transcription
- Sync: Updates metadata_manifest.json and copies MP3s to Outreach System static demos
"""

import os
import sys
import json
import subprocess
import torch
import numpy as np
import scipy.signal
import soundfile as sf
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from transformers import pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_DIR = os.path.join(BASE_DIR, "website_recordings_bundle")
WAV_24K_DIR = os.path.join(BUNDLE_DIR, "wav_24k_hd")
TELEPHONY_8K_DIR = os.path.join(BUNDLE_DIR, "telephony_8k")
MP3_DIR = os.path.join(BUNDLE_DIR, "mp3")

OUTREACH_STATIC_DIR = "/Users/haseebahmed/agent/Outreach System/backend/static/audio/demos"

PROMPT_PATH = "/Users/haseebahmed/agent/Spanish Voice model/spanish_voice/voices/female/prompt.pt"

# Warm, energetic, highly humanized Spanish customer service dialogue
SPANISH_SENTENCES = [
    {
        "text": "¡Hola! Qué gusto saludarle. Le doy la más cordial bienvenida a nuestro servicio de atención al cliente. Mi nombre es Sofía, su asistente de voz personal. ¿En qué le puedo colaborar el día de hoy?",
        "speed": 0.92,
        "pause_ms": 320
    },
    {
        "text": "¡Con muchísimo gusto! Permítame tan solo un momento mientras consulto los detalles de su cuenta en nuestro sistema.",
        "speed": 0.92,
        "pause_ms": 280
    },
    {
        "text": "¡Excelente noticia! Le confirmo que su pago ha sido procesado exitosamente y su pedido se encuentra completamente activo y confirmado.",
        "speed": 0.92,
        "pause_ms": 300
    },
    {
        "text": "Nuestro equipo ya preparó su paquete con mucho cuidado, y acaba de ser despachado directamente desde el centro de distribución principal.",
        "speed": 0.92,
        "pause_ms": 320
    },
    {
        "text": "Su entrega está programada para llegar mañana por la tarde, y ya le enviamos el enlace con el rastreo en vivo a su teléfono móvil. ¡Fue un verdadero placer atenderle, que tenga un día maravilloso!",
        "speed": 0.92,
        "pause_ms": 100
    }
]

def apply_studio_mastering(audio: np.ndarray, sr: int = 24000) -> np.ndarray:
    nyq = 0.5 * sr
    b, a = scipy.signal.butter(2, 50.0 / nyq, btype='high')
    cleaned = scipy.signal.lfilter(b, a, audio)
    peak = np.max(np.abs(cleaned))
    if peak > 0:
        cleaned = (cleaned / peak) * 0.90
    return cleaned.astype(np.float32)

def apply_telephony_filter(audio: np.ndarray, orig_sr: int = 24000):
    import torchaudio.functional as F
    nyq = 0.5 * orig_sr
    b, a = scipy.signal.butter(4, [300.0 / nyq, 3400.0 / nyq], btype='band')
    filtered = scipy.signal.lfilter(b, a, audio)
    
    tensor_data = torch.from_numpy(filtered.astype(np.float32)).unsqueeze(0)
    resampled_8k = F.resample(tensor_data, orig_freq=orig_sr, new_freq=8000).squeeze(0).numpy()
    
    peak = np.max(np.abs(resampled_8k))
    if peak > 0:
        resampled_8k = (resampled_8k / peak) * 0.92
    return resampled_8k.astype(np.float32), 8000

def convert_to_mp3(wav_path: str, mp3_path: str, bitrate: str = "192k"):
    cmd = [
        "ffmpeg", "-y",
        "-i", wav_path,
        "-codec:a", "libmp3lame",
        "-b:a", bitrate,
        "-ar", "24000" if "24k" in wav_path or "master" in wav_path else "8000",
        mp3_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def main():
    print("=" * 75)
    print("GENERATING PERFECT HUMANIZED SPANISH CUSTOMER SERVICE RECORDING")
    print("=" * 75)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading OmniVoice on {device}...")
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device)
    cfg = OmniVoiceGenerationConfig(num_step=24, guidance_scale=2.2, denoise=True, postprocess_output=True)

    print("Loading native energetic Colombian Spanish voice prompt...")
    from omnivoice.models.omnivoice import VoiceClonePrompt
    prompt = VoiceClonePrompt.load(PROMPT_PATH)

    print("Synthesizing humanized Spanish speech segments...")
    audio_segments = []
    text_transcripts = []

    for idx, item in enumerate(SPANISH_SENTENCES, 1):
        s_text = item["text"]
        s_speed = item["speed"]
        p_ms = item["pause_ms"]

        print(f"   [{idx}/{len(SPANISH_SENTENCES)}] {s_text[:50]}...")
        torch.manual_seed(2024 + idx * 43)
        aud = model.generate(
            text=s_text,
            language="es",
            voice_clone_prompt=prompt,
            generation_config=cfg,
            speed=s_speed
        )[0]

        audio_segments.append(aud)
        if idx < len(SPANISH_SENTENCES):
            pause_samples = int(24000 * (p_ms / 1000.0))
            audio_segments.append(np.zeros(pause_samples, dtype=np.float32))

        text_transcripts.append(s_text)

    full_audio = np.concatenate(audio_segments)

    # 1. 24kHz Studio HD Master
    master_24k = apply_studio_mastering(full_audio, 24000)
    dur_sec = len(master_24k) / 24000.0
    wav_24k_path = os.path.join(WAV_24K_DIR, "rec_05_spanish_customer_service_24k.wav")
    sf.write(wav_24k_path, master_24k, 24000)
    print(f"-> Saved 24kHz Studio HD Master: {wav_24k_path} ({dur_sec:.2f}s)")

    # 2. 8kHz Telephony Filter
    telephony_8k, sr_8k = apply_telephony_filter(master_24k, 24000)
    dur_8k_sec = len(telephony_8k) / float(sr_8k)
    wav_8k_path = os.path.join(TELEPHONY_8K_DIR, "rec_05_spanish_customer_service_8k_telephony.wav")
    sf.write(wav_8k_path, telephony_8k, sr_8k)
    print(f"-> Saved 8kHz Telephony PSTN: {wav_8k_path} ({dur_8k_sec:.2f}s)")

    # 3. Web MP3s
    mp3_24k_path = os.path.join(MP3_DIR, "rec_05_spanish_customer_service.mp3")
    mp3_8k_path = os.path.join(MP3_DIR, "rec_05_spanish_customer_service_telephony.mp3")
    convert_to_mp3(wav_24k_path, mp3_24k_path, "192k")
    convert_to_mp3(wav_8k_path, mp3_8k_path, "128k")
    print(f"-> Encoded Web MP3: {mp3_24k_path} (192kbps) & {mp3_8k_path} (128kbps)")

    # 4. Whisper ASR Validation
    print("Running Whisper ASR transcription check...")
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device="cpu")
    whisper_res = asr(wav_24k_path, generate_kwargs={"language": "es", "task": "transcribe"}, return_timestamps=True)
    whisper_text = whisper_res["text"]
    print(f"-> Whisper ASR Transcription: {whisper_text}")

    # Copy to Outreach System if folder exists
    if os.path.exists(OUTREACH_STATIC_DIR):
        dest_mp3 = os.path.join(OUTREACH_STATIC_DIR, "rec_05_spanish_customer_service.mp3")
        dest_tel = os.path.join(OUTREACH_STATIC_DIR, "rec_05_spanish_customer_service_telephony.mp3")
        subprocess.run(["cp", "-v", mp3_24k_path, dest_mp3], check=True)
        subprocess.run(["cp", "-v", mp3_8k_path, dest_tel], check=True)
        print(f"-> Synced MP3s to {OUTREACH_STATIC_DIR}")

    # 5. Update Manifest JSON
    manifest_path = os.path.join(BUNDLE_DIR, "metadata_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = []

    # Check if entry already exists, update or append
    entry = {
        "id": "rec_05_spanish_customer_service",
        "title": "Spanish Enterprise Customer Support & Order Tracking",
        "language_label": "Spanish (es / Español Internacional)",
        "language_code": "es",
        "industry": "Customer Care & Global Logistics",
        "description": "Polite and warm Spanish customer service representative verifying client identity, payment confirmation, and delivery scheduling.",
        "duration_seconds": round(dur_sec, 2),
        "duration_formatted": f"{int(dur_sec)}s",
        "full_transcript": "\n".join(text_transcripts),
        "whisper_asr_verification": whisper_text,
        "files": {
            "web_mp3": os.path.relpath(mp3_24k_path, BASE_DIR),
            "web_mp3_telephony": os.path.relpath(mp3_8k_path, BASE_DIR),
            "wav_24k_hd": os.path.relpath(wav_24k_path, BASE_DIR),
            "wav_8k_telephony": os.path.relpath(wav_8k_path, BASE_DIR)
        }
    }

    manifest = [m for m in manifest if m["id"] != "rec_05_spanish_customer_service"]
    manifest.append(entry)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"-> Updated manifest JSON: {manifest_path}")

    print("=" * 75)
    print("SPANISH MASTER RECORDING COMPLETED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    main()
