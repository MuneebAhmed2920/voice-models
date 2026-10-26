# 🎙️ OmniVoice Production TTS Engine

A clean, minimal, production-focused English Text-to-Speech (TTS) engine built on `k2-fsa/OmniVoice`. Designed specifically for low-latency, human-like voice synthesis in real-time customer service, AI voicebots, and telephony applications.

---

## 🌟 Key Features

* **English-Only Focused:** 100% deterministic text normalization for currency (`$129.50`), dates (`August 24th`), times (`3:30 PM`), phone numbers (`310-555-0182`), addresses, and acronyms.
* **4 Fixed Production Personas:** Precomputed in-RAM voice prompts for instant O(1) retrieval:
  * `female_1` (Dr. Sara) — Articulate, friendly, clear customer support
  * `female_2` (Emma) — Calm, warm, reassuring healthcare/finance
  * `male_1` (Dr. Alex) — Confident, resonant baritone technical specialist
  * `male_2` (Marcus) — Conversational, energetic appointment reminder
* **Dedicated Saved Voice Module:** Includes `saved_female_voice/` for standalone single-voice deployments.
* **Sub-Second TTFA on CPU:** Dynamic step scaling (4 steps for 1-word acks, 8 steps for phrases, 12 steps for sentences).
* **Conversational Gestures:** Built-in XML tag support for `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<breath>`, `<hesitate>`.
* **Zero Runtime Cloning Overhead:** Fixed voices are precomputed and frozen in RAM at startup.
* **Telephony Multi-Format Audio:** Native 24 kHz, 16 kHz, and 8 kHz ITU-T G.711 μ-law codecs.

---

## 📦 Quick Start

### 1. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 2. Synthesize Speech via CLI:
```bash
# Generate audio with Female 1 (Dr. Sara)
python3 generate.py "Hello! Your total balance of $129.50 is due on August 24th at 3:30 PM." --voice female_1 -o balance.wav

# Play aloud through speakers (-p flag)
python3 generate.py "Welcome to our customer service line!" --voice male_1 -p

# Interactive Live Console
python3 generate.py --interactive
```

---

## 💻 Python Library Usage

```python
from omnivoice_tts import get_engine

# Initialize engine (loads model and 4 precomputed voices once into RAM)
engine = get_engine(device="cpu")

# Instant synthesis
audio_bytes, metrics = engine.generate(
    text="<breath> Hello! Your appointment is scheduled for Monday at 3 PM. Please call 310-555-0182.",
    voice_id="female_1",    # 'female_1' | 'female_2' | 'male_1' | 'male_2'
    quality="fast",         # 'fast' (dynamic steps) | 'quality' (16 steps)
    sample_rate=24000,      # 24000 | 16000 | 8000
    output_format="wav"     # 'wav' | 'pcm16' | 'mulaw'
)

with open("appointment.wav", "wb") as f:
    f.write(audio_bytes)

print(f"Generated in {metrics['ttfa_sec']:.2f}s (Speed: {metrics['speed_x']:.2f}x real-time)")
```

---

## 🌐 Production REST API Server

### 1. Start Server:
```bash
python3 -m uvicorn omnivoice_tts.api.server:app --host 0.0.0.0 --port 8000
```

### 2. Request Synthesis (`POST /tts`):
```bash
curl -X POST "http://localhost:8000/tts" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Your total balance is $129.50. Please call 310-555-0182.",
       "voice_id": "female_1",
       "output_format": "wav",
       "sample_rate": 24000
     }' \
     --output balance.wav
```

---

## 📊 Latency & Performance Breakdown

| Stage | Operation | Latency (CPU) | Latency (GPU) |
| :--- | :--- | :---: | :---: |
| **1. Text Normalization** | Entity expansion (`$129.50` ➔ *"one hundred twenty-nine dollars..."*) | **`< 0.5 ms`** | **`< 0.5 ms`** |
| **2. Gesture Parsing** | Non-verbal tag sequencing (`<laugh>`, `<breath>`) | **`< 0.2 ms`** | **`< 0.2 ms`** |
| **3. Voice Retrieval** | Zero-overhead in-RAM lookup | **`0.0 ms`** | **`0.0 ms`** |
| **4. Flow Matching** | Iterative codebook diffusion (4–12 steps) | **`300 ms – 1.2s`** | **`35 ms – 70 ms`** |
| **5. Neural Codec** | Multi-codebook 24kHz decode | **`18 ms – 35 ms`** | **`2 ms – 5 ms`** |
| **6. Format Encode** | WAV / G.711 μ-law byte stream | **`< 2.0 ms`** | **`< 1.0 ms`** |
| **Total Real-Time TTFA** | **End-to-End Time to First Audio** | **`~0.8s – 1.8s`** | **`~40 ms – 80 ms`** |

---

## 🎧 High-Fidelity Audio Showcase Samples

The `samples/` directory provides production English enterprise voice models synthesized with high naturalness and crystal-clear pronunciation:

| Sample Asset | Format | Sample Rate | Description |
| :--- | :---: | :---: | :--- |
| `samples/rec_02_english_customer_service.mp3` | MP3 (192kbps) | 24,000 Hz | Enterprise Customer Service Account Verification & Delivery Scheduling (28.04s) |
| `samples/rec_02_english_customer_service_telephony.mp3` | MP3 (128kbps) | 8,000 Hz | PSTN / Telephony bandpass filtered phone line simulation (28.04s) |
| `samples/rec_02_english_customer_service_24k.wav` | Linear PCM WAV | 24,000 Hz | Studio Master HD Recording (-1.0 dBFS Peak Normalized) |
| `samples/rec_02_english_customer_service_8k_telephony.wav` | Linear PCM WAV | 8,000 Hz | Asterisk / PBX / G.711 Telephony Master |

