# 🎙️ OmniVoice TTS: Minimal Production English Speech Synthesis Engine

A clean, production-focused English Text-to-Speech (TTS) engine built on `k2-fsa/OmniVoice`. Designed specifically for low-latency, human-like voice synthesis in real-time customer service and telephonic applications.

---

## 🚀 Key Features

* **English-Only Optimized:** 100% deterministic text normalization for currency (`$129.50`), dates (`August 24th`), times (`3:30 PM`), phone numbers (`310-555-0182`), and acronyms.
* **4 Fixed Production Personas:** Precomputed in-RAM voice prompts for instant O(1) retrieval:
  * `female_1` (Dr. Sara) — Articulate, friendly, clear customer support
  * `female_2` (Emma) — Calm, warm, reassuring healthcare/finance
  * `male_1` (Dr. Alex) — Confident, resonant baritone technical specialist
  * `male_2` (Marcus) — Conversational, energetic appointment reminder
* **Ultra-Low Latency Flow Matching:** Dynamic step scaling (4 steps for 1-word acks, 8 steps for phrases, 12 steps for sentences).
* **Conversational Gestures:** Built-in XML tag support for `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<breath>`, `<hesitate>`.
* **Zero Runtime Cloning Overhead:** Fixed voices are precomputed and frozen in RAM at startup.
* **Telephony Multi-Format Audio:** Native 24 kHz, 16 kHz, and 8 kHz ITU-T G.711 μ-law codecs.

---

## 📁 Repository Structure

```
omnivoice_tts/
├── model/                  # OmniVoice neural speech backbone
├── voices/                 # 4 Precomputed voice tensors (.pt) & registry
│   ├── female_1.pt
│   ├── female_2.pt
│   ├── male_1.pt
│   ├── male_2.pt
│   └── registry.py
├── text/                   # English normalization & gesture parsing
│   ├── normalizer.py
│   └── gestures.py
├── audio/                  # Resampling & G.711 mu-law codecs
│   └── codecs.py
├── engine.py               # Core Persistent OmniVoice Engine
├── api/                    # Minimal FastAPI REST API (POST /tts)
│   └── server.py
├── samples/                # Pre-generated 4-voice audio demonstrations
└── benchmarks/             # Latency, p50/p95, and memory leak suites
```

---

## 💻 Python Usage (Direct Local API)

```python
from omnivoice_tts import get_engine

# 1. Initialize engine (loads model and 4 precomputed voices once)
engine = get_engine(device="cpu")

# 2. Synthesize English text with currency & gestures
audio_bytes, metrics = engine.generate(
    text="<breath> Hello! Your total balance is $129.50 due on August 24th at 3:30 PM.",
    voice_id="female_1",    # 'female_1' | 'female_2' | 'male_1' | 'male_2'
    quality="fast",         # 'fast' (dynamic steps) | 'quality' (16 steps)
    sample_rate=24000,      # 24000 | 16000 | 8000
    output_format="wav"     # 'wav' | 'pcm16' | 'mulaw'
)

# 3. Save audio file
with open("output.wav", "wb") as f:
    f.write(audio_bytes)

print(f"Generated in {metrics['ttfa_sec']:.2f}s (Speed: {metrics['speed_x']:.2f}x real-time)")
```

---

## 🌐 Minimal REST API Server

### 1. Start Server:
```bash
python3 -m uvicorn omnivoice_tts.api.server:app --host 0.0.0.0 --port 8000
```

### 2. Request Synthesis (`POST /tts`):
```bash
curl -X POST "http://localhost:8000/tts" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Your appointment is scheduled for August 24th at 3:30 PM. Please call 310-555-0182.",
       "voice_id": "male_1",
       "output_format": "wav",
       "sample_rate": 24000
     }' \
     --output appointment.wav
```

---

## 📊 Latency & Performance Breakdown

| Stage | Operation | Latency (CPU) | Latency (GPU) |
| :--- | :--- | :---: | :---: |
| 1. Text Normalization | Entity expansion (`$129.50` ➔ words) | **`< 0.5 ms`** | **`< 0.5 ms`** |
| 2. Gesture Parsing | Non-verbal tag sequencing (`<laugh>`) | **`< 0.2 ms`** | **`< 0.2 ms`** |
| 3. Voice Retrieval | Zero-overhead in-RAM lookup | **`0.0 ms`** | **`0.0 ms`** |
| 4. Flow Matching | Iterative codebook diffusion (4–12 steps) | **`300 ms – 1.2s`** | **`35 ms – 70 ms`** |
| 5. Neural Codec | Multi-codebook 24kHz decode | **`18 ms – 35 ms`** | **`2 ms – 5 ms`** |
| 6. Format Encode | WAV / G.711 μ-law byte stream | **`< 2.0 ms`** | **`< 1.0 ms`** |
| **Total Real-Time TTFA** | **End-to-End Time to First Audio** | **`~0.8s – 1.8s`** | **`~40 ms – 80 ms`** |
