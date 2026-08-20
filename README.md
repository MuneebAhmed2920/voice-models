# 🎙️ Clara Voice Model & Production OmniVoice TTS Engine
### Ultra-Low-Latency Real-Time Telephony TTS for Vicidial & AI Voice Bots (Windows NVIDIA CUDA FP16 / Linux)

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20FP16-ee4c2c)](https://pytorch.org/)
[![Vicidial](https://img.shields.io/badge/Vicidial-Asterisk%20AudioSocket-green)](https://www.vicidial.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## 🌟 Overview

This repository packages the **Clara Voice Model** (`voice_clara`) and a high-performance **Production Real-Time OmniVoice TTS Service** engineered specifically for **AI Voice Bots** connected to **Vicidial Call Centers** and **Asterisk PBX telephony networks**.

### Key Highlights:
- **Calibrated Voice Model (`voice_clara`)**: Warm, friendly, calm, and articulate American English female customer service representative with natural telephone cadence (default speed: `0.92`, guidance scale: `2.2`).
- **Precomputed Emotional Prompt Tensors**: Zero-latency runtime switching between `neutral`, `excited`, `sympathetic`, and `emotional` tone styles.
- **Organic Physiological Inhalation Breathing**: Integrated airway breath sounds (`prompt/breath_organic.wav`) for human realism on phone calls.
- **NVIDIA CUDA FP16 & Tensor Core Acceleration**: Native half-precision inference with Tensor Cores (`< 150ms` Time-To-First-Audio on GPU) and automatic CPU fallback.
- **Telephony Audio Pipeline**: Native 8,000 Hz PSTN resampling, 16,000 Hz HD Voice, Linear PCM16 framing (20ms frames / 320 bytes), G.711 μ-law, and G.711 a-law encoding.
- **Telephony Text Normalization**: Automated expansion of currencies (`$45.50`), dates (`08/24/2026`), times (`3:30 PM`), phone numbers, percentages, and addresses.
- **Bidirectional Streaming WebSocket (`/ws/tts`)**: Speculative clause boundary buffering for streaming LLM tokens with **instant caller barge-in cancellation**.
- **Asterisk AudioSocket TCP Bridge (Port 9092)**: Direct connection to Asterisk dialplans (`exten => 8300,1,AudioSocket(...)`).
- **OpenAI TTS API Compatibility (`/v1/audio/speech`)**: Drop-in replacement for existing voice bot frameworks.

---

## 📁 Repository Layout

```
clara-voice-model/
├── runtime/                        # Core Inference Engine
│   ├── engine.py                   # OmniVoiceEngine (CUDA FP16 / CPU singleton)
│   ├── normalizer.py               # Telephony text normalizer & emotion tag parser
│   ├── audio_processor.py          # Resampling (8k/16k/24k), PCM16/μ-law/a-law, 20ms framing
│   ├── streamer.py                 # Real-time clause token streamer & barge-in handler
│   ├── voice_manager.py            # Multi-voice & emotional prompt registry
│   └── __init__.py
├── server/                         # Network & Protocol Services
│   ├── app.py                      # FastAPI app factory with CORS & lifespan manager
│   ├── ws_endpoint.py              # Bidirectional WebSocket streaming protocol (/ws/tts)
│   ├── rest_endpoint.py            # REST endpoints (/v1/tts/synthesize, /v1/audio/speech, /health)
│   ├── audiosocket_server.py       # Native Asterisk AudioSocket TCP server (port 9092)
│   ├── schemas.py                  # Pydantic v2 data schemas
│   └── __init__.py
├── bridges/                        # Telephony Integration Bridges
│   ├── vicidial_bridge.py          # Asterisk dialplan generator & health checker
│   ├── mock_bot.py                 # Mock LLM Voice Bot simulating streaming & barge-in
│   └── __init__.py
├── prompt/                         # Calibrated Voice Prompts & Organic Breath Waveforms
│   ├── voice_prompt.pt             # Neutral baseline prompt tensor
│   ├── voice_prompt_excited.pt     # Excited / celebratory prompt tensor
│   ├── voice_prompt_sympathetic.pt # Sympathetic / apologetic prompt tensor
│   ├── voice_prompt_emotional.pt   # Emotional / warm prompt tensor
│   ├── breath_organic.wav          # Extracted physiological inhalation breath
│   └── reference.wav               # Master studio reference recording
├── config/                         # System Configurations
│   ├── config.json                 # Voice hyperparams (speed: 0.92, guidance: 2.2)
│   ├── metadata.json               # Provenance and Apache-2.0 license
│   └── server_config.yaml          # Host, port, precision, sample rate, ODE steps
├── benchmarks/                     # Latency & Concurrency Benchmarks
│   └── latency_benchmark.py        # Measures TTFA, RTF, memory, and call concurrency
├── tests/                          # Test Suite
│   ├── test_voice.py               # Package structure test
│   ├── test_normalizer.py          # Text normalizer test
│   ├── test_audio.py               # Audio processing test
│   ├── test_engine.py              # Engine synthesis test
│   ├── test_server.py              # FastAPI REST test
│   ├── test_websocket.py           # WebSocket & barge-in test
│   └── test_audiosocket.py         # Asterisk AudioSocket test
├── samples/                        # Generated Evaluation Benchmark Samples
│   └── generate_telephony_samples.py # Generates 8kHz PSTN audio samples
├── docs/                           # Integration Guides
│   ├── VICIDIAL_INTEGRATION.md     # Vicidial & Asterisk dialplan setup guide
│   ├── WINDOWS_CUDA_GUIDE.md       # Windows NVIDIA CUDA FP16 setup guide
│   └── API_REFERENCE.md            # Complete REST, WS, and AudioSocket documentation
├── start_server.bat                # Windows native startup script with CUDA check
├── start_server.sh                 # Linux / macOS startup script
├── Dockerfile                      # Production NVIDIA CUDA 12 Docker container
├── docker-compose.yml              # Docker Compose with GPU passthrough
├── requirements.txt                # Pinned production dependencies
└── pyproject.toml                  # Python package build configuration
```

---

## ⚡ Quick Start

### 1. Windows Native (NVIDIA CUDA FP16)
```cmd
:: 1. Clone repository
git clone https://github.com/MuneebAhmed2920/clara-voice-model.git
cd clara-voice-model

:: 2. Create virtual environment & install PyTorch with CUDA
python -m venv .venv
call .venv\Scripts\activate.bat
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

:: 3. Launch Server
start_server.bat
```

### 2. Linux / macOS
```bash
./start_server.sh
```

### 3. Docker (with NVIDIA GPU Passthrough)
```bash
docker compose up --build
```

The server will start and listen on:
- **REST & WebSocket API**: `http://localhost:8000` (Swagger UI at `http://localhost:8000/docs`)
- **Asterisk AudioSocket TCP**: `0.0.0.0:9092`

---

## 🚀 Telephony Usage Examples

### 1. Python API
```python
from runtime import get_engine

engine = get_engine(default_ode_steps=16)

# Generate 8kHz PSTN audio with emotion tag and currency expansion
audio_bytes, metrics = engine.synthesize(
    text="[excited] Thank you for calling Clara support! Your total refund of $45.50 was approved on 08/24/2026.",
    voice="voice_clara",
    sample_rate=8000,
    format_type="pcm16"
)

print(f"Generated {len(audio_bytes)} bytes in {metrics['ttfa_ms']}ms (RTF: {metrics['rtf']})")
```

### 2. REST API Telephony Synthesis
```bash
curl -X POST http://localhost:8000/v1/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "[sympathetic] I understand your concern. Please call us at (800) 555-0199.",
    "voice": "voice_clara",
    "sample_rate": 8000,
    "format": "pcm16",
    "ode_steps": 16
  }' \
  --output caller_audio.pcm
```

### 3. Real-Time Streaming WebSocket (`/ws/tts`) with Barge-In
```python
import asyncio
import json
import websockets

async def stream_to_caller():
    async with websockets.connect("ws://localhost:8000/ws/tts") as ws:
        # Start session
        await ws.send(json.dumps({"event": "start", "voice": "voice_clara", "sample_rate": 8000}))
        _ = await ws.recv()

        # Stream tokens as they arrive from LLM
        tokens = ["Yes, ", "absolutely! ", "I can ", "help you with that."]
        for tok in tokens:
            await ws.send(json.dumps({"event": "text_chunk", "text": tok}))
        
        await ws.send(json.dumps({"event": "flush"}))

asyncio.run(stream_to_caller())
```

---

## 📞 Vicidial & Asterisk Integration

1. Add the following to Asterisk `/etc/asterisk/extensions.conf`:
   ```ini
   [vicidial-ai-bot]
   exten => 8300,1,Answer()
    same => n,Set(CHANNEL(audioreadformat)=slin)
    same => n,Set(CHANNEL(audiowriteformat)=slin)
    same => n,AudioSocket(${UUID},127.0.0.1:9092)
    same => n,Hangup()
   ```
2. In Vicidial Admin, create a **Remote Agent** pointing to external extension `8300`.
3. Calls connected via predictive dialing are streamed directly to the AI Voice Bot and OmniVoice in real time.
4. See [docs/VICIDIAL_INTEGRATION.md](docs/VICIDIAL_INTEGRATION.md) for full dialplans, conference bridging, and jitter buffer configuration.

---

## 🧪 Testing & Verification

Run the full test suite:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Run latency benchmark:
```bash
python benchmarks/latency_benchmark.py
```

Run Mock Voice Bot streaming & barge-in simulation:
```bash
python bridges/mock_bot.py
```

---

## 📜 License & Provenance
The Clara Voice Model and OmniVoice runtime service are distributed under the **Apache-2.0 License**. Commercial use, redistribution, and telephony deployment are fully permitted.
