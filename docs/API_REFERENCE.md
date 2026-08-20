# 📖 OmniVoice TTS Engine API Reference

Comprehensive reference for the REST, Streaming WebSocket, and Asterisk AudioSocket interfaces.

---

## 🌐 1. REST API Reference

Base URL: `http://localhost:8000`

### A. Telephony Synthesis (`POST /v1/tts/synthesize`)
Generates speech audio formatted for telephony and voice applications.

#### Request Body (JSON):
```json
{
  "text": "[excited] Thank you for calling Clara support! Your total balance is $45.50 on 08/24/2026.",
  "voice": "voice_clara",
  "emotion": "excited",
  "speed": 0.92,
  "sample_rate": 8000,
  "format": "pcm16",
  "ode_steps": 16,
  "guidance_scale": 2.2,
  "normalize": true,
  "add_breath": false,
  "stream": false
}
```

#### Parameters:
| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `string` | **Required** | Text to speak. Supports emotion tags (`[excited]`, `[sympathetic]`, `[emotional]`, `[breath]`). |
| `voice` | `string` | `"voice_clara"` | Voice ID or alias (`voice_clara`, `clara`, `female_1`). |
| `emotion` | `string` | `null` | Style override: `neutral`, `excited`, `sympathetic`, `emotional`. |
| `speed` | `float` | `0.92` | Speaking cadence speed multiplier (`0.5` to `2.0`). |
| `sample_rate` | `int` | `8000` | Sample rate in Hz: `8000` (PSTN), `16000` (HD), `24000` (Native). |
| `format` | `string` | `"pcm16"` | Audio format: `pcm16`, `mulaw`, `alaw`, `wav`. |
| `ode_steps` | `int` | `16` | Flow-matching ODE steps (`16` for real-time `<150ms`, `32` for studio). |
| `normalize` | `bool` | `true` | Expands currency (`$45.50`), numbers, dates, times, phone numbers. |
| `add_breath` | `bool` | `false` | Prepends organic airway inhalation sound. |
| `stream` | `bool` | `false` | When true, streams audio in 20ms telephony chunks. |

#### Response:
- Binary audio stream / bytes with custom latency headers:
  - `X-TTS-TTFA-MS`: Time-to-first-audio in milliseconds.
  - `X-TTS-Inference-MS`: Total inference compute time.
  - `X-TTS-Audio-Duration`: Duration of synthesized audio in seconds.
  - `X-TTS-RTF`: Real-Time-Factor (`inference_sec / audio_sec`).
  - `X-TTS-Sample-Rate`: Output sampling rate.
  - `X-TTS-Format`: Audio codec format.

---

### B. OpenAI Speech Compatibility (`POST /v1/audio/speech`)
Drop-in replacement for OpenAI TTS API.

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "omnivoice",
    "input": "Hello from Clara customer representative!",
    "voice": "voice_clara",
    "response_format": "wav",
    "speed": 0.92
  }' \
  --output speech.wav
```

---

### C. System Health (`GET /health`)
```json
{
  "status": "ready",
  "service": "OmniVoice TTS Engine",
  "device": "cuda:0",
  "dtype": "torch.float16",
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "vram_used_mb": 1420.5,
  "vram_total_mb": 24564.0,
  "voices": [...]
}
```

---

## ⚡ 2. WebSocket Real-Time Streaming Protocol

Endpoint: `ws://localhost:8000/ws/tts`

### Flow Overview:
1. **Client sends `start`**:
   ```json
   {"event": "start", "voice": "voice_clara", "sample_rate": 8000, "format": "pcm16", "ode_steps": 16}
   ```
2. **Server replies `session_started`**:
   ```json
   {"event": "session_started", "session_id": "3b2e..."}
   ```
3. **Client streams LLM tokens as they arrive**:
   ```json
   {"event": "text_chunk", "text": "Yes, absolutely! "}
   {"event": "text_chunk", "text": "I can help you with your account today."}
   ```
4. **Server streams audio**:
   - `{"event": "audio_start", "clause_index": 0, "text": "Yes, absolutely!", "ttfa_ms": 118.4}`
   - Binary 20ms PCM16 frames (320 bytes each @ 8kHz)
   - `{"event": "audio_end", "clause_index": 0, "duration_sec": 1.25, "rtf": 0.09}`
5. **Caller Interruption (Barge-In)**:
   - Client sends: `{"event": "interrupt"}`
   - Server immediately terminates current audio output and replies `{"event": "interrupted"}`.

---

## ☎️ 3. Asterisk AudioSocket Protocol (Port 9092)

TCP Port: `9092`

Asterisk Dialplan:
```ini
exten => 8300,1,Answer()
 same => n,Set(CHANNEL(audioreadformat)=slin)
 same => n,Set(CHANNEL(audiowriteformat)=slin)
 same => n,AudioSocket(${UUID},127.0.0.1:9092)
 same => n,Hangup()
```

Frames sent/received follow the Asterisk AudioSocket specification (3-byte header + 320-byte PCM16 payload per 20ms frame).
