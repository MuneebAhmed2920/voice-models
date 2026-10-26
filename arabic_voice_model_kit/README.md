# 🎙️ Arabic Human-Realistic Conversational Voice Model Kit

Production-ready, ultra-realistic Arabic Voice Model Kit implementing the proven **7-Pillar Humanization Blueprint** originally engineered for the Clara Voice Model (`voice_clara`). 

Designed specifically for real-time conversational telephony AI agents, customer care bots, Asterisk PBX, Vicidial, and WebRTC streaming.

---

## 🌟 The 7 Humanization Pillars

| # | Pillar | Implementation in Arabic Kit | Effect on Speech |
|---|---|---|---|
| **1** | **Physiological Airway Inhalation** | Emotion-tailored breath prepend (`140ms` neutral, `95ms` excited, `210ms` sympathetic, `165ms` emotional) with raised-cosine crossfade. | Eliminates unnatural instant-onset speech; creates immediate human presence. |
| **2** | **Cadence & Speed Tuning** | Calibrated default `speed = 0.92x`. | Prevents rushed machine cadence; allows Arabic consonants (حروف الإطباق والحلق) to enunciate with natural warmth. |
| **3** | **Flow-Matching ODE Guidance** | Calibrated `guidance_scale = 2.2` & `class_temperature = 0.20`. | Prevents mechanical pitch rigidity; introduces organic micro-inflections in fundamental frequency ($F_0$). |
| **4** | **Multi-Emotion Prompt Tensors** | Precomputed ODE conditioning tensors for `neutral`, `excited` (`[متحمس]`), `sympathetic` (`[متعاطف]`), and `emotional` (`[مؤثر]`). | Adapts vocal energy, pitch range, and resonance seamlessly based on conversation sentiment. |
| **5** | **Studio Mastering & Proximity EQ** | **75 Hz HPF** (12 dB/oct rumble removal) + **280 Hz low-shelf** (+1.5 dB proximity warmth) + **4.0 kHz peaking filter** (+2.0 dB for emphatic/pharyngeal consonants `ح, ع, ص, ض, ط, ظ, ق`) + **soft-knee peak limiting**. | Studio telephone clarity without clipping or distortion on PSTN lines. |
| **6** | **Conversational Micro-Pauses** | Inter-clause comma pauses (`120ms`), period/question pauses (`280ms`), colon/dash pauses (`150ms`), and conjunction respiratory pauses (`و`, `فـ`, `لكن`, `يعني`, `طبعاً`). | Natural conversational respiratory rhythm. |
| **7** | **Deterministic Telephony Normalization** | Spoken expansion of currencies (`٤٥ ريال`, `١٢٠ جنيه`, `$1,250`), times (`٤:٣٠ مساءً`), dates (`15/09/2026`), phone numbers, and percentages (`95%`). | Reliable, error-free verbal articulation on phone calls. |

---

## 📁 Package Structure

```
arabic_voice_model_kit/
├── PROMPT_FOR_ANTIGRAVITY.md        # Master specification directive
├── config/
│   ├── config.json                  # Calibrated parameters (speed 0.92, guidance 2.2, sample_rate 24k)
│   ├── metadata.json                # Voice metadata, style description & license
│   └── server_config.yaml           # Telephony server configuration
├── prompt/
│   ├── reference.wav                # High-fidelity studio reference recording (24kHz)
│   ├── voice_prompt.pt              # Base neutral VoiceClonePrompt tensor
│   ├── voice_prompt_excited.pt      # Excited [متحمس] conditioning tensor
│   ├── voice_prompt_sympathetic.pt  # Sympathetic [متعاطف] conditioning tensor
│   ├── voice_prompt_emotional.pt    # Emotional [مؤثر] conditioning tensor
│   ├── breath_organic.wav           # Master 24kHz neutral airway breath (200ms)
│   ├── breath_excited.wav           # Short eager inhalation (100ms)
│   ├── breath_sympathetic.wav       # Deep soft inhalation (220ms)
│   ├── breath_emotional.wav         # Sincere resonant inhalation (160ms)
│   └── *_16k.wav / *_8k.wav         # Resampled telephony breath assets
├── runtime/
│   ├── __init__.py                  # Package exports
│   ├── arabic_breath_processor.py   # Emotion-tailored breathing & conjunction processor
│   ├── normalizer.py                # Deterministic Arabic telephony text normalizer
│   ├── audio_processor.py           # Proximity EQ chain, codecs (PCM16, μ-law, A-law, WAV), 20ms framing
│   ├── voice_manager.py             # Prompt tensor caching & emotion routing
│   ├── streamer.py                  # Punctuation micro-pauses & clause streaming
│   └── engine.py                    # Unified 7-pillar synthesis engine
├── server/
│   ├── __init__.py
│   ├── schemas.py                   # Pydantic request/response schemas
│   └── app.py                       # FastAPI & WebSocket streaming server
├── samples/
│   ├── benchmark_1_conversational_greeting_24k.wav  (and 8k telephony)
│   ├── benchmark_2_excited_confirmation_24k.wav      (and 8k telephony)
│   ├── benchmark_3_sympathetic_customer_care_24k.wav (and 8k telephony)
│   ├── benchmark_4_emotional_reassurance_24k.wav     (and 8k telephony)
│   └── benchmark_report.json        # Detailed audio metrics report
├── tests/
│   ├── test_arabic_breath_processor.py
│   ├── test_arabic_normalizer.py
│   ├── test_arabic_audio_processor.py
│   ├── test_arabic_voice_manager.py
│   ├── test_end_to_end.py
│   └── run_benchmarks.py            # Official Section 5 benchmark generator
└── README.md
```

---

## 🚀 Quickstart & Python Usage

### 1. Basic Synthesis (WAV / Studio 24kHz)

```python
from arabic_voice_model_kit.runtime.engine import get_engine

engine = get_engine()

# Synthesize conversational greeting with natural airway inhalation
audio_bytes, metrics = engine.synthesize(
    text="أهلاً وسهلاً بحضرتك، تفضل كيف أقدر أساعدك اليوم؟",
    voice="voice_arabic_clara",
    sample_rate=24000,
    format_type="wav",
    speed=0.92,
    guidance_scale=2.2
)

with open("greeting.wav", "wb") as f:
    f.write(audio_bytes)

print(f"Generated {metrics['audio_duration_sec']}s audio (RTF: {metrics['rtf']})")
```

### 2. Telephony Telecommunication Synthesis (8kHz μ-law / PCM16 for Asterisk/Vicidial)

```python
audio_bytes, metrics = engine.synthesize(
    text="[متحمس] ألف مبروك! تم تأكيد موعدك بنجاح ليوم الثلاثاء الساعة ٤:٣٠ مساءً.",
    sample_rate=8000,
    format_type="mulaw",  # or "pcm16", "alaw", "wav"
    ode_steps=16          # Ultra-low latency real-time voice
)
```

### 3. Real-Time Clause Streaming

```python
def token_stream():
    tokens = ["أهلاً ", "وسهلاً ", "بك. ", "كيف ", "أقدر ", "أساعدك ", "اليوم؟"]
    for t in tokens:
        yield t

for frame in engine.synthesize_stream(token_stream(), sample_rate=8000, frame_duration_ms=20):
    # Each frame is precisely 20ms (320 bytes @ 8kHz PCM16) ready for UDP/RTP socket
    send_to_telephony_channel(frame)
```

---

## 🧪 Benchmark Results (Official Section 5 Suite)

All 4 official benchmark sentences synthesized and verified:

| Benchmark | Input Text | Emotion | 24k Studio | 8k Telephony | Duration | RTF | Loudness |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Greeting** | `"أهلاً وسهلاً بحضرتك، تفضل كيف أقدر أساعدك اليوم؟"` | `neutral` | ✅ | ✅ | `5.31s` | `1.52` | `-26.0 dBFS` |
| **2. Excited** | `"[متحمس] ألف مبروك! تم تأكيد موعدك بنجاح ليوم الثلاثاء الساعة ٤:٣٠ مساءً."` | `excited` | ✅ | ✅ | `6.60s` | `1.08` | `-26.0 dBFS` |
| **3. Sympathetic** | `"[متعاطف] أعتذر جداً لحضرتك عن الإزعاج، وأنا هتابع الموضوع بنفسي حتى يتم حله بالكامل."` | `sympathetic` | ✅ | ✅ | `6.18s` | `1.12` | `-26.0 dBFS` |
| **4. Emotional** | `"[مؤثر] شكراً جزيلاً لثقتك الغالية فينا وشراكتنا الممتدة على مدار السنوات."` | `emotional` | ✅ | ✅ | `5.62s` | `1.15` | `-26.0 dBFS` |

*Note: In Benchmark 2, the telephony normalizer automatically expanded `الساعة ٤:٣٠ مساءً` to `الساعة الرابعة والنصف مساءً`.*

---

## 🌐 Running the Production Server

Start the FastAPI server:

```bash
uvicorn arabic_voice_model_kit.server.app:app --host 0.0.0.0 --port 8000
```

### Endpoints:
- `POST /synthesize`: Full JSON response with Base64 audio + latency metrics.
- `POST /synthesize/raw`: Returns raw binary stream (`audio/wav` or `application/octet-stream`).
- `GET /health`: Health status, GPU/VRAM metrics, ODE solver status.
- `GET /voices`: Lists installed voices, available emotions, and metadata.

---

## 🧪 Testing

Run the automated test suite:

```bash
PYTHONPATH=. .venv/bin/pytest arabic_voice_model_kit/tests/ -v
```

All 23 unit and integration tests pass with 100% success rate.
