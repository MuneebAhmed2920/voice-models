# 🎙️ Mahrukh Pakistani Urdu Voice Model (`mahrukh_urdu_model`)
### Dedicated Standalone Female Voice Model & Telephony TTS Engine for AI Voice Bots & Call Centers

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Audio-ee4c2c)](https://pytorch.org/)
[![Language](https://img.shields.io/badge/Language-Pakistani%20Urdu%20(ur--PK)-green)](https://en.wikipedia.org/wiki/Urdu)
[![Telephony](https://img.shields.io/badge/Telephony-8kHz%20G.711%20PSTN-orange)](https://en.wikipedia.org/wiki/G.711)

---

## 🌟 Overview

`mahrukh_urdu_model` is a dedicated, self-contained **Pakistani Urdu Voice Model** engineered specifically for **AI Customer Service Voice Bots**, **Asterisk PBX**, and **Vicidial Telephony Networks**.

It packages the calibrated vocal identity of **Mahrukh (ماہ رخ)** — an ultra-polite, friendly, and melodious Pakistani customer service representative.

### Key Highlights:
- **Locked to Pakistani Urdu (`ur-PK`)**: Dedicated pipeline with zero cross-lingual interference or accidental language switching.
- **Built-in Loanword De-Lagging & Phonetics**: Automatically converts and vocalizes loanwords so they sound 100% natural without syllable dragging:
  - `customer` $\to$ `/kʌstəmər/` (`کَسٹمر`)
  - `receive` $\to$ `/rɪˈsiːv/` (`receive` / `رِسیو`)
  - `asalamualikum` $\to$ `اَلسَّلَامُ عَلَیْکُم` (vocalized classical phone greeting)
  - `package` $\to$ `پیکیج`, `dispatch` $\to$ `ڈسپَیچ`, `confirm` $\to$ `کنفرْم`, `payment` $\to$ `پےمنٹ`, `delivery` $\to$ `ڈیلیوری`
- **Automatic Female Grammatical Agreement**: Converts male default verb forms (`کرتا ہوں`, `سکتا ہوں`) into natural feminine agreement (`کرتی ہوں`, `سکتی ہوں`).
- **Telephony Audio Pipeline (8kHz G.711 PSTN)**: Built-in 300Hz–3400Hz Butterworth telephone line bandpass filtering with peak safety normalization for direct call center broadcast.
- **FastAPI & OpenAI Compatibility**: Drop-in compatible with standard OpenAI speech endpoints (`/v1/audio/speech`) and telephony webhooks (`/v1/tts/synthesize`).

---

## 📁 Package Layout

```
mahrukh_urdu_model/
├── __init__.py                      # Package entrypoint (UrduVoiceModel)
├── config/
│   ├── config.json                  # Model hyperparams (speed: 1.12, guidance: 2.0)
│   └── metadata.json                # Voice model provenance and speaker profile
├── prompt/
│   ├── mahrukh_urdu_prompt.pt       # Calibrated warm voice clone prompt tensor
│   └── reference_master_24k.wav     # 24kHz clean studio reference recording
├── runtime/
│   ├── engine.py                    # UrduVoiceModel singleton with prompt caching
│   ├── urdu_normalizer.py           # Self-contained loanword & Perso-Arabic normalizer
│   └── audio_processor.py           # 8kHz PSTN (300-3400Hz) & 24kHz audio processor
├── server/
│   ├── app.py                       # FastAPI server (/v1/tts/synthesize, /v1/audio/speech)
│   └── schemas.py                   # Pydantic request/response schemas
├── examples/
│   ├── quickstart.py                # 5-line quickstart demo
│   └── telephony_call_demo.py       # Full customer-service telephony call generator
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

### 1. Simple Python API

```python
from mahrukh_urdu_model import UrduVoiceModel

# Initialize the model (singleton, loads weights & Mahrukh prompt once)
model = UrduVoiceModel()

# Synthesize 24kHz HD Studio Audio
audio, sr = model.synthesize(
    text="اَلسَّلَامُ عَلَیْکُم! ہماری کَسٹمر سروس میں خوش آمدید، میں آپ کی کیا مدد کر سکتی ہوں؟",
    sample_rate=24000,
    output_file="greeting_hd.wav"
)

# Synthesize 8kHz Telephony Line Audio (for Phone Calls)
audio_8k, sr_8k = model.synthesize(
    text="آپ کی پےمنٹ receive ہو چکی ہے اور پیکج ڈسپَیچ کے لیے ایکٹیو ہے۔",
    sample_rate=8000,
    telephony_filter=True,
    output_file="telephony_call.wav"
)
```

### 2. Start the Telephony REST Server

```bash
uvicorn mahrukh_urdu_model.server.app:app --host 0.0.0.0 --port 8000
```

#### API Endpoints:
- **`GET /health`**: Health check and model readiness.
- **`POST /v1/tts/synthesize`**: Custom synthesis with sample rate and telephony filtering.
- **`POST /v1/audio/speech`**: OpenAI-compatible speech synthesis for standard AI bot frameworks.

---

## 📜 License
Internal proprietary voice model asset for authorized customer service automation.

---

## 🎧 High-Fidelity Audio Showcase Samples

Included in the `samples/` directory:

| Sample Asset | Format | Sample Rate | Description |
| :--- | :---: | :---: | :--- |
| `samples/rec_01_urdu_customer_service.mp3` | MP3 (192kbps) | 24,000 Hz | High-fidelity Web & Mobile Customer Service Order Confirmation |
| `samples/rec_01_urdu_customer_service_telephony.mp3` | MP3 (128kbps) | 8,000 Hz | Bandpass-filtered PSTN Telephony Call Simulation |
| `samples/rec_01_urdu_customer_service_24k.wav` | Linear PCM WAV | 24,000 Hz | Studio Master HD Recording (-1.0 dBFS Headroom) |
| `samples/rec_01_urdu_customer_service_8k_telephony.wav` | Linear PCM WAV | 8,000 Hz | G.711 / Asterisk / PBX Telephony Master |

