# 🎙️ OmniVoice Telephony Models (Single Multi-Branch Repository)
### Production Real-Time Telephony TTS Engine for Vicidial & AI Voice Bots (Windows NVIDIA CUDA FP16 / Linux)

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20FP16-ee4c2c)](https://pytorch.org/)
[![Vicidial](https://img.shields.io/badge/Vicidial-Asterisk%20AudioSocket-green)](https://www.vicidial.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## 🌿 Speaker Model Branches

This repository hosts all production voice models organized by dedicated Git branches:

| Branch | Speaker | Gender | Pitch (F0) | Persona Style | Checkout Command |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **`clara`** | **Clara** | Female | `200.0 Hz` | Warm Telephone Customer Care Specialist | `git checkout clara` |
| **`alex`** | **Alex** | Male | `160.2 Hz` | Warm Tenor, Upbeat Support Specialist | `git checkout alex` |
| **`david`** | **David** | Male | `120.3 Hz` | Calm Baritone, Articulate Advisor | `git checkout david` |
| **`michael`** | **Michael** | Male | `108.2 Hz` | Deep Bass-Baritone, Authoritative Lead | `git checkout michael` |
| **`emma`** | **Emma** | Female | `186.1 Hz` | Soft, Empathetic Healthcare Concierge | `git checkout emma` |
| **`sarah`** | **Sarah** | Female | `217.6 Hz` | Confident, Sharp Business Advisor | `git checkout sarah` |
| **`olivia`** | **Olivia** | Female | `186.7 Hz` | Bright, Upbeat Lifestyle Concierge | `git checkout olivia` |

---

## 🚀 Quick Start (Any Branch)

```powershell
# 1. Clone specific speaker branch
git clone -b clara https://github.com/MuneebAhmed2920/voice-models.git
cd voice-models

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start server (Windows CUDA FP16)
start_server.bat
```

---

## ⚡ Architecture Highlights

- **NVIDIA CUDA FP16 Acceleration**: Native half-precision inference with Tensor Cores (`< 150ms` TTFA).
- **Asterisk AudioSocket (Port 9092)**: Direct TCP bridge to Asterisk dialplans (`exten => 8300,1,AudioSocket(...)`).
- **FastAPI Streaming WebSocket (`/ws/tts`)**: Speculative clause buffering with instant caller barge-in cancellation.
- **PSTN Telephony Audio Pipeline**: 8,000 Hz / 16,000 Hz Linear PCM16 framing (20ms frames / 320 bytes) & G.711 $\mu$-law/A-law.
- **Emotion Switching**: In-line tag parsing for `[excited]`, `[sympathetic]`, `[emotional]`, and physiological inhalation breathing.

---

## 📄 License
Permissive Voice Model Package licensed under [Apache-2.0](LICENSE).
