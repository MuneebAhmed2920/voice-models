# 🎙️ OmniVoice Telephony Models (Unified Multi-Language Repository)
### Production Real-Time Voice Models & Telephony TTS Engines for AI Voice Bots, Asterisk PBX & Vicidial

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20%2F%20MPS%20%2F%20CPU-ee4c2c)](https://pytorch.org/)
[![Telephony](https://img.shields.io/badge/Telephony-8kHz%20G.711%20PSTN-green)](https://en.wikipedia.org/wiki/G.711)
[![Asterisk AudioSocket](https://img.shields.io/badge/Asterisk-AudioSocket%20TCP-orange)](https://www.asterisk.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## 🌟 Architecture & Language Branches

This repository is the **single unified home for all production voice models**, organized into dedicated Git branches by language. Each branch contains the complete standalone engine, phonetic text normalizers, calibrated speaker prompts, telephony servers, and high-fidelity audio samples.

| Branch | Language / Region | Speaker Persona | Acoustic Profile | Key Technical Features | Checkout Command |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`urdu`** | **Pakistani Urdu (`ur-PK`)** | **Mahrukh (ماہ رخ)** | Warm, melodious female (`~200 Hz`) | Automatic loanword de-lagging, female grammatical agreement, 8kHz G.711 filter | `git checkout urdu` |
| **`spanish`** | **Spanish (`es` / Latin America)** | **Sofía** | Vibrant, energetic native female (`258 Hz`) | Native Latin American human acoustic conditioning, speed 0.92, natural respiration pauses (280-320ms) | `git checkout spanish` |
| **`english`** | **English (`en-US` / Int'l)** | **Clara / Sarah / Alex** | Warm, confident enterprise tone | Dynamic flow matching, 4 fixed personas, speculative sentence chunking | `git checkout english` |
| **`arabic`** | **Arabic (`ar-SA` / Gulf)** | **Fatima / Modern Standard** | Expressive, formal native tone | Full Tashkeel (diacritization) frontend, phoneme expansion, telephony framing | `git checkout arabic` |
| **`main`** | **Multilingual Hub** | **All Models & Orchestrator** | Multi-Engine Showcase | Telephony streaming bridges, Asterisk AudioSocket, Docker compose, benchmark suite | `git checkout main` |

---

## 🎧 Master Multi-Language Audio Showcase

The `samples/` directory on `main` provides high-fidelity audio demonstrations across all supported enterprise languages:

| Audio Demo ID | Language & Accent | Use Case Scenario | Call Duration | Web MP3 (192kbps) | Telephony MP3 (128kbps) |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **`REC-UR-01`** | **Pakistani Urdu (`ur-PK`)** | Order Confirmation & Logistics Dispatch | `25.90s` | [`rec_01_urdu_customer_service.mp3`](samples/rec_01_urdu_customer_service.mp3) | [`rec_01_urdu_customer_service_telephony.mp3`](samples/rec_01_urdu_customer_service_telephony.mp3) |
| **`REC-EN-01`** | **English (`en-US` / Int'l)** | Enterprise Customer Care & Scheduling | `28.04s` | [`rec_02_english_customer_service.mp3`](samples/rec_02_english_customer_service.mp3) | [`rec_02_english_customer_service_telephony.mp3`](samples/rec_02_english_customer_service_telephony.mp3) |
| **`REC-AR-01`** | **Arabic (`ar-SA` / Gulf)** | Inbound Support & Courier Dispatch | `28.58s` | [`rec_03_arabic_customer_service.mp3`](samples/rec_03_arabic_customer_service.mp3) | [`rec_03_arabic_customer_service_telephony.mp3`](samples/rec_03_arabic_customer_service_telephony.mp3) |
| **`REC-ES-01`** | **Spanish (`es` / Español)** | Humanized Order & Tracking Support | `59.40s` | [`rec_05_spanish_customer_service.mp3`](samples/rec_05_spanish_customer_service.mp3) | [`rec_05_spanish_customer_service_telephony.mp3`](samples/rec_05_spanish_customer_service_telephony.mp3) |
| **`REC-MULTI-01`** | **Trilingual Code-Switching** | Seamless Live Switching (`EN ⇄ UR ⇄ AR`) | `23.43s` | [`rec_04_trilingual_code_switching.mp3`](samples/rec_04_trilingual_code_switching.mp3) | [`rec_04_trilingual_code_switching_telephony.mp3`](samples/rec_04_trilingual_code_switching_telephony.mp3) |

---

## 🚀 Quick Start Guide

### 1. Clone a Specific Language Branch
To deploy or develop for a specific language, checkout its dedicated branch:
```bash
# Clone only the Urdu voice model
git clone -b urdu https://github.com/MuneebAhmed2920/voice-models.git

# Clone only the Spanish voice model
git clone -b spanish https://github.com/MuneebAhmed2920/voice-models.git

# Clone only the English voice model
git clone -b english https://github.com/MuneebAhmed2920/voice-models.git

# Clone only the Arabic voice model
git clone -b arabic https://github.com/MuneebAhmed2920/voice-models.git
```

### 2. Run the Multi-Engine Streaming Server (Main Hub)
```bash
# Install core dependencies
pip install -r requirements.txt

# Start Asterisk AudioSocket & FastAPI Streaming Server
./start_server.sh   # Linux / macOS
start_server.bat    # Windows (CUDA FP16)
```

---

## ⚡ Core Telephony Features

- **Sub-150ms First-Audio Latency (TTFA)**: Speculative clause tokenization and streaming flow matching.
- **Asterisk AudioSocket (Port 9092)**: Direct Asterisk dialplan integration via raw PCM TCP socket.
- **Dual Sampling Pipeline**: 24,000 Hz Studio HD Master and 8,000 Hz G.711 $\mu$-law bandpass filtered telephony audio.
- **Instant Caller Barge-In**: Immediate WebSocket audio flush on user interruption.

---

## 📄 License
Licensed under the [Apache-2.0 License](LICENSE).
