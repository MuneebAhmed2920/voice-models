# 🎙️ Emma Voice Model (`emma` Branch)
### Production Real-Time Telephony TTS for Vicidial & AI Voice Bots (Windows NVIDIA CUDA FP16 / Linux)

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20FP16-ee4c2c)](https://pytorch.org/)
[![Vicidial](https://img.shields.io/badge/Vicidial-Asterisk%20AudioSocket-green)](https://www.vicidial.org/)
[![Branch](https://img.shields.io/badge/Branch-emma-brightgreen.svg)](https://github.com/MuneebAhmed2920/voice-models/tree/emma)

---

## 🌟 Speaker Profile

- **Speaker**: Emma (Female)
- **Branch**: `emma`
- **Voice ID**: `voice_emma`
- **Fundamental Pitch (F0)**: `186.1 Hz`
- **Persona Style**: Adult female. Soft, soothing, empathetic patient care and customer concierge.
- **Default Speed**: `0.9` | **Guidance Scale**: `2.2`

---

## 🚀 Quick Start (Windows with NVIDIA GPU)

### 1. Clone this branch
```powershell
git clone -b emma https://github.com/MuneebAhmed2920/voice-models.git
cd voice-models
pip install -r requirements.txt
```

### 2. Launch Server
```cmd
start_server.bat
```
Server runs at `http://0.0.0.0:8000` with WebSocket at `ws://0.0.0.0:8000/ws/tts` and Asterisk AudioSocket on port `9092`.

---

## 📊 Evaluation Audio Samples

All evaluation WAV samples are available in `samples/`:
- `01_conversation.wav`: Baseline conversational customer care greeting
- `02_question.wav`: Scheduling question with upward prosody
- `03_numbers.wav`: Number normalization and inventory counts
- `04_money.wav`: Complex currency balances and percentages
- `05_phone_number.wav`: Digits grouped into natural phone cadence
- `06_date_time.wav`: Calendar appointments and times
- `07_long_sentence.wav`: Extended multi-clause utterance
- `08_breathing.wav`: Organic airway inhalation before speech
- `09_excited.wav`: Celebratory approval intonation
- `10_sympathetic.wav`: Apologetic customer care empathy
- `11_emotional.wav`: Deeply heartfelt customer loyalty appreciation

---

## 📄 License
Permissive Voice Model Package licensed under [Apache-2.0](LICENSE).
