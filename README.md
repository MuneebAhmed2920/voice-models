# 🎙️ Arabic OmniVoice Text-to-Speech (TTS) System
### نظام أومني فويس العربي المتقدم لتوليد الصوت بالذكاء الاصطناعي

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![OmniVoice](https://img.shields.io/badge/Base-k2--fsa%2FOmniVoice-purple.svg)](https://github.com/k2-fsa/OmniVoice)
[![VoiceTut-TTS](https://img.shields.io/badge/Fine--tuned-VoiceTut--TTS-orange.svg)](https://github.com/MohammedAly22/VoiceTuT-TTS)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Hardware-Apple%20Silicon%20(MPS)%20%7C%20NVIDIA%20CUDA-emerald.svg)]()

A state-of-the-art **Arabic Text-to-Speech (TTS) and Voice Cloning system** powered by **OmniVoice** and **VoiceTut-TTS**. Built on a non-autoregressive discrete diffusion language model architecture, it produces natural, highly expressive Arabic speech across multiple dialects with zero-shot voice cloning, real-time streaming, and automatic Tashkeel (harakat diacritization).

---

## ✨ Key Features | أبرز المميزات

1. 🗣️ **Dual-Engine Arabic Synthesis**:
   - **Modern Standard Arabic (الفصحى - MSA)**: Formal broadcast, news, documentary, and educational speech with full Tashkeel vocalization support.
   - **Egyptian Arabic (اللهجة المصرية)**: Everyday conversational Egyptian speech with seamless **Arabic ↔ English code-switching** (powered by the fine-tuned *VoiceTut-TTS* model).
   - **Regional Dialect Presets**: Gulf (خليجي), Levantine (شامي), and conversational profiles.

2. 🌬️ **Organic Breathing & Human Gestures (التنفس الفسيولوجي والمحادثة الطبيعية)**:
   - **Physiological Airway Inhalations**: Natural pre-speech inhalation (120ms–250ms) conditioned from clean studio recordings, bandpassed (180 Hz–5,500 Hz), RMS normalized (-28 dBFS), with raised-cosine fades.
   - **Emotion & Prosody Matrix**: Modulated inhalation duration, gain, pre-speech gaps, and acoustic EQ coloration across 4 emotional vocal states (`neutral`, `excited`, `sympathetic`, `emotional`).
   - **Arabic Syntactic Clause Splitting**: Micro-breath pauses placed naturally before major sentence connectors (`و`, `فـ`, `لكن`, `يعني`, `طبعاً`, `إذا`, `لو`).
   - **Seamless Zero-Crossing Splicing**: Eliminates boundary clicking and DC offsets.

3. 🧬 **Zero-Shot Voice Cloning (استنساخ الأصوات الفوري)**:
   - Clone any Arabic voice in seconds from a short (3–10s) reference audio clip or microphone recording.
   - No fine-tuning or retraining required.

4. 🎨 **Voice Design via Natural Language Prompts (تصميم الأصوات بالأوامر)**:
   - Describe the desired voice attributes in natural language:
     - e.g., *"صوت رجل وقور في الخمسين، نبرة إذاعية عميقة ولغة عربية فصيحة"*
     - e.g., *"female, calm documentary narrator, crisp Modern Standard Arabic"*

5. ✨ **Intelligent Arabic NLP & Tashkeel Pipeline**:
   - **Numeral & Currency Normalization**: Converts numbers (`123` -> `مائة وثلاثة وعشرون`), dates (`15/09/2026`), times (`3:30` -> `الساعة الثالثة والنصف`), currencies (`$` -> `دولار`, `ج.م` -> `جنيه`), and percentages (`25%`).
   - **Vocalization / Harakat Restorer**: Preserves user-provided diacritics and enriches functional words and particles to guarantee correct phonetic pronunciation.

6. ⚡ **Low-Latency & True Streaming**:
   - Sentence boundary chunking yields audio streams progressively for low time-to-first-audio (TTFA).
   - Supports 24 kHz high-fidelity studio audio output.

7. 🌐 **Arabic Interactive Web Studio & REST API**:
   - RTL dark glassmorphic web interface with waveform player, microphone recorder, conversational presets, and emotion toggles.
   - FastAPI server with `/synthesize`, `/synthesize_conversational`, `/clone`, and `/normalize` endpoints.

8. 🛠️ **Training & Fine-Tuning Pipeline**:
   - Complete dataset preprocessing and PyTorch diffusion LM fine-tuning harness for custom voice datasets.

---

## 🏛️ System Architecture

```
[Arabic Input Text (with optional tags: [متحمس], [متعاطف], [مؤثر], [تنفس], [وقفة])]
        │
        ▼
[Arabic Gesture Processor & Normalizer] ──► Extracts emotion & tags, splits clauses
        │
        ├─────────────────────────────┬─────────────────────────────┐
        ▼                             ▼                             ▼
[Physiological Breath Matrix]    [Speech Synthesizer]      [Emotional EQ Filters]
(180Hz–5500Hz, -28dBFS RMS,     (OmniVoice / VoiceTut)    (Presence Boost @ 4.2kHz,
 15ms/30ms raised-cosine fades)                            Warm Low-Mids @ 280Hz,
                                                           Chest Resonance @ 320Hz)
        │                             │                             │
        └─────────────────────────────┼─────────────────────────────┘
                                      ▼
                      [Zero-Crossing Seamless Splicer]
                                      │
                                      ▼
                [24kHz Organic Human Conversational Speech]
```

---

## 🚀 Quickstart

### 1. Installation

Clone this repository and run the automated setup script:
```bash
git clone <repo-url>
cd "omni voice Arabic"

# Automated environment setup (uses uv if available, or python venv)
chmod +x setup_env.sh
./setup_env.sh

# Activate the virtual environment
source .venv/bin/activate
```

Or install manually using `uv`:
```bash
uv venv --python python3.11 .venv
source .venv/bin/activate
uv pip install torch torchaudio soundfile scipy numpy librosa transformers accelerate pyarabic fastapi uvicorn gradio pydantic
uv pip install omnivoice voicetut-tts
```

---

### 2. Launch the Web Studio UI

Start the interactive Arabic voice studio:
```bash
python -m src.cli web --port 7860
```
Open your browser at `http://localhost:7860`.

Features in the Web Studio:
- **Studio TTS Tab**: Select from 12+ built-in male/female voices, adjust speed & steps, and preview live diacritics.
- **Voice Cloning Tab**: Upload reference audio or record via mic to instantly replicate any voice.
- **Voice Design Tab**: Create custom voices from descriptive text prompts.
- **Streaming Tab**: Long-form paragraph generation.

---

### 3. CLI Usage

#### Standard Synthesis
```bash
# Synthesize Modern Standard Arabic (MSA)
python -m src.cli synthesize "مرحباً بكم في استوديو أومني فويس العربي" \
    --voice msa_kareem \
    --output msa_news.wav

# Synthesize Egyptian Arabic with Code-Switching
python -m src.cli synthesize "عندي meeting الساعة 3:30 في الشركة" \
    --voice egy_mohamed \
    --output egyptian.wav
```

#### Zero-Shot Voice Cloning
```bash
python -m src.cli clone \
    --ref my_voice_sample.wav \
    --text "هذا الصوت تم استنساخه بالكامل من العينة المرجعية بدون تدريب مسبق." \
    --output cloned_output.wav
```

#### List Available Voices
```bash
python -m src.cli list-voices
```

#### Test Text Normalization
```bash
python -m src.cli normalize "بلغت الأرباح 1500 $ في 15/09/2026 مع نمو 20%"
```

---

### 4. Python API Usage

```python
from src.models.omnivoice_engine import ArabicOmniVoiceEngine

engine = ArabicOmniVoiceEngine()

# 1. Synthesize with preset voice
audio, sr = engine.synthesize(
    text="سَيِّدَاتِي وَآنِسَاتِي، أَهْلًا بِكُمْ فِي هَذَا المَوْجَزِ الصَّبَاحِيّ.",
    voice_id="msa_kareem",
    output_path="output_msa.wav"
)

# 2. Zero-Shot Voice Cloning
audio, sr = engine.synthesize(
    text="النهاردة بنجرب استنساخ الصوت بجودة عالية جداً.",
    ref_audio="speaker_sample.wav",
    output_path="output_cloned.wav"
)

# 3. Voice Design via Description
audio, sr = engine.synthesize(
    text="في أعماق البحار، أسرار لا تنتهي من الجمال والغموض.",
    instruct="female, calm and elegant voice, articulate Modern Standard Arabic, documentary narrator",
    output_path="output_documentary.wav"
)

# 4. Human Conversational Speech with Organic Breathing & Prosody
audio, sr = engine.synthesize_conversational(
    text="[متحمس] ألف مبروك! [تنفس] تم قبول طلبكم بنجاح، ويسعدنا جداً انضمامكم إلينا اليوم!",
    voice_id="msa_salma",
    include_breath=True,
    split_clauses=True,
    output_path="output_conversational.wav"
)
```

---

### 5. Launch REST API Server

```bash
python -m src.cli api --port 8000
```

#### API Endpoints:
- `POST /synthesize`: Generate speech from JSON payload (`text`, `voice_id`, `dialect`, `speed`, `num_step`).
- `POST /synthesize_conversational`: Humanized conversational synthesis with respiratory dynamics, clause micro-breaths, and emotion coloration.
- `POST /clone`: Zero-shot voice cloning with multipart audio upload.
- `GET /speakers`: List all available voices and metadata.
- `POST /normalize`: Test Arabic normalization and Tashkeel.
- `GET /health`: Engine status and GPU/MPS device info.

Example cURL request:
```bash
curl -X POST http://localhost:8000/synthesize_conversational \
     -H "Content-Type: application/json" \
     -d '{
       "text": "[متعاطف] أهلاً بك يا فندم، نعتذر عن أي تأخير، [تنفس] وبإذن الله هنحل المشكلة فوراً.",
       "voice_id": "msa_kareem",
       "include_breath": true
     }' \
     --output conversational_speech.wav
```

---

## 🌬️ Conversational Prosody & Organic Breathing Matrix

Human speech relies on physiological respiratory dynamics. This system implements an acoustic controller with 4 tailored emotional states:

| Emotion / Style | Arabic Context | Inhalation Duration | Relative Gain | Pre-Speech Gap | Acoustic EQ Coloration | Speed |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| **Neutral / Conversational** | Standard customer service / information | `140ms – 160ms` | `-28 dBFS` (`0.032`) | `50ms` | Balanced natural vocal tract | `1.00x` |
| **Excited / Upbeat (`[متحمس]`)** | Celebratory news / joyful agreement | `80ms – 110ms` | `-30 dBFS` (`0.024`) | `25ms` | Crisp presence (+2.0 dB @ 4.2 kHz) | `0.98x` |
| **Sympathetic / Caring (`[متعاطف]`)** | Apologetic / empathetic customer care | `180ms – 220ms` | `-26 dBFS` (`0.045`) | `75ms` | Warm low-mids (+1.8 dB @ 280 Hz) | `0.86x` |
| **Emotional / Sincere (`[مؤثر]`)** | Heartfelt gratitude / reassurance | `150ms – 180ms` | `-27 dBFS` (`0.038`) | `60ms` | Rich chest resonance (+1.5 dB @ 320 Hz) | `0.90x` |
| **High Pitch / Loud (`[نبرة_عالية]`)** | Vibrant excitement / elevated melodic contour | `80ms – 100ms` | `-30 dBFS` (`0.024`) | `20ms` | High harmonic presence (+2.5 dB @ 4.2 kHz) | `1.02x` |
| **Low Pitch / Deep (`[نبرة_منخفضة]`)** | Solemn / authoritative deep chest resonance | `200ms – 240ms` | `-26 dBFS` (`0.040`) | `70ms` | Deep subglottal boost (+2.5 dB @ 140–360 Hz) | `0.88x` |
| **Soft Whisper (`[همس]`)** | Intimate proximity / secretive tone | `110ms – 140ms` | `-32 dBFS` (`0.020`) | `35ms` | Attenuated sub-bass & airy breath boost (~5.5 kHz) | `0.92x` |

### Supported In-Line Gesture & Emotion Tags:
- **Pitch & Dynamics**: `[نبرة_عالية]` / `[high-pitch]` / `[صوت_عالي]`, `[نبرة_منخفضة]` / `[low-pitch]` / `[قرار]`, `[همس]` / `[whisper]`
- **Vocal Gestures**: `[ضحك]` / `[laughter]`, `[تنهد]` / `[sigh]`, `[شهقة]` / `[surprise-ah]`
- **Breathing & Pauses**: `[تنفس]` / `[breath]`, `[بدون_تنفس]` / `[no-breath]`, `[وقفة]` / `[pause]`, `<break time="300ms"/>`
- **Emotions & Styles**: `[متحمس]` / `[excited]`, `[متعاطف]` / `[sympathetic]`, `[مؤثر]` / `[emotional]`

### Arabic Syntactic Clause Splitting:
Micro-breath pauses are automatically inserted before major Arabic sentence connectors when clauses exceed 8 words:
- `و` (Wa- / And)
- `فـ` (Fa- / So)
- `لكن` / `ولكن` (Lakin / But)
- `يعني` (Ya'ni / Meaning)
- `طبعاً` / `أكيد` (Tab'an / Certainly)
- `إذا` / `لو` (Itha / If)

---

## 🏋️ Training & Fine-Tuning Pipeline

If you have your own Arabic audio dataset (podcasts, audiobooks, call center recordings), you can fine-tune OmniVoice:

### Step 1: Preprocess Audio & Transcripts
```bash
python -m training.prepare_arabic_dataset \
    --raw-dir /path/to/raw_audio_folder \
    --transcripts /path/to/transcripts.tsv \
    --output-dir data/processed_arabic \
    --dialect msa
```
This automatically:
- Converts audio to 24 kHz mono 16-bit WAV.
- Applies Arabic normalization and Tashkeel.
- Generates `train_manifest.jsonl` and `val_manifest.jsonl`.

### Step 2: Run Fine-Tuning
```bash
python -m training.train_diffusion \
    --train-manifest data/processed_arabic/train_manifest.jsonl \
    --val-manifest data/processed_arabic/val_manifest.jsonl \
    --output-dir checkpoints/my_arabic_model \
    --epochs 10 \
    --batch-size 4 \
    --lr 2e-5
```

---

## 🎭 Expressive Vocal Tags

OmniVoice supports non-verbal vocalization cues directly within the Arabic text:
- `[laughter]` : Adds a subtle, natural chuckle or laugh.
- `[sigh]` : Adds a natural exhale or sigh.
- `[surprise-ah]` : Expresses mild surprise or amazement.

Example:
```
"مرحباً بكم جميعاً! [laughter] لا أصدق أننا حققنا هذا الإنجاز اليوم [sigh] أخيراً ارتحنا."
```

---

## 📁 Repository Structure

```
.
├── setup_env.sh                  # Automated environment setup script
├── pyproject.toml                # Project specification and dependencies
├── README.md                     # Comprehensive documentation
├── src/
│   ├── arabic_nlp/               # Arabic NLP Preprocessing
│   │   ├── normalizer.py         # Numbers, currency, dates, times, code-switch
│   │   └── diacritizer.py        # Tashkeel Harakat vocalizer & density analyzer
│   ├── models/                   # Neural Models & Inference Core
│   │   ├── omnivoice_engine.py   # Unified OmniVoice & VoiceTut wrapper
│   │   └── voice_manager.py      # 12+ Studio voices & cloning registry
│   ├── app/                      # Applications & Interfaces
│   │   ├── web_studio.py         # Bilingual Gradio Web Studio (RTL dark UI)
│   │   └── api.py                # FastAPI REST & streaming audio server
│   └── cli.py                    # Command-line interface
├── training/                     # Training & Fine-Tuning Suite
│   ├── prepare_arabic_dataset.py # Audio resampling & transcript alignment
│   └── train_diffusion.py        # Discrete diffusion language model training loop
└── tests/
    └── test_arabic_tts.py        # Full unit test suite
```

---

## 📄 License
Released under the **Apache 2.0 License**. Free for personal, research, and commercial use.

---

## 🎧 High-Fidelity Audio Showcase Samples

The `samples/` directory provides native Modern Standard & Gulf Arabic voice models with full diacritization (Tashkeel) and telephony filtering:

| Sample Asset | Format | Sample Rate | Description |
| :--- | :---: | :---: | :--- |
| `samples/rec_03_arabic_customer_service.mp3` | MP3 (192kbps) | 24,000 Hz | Enterprise Arabic Inbound Support & Logistics Dispatch (28.58s) |
| `samples/rec_03_arabic_customer_service_telephony.mp3` | MP3 (128kbps) | 8,000 Hz | PSTN / Telephony bandpass filtered phone line simulation (28.58s) |
| `samples/rec_03_arabic_customer_service_24k.wav` | Linear PCM WAV | 24,000 Hz | Studio Master HD Recording (-1.0 dBFS Peak Normalized) |
| `samples/rec_03_arabic_customer_service_8k_telephony.wav` | Linear PCM WAV | 8,000 Hz | Asterisk / PBX / G.711 Telephony Master |
| `samples/perfect_arabic_recording.wav` | Linear PCM WAV | 24,000 Hz | Native benchmark reference demo with full diacritics |

