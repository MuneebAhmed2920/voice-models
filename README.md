# Spanish OmniVoice TTS

A clean, production-focused Spanish Text-to-Speech (TTS) runtime built on the OmniVoice base discrete diffusion model, integrated Latin American Spanish text normalization, and native voice conditioning.

> **Note on Architecture:** This repository does not introduce fine-tuned model weights. It packages the official OmniVoice base model together with a comprehensive Latin American Spanish normalization frontend and conditioning prompt embeddings.

---

## Voices

* **Female (`female` / `spanish_female`):** Colombian female persona (258 Hz mean $F_0$), conditioned via `Prompt01`.
* **Male (`male` / `spanish_male`):** Colombian male persona (125 Hz mean $F_0$), conditioned via `Prompt07`.

---

## Features

* **High-Fidelity Audio:** Native 24,000 Hz, mono PCM WAV output.
* **Dual Voice Personas:** Female and Male voice conditioning.
* **Automated Text Normalization:** Built-in rule-based normalization for neutral Latin American Spanish:
  * Cardinal and ordinal numbers
  * Currencies (e.g., `$25.50` $\rightarrow$ *"veinticinco dólares con cincuenta centavos"*)
  * Dates and times (e.g., `15 de octubre a las 3:30` $\rightarrow$ *"quince de octubre a las tres y media"*)
  * Phone numbers and addresses
  * Questions and inverted Spanish punctuation (`¿`, `¡`)
  * Accented vowels (`á`, `é`, `í`, `ó`, `ú`, `ü`, `ñ`)
* **Minimal Footprint:** Lightweight package structure with self-contained voice conditioning assets.

---

## Base Model & Weights

The base acoustic model is the official [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) along with the [eustlb/higgs-audio-v2-tokenizer](https://huggingface.co/eustlb/higgs-audio-v2-tokenizer).
* Model weights (~2.2 GB) download **automatically** on first execution via Hugging Face Hub and are cached in `~/.cache/huggingface/hub/`.
* No large multi-gigabyte model binaries are committed to this repository.

---

## Installation

```bash
# Clone the repository
git clone git@github.com:MuneebAhmed2920/spanish-omnivoice-tts.git
cd spanish-omnivoice-tts

# Install runtime dependencies
pip install -r requirements.txt
```

---

## Usage

### Python API

```python
from spanish_voice import SpanishVoice

# Initialize TTS synthesizer (defaults: female, 24 diffusion steps, 2.2 guidance)
tts = SpanishVoice()

# 1. Synthesize Female Speech
tts.generate(
    text="Hola, ¿cómo estás? Espero que tengas un excelente día.",
    voice="female",
    output="female_greeting.wav"
)

# 2. Synthesize Male Speech
tts.generate(
    text="Buenos días. Gracias por comunicarse con nosotros. ¿En qué puedo ayudarle?",
    voice="male",
    output="male_greeting.wav"
)

# 3. Automatic Normalization (Currency, Numbers, Dates)
tts.generate(
    text="El precio total es de $75.25 para su cita del 15 de octubre a las 3:30.",
    voice="female",
    output="appointment.wav"
)
```

### Command Line Interface (CLI)

```bash
# Female voice
python generate.py \
  --voice female \
  --text "Hola, ¿cómo estás? Espero que tengas un excelente día." \
  --output female.wav

# Male voice
python generate.py \
  --voice male \
  --text "Buenos días. Gracias por comunicarse con nosotros. ¿En qué puedo ayudarle?" \
  --output male.wav
```

CLI options:
* `--text`: Input Spanish text (required).
* `--voice`: Voice persona (`female` or `male`, default: `female`).
* `--output`: Output WAV file path (default: `output.wav`).
* `--steps`: Diffusion steps (default: `24` for highest quality).
* `--guidance`: Classifier-free guidance scale (default: `2.2`).
* `--seed`: Random seed for deterministic generation (default: `42`).
* `--device`: Compute device (`cpu` or `cuda`, default: `cpu`).

---

## Demo Reference Recordings

Two comprehensive full-length listening recordings (~12–13 minutes each) are included in the repository for evaluation and comparison:

* `recordings/final_spanish_female_full_test.wav` (34.74 MB, 12:38, 24 kHz mono)
* `recordings/final_spanish_male_full_test.wav` (36.53 MB, 13:18, 24 kHz mono)

Both recordings cover identical conversational test scripts spanning greetings, customer service dialogue, numbers, currency, dates, phonetic edge cases ($R$, $RR$, $\tilde{N}$, $LL$, $J$, $G$, $B/V$, $C/S/Z$, $CH$), lexical stress, and short responses.

---

## Spanish Whisper ASR Evaluation

This repository includes a standalone evaluation tool for blind ASR transcription using **Whisper `large-v3`** (via `faster-whisper`):

```bash
# 1. Install faster-whisper
pip install faster-whisper

# 2. Pre-download and cache Whisper large-v3 weights (~3.1 GB)
python transcribe_spanish.py --download-only

# 3. Transcribe generated Spanish speech
python transcribe_spanish.py output.wav
```

This outputs both clean plain text (`.txt`) and detailed word-level timestamped transcripts with confidence metrics (`.json`).

See [WHISPER_SETUP.md](WHISPER_SETUP.md) for complete details on hardware acceleration, programmatic Python usage, and Spanish linguistic evaluation guidelines.

---

## Repository Structure

```
spanish-omnivoice-tts/
├── spanish_voice/
│   ├── __init__.py       # Package entry point
│   ├── model.py          # SpanishVoice inference runtime
│   ├── normalizer.py     # Latin American text normalization
│   └── voices/
│       ├── female/
│       │   └── prompt.pt # Prompt01 female conditioning tensor
│       └── male/
│           └── prompt.pt # Prompt07 male conditioning tensor
├── recordings/
│   ├── final_spanish_female_full_test.wav
│   └── final_spanish_male_full_test.wav
├── generate.py           # CLI interface
├── transcribe_spanish.py # Whisper ASR evaluation tool
├── WHISPER_SETUP.md      # Whisper setup & evaluation guide
├── requirements.txt      # Minimal runtime dependencies
├── README.md
└── .gitignore
```

---

## 🎧 High-Fidelity Audio Showcase Samples

The `samples/` directory provides master recordings synthesized with authentic native Latin American acoustic conditioning (`mean F0: 258 Hz`), natural respiration pauses (280–320ms), and calibrated speech rate (`speed=0.92`):

| Sample Asset | Format | Sample Rate | Description |
| :--- | :---: | :---: | :--- |
| `samples/rec_05_spanish_customer_service.mp3` | MP3 (192kbps) | 24,000 Hz | High-energy, warm customer service order & delivery tracking demo (59.40s) |
| `samples/rec_05_spanish_customer_service_telephony.mp3` | MP3 (128kbps) | 8,000 Hz | PSTN / Telephony bandpass filtered phone line simulation (59.40s) |
| `samples/rec_05_spanish_customer_service_24k.wav` | Linear PCM WAV | 24,000 Hz | Studio Master HD Recording (-1.0 dBFS Peak Normalized) |
| `samples/rec_05_spanish_customer_service_8k_telephony.wav` | Linear PCM WAV | 8,000 Hz | Asterisk / PBX / G.711 Telephony Master |

