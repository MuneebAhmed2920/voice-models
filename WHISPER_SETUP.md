# Spanish Whisper Model Setup & ASR Evaluation Guide

This guide describes how to download, configure, and run the exact **Whisper `large-v3`** automatic speech recognition (ASR) setup used for evaluating the Spanish OmniVoice TTS models.

---

## 1. Model Specifications

* **Model Checkpoint:** `large-v3` (OpenAI Whisper)
* **Inference Engine:** `faster-whisper` (CTranslate2)
* **Language:** `es` (Spanish)
* **Task:** `transcribe`
* **Beam Size:** `5`
* **Word Timestamps:** `True`
* **Decoding Temperature:** `0.0`
* **Prompt Conditioning:** **None** (strict blind evaluation)

---

## 2. Installation & Requirements

Install `faster-whisper` into your Python environment:

```bash
pip install faster-whisper
```

### Hardware & Compute Types
* **NVIDIA GPU (CUDA):** Uses `float16` for high throughput and full precision.
* **Apple Silicon (Mac) / CPU:** Uses `int8` quantization for fast execution with minimal RAM usage (~1.5–2.5 GB).

Device selection is handled automatically by `transcribe_spanish.py`.

---

## 3. Pre-Downloading Model Weights

To pre-download and cache the Whisper `large-v3` model weights (approx. 3.1 GB) before running evaluations:

```bash
python transcribe_spanish.py --download-only
```

Weights will be cached locally in Hugging Face Hub cache (`~/.cache/huggingface/hub/`) and re-used for all subsequent transcriptions.

---

## 4. Transcribing Audio

### A. Transcribe a Single Generated WAV
```bash
python transcribe_spanish.py output_audio.wav
```
This generates:
* `output_audio_whisper.txt`: Clean, human-readable plain text.
* `output_audio_whisper.json`: Word-level timestamps, confidence probabilities, and segment metrics.

### B. Transcribe with Custom Output Paths
```bash
python transcribe_spanish.py audio.wav --output-json my_transcript.json --output-txt my_transcript.txt
```

### C. Transcribe an Entire Folder of Audio Files
```bash
python transcribe_spanish.py ./recordings/
```

---

## 5. Programmatic Python API

You can also use the Whisper evaluator directly inside your Python code:

```python
from transcribe_spanish import get_whisper_model, transcribe_audio

# Load model (auto-selects CUDA float16 or CPU int8)
model = get_whisper_model(model_size="large-v3")

# Transcribe Spanish audio
result = transcribe_audio(model, "audio.wav")

print("Spoken text:", result["text"])
for segment in result["segments"]:
    print(f"[{segment['start']}s - {segment['end']}s] {segment['text']}")
```

---

## 6. Important Spanish ASR Interpretation Notes

When evaluating Spanish TTS output with Whisper, keep in mind these expected formatting differences:
1. **Numbers to Numerals:** Whisper often transcribes spoken numbers as digits (e.g., spoken *"veinticinco dólares"* $\to$ transcribed *"25 dólares"*). This is standard ASR formatting, not a TTS error.
2. **Letter Names to Symbols:** Spoken Spanish letter names (*"eñe"*, *"jota"*, *"ge"*, *"be"*) are typically transcribed as characters (*"ñ"*, *"j"*, *"g"*, *"b"*).
3. **Accent Mark Headword Bias:** For isolated words differing only by accent marks (*"término" / "termino" / "terminó"*), Whisper’s language model tends to output the most frequent dictionary headword.
