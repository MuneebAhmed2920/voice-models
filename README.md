# 🎙️ Voice Package: Clara (Female Customer Service Representative) [`voice_clara`]

## Overview
- **Persona Name**: Clara
- **Voice ID**: `voice_clara` (Aliases: `clara`, `female_1`, `voice_female_1`)
- **Gender**: Female
- **Language / Accent**: `en-US` / General American (Conversational Telephone Representative)
- **Breathing Sound**: Integrated physiological airway inhalation sound (`prompt/breath_organic.wav`)
- **Sampling Rate**: 24,000 Hz (supports 8 kHz Telephony and 16 kHz Wideband resampling)
- **Default Speed**: `0.92` (Natural conversational speaking cadence)
- **Default Guidance Scale**: `2.2`

---

## 📁 Package Layout
```
voices/voice_clara/
├── prompt/
│   ├── voice_prompt.pt        # Calibrated OmniVoice prompt tensor
│   ├── reference.wav          # Master reference speech recording
│   └── breath_organic.wav     # Extracted organic airway inhalation breath
├── config/
│   ├── config.json            # Hyperparameters and runtime speed calibration
│   └── metadata.json          # License and provenance metadata
├── samples/                   # Generated evaluation benchmark audio samples
│   ├── clara_01_conversation.wav
│   ├── clara_02_question.wav
│   ├── clara_03_numbers.wav
│   ├── clara_04_money.wav
│   ├── clara_05_phone_number.wav
│   ├── clara_06_date_time.wav
│   ├── clara_07_long_sentence.wav
│   └── clara_08_breathing.wav
├── tests/
│   └── test_voice.py          # Standalone verification script
├── __init__.py                # Package interface
├── LICENSE                    # Permissive open license
└── README.md                  # Documentation
```

---

## 🚀 Usage

### Python API
```python
from runtime.generate_speech import get_engine

engine = get_engine()
wav_bytes, metrics = engine.generate(
    text="Hello! Thank you for calling. I can certainly help you with that.",
    voice="voice_clara",
    speed=0.92
)

with open("output.wav", "wb") as f:
    f.write(wav_bytes)
```
