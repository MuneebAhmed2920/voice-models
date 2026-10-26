# 🎙️ Master Directive for Antigravity: Arabic Human-Realistic Voice Model Blueprint

You are an expert Speech AI Engineer and Audio Signal Processing specialist.

Your task is to build, train, and configure the **Arabic Voice Model** using the proven humanization pipeline developed for the Clara Voice Model (`voice_clara`).

All local audio assets, reference recordings, prompt tensors, and configuration files are located on this system. You can fetch and copy them directly from the absolute file locations listed below.

---

## 📍 1. Absolute Local Asset Locations (Direct System Paths)

Antigravity can directly read and copy from these exact paths:

### A. Primary Organic Breathing Assets (Extracted from Clara):
- **Clara Master Inhalation Audio (`24kHz`)**:
  `/Users/haseebahmed/agent/clara/clara-voice-model/prompt/breath_organic.wav`
  *(Also available in this kit at: `/Users/haseebahmed/agent/arabic_voice_model_kit/prompt/breath_organic.wav`)*
- **All Master Emotion Breathing Audio Suite (`24k / 16k / 8k`)**:
  `/Users/haseebahmed/agent/arabic_breath_assets/`
  - `/Users/haseebahmed/agent/arabic_breath_assets/breath_organic.wav` (Neutral Inhalation, 200ms)
  - `/Users/haseebahmed/agent/arabic_breath_assets/breath_excited.wav` (Short Eager Inhalation, 100ms)
  - `/Users/haseebahmed/agent/arabic_breath_assets/breath_sympathetic.wav` (Deep Soft Inhalation, 220ms)
  - `/Users/haseebahmed/agent/arabic_breath_assets/breath_emotional.wav` (Sincere Resonant Inhalation, 160ms)

### B. Clara Reference Recording & Precomputed Prompt Tensors:
- **Clara Studio Audio Reference (`24kHz`)**:
  `/Users/haseebahmed/agent/clara/clara-voice-model/prompt/reference.wav`
- **Clara Baseline Neutral Prompt Tensor**:
  `/Users/haseebahmed/agent/clara/clara-voice-model/prompt/voice_prompt.pt`
- **Clara Excited Prompt Tensor (`[excited]`)**:
  `/Users/haseebahmed/agent/clara/clara-voice-model/prompt/voice_prompt_excited.pt`
- **Clara Sympathetic Prompt Tensor (`[sympathetic]`)**:
  `/Users/haseebahmed/agent/clara/clara-voice-model/prompt/voice_prompt_sympathetic.pt`
- **Clara Emotional Prompt Tensor (`[emotional]`)**:
  `/Users/haseebahmed/agent/clara/clara-voice-model/prompt/voice_prompt_emotional.pt`

### C. Other Reference Speakers (For Multi-Pitch Conditioning):
- **Alex (Male Tenor, F0 160 Hz)**: `/Users/haseebahmed/agent/omni voice 2 /voices/voice_alex/`
- **David (Male Baritone, F0 120 Hz)**: `/Users/haseebahmed/agent/omni voice 2 /voices/voice_david/`
- **Michael (Male Bass-Baritone, F0 108 Hz)**: `/Users/haseebahmed/agent/omni voice 2 /voices/voice_michael/`
- **Emma (Female Empathetic, F0 186 Hz)**: `/Users/haseebahmed/agent/omni voice 2 /voices/voice_emma/`
- **Sarah (Female Sharp, F0 217 Hz)**: `/Users/haseebahmed/agent/omni voice 2 /voices/voice_sarah/`
- **Olivia (Female Upbeat, F0 187 Hz)**: `/Users/haseebahmed/agent/omni voice 2 /voices/voice_olivia/`

---

## 🌟 2. The 7 Pillars that Make Clara Sound Human (Replicate for Arabic)

To eliminate any synthetic robotic TTS artifacts and make the Arabic voice sound like a real person on a telephone call:

### Pillar 1: Physiological Turn-Initial Airway Inhalation
- A real human speaker always takes a natural airway breath before speaking.
- Prepend the mastered `breath_organic.wav` with an emotion-dependent duration and gain envelope before the synthesized waveform.

### Pillar 2: Conversational Cadence & Speed Tuning (`0.90x – 0.94x`)
- **Default Speed**: Set `default_speed = 0.92` (instead of standard `1.0x`).
- Standard 1.0x TTS sounds rushed and machine-like. Slowing to 0.92x allows Arabic consonants (حروف الإطباق والحلق) to enunciate with natural warmth.

### Pillar 3: Flow-Matching ODE Guidance Calibration (`2.0 – 2.4`)
- **Guidance Scale**: Set `default_guidance_scale = 2.2`.
- High guidance (> 3.5) causes mechanical pitch rigidity. A guidance scale of 2.0 to 2.4 produces organic micro-inflections in fundamental frequency ($F_0$).

### Pillar 4: Multi-Emotion Prompt Tensor Conditioning
- Generate dedicated flow-matching conditioning tensors for:
  - `neutral` (Base conversational customer care)
  - `excited` / `[متحمس]` (Higher energy, tighter breath, +2 dB presence at 4.2 kHz)
  - `sympathetic` / `[متعاطف]` (Gentle pace, soft low-pass breath, +1.8 dB warmth at 280 Hz)
  - `emotional` / `[مؤثر]` (Heartfelt sincerity, balanced resonance at 320 Hz)

### Pillar 5: Studio Mastering & Proximity EQ Chain
- **High-Pass Filter (`75 Hz`, 12 dB/oct)**: Cleans DC rumble while preserving chest resonance.
- **Warmth Low-Shelf (`250 Hz – 320 Hz`, +1.5 dB)**: Simulates close-mic proximity effect for intimate telephone realism.
- **Consonant Presence (`3.5 kHz – 4.5 kHz`)**: Enhances Arabic pharyngeal and emphatic clarity (ح, ع, ص, ض, ط, ظ, ق).
- **Soft-Knee Peak Limiting**: Prevents digital clipping on PSTN/VoIP phone lines.

### Pillar 6: Natural Conversational Punctuation Micro-Pauses
- Inter-clause comma pauses: `120ms`
- Sentence period/question pauses: `280ms`
- Dash/colon pauses: `150ms`
- Automatic breath injection before coordinating conjunctions (`و`, `فـ`, `لكن`, `يعني`, `طبعاً`) in long clauses.

### Pillar 7: Deterministic Normalization for Arabic Telephony
- Expand dates, times, currencies (e.g. `٤٥ ريال`, `١٢٠ جنيه`, `$1,250`), phone numbers, and percentages into fully articulated spoken words.

---

## 🌬️ 3. Emotion-Tailored Breathing Dynamic Parameters

| Emotion Tag | Arabic Tag | Breath Duration | Gain Factor | Pre-Speech Gap | Acoustic EQ Boost |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Neutral** | `[طبيعي]` | `140ms` | `0.032` (`-28 dBFS`) | `50ms` | Balanced natural vocal tract |
| **Excited** | `[متحمس]` | `95ms` | `0.024` (`-30 dBFS`) | `25ms` | +2.0 dB presence @ 4.2 kHz |
| **Sympathetic** | `[متعاطف]` | `210ms` | `0.045` (`-26 dBFS`) | `75ms` | +1.8 dB low-mid warmth @ 280 Hz |
| **Emotional** | `[مؤثر]` | `165ms` | `0.038` (`-27 dBFS`) | `60ms` | +1.5 dB chest resonance @ 320 Hz |

---

## 💻 4. Python Integration Module (`runtime/arabic_breath_processor.py`)

```python
import os
import re
import numpy as np
import scipy.signal
import soundfile as sf

class ArabicBreathProcessor:
    def __init__(self, breath_path: str = "prompt/breath_organic.wav", sample_rate: int = 24000):
        self.sample_rate = sample_rate
        if not os.path.exists(breath_path):
            fallback = "/Users/haseebahmed/agent/arabic_voice_model_kit/prompt/breath_organic.wav"
            if os.path.exists(fallback):
                breath_path = fallback
            else:
                raise FileNotFoundError(f"Breath audio not found at: {breath_path}")
                
        self.raw_breath, self.raw_sr = sf.read(breath_path)
        if self.raw_sr != self.sample_rate:
            num_samples = int(len(self.raw_breath) * self.sample_rate / self.raw_sr)
            self.raw_breath = scipy.signal.resample(self.raw_breath, num_samples)
            
        self.raw_breath = self.raw_breath / (np.max(np.abs(self.raw_breath)) + 1e-8)

    def parse_arabic_tags(self, text: str):
        emotion = "neutral"
        include_breath = True

        if re.search(r"\[(متحمس|excited|happy)\]", text, re.IGNORECASE):
            emotion = "excited"
        elif re.search(r"\[(متعاطف|sympathetic|apologetic)\]", text, re.IGNORECASE):
            emotion = "sympathetic"
        elif re.search(r"\[(مؤثر|emotional|sincere)\]", text, re.IGNORECASE):
            emotion = "emotional"

        if re.search(r"\[(تنفس|breath|inhalation)\]", text, re.IGNORECASE):
            include_breath = True

        clean_text = re.sub(r"\[(متحمس|متعاطف|مؤثر|تنفس|طبيعي|excited|sympathetic|emotional|breath|pause)\]", "", text, flags=re.IGNORECASE)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        return clean_text, emotion, include_breath

    def create_tailored_breath(self, emotion: str = "neutral") -> np.ndarray:
        params = {
            "excited": {"duration_ms": 95, "gain": 0.024, "pre_gap_ms": 25},
            "sympathetic": {"duration_ms": 210, "gain": 0.045, "pre_gap_ms": 75},
            "emotional": {"duration_ms": 165, "gain": 0.038, "pre_gap_ms": 60},
            "neutral": {"duration_ms": 140, "gain": 0.032, "pre_gap_ms": 50},
        }
        cfg = params.get(emotion, params["neutral"])

        target_len = int((cfg["duration_ms"] / 1000.0) * self.sample_rate)
        breath = scipy.signal.resample(self.raw_breath, target_len) * cfg["gain"]

        fade_in = int(0.015 * self.sample_rate)
        fade_out = int(0.025 * self.sample_rate)
        breath[:fade_in] *= 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_in)))
        breath[-fade_out:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_out)))

        gap_len = int((cfg["pre_gap_ms"] / 1000.0) * self.sample_rate)
        gap = np.zeros(gap_len, dtype=np.float32)

        return np.concatenate([breath, gap])

    def inject_pre_speech_breath(self, speech_waveform: np.ndarray, emotion: str = "neutral", include_breath: bool = True) -> np.ndarray:
        if not include_breath:
            return speech_waveform
        breath_audio = self.create_tailored_breath(emotion)
        return np.concatenate([breath_audio, speech_waveform])
```

---

## 🧪 5. Arabic Human Realism Benchmark Suite

Test these sentences to verify human telephone presence:
1. **Conversational Greeting**: `"أهلاً وسهلاً بحضرتك، تفضل كيف أقدر أساعدك اليوم؟"`
2. **Excited Confirmation (`[متحمس]`)**: `"[متحمس] ألف مبروك! تم تأكيد موعدك بنجاح ليوم الثلاثاء الساعة ٤:٣٠ مساءً."`
3. **Sympathetic Customer Care (`[متعاطف]`)**: `"[متعاطف] أعتذر جداً لحضرتك عن الإزعاج، وأنا هتابع الموضوع بنفسي حتى يتم حله بالكامل."`
4. **Emotional Reassurance (`[مؤثر]`)**: `"[مؤثر] شكراً جزيلاً لثقتك الغالية فينا وشراكتنا الممتدة على مدار السنوات."`
