# 👩 Saved Female Voice (Dr. Sara)

This dedicated folder contains your exact saved female voice profile frozen into ready-to-use acoustic prompt tensors. You can use it to generate any speech anytime.

---

## 📁 Files in this Folder

* **`female_1_prompt.pt`** — Precomputed acoustic prompt tensor (0 ms loading overhead).
* **`female_1_reference.wav`** — Original master reference audio sample.
* **`voice_config.json`** — Voice configuration and acoustic hyperparameters.
* **`generate_speech.py`** — Standalone generator script to synthesize any text.
* **`test_saved_voice.wav`** — Pre-generated test audio.

---

## 🚀 How to Generate Speech with this Voice Anytime

### 1. Direct Command Line:
Synthesize any custom sentence and save it to a `.wav` file:
```bash
/opt/homebrew/bin/python3.11 saved_female_voice/generate_speech.py "Hello! Your confirmation code is 8842." -o my_message.wav
```

### 2. Play Directly Through Your Mac Speaker (`-p` flag):
```bash
/opt/homebrew/bin/python3.11 saved_female_voice/generate_speech.py "Thank you for calling customer care!" -p
```

### 3. Interactive Live Console:
Launch interactive mode to type multiple sentences and listen live:
```bash
/opt/homebrew/bin/python3.11 saved_female_voice/generate_speech.py --interactive
```

### 4. Telephony 8000 Hz Mode (for VICIDial / Phone Calls):
```bash
/opt/homebrew/bin/python3.11 saved_female_voice/generate_speech.py "Your total is $129.50." -sr 8000 -o phone_audio.wav
```
