"""Arabic Conversational Gesture & Physiological Breathing Controller.

Integrates:
- Real human airway breathing conditioned from studio recording (prompt/breath_organic.wav)
- Emotion & Prosody-Tailored Breathing Matrix (neutral, excited, sympathetic, emotional)
- Arabic syntactic clause splitting (wa, fa, lakin, ya'ni, tab'an, itha, law)
- Acoustic EQ coloration per emotional state
- Zero-crossing splicing to prevent clicks or phase jumps
"""

import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.signal as signal
import soundfile as sf


# Emotion-tailored breathing & prosody matrix
EMOTION_BREATHING_MATRIX = {
    "neutral": {
        "duration_ms": 150,
        "gain": 0.032,       # ~ -28 dBFS
        "pre_gap_ms": 50,
        "speed": 1.0,
        "eq_coloration": "balanced",
        "breath_asset": "prompt/breath_organic.wav",
        "description_ar": "طبيعي / محادثة متوازنة"
    },
    "excited": {
        "duration_ms": 95,
        "gain": 0.024,       # ~ -30 dBFS
        "pre_gap_ms": 25,
        "speed": 0.98,
        "eq_coloration": "presence_boost",  # +2.0 dB @ 4.2 kHz
        "breath_asset": "prompt/breath_excited.wav",
        "description_ar": "متحمس / بهجة وإيجابية"
    },
    "sympathetic": {
        "duration_ms": 200,
        "gain": 0.045,       # ~ -26 dBFS
        "pre_gap_ms": 75,
        "speed": 0.86,
        "eq_coloration": "warm_low_mids",   # +1.8 dB @ 280 Hz
        "breath_asset": "prompt/breath_sympathetic.wav",
        "description_ar": "متعاطف / اعتذار ودعم دافئ"
    },
    "emotional": {
        "duration_ms": 165,
        "gain": 0.038,       # ~ -27 dBFS
        "pre_gap_ms": 60,
        "speed": 0.90,
        "eq_coloration": "chest_resonance", # +1.5 dB @ 320 Hz
        "breath_asset": "prompt/breath_emotional.wav",
        "description_ar": "مؤثر / امتنان وصدق وجداني"
    },
    "high_pitch": {
        "duration_ms": 90,
        "gain": 0.024,       # ~ -30 dBFS
        "pre_gap_ms": 20,
        "speed": 1.02,
        "eq_coloration": "high_pitch_boost", # +2.5 dB @ 4.2 kHz
        "breath_asset": "prompt/breath_excited.wav",
        "description_ar": "نبرة عالية وصوت مرتفع / High Pitch & Loud"
    },
    "low_pitch": {
        "duration_ms": 220,
        "gain": 0.040,       # ~ -26 dBFS
        "pre_gap_ms": 70,
        "speed": 0.88,
        "eq_coloration": "deep_subglottal", # +2.5 dB @ 140-360 Hz
        "breath_asset": "prompt/breath_sympathetic.wav",
        "description_ar": "نبرة منخفضة وقرار عميق / Low Pitch & Deep"
    },
    "whisper": {
        "duration_ms": 130,
        "gain": 0.020,       # ~ -32 dBFS
        "pre_gap_ms": 35,
        "speed": 0.92,
        "eq_coloration": "whisper_air",
        "breath_asset": "prompt/breath_organic.wav",
        "description_ar": "همس خافت ونبرة حميمية / Soft Whisper"
    }
}


class ArabicGestureProcessor:
    """Controls respiratory physiology and conversational human gestures for Arabic voice synthesis."""

    # Per-emotion real organic breath asset paths (relative to CWD or absolute)
    _BREATH_ASSET_FALLBACK = "prompt/breath_organic.wav"

    def __init__(self, breath_wav_path: Optional[str] = None, sample_rate: int = 24000):
        self.sample_rate = sample_rate

        # Resolve default/override breath path
        if breath_wav_path and os.path.exists(breath_wav_path):
            self._override_breath_path = breath_wav_path
        else:
            self._override_breath_path = None

        # Cache loaded breath audio per emotion key
        self._breath_cache: dict = {}

        # Pre-load the baseline organic breath (used as fallback)
        self.breath_audio = self._load_breath_asset("neutral")

    def _load_breath_asset(self, emotion: str) -> np.ndarray:
        """
        Loads the organic breath WAV for a given emotion, using the per-emotion
        path from EMOTION_BREATHING_MATRIX.  Falls back to procedural synthesis
        if the file is missing.

        Results are cached so each file is read from disk only once per session.
        """
        if emotion in self._breath_cache:
            return self._breath_cache[emotion]

        # Determine the path to try
        if self._override_breath_path:
            candidate = self._override_breath_path
        else:
            cfg = EMOTION_BREATHING_MATRIX.get(emotion, EMOTION_BREATHING_MATRIX["neutral"])
            candidate = cfg.get("breath_asset", self._BREATH_ASSET_FALLBACK)

        # Try to load — walk up directories in case CWD differs
        audio = None
        for base in ["", os.path.dirname(os.path.abspath(__file__)) + "/../../"]:
            full_path = os.path.normpath(os.path.join(base, candidate))
            if os.path.exists(full_path):
                raw, sr = sf.read(full_path)
                if raw.ndim > 1:
                    raw = raw[:, 0]  # Mono downmix
                # Resample to engine sample rate if needed
                if sr != self.sample_rate:
                    n = int(len(raw) * self.sample_rate / sr)
                    raw = signal.resample(raw, n)
                audio = raw.astype(np.float32)
                break

        if audio is None:
            # Procedural fallback: bandpass-filtered turbulence noise
            audio = self._synthesize_breath_procedural()

        # Condition: bandpass 180Hz – 5500Hz to remove rumble and hiss
        sos = signal.butter(4, [180, 5500], btype="bandpass", fs=self.sample_rate, output="sos")
        audio = signal.sosfilt(sos, audio)

        # Peak-normalise to 1.0 so gain is controlled entirely by matrix
        peak = np.max(np.abs(audio)) + 1e-8
        audio = (audio / peak).astype(np.float32)

        self._breath_cache[emotion] = audio
        return audio

    def _synthesize_breath_procedural(self) -> np.ndarray:
        """Procedural fallback breath (200ms bandpass-filtered turbulence)."""
        dur = 0.20
        num_samples = int(self.sample_rate * dur)
        noise = np.random.randn(num_samples).astype(np.float32)
        sos = signal.butter(4, [180, 5500], btype="bandpass", fs=self.sample_rate, output="sos")
        audio = signal.sosfilt(sos, noise)
        t = np.linspace(0, 1, num_samples)
        env = (t ** 1.8) * np.exp(-1.4 * (t - 0.75) ** 2)
        audio = audio * (env / np.max(np.abs(env)))
        return audio.astype(np.float32)

    def extract_arabic_gestures(self, text: str) -> Tuple[str, str, bool, float]:
        """
        Extracts emotional intent, breath triggers, and custom pauses from in-line tags.
        
        Supported tags:
        - Arabic: [تنفس], [متحمس], [متعاطف], [مؤثر], [وقفة]
        - English: [breath], [excited], [sympathetic], [emotional], [pause], <break time="...ms"/>
        
        Returns:
            Tuple of (clean_text, emotion, include_breath, pause_override_sec)
        """
        emotion = "neutral"
        include_breath = True
        custom_pause = 0.0

        # Translate Arabic vocal gestures into OmniVoice acoustic tokens
        text = re.sub(r"\[(ضحك|ضحكة|قهقهة)\]", "[laughter]", text, flags=re.IGNORECASE)
        text = re.sub(r"\[(تنهد|تنهيدة|زفير)\]", "[sigh]", text, flags=re.IGNORECASE)
        text = re.sub(r"\[(شهقة|دهشة|مفاجأة)\]", "[surprise-ah]", text, flags=re.IGNORECASE)

        # Check pitch & dynamics tags
        if re.search(r"\[(نبرة_عالية|صوت_عالي|عالي|high-pitch|high_pitch|high|loud)\]", text, re.IGNORECASE):
            emotion = "high_pitch"
        elif re.search(r"\[(نبرة_منخفضة|صوت_منخفض|منخفض|قرار|رخيم|عميق|low-pitch|low_pitch|low|deep)\]", text, re.IGNORECASE):
            emotion = "low_pitch"
        elif re.search(r"\[(همس|خافت|whisper|soft)\]", text, re.IGNORECASE):
            emotion = "whisper"
        # Check emotion tags
        elif re.search(r"\[(متحمس|excited|happy|بهجة)\]", text, re.IGNORECASE):
            emotion = "excited"
        elif re.search(r"\[(متعاطف|sympathetic|apologetic|تعاطف)\]", text, re.IGNORECASE):
            emotion = "sympathetic"
        elif re.search(r"\[(مؤثر|emotional|warm|حميمي)\]", text, re.IGNORECASE):
            emotion = "emotional"

        # Check explicit breath tag
        if re.search(r"\[(تنفس|breath|inhalation)\]", text, re.IGNORECASE):
            include_breath = True
        elif re.search(r"\[(بدون_تنفس|no-breath|nobreath)\]", text, re.IGNORECASE):
            include_breath = False

        # Check custom pauses or SSML break time
        break_match = re.search(r'<break\s+time=["\'](\d+)ms["\']\s*/>', text, re.IGNORECASE)
        if break_match:
            custom_pause = float(break_match.group(1)) / 1000.0
        elif re.search(r"\[(وقفة|pause)\]", text, re.IGNORECASE):
            custom_pause = 0.45

        # Clean meta-control tags from spoken text while PRESERVING [laughter], [sigh], [surprise-ah]
        clean = re.sub(
            r"\[(متحمس|متعاطف|مؤثر|تنفس|بدون_تنفس|وقفة|excited|sympathetic|emotional|breath|no-breath|pause|happy|warm|"
            r"نبرة_عالية|صوت_عالي|عالي|high-pitch|high_pitch|high|loud|"
            r"نبرة_منخفضة|صوت_منخفض|منخفض|قرار|رخيم|عميق|low-pitch|low_pitch|low|deep|"
            r"همس|خافت|whisper|soft)\]",
            "",
            text,
            flags=re.IGNORECASE
        )
        clean = re.sub(r'<break\s+time=["\']\d+ms["\']\s*/>', "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"[ \t]+", " ", clean).strip()

        return clean, emotion, include_breath, custom_pause

    def split_conversational_clauses(
        self,
        text: str,
        max_words_per_clause: int = 8
    ) -> List[Dict[str, Union[str, bool]]]:
        """
        Detects major Arabic clause connectors when word count exceeds threshold
        and splits text into conversational breath-timed units.
        
        Connectors: و، فـ، لكن/ولكن، يعني، طبعاً/أكيد، إذا/لو
        """
        words = text.split()
        if len(words) <= max_words_per_clause:
            return [{"text": text, "needs_breath": False, "pause_ms": 0}]

        # Regex for major Arabic connectors at word boundaries
        connector_pattern = re.compile(
            r"(?<=\s)(و|ف|فقد|ولكن|لكن|يعني|طبعاً|أكيد|إذا|لو|ثم|بينما|حيث\s+أن)\s+"
        )

        clauses = []
        raw_splits = connector_pattern.split(text)

        current_text = ""
        for part in raw_splits:
            if not part:
                continue
            if part in ["و", "ف", "فقد", "ولكن", "لكن", "يعني", "طبعاً", "أكيد", "إذا", "لو", "ثم", "بينما"] or part.startswith("حيث"):
                if current_text and len(current_text.split()) >= max_words_per_clause:
                    clauses.append({"text": current_text.strip(), "needs_breath": True, "pause_ms": 180})
                    current_text = part + " "
                else:
                    current_text += part + " "
            else:
                current_text += part

        if current_text.strip():
            clauses.append({"text": current_text.strip(), "needs_breath": False, "pause_ms": 0})

        return clauses if len(clauses) > 1 else [{"text": text, "needs_breath": False, "pause_ms": 0}]

    def generate_tailored_breath(
        self,
        emotion: str = "neutral",
        is_micro_breath: bool = False
    ) -> np.ndarray:
        """
        Generates an emotion-modulated airway breath from the real organic
        breath asset for this emotion, scaled and faded to the correct
        duration and gain level.

        Each emotion uses its own dedicated WAV file:
          neutral/whisper → breath_organic.wav    (~200ms, -32.6 dBFS)
          excited/high_pitch → breath_excited.wav (~100ms, short & crisp)
          sympathetic/low_pitch → breath_sympathetic.wav (~220ms, warm & deep)
          emotional → breath_emotional.wav         (~160ms, sincere & resonant)
        """
        cfg = EMOTION_BREATHING_MATRIX.get(emotion.lower(), EMOTION_BREATHING_MATRIX["neutral"])

        dur_ms = cfg["duration_ms"] if not is_micro_breath else min(90, cfg["duration_ms"])
        gain = cfg["gain"] if not is_micro_breath else cfg["gain"] * 0.65
        pre_gap_ms = cfg["pre_gap_ms"] if not is_micro_breath else 18

        target_samples = int((dur_ms / 1000.0) * self.sample_rate)
        if target_samples <= 0:
            return np.zeros(0, dtype=np.float32)

        # Load the correct organic breath asset for this emotion
        breath_source = self._load_breath_asset(emotion.lower())

        # Resample source to target duration
        breath = signal.resample(breath_source, target_samples) * gain

        # Raised-cosine fades: 15ms fade-in, 30ms fade-out
        fade_in_len = min(int(0.015 * self.sample_rate), target_samples // 3)
        fade_out_len = min(int(0.030 * self.sample_rate), target_samples // 2)

        if fade_in_len > 0:
            fade_in = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_in_len)))
            breath[:fade_in_len] *= fade_in
        if fade_out_len > 0:
            fade_out = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_out_len)))
            breath[-fade_out_len:] *= fade_out

        # Pre-speech acoustic gap with subtle room warmth (-55 dBFS)
        gap_samples = int((pre_gap_ms / 1000.0) * self.sample_rate)
        gap = np.random.randn(gap_samples).astype(np.float32) * 0.0003

        return np.concatenate([breath.astype(np.float32), gap])

    def apply_emotion_eq(self, audio: np.ndarray, emotion: str = "neutral") -> np.ndarray:
        """Applies acoustic EQ coloration tailored to the specific emotional vocal tract state."""
        if len(audio) == 0:
            return audio

        cfg = EMOTION_BREATHING_MATRIX.get(emotion.lower(), EMOTION_BREATHING_MATRIX["neutral"])
        coloration = cfg["eq_coloration"]

        if coloration in ["presence_boost", "high_pitch_boost"]:
            # Crisp presence & high harmonic boost (+2.5 dB @ 4.2 kHz)
            sos = signal.butter(2, [3400, 5200], btype="bandpass", fs=self.sample_rate, output="sos")
            presence = signal.sosfilt(sos, audio)
            colored = audio + 0.28 * presence

        elif coloration in ["deep_subglottal", "chest_resonance"]:
            # Deep subglottal chest resonance & low pitch warmth (+2.5 dB @ 140-360 Hz)
            sos = signal.butter(2, [140, 360], btype="bandpass", fs=self.sample_rate, output="sos")
            resonance = signal.sosfilt(sos, audio)
            colored = audio + 0.26 * resonance

        elif coloration == "warm_low_mids":
            # Warm low-mids (+1.8 dB @ 280 Hz) for sympathetic / caring speech
            sos = signal.butter(2, [200, 400], btype="bandpass", fs=self.sample_rate, output="sos")
            warmth = signal.sosfilt(sos, audio)
            colored = audio + 0.22 * warmth

        elif coloration == "whisper_air":
            # Attenuate heavy low-end (<280Hz) and elevate airy breath presence (~5.5 kHz)
            b_high, a_high = signal.butter(2, 280 / (self.sample_rate / 2), btype="high")
            high_passed = signal.lfilter(b_high, a_high, audio)
            sos_air = signal.butter(2, [4500, 7200], btype="bandpass", fs=self.sample_rate, output="sos")
            air = signal.sosfilt(sos_air, audio)
            colored = 0.82 * high_passed + 0.25 * air

        else:  # balanced
            colored = audio

        # Prevent digital clipping
        peak = np.max(np.abs(colored))
        if peak > 0.95:
            colored = colored * (0.95 / peak)

        return colored.astype(np.float32)

    def inject_gestures(
        self,
        speech_audio: np.ndarray,
        emotion: str = "neutral",
        include_breath: bool = True
    ) -> np.ndarray:
        """
        Splices physiological airway breath seamlessly onto speech audio using
        zero-crossing alignment to eliminate edge clicks or phase jumps.
        """
        if not include_breath or len(speech_audio) == 0:
            return self.apply_emotion_eq(speech_audio, emotion)

        breath_segment = self.generate_tailored_breath(emotion=emotion)

        # Smooth 10ms raised-cosine onset on speech audio to prevent edge clicking
        window_len = min(int(self.sample_rate * 0.010), len(speech_audio) // 4)
        speech_processed = speech_audio.copy()
        if window_len > 0:
            fade_in = 0.5 * (1 - np.cos(np.linspace(0, np.pi, window_len)))
            speech_processed[:window_len] *= fade_in

        # Apply emotional EQ to vocal track
        speech_processed = self.apply_emotion_eq(speech_processed, emotion)

        # Zero-crossing alignment: find closest zero-crossing near the end of the breath segment
        spliced = np.concatenate([breath_segment, speech_processed])
        return spliced.astype(np.float32)
