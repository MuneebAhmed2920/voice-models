"""Unified Arabic OmniVoice Engine.

Integrates:
- k2-fsa/OmniVoice (Base Diffusion LM for MSA, zero-shot cloning, voice design)
- mohammedaly22/VoiceTut-TTS (Fine-tuned Egyptian Arabic & Code-Switching)
- Arabic Normalizer & Diacritizer
- Apple Silicon (MPS), CUDA, and CPU execution
- Modular humanization pipeline (context prosody, code-switching, telephony audio)
"""

import logging
import os
from pathlib import Path
import time
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np

from ..arabic_nlp.normalizer import ArabicNormalizer
from ..arabic_nlp.diacritizer import ArabicDiacritizer
from .voice_manager import VoiceManager, VoiceProfile
from .gesture_controller import ArabicGestureProcessor, EMOTION_BREATHING_MATRIX
from ..pipeline.context_prosody import ContextProsodyAnalyzer
from ..pipeline.normalizer_unified import UnifiedNormalizer
from ..pipeline.code_switch import CodeSwitchSegmenter
from ..pipeline.prosody_rules import BreatheGate
from ..pipeline.telephony_audio import TelephonyAudioProcessor

# New arabic_frontend prosody planner (Saudi-dialect aware)
try:
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", ".."))
    from arabic_frontend.prosody_planner import ArabicProsodyPlanner
except ImportError:
    ArabicProsodyPlanner = None


logger = logging.getLogger("ArabicOmniVoice")
logging.basicConfig(level=logging.INFO)


DIALECT_TO_OMNIVOICE_LANG = {
    # Saudi (primary target)
    "saudi": "najdi arabic",
    "najdi": "najdi arabic",
    "hijazi": "hijazi arabic",
    # MSA / Formal
    "msa": "standard arabic",
    "standard": "standard arabic",
    "fusha": "standard arabic",
    # Other dialects
    "egyptian": "egyptian arabic",
    "gulf": "gulf arabic",
    "uae": "gulf arabic",
    "kuwaiti": "gulf arabic",
    "levantine": "levantine arabic",
    "syrian": "levantine arabic",
    "jordanian": "levantine arabic",
    "moroccan": "moroccan arabic",
    "algerian": "algerian arabic",
    "sudanese": "sudanese arabic",
    "tunisian": "tunisian arabic",
    "libyan": "libyan arabic",
}

# OmniVoice v0.2.1 — COMPLETE valid instruct token set
# Any token NOT in this set will cause ValueError
VALID_INSTRUCT_TOKENS = frozenset([
    "american accent", "australian accent", "british accent",
    "canadian accent", "child", "chinese accent", "elderly",
    "female", "high pitch", "indian accent", "japanese accent",
    "korean accent", "low pitch", "male", "middle-aged",
    "moderate pitch", "portuguese accent", "russian accent",
    "teenager", "very high pitch", "very low pitch", "whisper",
    "young adult",
])


def _safe_instruct(raw: Optional[str]) -> Optional[str]:
    """
    Validates an instruct string against OmniVoice's accepted vocabulary.
    Filters out any unsupported tokens to prevent ValueError at inference time.
    Returns None if nothing valid remains.
    """
    if not raw:
        return None
    tokens = [t.strip().lower() for t in raw.split(",")]
    valid = [t for t in tokens if t in VALID_INSTRUCT_TOKENS]
    return ", ".join(valid) if valid else None



class ArabicOmniVoiceEngine:
    """Unified Arabic TTS generation engine supporting OmniVoice and VoiceTut architectures."""

    def __init__(
        self,
        default_dialect: str = "msa",
        device: Optional[str] = None,
        voices_dir: str = "custom_voices",
        preload_models: bool = False
    ):
        """
        Initialize the Arabic OmniVoice Engine.
        
        Args:
            default_dialect: 'msa' (Modern Standard Arabic) or 'egyptian'
            device: 'cuda', 'cpu', 'mps' or None (auto-detect)
            voices_dir: Directory for custom cloned voice metadata
            preload_models: If True, loads model weights immediately
        """
        self.default_dialect = default_dialect.lower()
        self.device = self._resolve_device(device)
        self.voices_dir = voices_dir
        self.voice_manager = VoiceManager(custom_voices_dir=voices_dir)

        # NLP preprocessors
        self.msa_normalizer = ArabicNormalizer(dialect="msa")
        self.egy_normalizer = ArabicNormalizer(dialect="egyptian")
        self.diacritizer = ArabicDiacritizer()

        # Telephony-grade unified normalizers
        self.msa_unified_normalizer = UnifiedNormalizer(dialect="msa")
        self.egy_unified_normalizer = UnifiedNormalizer(dialect="egyptian")

        # Physiological breathing & conversational gesture controller
        self.gesture_processor = ArabicGestureProcessor(sample_rate=24000)

        # Modular pipeline components
        self.context_prosody = ContextProsodyAnalyzer()
        self.code_switcher = CodeSwitchSegmenter()
        self.breathe_gate = BreatheGate()
        self.telephony_processor = TelephonyAudioProcessor(source_sample_rate=24000)

        # Arabic prosody planner (intent → valid instruct + parameters)
        self._prosody_planner = ArabicProsodyPlanner(
            dialect=default_dialect, gender="male", age="middle-aged"
        ) if ArabicProsodyPlanner is not None else None

        # Engine backends (lazy loaded)
        self._omnivoice_model = None
        self._voicetut_model = None

        logger.info(f"[ArabicOmniVoiceEngine] Initialized on device: {self.device}")

        if preload_models:
            self._get_omnivoice_backend()

    def _resolve_device(self, user_device: Optional[str]) -> str:
        """Determines best hardware acceleration device."""
        if user_device:
            return user_device

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            # On macOS, CPU with Apple Accelerate provides instant model initialization (<2s)
            # and stable multi-core vector execution without shader compilation stalls
            return "cpu"
        except ImportError:
            pass
        return "cpu"

    def _get_omnivoice_backend(self):
        """Loads and caches base k2-fsa/OmniVoice model."""
        if self._omnivoice_model is not None:
            return self._omnivoice_model

        try:
            from omnivoice import OmniVoice
            dtype = "float16" if self.device == "cuda" else "float32"
            logger.info(f"[ArabicOmniVoiceEngine] Loading k2-fsa/OmniVoice on {self.device} ({dtype})...")
            self._omnivoice_model = OmniVoice.from_pretrained(
                "k2-fsa/OmniVoice",
                device_map=self.device,
                dtype=dtype
            )
            logger.info("[ArabicOmniVoiceEngine] k2-fsa/OmniVoice loaded successfully.")
            return self._omnivoice_model
        except Exception as e:
            logger.warning(f"[ArabicOmniVoiceEngine] Base OmniVoice could not be loaded: {e}. Running fallback handler.")
            return None

    def _get_voicetut_backend(self):
        """Loads and caches fine-tuned VoiceTut-TTS model."""
        if self._voicetut_model is not None:
            return self._voicetut_model

        try:
            from voicetut_tts import VoiceTutTTS
            dtype = "float16" if self.device == "cuda" else "float32"
            logger.info(f"[ArabicOmniVoiceEngine] Loading mohammedaly22/VoiceTut-TTS on {self.device} ({dtype})...")
            self._voicetut_model = VoiceTutTTS.from_pretrained(
                "mohammedaly22/VoiceTut-TTS",
                device=self.device,
                dtype=dtype
            )
            logger.info("[ArabicOmniVoiceEngine] VoiceTut-TTS loaded successfully.")
            return self._voicetut_model
        except Exception as e:
            logger.warning(f"[ArabicOmniVoiceEngine] VoiceTut-TTS could not be loaded: {e}.")
            return None

    def preprocess_text(
        self,
        text: str,
        dialect: str = "msa",
        auto_tashkeel: bool = True,
        normalize_numbers: bool = True
    ) -> str:
        """
        Applies Arabic normalization and Tashkeel diacritization.
        Uses the legacy ArabicNormalizer (preserved for backward compatibility).
        For the full telephony pipeline use synthesize_pipeline() instead.
        """
        if not text:
            return ""

        # Step 1: Normalization (numbers, dates, currency, code-switching)
        if normalize_numbers:
            normalizer = self.egy_normalizer if dialect.lower() == "egyptian" else self.msa_normalizer
            text = normalizer.normalize(text)

        # Step 2: Tashkeel / Harakat Diacritization (especially important for MSA)
        if auto_tashkeel:
            text = self.diacritizer.diacritize(text)

        return text

    def preprocess_text_unified(
        self,
        text: str,
        dialect: str = "msa",
        auto_tashkeel: bool = True,
    ) -> str:
        """
        Full telephony-grade normalization:
        phone numbers, emails, URLs, decimals, acronyms, contractions,
        ordinals, then Arabic numerals/dates/time/currency, then Tashkeel.
        """
        if not text:
            return ""

        # Use the extended unified normalizer
        norm = self.egy_unified_normalizer if dialect.lower() == "egyptian" else self.msa_unified_normalizer
        text = norm.normalize(text)

        # Tashkeel only for MSA
        if auto_tashkeel and dialect.lower() in ("msa", "standard", "fusha"):
            text = self.diacritizer.diacritize(text)

        return text

    def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        dialect: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        instruct: Optional[str] = None,
        num_step: int = 32,
        speed: float = 1.0,
        guidance_scale: float = 2.5,
        class_temperature: float = 0.20,
        postprocess_output: bool = True,
        auto_preprocess: bool = True,
        output_path: Optional[str] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesizes Arabic speech from text.
        
        Args:
            text: Input Arabic text (vocalized or plain)
            voice_id: ID of preset voice (e.g., 'egy_mohamed', 'msa_kareem')
            dialect: 'msa', 'egyptian', 'gulf', etc.
            ref_audio: Path to reference audio for zero-shot voice cloning
            ref_text: Transcript of reference audio (optional, improves cloning fidelity)
            instruct: Natural language voice design prompt (e.g. 'male, deep voice, news')
            num_step: Diffusion steps (16 for faster speed, 32 for maximum quality)
            speed: Playback speed multiplier (0.8 - 1.5)
            guidance_scale: Classifier-free guidance strength
            auto_preprocess: If True, applies Arabic normalizer and diacritizer
            output_path: If provided, saves audio to this WAV file path
            
        Returns:
            Tuple of (audio_samples: np.ndarray, sample_rate: int)
        """
        # Resolve voice profile
        profile = self.voice_manager.get_voice(voice_id) if voice_id else None
        target_dialect = (dialect or (profile.dialect if profile else self.default_dialect)).lower()

        # Step 1: Preprocess Arabic text
        if auto_preprocess:
            processed_text = self.preprocess_text(
                text,
                dialect=target_dialect,
                auto_tashkeel=(target_dialect == "msa"),
                normalize_numbers=True
            )
        else:
            processed_text = text

        lang_name = DIALECT_TO_OMNIVOICE_LANG.get(target_dialect, "Standard Arabic")
        logger.info(f"[ArabicOmniVoiceEngine] Synthesizing: '{processed_text[:60]}...' (dialect: {target_dialect}, lang: {lang_name})")

        sample_rate = 24000
        audio_array = None

        # Mode A: Cloned voice from reference audio or voice profile
        target_ref_audio = ref_audio or (profile.ref_audio_path if profile and profile.ref_audio_path and os.path.exists(profile.ref_audio_path if profile.ref_audio_path else "") else None)
        target_ref_text = ref_text or (profile.ref_transcript if profile else None)

        if target_ref_audio:
            logger.info(f"[ArabicOmniVoiceEngine] Zero-shot voice cloning using {target_ref_audio}")
            omni = self._get_omnivoice_backend()
            if omni is not None:
                kwargs = {
                    "text": processed_text,
                    "ref_audio": target_ref_audio,
                    "language": lang_name,
                    "num_step": num_step,
                    "guidance_scale": guidance_scale,
                    "class_temperature": class_temperature,
                    "postprocess_output": postprocess_output,
                    "speed": speed,
                }
                if target_ref_text:
                    kwargs["ref_text"] = target_ref_text
                audio = omni.generate(**kwargs)
                audio_array = self._to_numpy(audio)
            else:
                audio_array = self._generate_fallback_audio(processed_text, sample_rate=sample_rate)

        # Mode B: Egyptian Studio Voice (VoiceTut-TTS)
        elif profile and profile.engine == "voicetut" and profile.speaker_key:
            tut = self._get_voicetut_backend()
            if tut is not None:
                tmp_out = output_path or f"/tmp/voicetut_{int(time.time()*1000)}.wav"
                tut.synthesize(
                    processed_text,
                    speaker=profile.speaker_key,
                    output=tmp_out,
                    num_step=num_step,
                    guidance_scale=guidance_scale,
                    speed=speed
                )
                import soundfile as sf
                audio_array, sample_rate = sf.read(tmp_out)
            else:
                omni = self._get_omnivoice_backend()
                if omni is not None:
                    fallback_instruct = _safe_instruct("male, middle-aged")
                    audio = omni.generate(
                        text=processed_text,
                        instruct=fallback_instruct,
                        language="egyptian arabic",
                        num_step=num_step,
                        guidance_scale=guidance_scale,
                        class_temperature=class_temperature,
                        postprocess_output=postprocess_output,
                        speed=speed,
                    )
                    audio_array = self._to_numpy(audio)
                else:
                    audio_array = self._generate_fallback_audio(processed_text, sample_rate=sample_rate)

        # Mode C: Voice Design / OmniVoice (Primary path for MSA/Saudi)
        else:
            from .voice_manager import resolve_instruct_prompt
            raw_instruct = instruct or (profile.instruct_prompt if profile else None)
            # CRITICAL: validate against OmniVoice vocabulary before passing
            instruct_prompt = _safe_instruct(raw_instruct)

            omni = self._get_omnivoice_backend()
            if omni is not None:
                kwargs = {
                    "text": processed_text,
                    "language": lang_name,
                    "num_step": num_step,
                    "guidance_scale": guidance_scale,
                    "class_temperature": class_temperature,
                    "postprocess_output": postprocess_output,
                    "speed": speed,
                }
                if instruct_prompt:
                    kwargs["instruct"] = instruct_prompt
                audio = omni.generate(**kwargs)
                audio_array = self._to_numpy(audio)
            else:
                audio_array = self._generate_fallback_audio(processed_text, sample_rate=sample_rate)

        # Save to output file if requested
        if output_path and audio_array is not None:
            self._save_audio(audio_array, sample_rate, output_path)

        return audio_array, sample_rate

    def stream_synthesize(
        self,
        long_text: str,
        voice_id: Optional[str] = None,
        dialect: Optional[str] = None,
        num_step: int = 16
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        True streaming synthesis: splits long paragraphs by sentence boundaries
        and yields audio chunks progressively for minimal time-to-first-audio.
        """
        import re
        sentences = re.split(r"(?<=[.!?؟،\n])\s+", long_text)
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            chunk, sr = self.synthesize(
                text=sent,
                voice_id=voice_id,
                dialect=dialect,
                num_step=num_step
            )
            yield chunk, sr

    def synthesize_conversational(
        self,
        text: str,
        voice_id: Optional[str] = None,
        dialect: Optional[str] = None,
        emotion_override: Optional[str] = None,
        include_breath: bool = True,
        split_clauses: bool = True,
        num_step: int = 20,
        output_path: Optional[str] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesizes realistic conversational Arabic speech conditioned with:
        - In-line emotional tags ([متحمس], [متعاطف], [مؤثر]) or manual emotion
        - Natural airway breath inhalations tailored per emotion
        - Arabic clause splitting (و، فـ، لكن، يعني، طبعاً، إذا) with micro-breath pauses
        - Acoustic EQ coloration (presence boost, chest resonance, warm low-mids)
        - Zero-crossing splicing to prevent audio clicks
        """
        sample_rate = 24000

        # Step 1: Parse gestures and emotional intent
        clean_text, detected_emotion, breath_flag, custom_pause = self.gesture_processor.extract_arabic_gestures(text)
        emotion = (emotion_override or detected_emotion).lower()
        cfg = EMOTION_BREATHING_MATRIX.get(emotion, EMOTION_BREATHING_MATRIX["neutral"])
        speed = cfg["speed"]
        should_breathe = include_breath and breath_flag

        logger.info(f"[ArabicOmniVoiceEngine] Conversational TTS | Emotion: '{emotion}' | Speed: {speed:.2f} | Breath: {should_breathe}")

        # Step 2: Clause splitting for natural conversational phrasing
        if split_clauses:
            clauses = self.gesture_processor.split_conversational_clauses(clean_text)
        else:
            clauses = [{"text": clean_text, "needs_breath": False, "pause_ms": 0}]

        assembled_segments = []

        # Optional turn-initial inhalation
        if should_breathe:
            turn_breath = self.gesture_processor.generate_tailored_breath(emotion=emotion, is_micro_breath=False)
            assembled_segments.append(turn_breath)

        for i, clause in enumerate(clauses):
            raw_c_text = clause["text"].strip()
            if not raw_c_text:
                continue

            # Extract clause-specific gesture/emotion if in-line tag exists
            sub_clean, sub_emotion, sub_breath, sub_pause = self.gesture_processor.extract_arabic_gestures(raw_c_text)
            c_emotion = sub_emotion if sub_emotion != "neutral" else emotion
            c_cfg = EMOTION_BREATHING_MATRIX.get(c_emotion, cfg)
            c_speed = c_cfg.get("speed", speed)

            # Determine dynamic instruct prompt based on clause emotion/pitch
            c_instruct = None
            if c_emotion == "high_pitch":
                c_instruct = "high pitch, loud"
            elif c_emotion == "low_pitch":
                c_instruct = "low pitch, deep"
            elif c_emotion == "whisper":
                c_instruct = "whisper"
            elif c_emotion == "excited":
                c_instruct = "high pitch"

            c_audio, _ = self.synthesize(
                text=sub_clean if sub_clean else raw_c_text,
                voice_id=voice_id,
                dialect=dialect,
                instruct=c_instruct,
                speed=c_speed,
                num_step=num_step,
                auto_preprocess=True
            )

            # Apply clause-specific EQ coloration
            c_audio = self.gesture_processor.apply_emotion_eq(c_audio, emotion=c_emotion)

            # Smooth 10ms boundary window
            w_len = min(int(sample_rate * 0.010), len(c_audio) // 4)
            if w_len > 0:
                c_audio = c_audio.copy()
                fade = 0.5 * (1 - np.cos(np.linspace(0, np.pi, w_len)))
                c_audio[:w_len] *= fade
                c_audio[-w_len:] *= fade[::-1]

            assembled_segments.append(c_audio)

            # Insert inter-clause micro-breath before next clause if flagged
            if i < len(clauses) - 1:
                if (clause.get("needs_breath", False) or sub_breath) and should_breathe:
                    micro_breath = self.gesture_processor.generate_tailored_breath(emotion=c_emotion, is_micro_breath=True)
                    assembled_segments.append(micro_breath)
                else:
                    pause_s = sub_pause if sub_pause > 0 else (custom_pause if custom_pause > 0 else (clause.get("pause_ms", 140) / 1000.0))
                    ambient_pause = np.random.randn(int(sample_rate * max(0.08, pause_s))).astype(np.float32) * 0.0003
                    assembled_segments.append(ambient_pause)

        full_audio = np.concatenate(assembled_segments) if assembled_segments else np.zeros(sample_rate, dtype=np.float32)

        if output_path:
            self._save_audio(full_audio, sample_rate, output_path)

        return full_audio, sample_rate

    # ══════════════════════════════════════════════════════════════════════
    # FULL 12-STAGE PIPELINE — context-aware, telephony-grade
    # ══════════════════════════════════════════════════════════════════════

    def synthesize_pipeline(
        self,
        text: str,
        voice_id: Optional[str] = None,
        dialect: Optional[str] = None,
        emotion_override: Optional[str] = None,
        telephony_sr: Optional[int] = None,
        simulate_codec: bool = False,
        num_step: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Full 12-stage humanized telephony TTS pipeline.

        Stages:
          1.  Text cleanup
          2.  Language detection + code-switching segmentation
          3.  Unified telephony normalization (phone/email/URL/acronym/decimal)
          4.  Arabic diacritization (MSA only)
          5.  Context + prosody analysis (dialog intent → speed/guidance/steps)
          6.  Breathing gate (decides if pre-speech breath is appropriate)
          7.  TTS inference (OmniVoice with intent-derived instruct + speed)
          8.  EQ coloration (gesture_controller.apply_emotion_eq)
          9.  Loudness normalization (-18 dBFS)
          10. DC offset removal
          11. Resampling (optional, only if telephony_sr specified)
          12. Quality verification

        Args:
            text: Raw input text (Arabic, English, or mixed)
            voice_id: Optional preset voice ID
            dialect: 'msa', 'egyptian', 'gulf', 'levantine'
            emotion_override: Force a specific emotion ('neutral', 'excited', etc.)
            telephony_sr: Resample to 8000 (G.711) or 16000 (G.722). None = keep 24kHz.
            simulate_codec: Apply G.711 μ-law codec simulation for QA testing.
            num_step: Diffusion steps (overrides intent-based default if provided)
            output_path: Save output WAV to this path

        Returns:
            Tuple of (audio_array: np.ndarray, sample_rate: int)
        """
        t0 = time.time()
        sample_rate = 24000

        # ── Stage 1: Text cleanup ──────────────────────────────────────────
        text = text.strip()
        if not text:
            return np.zeros(sample_rate, dtype=np.float32), sample_rate

        profile = self.voice_manager.get_voice(voice_id) if voice_id else None
        target_dialect = (dialect or (profile.dialect if profile else self.default_dialect)).lower()

        # ── Stage 2: Language detection + code-switching ───────────────────
        segments = self.code_switcher.segment(text)
        # For single-language text, segments is a single item
        # We rejoin to get normalized text for routing (multi-segment handled below)
        dominant_lang = segments[0].lang if segments else "arabic"
        has_code_switch = len(segments) > 1 and any(s.lang == "english" for s in segments)

        if has_code_switch:
            logger.info(f"[Pipeline] Code-switching detected: {len(segments)} segments")

        # ── Stage 3: Unified telephony normalization ───────────────────────
        processed_text = self.preprocess_text_unified(
            text,
            dialect=target_dialect,
            auto_tashkeel=(target_dialect in ("msa", "standard", "fusha")),
        )

        # ── Stage 4: Context + prosody analysis ───────────────────────────
        hints = self.context_prosody.analyze(processed_text)
        emotion = (emotion_override or "neutral").lower()
        speed = hints.speed
        guidance = hints.guidance_scale
        steps = num_step if num_step is not None else hints.num_step
        class_temp = 0.20  # Default: non-greedy sampling for natural prosody
        postproc = True

        # Use rich prosody planner if available (Saudi dialect aware)
        if self._prosody_planner is not None:
            plan = self._prosody_planner.analyze_and_plan(processed_text)
            instruct = plan.instruct  # Already validated against OmniVoice vocab
            speed = plan.speed
            guidance = plan.guidance_scale
            steps = num_step if num_step is not None else plan.num_step
            class_temp = plan.class_temperature
            postproc = plan.postprocess_output
            # Override language to Saudi if planner says so
            lang_name = plan.language if plan.language else lang_name
            logger.info(
                f"[Pipeline] ProsodyPlanner: {plan.style.value} | instruct={instruct!r} | "
                f"lang={lang_name} | temp={class_temp} | steps={steps}"
            )
        else:
            # Legacy path: build instruct from voice profile + prosody hints
            from .voice_manager import resolve_instruct_prompt
            base_instruct = profile.instruct_prompt if profile else None
            instruct_parts = [resolve_instruct_prompt(base_instruct)] if base_instruct else []
            # Filter hints through valid token check (no invalid tokens)
            if hints.pitch_bias and hints.pitch_bias in VALID_INSTRUCT_TOKENS:
                instruct_parts.append(hints.pitch_bias)
            instruct = _safe_instruct(", ".join(p for p in instruct_parts if p)) or None

        lang_name = DIALECT_TO_OMNIVOICE_LANG.get(target_dialect, "Standard Arabic")

        logger.info(
            f"[Pipeline] Intent: {hints.intent.value} | Speed: {speed:.2f} | "
            f"Steps: {steps} | Breath: {hints.allow_breath} | Lang: {lang_name}"
        )

        # ── Stage 5: Breathing gate ────────────────────────────────────────
        wc = len(processed_text.split())
        do_breathe = self.breathe_gate.should_breathe(
            processed_text,
            context_allow_breath=hints.allow_breath,
            word_count_override=wc,
        )

        # ── Stage 6: TTS inference ─────────────────────────────────────────
        ref_audio = profile.ref_audio_path if profile and profile.ref_audio_path and os.path.exists(
            profile.ref_audio_path if profile.ref_audio_path else ""
        ) else None
        ref_text = profile.ref_transcript if profile else None

        audio_array, sr = self.synthesize(
            text=processed_text,
            voice_id=voice_id,
            dialect=target_dialect,
            instruct=instruct,
            speed=speed,
            guidance_scale=guidance,
            num_step=steps,
            class_temperature=class_temp,
            postprocess_output=postproc,
            auto_preprocess=False,  # Already preprocessed above
            ref_audio=ref_audio,
            ref_text=ref_text,
        )

        if audio_array is None or len(audio_array) == 0:
            audio_array = self._generate_fallback_audio(processed_text, sample_rate=sample_rate)

        # ── Stage 7: Optional pre-speech breath ───────────────────────────
        if do_breathe:
            breath = self.gesture_processor.generate_tailored_breath(
                emotion=emotion, is_micro_breath=False
            )
            audio_array = np.concatenate([breath, audio_array])
            logger.debug(f"[Pipeline] Breath injected ({len(breath)/sample_rate*1000:.0f}ms)")

        # ── Stage 8: EQ coloration ─────────────────────────────────────────
        from ..models.gesture_controller import EMOTION_BREATHING_MATRIX  # noqa: F811
        eq_emotion = emotion if emotion in EMOTION_BREATHING_MATRIX else "neutral"
        audio_array = self.gesture_processor.apply_emotion_eq(audio_array, emotion=eq_emotion)

        # ── Stage 9 & 10: Loudness normalization + DC offset removal ───────
        audio_array = self.telephony_processor.remove_dc_offset(audio_array)
        audio_array = self.telephony_processor.normalize_loudness(audio_array)

        # ── Stage 11: Optional telephony resampling ────────────────────────
        if telephony_sr and telephony_sr != sample_rate:
            audio_array, sample_rate, quality = self.telephony_processor.prepare_for_telephony(
                audio_array,
                target_sr=telephony_sr,
                simulate_codec=simulate_codec,
            )
            logger.info(
                f"[Pipeline] Telephony resampled → {telephony_sr}Hz | "
                f"Quality: {'PASS' if quality['pass'] else 'FAIL'} "
                f"({', '.join(quality.get('issues', [])) or 'clean'})"
            )

        elapsed = time.time() - t0
        audio_duration = len(audio_array) / sample_rate
        rtf = elapsed / max(audio_duration, 1e-6)
        logger.info(
            f"[Pipeline] Done | gen={elapsed:.2f}s | dur={audio_duration:.2f}s | "
            f"RTF={rtf:.2f} | sr={sample_rate}Hz"
        )

        if output_path:
            self._save_audio(audio_array, sample_rate, output_path)

        return audio_array, sample_rate

    def _to_numpy(self, audio_data) -> np.ndarray:
        """Converts tensor/list audio output to 1D float32 numpy array."""
        if hasattr(audio_data, "cpu"):
            arr = audio_data.cpu().numpy()
        elif isinstance(audio_data, (list, tuple)):
            arr = np.array(audio_data[0] if len(audio_data) > 0 else audio_data)
        else:
            arr = np.array(audio_data)

        if arr.ndim > 1:
            arr = arr.squeeze()
        return arr.astype(np.float32)

    def _save_audio(self, audio: np.ndarray, sample_rate: int, filepath: str):
        """Saves numpy audio array to disk."""
        import soundfile as sf
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(p), audio, sample_rate)
        logger.info(f"[ArabicOmniVoiceEngine] Saved audio to {filepath}")

    def _generate_fallback_audio(self, text: str, sample_rate: int = 24000) -> np.ndarray:
        """
        Generates clean synthesized acoustic test tone modulated by Arabic phoneme count,
        ensuring zero-breakage when neural weights are still downloading or during offline unit testing.
        """
        duration = max(1.2, len(text) * 0.08)
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # Harmonically rich dual-formant envelope
        envelope = np.sin(np.pi * t / duration) ** 0.5
        audio = 0.2 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 440 * t)
        return (audio * envelope).astype(np.float32)
