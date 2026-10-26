"""
Arabic Voice Engine: Core Synthesis Orchestrator (The 7 Pillars)
Combines OmniVoice flow-matching with calibrated parameters (speed 0.92, guidance 2.2, class_temp 0.20),
physiological airway breathing, studio mastering EQ, and telephony codecs.
"""

import os
import time
import logging
import threading
from typing import Dict, Optional, Tuple, Any, Generator, List
import numpy as np
import torch

try:
    from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig, VoiceClonePrompt
except ImportError:
    OmniVoice = None
    OmniVoiceGenerationConfig = None
    VoiceClonePrompt = None

from .voice_manager import ArabicVoiceManager
from .normalizer import ArabicTelephonyNormalizer
from .arabic_breath_processor import ArabicBreathProcessor
from .audio_processor import ArabicAudioProcessor
from .streamer import ArabicClauseStreamer

logger = logging.getLogger("ArabicVoiceEngine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class ArabicVoiceEngine:
    """
    Production-grade Arabic Voice Engine implementing the Clara 7-Pillar Humanization Blueprint.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(
        self,
        model_name_or_path: str = "k2-fsa/OmniVoice",
        device: Optional[str] = None,
        precision: Optional[str] = "fp16",
        default_ode_steps: int = 16,
        default_sample_rate: int = 8000,
        warmup: bool = True
    ):
        self.model_name_or_path = model_name_or_path
        self.requested_device = device
        self.requested_precision = precision
        self.default_ode_steps = default_ode_steps
        self.default_sample_rate = default_sample_rate

        self.device, self.torch_dtype = self._configure_device_and_precision(device, precision)

        # Initialize core components
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.voice_manager = ArabicVoiceManager(base_dir=base_dir)
        self.normalizer = ArabicTelephonyNormalizer()
        self.breath_processor = ArabicBreathProcessor(
            prompt_dir=os.path.join(base_dir, "prompt"),
            sample_rate=24000
        )
        self.audio_processor = ArabicAudioProcessor(
            default_sample_rate=default_sample_rate,
            apply_mastering_eq=True
        )

        self.model = None
        self.inference_lock = threading.Lock()

        # Load OmniVoice model
        self._load_model()

        if warmup and self.model is not None:
            self._warmup()

    def _configure_device_and_precision(self, device: Optional[str], precision: Optional[str]) -> Tuple[torch.device, torch.dtype]:
        if device is None or device == "auto":
            if torch.cuda.is_available():
                target_device = torch.device("cuda:0")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                target_device = torch.device("mps")
            else:
                target_device = torch.device("cpu")
        else:
            target_device = torch.device(device)

        prec = (precision or "fp16").lower()
        if target_device.type == "cuda":
            if prec in ["fp16", "float16"]:
                target_dtype = torch.float16
            elif prec in ["bf16", "bfloat16"]:
                target_dtype = torch.bfloat16
            else:
                target_dtype = torch.float32
        else:
            target_dtype = torch.float32

        logger.info(f"Configured hardware target: device={target_device}, dtype={target_dtype}")
        return target_device, target_dtype

    def _load_model(self):
        if OmniVoice is None:
            logger.error("OmniVoice package is not installed.")
            return

        logger.info(f"Loading OmniVoice checkpoint from {self.model_name_or_path}...")
        t0 = time.time()
        try:
            # First check local HuggingFace cache
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--k2-fsa--OmniVoice")
            if os.path.exists(cache_dir):
                self.model = OmniVoice.from_pretrained(
                    self.model_name_or_path,
                    torch_dtype=self.torch_dtype,
                    local_files_only=True
                )
            else:
                self.model = OmniVoice.from_pretrained(
                    self.model_name_or_path,
                    torch_dtype=self.torch_dtype
                )

            self.model.to(self.device)
            self.model.eval()
            logger.info(f"✅ OmniVoice model loaded successfully in {time.time() - t0:.2f}s.")
        except Exception as e:
            logger.error(f"Failed to load OmniVoice model: {e}")
            raise

    def _warmup(self):
        """Pre-warms the ODE solver to eliminate cold-start latency."""
        try:
            prompt = self.voice_manager.get_prompt("voice_arabic_clara", "neutral", device=self.device)
            if prompt is not None and self.model is not None:
                logger.info("Executing engine warm-up cycle...")
                t0 = time.time()
                cfg = OmniVoiceGenerationConfig(num_step=8, guidance_scale=2.0, denoise=True)
                with self.inference_lock:
                    with torch.no_grad():
                        self.model.generate(
                            text="مرحباً",
                            voice_clone_prompt=prompt,
                            language="standard arabic",
                            speed=0.92,
                            generation_config=cfg
                        )
                logger.info(f"✅ Pipeline warm-up finished in {time.time() - t0:.2f}s.")
        except Exception as e:
            logger.warning(f"Warm-up notice (non-fatal): {e}")

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = "voice_arabic_clara",
        emotion: Optional[str] = None,
        language: str = "najdi arabic",
        speed: Optional[float] = None,
        sample_rate: Optional[int] = None,
        format_type: str = "pcm16",
        ode_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        class_temperature: Optional[float] = None,
        normalize_text: bool = True,
        add_breath: Optional[bool] = None,
        apply_eq: bool = True
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Synthesizes high-naturalness Arabic speech.
        
        Args:
            text: Input text string (supports [متحمس], [متعاطف], [مؤثر], [طبيعي], numbers, dates, currencies)
            voice: Voice model ID (default: voice_arabic_clara)
            emotion: Emotion override (neutral, excited, sympathetic, emotional)
            language: Arabic dialect ('najdi arabic', 'standard arabic', 'hijazi arabic', etc.)
            speed: Speaking speed multiplier (calibrated default: 0.92)
            sample_rate: Output sample rate (8000 for PSTN Vicidial, 16000 for HD, 24000 for native)
            format_type: Encoded format: 'pcm16', 'mulaw', 'alaw', 'wav'
            ode_steps: Flow-matching steps (16 for real-time telephony, 24-32 for studio)
            guidance_scale: CFG guidance scale (calibrated default: 2.2)
            class_temperature: Class sampling temperature (default: 0.20 for organic pitch micro-inflections)
            normalize_text: Whether to apply deterministic Arabic telephony normalization
            add_breath: Whether to prepend emotion-tailored airway breath
            apply_eq: Whether to apply Pillar 5 studio mastering & proximity EQ chain
            
        Returns:
            (encoded_audio_bytes, metrics_dict)
        """
        start_time = time.time()

        # 1. Voice and Parameter Resolution
        canonical_voice = self.voice_manager.resolve_voice_id(voice)
        voice_cfg = self.voice_manager.get_config(canonical_voice)

        target_speed = speed if speed is not None else voice_cfg.get("default_speed", 0.92)
        target_guidance = guidance_scale if guidance_scale is not None else voice_cfg.get("default_guidance_scale", 2.2)
        target_temp = class_temperature if class_temperature is not None else voice_cfg.get("class_temperature", 0.20)
        target_sr = sample_rate if sample_rate is not None else self.default_sample_rate
        target_ode_steps = ode_steps if ode_steps is not None else self.default_ode_steps

        # 2. Text Normalization and Emotion Tag Parsing (Pillars 1 & 7)
        has_breath_tag = True
        parsed_emotion = "neutral"
        if normalize_text:
            processed_text, parsed_emotion, has_breath_tag = self.normalizer.process(text)
        else:
            processed_text, parsed_emotion, has_breath_tag = self.breath_processor.parse_arabic_tags(text)

        effective_emotion = emotion if emotion else parsed_emotion
        effective_breath = add_breath if add_breath is not None else has_breath_tag

        if not processed_text:
            empty_bytes, _ = self.audio_processor.process_and_encode(
                np.array([], dtype=np.float32),
                orig_sr=24000,
                target_sr=target_sr,
                format_type=format_type
            )
            return empty_bytes, {
                "ttfa_ms": 0.0,
                "inference_time_ms": 0.0,
                "audio_duration_sec": 0.0,
                "rtf": 0.0,
                "text": text,
                "normalized_text": "",
                "sample_rate": target_sr,
                "format": format_type
            }

        # 3. Retrieve Multi-Emotion Voice Prompt Tensor (Pillar 4)
        prompt = self.voice_manager.get_prompt(canonical_voice, effective_emotion, device=self.device)
        if prompt is None:
            raise ValueError(f"No prompt tensor available for '{canonical_voice}' (emotion: '{effective_emotion}')")

        # 4. Configure ODE Flow-Matching (Pillar 3)
        gen_cfg = OmniVoiceGenerationConfig(
            num_step=target_ode_steps,
            guidance_scale=target_guidance,
            denoise=True
        )

        inference_start = time.time()

        # 5. Model Inference (Pillar 2: Cadence 0.92x)
        with self.inference_lock:
            with torch.no_grad():
                audios = self.model.generate(
                    text=processed_text,
                    voice_clone_prompt=prompt,
                    language=language,
                    speed=target_speed,
                    generation_config=gen_cfg,
                    class_temperature=target_temp
                )

        inference_end = time.time()
        raw_audio = audios[0] if isinstance(audios, (list, tuple)) else audios
        if hasattr(raw_audio, "cpu"):
            raw_audio = raw_audio.cpu().numpy()
        raw_audio = np.asarray(raw_audio, dtype=np.float32)

        model_sr = getattr(self.model, "sampling_rate", 24000) or 24000

        # 6. Physiological Airway Inhalation Injection (Pillar 1)
        if effective_breath:
            raw_audio = self.breath_processor.inject_pre_speech_breath(
                raw_audio,
                emotion=effective_emotion,
                include_breath=True
            )

        # 7. Studio Mastering & Proximity EQ Chain + Codec Encoding (Pillar 5)
        encoded_bytes, audio_dur = self.audio_processor.process_and_encode(
            raw_audio,
            orig_sr=model_sr,
            target_sr=target_sr,
            format_type=format_type,
            apply_eq=apply_eq,
            normalize_volume=True
        )

        total_time = time.time() - start_time
        inference_time_ms = (inference_end - inference_start) * 1000.0
        ttfa_ms = total_time * 1000.0
        rtf = (inference_end - inference_start) / max(0.001, audio_dur)

        metrics = {
            "ttfa_ms": round(ttfa_ms, 2),
            "inference_time_ms": round(inference_time_ms, 2),
            "audio_duration_sec": round(audio_dur, 3),
            "rtf": round(rtf, 4),
            "text": text,
            "normalized_text": processed_text,
            "voice": canonical_voice,
            "emotion": effective_emotion,
            "language": language,
            "sample_rate": target_sr,
            "format": format_type,
            "ode_steps": target_ode_steps,
            "speed": target_speed,
            "guidance_scale": target_guidance,
            "class_temperature": target_temp,
            "has_breath": effective_breath
        }

        return encoded_bytes, metrics

    def synthesize_stream(
        self,
        text_generator: Generator[str, None, None],
        voice: Optional[str] = "voice_arabic_clara",
        language: str = "najdi arabic",
        sample_rate: int = 8000,
        format_type: str = "pcm16",
        frame_duration_ms: int = 20
    ) -> Generator[bytes, None, None]:
        """
        Streams audio frames (e.g. 20ms telephony frames) as tokens arrive from LLM.
        """
        streamer = ArabicClauseStreamer()
        first_clause = True

        for token in text_generator:
            clauses = streamer.feed_token(token)
            for clause, pause_ms in clauses:
                audio_bytes, _ = self.synthesize(
                    text=clause,
                    voice=voice,
                    language=language,
                    sample_rate=sample_rate,
                    format_type=format_type,
                    add_breath=first_clause
                )
                first_clause = False

                # Yield in 20ms frames
                bytes_per_sample = 2 if format_type == "pcm16" else 1
                for frame in self.audio_processor.frame_audio(
                    audio_bytes,
                    sample_rate=sample_rate,
                    frame_duration_ms=frame_duration_ms,
                    bytes_per_sample=bytes_per_sample
                ):
                    yield frame

                # Yield inter-clause pause silence
                pause_samples = int(sample_rate * (pause_ms / 1000.0))
                pause_bytes = b"\x00" * (pause_samples * bytes_per_sample)
                for frame in self.audio_processor.frame_audio(
                    pause_bytes,
                    sample_rate=sample_rate,
                    frame_duration_ms=frame_duration_ms,
                    bytes_per_sample=bytes_per_sample
                ):
                    yield frame

        # Flush remaining buffer
        for clause, pause_ms in streamer.flush():
            audio_bytes, _ = self.synthesize(
                text=clause,
                voice=voice,
                language=language,
                sample_rate=sample_rate,
                format_type=format_type,
                add_breath=False
            )
            bytes_per_sample = 2 if format_type == "pcm16" else 1
            for frame in self.audio_processor.frame_audio(
                audio_bytes,
                sample_rate=sample_rate,
                frame_duration_ms=frame_duration_ms,
                bytes_per_sample=bytes_per_sample
            ):
                yield frame

    def get_status(self) -> Dict[str, Any]:
        """Returns engine and hardware status."""
        vram_used_mb = 0
        vram_total_mb = 0
        gpu_name = None

        if self.device.type == "cuda" and torch.cuda.is_available():
            vram_used_mb = round(torch.cuda.memory_allocated(self.device) / (1024 * 1024), 2)
            vram_total_mb = round(torch.cuda.get_device_properties(self.device).total_memory / (1024 * 1024), 2)
            gpu_name = torch.cuda.get_device_name(self.device)

        return {
            "status": "ready" if self.model is not None else "uninitialized",
            "device": str(self.device),
            "dtype": str(self.torch_dtype),
            "gpu_name": gpu_name,
            "vram_used_mb": vram_used_mb,
            "vram_total_mb": vram_total_mb,
            "model_path": self.model_name_or_path,
            "default_ode_steps": self.default_ode_steps,
            "default_sample_rate": self.default_sample_rate,
            "voices": self.voice_manager.list_voices()
        }


def get_engine(
    model_name_or_path: str = "k2-fsa/OmniVoice",
    device: Optional[str] = None,
    precision: Optional[str] = "fp16",
    default_ode_steps: int = 16,
    default_sample_rate: int = 8000,
    warmup: bool = True
) -> ArabicVoiceEngine:
    """Thread-safe Singleton provider for ArabicVoiceEngine."""
    if ArabicVoiceEngine._instance is None:
        with ArabicVoiceEngine._lock:
            if ArabicVoiceEngine._instance is None:
                ArabicVoiceEngine._instance = ArabicVoiceEngine(
                    model_name_or_path=model_name_or_path,
                    device=device,
                    precision=precision,
                    default_ode_steps=default_ode_steps,
                    default_sample_rate=default_sample_rate,
                    warmup=warmup
                )
    return ArabicVoiceEngine._instance
