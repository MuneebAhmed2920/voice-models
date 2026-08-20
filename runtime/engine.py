"""
OmniVoice Production Engine (Windows NVIDIA CUDA FP16 & CPU Fallback).
Ultra-low-latency, thread-safe Speech Synthesis Engine for Telephony and AI Voice Bots.
"""

import os
import time
import logging
import threading
from typing import Tuple, Dict, Any, Optional, Union, List

import numpy as np
import torch

try:
    from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig, VoiceClonePrompt
except ImportError:
    OmniVoice = None
    OmniVoiceGenerationConfig = None
    VoiceClonePrompt = None

from runtime.normalizer import TelephonyTextNormalizer
from runtime.audio_processor import AudioProcessor
from runtime.voice_manager import VoiceManager

logger = logging.getLogger("omnivoice.engine")


class OmniVoiceEngine:
    """
    Production real-time OmniVoice TTS inference engine.
    Optimized for NVIDIA CUDA FP16 Tensor Cores and Vicidial Telephony Pipelines.
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
        self.default_ode_steps = default_ode_steps
        self.default_sample_rate = default_sample_rate
        self.inference_lock = threading.Lock()

        # 1. Device and Precision Configuration
        self.device, self.torch_dtype = self._configure_device_and_precision(device, precision)
        
        # 2. Subsystem Initializations
        self.normalizer = TelephonyTextNormalizer()
        self.audio_processor = AudioProcessor(default_sample_rate=default_sample_rate)
        self.voice_manager = VoiceManager()

        # 3. Model Loading
        self.model = self._load_model()

        # 4. Warm-up
        if warmup and self.model is not None:
            self._warmup()

    @staticmethod
    def _configure_device_and_precision(
        requested_device: Optional[str],
        requested_precision: Optional[str]
    ) -> Tuple[torch.device, torch.dtype]:
        """
        Configures CUDA FP16 or CPU fallback with Tensor Core acceleration.
        """
        if requested_device:
            dev = torch.device(requested_device)
        elif torch.cuda.is_available():
            dev = torch.device("cuda:0")
        else:
            dev = torch.device("cpu")

        if dev.type == "cuda":
            # Enable Tensor Core acceleration for NVIDIA Ampere, Ada, Hopper, Turing GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            
            if requested_precision == "fp32":
                dtype = torch.float32
            elif requested_precision == "bf16" and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16  # Default FP16 for ultra-fast generation
            logger.info(f"🚀 CUDA Initialized on {torch.cuda.get_device_name(dev)} with {dtype} precision.")
        else:
            # CPU fallback
            cpu_threads = max(1, os.cpu_count() or 4)
            torch.set_num_threads(cpu_threads)
            dtype = torch.float32
            logger.info(f"ℹ️ CPU fallback mode with {cpu_threads} threads and {dtype} precision.")

        return dev, dtype

    def _load_model(self):
        """Loads OmniVoice model with specified precision onto target device."""
        if OmniVoice is None:
            logger.error("OmniVoice package not installed.")
            return None

        logger.info(f"Loading OmniVoice from '{self.model_name_or_path}' onto {self.device} ({self.torch_dtype})...")
        t0 = time.time()
        
        try:
            model = OmniVoice.from_pretrained(
                self.model_name_or_path,
                torch_dtype=self.torch_dtype
            )
            model = model.to(self.device)
            model.eval()
            logger.info(f"✅ OmniVoice model loaded in {time.time() - t0:.2f}s.")
            return model
        except Exception as e:
            logger.error(f"Failed to load model on {self.device} in {self.torch_dtype}: {e}")
            if self.device.type == "cuda":
                logger.warning("Attempting fallback to CPU float32...")
                self.device = torch.device("cpu")
                self.torch_dtype = torch.float32
                model = OmniVoice.from_pretrained(self.model_name_or_path, torch_dtype=torch.float32)
                model = model.to(self.device)
                model.eval()
                return model
            raise

    def _warmup(self):
        """Runs a fast dummy inference to pre-warm CUDA kernels, prompt caches, and pipelines."""
        logger.info("Pre-warming OmniVoice inference pipeline...")
        try:
            prompt = self.voice_manager.get_prompt("voice_clara", "neutral")
            if prompt is not None:
                gen_cfg = OmniVoiceGenerationConfig(
                    num_step=16,
                    guidance_scale=2.2,
                    denoise=True
                )
                t0 = time.time()
                with self.inference_lock:
                    if self.device.type == "cuda" and self.torch_dtype == torch.float16:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            _ = self.model.generate(
                                text="Ready.",
                                voice_clone_prompt=prompt,
                                speed=0.92,
                                generation_config=gen_cfg
                            )
                    else:
                        _ = self.model.generate(
                            text="Ready.",
                            voice_clone_prompt=prompt,
                            speed=0.92,
                            generation_config=gen_cfg
                        )
                logger.info(f"✅ Pipeline warm-up completed in {time.time() - t0:.2f}s.")
        except Exception as e:
            logger.warning(f"Warm-up notice (non-fatal): {e}")

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = "voice_clara",
        emotion: Optional[str] = None,
        speed: Optional[float] = None,
        sample_rate: Optional[int] = None,
        format_type: str = "pcm16",
        ode_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        normalize_text: bool = True,
        add_breath: Optional[bool] = None
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Synthesizes speech from text.
        
        Args:
            text: Input text string (can contain emotion tags like [excited], numbers, currencies, etc.)
            voice: Voice ID or alias (default: voice_clara)
            emotion: Optional emotion override (neutral, excited, sympathetic, emotional)
            speed: Speaking speed multiplier (default: calibrated 0.92)
            sample_rate: Target audio sample rate (8000 for PSTN Vicidial, 16000 for HD, 24000 for native)
            format_type: Encoded output format: 'pcm16', 'mulaw', 'alaw', 'wav'
            ode_steps: Flow-matching ODE steps (16 for low-latency real-time, 32 for studio)
            guidance_scale: CFG guidance scale (default: 2.2)
            normalize_text: Whether to apply telephony text normalization
            add_breath: Whether to prepend organic breathing sound
            
        Returns:
            (encoded_audio_bytes, metrics_dict)
        """
        start_time = time.time()
        
        # 1. Voice & Config Resolution
        canonical_voice = self.voice_manager.resolve_voice_id(voice)
        voice_cfg = self.voice_manager.get_config(canonical_voice)

        target_speed = speed if speed is not None else voice_cfg.get("default_speed", 0.92)
        target_guidance = guidance_scale if guidance_scale is not None else voice_cfg.get("default_guidance_scale", 2.2)
        target_sr = sample_rate if sample_rate is not None else self.default_sample_rate
        target_ode_steps = ode_steps if ode_steps is not None else self.default_ode_steps

        # 2. Text Normalization and Emotion Tag Parsing
        has_breath_tag = False
        parsed_emotion = None
        if normalize_text:
            processed_text, parsed_emotion, has_breath_tag = self.normalizer.process(text)
        else:
            processed_text = text.strip()

        # Emotion priority: explicit argument > tag in text > default neutral
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

        # 3. Retrieve Precomputed Voice Prompt
        prompt = self.voice_manager.get_prompt(canonical_voice, effective_emotion, device=self.device)
        if prompt is None:
            raise ValueError(f"No prompt available for voice '{canonical_voice}' (emotion: '{effective_emotion}')")

        # 4. Configure ODE Generation
        gen_cfg = OmniVoiceGenerationConfig(
            num_step=target_ode_steps,
            guidance_scale=target_guidance,
            denoise=True
        )

        inference_start = time.time()
        
        # 5. Model Inference (Thread-safe)
        with self.inference_lock:
            if self.device.type == "cuda" and self.torch_dtype == torch.float16:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    audios = self.model.generate(
                        text=processed_text,
                        voice_clone_prompt=prompt,
                        speed=target_speed,
                        generation_config=gen_cfg
                    )
            else:
                audios = self.model.generate(
                    text=processed_text,
                    voice_clone_prompt=prompt,
                    speed=target_speed,
                    generation_config=gen_cfg
                )

        inference_end = time.time()
        raw_audio = audios[0] if isinstance(audios, (list, tuple)) else audios
        model_sr = getattr(self.model, "sampling_rate", 24000) or 24000

        # 6. Post-Processing & Telephony Resampling / Encoding
        encoded_bytes, audio_dur = self.audio_processor.process_and_encode(
            raw_audio,
            orig_sr=model_sr,
            target_sr=target_sr,
            format_type=format_type,
            has_breath=effective_breath
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
            "emotion": effective_emotion or "neutral",
            "sample_rate": target_sr,
            "format": format_type,
            "ode_steps": target_ode_steps,
            "speed": target_speed,
            "has_breath": effective_breath
        }

        return encoded_bytes, metrics

    def get_status(self) -> Dict[str, Any]:
        """Returns hardware, model, and memory status."""
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
) -> OmniVoiceEngine:
    """
    Thread-safe Singleton provider for OmniVoiceEngine.
    """
    if OmniVoiceEngine._instance is None:
        with OmniVoiceEngine._lock:
            if OmniVoiceEngine._instance is None:
                OmniVoiceEngine._instance = OmniVoiceEngine(
                    model_name_or_path=model_name_or_path,
                    device=device,
                    precision=precision,
                    default_ode_steps=default_ode_steps,
                    default_sample_rate=default_sample_rate,
                    warmup=warmup
                )
    return OmniVoiceEngine._instance
