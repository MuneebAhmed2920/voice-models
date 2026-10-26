"""
Production OmniVoice TTS Engine.
Dedicated TEXT -> OMNIVOICE -> AUDIO synthesis pipeline.
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any, Literal, Generator

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice_tts.text.normalizer import normalize_english_text
from omnivoice_tts.text.gestures import GestureParser
from omnivoice_tts.audio.codecs import resample_audio, encode_audio, AudioFormat
from omnivoice_tts.voices.registry import VoiceRegistry

QualityMode = Literal["fast", "quality"]


class OmniVoiceTTSEngine:
    """
    Dedicated Minimal Production English TTS Engine.
    Supports 4 fixed precomputed voices, dynamic flow step scaling,
    call-center text normalization, and conversational gestures.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._is_ready = False
        
        print("⚡ [OmniVoiceTTS] Booting Production TTS Engine...")
        t0 = time.perf_counter()
        self.model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=self.device)
        self.registry = VoiceRegistry()
        self.gesture_parser = GestureParser(sample_rate=24000)
        
        # Warmup at startup to compile execution graphs & preallocate buffers
        self._warmup()
        self._is_ready = True
        print(f"✅ [OmniVoiceTTS] Engine READY in {(time.perf_counter() - t0):.2f}s | Loaded 4 Voices: {self.registry.list_voices()}")

    def _warmup(self):
        """Runs a tiny 1-step warmup synthesis to enter READY state."""
        warm_cfg = OmniVoiceGenerationConfig(num_step=4, guidance_scale=1.0)
        self.model.generate(
            text="Hello.",
            language="en",
            instruct="female, american accent, young adult, moderate pitch",
            generation_config=warm_cfg,
            speed=1.0
        )

    def _get_dynamic_steps(self, word_count: int, quality: QualityMode) -> int:
        if quality == "fast":
            if word_count <= 2:
                return 6
            elif word_count <= 7:
                return 10
            else:
                return 16
        # Quality mode: 16 steps (Exact original sound)
        return 16


    @torch.inference_mode()
    def generate(
        self,
        text: str,
        voice_id: str = "female_1",
        speed: float = 1.0,
        quality: QualityMode = "fast",
        sample_rate: int = 24000,
        output_format: AudioFormat = "wav",
        include_breath: bool = False,
        include_cough: bool = False,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Synthesizes text into audio bytes.
        
        Args:
            text: Input English text (supports <laugh>, <sigh>, <breath>, <cough>, numbers, currency)
            voice_id: 'female_1' | 'female_2' | 'male_1' | 'male_2'
            speed: Speaking speed factor (1.0 = normal)
            quality: 'fast' (dynamic 4-12 steps) | 'quality' (16 steps)
            sample_rate: 24000 | 16000 | 8000
            output_format: 'wav' | 'pcm16' | 'mulaw'
            include_breath: Whether to prepend a subtle airway inhale
            include_cough: Whether to prepend a subtle throat clearing (*ahem*)
            
        Returns:
            - audio_bytes: Encoded bytes in target format
            - metrics: Generation diagnostics (TTFA, RTF, word count, steps)
        """
        t_start = time.perf_counter()

        # 1. English Normalization
        t_norm_0 = time.perf_counter()
        normalized_text = normalize_english_text(text)
        t_norm_ms = (time.perf_counter() - t_norm_0) * 1000.0

        # 2. Gesture & Emotion Parsing
        cleaned_text, emotion_mod = self.gesture_parser.parse(normalized_text)
        words = cleaned_text.split()
        word_count = len(words)

        # 3. Voice Profile Retrieval (0 ms overhead)
        voice_profile = self.registry.get(voice_id)
        base_speed = voice_profile.get("speed", 1.0) * speed

        instruct = voice_profile.get("instruct", "female, american accent, young adult, moderate pitch")
        if emotion_mod:
            instruct = f"{instruct}, {emotion_mod}"

        steps = self._get_dynamic_steps(word_count, quality)

        gen_cfg = OmniVoiceGenerationConfig(
            num_step=steps,
            guidance_scale=2.2,
            denoise=True,
            postprocess_output=True,
            pad_duration=0.08,
            fade_duration=0.08
        )


        # 4. Flow-Matching Generation
        t_infer_0 = time.perf_counter()
        if "prompt_obj" in voice_profile and voice_profile["prompt_obj"] is not None:
            audio_list = self.model.generate(
                text=cleaned_text,
                language="en",
                voice_clone_prompt=voice_profile["prompt_obj"],
                generation_config=gen_cfg,
                speed=base_speed
            )
        else:
            audio_list = self.model.generate(
                text=cleaned_text,
                language="en",
                instruct=instruct,
                generation_config=gen_cfg,
                speed=base_speed
            )
        t_infer_sec = time.perf_counter() - t_infer_0

        raw_speech = np.asarray(audio_list[0], dtype=np.float32).reshape(-1)

        # 5. Splice Non-Verbal Acoustic Signals
        pieces = []
        is_male = "male" in voice_id
        if include_cough:
            pieces.append(self.gesture_parser.generate_breath(duration_sec=0.15, intensity=0.10, is_male=is_male))
        if include_breath:
            pieces.append(self.gesture_parser.generate_breath(duration_sec=0.25, intensity=0.12, is_male=is_male))

        pieces.append(raw_speech)
        combined_24k = np.concatenate(pieces) if len(pieces) > 1 else raw_speech

        # 6. Resample & Format Encode
        resampled = resample_audio(combined_24k, orig_sr=24000, target_sr=sample_rate)
        audio_bytes = encode_audio(resampled, sample_rate=sample_rate, output_format=output_format)

        total_latency_sec = time.perf_counter() - t_start
        audio_duration_sec = len(resampled) / float(sample_rate)
        rtf = total_latency_sec / audio_duration_sec if audio_duration_sec > 0 else 0

        metrics = {
            "voice_id": voice_id,
            "persona": voice_profile.get("persona", voice_id),
            "text_original": text,
            "text_normalized": normalized_text,
            "word_count": word_count,
            "steps_used": steps,
            "quality_mode": quality,
            "norm_latency_ms": t_norm_ms,
            "infer_sec": t_infer_sec,
            "ttfa_sec": total_latency_sec,
            "audio_duration_sec": audio_duration_sec,
            "rtf": rtf,
            "speed_x": (1.0 / rtf) if rtf > 0 else 0
        }

        return audio_bytes, metrics


# Singleton Engine Helper
_GLOBAL_ENGINE: Optional[OmniVoiceTTSEngine] = None

def get_engine(device: str = "cpu") -> OmniVoiceTTSEngine:
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = OmniVoiceTTSEngine(device=device)
    return _GLOBAL_ENGINE
