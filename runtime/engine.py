#!/usr/bin/env python3
"""
runtime/engine.py

Core inference engine for the Mahrukh Pakistani Urdu Voice Model.
Encapsulates model weights, caches Mahrukh's warm voice prompt, and enforces
Urdu language routing and telephony audio post-processing.
"""

import os
import json
import torch
import numpy as np
import soundfile as sf
from typing import Optional, Union, Tuple
from omnivoice import OmniVoice, OmniVoiceGenerationConfig

from .urdu_normalizer import normalize_urdu_text
from .audio_processor import process_telephony_audio, to_wav_bytes

MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(MODULE_DIR, "config", "config.json")
PROMPT_PATH = os.path.join(MODULE_DIR, "prompt", "mahrukh_urdu_prompt.pt")

class UrduVoiceModel:
    """
    Dedicated Standalone Pakistani Urdu Voice Model (Mahrukh).
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(UrduVoiceModel, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, device: Optional[str] = None):
        if self._initialized:
            return

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"[UrduVoiceModel] Loading model on device: {self.device}...")
        self.model = OmniVoice.from_pretrained(
            self.config.get("base_model", "k2-fsa/OmniVoice"),
            device_map=self.device
        )
        
        print(f"[UrduVoiceModel] Loading Mahrukh warm voice prompt from {PROMPT_PATH}...")
        self.prompt = torch.load(PROMPT_PATH, map_location=self.device, weights_only=False)
        
        self.default_speed = float(self.config.get("default_speed", 0.88))
        self.default_guidance = float(self.config.get("default_guidance_scale", 1.85))
        self.default_steps = int(self.config.get("default_num_step", 36))
        self.female_mode = bool(self.config.get("female_mode", True))
        self._initialized = True
        print("[UrduVoiceModel] Ready for inference.")

    def synthesize(
        self,
        text: str,
        speed: Optional[float] = None,
        sample_rate: int = 24000,
        telephony_filter: Optional[bool] = None,
        num_step: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        output_file: Optional[str] = None,
        return_bytes: bool = False
    ) -> Union[Tuple[np.ndarray, int], bytes]:
        """
        Synthesizes Urdu text using Mahrukh's calibrated voice.

        Args:
            text: Raw input text (Urdu, English loanwords, or phone numbers)
            speed: Speaking rate multiplier (default: 0.92)
            sample_rate: Output sample rate: 24000 (HD studio) or 8000 (telephony PSTN)
            telephony_filter: Whether to apply 300Hz-3400Hz telephone filter (defaults to True if sample_rate==8000)
            num_step: Diffusion steps (default: 32 for human acoustic realism)
            guidance_scale: Classifier-free guidance scale (default: 2.0)
            output_file: Optional filepath to save the WAV output
            return_bytes: If True, returns encoded WAV byte buffer

        Returns:
            (audio_numpy, sample_rate) or WAV bytes
        """
        # 1. Phonetic and grammatical normalization
        normalized_text = normalize_urdu_text(text, female_mode=self.female_mode)
        
        # 2. Generation config
        steps = num_step if num_step is not None else self.default_steps
        guidance = guidance_scale if guidance_scale is not None else self.default_guidance
        cfg = OmniVoiceGenerationConfig(
            num_step=steps,
            guidance_scale=guidance
        )
        target_speed = speed if speed is not None else self.default_speed

        # 3. Model inference locked to Urdu ('ur')
        raw_audio = self.model.generate(
            text=normalized_text,
            language="ur",
            voice_clone_prompt=self.prompt,
            generation_config=cfg,
            speed=target_speed
        )[0]

        # 4. Audio post-processing & resampling
        apply_filter = telephony_filter if telephony_filter is not None else (sample_rate == 8000)
        final_audio, final_sr = process_telephony_audio(
            raw_audio,
            orig_sr=24000,
            target_sr=sample_rate,
            enable_filter=apply_filter
        )

        # 5. Output handling
        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            sf.write(output_file, final_audio, final_sr)

        if return_bytes:
            return to_wav_bytes(final_audio, final_sr)

        return final_audio, final_sr
