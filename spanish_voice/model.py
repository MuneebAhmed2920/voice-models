"""Spanish Voice Model Module.

High-fidelity Spanish TTS synthesizing 24 kHz audio using OmniVoice base model,
Latin American Spanish text normalization, and native Colombian voice conditioning.

Supported voices:
- female / spanish_female (Prompt_01: Colombian Female, 258 Hz mean F0)
- male / spanish_male (Prompt_07: Colombian Male, 125 Hz mean F0)
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Union

import numpy as np
import soundfile as sf
import torch
from omnivoice import OmniVoice, OmniVoiceGenerationConfig, VoiceClonePrompt

from .normalizer import normalize_spanish


class SpanishVoice:
    """High-quality Spanish TTS voice synthesizer wrapping OmniVoice."""

    VOICE_MAP: Dict[str, str] = {
        "female": "female",
        "spanish_female": "female",
        "prompt_01": "female",
        "f": "female",
        "male": "male",
        "spanish_male": "male",
        "prompt_07": "male",
        "m": "male",
    }

    def __init__(
        self,
        voice: str = "female",
        device: str = "cpu",
        diffusion_steps: int = 24,
        guidance_scale: float = 2.2,
        model_name: str = "k2-fsa/OmniVoice",
    ):
        """Initialize SpanishVoice TTS engine.

        Args:
            voice: Default voice to use ('female' / 'spanish_female' or 'male' / 'spanish_male').
            device: Compute device ('cpu', 'cuda', etc.).
            diffusion_steps: Number of diffusion steps (default: 24 for highest quality).
            guidance_scale: Classifier-free guidance scale (default: 2.2).
            model_name: HuggingFace model repository or local path for OmniVoice.
        """
        self.device = device
        self.default_voice = self._resolve_voice_key(voice)
        self.diffusion_steps = diffusion_steps
        self.guidance_scale = guidance_scale
        self.model_name = model_name

        self._package_dir = os.path.dirname(os.path.abspath(__file__))
        self._voices_dir = os.path.join(self._package_dir, "voices")

        # Load OmniVoice model
        self.model = OmniVoice.from_pretrained(model_name, device_map=self.device)
        self.model.eval()
        self.sample_rate = getattr(self.model, "sampling_rate", 24000)

        # In-memory prompt cache
        self._prompt_cache: Dict[str, VoiceClonePrompt] = {}
        self._load_prompts()

    def _resolve_voice_key(self, voice_name: str) -> str:
        key = voice_name.strip().lower()
        if key not in self.VOICE_MAP:
            valid = ["female", "male", "spanish_female", "spanish_male"]
            raise ValueError(
                f"Unknown voice '{voice_name}'. Valid choices are: {valid}"
            )
        return self.VOICE_MAP[key]

    def _load_prompts(self) -> None:
        """Preload both female and male voice conditioning prompts into memory."""
        for v in ["female", "male"]:
            prompt_path = os.path.join(self._voices_dir, v, "prompt.pt")
            if not os.path.exists(prompt_path):
                raise FileNotFoundError(
                    f"Required voice prompt file not found at: {prompt_path}"
                )
            self._prompt_cache[v] = VoiceClonePrompt.load(prompt_path)

    @property
    def voice(self) -> str:
        return self.default_voice

    @property
    def available_voices(self) -> List[str]:
        return ["female", "male"]

    def _validate_audio(self, audio: np.ndarray, target_text: str) -> None:
        """Validate synthesized audio against numerical instability and gross defects."""
        if audio is None or len(audio) == 0:
            raise RuntimeError("TTS synthesis produced empty audio array.")

        if np.isnan(audio).any() or np.isinf(audio).any():
            raise RuntimeError("TTS synthesis produced numerical NaN/Inf values.")

        peak = float(np.max(np.abs(audio)))
        if peak < 1e-4:
            raise RuntimeError("TTS synthesis produced near-zero energy silence.")

        duration_s = len(audio) / float(self.sample_rate)
        # Basic sanity check on duration vs text length
        num_chars = len(target_text.strip())
        est_syllables = max(1, num_chars // 3)
        expected_min = max(0.40, est_syllables * 0.07)
        if duration_s < expected_min:
            raise RuntimeError(
                f"TTS audio unexpectedly short ({duration_s:.2f}s < {expected_min:.2f}s) for '{target_text}'."
            )

    def generate(
        self,
        text: str,
        output: Optional[str] = None,
        voice: Optional[str] = None,
        seed: int = 42,
        diffusion_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        **kwargs,
    ) -> np.ndarray:
        """Synthesize high-quality Spanish speech from text.

        Pipeline:
        Raw Spanish Text
          -> Normalization (numbers, dates, currency, accents)
          -> Voice Prompt Conditioning (female / male)
          -> OmniVoice 24 kHz Discrete Diffusion Synthesis
          -> Acoustic Safety Verification
          -> 24 kHz WAV Audio Output

        Args:
            text: Input Spanish text (e.g., 'Hola, ¿cómo estás?').
            output: Optional path to save the resulting audio as a 24 kHz mono PCM WAV file.
            voice: Voice override ('female' or 'male'). If None, uses default voice.
            seed: Random seed for deterministic generation (default: 42).
            diffusion_steps: Number of diffusion steps override (default: 24).
            guidance_scale: Guidance scale override (default: 2.2).

        Returns:
            1D float32 numpy array containing the synthesized audio at 24,000 Hz.
        """
        # Handle flexible argument orders (e.g., tts.generate(text, voice="female", output="output.wav"))
        if "voice" in kwargs and voice is None:
            voice = kwargs["voice"]
        if "output" in kwargs and output is None:
            output = kwargs["output"]

        # If positional second arg was a voice name rather than output file path
        if output and output.strip().lower() in self.VOICE_MAP:
            if voice is None or voice.strip().lower() not in self.VOICE_MAP:
                voice, output = output, voice

        # 1. Normalize Spanish text
        normalized_text = normalize_spanish(text)

        # 2. Select voice conditioning prompt
        active_voice = self._resolve_voice_key(voice) if voice else self.default_voice
        prompt_obj = self._prompt_cache[active_voice]

        # 3. Configure synthesis parameters
        steps = diffusion_steps if diffusion_steps is not None else self.diffusion_steps
        guidance = guidance_scale if guidance_scale is not None else self.guidance_scale

        gen_config = OmniVoiceGenerationConfig(
            num_step=steps,
            guidance_scale=guidance,
            denoise=True,
            postprocess_output=True,
        )

        # 4. Generate audio with safety retry if needed
        audio: Optional[np.ndarray] = None
        active_seed = seed
        last_error = None

        for attempt in range(2):
            torch.manual_seed(active_seed)
            np.random.seed(active_seed)

            try:
                gen_out = self.model.generate(
                    text=normalized_text,
                    language="es",
                    voice_clone_prompt=prompt_obj,
                    generation_config=gen_config,
                )[0]
                self._validate_audio(gen_out, normalized_text)
                audio = gen_out
                break
            except Exception as e:
                last_error = e
                active_seed = (active_seed + 101) % 10000

        if audio is None:
            raise RuntimeError(f"Spanish TTS generation failed: {last_error}")

        # 5. Save to WAV if requested
        if output:
            out_dir = os.path.dirname(os.path.abspath(output))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            sf.write(output, audio, self.sample_rate, subtype="PCM_16")

        return audio
