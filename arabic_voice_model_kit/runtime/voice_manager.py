"""
Arabic Voice Manager: Prompt Tensor Cache & Multi-Emotion Conditioning (Pillar 4)
"""

import os
import json
import torch
from typing import Dict, Optional, Any, List


class ArabicVoiceManager:
    """
    Manages Arabic voice models, emotional prompt tensors (neutral, excited, sympathetic, emotional),
    and runtime configurations. Maintains an in-memory cache of VoiceClonePrompt objects.
    """

    EMOTION_MAP = {
        "neutral": "neutral",
        "طبيعي": "neutral",
        "default": "neutral",
        "excited": "excited",
        "متحمس": "excited",
        "فرحان": "excited",
        "happy": "excited",
        "sympathetic": "sympathetic",
        "متعاطف": "sympathetic",
        "اعتذار": "sympathetic",
        "caring": "sympathetic",
        "apologetic": "sympathetic",
        "emotional": "emotional",
        "مؤثر": "emotional",
        "صادق": "emotional",
        "sincere": "emotional",
        "warm": "emotional"
    }

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.base_dir = base_dir

        self.prompt_dir = os.path.join(self.base_dir, "prompt")
        self.config_dir = os.path.join(self.base_dir, "config")

        self.prompt_cache: Dict[str, Any] = {}
        self.voice_configs: Dict[str, Dict[str, Any]] = {}
        self.voice_metadata: Dict[str, Dict[str, Any]] = {}

        self.aliases: Dict[str, str] = {
            "arabic_clara": "voice_arabic_clara",
            "voice_arabic": "voice_arabic_clara",
            "clara_arabic": "voice_arabic_clara",
            "arabic": "voice_arabic_clara",
            "default": "voice_arabic_clara",
            "ar-sa": "voice_arabic_clara",
            "voice_arabic_clara": "voice_arabic_clara"
        }

        self._load_arabic_voice()

    def _load_arabic_voice(self):
        """Loads and caches configurations, metadata, and multi-emotion prompt tensors."""
        config_path = os.path.join(self.config_dir, "config.json")
        metadata_path = os.path.join(self.config_dir, "metadata.json")

        cfg = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            cfg = {
                "voice_id": "voice_arabic_clara",
                "name": "Arabic Conversational Voice",
                "language": "ar-SA",
                "default_guidance_scale": 2.2,
                "default_speed": 0.92,
                "default_ode_steps": 16,
                "class_temperature": 0.20,
                "sample_rate": 24000
            }
        self.voice_configs["voice_arabic_clara"] = cfg

        meta = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        self.voice_metadata["voice_arabic_clara"] = meta

        # Load multi-emotion prompt tensors
        prompt_files = {
            "neutral": "voice_prompt.pt",
            "excited": "voice_prompt_excited.pt",
            "sympathetic": "voice_prompt_sympathetic.pt",
            "emotional": "voice_prompt_emotional.pt"
        }

        for emo_key, fname in prompt_files.items():
            fpath = os.path.join(self.prompt_dir, fname)
            if not os.path.exists(fpath):
                # Fallback to Clara's prompt tensors
                clara_path = f"/Users/haseebahmed/agent/clara/clara-voice-model/prompt/{fname}"
                if os.path.exists(clara_path):
                    fpath = clara_path

            if os.path.exists(fpath):
                try:
                    tensor_prompt = torch.load(fpath, map_location="cpu", weights_only=False)
                    cache_key = f"voice_arabic_clara:{emo_key}"
                    self.prompt_cache[cache_key] = tensor_prompt
                except Exception as e:
                    print(f"Warning: Failed loading prompt tensor {fpath}: {e}")

        # Set default fallback
        if "voice_arabic_clara:neutral" in self.prompt_cache:
            self.prompt_cache["voice_arabic_clara:default"] = self.prompt_cache["voice_arabic_clara:neutral"]

    def resolve_voice_id(self, voice: Optional[str]) -> str:
        """Resolves voice name or alias to canonical voice_id."""
        if not voice:
            return "voice_arabic_clara"
        cleaned = voice.strip().lower()
        return self.aliases.get(cleaned, "voice_arabic_clara")

    def resolve_emotion(self, emotion: Optional[str]) -> str:
        """Maps Arabic or English emotion tag to canonical emotion key."""
        if not emotion:
            return "neutral"
        cleaned = emotion.strip().lower()
        return self.EMOTION_MAP.get(cleaned, "neutral")

    def get_prompt(
        self,
        voice: Optional[str] = "voice_arabic_clara",
        emotion: Optional[str] = None,
        device: Optional[torch.device] = None
    ) -> Optional[Any]:
        """
        Retrieves the preloaded VoiceClonePrompt for the specified voice and emotion style.
        """
        canonical_voice = self.resolve_voice_id(voice)
        emo = self.resolve_emotion(emotion)

        cache_key = f"{canonical_voice}:{emo}"
        prompt = self.prompt_cache.get(cache_key)

        # Fallbacks
        if prompt is None:
            prompt = self.prompt_cache.get(f"{canonical_voice}:neutral")
        if prompt is None:
            prompt = self.prompt_cache.get("voice_arabic_clara:neutral")

        return prompt

    def get_config(self, voice: Optional[str] = "voice_arabic_clara") -> Dict[str, Any]:
        canonical_voice = self.resolve_voice_id(voice)
        return self.voice_configs.get(canonical_voice, self.voice_configs.get("voice_arabic_clara", {}))

    def get_metadata(self, voice: Optional[str] = "voice_arabic_clara") -> Dict[str, Any]:
        canonical_voice = self.resolve_voice_id(voice)
        return self.voice_metadata.get(canonical_voice, self.voice_metadata.get("voice_arabic_clara", {}))

    def list_voices(self) -> List[Dict[str, Any]]:
        voices = []
        for voice_id, cfg in self.voice_configs.items():
            meta = self.voice_metadata.get(voice_id, {})
            emotions = [
                k.split(":")[1]
                for k in self.prompt_cache.keys()
                if k.startswith(f"{voice_id}:") and k.split(":")[1] != "default"
            ]
            voices.append({
                "voice_id": voice_id,
                "name": cfg.get("name", voice_id),
                "gender": cfg.get("gender", "female"),
                "language": cfg.get("language", "ar-SA"),
                "accent": cfg.get("accent", ""),
                "default_speed": cfg.get("default_speed", 0.92),
                "default_guidance_scale": cfg.get("default_guidance_scale", 2.2),
                "available_emotions": sorted(list(set(emotions))),
                "style": meta.get("style", ""),
                "license": meta.get("license", "Apache-2.0")
            })
        return voices
