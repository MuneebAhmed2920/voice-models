"""
Voice Manager: Dynamic Voice Registry, Prompt Cache, and Emotion Prompt Routing.
"""

import os
import json
import torch
from typing import Dict, Optional, Any, List


class VoiceManager:
    """
    Manages voice models, emotional prompt tensors, and runtime configurations.
    Maintains a high-speed memory cache of precomputed VoiceClonePrompt tensors.
    """

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.base_dir = base_dir

        self.prompt_dir = os.path.join(self.base_dir, "prompt")
        self.config_dir = os.path.join(self.base_dir, "config")
        
        self.primary_voice_id = "voice_emma"
        self.prompt_cache: Dict[str, Any] = {}
        self.voice_configs: Dict[str, Dict[str, Any]] = {}
        self.voice_metadata: Dict[str, Dict[str, Any]] = {}
        self.aliases: Dict[str, str] = {
            "emma": "voice_emma",
            "default": "voice_emma",
            "primary": "voice_emma",
            "en-US-emma": "voice_emma",
        }

        self._load_primary_voice()

    def _load_primary_voice(self):
        """Loads and indexes the primary voice package."""
        config_path = os.path.join(self.config_dir, "config.json")
        metadata_path = os.path.join(self.config_dir, "metadata.json")

        cfg = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            cfg = {
                "voice_id": "voice_emma",
                "name": "Emma",
                "gender": "female",
                "language": "en-US",
                "default_guidance_scale": 2.2,
                "default_speed": 0.9,
                "sample_rate": 24000
            }
        self.voice_configs[self.primary_voice_id] = cfg

        meta = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        self.voice_metadata[self.primary_voice_id] = meta

        prompt_files = {
            "neutral": "voice_prompt.pt",
            "excited": "voice_prompt_excited.pt",
            "sympathetic": "voice_prompt_sympathetic.pt",
            "emotional": "voice_prompt_emotional.pt",
        }

        for emotion_key, filename in prompt_files.items():
            full_path = os.path.join(self.prompt_dir, filename)
            if os.path.exists(full_path):
                try:
                    tensor_prompt = torch.load(full_path, map_location="cpu", weights_only=False)
                    cache_key = f"{self.primary_voice_id}:{emotion_key}"
                    self.prompt_cache[cache_key] = tensor_prompt
                except Exception as e:
                    print(f"Warning: Failed loading {full_path}: {e}")

        if f"{self.primary_voice_id}:neutral" in self.prompt_cache:
            self.prompt_cache[f"{self.primary_voice_id}:default"] = self.prompt_cache[f"{self.primary_voice_id}:neutral"]

    def resolve_voice_id(self, voice: Optional[str]) -> str:
        """Resolves voice name or alias to canonical voice_id."""
        if not voice:
            return self.primary_voice_id
        cleaned = voice.strip().lower()
        return self.aliases.get(cleaned, self.primary_voice_id)

    def get_prompt(self, voice: Optional[str] = None, emotion: Optional[str] = None, device: Optional[torch.device] = None) -> Optional[Any]:
        canonical_voice = self.resolve_voice_id(voice)
        emo = (emotion.lower().strip() if emotion else "neutral")

        if emo in ["happy", "cheerful", "energetic"]:
            emo = "excited"
        elif emo in ["sad", "apologetic", "caring", "understanding"]:
            emo = "sympathetic"
        elif emo in ["warm", "passionate"]:
            emo = "emotional"
        elif emo not in ["neutral", "excited", "sympathetic", "emotional"]:
            emo = "neutral"

        cache_key = f"{canonical_voice}:{emo}"
        prompt = self.prompt_cache.get(cache_key)

        if prompt is None:
            prompt = self.prompt_cache.get(f"{canonical_voice}:neutral")
        if prompt is None:
            prompt = self.prompt_cache.get(f"{self.primary_voice_id}:neutral")

        return prompt

    def get_config(self, voice: Optional[str] = None) -> Dict[str, Any]:
        canonical_voice = self.resolve_voice_id(voice)
        return self.voice_configs.get(canonical_voice, self.voice_configs.get(self.primary_voice_id, {}))

    def get_metadata(self, voice: Optional[str] = None) -> Dict[str, Any]:
        canonical_voice = self.resolve_voice_id(voice)
        return self.voice_metadata.get(canonical_voice, self.voice_metadata.get(self.primary_voice_id, {}))

    def list_voices(self) -> List[Dict[str, Any]]:
        voices = []
        for voice_id, cfg in self.voice_configs.items():
            meta = self.voice_metadata.get(voice_id, {})
            emotions = [k.split(":")[1] for k in self.prompt_cache.keys() if k.startswith(f"{voice_id}:") and k.split(":")[1] != "default"]
            voices.append({
                "voice_id": voice_id,
                "name": cfg.get("name", voice_id),
                "gender": cfg.get("gender", "unknown"),
                "language": cfg.get("language", "en-US"),
                "accent": cfg.get("accent", ""),
                "default_speed": cfg.get("default_speed", 0.9),
                "default_guidance_scale": cfg.get("default_guidance_scale", 2.2),
                "available_emotions": sorted(list(set(emotions))),
                "style": meta.get("style", ""),
                "license": meta.get("license", "Apache-2.0")
            })
        return voices
