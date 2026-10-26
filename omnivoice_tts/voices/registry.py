"""
In-Memory Fixed Voice Registry for omnivoice_tts.
Zero-overhead O(1) voice profile lookup.
"""

import os
import torch
from typing import Dict, Any, List

VOICES_DIR = os.path.dirname(os.path.abspath(__file__))
SUPPORTED_VOICES = ["female_1", "female_2", "male_1", "male_2"]

class VoiceRegistry:
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_all()

    def _load_all(self):
        for v_id in SUPPORTED_VOICES:
            pt_path = os.path.join(VOICES_DIR, f"{v_id}.pt")
            if os.path.exists(pt_path):
                self._cache[v_id] = torch.load(pt_path, map_location="cpu", weights_only=False)
            else:
                self._cache[v_id] = {
                    "voice_id": v_id,
                    "persona": v_id,
                    "instruct": "female, american accent, young adult, moderate pitch",
                    "speed": 1.0
                }

    def get(self, voice_id: str) -> Dict[str, Any]:
        """Returns precomputed voice profile with 0ms overhead."""
        if voice_id not in self._cache:
            voice_id = "female_1" # Default safe fallback
        return self._cache[voice_id]

    def list_voices(self) -> List[str]:
        return list(self._cache.keys())
