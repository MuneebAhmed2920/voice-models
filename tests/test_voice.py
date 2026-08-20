"""
Standalone verification test suite for voice_clara package.
"""

import os
import torch

_VOICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_package_structure():
    assert os.path.exists(os.path.join(_VOICE_DIR, "prompt", "voice_prompt.pt")), "Missing voice_prompt.pt"
    assert os.path.exists(os.path.join(_VOICE_DIR, "config", "config.json")), "Missing config.json"
    assert os.path.exists(os.path.join(_VOICE_DIR, "config", "metadata.json")), "Missing metadata.json"

    prompt = torch.load(os.path.join(_VOICE_DIR, "prompt", "voice_prompt.pt"), map_location="cpu", weights_only=False)
    assert prompt is not None, "Prompt could not be loaded"
    print("✅ voice_clara package structure verified successfully.")


if __name__ == "__main__":
    test_package_structure()
