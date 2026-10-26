"""Spanish Voice TTS Package.

Production-grade Spanish TTS powered by OmniVoice base model,
Latin American Spanish text normalization, and native Colombian voice conditioning.

Voices:
- female / spanish_female (Prompt_01: Colombian Female, 258 Hz mean F0)
- male / spanish_male (Prompt_07: Colombian Male, 125 Hz mean F0)
"""

from .model import SpanishVoice
from .normalizer import normalize_spanish

__all__ = ["SpanishVoice", "normalize_spanish"]
__version__ = "1.0.0"
