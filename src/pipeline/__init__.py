"""Modular humanization pipeline for real-time Arabic TTS telephony."""

from .context_prosody import ContextProsodyAnalyzer, DialogIntent, ProsodyHints
from .normalizer_unified import UnifiedNormalizer
from .code_switch import CodeSwitchSegmenter, TextSegment
from .telephony_audio import TelephonyAudioProcessor
from .prosody_rules import BreatheGate

__all__ = [
    "ContextProsodyAnalyzer",
    "DialogIntent",
    "ProsodyHints",
    "UnifiedNormalizer",
    "CodeSwitchSegmenter",
    "TextSegment",
    "TelephonyAudioProcessor",
    "BreatheGate",
]
