"""
Pydantic Schemas for Arabic Voice Model API
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


class ArabicSynthesizeRequest(BaseModel):
    """Full-featured telephony Arabic TTS synthesis request."""
    text: str = Field(..., description="Arabic text to synthesize. Supports emotion tags ([متحمس], [متعاطف], [مؤثر], [طبيعي]).")
    voice: Optional[str] = Field("voice_arabic_clara", description="Voice ID or alias.")
    emotion: Optional[str] = Field(None, description="Emotion style: neutral, excited, sympathetic, emotional.")
    language: Optional[str] = Field("najdi arabic", description="Arabic dialect (najdi arabic, standard arabic, hijazi arabic).")
    speed: Optional[float] = Field(0.92, description="Speaking cadence multiplier (calibrated default: 0.92).", ge=0.5, le=2.0)
    sample_rate: Optional[int] = Field(8000, description="Target sample rate (8000 for PSTN, 16000 for HD, 24000 for native).")
    format: Optional[Literal["pcm16", "mulaw", "alaw", "wav", "raw"]] = Field("pcm16", description="Audio encoding format.")
    ode_steps: Optional[int] = Field(16, description="Flow-matching ODE steps (16 for real-time, 24-32 for studio).", ge=4, le=64)
    guidance_scale: Optional[float] = Field(2.2, description="Classifier-free guidance scale (calibrated default: 2.2).", ge=1.0, le=5.0)
    class_temperature: Optional[float] = Field(0.20, description="Class sampling temperature (calibrated default: 0.20).", ge=0.0, le=1.0)
    normalize_text: Optional[bool] = Field(True, description="Whether to apply deterministic telephony normalization.")
    add_breath: Optional[bool] = Field(True, description="Whether to prepend emotion-tailored physiological airway breath.")
    apply_eq: Optional[bool] = Field(True, description="Whether to apply Pillar 5 studio mastering & proximity EQ chain.")


class ArabicSynthesizeResponse(BaseModel):
    """Response payload containing encoded audio and latency metrics."""
    audio_base64: str = Field(..., description="Base64-encoded audio bytes.")
    audio_duration_sec: float
    inference_time_ms: float
    ttfa_ms: float
    rtf: float
    sample_rate: int
    format: str
    voice: str
    emotion: str
    normalized_text: str
    has_breath: bool
