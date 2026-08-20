"""
Pydantic Schemas for OmniVoice TTS Service (REST & WebSocket APIs).
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


class SynthesizeRequest(BaseModel):
    """Full-featured telephony TTS synthesis request."""
    text: str = Field(..., description="Text to synthesize. Can include emotion tags like [excited], [sympathetic].", example="Thank you for calling Clara support. Your balance is $45.50.")
    voice: Optional[str] = Field("voice_clara", description="Voice ID or alias (voice_clara, clara, female_1).")
    emotion: Optional[str] = Field(None, description="Emotion style: neutral, excited, sympathetic, emotional.")
    speed: Optional[float] = Field(None, description="Cadence multiplier (calibrated default: 0.92 for Clara).", ge=0.5, le=2.0)
    sample_rate: Optional[int] = Field(8000, description="Target sample rate in Hz: 8000 (PSTN), 16000 (HD Voice), 24000 (Native).")
    format: Optional[str] = Field("pcm16", description="Audio format: pcm16, mulaw, alaw, wav.")
    ode_steps: Optional[int] = Field(None, description="Flow-matching ODE steps (16 for low latency, 32 for studio).", ge=4, le=64)
    guidance_scale: Optional[float] = Field(None, description="CFG guidance scale (default: 2.2).", ge=1.0, le=5.0)
    normalize: bool = Field(True, description="Whether to apply telephony number/currency/date normalization.")
    add_breath: Optional[bool] = Field(None, description="Whether to prepend physiological breath sound.")
    stream: bool = Field(False, description="Whether to stream audio chunks.")


class OpenAISpeechRequest(BaseModel):
    """OpenAI /v1/audio/speech compatibility request schema."""
    model: str = Field("omnivoice", description="Model identifier.")
    input: str = Field(..., description="The text to generate audio for.")
    voice: str = Field("voice_clara", description="The voice to use.")
    response_format: Optional[str] = Field("wav", description="The audio format: wav, pcm, mulaw, alaw, mp3.")
    speed: Optional[float] = Field(1.0, description="The speed of the generated audio (0.5 to 2.0).")


class VoiceMetadataItem(BaseModel):
    voice_id: str
    name: str
    gender: str
    language: str
    accent: str
    default_speed: float
    default_guidance_scale: float
    available_emotions: List[str]
    style: str
    license: str


class HealthResponse(BaseModel):
    status: str
    service: str = "OmniVoice TTS Engine"
    device: str
    dtype: str
    gpu_name: Optional[str] = None
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    voices: List[VoiceMetadataItem]


class MetricsResponse(BaseModel):
    total_requests: int
    total_audio_seconds: float
    avg_ttfa_ms: float
    avg_rtf: float
    active_connections: int
    gpu_vram_used_mb: float
