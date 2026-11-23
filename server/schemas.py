from pydantic import BaseModel, Field
from typing import Optional

class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="English text to synthesize")
    speed: Optional[float] = Field(1.0, ge=0.5, le=2.0, description="Speaking rate multiplier")
    sample_rate: Optional[int] = Field(24000, description="Output sample rate: 24000 or 8000")
    telephony_filter: Optional[bool] = Field(None, description="Apply 300Hz-3400Hz telephone line bandpass filter")
    num_step: Optional[int] = Field(32, ge=8, le=64, description="Diffusion steps")
    guidance_scale: Optional[float] = Field(2.0, ge=1.0, le=5.0, description="Guidance scale")

class OpenAISpeechRequest(BaseModel):
    model: str = Field("marukh_english", description="Model identifier")
    input: str = Field(..., description="The text to generate audio for")
    voice: Optional[str] = Field("marukh", description="Voice persona")
    response_format: Optional[str] = Field("wav", description="Audio format: wav")
    speed: Optional[float] = Field(1.0, ge=0.25, le=4.0, description="Speed of generated audio")
