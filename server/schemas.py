from pydantic import BaseModel, Field
from typing import Optional

class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="Raw text in Urdu or code-switched loanwords to synthesize")
    speed: Optional[float] = Field(1.12, description="Speaking rate multiplier (e.g. 1.0 to 1.25)")
    sample_rate: Optional[int] = Field(24000, description="Target sample rate: 24000 (HD) or 8000 (PSTN telephony)")
    telephony_filter: Optional[bool] = Field(None, description="Enable 300Hz-3400Hz telephone line bandpass filter")

class OpenAISpeechRequest(BaseModel):
    input: str = Field(..., description="Text input to speak")
    model: Optional[str] = Field("mahrukh_urdu", description="Model ID")
    voice: Optional[str] = Field("mahrukh", description="Voice identifier")
    speed: Optional[float] = Field(1.0, description="Speech rate")
    response_format: Optional[str] = Field("wav", description="Audio format (wav supported)")
