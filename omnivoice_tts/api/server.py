"""
Minimal High-Performance REST API for omnivoice_tts.
Endpoint: POST /tts
"""

from __future__ import annotations

import os
import sys
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from typing import Optional, Literal

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from omnivoice_tts.engine import get_engine, QualityMode
from omnivoice_tts.audio.codecs import AudioFormat

app = FastAPI(title="OmniVoice TTS Engine", version="1.0.0")

class TTSRequest(BaseModel):
    text: str = Field(..., description="English text to synthesize")
    voice_id: Literal["female_1", "female_2", "male_1", "male_2"] = Field(default="female_1")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    quality: QualityMode = Field(default="fast")
    sample_rate: Literal[8000, 16000, 24000] = Field(default=24000)
    output_format: AudioFormat = Field(default="wav")
    include_breath: bool = Field(default=False)

@app.on_event("startup")
def startup_event():
    get_engine()

@app.get("/health")
def health_check():
    engine = get_engine()
    return {
        "status": "READY",
        "voices": engine.registry.list_voices(),
        "model": "k2-fsa/OmniVoice-English-Production"
    }

@app.post("/tts")
def synthesize(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    engine = get_engine()
    audio_bytes, metrics = engine.generate(
        text=req.text,
        voice_id=req.voice_id,
        speed=req.speed,
        quality=req.quality,
        sample_rate=req.sample_rate,
        output_format=req.output_format,
        include_breath=req.include_breath
    )
    
    media_type = "audio/wav" if req.output_format == "wav" else "application/octet-stream"
    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={
            "X-TTFA-Sec": str(round(metrics["ttfa_sec"], 3)),
            "X-RTF": str(round(metrics["rtf"], 3)),
            "X-Steps": str(metrics["steps_used"]),
            "X-Voice": req.voice_id
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
