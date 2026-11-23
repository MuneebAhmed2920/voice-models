#!/usr/bin/env python3
"""
server/app.py

FastAPI REST & Streaming Server for Marukh English Voice Model.
Endpoints:
- GET  /health              -> Health check & model readiness
- POST /v1/tts/synthesize   -> Full parameter synthesis
- POST /v1/audio/speech     -> Drop-in OpenAI speech format
"""

import io
from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from ..runtime.engine import EnglishVoiceModel
from .schemas import SynthesizeRequest, OpenAISpeechRequest

model_instance: EnglishVoiceModel = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_instance
    print("[Server] Initializing EnglishVoiceModel singleton...")
    model_instance = EnglishVoiceModel()
    yield
    print("[Server] Shutting down.")

app = FastAPI(
    title="Marukh English Voice Model API",
    description="Dedicated production English female voice model of Mahrukh for telephony and AI voice agents.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {
        "status": "ready" if model_instance is not None else "initializing",
        "model": "marukh_english",
        "speaker": "Mahrukh (ماہ رخ)",
        "language": "en"
    }

@app.post("/v1/tts/synthesize")
def synthesize(req: SynthesizeRequest):
    if not model_instance:
        raise HTTPException(status_code=503, detail="Model is still initializing.")
    
    wav_bytes = model_instance.synthesize(
        text=req.text,
        speed=req.speed,
        sample_rate=req.sample_rate,
        telephony_filter=req.telephony_filter,
        num_step=req.num_step,
        guidance_scale=req.guidance_scale,
        return_bytes=True
    )
    return Response(content=wav_bytes, media_type="audio/wav")

@app.post("/v1/audio/speech")
def openai_speech(req: OpenAISpeechRequest):
    if not model_instance:
        raise HTTPException(status_code=503, detail="Model is still initializing.")
    
    wav_bytes = model_instance.synthesize(
        text=req.input,
        speed=req.speed,
        sample_rate=24000,
        return_bytes=True
    )
    return Response(content=wav_bytes, media_type="audio/wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
