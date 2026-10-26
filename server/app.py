#!/usr/bin/env python3
"""
server/app.py

FastAPI production server for Mahrukh Urdu Voice Model.
Provides telephony synthesis endpoints and OpenAI TTS API drop-in compatibility.
"""

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from ..runtime.engine import UrduVoiceModel
from .schemas import SynthesizeRequest, OpenAISpeechRequest

model: UrduVoiceModel = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("[Server] Initializing Mahrukh Urdu Voice Model...")
    model = UrduVoiceModel()
    yield
    print("[Server] Shutting down.")

app = FastAPI(
    title="Mahrukh Urdu Voice Model API",
    description="Standalone Pakistani Urdu Telephony TTS Engine (Mahrukh)",
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
def health():
    return {
        "status": "healthy",
        "model": "mahrukh_urdu",
        "speaker": "Mahrukh",
        "language": "ur",
        "initialized": model is not None
    }

@app.post("/v1/tts/synthesize")
def synthesize(req: SynthesizeRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still loading")
    
    try:
        wav_bytes = model.synthesize(
            text=req.text,
            speed=req.speed,
            sample_rate=req.sample_rate or 24000,
            telephony_filter=req.telephony_filter,
            return_bytes=True
        )
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/audio/speech")
def openai_speech_compat(req: OpenAISpeechRequest):
    """OpenAI TTS drop-in compatibility endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    try:
        wav_bytes = model.synthesize(
            text=req.input,
            speed=req.speed * 1.12 if req.speed else 1.12,
            sample_rate=24000,
            return_bytes=True
        )
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
