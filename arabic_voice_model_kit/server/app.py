"""
Arabic Voice Model API Server (FastAPI + WebSocket Telephony Stream)
"""

import base64
import os
import io
import time
from typing import Optional
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from arabic_voice_model_kit.runtime.engine import get_engine, ArabicVoiceEngine
from arabic_voice_model_kit.server.schemas import ArabicSynthesizeRequest, ArabicSynthesizeResponse

app = FastAPI(
    title="Arabic Human-Realistic Voice Model Server",
    description="Real-time conversational Arabic voice synthesis with 7-pillar humanization (airway breathing, proximity EQ, 0.92x cadence).",
    version="1.0.0"
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
    """Returns engine, hardware, and voice status."""
    engine = get_engine()
    status = engine.get_status()
    return {"status": "healthy", "engine": status}


@app.get("/voices")
def list_voices():
    """Lists available Arabic voices and supported emotion styles."""
    engine = get_engine()
    return {"voices": engine.voice_manager.list_voices()}


@app.post("/synthesize")
def synthesize(req: ArabicSynthesizeRequest):
    """
    Synthesizes speech and returns JSON with Base64 audio + latency metrics.
    """
    engine = get_engine()
    try:
        audio_bytes, metrics = engine.synthesize(
            text=req.text,
            voice=req.voice,
            emotion=req.emotion,
            language=req.language or "najdi arabic",
            speed=req.speed,
            sample_rate=req.sample_rate,
            format_type=req.format or "pcm16",
            ode_steps=req.ode_steps,
            guidance_scale=req.guidance_scale,
            class_temperature=req.class_temperature,
            normalize_text=req.normalize_text if req.normalize_text is not None else True,
            add_breath=req.add_breath if req.add_breath is not None else True,
            apply_eq=req.apply_eq if req.apply_eq is not None else True
        )
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        return ArabicSynthesizeResponse(
            audio_base64=audio_b64,
            audio_duration_sec=metrics["audio_duration_sec"],
            inference_time_ms=metrics["inference_time_ms"],
            ttfa_ms=metrics["ttfa_ms"],
            rtf=metrics["rtf"],
            sample_rate=metrics["sample_rate"],
            format=metrics["format"],
            voice=metrics["voice"],
            emotion=metrics["emotion"],
            normalized_text=metrics["normalized_text"],
            has_breath=metrics["has_breath"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize/raw")
def synthesize_raw(req: ArabicSynthesizeRequest):
    """
    Synthesizes speech and returns binary audio stream with appropriate Content-Type.
    """
    engine = get_engine()
    try:
        audio_bytes, metrics = engine.synthesize(
            text=req.text,
            voice=req.voice,
            emotion=req.emotion,
            language=req.language or "najdi arabic",
            speed=req.speed,
            sample_rate=req.sample_rate,
            format_type=req.format or "wav",
            ode_steps=req.ode_steps,
            guidance_scale=req.guidance_scale,
            class_temperature=req.class_temperature,
            normalize_text=req.normalize_text if req.normalize_text is not None else True,
            add_breath=req.add_breath if req.add_breath is not None else True,
            apply_eq=req.apply_eq if req.apply_eq is not None else True
        )
        fmt = (req.format or "wav").lower()
        content_type = "audio/wav" if fmt == "wav" else "application/octet-stream"
        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "X-Audio-Duration": str(metrics["audio_duration_sec"]),
                "X-RTF": str(metrics["rtf"]),
                "X-TTFA-MS": str(metrics["ttfa_ms"])
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
