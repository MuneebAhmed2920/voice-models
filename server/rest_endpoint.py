"""
REST API Endpoints for OmniVoice TTS Engine.
Includes Telephony Synthesize, OpenAI Speech API compatibility, Health & Metrics.
"""

import time
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import Response, StreamingResponse
import numpy as np
import torch

from runtime.engine import get_engine, OmniVoiceEngine
from server.schemas import (
    SynthesizeRequest,
    OpenAISpeechRequest,
    HealthResponse,
    MetricsResponse,
    VoiceMetadataItem
)

router = APIRouter()

# Global metrics tracking
_METRICS = {
    "total_requests": 0,
    "total_audio_seconds": 0.0,
    "total_ttfa_ms": 0.0,
    "total_rtf": 0.0,
    "active_connections": 0
}


def get_engine_dependency() -> OmniVoiceEngine:
    return get_engine()


@router.get("/health", response_model=HealthResponse)
def health_check(engine: OmniVoiceEngine = Depends(get_engine_dependency)):
    """Health check returning GPU / CPU status, VRAM usage, and active voices."""
    status = engine.get_status()
    voices_meta = [VoiceMetadataItem(**v) for v in status.get("voices", [])]
    return HealthResponse(
        status=status.get("status", "unknown"),
        device=status.get("device", "unknown"),
        dtype=status.get("dtype", "unknown"),
        gpu_name=status.get("gpu_name"),
        vram_used_mb=status.get("vram_used_mb", 0.0),
        vram_total_mb=status.get("vram_total_mb", 0.0),
        voices=voices_meta
    )


@router.get("/v1/voices")
def list_voices(engine: OmniVoiceEngine = Depends(get_engine_dependency)):
    """Returns list of registered voices and supported emotion styles."""
    return {"voices": engine.voice_manager.list_voices()}


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(engine: OmniVoiceEngine = Depends(get_engine_dependency)):
    """Returns real-time synthesis performance statistics."""
    req_count = max(1, _METRICS["total_requests"])
    avg_ttfa = _METRICS["total_ttfa_ms"] / req_count if _METRICS["total_requests"] > 0 else 0.0
    avg_rtf = _METRICS["total_rtf"] / req_count if _METRICS["total_requests"] > 0 else 0.0

    vram = 0.0
    if engine.device.type == "cuda" and torch.cuda.is_available():
        vram = round(torch.cuda.memory_allocated(engine.device) / (1024 * 1024), 2)

    return MetricsResponse(
        total_requests=_METRICS["total_requests"],
        total_audio_seconds=round(_METRICS["total_audio_seconds"], 2),
        avg_ttfa_ms=round(avg_ttfa, 2),
        avg_rtf=round(avg_rtf, 4),
        active_connections=_METRICS["active_connections"],
        gpu_vram_used_mb=vram
    )


@router.post("/v1/tts/synthesize")
def synthesize_telephony(
    req: SynthesizeRequest,
    engine: OmniVoiceEngine = Depends(get_engine_dependency)
):
    """
    Primary Telephony Synthesis Endpoint.
    Returns raw PCM16, G.711 μ-law, G.711 a-law, or WAV audio formatted for Vicidial/Asterisk.
    """
    try:
        audio_bytes, metrics = engine.synthesize(
            text=req.text,
            voice=req.voice,
            emotion=req.emotion,
            speed=req.speed,
            sample_rate=req.sample_rate,
            format_type=req.format,
            ode_steps=req.ode_steps,
            guidance_scale=req.guidance_scale,
            normalize_text=req.normalize,
            add_breath=req.add_breath
        )

        # Update metrics
        _METRICS["total_requests"] += 1
        _METRICS["total_audio_seconds"] += metrics.get("audio_duration_sec", 0.0)
        _METRICS["total_ttfa_ms"] += metrics.get("ttfa_ms", 0.0)
        _METRICS["total_rtf"] += metrics.get("rtf", 0.0)

        # Media type header mapping
        media_type = "application/octet-stream"
        fmt = (req.format or "pcm16").lower()
        if fmt == "wav":
            media_type = "audio/wav"
        elif fmt in ["mulaw", "pcmu", "ulaw"]:
            media_type = "audio/basic"
        elif fmt in ["alaw", "pcma"]:
            media_type = "audio/x-alaw-basic"
        elif fmt == "pcm16":
            media_type = "audio/l16"

        headers = {
            "X-TTS-TTFA-MS": str(metrics.get("ttfa_ms", 0.0)),
            "X-TTS-Inference-MS": str(metrics.get("inference_time_ms", 0.0)),
            "X-TTS-Audio-Duration": str(metrics.get("audio_duration_sec", 0.0)),
            "X-TTS-RTF": str(metrics.get("rtf", 0.0)),
            "X-TTS-Sample-Rate": str(metrics.get("sample_rate", 8000)),
            "X-TTS-Format": fmt,
            "X-TTS-Emotion": str(metrics.get("emotion", "neutral")),
            "X-TTS-Normalized-Text": str(metrics.get("normalized_text", ""))[:200]
        }

        if req.stream:
            # Stream in 20ms telephony frames
            frames = engine.audio_processor.frame_audio(
                audio_bytes,
                sample_rate=req.sample_rate or 8000,
                frame_duration_ms=20,
                bytes_per_sample=1 if fmt in ["mulaw", "alaw"] else 2
            )
            return StreamingResponse(frames, media_type=media_type, headers=headers)

        return Response(content=audio_bytes, media_type=media_type, headers=headers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS synthesis error: {str(e)}")


@router.post("/v1/audio/speech")
def openai_speech_compatibility(
    req: OpenAISpeechRequest,
    engine: OmniVoiceEngine = Depends(get_engine_dependency)
):
    """
    OpenAI-compatible Audio Speech API endpoint.
    Direct drop-in replacement for OpenAI TTS API in existing voice bot frameworks.
    """
    try:
        format_mapping = {
            "mp3": "wav",  # High quality lossless container fallback
            "opus": "pcm16",
            "aac": "wav",
            "flac": "wav",
            "wav": "wav",
            "pcm": "pcm16",
            "mulaw": "mulaw"
        }
        target_fmt = format_mapping.get(req.response_format.lower(), "wav")
        sample_rate = 24000 if target_fmt == "wav" else 8000

        audio_bytes, metrics = engine.synthesize(
            text=req.input,
            voice=req.voice,
            speed=req.speed,
            sample_rate=sample_rate,
            format_type=target_fmt,
            normalize_text=True
        )

        _METRICS["total_requests"] += 1
        _METRICS["total_audio_seconds"] += metrics.get("audio_duration_sec", 0.0)
        _METRICS["total_ttfa_ms"] += metrics.get("ttfa_ms", 0.0)
        _METRICS["total_rtf"] += metrics.get("rtf", 0.0)

        media_type = "audio/wav" if target_fmt == "wav" else "audio/l16"
        return Response(content=audio_bytes, media_type=media_type)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI TTS API error: {str(e)}")
