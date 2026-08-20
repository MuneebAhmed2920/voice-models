"""
Bidirectional WebSocket Protocol for AI Voice Bots & Asterisk Telephony Bridges.
Supports real-time token streaming, speculative clause TTS synthesis, and instant barge-in cancellation.
"""

import asyncio
import json
import uuid
import time
import logging
from typing import Optional, Dict, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np

from runtime.engine import get_engine, OmniVoiceEngine
from runtime.streamer import ClauseStreamer

logger = logging.getLogger("omnivoice.ws")
ws_router = APIRouter()


@ws_router.websocket("/ws/tts")
@ws_router.websocket("/ws/stream")
async def websocket_tts_endpoint(websocket: WebSocket):
    """
    Real-time streaming WebSocket endpoint for Voice Bots and Asterisk Audio Bridges.
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"WebSocket client connected: session_id={session_id}")

    engine: OmniVoiceEngine = get_engine()
    streamer = ClauseStreamer()

    # Session settings (defaults)
    session_cfg = {
        "voice": "voice_clara",
        "emotion": None,
        "speed": 0.92,
        "sample_rate": 8000,
        "format": "pcm16",
        "ode_steps": 16,
        "guidance_scale": 2.2,
        "normalize": True,
        "frame_duration_ms": 20
    }

    # Queue for synthesizing clauses asynchronously
    clause_queue = asyncio.Queue()
    is_interrupted = False

    async def _synthesize_and_stream_clause(clause_text: str, clause_idx: int):
        nonlocal is_interrupted
        if is_interrupted or not clause_text.strip():
            return

        t0 = time.time()
        try:
            # Run blocking synthesis in thread pool to keep asyncio event loop responsive
            loop = asyncio.get_running_loop()
            audio_bytes, metrics = await loop.run_in_executor(
                None,
                lambda: engine.synthesize(
                    text=clause_text,
                    voice=session_cfg["voice"],
                    emotion=session_cfg["emotion"],
                    speed=session_cfg["speed"],
                    sample_rate=session_cfg["sample_rate"],
                    format_type=session_cfg["format"],
                    ode_steps=session_cfg["ode_steps"],
                    guidance_scale=session_cfg["guidance_scale"],
                    normalize_text=session_cfg["normalize"]
                )
            )

            if is_interrupted:
                return

            # Send audio_start event
            await websocket.send_text(json.dumps({
                "event": "audio_start",
                "session_id": session_id,
                "clause_index": clause_idx,
                "text": clause_text,
                "normalized_text": metrics.get("normalized_text", ""),
                "emotion": metrics.get("emotion", "neutral"),
                "sample_rate": session_cfg["sample_rate"],
                "format": session_cfg["format"],
                "ttfa_ms": metrics.get("ttfa_ms", 0.0)
            }))

            # Stream audio frames in 20ms telephony chunks
            bytes_per_sample = 1 if session_cfg["format"] in ["mulaw", "alaw"] else 2
            frames = list(engine.audio_processor.frame_audio(
                audio_bytes,
                sample_rate=session_cfg["sample_rate"],
                frame_duration_ms=session_cfg["frame_duration_ms"],
                bytes_per_sample=bytes_per_sample
            ))

            for frame in frames:
                if is_interrupted:
                    break
                await websocket.send_bytes(frame)
                # Slight yield to allow interleaving control messages
                await asyncio.sleep(0.001)

            if not is_interrupted:
                # Send audio_end event
                await websocket.send_text(json.dumps({
                    "event": "audio_end",
                    "session_id": session_id,
                    "clause_index": clause_idx,
                    "duration_sec": metrics.get("audio_duration_sec", 0.0),
                    "rtf": metrics.get("rtf", 0.0),
                    "total_time_ms": round((time.time() - t0) * 1000.0, 2)
                }))

        except Exception as e:
            logger.error(f"Error synthesizing clause '{clause_text}': {e}")
            await websocket.send_text(json.dumps({
                "event": "error",
                "session_id": session_id,
                "clause_index": clause_idx,
                "message": str(e)
            }))

    try:
        while True:
            raw_message = await websocket.receive()
            if raw_message.get("type") == "websocket.disconnect":
                break

            text_data = raw_message.get("text")
            if not text_data:
                continue

            try:
                data = json.loads(text_data)
            except json.JSONDecodeError:
                continue

            event_type = data.get("event", "").lower()

            if event_type == "start":
                is_interrupted = False
                streamer.reset()
                session_cfg["voice"] = data.get("voice", session_cfg["voice"])
                session_cfg["emotion"] = data.get("emotion", session_cfg["emotion"])
                session_cfg["speed"] = float(data.get("speed", session_cfg["speed"]))
                session_cfg["sample_rate"] = int(data.get("sample_rate", session_cfg["sample_rate"]))
                session_cfg["format"] = data.get("format", session_cfg["format"])
                session_cfg["ode_steps"] = int(data.get("ode_steps", session_cfg["ode_steps"]))
                session_cfg["guidance_scale"] = float(data.get("guidance_scale", session_cfg["guidance_scale"]))
                session_cfg["normalize"] = bool(data.get("normalize", session_cfg["normalize"]))

                await websocket.send_text(json.dumps({
                    "event": "session_started",
                    "session_id": session_id,
                    "config": session_cfg
                }))

            elif event_type == "text_chunk":
                is_interrupted = False
                token = data.get("text", "")
                if token:
                    clauses = streamer.feed_token(token)
                    for clause in clauses:
                        await _synthesize_and_stream_clause(clause, streamer.clause_index)

            elif event_type == "flush":
                if not is_interrupted:
                    tail_clauses = streamer.flush()
                    for clause in tail_clauses:
                        await _synthesize_and_stream_clause(clause, streamer.clause_index)
                    await websocket.send_text(json.dumps({
                        "event": "stream_completed",
                        "session_id": session_id,
                        "total_clauses": streamer.clause_index
                    }))

            elif event_type == "interrupt":
                # Caller Barge-In detected: cancel in-flight audio immediately
                is_interrupted = True
                streamer.interrupt()
                logger.info(f"Barge-in interrupt received for session {session_id}")
                await websocket.send_text(json.dumps({
                    "event": "interrupted",
                    "session_id": session_id,
                    "timestamp": time.time()
                }))

            elif event_type == "stop":
                is_interrupted = True
                streamer.interrupt()
                await websocket.send_text(json.dumps({
                    "event": "session_stopped",
                    "session_id": session_id
                }))
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: session_id={session_id}")
    except Exception as e:
        logger.error(f"WebSocket session error: {e}")
    finally:
        is_interrupted = True
        streamer.interrupt()
