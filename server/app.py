"""
FastAPI Application Factory for OmniVoice Production TTS Service.
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from runtime.engine import get_engine
from server.rest_endpoint import router as rest_router
from server.ws_endpoint import ws_router
from server.audiosocket_server import AsteriskAudioSocketServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("omnivoice.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown routines."""
    logger.info("Initializing OmniVoice Production TTS Service...")
    
    # 1. Initialize Engine & Warm up
    engine = get_engine(warmup=True)
    logger.info(f"OmniVoice Engine initialized on {engine.device} ({engine.torch_dtype})")

    # 2. Optionally start Asterisk AudioSocket Server
    enable_audiosocket = os.environ.get("ENABLE_AUDIOSOCKET", "true").lower() in ["1", "true", "yes"]
    audiosocket_port = int(os.environ.get("AUDIOSOCKET_PORT", "9092"))
    
    audiosocket_server = None
    if enable_audiosocket:
        try:
            audiosocket_server = AsteriskAudioSocketServer(host="0.0.0.0", port=audiosocket_port)
            await audiosocket_server.start()
        except Exception as e:
            logger.warning(f"Could not start Asterisk AudioSocket server on port {audiosocket_port}: {e}")

    yield

    # Shutdown routine
    logger.info("Shutting down OmniVoice TTS Service...")
    if audiosocket_server:
        await audiosocket_server.stop()


def create_app() -> FastAPI:
    """Creates and configures the production FastAPI instance."""
    app = FastAPI(
        title="OmniVoice Telephony & Voice Bot TTS Service",
        description="Production Ultra-Low-Latency TTS Engine optimized for Windows NVIDIA CUDA FP16 & Vicidial Call Centers.",
        version="1.0.0",
        lifespan=lifespan
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount REST and WebSocket routes
    app.include_router(rest_router)
    app.include_router(ws_router)

    @app.get("/")
    def root_info():
        return {
            "service": "OmniVoice TTS Telephony Engine",
            "status": "online",
            "docs": "/docs",
            "websocket_endpoint": "/ws/tts",
            "openai_compatible_endpoint": "/v1/audio/speech",
            "telephony_endpoint": "/v1/tts/synthesize",
            "health": "/health",
            "metrics": "/metrics"
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("server.app:app", host=host, port=port, reload=False, log_level="info")
