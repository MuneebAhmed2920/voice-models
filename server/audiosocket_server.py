"""
Native Asterisk AudioSocket Protocol Server.
Enables direct Asterisk PBX / Vicidial dialplan connectivity over TCP (port 9092)
for zero-overhead 8kHz/16kHz Linear PCM16 bidirectional telephony audio streaming.
"""

import asyncio
import struct
import uuid
import logging
from typing import Optional, Callable, Dict, Any

from runtime.engine import get_engine, OmniVoiceEngine

logger = logging.getLogger("omnivoice.audiosocket")

# AudioSocket Protocol Constants (Asterisk app_audiosocket RFC)
TYPE_UUID = 0x01        # Connection UUID (16 raw bytes)
TYPE_SILENCE = 0x02     # Silence / Keep-alive frame
TYPE_AUDIO = 0x10       # Linear 16-bit Signed PCM Audio (mono)
TYPE_ERROR = 0xFF       # Error / Disconnect

HEADER_FORMAT = ">BH"   # 1 byte type + 2 bytes length (Big-Endian)
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 3 bytes


class AsteriskAudioSocketSession:
    """
    Manages an active Asterisk AudioSocket call session.
    """

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, sample_rate: int = 8000):
        self.reader = reader
        self.writer = writer
        self.sample_rate = sample_rate
        self.call_uuid: Optional[str] = None
        self.is_active = True
        self.engine: OmniVoiceEngine = get_engine()

    async def send_audio_frame(self, pcm16_bytes: bytes):
        """Sends a PCM16 audio frame to Asterisk channel."""
        if not self.is_active or self.writer.is_closing():
            return
        
        # AudioSocket message: [TYPE (1B)] [LENGTH (2B)] [PAYLOAD (NB)]
        header = struct.pack(HEADER_FORMAT, TYPE_AUDIO, len(pcm16_bytes))
        self.writer.write(header + pcm16_bytes)
        await self.writer.drain()

    async def speak_text(self, text: str, voice: str = "voice_clara", speed: float = 0.92):
        """
        Synthesizes text and streams 20ms frames directly to Asterisk AudioSocket.
        """
        if not self.is_active:
            return

        loop = asyncio.get_running_loop()
        audio_bytes, metrics = await loop.run_in_executor(
            None,
            lambda: self.engine.synthesize(
                text=text,
                voice=voice,
                speed=speed,
                sample_rate=self.sample_rate,
                format_type="pcm16"
            )
        )

        frames = self.engine.audio_processor.frame_audio(
            audio_bytes,
            sample_rate=self.sample_rate,
            frame_duration_ms=20,
            bytes_per_sample=2
        )

        for frame in frames:
            if not self.is_active:
                break
            await self.send_audio_frame(frame)
            await asyncio.sleep(0.018)  # 20ms real-time pacing


class AsteriskAudioSocketServer:
    """
    TCP Server hosting the Asterisk AudioSocket bridge.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 9092, sample_rate: int = 8000):
        self.host = host
        self.port = port
        self.sample_rate = sample_rate
        self.server: Optional[asyncio.Server] = None
        self.sessions: Dict[str, AsteriskAudioSocketSession] = {}

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handles incoming Asterisk AudioSocket TCP connection."""
        session = AsteriskAudioSocketSession(reader, writer, sample_rate=self.sample_rate)
        peer = writer.get_extra_info("peername")
        logger.info(f"Asterisk AudioSocket connection from {peer}")

        try:
            while session.is_active:
                # Read 3-byte AudioSocket header
                header_data = await reader.readexactly(HEADER_SIZE)
                msg_type, payload_len = struct.unpack(HEADER_FORMAT, header_data)

                # Read payload
                payload = await reader.readexactly(payload_len) if payload_len > 0 else b""

                if msg_type == TYPE_UUID:
                    # 16-byte UUID from Asterisk
                    if len(payload) == 16:
                        session.call_uuid = str(uuid.UUID(bytes=payload))
                    else:
                        session.call_uuid = payload.decode("utf-8", errors="ignore")
                    self.sessions[session.call_uuid] = session
                    logger.info(f"Asterisk Call UUID registered: {session.call_uuid}")

                elif msg_type == TYPE_AUDIO:
                    # Inbound caller audio frame (e.g. 320 bytes PCM16 for 20ms @ 8kHz)
                    # Forward to STT / Voice Bot if connected
                    pass

                elif msg_type == TYPE_SILENCE:
                    pass

                elif msg_type == TYPE_ERROR:
                    logger.info(f"Asterisk channel closed for session {session.call_uuid}")
                    break

        except (asyncio.IncompleteReadError, ConnectionResetError):
            logger.info("Asterisk channel disconnected.")
        except Exception as e:
            logger.error(f"AudioSocket session error: {e}")
        finally:
            session.is_active = False
            if session.call_uuid and session.call_uuid in self.sessions:
                del self.sessions[session.call_uuid]
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def start(self):
        """Starts the AudioSocket TCP listener."""
        self.server = await asyncio.start_server(self.handle_client, self.host, self.port)
        logger.info(f"🚀 Asterisk AudioSocket server listening on {self.host}:{self.port}")

    async def stop(self):
        """Stops the AudioSocket TCP listener."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("Asterisk AudioSocket server stopped.")
