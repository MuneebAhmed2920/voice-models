"""
Mock AI Voice Bot Client.
Simulates real-time LLM token streaming, clause emission, audio reception, and caller barge-in.
"""

import sys
import json
import time
import asyncio
from typing import List, Optional

try:
    import websockets
except ImportError:
    websockets = None


class MockAIVoiceBot:
    """
    Simulates an LLM conversational agent interacting with OmniVoice TTS service via WebSocket.
    """

    def __init__(self, ws_url: str = "ws://127.0.0.1:8000/ws/tts"):
        self.ws_url = ws_url
        self.audio_bytes_received = 0
        self.frames_received = 0
        self.events_received: List[Dict[str, Any]] = []

    async def run_simulation(
        self,
        token_stream: List[str],
        voice: str = "voice_clara",
        sample_rate: int = 8000,
        simulate_bargein: bool = False,
        bargein_delay_sec: float = 2.0
    ):
        """
        Runs full end-to-end token streaming simulation.
        """
        if websockets is None:
            print("Error: 'websockets' library is required to run MockAIVoiceBot.")
            return

        print(f"Connecting to OmniVoice TTS WebSocket at {self.ws_url}...")
        async with websockets.connect(self.ws_url) as ws:
            # 1. Start handshake
            start_payload = {
                "event": "start",
                "voice": voice,
                "sample_rate": sample_rate,
                "format": "pcm16",
                "speed": 0.92,
                "ode_steps": 16,
                "normalize": True
            }
            await ws.send(json.dumps(start_payload))
            print(f"Sent 'start' handshake: {start_payload}")

            # Background listener task for incoming audio and event messages
            async def receiver():
                try:
                    while True:
                        msg = await ws.recv()
                        if isinstance(msg, bytes):
                            self.audio_bytes_received += len(msg)
                            self.frames_received += 1
                        else:
                            evt = json.loads(msg)
                            self.events_received.append(evt)
                            evt_name = evt.get("event")
                            if evt_name == "session_started":
                                print(f"✅ Session started: {evt.get('session_id')}")
                            elif evt_name == "audio_start":
                                print(f"🎙️ Audio Start [Clause {evt.get('clause_index')}]: '{evt.get('text')}' (TTFA: {evt.get('ttfa_ms')}ms)")
                            elif evt_name == "audio_end":
                                print(f"⏹️ Audio End [Clause {evt.get('clause_index')}]: Duration: {evt.get('duration_sec')}s, RTF: {evt.get('rtf')}")
                            elif evt_name == "interrupted":
                                print(f"⚡ Barge-in interruption confirmed by server.")
                            elif evt_name == "stream_completed":
                                print(f"🎉 Stream completed. Total clauses: {evt.get('total_clauses')}")
                                break
                            elif evt_name == "error":
                                print(f"❌ Server Error: {evt.get('message')}")
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    print(f"Receiver error: {e}")

            recv_task = asyncio.create_task(receiver())

            # 2. Feed LLM tokens with conversational pacing
            start_time = time.time()
            for token in token_stream:
                await asyncio.sleep(0.06)  # ~16 tokens/sec LLM generation speed
                await ws.send(json.dumps({
                    "event": "text_chunk",
                    "text": token
                }))

                # Simulate caller barge-in interruption if enabled
                if simulate_bargein and (time.time() - start_time) > bargein_delay_sec:
                    print("\n🛑 Caller spoke! Triggering barge-in interruption...")
                    await ws.send(json.dumps({"event": "interrupt"}))
                    await asyncio.sleep(0.5)
                    break

            if not simulate_bargein:
                # 3. Flush at end of turn
                await ws.send(json.dumps({"event": "flush"}))

            # Wait for remaining audio to play out or receiver to finish
            await asyncio.sleep(3.0)
            recv_task.cancel()

            print(f"\n--- Simulation Results ---")
            print(f"Total PCM16 Audio Received: {self.audio_bytes_received} bytes (~{self.audio_bytes_received / (sample_rate * 2):.2f}s)")
            print(f"Total 20ms Frames Received: {self.frames_received}")
            print(f"Total Events: {len(self.events_received)}")


if __name__ == "__main__":
    sample_tokens = [
        "[excited] Hello! ", "Thank you for ", "calling Clara ", "customer care. ",
        "I see your ", "appointment is confirmed ", "for August 24th ", "at 3:30 PM. ",
        "Is there anything ", "else I can assist ", "you with today?"
    ]
    bot = MockAIVoiceBot()
    asyncio.run(bot.run_simulation(sample_tokens, simulate_bargein=False))
