"""
Unit tests for WebSocket Streaming Endpoint and Barge-in Protocol.
"""

import json
import unittest
from fastapi.testclient import TestClient
from server.app import create_app


class TestWebSocketEndpoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = TestClient(cls.app)

    def test_websocket_streaming_flow(self):
        with self.client.websocket_connect("/ws/tts") as websocket:
            # 1. Send start handshake
            websocket.send_text(json.dumps({
                "event": "start",
                "voice": "voice_clara",
                "sample_rate": 8000,
                "format": "pcm16",
                "ode_steps": 16
            }))

            start_resp = json.loads(websocket.receive_text())
            self.assertEqual(start_resp.get("event"), "session_started")

            # 2. Feed text chunk with single clause
            websocket.send_text(json.dumps({
                "event": "text_chunk",
                "text": "Hello!"
            }))

            # Expect audio_start event
            audio_start_evt = json.loads(websocket.receive_text())
            self.assertEqual(audio_start_evt.get("event"), "audio_start")
            self.assertEqual(audio_start_evt.get("sample_rate"), 8000)

            # Receive binary audio chunks and audio_end event
            received_bytes = 0
            while True:
                msg = websocket.receive()
                if "bytes" in msg and msg["bytes"]:
                    received_bytes += len(msg["bytes"])
                elif "text" in msg and msg["text"]:
                    data = json.loads(msg["text"])
                    if data.get("event") == "audio_end":
                        break

            self.assertGreater(received_bytes, 0)

            # 3. Test barge-in interrupt event
            websocket.send_text(json.dumps({"event": "interrupt"}))
            interrupt_resp = json.loads(websocket.receive_text())
            self.assertEqual(interrupt_resp.get("event"), "interrupted")


if __name__ == "__main__":
    unittest.main()
