"""
Unit tests for FastAPI REST Endpoints.
"""

import unittest
from fastapi.testclient import TestClient
from server.app import create_app


class TestServerRESTEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = TestClient(cls.app)

    def test_root_endpoint(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "online")

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ready")
        self.assertIn("device", data)
        self.assertIn("voices", data)

    def test_voices_endpoint(self):
        resp = self.client.get("/v1/voices")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(len(data["voices"]) >= 1)
        self.assertEqual(data["voices"][0]["voice_id"], "voice_clara")

    def test_metrics_endpoint(self):
        resp = self.client.get("/metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("total_requests", data)
        self.assertIn("avg_ttfa_ms", data)

    def test_synthesize_endpoint(self):
        payload = {
            "text": "[excited] Hello! Your order is ready.",
            "voice": "voice_clara",
            "sample_rate": 8000,
            "format": "pcm16",
            "ode_steps": 16
        }
        resp = self.client.post("/v1/tts/synthesize", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(len(resp.content), 0)
        self.assertIn("X-TTS-TTFA-MS", resp.headers)
        self.assertEqual(resp.headers.get("X-TTS-Format"), "pcm16")

    def test_openai_speech_endpoint(self):
        payload = {
            "model": "omnivoice",
            "input": "Thank you for using OpenAI speech compatible endpoint.",
            "voice": "voice_clara",
            "response_format": "wav"
        }
        resp = self.client.post("/v1/audio/speech", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.content.startswith(b"RIFF"))


if __name__ == "__main__":
    unittest.main()
