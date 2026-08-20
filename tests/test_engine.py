"""
Unit and integration tests for OmniVoiceEngine and VoiceManager.
"""

import unittest
from runtime.voice_manager import VoiceManager
from runtime.engine import get_engine, OmniVoiceEngine


class TestEngineAndVoices(unittest.TestCase):

    def setUp(self):
        self.vm = VoiceManager()
        self.engine = get_engine(default_ode_steps=16, warmup=False)

    def test_voice_manager_prompts(self):
        voices = self.vm.list_voices()
        self.assertTrue(len(voices) >= 1)
        clara = voices[0]
        self.assertEqual(clara["voice_id"], "voice_clara")
        self.assertIn("neutral", clara["available_emotions"])
        self.assertIn("excited", clara["available_emotions"])

        prompt_neutral = self.vm.get_prompt("clara", "neutral")
        self.assertIsNotNone(prompt_neutral)
        self.assertIsNotNone(prompt_neutral.ref_text)

        prompt_excited = self.vm.get_prompt("clara", "excited")
        self.assertIsNotNone(prompt_excited)

    def test_engine_status(self):
        status = self.engine.get_status()
        self.assertEqual(status["status"], "ready")
        self.assertIn("device", status)
        self.assertIn("dtype", status)

    def test_engine_synthesis(self):
        audio_bytes, metrics = self.engine.synthesize(
            text="Testing engine synthesis.",
            voice="voice_clara",
            sample_rate=8000,
            format_type="pcm16",
            ode_steps=16
        )
        self.assertGreater(len(audio_bytes), 0)
        self.assertEqual(metrics["sample_rate"], 8000)
        self.assertEqual(metrics["format"], "pcm16")
        self.assertGreater(metrics["audio_duration_sec"], 0.0)


if __name__ == "__main__":
    unittest.main()
