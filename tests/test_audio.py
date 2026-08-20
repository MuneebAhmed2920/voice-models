"""
Unit tests for AudioProcessor (resampling, formats, framing, breath).
"""

import unittest
import numpy as np
from runtime.audio_processor import AudioProcessor


class TestAudioProcessor(unittest.TestCase):

    def setUp(self):
        self.proc = AudioProcessor(default_sample_rate=8000)
        # 1.0 second 24kHz sine tone
        t = np.linspace(0, 1.0, 24000, dtype=np.float32)
        self.sine_24k = np.sin(2 * np.pi * 440.0 * t) * 0.4

    def test_resample(self):
        resampled_8k = self.proc.resample(self.sine_24k, 24000, 8000)
        self.assertEqual(len(resampled_8k), 8000)

        resampled_16k = self.proc.resample(self.sine_24k, 24000, 16000)
        self.assertEqual(len(resampled_16k), 16000)

    def test_float_to_pcm16_and_back(self):
        pcm = self.proc.float_to_pcm16(self.sine_24k)
        self.assertEqual(len(pcm), 24000 * 2)  # 2 bytes per sample

        recovered = self.proc.pcm16_to_float(pcm)
        self.assertEqual(len(recovered), 24000)
        np.testing.assert_allclose(recovered, self.sine_24k, atol=1e-4)

    def test_mulaw_encoding(self):
        mulaw_bytes, dur = self.proc.process_and_encode(self.sine_24k, 24000, 8000, "mulaw")
        self.assertEqual(len(mulaw_bytes), 8000)  # 1 byte per sample @ 8kHz
        self.assertAlmostEqual(dur, 1.0, places=2)

    def test_alaw_encoding(self):
        alaw_bytes, dur = self.proc.process_and_encode(self.sine_24k, 24000, 8000, "alaw")
        self.assertEqual(len(alaw_bytes), 8000)

    def test_wav_container(self):
        wav_bytes, dur = self.proc.process_and_encode(self.sine_24k, 24000, 8000, "wav")
        # Header (44 bytes) + 8000 samples * 2 bytes = 16044 bytes
        self.assertEqual(len(wav_bytes), 16044)
        self.assertTrue(wav_bytes.startswith(b"RIFF"))

    def test_telephony_20ms_framing(self):
        pcm8k, _ = self.proc.process_and_encode(self.sine_24k, 24000, 8000, "pcm16")
        frames = list(self.proc.frame_audio(pcm8k, sample_rate=8000, frame_duration_ms=20, bytes_per_sample=2))
        self.assertEqual(len(frames), 50)  # 50 frames per second
        for frame in frames:
            self.assertEqual(len(frame), 320)  # 160 samples * 2 bytes = 320 bytes


if __name__ == "__main__":
    unittest.main()
