"""Comprehensive test suite for Arabic OmniVoice TTS system."""

import os
import unittest
import numpy as np

from src.arabic_nlp.diacritizer import ArabicDiacritizer
from src.arabic_nlp.normalizer import ArabicNormalizer
from src.models.gesture_controller import ArabicGestureProcessor, EMOTION_BREATHING_MATRIX
from src.models.omnivoice_engine import ArabicOmniVoiceEngine
from src.models.voice_manager import BUILTIN_VOICES, VoiceManager


class TestArabicNormalizer(unittest.TestCase):
    """Unit tests for Arabic Normalizer."""

    def setUp(self):
        self.msa_norm = ArabicNormalizer(dialect="msa")
        self.egy_norm = ArabicNormalizer(dialect="egyptian")

    def test_numbers_msa(self):
        self.assertEqual(self.msa_norm.number_to_arabic(0), "صفر")
        self.assertEqual(self.msa_norm.number_to_arabic(5), "خمسة")
        self.assertEqual(self.msa_norm.number_to_arabic(15), "خمسة عشر")
        self.assertEqual(self.msa_norm.number_to_arabic(100), "مائة")
        self.assertEqual(self.msa_norm.number_to_arabic(123), "مائة وثلاثة وعشرون")
        self.assertEqual(self.msa_norm.number_to_arabic(1000), "ألف")
        self.assertEqual(self.msa_norm.number_to_arabic(2026), "ألفان وستة وعشرون")

    def test_numbers_egyptian(self):
        self.assertEqual(self.egy_norm.number_to_arabic(2), "اتنين")
        self.assertEqual(self.egy_norm.number_to_arabic(13), "تلاتاشر")
        self.assertEqual(self.egy_norm.number_to_arabic(100), "مية")
        self.assertEqual(self.egy_norm.number_to_arabic(200), "ميتين")

    def test_eastern_arabic_numerals(self):
        # ٠١٢٣٤٥٦٧٨٩ -> 0123456789
        text = "وصل ٥٠ طالباً"
        normalized = self.msa_norm.normalize(text)
        self.assertIn("خمسون", normalized)

    def test_currency(self):
        text = "السعر 150 $"
        normalized = self.msa_norm.normalize(text)
        self.assertIn("دولار", normalized)

    def test_time(self):
        text = "الموعد الساعة 3:30"
        normalized = self.msa_norm.normalize(text)
        self.assertIn("الثالثة والنصف", normalized)

    def test_percentage(self):
        text = "النسبة 25%"
        normalized = self.msa_norm.normalize(text)
        self.assertIn("بالمائة", normalized)

    def test_abbreviations(self):
        text = "حضر د. أحمد و م. خالد إلخ"
        normalized = self.msa_norm.normalize(text)
        self.assertIn("دكتور", normalized)
        self.assertIn("مهندس", normalized)
        self.assertIn("إلى آخره", normalized)


class TestArabicDiacritizer(unittest.TestCase):
    """Unit tests for Arabic Diacritizer."""

    def setUp(self):
        self.diacritizer = ArabicDiacritizer()

    def test_diacritic_density(self):
        bare = "مرحبا بكم"
        vocalized = "مَرْحَبًا بِكُمْ"
        self.assertEqual(self.diacritizer.diacritic_density(bare), 0.0)
        self.assertGreater(self.diacritizer.diacritic_density(vocalized), 0.5)

    def test_common_word_vocalization(self):
        text = "في هذا اليوم شكرا لكم"
        vocalized = self.diacritizer.diacritize(text)
        self.assertIn("فِي", vocalized)
        self.assertIn("هَٰذَا", vocalized)
        self.assertIn("شُكْرًا", vocalized)

    def test_preserve_existing_vocalization(self):
        text = "قَرَأَ الطَّالِبُ الكِتَابَ"
        vocalized = self.diacritizer.diacritize(text)
        # Should preserve exact vowels
        self.assertEqual(text, vocalized)


class TestVoiceManager(unittest.TestCase):
    """Unit tests for Voice Manager."""

    def setUp(self):
        self.vm = VoiceManager()

    def test_builtin_voices_count(self):
        voices = self.vm.list_voices()
        self.assertGreaterEqual(len(voices), 10)

    def test_dialect_filter(self):
        egy_voices = self.vm.list_voices(dialect="egyptian")
        msa_voices = self.vm.list_voices(dialect="msa")
        self.assertTrue(all(v.dialect == "egyptian" for v in egy_voices))
        self.assertTrue(all(v.dialect == "msa" for v in msa_voices))

    def test_get_voice_by_id(self):
        v = self.vm.get_voice("egy_mohamed")
        self.assertIsNotNone(v)
        self.assertEqual(v.gender, "male")
        self.assertEqual(v.engine, "voicetut")


class TestArabicOmniVoiceEngine(unittest.TestCase):
    """Unit tests for OmniVoice Engine."""

    def setUp(self):
        self.engine = ArabicOmniVoiceEngine()

    def test_device_detected(self):
        self.assertIn(self.engine.device, ["mps", "cuda", "cpu"])

    def test_text_preprocessing(self):
        raw = "عندي 250 جنيه والميعاد 3:30"
        processed = self.engine.preprocess_text(raw, dialect="egyptian")
        self.assertNotIn("250", processed)
        self.assertNotIn("3:30", processed)

    def test_synthesize_audio_output(self):
        text = "مرحباً بكم في نظام أومني فويس العربي"
        audio, sr = self.engine.synthesize(text, voice_id="msa_kareem", num_step=8)
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(sr, 24000)
        self.assertGreater(len(audio), 0)

    def test_stream_synthesize(self):
        text = "الجملة الأولى. الجملة الثانية."
        chunks = list(self.engine.stream_synthesize(text, num_step=6))
        self.assertGreaterEqual(len(chunks), 2)
        for chunk, sr in chunks:
            self.assertIsInstance(chunk, np.ndarray)
            self.assertEqual(sr, 24000)


class TestArabicGestureProcessor(unittest.TestCase):
    """Unit tests for Arabic Conversational Gestures, Breathing & Prosody."""

    def setUp(self):
        self.processor = ArabicGestureProcessor()

    def test_extract_emotion_tags(self):
        # Arabic tags
        clean, emotion, breath, pause = self.processor.extract_arabic_gestures("[متحمس] أهلاً بك يا صديقي!")
        self.assertEqual(emotion, "excited")
        self.assertNotIn("[متحمس]", clean)
        self.assertTrue(breath)

        clean, emotion, breath, pause = self.processor.extract_arabic_gestures("[متعاطف] نعتذر عن أي إزعاج")
        self.assertEqual(emotion, "sympathetic")
        self.assertNotIn("[متعاطف]", clean)

        clean, emotion, breath, pause = self.processor.extract_arabic_gestures("[مؤثر] شكراً جزيلاً لكم جميعاً")
        self.assertEqual(emotion, "emotional")
        self.assertNotIn("[مؤثر]", clean)

    def test_extract_english_and_ssml_tags(self):
        clean, emotion, breath, pause = self.processor.extract_arabic_gestures("[excited] Great news!")
        self.assertEqual(emotion, "excited")

        clean, emotion, breath, pause = self.processor.extract_arabic_gestures("Text with <break time=\"300ms\"/> pause")
        self.assertAlmostEqual(pause, 0.3)
        self.assertNotIn("<break", clean)

    def test_pitch_and_vocal_gesture_tags(self):
        # High pitch / loud
        clean, emotion, _, _ = self.processor.extract_arabic_gestures("[نبرة_عالية] مرحباً بكم جميعاً!")
        self.assertEqual(emotion, "high_pitch")
        self.assertNotIn("[نبرة_عالية]", clean)

        # Low pitch / deep
        clean, emotion, _, _ = self.processor.extract_arabic_gestures("[نبرة_منخفضة] في هذا اللقاء الهام")
        self.assertEqual(emotion, "low_pitch")
        self.assertNotIn("[نبرة_منخفضة]", clean)

        # Whisper
        clean, emotion, _, _ = self.processor.extract_arabic_gestures("[همس] هذا سر بيننا فقط")
        self.assertEqual(emotion, "whisper")
        self.assertNotIn("[همس]", clean)

        # Vocal gesture token mapping
        clean, _, _, _ = self.processor.extract_arabic_gestures("يا هلا والله [ضحك] نورتنا [تنهد]")
        self.assertIn("[laughter]", clean)
        self.assertIn("[sigh]", clean)
        self.assertNotIn("[ضحك]", clean)
        self.assertNotIn("[تنهد]", clean)

    def test_tailored_breath_generation(self):
        sr = 24000
        # Excited breath should be shorter (~95ms duration + 25ms gap)
        excited_breath = self.processor.generate_tailored_breath(emotion="excited")
        # Sympathetic breath should be longer (~200ms duration + 75ms gap)
        sympathetic_breath = self.processor.generate_tailored_breath(emotion="sympathetic")

        self.assertIsInstance(excited_breath, np.ndarray)
        self.assertIsInstance(sympathetic_breath, np.ndarray)
        self.assertGreater(len(sympathetic_breath), len(excited_breath))
        # Zero NaN or Inf
        self.assertFalse(np.isnan(excited_breath).any())
        self.assertFalse(np.isnan(sympathetic_breath).any())

    def test_clause_splitting(self):
        short_text = "مرحباً بكم اليوم"
        clauses = self.processor.split_conversational_clauses(short_text, max_words_per_clause=8)
        self.assertEqual(len(clauses), 1)

        long_text = "نحن سعداء جداً بوجودكم معنا في هذه الجلسة الافتتاحية ولكن لدينا بعض النقاط الهامة للمناقشة"
        clauses = self.processor.split_conversational_clauses(long_text, max_words_per_clause=6)
        self.assertGreater(len(clauses), 1)

    def test_apply_emotion_eq(self):
        test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 12000)).astype(np.float32)
        colored_excited = self.processor.apply_emotion_eq(test_audio, emotion="excited")
        colored_sympathetic = self.processor.apply_emotion_eq(test_audio, emotion="sympathetic")

        self.assertEqual(len(colored_excited), len(test_audio))
        self.assertEqual(len(colored_sympathetic), len(test_audio))
        self.assertLessEqual(np.max(np.abs(colored_excited)), 0.96)

    def test_inject_gestures(self):
        speech = np.zeros(24000, dtype=np.float32)
        spliced = self.processor.inject_gestures(speech, emotion="neutral", include_breath=True)
        # Spliced audio should contain breath segment + speech
        self.assertGreater(len(spliced), len(speech))

        no_breath = self.processor.inject_gestures(speech, emotion="neutral", include_breath=False)
        self.assertEqual(len(no_breath), len(speech))


if __name__ == "__main__":
    unittest.main()
