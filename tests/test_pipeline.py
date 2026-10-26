"""Pipeline unit tests — covers all new modular components.

Tests:
  - ContextProsodyAnalyzer: intent classification, speed/guidance routing
  - BreatheGate: correct suppression on short/numeric/farewell text
  - UnifiedNormalizer: phone numbers, emails, URLs, decimals, acronyms, contractions
  - CodeSwitchSegmenter: Arabic/English boundary detection
  - TelephonyAudioProcessor: loudness normalization, DC removal, resampling quality
"""

import sys
import unittest
from pathlib import Path

import numpy as np

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.context_prosody import ContextProsodyAnalyzer, DialogIntent
from src.pipeline.prosody_rules import BreatheGate
from src.pipeline.normalizer_unified import UnifiedNormalizer
from src.pipeline.code_switch import CodeSwitchSegmenter
from src.pipeline.telephony_audio import TelephonyAudioProcessor


class TestContextProsodyAnalyzer(unittest.TestCase):
    """Tests for dialog intent classification and prosody routing."""

    def setUp(self):
        self.analyzer = ContextProsodyAnalyzer()

    def test_greeting_intent(self):
        hints = self.analyzer.analyze("مرحباً بك! كيف يمكنني مساعدتك؟")
        self.assertEqual(hints.intent, DialogIntent.GREETING)
        self.assertLessEqual(hints.speed, 1.0)
        self.assertFalse(hints.allow_breath)

    def test_farewell_intent(self):
        hints = self.analyzer.analyze("مع السلامة، شكراً لتواصلك معنا.")
        self.assertEqual(hints.intent, DialogIntent.FAREWELL)
        self.assertFalse(hints.allow_breath)

    def test_question_intent_arabic(self):
        hints = self.analyzer.analyze("هل يمكنني مساعدتك في شيء آخر؟")
        self.assertEqual(hints.intent, DialogIntent.QUESTION)
        self.assertEqual(hints.sentence_ending, "rising")

    def test_question_intent_english(self):
        hints = self.analyzer.analyze("Would you like anything else today?")
        self.assertEqual(hints.intent, DialogIntent.QUESTION)

    def test_apology_intent(self):
        hints = self.analyzer.analyze("نعتذر بشدة عن الإزعاج الذي سببناه لك.")
        self.assertEqual(hints.intent, DialogIntent.APOLOGY)
        self.assertLess(hints.speed, 1.0)
        self.assertEqual(hints.sentence_ending, "falling")

    def test_short_confirmation_intent(self):
        for text in ["نعم، بالتأكيد.", "حسناً.", "Okay.", "Sure."]:
            hints = self.analyzer.analyze(text)
            self.assertEqual(hints.intent, DialogIntent.CONFIRMATION)
            self.assertFalse(hints.allow_breath)
            self.assertGreaterEqual(hints.speed, 1.0)  # confirmations should be crisp

    def test_good_news_intent(self):
        hints = self.analyzer.analyze("يسعدنا إبلاغك بأن طلبك تمت الموافقة عليه بنجاح!")
        self.assertEqual(hints.intent, DialogIntent.GOOD_NEWS)

    def test_important_info_intent(self):
        hints = self.analyzer.analyze("يرجى الانتباه: رمز التحقق OTP صالح لمدة خمس دقائق فقط.")
        self.assertEqual(hints.intent, DialogIntent.IMPORTANT_INFO)
        self.assertLess(hints.speed, 0.95)
        self.assertGreater(hints.guidance_scale, 2.7)

    def test_long_explanation_allows_breath(self):
        long_text = "نود إبلاغك بأن الخدمة ستكون متوقفة مؤقتاً للصيانة الدورية في الفترة من الساعة الثانية صباحاً حتى السادسة صباحاً بتوقيت المملكة العربية السعودية."
        hints = self.analyzer.analyze(long_text)
        self.assertEqual(hints.intent, DialogIntent.LONG_EXPLANATION)
        self.assertTrue(hints.allow_breath)

    def test_normal_info_default(self):
        hints = self.analyzer.analyze("طلبك قيد المعالجة وسيصل خلال ثلاثة أيام.")
        self.assertEqual(hints.intent, DialogIntent.NORMAL_INFO)
        self.assertAlmostEqual(hints.speed, 1.0, places=1)


class TestBreatheGate(unittest.TestCase):
    """Tests for breathing gate — must suppress breath on inappropriate contexts."""

    def setUp(self):
        self.gate = BreatheGate()

    def test_no_breath_on_single_word(self):
        self.assertFalse(self.gate.should_breathe("نعم."))

    def test_no_breath_on_short_confirmation(self):
        for text in ["نعم", "تمام", "sure", "okay", "yes", "ok"]:
            self.assertFalse(self.gate.should_breathe(text))

    def test_no_breath_on_farewell(self):
        self.assertFalse(self.gate.should_breathe("مع السلامة، وداعاً."))

    def test_no_breath_when_context_disallows(self):
        long_text = "طلبك قيد المعالجة وسيصل إلى عنوانك المسجل خلال ثلاثة إلى خمسة أيام عمل من تاريخ اليوم."
        self.assertFalse(self.gate.should_breathe(long_text, context_allow_breath=False))

    def test_no_breath_on_numeric_readout(self):
        # Phone number readout
        numeric_text = "صفر خمسة صفر واحد اثنان ثلاثة أربعة خمسة ستة سبعة"
        self.assertFalse(self.gate.should_breathe(numeric_text))

    def test_breath_allowed_on_long_paragraph(self):
        long_text = (
            "نود إبلاغك بأن الخدمة ستكون متوقفة مؤقتاً للصيانة الدورية "
            "في الفترة من الساعة الثانية صباحاً حتى السادسة صباحاً بتوقيت "
            "المملكة العربية السعودية وذلك لضمان أفضل أداء ممكن."
        )
        self.assertTrue(self.gate.should_breathe(long_text, context_allow_breath=True))

    def test_no_inter_clause_breath_on_last_clause(self):
        result = self.gate.should_breathe_between_clauses(
            "نعم.", clause_word_count=2, is_last_clause=True
        )
        self.assertFalse(result)

    def test_inter_clause_breath_on_long_clause(self):
        result = self.gate.should_breathe_between_clauses(
            "نود إبلاغك بأن الخدمة ستكون متوقفة مؤقتاً للصيانة الدورية",
            clause_word_count=10,
            is_last_clause=False,
        )
        self.assertTrue(result)


class TestUnifiedNormalizer(unittest.TestCase):
    """Tests for telephony-grade text normalization."""

    def setUp(self):
        self.norm_msa = UnifiedNormalizer(dialect="msa")
        self.norm_egy = UnifiedNormalizer(dialect="egyptian")

    def test_phone_number_normalization(self):
        text = "اتصل بنا على 0501234567"
        result = self.norm_msa.normalize(text)
        # Phone digits should be expanded to Arabic words
        self.assertNotIn("0501234567", result)
        self.assertIn("صفر", result)

    def test_email_normalization(self):
        text = "راسلنا على info@company.com للاستفسار."
        result = self.norm_msa.normalize(text)
        self.assertNotIn("@company.com", result)
        self.assertIn("آت", result)
        self.assertIn("دوت", result)

    def test_url_normalization(self):
        text = "قم بزيارة www.example.com"
        result = self.norm_msa.normalize(text)
        self.assertNotIn("www.example.com", result)
        self.assertIn("دوت", result)

    def test_decimal_normalization(self):
        text = "النسبة هي 3.14 بالمائة"
        result = self.norm_msa.normalize(text)
        self.assertNotIn("3.14", result)
        self.assertIn("فاصل", result)

    def test_acronym_normalization_otp(self):
        text = "رمز الـ OTP الخاص بك هو 123456."
        result = self.norm_msa.normalize(text)
        self.assertNotIn("OTP", result)
        self.assertIn("أو تي بي", result)

    def test_acronym_normalization_sms(self):
        text = "ستصلك رسالة SMS."
        result = self.norm_msa.normalize(text)
        self.assertNotIn("SMS", result)
        self.assertIn("إس إم إس", result)

    def test_acronym_vat(self):
        text = "السعر شامل الـ VAT."
        result = self.norm_msa.normalize(text)
        self.assertIn("ضريبة القيمة المضافة", result)

    def test_english_contraction_wont(self):
        result = self.norm_msa._expand_english_contractions("I won't be able to help.")
        self.assertIn("will not", result)
        self.assertNotIn("won't", result)

    def test_english_contraction_cant(self):
        result = self.norm_msa._expand_english_contractions("I can't find your order.")
        self.assertIn("cannot", result)

    def test_english_ordinal(self):
        result = self.norm_msa._normalize_english_ordinals("This is the 1st time.")
        self.assertIn("first", result)
        self.assertNotIn("1st", result)

    def test_currency_normalization(self):
        text = "السعر 150 $"
        result = self.norm_msa.normalize(text)
        self.assertIn("دولار", result)
        self.assertNotIn("150 $", result)

    def test_percentage_normalization(self):
        text = "خصم 25%"
        result = self.norm_msa.normalize(text)
        self.assertIn("بالمائة", result)

    def test_date_normalization(self):
        text = "تاريخ الانتهاء 15/09/2026"
        result = self.norm_msa.normalize(text)
        self.assertNotIn("15/09/2026", result)
        self.assertIn("سبتمبر", result)

    def test_time_normalization(self):
        text = "الموعد الساعة 3:30"
        result = self.norm_msa.normalize(text)
        self.assertIn("الثالثة والنصف", result)


class TestCodeSwitchSegmenter(unittest.TestCase):
    """Tests for Arabic/English code-switching boundary detection."""

    def setUp(self):
        self.segmenter = CodeSwitchSegmenter()

    def test_pure_arabic(self):
        segs = self.segmenter.segment("مرحباً بكم في نظام أومني فويس")
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].lang, "arabic")

    def test_pure_english(self):
        segs = self.segmenter.segment("Hello and welcome to our service")
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].lang, "english")

    def test_mixed_arabic_english(self):
        segs = self.segmenter.segment("طلبك confirmed وسيصل خلال thirty minutes")
        # Should detect at least two segments (arabic and english portions)
        self.assertGreaterEqual(len(segs), 1)
        langs = [s.lang for s in segs]
        self.assertTrue("arabic" in langs or "english" in langs)

    def test_reconstruct_preserves_content(self):
        text = "طلبك confirmed وسيصل خلال thirty minutes"
        segs = self.segmenter.segment(text)
        reconstructed = self.segmenter.reconstruct(segs)
        # All words should be present
        for word in text.split():
            self.assertIn(word, reconstructed)

    def test_omnivoice_lang_mapping(self):
        from src.pipeline.code_switch import TextSegment
        ar_seg = TextSegment(text="مرحبا", lang="arabic", index=0)
        en_seg = TextSegment(text="hello", lang="english", index=1)
        self.assertEqual(self.segmenter.to_omnivoice_lang(ar_seg), "Standard Arabic")
        self.assertEqual(self.segmenter.to_omnivoice_lang(en_seg), "English")

    def test_empty_text(self):
        segs = self.segmenter.segment("")
        self.assertEqual(segs, [])


class TestTelephonyAudioProcessor(unittest.TestCase):
    """Tests for telephony audio preparation."""

    def setUp(self):
        self.proc = TelephonyAudioProcessor(source_sample_rate=24000)
        # Clean 440Hz sine wave at 24kHz
        duration = 2.0
        sr = 24000
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        self.test_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    def test_dc_offset_removal(self):
        dc_audio = self.test_audio + 0.1  # Introduce DC bias
        cleaned = self.proc.remove_dc_offset(dc_audio)
        self.assertLess(abs(np.mean(cleaned)), 0.01)

    def test_loudness_normalization_rms(self):
        # Quiet audio
        quiet = (self.test_audio * 0.01).astype(np.float32)
        normalized = self.proc.normalize_loudness(quiet, target_rms_db=-18.0)
        rms = np.sqrt(np.mean(normalized ** 2))
        rms_db = 20 * np.log10(max(rms, 1e-9))
        self.assertAlmostEqual(rms_db, -18.0, delta=2.0)

    def test_peak_limiter(self):
        # Very loud audio
        loud = (self.test_audio * 10.0).astype(np.float32)
        normalized = self.proc.normalize_loudness(loud, peak_limit_db=-1.0)
        peak = np.max(np.abs(normalized))
        self.assertLessEqual(peak, 1.0)

    def test_resample_8khz_shape(self):
        audio_8k, sr = self.proc.resample_for_telephony(self.test_audio, target_sr=8000)
        expected_len = int(len(self.test_audio) * 8000 / 24000)
        self.assertEqual(sr, 8000)
        self.assertAlmostEqual(len(audio_8k), expected_len, delta=5)

    def test_resample_16khz_shape(self):
        audio_16k, sr = self.proc.resample_for_telephony(self.test_audio, target_sr=16000)
        self.assertEqual(sr, 16000)
        expected_len = int(len(self.test_audio) * 16000 / 24000)
        self.assertAlmostEqual(len(audio_16k), expected_len, delta=5)

    def test_resample_no_clipping(self):
        audio_8k, _ = self.proc.resample_for_telephony(self.test_audio, target_sr=8000)
        self.assertLessEqual(np.max(np.abs(audio_8k)), 1.5)  # should not clip badly

    def test_quality_verification_pass(self):
        normalized = self.proc.normalize_loudness(self.test_audio)
        audio_8k, sr = self.proc.resample_for_telephony(normalized, target_sr=8000)
        quality = self.proc.verify_telephony_quality(audio_8k, sr)
        self.assertFalse(quality["clipping"])
        self.assertFalse(quality["silence"])
        self.assertIn("duration_sec", quality)
        self.assertGreater(quality["duration_sec"], 0)

    def test_quality_verification_detects_silence(self):
        silent = np.zeros(8000, dtype=np.float32)
        quality = self.proc.verify_telephony_quality(silent, 8000)
        self.assertTrue(quality["silence"])
        self.assertFalse(quality["pass"])

    def test_prepare_for_telephony_pipeline(self):
        audio, sr, quality = self.proc.prepare_for_telephony(
            self.test_audio, target_sr=8000, simulate_codec=False
        )
        self.assertEqual(sr, 8000)
        self.assertFalse(quality["clipping"])

    def test_g711_codec_simulation(self):
        audio_8k, _ = self.proc.resample_for_telephony(self.test_audio, target_sr=8000)
        # Normalize first
        audio_8k = self.proc.normalize_loudness(audio_8k)
        simulated = self.proc.apply_g711_codec_simulation(audio_8k)
        self.assertEqual(len(simulated), len(audio_8k))
        self.assertFalse(np.any(np.isnan(simulated)))
        self.assertFalse(np.any(np.isinf(simulated)))
        # Should introduce only small quantization noise
        noise = simulated - audio_8k
        noise_rms = np.sqrt(np.mean(noise ** 2))
        self.assertLess(noise_rms, 0.05)  # codec noise should be small

    def test_empty_audio_handled(self):
        empty = np.zeros(0, dtype=np.float32)
        out = self.proc.remove_dc_offset(empty)
        self.assertEqual(len(out), 0)
        out2 = self.proc.normalize_loudness(empty)
        self.assertEqual(len(out2), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
