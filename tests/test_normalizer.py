"""
Unit tests for TelephonyTextNormalizer and emotion tag extraction.
"""

import unittest
from runtime.normalizer import TelephonyTextNormalizer


class TestTelephonyTextNormalizer(unittest.TestCase):

    def setUp(self):
        self.norm = TelephonyTextNormalizer()

    def test_currency_normalization(self):
        text = "Your total is $45.50 and the discount is $5.00."
        res, emo, breath = self.norm.process(text)
        self.assertIn("forty-five dollars and fifty cents", res)
        self.assertIn("five dollars", res)

    def test_phone_number_normalization(self):
        text = "Call us at (800) 555-0199 or 888-123-4567."
        res, emo, breath = self.norm.process(text)
        self.assertIn("eight zero zero, five five five, zero one nine nine", res)

    def test_date_and_time_normalization(self):
        text = "Your appointment is on 08/24/2026 at 3:30 PM."
        res, emo, breath = self.norm.process(text)
        self.assertIn("August twenty-fourth", res)
        self.assertIn("twenty twenty-six", res)
        self.assertIn("three thirty PM", res)

    def test_emotion_tags(self):
        text = "[excited] Welcome to our team! [breath]"
        res, emo, breath = self.norm.process(text)
        self.assertEqual(emo, "excited")
        self.assertTrue(breath)
        self.assertEqual(res, "Welcome to our team!")

    def test_sympathetic_emotion(self):
        text = "[sympathetic] I am very sorry for the inconvenience."
        res, emo, breath = self.norm.process(text)
        self.assertEqual(emo, "sympathetic")
        self.assertNotIn("[sympathetic]", res)

    def test_abbreviations(self):
        text = "Meet with Dr. Watson at 123 Main St. with 15% off."
        res, emo, breath = self.norm.process(text)
        self.assertIn("Doctor Watson", res)
        self.assertIn("Main Street", res)
        self.assertIn("fifteen percent", res)


if __name__ == "__main__":
    unittest.main()
