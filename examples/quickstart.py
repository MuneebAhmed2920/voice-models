#!/usr/bin/env python3
"""
examples/quickstart.py

Quickstart demonstration using Mahrukh Pakistani Urdu Voice Model.
"""

import os
import sys

# Ensure mahrukh_urdu_model can be imported directly
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_ROOT)

from runtime.engine import UrduVoiceModel

def main():
    print("Initializing Mahrukh Urdu Voice Model...")
    model = UrduVoiceModel()

    text = "اَلسَّلَامُ عَلَیْکُم! ہماری کَسٹمر سروس میں خوش آمدید، میں ماہ رخ بات کر رہی ہوں۔"
    output_wav = os.path.join(PACKAGE_ROOT, "outputs", "quickstart_demo.wav")

    print(f"Synthesizing: '{text}'...")
    audio, sr = model.synthesize(text, output_file=output_wav)
    print(f"Successfully generated: {output_wav} ({len(audio)/sr:.2f}s, {sr}Hz)")

if __name__ == "__main__":
    main()
