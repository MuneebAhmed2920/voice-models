from setuptools import setup, find_packages

setup(
    name="omnivoice_tts",
    version="1.0.0",
    description="Clean, minimal, production-focused English Speech Synthesis Engine based on OmniVoice.",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "omnivoice>=0.2.1",
        "soundfile>=0.12.1",
        "scipy>=1.11.0",
        "numpy>=1.24.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
        "psutil>=5.9.0",
        "librosa>=0.10.0"
    ],
    entry_points={
        "console_scripts": [
            "omnivoice-server=omnivoice_tts.api.server:main",
            "omnivoice-cli=generate:main"
        ]
    }
)
