from setuptools import setup, find_packages

setup(
    name="marukh-english-model",
    version="1.0.0",
    description="Production-grade English female voice model of Mahrukh for AI voice agents and telephony",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Muneeb Ahmed",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "marukh_english_model": [
            "config/*.json",
            "prompt/*.pt",
            "prompt/*.wav",
            "samples/*.wav"
        ]
    },
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "soundfile>=0.12.1",
        "scipy>=1.10.0",
        "librosa>=0.10.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0"
    ],
    python_requires=">=3.9",
)
