from setuptools import setup, find_packages

setup(
    name="mahrukh-urdu-model",
    version="1.0.0",
    description="Standalone Female Pakistani Urdu Voice Model and Telephony TTS Engine (Mahrukh)",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="MuneebAhmed2920",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "mahrukh_urdu_model": [
            "config/*.json",
            "prompt/*.pt",
            "prompt/*.wav"
        ]
    },
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy",
        "scipy",
        "soundfile",
        "omnivoice",
        "fastapi",
        "uvicorn",
        "pydantic"
    ],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
