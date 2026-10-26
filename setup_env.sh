#!/bin/bash
set -e

echo "========================================================"
echo "  🎙️ Arabic OmniVoice TTS - Automated Setup"
echo "========================================================"

# Check for Python 3.11 or 3.12
PYTHON_BIN=""
if command -v python3.11 &>/dev/null; then
    PYTHON_BIN="python3.11"
elif command -v python3.12 &>/dev/null; then
    PYTHON_BIN="python3.12"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
else
    echo "❌ Error: Python 3.11 or 3.12 is required."
    exit 1
fi

echo "✅ Using Python: $($PYTHON_BIN --version)"

# Create virtual environment if not present
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment in .venv..."
    if command -v uv &>/dev/null; then
        uv venv --python $PYTHON_BIN .venv
    else
        $PYTHON_BIN -m venv .venv
    fi
fi

source .venv/bin/activate

echo "📦 Installing core dependencies..."
if command -v uv &>/dev/null; then
    uv pip install torch torchaudio soundfile scipy numpy librosa transformers accelerate pyarabic fastapi uvicorn gradio pydantic
    uv pip install omnivoice voicetut-tts
else
    pip install --upgrade pip
    pip install torch torchaudio soundfile scipy numpy librosa transformers accelerate pyarabic fastapi uvicorn gradio pydantic
    pip install omnivoice voicetut-tts
fi

echo ""
echo "========================================================"
echo "  🎉 Setup Complete!"
echo "========================================================"
echo "  To activate the environment:"
echo "    source .venv/bin/activate"
echo ""
echo "  To start the Interactive Web Studio:"
echo "    python -m src.cli web"
echo ""
echo "  To launch the REST API server:"
echo "    python -m src.cli api"
echo ""
echo "  To test synthesis via CLI:"
echo "    python -m src.cli synthesize 'مرحباً بكم في نظام أومني فويس العربي' -o output.wav"
echo "========================================================"
