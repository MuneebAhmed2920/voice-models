#!/usr/bin/env bash
set -e

echo "==============================================================================="
echo "🚀 OmniVoice Production Real-Time TTS Server (Linux / macOS)"
echo "==============================================================================="

# Virtual environment activation if present
if [ -f ".venv/bin/activate" ]; then
    echo "[INFO] Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    echo "[INFO] Activating virtual environment (venv)..."
    source venv/bin/activate
fi

# Detect Python interpreter
if command -v python3.11 &> /dev/null; then
    PY_BIN=python3.11
elif command -v python3 &> /dev/null; then
    PY_BIN=python3
else
    PY_BIN=python
fi

echo "[INFO] Using Python: $($PY_BIN --version)"
$PY_BIN -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU Fallback\"}')"

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export ENABLE_AUDIOSOCKET="${ENABLE_AUDIOSOCKET:-true}"
export AUDIOSOCKET_PORT="${AUDIOSOCKET_PORT:-9092}"

echo "[STARTING] Launching OmniVoice Server on http://${HOST}:${PORT} (AudioSocket: ${AUDIOSOCKET_PORT})..."
$PY_BIN -m uvicorn server.app:app --host "${HOST}" --port "${PORT}" --log-level info
