@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

echo ===============================================================================
echo 🚀 OmniVoice Production Real-Time TTS Server (Windows NVIDIA CUDA FP16)
echo ===============================================================================
echo.

:: 1. Check Python installation
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in system PATH.
    echo Please install Python 3.10 or 3.11 from python.org and check 'Add to PATH'.
    pause
    exit /b 1
)

:: 2. Activate Virtual Environment if exists
if exist .venv\Scripts\activate.bat (
    echo [INFO] Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) else if exist venv\Scripts\activate.bat (
    echo [INFO] Activating virtual environment (venv)...
    call venv\Scripts\activate.bat
) else (
    echo [INFO] No virtual environment found. Using system Python.
)

:: 3. Detect NVIDIA CUDA GPU Hardware & Precision
echo [INFO] Detecting NVIDIA GPU and PyTorch CUDA environment...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU Fallback\"}'); print(f'PyTorch Version: {torch.__version__}')"

if %errorlevel% neq 0 (
    echo.
    echo [WARNING] PyTorch or required dependencies not found.
    echo Installing requirements from requirements.txt...
    pip install -r requirements.txt
)

:: 4. Set Telephony & Server Environment Variables
if "%HOST%"=="" set HOST=0.0.0.0
if "%PORT%"=="" set PORT=8000
if "%ENABLE_AUDIOSOCKET%"=="" set ENABLE_AUDIOSOCKET=true
if "%AUDIOSOCKET_PORT%"=="" set AUDIOSOCKET_PORT=9092

echo.
echo [CONFIG] Host: %HOST%
echo [CONFIG] REST & WebSocket Port: %PORT%
echo [CONFIG] Asterisk AudioSocket Port: %AUDIOSOCKET_PORT%
echo.
echo [STARTING] Launching OmniVoice FastAPI / Uvicorn Server...
echo.

python -m uvicorn server.app:app --host %HOST% --port %PORT% --log-level info

pause
