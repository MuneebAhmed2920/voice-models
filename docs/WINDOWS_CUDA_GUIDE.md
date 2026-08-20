# 🪟 Windows NVIDIA CUDA FP16 Setup & Performance Optimization Guide

This guide details how to install, configure, and maximize real-time speech generation throughput with **OmniVoice TTS Engine** on **Windows 10/11 / Windows Server 2022** using **NVIDIA CUDA FP16** and Tensor Core acceleration.

---

## 🖥️ 1. Hardware & System Requirements

- **GPU**: NVIDIA RTX 3000 series (RTX 3060, 3070, 3080, 3090), RTX 4000 series (RTX 4070, 4080, 4090), NVIDIA A10G, A40, L4, or T4.
- **VRAM**: Minimum 6 GB VRAM (8+ GB recommended for high concurrent call volume).
- **RAM**: 16 GB System RAM.
- **Operating System**: Windows 10 (64-bit), Windows 11 (64-bit), or Windows Server 2022.

---

## ⚡ 2. Step-by-Step Installation

### Step 1: Install NVIDIA Driver & CUDA Toolkit
1. Download and install the latest NVIDIA Driver (v535.xx or higher) from [nvidia.com/drivers](https://www.nvidia.com/drivers).
2. Download and install **NVIDIA CUDA Toolkit 12.1** (or 12.4) from [developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit).

### Step 2: Install Python 3.10 or 3.11
1. Download Python from [python.org/downloads](https://www.python.org/downloads).
2. **Important**: Check the box **"Add Python to PATH"** during installation.

### Step 3: Install PyTorch with CUDA 12.1 Support
Open PowerShell or Command Prompt in the repository directory and run:

```cmd
:: Create and activate virtual environment
python -m venv .venv
call .venv\Scripts\activate.bat

:: Install PyTorch with CUDA 12.1 Wheels
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

:: Install service dependencies
pip install -r requirements.txt
```

### Step 4: Verify CUDA Acceleration
Run the verification command:
```cmd
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0)); print('FP16 Tensor Cores Active:', torch.cuda.get_device_capability(0))"
```

---

## 🚀 3. Windows Performance Tuning for Ultra-Low Latency

### A. Enable Hardware-Accelerated GPU Scheduling (HAGS)
1. Open Windows **Settings ➔ System ➔ Display ➔ Graphics Settings**.
2. Toggle **Hardware-accelerated GPU scheduling** to **ON**.
3. Restart your computer.

### B. NVIDIA Control Panel 3D Settings
1. Open **NVIDIA Control Panel ➔ Manage 3D settings ➔ Global Settings / Program Settings**.
2. Select `python.exe` and set:
   - **Power management mode**: `Prefer maximum performance`
   - **CUDA - GPUs**: `All`
   - **Low Latency Mode**: `Ultra`
   - **Threaded optimization**: `On`

### C. Launching the Service
Simply double-click `start_server.bat` or run:
```cmd
start_server.bat
```

The server will automatically detect your NVIDIA GPU, pre-load the Clara voice models in `torch.float16`, run a warm-up inference pass, and start listening on ports:
- **Port 8000**: REST API (`/v1/tts/synthesize`, `/v1/audio/speech`) & WebSocket (`/ws/tts`)
- **Port 9092**: Native Asterisk AudioSocket TCP listener
