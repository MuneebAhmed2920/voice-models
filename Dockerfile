# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system audio dependencies and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    sox \
    libsox-fmt-all \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies
COPY requirements.txt pyproject.toml /app/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY . /app/

# Expose ports: 8000 for REST & WebSocket, 9092 for Asterisk AudioSocket
EXPOSE 8000 9092

ENV HOST=0.0.0.0
ENV PORT=8000
ENV ENABLE_AUDIOSOCKET=true
ENV AUDIOSOCKET_PORT=9092
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
