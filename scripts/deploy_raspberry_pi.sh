#!/usr/bin/env bash
# Turnkey deployment script for Raspberry Pi 3/4/5 (64-bit Raspberry Pi OS)
# Sets up Python Flask, Persistent ChromaDB, Ollama (nomic-embed-text + Llama 3.2 1B / Phi-3 Mini),
# and offline Arabic/English TTS (espeak-ng) for standalone edge operation.

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

echo "[1/5] Installing system packages (Python venv, espeak-ng offline TTS, Chromium kiosk)..."
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip espeak-ng curl

echo "[2/5] Creating Python virtual environment and installing Edge AI dependencies..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt chromadb

echo "[3/5] Installing local Ollama engine for 100% offline LLM + nomic-embed-text..."
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi

echo "[4/5] Pulling quantized 4-bit edge models..."
ollama pull nomic-embed-text
TOTAL_MEM_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
if [ "$TOTAL_MEM_MB" -ge 3500 ]; then
  echo "Detected >= 4GB RAM (Raspberry Pi 4/5): pulling qwen2.5:1.5b-instruct-q4_K_M (~986 MB, bilingual AR/EN)"
  ollama pull qwen2.5:1.5b-instruct-q4_K_M
else
  echo "Detected <= 2GB RAM (Raspberry Pi 3/4 2GB): pulling qwen2.5:0.5b-instruct-q4_K_M (~398 MB, ultra-light AR/EN)"
  ollama pull qwen2.5:0.5b-instruct-q4_K_M
fi

echo "[5/5] Creating systemd auto-start service..."
SERVICE_FILE="/etc/systemd/system/campus-edge-ai.service"
sudo tee "$SERVICE_FILE" >/dev/null <<EOF
[Unit]
Description=Edge AI Avatar Campus Assistant (Offline Flask + ChromaDB RAG)
After=network.target ollama.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
Environment=PORT=5050
ExecStart=$APP_DIR/.venv/bin/python $APP_DIR/app.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now campus-edge-ai.service

echo "Deployment complete. Run ./scripts/kiosk_start.sh to open 7-inch touch Kiosk Mode."
