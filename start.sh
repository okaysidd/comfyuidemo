#!/usr/bin/env bash
set -euo pipefail

# extra safety: hide GPUs from torch
export CUDA_VISIBLE_DEVICES=-1
export FORCE_CPU=1

# Start ComfyUI (headless) on CPU
# Remove --listen/--port if you’re not exposing UI; keep --cpu.
python -u /app/ComfyUI/main.py --cpu --listen 0.0.0.0 --port 8188 &

# Start your FastAPI wrapper
# If you use uvicorn directly from main.py:app
uvicorn main:app --host 0.0.0.0 --port 8000
