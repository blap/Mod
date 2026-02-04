#!/bin/bash

echo "[Inference-PIO] Starting..."

# Check if python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed."
    exit 1
fi

# Check for virtual environment
if [ -d ".venv" ]; then
    echo "[Inference-PIO] Activating virtual environment..."
    source .venv/bin/activate
fi

# Check dependencies
if ! python3 -c "import rich" &> /dev/null; then
    echo "[Inference-PIO] Installing dependencies..."
    pip install -r requirements.txt
fi

echo "[Inference-PIO] Launching CLI..."
python3 src/inference_pio/__main__.py "$@"
