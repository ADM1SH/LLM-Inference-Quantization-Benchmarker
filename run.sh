#!/bin/bash

# Exit on any failure
set -e

echo "===================================================="
echo "LLM Inference & Quantization Benchmarker Orchestrator"
echo "===================================================="

# 1. Navigate to the project root directory (absolute path safety)
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

# 2. Setup Python virtual environment
if [ ! -d "venv" ]; then
    echo "[Setup] Creating Python virtual environment..."
    python3 -m venv venv
else
    echo "[Setup] Virtual environment already exists."
fi

# 3. Activate venv
echo "[Setup] Activating virtual environment..."
source venv/bin/activate

# 4. Install dependencies
echo "[Setup] Installing / updating dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. Run the python benchmarking engine
echo "[Runner] Launching benchmarking suite..."
python benchmark.py

# 6. Inform user and launch the dashboard web server
echo ""
echo "===================================================="
echo "Benchmarking execution completed successfully!"
echo "Starting local dashboard server on http://localhost:8000"
echo "===================================================="
echo "Open your browser and visit: http://localhost:8000"
echo "Press Ctrl+C to stop the server."
echo "===================================================="
echo ""

python3 -m http.server 8000 --directory web
