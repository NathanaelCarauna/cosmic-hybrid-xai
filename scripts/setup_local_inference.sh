#!/usr/bin/env bash
set -euo pipefail

# Setup script for local SLM inference on AMD RX 6700 XT (gfx1031)
# Three tiers: CPU (no sudo) → Vulkan (sudo) → ROCm (sudo + heavy)
#
# Usage:
#   ./setup_local_inference.sh cpu      # CPU-only (default, no sudo)
#   ./setup_local_inference.sh vulkan   # Vulkan backend (requires sudo)
#   ./setup_local_inference.sh rocm     # ROCm/HIP backend (requires sudo)

BACKEND="${1:-cpu}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models/gguf"

echo "=== Hybrid-XAI Local Inference Setup ==="
echo "Backend: $BACKEND"
echo "Project: $PROJECT_DIR"
echo ""

mkdir -p "$MODELS_DIR"

install_cpu() {
    echo "[1/3] Installing llama-cpp-python (CPU)..."
    pip install --upgrade "llama-cpp-python>=0.3.0" 2>&1 | tail -5

    echo "[2/3] Installing huggingface-hub for model downloads..."
    pip install --upgrade huggingface-hub 2>&1 | tail -3

    echo "[3/3] Downloading Phi-4-mini Q4_K_M (~2.5GB)..."
    if [ ! -f "$MODELS_DIR/Phi-4-mini-instruct-Q4_K_M.gguf" ]; then
        huggingface-cli download \
            bartowski/Phi-4-mini-instruct-GGUF \
            Phi-4-mini-instruct-Q4_K_M.gguf \
            --local-dir "$MODELS_DIR" \
            --local-dir-use-symlinks False
    else
        echo "  Model already exists, skipping download."
    fi
}

install_vulkan() {
    echo "[1/4] Installing Vulkan SDK..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq libvulkan-dev vulkan-tools 2>&1 | tail -3

    echo "[2/4] Installing llama-cpp-python with Vulkan..."
    CMAKE_ARGS="-DGGML_VULKAN=on" pip install --upgrade --force-reinstall --no-cache-dir "llama-cpp-python>=0.3.0" 2>&1 | tail -5

    echo "[3/4] Installing huggingface-hub..."
    pip install --upgrade huggingface-hub 2>&1 | tail -3

    echo "[4/4] Downloading Phi-4-mini Q4_K_M..."
    if [ ! -f "$MODELS_DIR/Phi-4-mini-instruct-Q4_K_M.gguf" ]; then
        huggingface-cli download \
            bartowski/Phi-4-mini-instruct-GGUF \
            Phi-4-mini-instruct-Q4_K_M.gguf \
            --local-dir "$MODELS_DIR" \
            --local-dir-use-symlinks False
    fi
}

install_rocm() {
    echo "[1/5] Installing ROCm 6.x for gfx1031..."
    echo "  See: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
    sudo apt-get update -qq
    sudo apt-get install -y -qq "linux-headers-$(uname -r)" 2>&1 | tail -3

    wget -q https://repo.radeon.com/amdgpu-install/6.3.3/ubuntu/jammy/amdgpu-install_6.3.60303-1_all.deb -O /tmp/amdgpu-install.deb
    sudo apt-get install -y -qq /tmp/amdgpu-install.deb 2>&1 | tail -3
    sudo amdgpu-install --usecase=rocm --no-dkms -y 2>&1 | tail -10

    echo "[2/5] Adding user to render/video groups..."
    sudo usermod -aG render,video "$USER"

    echo "[3/5] Installing llama-cpp-python with ROCm/HIP..."
    CMAKE_ARGS="-DGGML_HIP=on -DAMDGPU_TARGETS=gfx1031" \
        pip install --upgrade --force-reinstall --no-cache-dir "llama-cpp-python>=0.3.0" 2>&1 | tail -5

    echo "[4/5] Installing huggingface-hub..."
    pip install --upgrade huggingface-hub 2>&1 | tail -3

    echo "[5/5] Downloading Phi-4-mini Q4_K_M..."
    if [ ! -f "$MODELS_DIR/Phi-4-mini-instruct-Q4_K_M.gguf" ]; then
        huggingface-cli download \
            bartowski/Phi-4-mini-instruct-GGUF \
            Phi-4-mini-instruct-Q4_K_M.gguf \
            --local-dir "$MODELS_DIR" \
            --local-dir-use-symlinks False
    fi

    echo ""
    echo "NOTE: Log out and log back in for group changes to take effect."
}

validate() {
    echo ""
    echo "=== Validating setup ==="
    python3 -c "
from llama_cpp import Llama
import os
model_path = os.path.join('$MODELS_DIR', 'Phi-4-mini-instruct-Q4_K_M.gguf')
if not os.path.exists(model_path):
    print('ERROR: Model not found at', model_path)
    exit(1)
print('Loading model (this may take 10-30s on first run)...')
llm = Llama(model_path=model_path, n_ctx=2048, n_threads=6, verbose=False)
resp = llm.create_chat_completion(
    messages=[{'role': 'user', 'content': 'Say exactly: INFERENCE OK'}],
    max_tokens=16, temperature=0
)
text = resp['choices'][0]['message']['content'].strip()
print(f'Response: {text}')
if 'OK' in text.upper():
    print('SUCCESS: Local inference is working.')
else:
    print('WARNING: Got unexpected response, but model loaded.')
"
}

case "$BACKEND" in
    cpu)     install_cpu ;;
    vulkan)  install_vulkan ;;
    rocm)    install_rocm ;;
    *)       echo "Unknown backend: $BACKEND. Use: cpu, vulkan, or rocm"; exit 1 ;;
esac

validate

echo ""
echo "=== Setup complete ==="
echo "To download additional models:"
echo "  huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q4_k_m.gguf --local-dir $MODELS_DIR"
echo "  huggingface-cli download bartowski/gemma-3-4b-it-GGUF gemma-3-4b-it-Q4_K_M.gguf --local-dir $MODELS_DIR"
