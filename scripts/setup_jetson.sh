#!/usr/bin/env bash
# Desk Buddy — Jetson AGX Orin setup script
# Run on the Orin: bash scripts/setup_jetson.sh
#
# This script:
#   1. Installs system packages (apt)
#   2. Creates a Python venv with system-site-packages
#   3. Installs PyTorch from NVIDIA wheels (auto-detects JetPack version)
#   4. Installs remaining Python dependencies
#   5. Downloads Piper TTS voice model
#
# Prerequisites: JetPack 5.x or 6.x installed on the Orin.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Detect JetPack version ──────────────────────────────────────
detect_jetpack() {
    if [ ! -f /etc/nv_tegra_release ]; then
        error "Not running on a Jetson (no /etc/nv_tegra_release). Aborting."
    fi

    local release_line
    release_line=$(head -1 /etc/nv_tegra_release)
    info "Tegra release: $release_line"

    if echo "$release_line" | grep -q "R35"; then
        JETPACK_MAJOR=5
        info "Detected JetPack 5.x"
    elif echo "$release_line" | grep -q "R36"; then
        JETPACK_MAJOR=6
        info "Detected JetPack 6.x"
    else
        warn "Unknown JetPack version. Assuming JP5.x."
        JETPACK_MAJOR=5
    fi

    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    info "Python version: $PYTHON_VERSION"
}

# ── Step 1: System packages ─────────────────────────────────────
install_system_packages() {
    info "Installing system packages..."
    sudo apt update && sudo apt upgrade -y

    sudo apt install -y \
        build-essential cmake pkg-config git wget curl \
        v4l-utils libv4l-dev \
        portaudio19-dev libasound2-dev pulseaudio alsa-utils \
        sox libsox-fmt-all \
        bluetooth bluez libbluetooth-dev \
        libopenblas-dev liblapack-dev \
        libjpeg-dev libpng-dev libtiff-dev \
        libavcodec-dev libavformat-dev libswscale-dev \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
        python3-dev python3-pip python3-venv \
        libhdf5-dev \
        espeak-ng

    info "System packages installed."
}

# ── Step 2: Python venv ─────────────────────────────────────────
setup_venv() {
    VENV_DIR="$HOME/desk-buddy/venv"

    if [ -d "$VENV_DIR" ]; then
        info "Virtual environment already exists at $VENV_DIR"
    else
        info "Creating virtual environment at $VENV_DIR ..."
        mkdir -p "$HOME/desk-buddy"
        python3 -m venv "$VENV_DIR" --system-site-packages
    fi

    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip setuptools wheel
    info "Virtual environment ready."
}

# ── Step 3: PyTorch ──────────────────────────────────────────────
install_pytorch() {
    # Check if PyTorch with CUDA is already installed
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        local ver
        ver=$(python3 -c "import torch; print(torch.__version__)")
        info "PyTorch $ver with CUDA already installed. Skipping."
        return
    fi

    info "Installing PyTorch from NVIDIA Jetson wheels..."

    if [ "$JETPACK_MAJOR" -eq 5 ]; then
        # JP5.x — Python 3.8, CUDA 11.4
        pip install --no-cache-dir \
            https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl || {
            warn "PyTorch wheel download failed. Check https://forums.developer.nvidia.com/t/pytorch-for-jetson/ for updated URLs."
            warn "You may need to find the correct wheel for your exact JetPack version."
            return 1
        }
    elif [ "$JETPACK_MAJOR" -eq 6 ]; then
        # JP6.x — Python 3.10, CUDA 12.2
        pip install --no-cache-dir \
            https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0a0+ebedce2.nv24.05-cp310-cp310-linux_aarch64.whl || {
            warn "PyTorch wheel download failed. Check https://forums.developer.nvidia.com/t/pytorch-for-jetson/ for updated URLs."
            return 1
        }
    fi

    # Verify
    if python3 -c "import torch; assert torch.cuda.is_available()"; then
        info "PyTorch installed with CUDA support."
    else
        warn "PyTorch installed but CUDA not available. Check your JetPack installation."
    fi
}

# ── Step 4: OpenCV check ────────────────────────────────────────
check_opencv() {
    if python3 -c "import cv2" 2>/dev/null; then
        local ver
        ver=$(python3 -c "import cv2; print(cv2.__version__)")
        info "OpenCV $ver available (from JetPack system packages)."
    else
        warn "OpenCV not found. Installing system package..."
        sudo apt install -y python3-opencv
    fi
}

# ── Step 5: Python dependencies ──────────────────────────────────
install_python_deps() {
    local REPO_DIR="$HOME/desk-buddy/cs347w26-team7-desk-buddy"

    if [ ! -f "$REPO_DIR/requirements-jetson.txt" ]; then
        error "requirements-jetson.txt not found at $REPO_DIR. Clone the repo first."
    fi

    info "Installing Python dependencies from requirements-jetson.txt..."
    pip install -r "$REPO_DIR/requirements-jetson.txt"

    info "Building llama-cpp-python with CUDA support (this may take 5-10 minutes)..."
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install 'llama-cpp-python>=0.2.0' --no-cache-dir || {
        warn "llama-cpp-python build failed. The LLM agent will run in simulation mode."
        warn "Ensure nvcc is on your PATH and retry manually."
    }

    info "Python dependencies installed."
}

# ── Step 6: Install Piper TTS binary + voice ────────────────────
install_piper_binary() {
    if command -v piper &>/dev/null; then
        info "Piper binary already installed: $(piper --version 2>&1 || true)"
        return
    fi

    info "Downloading Piper TTS aarch64 binary..."
    local PIPER_VERSION="2023.11.14-2"
    local PIPER_URL="https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_aarch64.tar.gz"
    local TMP_DIR
    TMP_DIR=$(mktemp -d)

    wget -q -O "$TMP_DIR/piper.tar.gz" "$PIPER_URL" || {
        warn "Piper binary download failed. espeak-ng will be used as TTS fallback."
        rm -rf "$TMP_DIR"
        return 1
    }

    tar -xzf "$TMP_DIR/piper.tar.gz" -C "$TMP_DIR"
    sudo install -m 755 "$TMP_DIR/piper/piper" /usr/local/bin/piper
    # Install shared libraries alongside the binary
    if [ -d "$TMP_DIR/piper/lib" ]; then
        sudo cp "$TMP_DIR/piper/lib"/* /usr/local/lib/
        sudo ldconfig
    fi
    rm -rf "$TMP_DIR"
    info "Piper binary installed to /usr/local/bin/piper"
}

download_piper_voice() {
    local PIPER_DIR="$HOME/.local/share/piper"
    local VOICE_ONNX="$PIPER_DIR/en_US-lessac-medium.onnx"

    if [ -f "$VOICE_ONNX" ]; then
        info "Piper TTS voice already downloaded."
        return
    fi

    info "Downloading Piper TTS voice model..."
    mkdir -p "$PIPER_DIR"
    wget -q -O "$VOICE_ONNX" \
        https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
    wget -q -O "${VOICE_ONNX}.json" \
        https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
    info "Piper TTS voice downloaded."
}

# ── Step 7: Create data directories ─────────────────────────────
create_dirs() {
    local REPO_DIR="$HOME/desk-buddy/cs347w26-team7-desk-buddy"
    mkdir -p "$REPO_DIR/data/state_logs" \
             "$REPO_DIR/data/trained_models" \
             "$REPO_DIR/models"
    info "Data directories created."
}

# ── Main ─────────────────────────────────────────────────────────
main() {
    info "=== Desk Buddy — Jetson AGX Orin Setup ==="
    echo

    detect_jetpack
    install_system_packages
    setup_venv
    install_pytorch
    check_opencv
    create_dirs
    install_python_deps
    install_piper_binary
    download_piper_voice

    # ── LD_PRELOAD fix for static TLS on aarch64 ──
    # Libraries using initial-exec TLS model (e.g. libgomp from OpenMP) cause
    # "cannot allocate memory in static TLS block" when loaded via dlopen().
    # Preloading ensures they get a TLS slot at process start.
    local LD_PRELOAD_LINE='export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1'
    if ! grep -qF "$LD_PRELOAD_LINE" ~/.bashrc 2>/dev/null; then
        info "Adding LD_PRELOAD fix for static TLS to ~/.bashrc ..."
        echo "" >> ~/.bashrc
        echo "# Desk Buddy: fix static TLS allocation error on aarch64" >> ~/.bashrc
        echo "$LD_PRELOAD_LINE" >> ~/.bashrc
    else
        info "LD_PRELOAD fix already in ~/.bashrc."
    fi

    echo
    info "=== Setup complete! ==="
    info ""
    info "Next steps:"
    info "  0. Apply LD_PRELOAD fix (already added to ~/.bashrc, reload with: source ~/.bashrc)"
    info "     Or for this shell: export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1"
    info "  1. Transfer models from dev machine (see docs/JETSON_DEPLOY.md Step 8)"
    info "  2. Copy Jetson config: cp config/pipeline.jetson.yaml config/pipeline.yaml"
    info "  3. Run verification:  python3 scripts/verify_jetson.py"
    info "  4. Start Desk Buddy:  python3 -m src.main --no-desk"
}

main "$@"
