# Deploying Desk Buddy to NVIDIA Jetson AGX Orin 64GB

This guide covers deploying the full Desk Buddy system (perception pipeline + voice I/O + LLM agent + desk control) to an AGX Orin 64GB running JetPack 5.x.

**Key constraints:**
- PyTorch requires NVIDIA's Jetson-specific wheels (not PyPI)
- MediaPipe has limited ARM64 support
- OpenCV should come from JetPack (not pip)

---

## Step 0: Verify JetPack Version & Environment

SSH into the Orin and confirm your setup:

```bash
# JetPack version (R35 = JP5.x, R36 = JP6.x)
head -1 /etc/nv_tegra_release

# Specific JetPack version
dpkg -l | grep nvidia-jetpack

# CUDA version (JP5.x → CUDA 11.4)
nvcc --version

# Python version (JP5.x/Ubuntu 20.04 → Python 3.8)
python3 --version

# Disk space (need ~15GB)
df -h /

# GPU info
tegrastats   # Ctrl+C after a few lines

# Webcam
ls /dev/video*
v4l2-ctl --list-devices

# Audio devices
arecord -l   # microphone
aplay -l     # speakers
```

**Expected for JetPack 5.x:** Ubuntu 20.04, Python 3.8, CUDA 11.4, cuDNN 8.6+, TensorRT 8.5+.

If you see R36 / Ubuntu 22.04 / Python 3.10, you're on JetPack 6.x — the steps below still work but PyTorch wheel URLs differ (noted inline).

---

## Step 1: Install System Packages

Run `scripts/setup_jetson.sh` to automate this, or manually:

```bash
sudo apt update && sudo apt upgrade -y

# Build tools
sudo apt install -y build-essential cmake pkg-config git wget curl

# Video/camera
sudo apt install -y v4l-utils libv4l-dev

# Audio (for PyAudio, microphone, speaker)
sudo apt install -y portaudio19-dev libasound2-dev pulseaudio alsa-utils
sudo apt install -y sox libsox-fmt-all

# Bluetooth (for BLE desk control)
sudo apt install -y bluetooth bluez libbluetooth-dev

# Image/media libraries (OpenCV deps)
sudo apt install -y libopenblas-dev liblapack-dev
sudo apt install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

# Python dev headers
sudo apt install -y python3-dev python3-pip python3-venv

# HDF5
sudo apt install -y libhdf5-dev
```

Verify audio:
```bash
arecord -d 3 -f cd /tmp/test.wav && aplay /tmp/test.wav
```

---

## Step 2: Python Virtual Environment

Use `--system-site-packages` so the venv sees JetPack's OpenCV and system PyTorch:

```bash
mkdir -p ~/desk-buddy && cd ~/desk-buddy
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

---

## Step 3: Install PyTorch (NVIDIA Jetson Wheels)

**Do NOT `pip install torch` from PyPI** — those are x86-only.

```bash
# Check if PyTorch already came with JetPack:
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If it prints `True`, skip ahead. Otherwise:

**For JetPack 5.x (CUDA 11.4, Python 3.8):**
```bash
pip install --no-cache-dir \
  https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
```

**For JetPack 6.x (CUDA 12.2, Python 3.10):**
```bash
pip install --no-cache-dir \
  https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0a0+ebedce2.nv24.05-cp310-cp310-linux_aarch64.whl
```

> **Note:** Exact wheel URLs change. Check https://forums.developer.nvidia.com/t/pytorch-for-jetson/ for your JetPack + Python version.

Install torchvision:
```bash
# Find matching wheel on the same NVIDIA page, OR build from source:
pip install --no-cache-dir \
  https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torchvision-0.16.1+fdea156-cp38-cp38-linux_aarch64.whl

# If no wheel available, build from source:
# git clone --branch v0.16.0 https://github.com/pytorch/vision.git /tmp/torchvision
# cd /tmp/torchvision && pip install -e . && cd ~/desk-buddy
```

**Verify — MUST print True:**
```bash
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

---

## Step 4: Install MediaPipe (ARM64)

MediaPipe's official PyPI package is x86-only. Try in order:

**Option A: pip install (may work with newer versions)**
```bash
pip install mediapipe
```

**Option B: Build from source**
```bash
sudo apt install -y openjdk-11-jdk
wget -O /usr/local/bin/bazel \
  https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-arm64
sudo chmod +x /usr/local/bin/bazel

git clone https://github.com/google/mediapipe.git /tmp/mediapipe
cd /tmp/mediapipe
python3 setup.py gen_protos && python3 setup.py bdist_wheel
pip install dist/mediapipe-*.whl
cd ~/desk-buddy
```

**Option C:** Search for community `mediapipe aarch64` wheels matching your Python version.

**Verify:**
```bash
python3 -c "import mediapipe as mp; print(f'MediaPipe {mp.__version__} OK')"
```

---

## Step 5: Verify OpenCV (JetPack Build)

JetPack ships OpenCV with CUDA + GStreamer. **Do NOT pip install opencv-python.**

```bash
python3 -c "
import cv2
print(f'OpenCV: {cv2.__version__}')
print(f'CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}')
"
```

If not available: `sudo apt install -y python3-opencv`

---

## Step 6: Clone & Setup the Repo

```bash
cd ~/desk-buddy
git clone <repo-url> cs347w26-team7-desk-buddy
cd cs347w26-team7-desk-buddy
git checkout main

mkdir -p data/state_logs data/trained_models models
```

---

## Step 7: Install Python Dependencies

Use the Jetson-specific requirements file (excludes opencv-python, torch, torchvision):

```bash
cd ~/desk-buddy/cs347w26-team7-desk-buddy
source ~/desk-buddy/venv/bin/activate

pip install -r requirements-jetson.txt

# LLM with CUDA support (takes 5-10 min to build)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install 'llama-cpp-python>=0.2.0' --no-cache-dir
```

---

## Step 8: Transfer & Download Models

### Auto-downloaded on first run:
| Model | Size | Trigger |
|-------|------|---------|
| YOLOv8s | 22 MB | `PersonDetector.__init__` |
| MediaPipe Pose Landmarker | 9 MB | `PoseEstimator._ensure_model()` |
| MediaPipe Face Landmarker | 3.6 MB | `GazeTracker._ensure_model()` |
| Whisper "small" | ~500 MB | `faster-whisper` on first STT call |
| OpenWakeWord | ~100 MB | `WakeWordDetector._init_model()` |

### Transfer from dev machine:
```bash
# Posture models
scp data/trained_models/posture_model.pkl   user@orin:~/desk-buddy/cs347w26-team7-desk-buddy/data/trained_models/
scp data/trained_models/posture_cnn.pt      user@orin:~/desk-buddy/cs347w26-team7-desk-buddy/data/trained_models/

# Calibration profile (optional — can re-calibrate on Orin)
scp data/calibration_profile.json user@orin:~/desk-buddy/cs347w26-team7-desk-buddy/data/
```

### Llama 3.1 8B GGUF (~4.6 GB):
```bash
# Option A: SCP from Mac
scp models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf user@orin:~/desk-buddy/cs347w26-team7-desk-buddy/models/

# Option B: Download on Orin
cd ~/desk-buddy/cs347w26-team7-desk-buddy/models
wget https://huggingface.co/TheBloke/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

### Piper TTS voice:
```bash
mkdir -p ~/.local/share/piper && cd ~/.local/share/piper
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

---

## Step 9: Configuration

Use the Jetson-optimized config:

```bash
cd ~/desk-buddy/cs347w26-team7-desk-buddy
cp config/pipeline.jetson.yaml config/pipeline.yaml
```

Or manually edit `config/pipeline.yaml`:
- `person_detector.device`: `"cpu"` → `"cuda:0"`
- `voice.stt.model`: `"small"` → `"medium"` (optional, Orin has headroom)
- `desk.enabled`: `false` if no BLE desk

No source code changes are needed — all components auto-detect CUDA.

---

## Step 10: Component Verification

Run the automated verification script:

```bash
python3 scripts/verify_jetson.py
```

This tests camera, YOLO, MediaPipe, CNN, Whisper, audio, TTS, LLM, and the full pipeline.

---

## Step 11: Run the Full System

**First run (with calibration):**
```bash
python3 -m src.main --no-desk
```

**Subsequent runs:**
```bash
python3 -m src.main --no-desk --skip-calibration
```

**Headless (production):**
```bash
python3 -m src.main --no-desk --skip-calibration --no-display
```

---

## Step 12: Production — systemd Service

Install the service for headless auto-start:

```bash
# Edit the service file with your username
sudo cp scripts/deskbuddy.service /etc/systemd/system/
sudo nano /etc/systemd/system/deskbuddy.service  # set your username

sudo systemctl daemon-reload
sudo systemctl enable deskbuddy
sudo systemctl start deskbuddy
sudo journalctl -u deskbuddy -f    # tail logs
```

---

## Step 13: Performance Optimization

**Max performance mode:**
```bash
sudo nvpmodel -m 0       # MAXN mode
sudo jetson_clocks        # lock clocks at max
```

**Export YOLOv8 to TensorRT (2-3x faster):**
```bash
python3 -c "
from ultralytics import YOLO
model = YOLO('models/yolov8s.pt')
model.export(format='engine', half=True, device=0)
"
# Then update pipeline.yaml: model: 'models/yolov8s.engine'
```

**Expected VRAM usage (64GB unified memory):**
| Component | VRAM |
|-----------|------|
| YOLOv8s CUDA | ~200 MB |
| MediaPipe Pose/Face | ~100 MB (CPU) |
| Posture CNN | ~50 MB |
| Whisper "small" | ~500 MB |
| Piper TTS | ~500 MB |
| Llama 3.1 8B Q4 | ~5 GB |
| **Total** | **~6-7 GB** (of 64 GB) |

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `torch.cuda.is_available()` → False | PyTorch from PyPI (x86) | Reinstall from NVIDIA wheel (Step 3) |
| `No matching distribution for mediapipe` | No aarch64 wheel | Build from source (Step 4 Option B) |
| `cv2.imshow` fails headless | No X display | Use `--no-display` flag |
| Camera not found | USB/V4L2 issue | `ls /dev/video*`, try different USB port |
| PyAudio "no default input" | PortAudio missing | `sudo apt install portaudio19-dev` |
| llama-cpp build fails | nvcc not on PATH | `export PATH=/usr/local/cuda/bin:$PATH` then rebuild |
| TTS no audio on Linux | Piper voice not downloaded | Download `.onnx` + `.json` to `~/.local/share/piper/` (Step 8) |
| sklearn pickle version mismatch | Different sklearn version | `pip install scikit-learn==<same version as dev machine>` |
| Low FPS | Power mode / no TensorRT | `sudo nvpmodel -m 0 && sudo jetson_clocks`, export to TensorRT (Step 13) |
