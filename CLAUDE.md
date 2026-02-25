# Desk Buddy - Voice-Enabled Posture & Focus Assistant

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# For voice features (macOS: run pyobjc-core first)
pip install pyobjc-core
pip install -r requirements-voice.txt

# Run the full assistant
python -m src.main

# Run without voice (perception only)
python -m src.main --no-voice

# Run without desk control
python -m src.main --no-desk

# Skip calibration (use saved profile)
python -m src.main --skip-calibration
```

On first run:
1. Downloads YOLOv8s (~22 MB) and MediaPipe models
2. Runs 10-second calibration — sit with **good** posture
3. Begins tracking with voice commands enabled

### LLM Setup (Optional)

For the AI agent to work beyond simulation mode:

```bash
# Install llama-cpp-python
pip install llama-cpp-python

# With Metal GPU (macOS):
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# With CUDA (Linux):
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

Download a GGUF model (e.g., Llama 3.1 8B) to `~/models/` or `./models/`.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DESK BUDDY ASSISTANT                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │   PERCEPTION │     │  STATE LOG   │     │    AGENT     │    │
│  │   PIPELINE   │────►│   SYSTEM     │────►│   (LLM)      │    │
│  │              │     │              │     │              │    │
│  │ • Posture    │     │ • History    │     │ • Llama 8B   │    │
│  │ • Gaze       │     │ • Events     │     │ • Reasoning  │    │
│  │ • Phone      │     │ • Summaries  │     │ • Decisions  │    │
│  │ • Focus      │     │              │     │              │    │
│  └──────────────┘     └──────────────┘     └──────┬───────┘    │
│                                                    │            │
│                              ┌─────────────────────┘            │
│                              ▼                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │    VOICE     │     │   ALERT      │     │    VOICE     │    │
│  │    INPUT     │────►│   ENGINE     │────►│   OUTPUT     │    │
│  │              │     │              │     │              │    │
│  │ • Wake word  │     │ • Rules      │     │ • Piper TTS  │    │
│  │ • Whisper    │     │ • Desk ctrl  │     │ • Speaker    │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
cs347w26-team7-desk-buddy/
├── requirements.txt              # Core dependencies
├── requirements-voice.txt        # Voice/LLM dependencies
├── config/pipeline.yaml          # Unified config
├── data/
│   ├── calibration_profile.json  # Auto-generated calibration
│   ├── state_logs/               # Session state logs (JSONL)
│   ├── posture_sessions/         # Training data
│   └── trained_models/           # Classifier artifacts
│
├── src/
│   ├── __init__.py
│   ├── main.py                   # ★ Unified entry point
│   │
│   ├── perception/               # Computer vision pipeline
│   │   ├── video_source.py       # Webcam with frame-drop
│   │   ├── person_detector.py    # YOLOv8s person + phone
│   │   ├── primary_tracker.py    # ByteTrack + sticky selection
│   │   ├── pose_estimator.py     # MediaPipe Pose (33 landmarks)
│   │   ├── posture_features.py   # 7-feature vector extraction
│   │   ├── calibration.py        # Z-score normalization
│   │   ├── posture_model.py      # LogisticRegression classifier
│   │   ├── posture_state.py      # EWMA + hysteresis state machine
│   │   ├── gaze_tracker.py       # Head pose via solvePnP
│   │   ├── focus_estimator.py    # Multi-signal focus fusion
│   │   ├── presence_detector.py  # Seated/standing/away detection
│   │   ├── state_logger.py       # ★ StateSnapshot logging
│   │   ├── state_history.py      # ★ Ring buffer + query API
│   │   ├── state_summarizer.py   # ★ NL summaries for agent
│   │   └── skeleton_renderer.py  # LAViTSPose-style rendering
│   │
│   ├── voice/                    # ★ Voice I/O module
│   │   ├── audio_manager.py      # Mic/speaker handling
│   │   ├── wake_word.py          # OpenWakeWord ("Hey Jarvis")
│   │   ├── speech_to_text.py     # Whisper ASR
│   │   └── text_to_speech.py     # Piper TTS
│   │
│   ├── desk/                     # ★ BLE desk control
│   │   └── desk_client.py        # Async sit/stand/nudge
│   │
│   └── agent/                    # ★ LLM agent module
│       ├── llm_client.py         # Llama 3.1 8B (llama.cpp)
│       ├── agent_core.py         # Query processing + intents
│       ├── focus_session.py      # Smart productivity timer
│       └── alert_engine.py       # Adaptive rules + desk actions
│
└── scripts/
    ├── run_pipeline.py           # Legacy perception-only demo
    ├── collect_posture_sessions.py
    ├── train_posture.py
    └── test_gaze.py
```

---

## Voice Commands

**Wake word:** "Hey Jarvis" (configurable in `config/pipeline.yaml`)

| Command | Response |
|---------|----------|
| "How's my posture?" | Current posture status with specific feedback |
| "How long have I been slouching?" | Duration in bad posture state |
| "Start a focus session" | Begins 25-min Pomodoro timer |
| "Start a 50 minute focus" | Custom duration focus session |
| "Take a break" | Starts break, may lower desk |
| "How am I doing?" | Overall status summary |
| "End session" | Ends focus session with stats |
| "Stand up" / "Sit down" | Moves desk (if connected) |
| "Give me a posture tip" | Random ergonomic advice |

---

## State Logging & History API

The `StateLogger` captures perception state at ~1 Hz for agent context and queries.

```python
from src.perception import StateLogger

logger = StateLogger()
logger.start_session()

# In perception loop:
logger.log(posture=posture, gaze=gaze, focus=focus, ...)

# Query history:
history = logger.get_history()
history.duration_in_state("posture", "bad")           # seconds in current state
history.state_ratio("focus", "focused", 300)          # % focused in last 5 min
history.get_trend("posture_smoothed_prob", 300)       # trend analysis
history.get_summary(3600)                              # full summary dict
```

### StateSnapshot Fields

| Field | Type | Description |
|-------|------|-------------|
| `posture_state` | str | "good" / "bad" / "unknown" |
| `posture_raw_prob` | float | Raw p_bad from classifier |
| `posture_smoothed_prob` | float | EWMA-smoothed probability |
| `torso_pitch` | float | Forward lean angle (degrees) |
| `forward_lean_z` | float | Shoulder-hip depth difference |
| `gaze_pitch/yaw/roll` | float | Head pose angles |
| `attention_state` | str | "focused" / "looking_away" / "looking_down" |
| `phone_detected` | bool | Phone in hand |
| `presence_state` | str | "seated" / "standing" / "away" |
| `focus_state` | str | "focused" / "distracted" / "away" |
| `focus_factors` | list | Contributing factors |

---

## Focus Sessions

The `FocusSessionManager` provides Pomodoro-style timers with adaptive suggestions.

```python
from src.agent import FocusSessionManager

session = FocusSessionManager(history)
session.start_focus(duration_min=25)

# In main loop:
suggestion = session.check_and_suggest()
if suggestion:
    tts.speak(suggestion.message)

# Adaptive features:
# - Early break suggestion if focus degrades
# - Posture warnings during session
# - Session completion summaries with stats
```

---

## Adaptive Alerts

The `AlertEngine` triggers voice and desk actions based on rules:

| Rule | Condition | Action |
|------|-----------|--------|
| `focus_posture_degrading` | >60% bad posture in 5 min | Silent desk nudge |
| `focus_severe_slouch` | 15 min continuous bad | Stand desk + voice |
| `idle_bad_posture` | 10 min bad (not in focus) | Stand desk + voice |
| `sitting_too_long` | 1 hour seated | Voice reminder |
| `standing_too_long` | 45 min standing | Voice reminder |
| `good_posture_streak` | 30 min good posture | Encouragement |
| `phone_distraction` | 2 min phone during focus | Voice nudge |

Rules adapt based on session context (gentler during focus sessions).

---

## Configuration

All settings in `config/pipeline.yaml`:

```yaml
# State logging
state_logger:
  log_interval_seconds: 1.0
  output_dir: "data/state_logs"
  max_memory_snapshots: 3600

# Voice I/O
voice:
  wake_word:
    phrase: "hey_jarvis"  # or: alexa, hey_mycroft, ok_google
    sensitivity: 0.5
  stt:
    model: "base"  # tiny, base, small, medium, large-v2
  tts:
    voice: "en_US-lessac-medium"
    speed: 1.0

# LLM Agent
agent:
  model: "llama-3.1-8b"
  max_tokens: 150
  temperature: 0.7

# Alerts
alerts:
  enabled: true
  positive_reinforcement: true

# Desk control
desk:
  enabled: false  # Enable when sitstand repo available
  sitstand_path: "../sitstand"
```

---

## Perception Pipeline

```
Camera Frame (640×480)
    │
    ├─[every Nth frame]─► PersonDetector (YOLOv8s)
    │                         │
    │                         ▼
    │                     PrimaryTracker (ByteTrack)
    │                         │
    │                         ▼
    ├─[every frame]──────► PoseEstimator (MediaPipe 33 landmarks)
    │                         │
    │                         ▼
    │                     extract_features() → PostureFeatures (7D)
    │                         │
    │                         ▼
    │                     CalibrationManager.normalize() → z-scores
    │                         │
    │                         ▼
    │                     PostureClassifier.predict() → p_bad
    │                         │
    │                         ▼
    │                     PostureStateMachine → GOOD / BAD / UNKNOWN
    │                         │
    │                         ▼
    │                     StateLogger.log() → history
    │                         │
    │                         ▼
    └────────────────────► FocusEstimator → FOCUSED / DISTRACTED / AWAY
                              │
                              ▼
                          AlertEngine.check() → voice/desk actions
```

### PostureFeatures (7D Vector)

| Feature | Description |
|---------|-------------|
| `torso_pitch` | Hip→shoulder angle vs vertical (degrees) |
| `head_forward_ratio` | Ear-shoulder offset / shoulder width |
| `shoulder_roll` | Shoulder line tilt (degrees) |
| `lateral_lean` | Horizontal offset / shoulder width |
| `head_tilt` | Ear line tilt (degrees) |
| `avg_visibility` | Mean landmark visibility |
| `forward_lean_z` | Shoulder z - hip z (negative = forward) |

---

## Hardware Requirements

### Development (MacBook/Desktop)
- Webcam
- Microphone (for voice commands)
- ~4GB RAM for perception
- ~8GB additional for LLM (Llama 8B Q4)

### Production (AGX Orin 64GB)
| Component | VRAM | Notes |
|-----------|------|-------|
| Perception | ~2GB | YOLOv8 + MediaPipe |
| Whisper medium | ~2GB | ASR |
| Piper TTS | ~500MB | Voice output |
| Llama 3.1 8B (Q4) | ~6GB | Agent reasoning |
| **Total** | **~10-12GB** | Plenty of headroom |

---

## Future: LAViTSPose-Inspired Pipeline

Target architecture for multi-person posture recognition on AGX Orin:

1. **YOLOv8l + BoT-SORT** — Person detection & tracking
2. **RTMPose-l** — 33-landmark pose estimation
3. **Rectangle-based skeleton rendering** — ω=4 limb width
4. **MLiT classifier** — ViT with SDC + learnable temperature
5. **L2CS-Net** — Direct gaze prediction (replaces solvePnP)
6. **TensorRT FP16** — All models optimized

See full LAViTSPose architecture details in the [original plan](./PLAN.md).

---

## Development Notes

- MediaPipe requires RGB input; OpenCV reads BGR
- Wake word uses OpenWakeWord pre-trained models (custom requires training)
- LLM runs in simulation mode if llama-cpp-python not installed
- Desk control requires BLE-enabled standing desk + sitstand repo
- State logs saved as JSONL in `data/state_logs/`
- Calibration profile saved to `data/calibration_profile.json`

## References

- [LAViTSPose](https://www.mdpi.com/1099-4300/27/12/1196) - Sitting Posture Recognition
- [L2CS-Net](https://github.com/Ahmednull/L2CS-Net) - Gaze Estimation
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - CTranslate2 Whisper
- [Piper](https://github.com/rhasspy/piper) - Fast TTS
- [OpenWakeWord](https://github.com/dscripka/openWakeWord) - Wake word detection
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Local LLM inference
