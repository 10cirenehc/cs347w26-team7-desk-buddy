# Desk Buddy - Perception Layer

## Quick Start (Current Prototype)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/run_perception.py
```

## Current Prototype Structure

```
cs347w26-team7-desk-buddy/
├── requirements.txt
├── config/thresholds.yaml
├── src/perception/
│   ├── camera.py               # Webcam capture utility
│   ├── posture_detector.py     # MediaPipe Pose + rule-based classification
│   ├── phone_detector.py       # YOLOv8 phone detection
│   ├── gaze_tracker.py         # MediaPipe Face Mesh + solvePnP head pose
│   └── focus_estimator.py      # Signal fusion into focus state
└── scripts/
    ├── test_posture.py
    ├── test_phone_detection.py
    ├── test_gaze.py
    └── run_perception.py
```

### Prototype Limitations

1. **Rule-based posture classification is fragile** - geometric thresholds on landmarks don't generalize across camera angles or body types
2. **Gaze tracking assumes frontal camera** - solvePnP head pose only works when camera faces the user
3. **Single-person only**
4. **MediaPipe pose detection is noisy** for seated upper-body views
5. **No occlusion handling**

---

## Target Architecture: LAViTSPose-Inspired Pipeline on AGX Orin

Based on [LAViTSPose](https://www.mdpi.com/1099-4300/27/12/1196) (MDPI Entropy, Nov 2025), a cascaded detection-segmentation-classification framework designed specifically for multi-person sitting posture recognition under occlusion.

### Why LAViTSPose is the Right Model

LAViTSPose solves our exact problem:
- Multi-person sitting posture recognition in classrooms/offices
- Handles occlusion (desks, chairs, overlapping people)
- Works from arbitrary camera angles (classifies skeleton images, not raw geometry)
- Lightweight enough for real-time on edge devices
- Three-stage cascade suppresses errors at each stage

### Key Insight: Classify Skeletons, Not Geometry

Our current approach computes angles/distances from landmarks and applies thresholds. This is fragile because:
- Thresholds don't generalize across camera angles
- Small landmark noise causes state flicker
- Can't distinguish subtle posture variations

LAViTSPose instead:
1. Extracts skeleton keypoints
2. Renders them as a **binary skeleton image** (rectangle-based, not thin lines)
3. Feeds the skeleton image to a **trained ViT classifier**

This abstracts away camera perspective - the classifier learns posture from structural shape, not raw coordinates.

### Proposed Pipeline for Desk Buddy on AGX Orin

```
Camera Frame (640x480 or higher)
    │
    ▼
┌─────────────────────────────────────────────┐
│ Stage 1: PERSON DETECTION + TRACKING        │
│                                             │
│  YOLOv8m/l + BoT-SORT                      │
│  - Detect all people in frame               │
│  - Stable person IDs across frames          │
│  - RaIoU-style loss for tight crops         │
│  - TensorRT optimized                       │
│  - Also detect phones (COCO class 67)       │
└──────────────┬──────────────────────────────┘
               │ Per-person ROI crops
               ▼
┌─────────────────────────────────────────────┐
│ Stage 2: PER-PERSON PARSING                 │
│                                             │
│  2a. Segmentation (ESBody-style)            │
│      - Remove cross-person leakage (Reno)   │
│      - Estimate occlusion + head orient.    │
│      - Route to HB/WB classification branch │
│                                             │
│  2b. Pose Estimation                        │
│      RTMPose-l or ViTPose-B                 │
│      - Extract skeleton keypoints           │
│      - Render rectangle-based skeleton      │
│      - TensorRT optimized                   │
│                                             │
│  2c. Gaze Estimation                        │
│      L2CS-Net (ResNet50) or Gaze360         │
│      - Face crop → pitch/yaw prediction     │
│      - Works from any camera angle           │
│      - No solvePnP needed                   │
└──────────────┬──────────────────────────────┘
               │ Skeleton image + gaze + phone + occlusion cues
               ▼
┌─────────────────────────────────────────────┐
│ Stage 3: CLASSIFICATION + FUSION            │
│                                             │
│  3a. Posture Classification (MLiT-style)    │
│      - Skeleton image → ViT classifier      │
│      - SDC for local inductive bias         │
│      - Learnable temperature for stability  │
│      - HB/WB branches for occluded cases    │
│      Output: posture category per person    │
│                                             │
│  3b. Focus State Fusion                     │
│      - Posture state (good/slouch/lean/...) │
│      - Gaze direction (at screen / away)    │
│      - Phone detected near person           │
│      - Head orientation from ESBody         │
│      - Temporal smoothing                   │
│      Output: FOCUSED / DISTRACTED / AWAY    │
└─────────────────────────────────────────────┘
```

### Component Selection for AGX Orin (275 TOPS)

| Component | Model | Why | Params | Est. Speed on Orin |
|-----------|-------|-----|--------|-------------------|
| Person Detection | YOLOv8l or YOLOv11l | Accurate, TensorRT native | ~43M | 100+ FPS |
| Multi-Person Tracking | BoT-SORT | Best MOTA/IDF1 on MOT17, re-ID capable | minimal | adds ~5ms |
| Person Segmentation | BodyPix 2.0 (MobileNet) or SAM2-tiny | ESBody uses BodyPix; SAM2 for higher quality | 4-39M | 60-90 FPS |
| Pose Estimation | RTMPose-l | 75.8 AP on COCO, 130+ FPS on GPU | ~28M | 200+ FPS |
| Skeleton Rendering | Custom (rectangle-based) | LAViTSPose approach, width ω=4 | none | <1ms |
| Posture Classifier | MLiT (compact ViT) | SDC + learnable temp, small-data friendly | ~5-10M | 500+ FPS |
| Gaze Estimation | L2CS-Net (ResNet50) | 3.92° on MPIIGaze, works unconstrained | ~25M | 100+ FPS |
| Phone Detection | YOLOv8l (shared with person det.) | COCO class 67, same model | shared | shared |

**Total estimated per-frame latency on AGX Orin with TensorRT: ~25-40ms (25-40 FPS)**

This is well within real-time for multi-person scenes. With all models converted to TensorRT FP16, the AGX Orin has ample headroom.

### LAViTSPose Concepts to Adopt

#### 1. RaIoU Loss (Detection)
- Standard IoU losses produce loose boxes in occluded seated scenes
- RaIoU adds dataset-derived priors on box width/height/aspect ratio
- Zero gradient for in-range predictions, Huber penalty for outliers
- **For us**: Fine-tune YOLO with RaIoU on classroom/office seated data

#### 2. ESBody: Reno + APF (Segmentation)
- **Reno**: Suppresses boundary-connected foreground from neighboring people. Uses BodyPix foreground mask, filters components touching the ROI boundary that likely belong to adjacent people
- **APF**: Analyzes body part probabilities to determine lower-body occlusion ratio and coarse head orientation. No training needed - pure geometric heuristics on BodyPix output
- **Routing**: If lower-body visibility < 15%, route to half-body classifier; otherwise whole-body
- **For us**: Adopt directly. BodyPix runs on MobileNet, very lightweight

#### 3. Rectangle-Based Skeleton (Pose → Image)
- Instead of thin lines, limbs rendered as solid rectangles (width ω=4)
- Enhances structural continuity and feature density
- More robust to keypoint noise than thin-line skeletons
- Rendered onto 224×224 canvas with letterbox scaling
- **For us**: Replace our geometric metrics with skeleton image classification

#### 4. MLiT Classifier (Classification)
- Compact ViT with two key additions:
  - **SDC (Spatial Displacement Contact)**: Concatenates spatially shifted versions of the input before patch embedding. Injects local inductive bias without convolutions
  - **Learnable Temperature**: Trainable scalar τ in softmax attention. Stabilizes training on small datasets
- Two branches: HB (half-body) and WB (whole-body), selected by APF occlusion routing
- **For us**: Train on sitting posture dataset (USSP or custom-collected)

### Gaze Estimation: L2CS-Net

Replaces our fragile solvePnP approach. Key advantages:
- Predicts gaze pitch/yaw directly from a face crop
- Works from any camera angle (trained on Gaze360 dataset with full 360° coverage)
- 3.92° mean angular error on MPIIGaze
- Simple API: `gaze_pipeline.step(frame)` → pitch, yaw per detected face
- ResNet50 backbone, ONNX/TensorRT exportable
- [github.com/Ahmednull/L2CS-Net](https://github.com/Ahmednull/L2CS-Net)

For Desk Buddy, gaze direction relative to body orientation (from pose landmarks) gives us angle-independent attention tracking.

### TensorRT Deployment on AGX Orin

All models should be converted to TensorRT for production:

```
PyTorch model → ONNX export → TensorRT engine (FP16)
```

- TensorRT provides 3-6x speedup over raw PyTorch on Jetson
- FP16 inference with negligible accuracy loss
- Can use DLA (Deep Learning Accelerator) cores on Orin to offload some models, freeing GPU
- JetPack 6.x includes CUDA, cuDNN, TensorRT pre-installed
- [NVIDIA TensorRT SDK](https://developer.nvidia.com/tensorrt)

### Data Collection & Training Plan

LAViTSPose was evaluated on the USSP (University Sitting Student Posture) dataset. For Desk Buddy:

1. **Collect classroom/office seated posture data** using the AGX Orin camera
2. **Annotate posture categories**: Upright, Slouching, Leaning-sideways, Head-on-desk, etc.
3. **Fine-tune YOLO** with RaIoU loss on seated indoor scenes
4. **Train MLiT classifier** on skeleton images extracted via RTMPose
5. **Fine-tune L2CS-Net** on gaze data if needed (pre-trained may suffice)

### Optional: V-JEPA for Activity Understanding

With AGX Orin compute budget, V-JEPA 2 (ViT-L, 300M) can run alongside the main pipeline as a complementary signal:
- Classify overall activity from short video clips (working, chatting, sleeping, on phone)
- Self-supervised pre-training means it works without labeled data
- Complements geometric posture classification with semantic understanding
- [github.com/facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)

---

## Development Notes

- MediaPipe requires RGB input; OpenCV reads BGR by default
- YOLOv8n model auto-downloads on first phone detection run (~6MB)
- MediaPipe pose/face models auto-download to `src/perception/models/`
- Focus estimator uses temporal smoothing to prevent state flickering
- Posture shoulder angle is normalized to handle mirrored webcam views
- AGX Orin should run JetPack 6.x with DeepStream 7.1 (DeepStream 8.0 is Thor-only)

## References

- [LAViTSPose](https://www.mdpi.com/1099-4300/27/12/1196) - Lightweight Cascaded Framework for Sitting Posture Recognition (MDPI Entropy, 2025)
- [L2CS-Net](https://github.com/Ahmednull/L2CS-Net) - Fine-Grained Gaze Estimation in Unconstrained Environments
- [RTMPose](https://arxiv.org/abs/2303.07399) - Real-Time Multi-Person Pose Estimation
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) - Vision Transformer for Pose Estimation
- [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - Robust Multi-Pedestrian Tracking
- [SAM2](https://github.com/facebookresearch/sam2) - Segment Anything Model 2
- [V-JEPA 2](https://github.com/facebookresearch/vjepa2) - Self-supervised Video Understanding
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) - Inference Optimization
- [NVIDIA DeepStream](https://developer.nvidia.com/deepstream-sdk) - Video Analytics Pipeline
