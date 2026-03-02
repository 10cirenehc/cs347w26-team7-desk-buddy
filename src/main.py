#!/usr/bin/env python3
"""
Desk Buddy - Voice-Enabled Posture and Focus Assistant

Unified entry point that combines:
- Perception pipeline (posture, gaze, focus detection)
- State logging and history
- Voice I/O (wake word, speech-to-text, text-to-speech)
- LLM agent for query processing
- Focus session management
- Adaptive alerts with desk control

Usage:
    python -m src.main
    python -m src.main --no-voice  # Without voice features
    python -m src.main --no-desk   # Without desk control
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/pipeline.yaml") -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if path.exists():
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}


class DeskBuddyApp:
    """
    Main application class for Desk Buddy.

    Orchestrates all components and runs the main loop.
    """

    def __init__(
        self,
        config_path: str = "config/pipeline.yaml",
        enable_voice: bool = True,
        enable_desk: bool = True,
        enable_display: bool = True,
        demo_mode: bool = False,
    ):
        """
        Initialize Desk Buddy application.

        Args:
            config_path: Path to configuration file
            enable_voice: Enable voice I/O
            enable_desk: Enable desk control
            enable_display: Show video display
            demo_mode: Use shortened alert thresholds for demo/presentation
        """
        self.config_path = config_path
        self.enable_voice = enable_voice
        self.enable_desk = enable_desk
        self.enable_display = enable_display
        self.demo_mode = demo_mode

        self.config = load_config(config_path)
        self._running = False

        # Components (initialized in setup)
        self.video_source = None
        self.person_detector = None
        self.tracker = None
        self.pose_estimator = None
        self.calibration = None
        self.calibration_profile = None  # Stores the active profile
        self.classifier = None
        self.posture_state = None
        self.focus_estimator = None
        self.presence_detector = None

        # CNN posture classifier (loaded in setup if available)
        self.cnn_model = None
        self.cnn_device = "cpu"
        self.cnn_threshold = 0.55
        self.use_depth_images = False

        self.state_logger = None
        self.audio_manager = None
        self.wake_word = None
        self.stt = None
        self.tts = None
        self.desk = None
        self.llm = None
        self.agent = None
        self.session = None
        self.alerts = None

    async def setup(self) -> bool:
        """
        Initialize all components.

        Returns:
            True if setup successful
        """
        logger.info("Initializing Desk Buddy...")

        try:
            # ----- Perception Pipeline -----
            from .perception import (
                VideoSource, PersonDetector, PrimaryTracker,
                PoseEstimator, CalibrationManager, PostureClassifier,
                PostureStateMachine, FocusEstimator,
                PresenceDetector, StateLogger
            )

            logger.info("Setting up perception pipeline...")

            self.video_source = VideoSource(config_path=self.config_path)
            if not self.video_source.open():
                logger.error("Failed to open video source (camera)")
                return False

            self.person_detector = PersonDetector(config_path=self.config_path)
            self.tracker = PrimaryTracker(config_path=self.config_path)
            self.pose_estimator = PoseEstimator(config_path=self.config_path)
            self.calibration = CalibrationManager(config_path=self.config_path)
            self.classifier = PostureClassifier(config_path=self.config_path)
            self.posture_state = PostureStateMachine(config_path=self.config_path)
            self.focus_estimator = FocusEstimator(config_path=self.config_path)
            self.presence_detector = PresenceDetector()

            # ----- CNN Posture Model (optional) -----
            # Try ONNX first (portable, no torch needed), then fall back to .pt
            cnn_config = self.config.get('posture_cnn', {})
            cnn_loaded = False

            # 1) ONNX path
            onnx_path = Path(cnn_config.get('onnx_path', 'data/trained_models/posture_cnn.onnx'))
            if not cnn_loaded and onnx_path.exists():
                try:
                    from .perception.posture_cnn import load_onnx_model
                    self.cnn_model, cnn_metadata = load_onnx_model(str(onnx_path))
                    self.cnn_model.eval()
                    self.cnn_device = "cpu"  # ONNX runtime handles device internally

                    in_channels = cnn_metadata.get("image_channels", 1)
                    self.use_depth_images = (in_channels == 3)
                    if cnn_metadata:
                        self.cnn_threshold = max(0.55, cnn_metadata.get('optimal_threshold', 0.55))

                    logger.info(f"CNN posture model loaded via ONNX "
                                f"(channels={in_channels}, threshold={self.cnn_threshold:.2f})")
                    cnn_loaded = True
                except ImportError:
                    logger.info("onnxruntime not installed, trying PyTorch .pt fallback")

            # 2) PyTorch .pt path
            cnn_pt_path = Path(cnn_config.get('model_path', 'data/trained_models/posture_cnn.pt'))
            if not cnn_loaded and cnn_pt_path.exists():
                try:
                    import torch
                    from .perception.posture_cnn import load_model as load_cnn_model

                    self.cnn_device = (
                        "cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available()
                        else "cpu"
                    )
                    self.cnn_model, cnn_metadata = load_cnn_model(str(cnn_pt_path), device=self.cnn_device)
                    self.cnn_model.eval()

                    in_channels = self.cnn_model.cnn.conv1.in_channels
                    self.use_depth_images = (in_channels == 3)
                    if cnn_metadata:
                        self.cnn_threshold = max(0.55, cnn_metadata.get('optimal_threshold', 0.55))

                    logger.info(f"CNN posture model loaded on {self.cnn_device} "
                                f"(channels={in_channels}, threshold={self.cnn_threshold:.2f})")
                    cnn_loaded = True
                except ImportError:
                    logger.info("torch not installed, using LogisticRegression fallback")

            if not cnn_loaded:
                logger.info("No CNN model available, using LogisticRegression fallback")

            # ----- State Logging -----
            logger.info("Setting up state logging...")

            state_logger_config = self.config.get('state_logger', {})
            self.state_logger = StateLogger(
                log_interval_seconds=state_logger_config.get('log_interval_seconds', 1.0),
                max_memory_snapshots=state_logger_config.get('max_memory_snapshots', 3600),
                output_dir=state_logger_config.get('output_dir', 'data/state_logs'),
            )

            # ----- Voice I/O -----
            if self.enable_voice:
                logger.info("Setting up voice I/O...")
                await self._setup_voice()

            # ----- Desk Control -----
            if self.enable_desk:
                logger.info("Setting up desk control...")
                await self._setup_desk()

            # ----- Agent -----
            logger.info("Setting up agent...")
            await self._setup_agent()

            logger.info("Desk Buddy initialized successfully!")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _setup_voice(self) -> None:
        """Setup voice I/O components."""
        try:
            from .voice import AudioManager, WakeWordDetector, SpeechToText, TextToSpeech

            voice_config = self.config.get('voice', {})

            # Audio manager
            self.audio_manager = AudioManager()

            # Wake word detector
            wake_config = voice_config.get('wake_word', {})
            self.wake_word = WakeWordDetector(
                wake_phrase=wake_config.get('phrase', 'hey_jarvis'),
                sensitivity=wake_config.get('sensitivity', 0.5),
            )

            # Speech-to-text
            stt_config = voice_config.get('stt', {})
            self.stt = SpeechToText(
                model_size=stt_config.get('model', 'base'),
                language=stt_config.get('language', 'en'),
            )

            # Text-to-speech
            tts_config = voice_config.get('tts', {})
            self.tts = TextToSpeech(
                voice=tts_config.get('voice', 'en_US-lessac-medium'),
                speed=tts_config.get('speed', 1.0),
                audio_manager=self.audio_manager,
            )

            logger.info("Voice I/O ready")

        except Exception as e:
            logger.warning(f"Voice setup failed: {e}")
            self.enable_voice = False

    async def _setup_desk(self) -> None:
        """Setup desk control."""
        try:
            from .desk import DeskClient

            desk_config = self.config.get('desk', {})
            self.desk = DeskClient(
                sitstand_path=desk_config.get('sitstand_path'),
                enabled=desk_config.get('enabled', True),
            )

            await self.desk.connect()
            logger.info("Desk control ready")

        except Exception as e:
            logger.warning(f"Desk setup failed: {e}")
            self.enable_desk = False

    async def _setup_agent(self) -> None:
        """Setup LLM agent and related components."""
        from .agent import LLMClient, DeskBuddyAgent, FocusSessionManager, AlertEngine

        agent_config = self.config.get('agent', {})

        # LLM client
        self.llm = LLMClient(
            model=agent_config.get('model', 'llama-3.1-8b'),
            max_tokens=agent_config.get('max_tokens', 150),
            temperature=agent_config.get('temperature', 0.7),
        )

        # Focus session manager
        history = self.state_logger.get_history()
        self.session = FocusSessionManager(history=history, demo_mode=self.demo_mode)

        # Main agent
        self.agent = DeskBuddyAgent(
            llm=self.llm,
            history=history,
            session=self.session,
            desk_callback=self._handle_desk_command if self.desk else None,
        )

        # Alert engine
        alerts_config = self.config.get('alerts', {})
        self.alerts = AlertEngine(
            desk_client=self.desk,
            tts=self.tts,
            enabled=alerts_config.get('enabled', True),
            demo_mode=self.demo_mode,
        )

    async def _handle_desk_command(self, command: str) -> None:
        """Handle desk command from agent."""
        if not self.desk:
            return

        if command == "stand":
            await self.desk.stand()
        elif command == "sit":
            await self.desk.sit()

    async def run_calibration(self) -> bool:
        """
        Run calibration phase.

        Returns:
            True if calibration successful
        """
        from .perception import extract_features, CalibrationManager

        calibration_config = self.config.get('calibration', {})
        duration = calibration_config.get('duration_seconds', 10)

        logger.info(f"Starting {duration}-second calibration. Sit with GOOD posture...")

        if self.tts:
            self.tts.speak("Starting calibration. Please sit with good posture.")

        self.calibration.start()
        self.presence_detector.start_calibration()

        start_time = time.time()
        detect_counter = 0
        detect_every_n = self.config.get('pipeline', {}).get('detect_every_n', 3)
        primary_bbox = None

        while time.time() - start_time < duration:
            frame = self.video_source.read()
            if frame is None:
                continue

            # Detection
            detect_counter += 1
            if detect_counter >= detect_every_n or primary_bbox is None:
                detect_counter = 0
                persons, phones = self.person_detector.detect(frame)
                tracked = self.tracker.update(persons, frame)
                primary_person = self.tracker.get_primary()
                primary_bbox = primary_person.bbox if primary_person else None

            if primary_bbox is None:
                continue

            # Pose estimation (PoseEstimator handles BGR→RGB internally)
            keypoints = self.pose_estimator.estimate(frame, primary_bbox)

            if keypoints and keypoints.avg_visibility > 0.5:
                features = extract_features(keypoints)
                self.calibration.add_sample(features)
                self.presence_detector.add_calibration_sample(keypoints)

            # Display progress
            if self.enable_display:
                elapsed = time.time() - start_time
                progress = f"Calibrating... {int(duration - elapsed)}s"
                cv2.putText(frame, progress, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Desk Buddy - Calibration", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return False

        # Finish calibration
        profile = self.calibration.finish()
        self.presence_detector.finish_calibration()

        if profile and profile.n_samples > 0:
            self.calibration_profile = profile
            # Save profile
            calibration_path = self.config.get('calibration', {}).get('save_path', 'data/calibration_profile.json')
            CalibrationManager.save(profile, calibration_path)
            logger.info(f"Calibration complete: {profile.n_samples} samples")
            if self.tts:
                self.tts.speak("Calibration complete. I'm now tracking your posture.")
            return True
        else:
            logger.error("Calibration failed - not enough samples")
            if self.tts:
                self.tts.speak("Calibration failed. Please try again.")
            return False

    async def run(self) -> None:
        """
        Run the main application loop.
        """
        from .perception import extract_features

        # Lazy imports for CNN path (avoid if torch not installed)
        if self.cnn_model is not None:
            import torch
            import numpy as np
            from .perception import render_skeleton, render_skeleton_depth

        self._running = True
        self.state_logger.start_session()

        # Start voice input if enabled
        if self.enable_voice and self.audio_manager and self.wake_word:
            self.audio_manager.start_input_stream()
            self.wake_word.start(self.audio_manager)

        logger.info("Desk Buddy is running. Press 'q' to quit.")

        # Test video source
        test_frame = self.video_source.read()
        if test_frame is None:
            logger.error("Video source not working - cannot read frames!")
            logger.error("Check if camera is connected and not in use by another app")
            return
        else:
            logger.info(f"Video source OK: {test_frame.shape}")

        detect_counter = 0
        detect_every_n = self.config.get('pipeline', {}).get('detect_every_n', 3)
        primary_bbox = None
        phones = []
        last_alert_check = 0
        alert_check_interval = 1.0  # Check alerts every second

        try:
            while self._running:
                frame = self.video_source.read()
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue

                # ----- Perception -----

                # Person detection (every N frames)
                detect_counter += 1
                if detect_counter >= detect_every_n or primary_bbox is None:
                    detect_counter = 0
                    persons, phones = self.person_detector.detect(frame)
                    tracked = self.tracker.update(persons, frame)
                    primary_person = self.tracker.get_primary()
                    primary_bbox = primary_person.bbox if primary_person else None

                # Variables for state logging
                posture = None
                features = None
                keypoints = None
                presence = None
                focus = None
                phone_detected = False
                phone_confidence = 0.0

                if primary_bbox is not None:
                    # Pose estimation (PoseEstimator handles BGR→RGB internally)
                    keypoints = self.pose_estimator.estimate(frame, primary_bbox)

                    if keypoints:
                        # Features and classification
                        features = extract_features(keypoints)

                        # CNN-first classification with LR fallback
                        if self.cnn_model is not None:
                            # CNN path (does not need calibration)
                            if self.use_depth_images:
                                skel_img = render_skeleton_depth(
                                    keypoints.landmarks, output_size=224, upper_body_only=True
                                )
                                skel_tensor = torch.from_numpy(skel_img).permute(2, 0, 1).float() / 255.0
                            else:
                                skel_img = render_skeleton(
                                    keypoints.landmarks, output_size=224, upper_body_only=True
                                )
                                skel_tensor = torch.from_numpy(skel_img).unsqueeze(0).float() / 255.0

                            skel_tensor = skel_tensor.unsqueeze(0).to(self.cnn_device)

                            feat_tensor = None
                            if self.cnn_model.use_features:
                                feat_array = np.array([
                                    features.torso_pitch,
                                    features.head_forward_ratio,
                                    features.shoulder_roll,
                                    features.lateral_lean,
                                    features.head_tilt,
                                    features.avg_visibility,
                                    features.forward_lean_z,
                                ], dtype=np.float32)
                                feat_tensor = torch.from_numpy(feat_array).unsqueeze(0).to(self.cnn_device)

                            with torch.no_grad():
                                raw_p = self.cnn_model.predict_proba(skel_tensor, feat_tensor).item()

                            # Threshold rescaling for state machine compatibility
                            if self.cnn_threshold != 0.5:
                                if raw_p < self.cnn_threshold:
                                    raw_p = raw_p * (0.5 / self.cnn_threshold)
                                else:
                                    raw_p = 0.5 + (raw_p - self.cnn_threshold) * (0.5 / (1 - self.cnn_threshold))

                            posture = self.posture_state.update(raw_p, features.avg_visibility)
                        else:
                            # LogisticRegression fallback (requires calibration)
                            normalized = None
                            if self.calibration_profile:
                                from .perception import CalibrationManager
                                normalized = CalibrationManager.normalize(features, self.calibration_profile)

                            if normalized is not None:
                                classification = self.classifier.predict(normalized)
                                posture = self.posture_state.update(
                                    classification.p_bad,
                                    features.avg_visibility
                                )

                        # Presence detection
                        presence = self.presence_detector.detect(keypoints)

                    # Phone detection (check overlap with primary person)
                    for phone_bbox in phones:
                        # Check if phone bbox overlaps with primary person bbox
                        overlap_x = max(0, min(phone_bbox.x2, primary_bbox.x2) - max(phone_bbox.x1, primary_bbox.x1))
                        overlap_y = max(0, min(phone_bbox.y2, primary_bbox.y2) - max(phone_bbox.y1, primary_bbox.y1))
                        if overlap_x > 0 and overlap_y > 0:
                            phone_detected = True
                            phone_confidence = phone_bbox.confidence
                            break

                # Focus estimation (outside primary_bbox block so AWAY is detected)
                focus = self.focus_estimator.estimate(
                    posture=posture,
                    phone_detected_in_hand=phone_detected,
                    phone_confidence=phone_confidence,
                    presence=presence,
                )

                # ----- State Logging -----
                self.state_logger.log(
                    posture=posture,
                    features=features,
                    gaze=None,
                    phone_detected=phone_detected,
                    phone_confidence=phone_confidence,
                    presence=presence,
                    focus=focus,
                )

                # ----- Voice Commands -----
                if self.enable_voice and self.wake_word and self.wake_word.detected():
                    await self._handle_voice_command()

                # ----- Focus Session -----
                if self.session:
                    suggestion = self.session.check_and_suggest()
                    if suggestion and self.tts:
                        self.tts.speak(suggestion.message)

                # ----- Adaptive Alerts -----
                now = time.time()
                if now - last_alert_check >= alert_check_interval:
                    last_alert_check = now
                    if self.alerts:
                        history = self.state_logger.get_history()
                        await self.alerts.check_and_execute(history, self.session)

                # ----- Display -----
                if self.enable_display:
                    display_frame = self._draw_overlay(
                        frame, posture, focus, presence,
                        phone_detected, features,
                    )
                    cv2.imshow("Desk Buddy", display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('c'):
                        # Manual calibration trigger
                        await self.run_calibration()

                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.001)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            await self.shutdown()

    async def _handle_voice_command(self) -> None:
        """Handle voice command after wake word detected."""
        if not self.stt or not self.agent:
            return

        logger.info("Wake word detected, listening for command...")

        if self.tts:
            self.tts.speak("Yes?")

        # Listen and transcribe
        result = self.stt.listen_and_transcribe(
            self.audio_manager,
            timeout_seconds=10,
        )

        if not result.text:
            logger.info("No speech detected")
            return

        logger.info(f"Transcribed: {result.text}")

        # Process with agent
        response = self.agent.process_query(result.text)

        # Speak response
        if self.tts:
            self.tts.speak(response)

        # Execute pending desk action if any
        pending = self.agent.get_pending_desk_action()
        if pending:
            await self._handle_desk_command(pending)

    def _draw_overlay(self, frame, posture, focus, presence,
                       phone_detected=False, features=None):
        """Draw status overlay on frame."""
        display = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        h, w = display.shape[:2]
        y = 30

        # ── Left column: primary status ──

        # Posture status
        if posture:
            color = (0, 255, 0) if posture.state.value == "good" else (0, 0, 255)
            label = posture.state.value.upper()
            cv2.putText(display, f"Posture: {label}", (10, y), font, 0.7, color, 2)
            y += 25
            cv2.putText(display, f"p_bad: {posture.raw_prob:.2f}  smooth: {posture.smoothed_prob:.2f}",
                       (10, y), font, 0.4, (200, 200, 200), 1)
            y += 18
            method = "CNN" if self.cnn_model else "LR"
            cv2.putText(display, f"method: {method}", (10, y), font, 0.4, (200, 200, 200), 1)
            y += 22

        # Focus status
        if focus:
            focus_colors = {
                "focused": (0, 255, 0),
                "distracted": (0, 0, 255),
                "away": (128, 128, 128),
            }
            color = focus_colors.get(focus.state.value, (255, 255, 255))
            cv2.putText(display, f"Focus: {focus.state.value.upper()}", (10, y), font, 0.7, color, 2)
            y += 25
            factors_str = ", ".join(focus.contributing_factors[:3])
            cv2.putText(display, f"factors: {factors_str}", (10, y), font, 0.4, (200, 200, 200), 1)
            y += 22

        # Presence
        if presence:
            cv2.putText(display, f"Presence: {presence.state.value}",
                       (10, y), font, 0.5, (255, 255, 255), 1)
            y += 22

        # Phone
        if phone_detected:
            cv2.putText(display, "PHONE DETECTED", (10, y), font, 0.5, (0, 0, 255), 1)
            y += 22

        # Session status
        if self.session and self.session.is_active:
            status = self.session.get_status()
            remaining = status.get('remaining_min', 0)
            phase = status.get('phase', 'unknown')
            cv2.putText(display, f"Session: {phase} ({remaining:.0f} min left)",
                       (10, y), font, 0.5, (0, 255, 255), 1)
            y += 22

        # ── Right column: desk + features ──

        rx = w - 250
        ry = 30

        # Desk info
        if self.desk and self.enable_desk:
            desk_status = self.desk.get_status()
            desk_color = (0, 255, 0) if desk_status.state.value == "connected" else (128, 128, 128)
            cv2.putText(display, f"Desk: {desk_status.state.value}",
                       (rx, ry), font, 0.5, desk_color, 1)
            ry += 20
            cv2.putText(display, f"Position: {desk_status.position.value}",
                       (rx, ry), font, 0.5, (200, 200, 200), 1)
            ry += 20
            height = desk_status.height_cm
            if height is not None:
                inches = height / 2.54
                cv2.putText(display, f"Height: {height:.0f}cm / {inches:.1f}in",
                           (rx, ry), font, 0.5, (200, 200, 200), 1)
                ry += 20
            if desk_status.last_command:
                cv2.putText(display, f"Last cmd: {desk_status.last_command}",
                           (rx, ry), font, 0.4, (160, 160, 160), 1)
                ry += 18
            ry += 10

        # Posture features
        if features:
            cv2.putText(display, "Features:", (rx, ry), font, 0.45, (180, 180, 180), 1)
            ry += 18
            fwd_color = (0, 0, 255) if features.forward_lean_z < -0.05 else (180, 180, 180)
            cv2.putText(display, f"pitch: {features.torso_pitch:.1f}  fwd_z: {features.forward_lean_z:.3f}",
                       (rx, ry), font, 0.38, fwd_color, 1)
            ry += 16
            cv2.putText(display, f"head_fwd: {features.head_forward_ratio:.2f}  roll: {features.shoulder_roll:.1f}",
                       (rx, ry), font, 0.38, (180, 180, 180), 1)
            ry += 16
            cv2.putText(display, f"lean: {features.lateral_lean:.2f}  tilt: {features.head_tilt:.1f}",
                       (rx, ry), font, 0.38, (180, 180, 180), 1)
            ry += 16
            cv2.putText(display, f"vis: {features.avg_visibility:.2f}",
                       (rx, ry), font, 0.38, (180, 180, 180), 1)
            ry += 20

        # Duration in current states (from history)
        if self.state_logger:
            history = self.state_logger.get_history()
            if posture:
                dur = history.duration_in_state("posture", posture.state.value)
                if dur > 0:
                    m, s = divmod(int(dur), 60)
                    cv2.putText(display, f"In {posture.state.value}: {m}m{s:02d}s",
                               (rx, ry), font, 0.4, (180, 180, 180), 1)
                    ry += 18
            if focus:
                dur = history.duration_in_state("focus", focus.state.value)
                if dur > 0:
                    m, s = divmod(int(dur), 60)
                    cv2.putText(display, f"In {focus.state.value}: {m}m{s:02d}s",
                               (rx, ry), font, 0.4, (180, 180, 180), 1)
                    ry += 18

        # Instructions
        cv2.putText(display, "Press 'q' to quit, 'c' to recalibrate",
                   (10, h - 10), font, 0.4, (128, 128, 128), 1)

        return display

    async def shutdown(self) -> None:
        """Clean up and shutdown."""
        logger.info("Shutting down...")

        self._running = False

        # End session
        if self.state_logger:
            summary = self.state_logger.end_session()
            logger.info(f"Session summary: {summary}")

        # Stop voice
        if self.wake_word:
            self.wake_word.stop()
        if self.audio_manager:
            self.audio_manager.close()

        # Disconnect desk
        if self.desk:
            await self.desk.disconnect()

        # Close video
        if self.video_source:
            self.video_source.release()

        # Close display
        cv2.destroyAllWindows()

        logger.info("Shutdown complete")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Desk Buddy - Voice-Enabled Posture Assistant")
    parser.add_argument("--config", default="config/pipeline.yaml", help="Path to config file")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice features")
    parser.add_argument("--no-desk", action="store_true", help="Disable desk control")
    parser.add_argument("--no-display", action="store_true", help="Disable video display")
    parser.add_argument("--skip-calibration", action="store_true", help="Skip calibration (use saved)")
    parser.add_argument("--demo", action="store_true", help="Demo mode with shortened alert thresholds")
    args = parser.parse_args()

    # Create app
    app = DeskBuddyApp(
        config_path=args.config,
        enable_voice=not args.no_voice,
        enable_desk=not args.no_desk,
        enable_display=not args.no_display,
        demo_mode=args.demo,
    )

    # Setup
    if not await app.setup():
        logger.error("Failed to initialize, exiting")
        sys.exit(1)

    # Calibration - check if saved profile exists
    calibration_path = app.config.get('calibration', {}).get('save_path', 'data/calibration_profile.json')
    has_saved_calibration = Path(calibration_path).exists()
    logger.info(f"Calibration path: {calibration_path}, exists: {has_saved_calibration}")

    if not args.skip_calibration and not has_saved_calibration:
        if not await app.run_calibration():
            logger.warning("Calibration failed, continuing with defaults")
    elif has_saved_calibration:
        # Load existing calibration
        from .perception import CalibrationManager
        profile = CalibrationManager.load(calibration_path)
        app.calibration_profile = profile
        logger.info(f"Loaded saved calibration ({profile.n_samples} samples)")

    # Handle signals
    def signal_handler(sig, frame):
        logger.info("Received signal, shutting down...")
        app._running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())
