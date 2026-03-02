"""
Wake word detection using OpenWakeWord.

Listens for "Hey Desk Buddy" (or custom phrase) to activate voice commands.
"""

import logging
import threading
import time
from typing import Optional, Callable, List
import numpy as np

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """
    Wake word detection using OpenWakeWord.

    Listens for a specific wake phrase and triggers a callback or
    sets a flag when detected.

    Usage:
        detector = WakeWordDetector(wake_phrase="hey_jarvis")
        detector.start(audio_manager)

        # Option 1: Check flag
        if detector.detected():
            handle_wake()

        # Option 2: Use callback
        detector = WakeWordDetector(on_wake=handle_wake)
        detector.start(audio_manager)
    """

    # Pre-trained model names available in OpenWakeWord
    AVAILABLE_MODELS = [
        "hey_jarvis",
        "alexa",
        "hey_mycroft",
        "ok_google",
        "hey_siri",
    ]

    def __init__(
        self,
        wake_phrase: str = "hey_jarvis",
        sensitivity: float = 0.5,
        on_wake: Optional[Callable[[], None]] = None,
        cooldown_seconds: float = 2.0,
    ):
        """
        Initialize wake word detector.

        Args:
            wake_phrase: Wake phrase model to use (see AVAILABLE_MODELS)
            sensitivity: Detection sensitivity (0.0 to 1.0)
            on_wake: Callback function when wake word detected
            cooldown_seconds: Minimum time between detections
        """
        self.wake_phrase = wake_phrase
        self.sensitivity = sensitivity
        self.on_wake = on_wake
        self.cooldown_seconds = cooldown_seconds

        self._model = None
        self._running = False
        self._detected_flag = threading.Event()
        self._last_detection_time = 0.0
        self._audio_manager = None
        self._listen_thread: Optional[threading.Thread] = None

        # Initialize model
        self._init_model()

    def _init_model(self) -> bool:
        """Initialize the OpenWakeWord model."""
        try:
            import openwakeword
            from openwakeword.model import Model

            # OpenWakeWord requires specific models to be downloaded
            # For custom phrases, we'd need to train a custom model
            # For now, use pre-trained models
            # Download model if missing
            try:
                from openwakeword.utils import download_models
                download_models(model_names=[self.wake_phrase])
            except Exception as dl_err:
                logger.debug(f"Model download skipped: {dl_err}")

            self._model = Model(
                wakeword_models=[self.wake_phrase],
            )

            logger.info(f"Loaded wake word model: {self.wake_phrase}")
            return True

        except ImportError as e:
            logger.warning(f"OpenWakeWord not available: {e}")
            logger.warning("Install with: pip install openwakeword")
            return False
        except Exception as e:
            logger.error(f"Failed to load wake word model: {e}")
            logger.info(f"Available models: {self.AVAILABLE_MODELS}")
            return False

    def start(self, audio_manager) -> bool:
        """
        Start wake word detection.

        Args:
            audio_manager: AudioManager instance for mic input

        Returns:
            True if started successfully
        """
        if self._model is None:
            logger.warning("Wake word model not loaded, detection disabled")
            return False

        if self._running:
            logger.warning("Wake word detection already running")
            return True

        self._audio_manager = audio_manager
        self._running = True
        self._detected_flag.clear()

        # Start background listening thread
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()

        logger.info("Wake word detection started")
        return True

    def stop(self) -> None:
        """Stop wake word detection."""
        self._running = False

        if self._listen_thread:
            self._listen_thread.join(timeout=2.0)
            self._listen_thread = None

        logger.info("Wake word detection stopped")

    def _listen_loop(self) -> None:
        """Background thread for wake word detection."""
        # OpenWakeWord expects 80ms chunks at 16kHz = 1280 samples
        chunk_samples = 1280

        while self._running:
            try:
                # Read audio chunk
                audio = self._audio_manager.read_audio(timeout=0.1)
                if audio is None:
                    continue

                # OpenWakeWord expects int16 audio
                if audio.dtype != np.int16:
                    audio = (audio * 32767).astype(np.int16)

                # Process through model
                predictions = self._model.predict(audio)

                # Check for wake word
                for model_name, score in predictions.items():
                    if score >= self.sensitivity:
                        self._handle_detection()
                        break

            except Exception as e:
                if self._running:
                    logger.error(f"Error in wake word detection: {e}")
                time.sleep(0.1)

    def _handle_detection(self) -> None:
        """Handle wake word detection."""
        now = time.time()

        # Check cooldown
        if now - self._last_detection_time < self.cooldown_seconds:
            return

        self._last_detection_time = now
        self._detected_flag.set()

        logger.info(f"Wake word detected: {self.wake_phrase}")

        # Call callback if registered
        if self.on_wake:
            try:
                self.on_wake()
            except Exception as e:
                logger.error(f"Error in wake word callback: {e}")

    def detected(self) -> bool:
        """
        Check if wake word was detected.

        Clears the flag after checking.

        Returns:
            True if wake word was detected since last check
        """
        was_detected = self._detected_flag.is_set()
        self._detected_flag.clear()
        return was_detected

    def wait_for_wake(self, timeout: Optional[float] = None) -> bool:
        """
        Block until wake word detected.

        Args:
            timeout: Max time to wait (None = wait forever)

        Returns:
            True if wake word detected, False if timeout
        """
        result = self._detected_flag.wait(timeout=timeout)
        self._detected_flag.clear()
        return result

    def reset(self) -> None:
        """Clear detection flag."""
        self._detected_flag.clear()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_available(self) -> bool:
        return self._model is not None
