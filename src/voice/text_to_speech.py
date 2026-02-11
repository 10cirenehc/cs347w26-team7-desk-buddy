"""
Text-to-speech using Piper TTS.

Provides fast, natural-sounding voice output for assistant responses.
"""

import logging
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VoiceInfo:
    """Information about a TTS voice."""
    name: str
    language: str
    quality: str  # "low", "medium", "high"
    sample_rate: int
    path: Optional[str] = None


class TextToSpeech:
    """
    Text-to-speech using Piper TTS.

    Provides fast, natural-sounding voice synthesis for assistant responses.

    Usage:
        tts = TextToSpeech(voice="en_US-lessac-medium")

        # Blocking speech
        tts.speak("Hello, how can I help you?")

        # Non-blocking speech
        tts.speak_async("Processing your request...")

        # Get audio for custom playback
        audio = tts.synthesize("Some text")
        audio_manager.play_audio(audio)
    """

    # Popular Piper voices
    VOICES = {
        "en_US-lessac-medium": {
            "language": "en",
            "quality": "medium",
            "sample_rate": 22050,
        },
        "en_US-lessac-high": {
            "language": "en",
            "quality": "high",
            "sample_rate": 22050,
        },
        "en_US-amy-medium": {
            "language": "en",
            "quality": "medium",
            "sample_rate": 22050,
        },
        "en_US-ryan-medium": {
            "language": "en",
            "quality": "medium",
            "sample_rate": 22050,
        },
        "en_GB-alan-medium": {
            "language": "en",
            "quality": "medium",
            "sample_rate": 22050,
        },
    }

    def __init__(
        self,
        voice: str = "en_US-lessac-medium",
        speed: float = 1.0,
        models_dir: Optional[str] = None,
        audio_manager=None,
    ):
        """
        Initialize text-to-speech.

        Args:
            voice: Voice model name (see VOICES)
            speed: Speech speed multiplier (0.5 = slow, 1.0 = normal, 2.0 = fast)
            models_dir: Directory for voice models
            audio_manager: AudioManager for playback (optional)
        """
        self.voice = voice
        self.speed = speed
        self.models_dir = Path(models_dir) if models_dir else Path.home() / ".local" / "share" / "piper"
        self.audio_manager = audio_manager

        self._piper_available = False
        self._voice_path: Optional[Path] = None
        self._speaking = False
        self._speak_thread: Optional[threading.Thread] = None

        # Initialize Piper
        self._init_piper()

    def _init_piper(self) -> bool:
        """Initialize Piper TTS."""
        # Check if piper is installed
        try:
            result = subprocess.run(
                ["piper", "--help"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                self._piper_available = True
                logger.info("Piper TTS available (CLI)")
            else:
                logger.warning("Piper CLI not working properly")
        except FileNotFoundError:
            logger.warning("Piper CLI not found. Trying Python module...")

            # Try Python module
            try:
                import piper
                self._piper_available = True
                logger.info("Piper TTS available (Python module)")
            except ImportError:
                logger.warning("Piper TTS not installed. Voice output disabled.")
                logger.warning("Install with: pip install piper-tts")
                return False

        # Check for voice model
        self._voice_path = self._find_voice_model()
        if self._voice_path:
            logger.info(f"Using voice model: {self._voice_path}")
        else:
            logger.warning(f"Voice model not found: {self.voice}")
            logger.info("Voice models will be downloaded on first use")

        return self._piper_available

    def _find_voice_model(self) -> Optional[Path]:
        """Find voice model file."""
        # Check common locations
        search_paths = [
            self.models_dir / f"{self.voice}.onnx",
            self.models_dir / self.voice / f"{self.voice}.onnx",
            Path(f"/usr/share/piper/voices/{self.voice}.onnx"),
            Path.cwd() / "models" / "piper" / f"{self.voice}.onnx",
        ]

        for path in search_paths:
            if path.exists():
                return path

        return None

    def speak(self, text: str, blocking: bool = True) -> bool:
        """
        Speak text through speakers.

        Args:
            text: Text to speak
            blocking: Whether to block until speech complete

        Returns:
            True if speech started/completed successfully
        """
        if not text.strip():
            return True

        if not self._piper_available:
            logger.warning(f"TTS unavailable. Would say: {text}")
            return False

        if blocking:
            return self._speak_blocking(text)
        else:
            self.speak_async(text)
            return True

    def _speak_blocking(self, text: str) -> bool:
        """Speak text (blocking)."""
        self._speaking = True

        try:
            audio = self.synthesize(text)
            if audio is None:
                return False

            # Play audio
            if self.audio_manager:
                self.audio_manager.play_audio(audio, blocking=True)
            else:
                self._play_with_system(audio)

            return True

        except Exception as e:
            logger.error(f"Error speaking: {e}")
            return False

        finally:
            self._speaking = False

    def speak_async(self, text: str) -> None:
        """
        Speak text without blocking.

        Args:
            text: Text to speak
        """
        if self._speak_thread and self._speak_thread.is_alive():
            logger.warning("Already speaking, queuing...")
            # Could implement a queue here

        self._speak_thread = threading.Thread(
            target=self._speak_blocking,
            args=(text,),
            daemon=True,
        )
        self._speak_thread.start()

    def synthesize(self, text: str) -> Optional[np.ndarray]:
        """
        Synthesize text to audio array.

        Args:
            text: Text to synthesize

        Returns:
            Audio as int16 numpy array, or None on error
        """
        if not self._piper_available:
            return None

        try:
            # Try Python module first
            try:
                import piper

                # Use piper-tts Python API
                voice = piper.PiperVoice.load(
                    str(self._voice_path) if self._voice_path else self.voice
                )
                audio = voice.synthesize(text, length_scale=1.0 / self.speed)
                return np.array(audio, dtype=np.int16)

            except (ImportError, Exception) as e:
                logger.debug(f"Piper Python API not available: {e}")
                pass

            # Fall back to CLI
            return self._synthesize_cli(text)

        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            return None

    def _synthesize_cli(self, text: str) -> Optional[np.ndarray]:
        """Synthesize using Piper CLI."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            cmd = ["piper", "--model", self.voice, "--output_file", output_path]

            # Add length scale for speed
            if self.speed != 1.0:
                cmd.extend(["--length_scale", str(1.0 / self.speed)])

            # Run piper
            result = subprocess.run(
                cmd,
                input=text,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.error(f"Piper error: {result.stderr}")
                return None

            # Read output WAV
            import wave
            with wave.open(output_path, 'rb') as wf:
                audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                return audio

        finally:
            # Clean up temp file
            Path(output_path).unlink(missing_ok=True)

    def _play_with_system(self, audio: np.ndarray) -> None:
        """Play audio using system tools."""
        # Save to temp file and play with system audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            import wave
            voice_info = self.VOICES.get(self.voice, {"sample_rate": 22050})
            sample_rate = voice_info.get("sample_rate", 22050)

            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio.tobytes())

            # Try different system players
            for player in ["aplay", "afplay", "paplay", "play"]:
                try:
                    subprocess.run(
                        [player, temp_path],
                        capture_output=True,
                        check=True,
                    )
                    break
                except (FileNotFoundError, subprocess.CalledProcessError):
                    continue

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def stop(self) -> None:
        """Stop current speech."""
        self._speaking = False
        # Would need more sophisticated handling for actual interruption

    def list_voices(self) -> List[VoiceInfo]:
        """List available voices."""
        voices = []
        for name, info in self.VOICES.items():
            voices.append(VoiceInfo(
                name=name,
                language=info["language"],
                quality=info["quality"],
                sample_rate=info["sample_rate"],
            ))
        return voices

    def set_voice(self, voice: str) -> bool:
        """
        Change the voice.

        Args:
            voice: New voice name

        Returns:
            True if voice set successfully
        """
        self.voice = voice
        self._voice_path = self._find_voice_model()
        return self._voice_path is not None or self._piper_available

    @property
    def is_available(self) -> bool:
        return self._piper_available

    @property
    def is_speaking(self) -> bool:
        return self._speaking
