"""
Text-to-speech with Piper TTS, espeak-ng, and macOS 'say' fallback.

Provides voice output for assistant responses.
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
    Text-to-speech with Piper TTS, espeak-ng, and macOS 'say' fallback.

    Tries backends in order:
    1. Piper Python API (if voice model is found/downloadable)
    2. Piper CLI (requires voice .onnx model on disk)
    3. espeak-ng (common Linux fallback)
    4. macOS 'say' command (always available on macOS)

    Usage:
        tts = TextToSpeech(voice="en_US-lessac-medium")

        # Non-blocking (default) — won't freeze the main loop
        tts.speak("Hello!")

        # Blocking — wait until speech finishes
        tts.speak("Hello!", blocking=True)
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

    # Ordered fallback chain for runtime TTS attempts
    _SPEAK_CHAIN = ["espeak", "piper_python", "piper_cli", "say"]

    def __init__(
        self,
        voice: str = "en_US-lessac-medium",
        speed: float = 1.0,
        models_dir: Optional[str] = None,
        audio_manager=None,
        audio_device: Optional[str] = None,
    ):
        self.voice = voice
        self.speed = speed
        self.models_dir = Path(models_dir) if models_dir else Path.home() / ".local" / "share" / "piper"
        self.audio_manager = audio_manager
        self.audio_device = audio_device

        self._backend: Optional[str] = None  # "piper_python", "piper_cli", "espeak", "say"
        self._voice_path: Optional[Path] = None
        self._speaking = False
        self._speak_thread: Optional[threading.Thread] = None

        self._init_backend()

    def _init_backend(self) -> None:
        """Detect and initialize the best available TTS backend."""

        # 1. Try espeak-ng first (fast, reliable on Linux/Jetson)
        try:
            result = subprocess.run(
                ["espeak-ng", "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                self._backend = "espeak"
                logger.info("TTS backend: espeak-ng")
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # 2. Try Piper Python API
        try:
            import piper
            self._voice_path = self._find_voice_model()
            if self._voice_path:
                self._backend = "piper_python"
                logger.info(f"TTS backend: Piper Python (voice: {self._voice_path})")
                return
            else:
                logger.info("Piper Python available but voice model not found locally")
        except ImportError:
            pass

        # 3. Try Piper CLI
        try:
            result = subprocess.run(
                ["piper", "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                self._voice_path = self._find_voice_model()
                if self._voice_path:
                    self._backend = "piper_cli"
                    logger.info(f"TTS backend: Piper CLI (voice: {self._voice_path})")
                    return
                else:
                    logger.warning("Piper CLI available but voice model not found")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # 4. macOS 'say' fallback
        try:
            result = subprocess.run(
                ["say", ""],
                capture_output=True, timeout=3,
            )
            self._backend = "say"
            logger.info("TTS backend: macOS 'say'")
            return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        logger.warning("No TTS backend available. Voice output disabled.")

    def _find_voice_model(self) -> Optional[Path]:
        """Find voice model file."""
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

    def speak(self, text: str, blocking: bool = False) -> bool:
        """
        Speak text through speakers.

        Args:
            text: Text to speak
            blocking: Whether to block until speech complete (default: False)

        Returns:
            True if speech started/completed successfully
        """
        if not text.strip():
            return True

        if self._backend is None:
            logger.warning(f"TTS unavailable. Would say: {text}")
            return False

        if blocking:
            return self._speak_impl(text)
        else:
            self.speak_async(text)
            return True

    def _speak_impl(self, text: str) -> bool:
        """Speak text with runtime fallback chain (called from any thread)."""
        self._speaking = True
        try:
            methods = {
                "piper_python": self._speak_piper_python,
                "piper_cli": self._speak_piper_cli,
                "espeak": self._speak_espeak,
                "say": self._speak_say,
            }

            # Start from the selected backend and fall through on failure
            try:
                start = self._SPEAK_CHAIN.index(self._backend)
            except (ValueError, TypeError):
                start = 0

            for name in self._SPEAK_CHAIN[start:]:
                try:
                    if methods[name](text):
                        if name != self._backend:
                            logger.info(f"TTS fell back to '{name}' backend")
                        return True
                except Exception as e:
                    logger.debug(f"TTS backend '{name}' failed: {e}")

            logger.error("All TTS backends failed")
            return False
        finally:
            self._speaking = False

    def _speak_say(self, text: str) -> bool:
        """Speak using macOS 'say' command."""
        rate = int(200 * self.speed)
        result = subprocess.run(
            ["say", "-r", str(rate), text],
            capture_output=True,
        )
        return result.returncode == 0

    def _speak_piper_python(self, text: str) -> bool:
        """Speak using Piper Python API."""
        from piper import PiperVoice
        from piper.config import SynthesisConfig

        voice = PiperVoice.load(str(self._voice_path))
        syn_config = SynthesisConfig(length_scale=1.0 / self.speed)

        # synthesize() yields AudioChunk objects with audio_float_array
        chunks = []
        for chunk in voice.synthesize(text, syn_config=syn_config):
            int16_audio = (chunk.audio_float_array * 32767).astype(np.int16)
            chunks.append(int16_audio)

        audio_np = np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)

        if self.audio_manager:
            self.audio_manager.play_audio(audio_np, blocking=True)
        else:
            self._play_with_system(audio_np)
        return True

    def _speak_piper_cli(self, text: str) -> bool:
        """Speak using Piper CLI."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            cmd = ["piper", "--model", str(self._voice_path), "--output_file", output_path]
            if self.speed != 1.0:
                cmd.extend(["--length_scale", str(1.0 / self.speed)])

            result = subprocess.run(cmd, input=text, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Piper CLI error: {result.stderr}")
                return False

            import wave
            with wave.open(output_path, 'rb') as wf:
                audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

            if self.audio_manager:
                self.audio_manager.play_audio(audio, blocking=True)
            else:
                self._play_with_system(audio)
            return True
        finally:
            Path(output_path).unlink(missing_ok=True)

    def _speak_espeak(self, text: str) -> bool:
        """Speak using espeak-ng (Linux fallback)."""
        rate = int(175 * self.speed)
        if self.audio_device:
            # Pipe through aplay to reach the correct ALSA device
            espeak = subprocess.Popen(
                ["espeak-ng", "-s", str(rate), "--stdout", text],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            aplay = subprocess.Popen(
                ["aplay", "-D", self.audio_device],
                stdin=espeak.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            espeak.stdout.close()
            aplay.communicate()
            return aplay.returncode == 0
        else:
            result = subprocess.run(
                ["espeak-ng", "-s", str(rate), text],
                capture_output=True,
            )
            return result.returncode == 0

    def speak_async(self, text: str) -> None:
        """Speak text without blocking."""
        if self._speak_thread and self._speak_thread.is_alive():
            logger.debug("Already speaking, skipping")
            return

        self._speak_thread = threading.Thread(
            target=self._speak_impl,
            args=(text,),
            daemon=True,
        )
        self._speak_thread.start()

    def synthesize(self, text: str) -> Optional[np.ndarray]:
        """Synthesize text to audio array (for Piper backends only)."""
        if self._backend == "piper_python":
            try:
                from piper import PiperVoice
                from piper.config import SynthesisConfig

                voice = PiperVoice.load(str(self._voice_path))
                syn_config = SynthesisConfig(length_scale=1.0 / self.speed)
                chunks = [
                    (chunk.audio_float_array * 32767).astype(np.int16)
                    for chunk in voice.synthesize(text, syn_config=syn_config)
                ]
                return np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)
            except Exception as e:
                logger.error(f"Piper synthesis error: {e}")
        return None

    def _play_with_system(self, audio: np.ndarray) -> None:
        """Play audio using system tools."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            import wave
            voice_info = self.VOICES.get(self.voice, {"sample_rate": 22050})
            sample_rate = voice_info.get("sample_rate", 22050)

            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio.tobytes())

            for player in ["afplay", "aplay", "paplay", "play"]:
                try:
                    cmd = [player]
                    if player == "aplay" and self.audio_device:
                        cmd += ["-D", self.audio_device]
                    cmd.append(temp_path)
                    subprocess.run(cmd, capture_output=True, check=True)
                    break
                except (FileNotFoundError, subprocess.CalledProcessError):
                    continue
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def stop(self) -> None:
        """Stop current speech."""
        self._speaking = False

    def list_voices(self) -> List[VoiceInfo]:
        """List available voices."""
        return [
            VoiceInfo(name=n, language=i["language"], quality=i["quality"], sample_rate=i["sample_rate"])
            for n, i in self.VOICES.items()
        ]

    def set_voice(self, voice: str) -> bool:
        """Change the voice."""
        self.voice = voice
        self._voice_path = self._find_voice_model()
        return self._backend is not None

    @property
    def is_available(self) -> bool:
        return self._backend is not None

    @property
    def is_speaking(self) -> bool:
        return self._speaking
