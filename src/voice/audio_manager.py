"""
Audio device management for voice I/O.

Handles microphone input and speaker output with:
- Device enumeration and selection
- Audio stream management
- Buffer handling
"""

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Callable
import numpy as np

logger = logging.getLogger(__name__)

# Audio parameters
SAMPLE_RATE = 16000  # 16kHz for speech recognition
CHANNELS = 1  # Mono
CHUNK_SIZE = 512  # ~32ms at 16kHz
DTYPE = np.int16


@dataclass
class AudioConfig:
    """Audio configuration."""
    sample_rate: int = SAMPLE_RATE
    channels: int = CHANNELS
    chunk_size: int = CHUNK_SIZE
    input_device: Optional[int] = None  # None = default
    output_device: Optional[int] = None  # None = default


@dataclass
class AudioDevice:
    """Audio device info."""
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    is_default_input: bool = False
    is_default_output: bool = False


class AudioManager:
    """
    Manages audio input and output devices.

    Provides:
    - Device enumeration
    - Microphone input streaming
    - Speaker output
    - Audio level monitoring

    Usage:
        audio = AudioManager()
        audio.start_input_stream(callback=on_audio)
        ...
        audio.stop_input_stream()
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialize audio manager.

        Args:
            config: Audio configuration (uses defaults if None)
        """
        self.config = config or AudioConfig()

        self._pyaudio = None
        self._input_stream = None
        self._output_stream = None
        self._input_callback: Optional[Callable[[np.ndarray], None]] = None
        self._input_thread: Optional[threading.Thread] = None
        self._running = False
        self._audio_queue: queue.Queue = queue.Queue()

        # Try to initialize PyAudio
        self._init_pyaudio()

    def _init_pyaudio(self) -> bool:
        """Initialize PyAudio."""
        try:
            import pyaudio
            self._pyaudio = pyaudio.PyAudio()
            return True
        except ImportError:
            logger.warning("PyAudio not installed. Audio features will be disabled.")
            logger.warning("Install with: pip install pyaudio")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize PyAudio: {e}")
            return False

    def list_devices(self) -> List[AudioDevice]:
        """List available audio devices."""
        devices = []

        if self._pyaudio is None:
            return devices

        try:
            default_input = self._pyaudio.get_default_input_device_info()
            default_input_idx = default_input.get('index', -1)
        except Exception:
            default_input_idx = -1

        try:
            default_output = self._pyaudio.get_default_output_device_info()
            default_output_idx = default_output.get('index', -1)
        except Exception:
            default_output_idx = -1

        for i in range(self._pyaudio.get_device_count()):
            try:
                info = self._pyaudio.get_device_info_by_index(i)
                devices.append(AudioDevice(
                    index=i,
                    name=info.get('name', f'Device {i}'),
                    max_input_channels=int(info.get('maxInputChannels', 0)),
                    max_output_channels=int(info.get('maxOutputChannels', 0)),
                    default_sample_rate=float(info.get('defaultSampleRate', 16000)),
                    is_default_input=(i == default_input_idx),
                    is_default_output=(i == default_output_idx),
                ))
            except Exception as e:
                logger.warning(f"Failed to get info for device {i}: {e}")

        return devices

    def list_input_devices(self) -> List[AudioDevice]:
        """List available input (microphone) devices."""
        return [d for d in self.list_devices() if d.max_input_channels > 0]

    def list_output_devices(self) -> List[AudioDevice]:
        """List available output (speaker) devices."""
        return [d for d in self.list_devices() if d.max_output_channels > 0]

    def start_input_stream(
        self,
        callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> bool:
        """
        Start microphone input stream.

        Args:
            callback: Function called with each audio chunk (int16 numpy array)

        Returns:
            True if stream started successfully
        """
        if self._pyaudio is None:
            logger.error("PyAudio not available")
            return False

        if self._running:
            logger.warning("Input stream already running")
            return True

        self._input_callback = callback

        try:
            import pyaudio

            self._input_stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=self.config.input_device,
                frames_per_buffer=self.config.chunk_size,
            )

            self._running = True
            self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
            self._input_thread.start()

            logger.info("Audio input stream started")
            return True

        except Exception as e:
            logger.error(f"Failed to start input stream: {e}")
            return False

    def _input_loop(self) -> None:
        """Background thread for reading audio input."""
        while self._running and self._input_stream:
            try:
                data = self._input_stream.read(
                    self.config.chunk_size,
                    exception_on_overflow=False
                )
                audio = np.frombuffer(data, dtype=np.int16)

                # Add to queue for read_audio()
                try:
                    self._audio_queue.put_nowait(audio)
                except queue.Full:
                    pass  # Drop oldest

                # Call callback if registered
                if self._input_callback:
                    self._input_callback(audio)

            except Exception as e:
                if self._running:
                    logger.error(f"Error in input loop: {e}")
                break

    def stop_input_stream(self) -> None:
        """Stop microphone input stream."""
        self._running = False

        if self._input_stream:
            try:
                self._input_stream.stop_stream()
                self._input_stream.close()
            except Exception as e:
                logger.warning(f"Error closing input stream: {e}")
            self._input_stream = None

        if self._input_thread:
            self._input_thread.join(timeout=1.0)
            self._input_thread = None

        logger.info("Audio input stream stopped")

    def read_audio(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Read audio chunk from queue.

        Args:
            timeout: Max time to wait for audio

        Returns:
            Audio chunk as int16 numpy array, or None if timeout
        """
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def read_audio_seconds(self, seconds: float) -> np.ndarray:
        """
        Read specified duration of audio.

        Args:
            seconds: Duration to record

        Returns:
            Audio as int16 numpy array
        """
        chunks = []
        samples_needed = int(self.config.sample_rate * seconds)
        samples_collected = 0

        while samples_collected < samples_needed:
            chunk = self.read_audio(timeout=0.5)
            if chunk is not None:
                chunks.append(chunk)
                samples_collected += len(chunk)

        if not chunks:
            return np.array([], dtype=np.int16)

        audio = np.concatenate(chunks)
        return audio[:samples_needed]

    def play_audio(self, audio: np.ndarray, blocking: bool = True) -> bool:
        """
        Play audio through speaker.

        Args:
            audio: Audio as int16 numpy array
            blocking: Whether to block until playback complete

        Returns:
            True if playback successful
        """
        if self._pyaudio is None:
            logger.error("PyAudio not available")
            return False

        try:
            import pyaudio

            stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                output=True,
                output_device_index=self.config.output_device,
            )

            # Ensure audio is int16
            if audio.dtype != np.int16:
                audio = (audio * 32767).astype(np.int16)

            stream.write(audio.tobytes())

            if blocking:
                stream.stop_stream()
                stream.close()
            else:
                # Close in background
                def close_stream():
                    import time
                    time.sleep(len(audio) / self.config.sample_rate + 0.1)
                    stream.stop_stream()
                    stream.close()
                threading.Thread(target=close_stream, daemon=True).start()

            return True

        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
            return False

    def get_audio_level(self, audio: np.ndarray) -> float:
        """
        Calculate RMS audio level.

        Args:
            audio: Audio as numpy array

        Returns:
            RMS level (0.0 to 1.0 for int16 audio)
        """
        if len(audio) == 0:
            return 0.0

        rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
        # Normalize to 0-1 range for int16
        return min(1.0, rms / 32768.0)

    def close(self) -> None:
        """Release all audio resources."""
        self.stop_input_stream()

        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None

        logger.info("Audio manager closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def is_available(self) -> bool:
        return self._pyaudio is not None

    @property
    def is_input_active(self) -> bool:
        return self._running
