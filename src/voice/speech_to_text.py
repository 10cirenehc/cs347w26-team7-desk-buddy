"""
Speech-to-text using Whisper (faster-whisper or openai-whisper fallback).

Transcribes voice commands after wake word detection.
Tries faster-whisper first (faster, needs ctranslate2), then falls back to
openai-whisper (PyTorch-based, works on Jetson AGX Orin out of the box).
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result from speech-to-text."""
    text: str
    language: str
    confidence: float
    duration_seconds: float
    segments: List[dict]


class SpeechToText:
    """
    Speech-to-text using Whisper (faster-whisper or openai-whisper).

    Tries faster-whisper first (CTranslate2, faster inference), then falls back
    to openai-whisper (PyTorch, works on Jetson where ctranslate2 is unavailable).

    Usage:
        stt = SpeechToText(model_size="base")

        # Transcribe audio array
        result = stt.transcribe(audio)
        print(result.text)

        # Listen from microphone and transcribe
        result = stt.listen_and_transcribe(audio_manager, timeout=10)
    """

    # Available model sizes (speed vs accuracy tradeoff)
    MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
        language: str = "en",
    ):
        """
        Initialize speech-to-text.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v2, large-v3)
            device: Device to use ("cpu", "cuda", "auto")
            compute_type: Computation type ("int8", "float16", "float32", "auto")
            language: Language code for transcription (e.g., "en")
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language

        self._model = None
        self._backend = None  # "faster_whisper" or "openai_whisper"
        self._init_model()

    def _init_model(self) -> bool:
        """Initialize the Whisper model, trying backends in order."""
        # Try 1: faster-whisper (faster, but needs ctranslate2)
        if self._init_faster_whisper():
            return True
        # Try 2: openai-whisper (uses PyTorch directly, works on Jetson)
        if self._init_openai_whisper():
            return True
        logger.warning("No Whisper backend available. Speech recognition disabled.")
        logger.warning("Install one of: pip install faster-whisper  OR  pip install openai-whisper")
        return False

    def _init_faster_whisper(self) -> bool:
        """Try to initialize using faster-whisper (CTranslate2 backend)."""
        try:
            from faster_whisper import WhisperModel

            # Determine device
            device = self.device
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            # Determine compute type
            compute_type = self.compute_type
            if compute_type == "auto":
                compute_type = "float16" if device == "cuda" else "int8"

            logger.info(f"Loading Whisper model (faster-whisper): {self.model_size} on {device}")
            self._model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute_type,
            )
            self._backend = "faster_whisper"
            logger.info("Whisper model loaded (faster-whisper backend)")
            return True

        except ImportError as e:
            logger.info(f"faster-whisper not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load faster-whisper model: {e}")
            return False

    def _init_openai_whisper(self) -> bool:
        """Try to initialize using openai-whisper (PyTorch backend)."""
        try:
            import whisper

            device = self.device
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            logger.info(f"Loading Whisper model (openai-whisper): {self.model_size} on {device}")
            self._model = whisper.load_model(self.model_size, device=device)
            self._backend = "openai_whisper"
            logger.info("Whisper model loaded (openai-whisper backend)")
            return True

        except ImportError as e:
            logger.info(f"openai-whisper not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load openai-whisper model: {e}")
            return False

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio: Audio as numpy array (int16 or float32)
            sample_rate: Audio sample rate (should be 16kHz)

        Returns:
            TranscriptionResult with text and metadata
        """
        if self._model is None:
            return TranscriptionResult(
                text="",
                language="unknown",
                confidence=0.0,
                duration_seconds=0.0,
                segments=[],
            )

        start_time = time.time()

        # Convert to float32 normalized [-1, 1] if needed
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Resample if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            # Simple resampling using scipy if available
            try:
                from scipy import signal
                audio = signal.resample(audio, int(len(audio) * 16000 / sample_rate))
            except ImportError:
                logger.warning(f"Audio sample rate is {sample_rate}, expected 16000")

        # Dispatch to the appropriate backend
        if self._backend == "openai_whisper":
            return self._transcribe_openai_whisper(audio, start_time)
        else:
            return self._transcribe_faster_whisper(audio, start_time)

    def _transcribe_faster_whisper(
        self, audio: np.ndarray, start_time: float
    ) -> TranscriptionResult:
        """Transcribe using faster-whisper backend."""
        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            vad_filter=True,  # Filter out silence
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=400,
            ),
        )

        # Collect segments
        segment_list = []
        text_parts = []
        for segment in segments:
            segment_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "confidence": segment.avg_logprob,
            })
            text_parts.append(segment.text)

        full_text = " ".join(text_parts).strip()
        duration = time.time() - start_time

        # Calculate average confidence
        confidence = self._compute_confidence(segment_list)

        return TranscriptionResult(
            text=full_text,
            language=info.language if info else self.language,
            confidence=confidence,
            duration_seconds=duration,
            segments=segment_list,
        )

    def _transcribe_openai_whisper(
        self, audio: np.ndarray, start_time: float
    ) -> TranscriptionResult:
        """Transcribe using openai-whisper backend."""
        # Determine if fp16 is appropriate (only on CUDA)
        use_fp16 = False
        try:
            import torch
            use_fp16 = next(self._model.parameters()).is_cuda
        except Exception:
            pass

        result = self._model.transcribe(
            audio,
            language=self.language,
            fp16=use_fp16,
        )

        # Collect segments (openai-whisper returns dicts)
        segment_list = []
        for seg in result.get("segments", []):
            segment_list.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "confidence": seg.get("avg_logprob", 0.0),
            })

        full_text = result.get("text", "").strip()
        duration = time.time() - start_time

        # Calculate average confidence
        confidence = self._compute_confidence(segment_list)

        return TranscriptionResult(
            text=full_text,
            language=result.get("language", self.language),
            confidence=confidence,
            duration_seconds=duration,
            segments=segment_list,
        )

    @staticmethod
    def _compute_confidence(segment_list: List[dict]) -> float:
        """Compute average confidence from segment log probabilities."""
        if not segment_list:
            return 0.0
        avg_logprob = sum(s["confidence"] for s in segment_list) / len(segment_list)
        # Convert log probability to rough confidence score
        return min(1.0, max(0.0, (avg_logprob + 1.0) / 1.0))

    def listen_and_transcribe(
        self,
        audio_manager,
        timeout_seconds: float = 10.0,
        silence_threshold: float = 0.01,
        silence_duration: float = 1.5,
        min_audio_duration: float = 0.5,
    ) -> TranscriptionResult:
        """
        Listen from microphone until silence, then transcribe.

        Args:
            audio_manager: AudioManager instance for mic input
            timeout_seconds: Max time to listen
            silence_threshold: Audio level threshold for silence detection
            silence_duration: Seconds of silence before stopping
            min_audio_duration: Minimum audio to collect before allowing silence stop

        Returns:
            TranscriptionResult with transcribed text
        """
        logger.info("Listening for speech...")

        chunks = []
        silence_start = None
        start_time = time.time()
        has_speech = False

        while True:
            elapsed = time.time() - start_time

            # Check timeout
            if elapsed >= timeout_seconds:
                logger.info("Listen timeout reached")
                break

            # Read audio
            chunk = audio_manager.read_audio(timeout=0.1)
            if chunk is None:
                continue

            chunks.append(chunk)
            level = audio_manager.get_audio_level(chunk)

            # Detect speech/silence
            if level > silence_threshold:
                has_speech = True
                silence_start = None
            else:
                if silence_start is None:
                    silence_start = time.time()
                elif has_speech and elapsed > min_audio_duration:
                    # Check if silence duration met
                    if time.time() - silence_start >= silence_duration:
                        logger.info("Silence detected, stopping")
                        break

        if not chunks:
            return TranscriptionResult(
                text="",
                language=self.language,
                confidence=0.0,
                duration_seconds=0.0,
                segments=[],
            )

        # Concatenate audio
        audio = np.concatenate(chunks)
        logger.info(f"Collected {len(audio) / 16000:.1f}s of audio")

        # Transcribe
        return self.transcribe(audio)

    def transcribe_stream(
        self,
        audio_generator,
        chunk_duration_seconds: float = 0.5,
    ):
        """
        Transcribe audio stream in real-time (generator-based).

        Args:
            audio_generator: Generator yielding audio chunks
            chunk_duration_seconds: Duration of each chunk

        Yields:
            Partial TranscriptionResult as audio is processed
        """
        # This would require streaming Whisper support
        # For now, just collect and transcribe
        logger.warning("Streaming transcription not yet implemented, using batch mode")

        chunks = list(audio_generator)
        if chunks:
            audio = np.concatenate(chunks)
            yield self.transcribe(audio)

    @property
    def is_available(self) -> bool:
        return self._model is not None
