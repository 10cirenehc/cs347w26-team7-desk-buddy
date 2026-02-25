"""
Voice I/O module for Desk Buddy.

Provides:
- Wake word detection (OpenWakeWord)
- Speech-to-text (Whisper via faster-whisper)
- Text-to-speech (Piper TTS)
- Audio device management
"""

from .audio_manager import AudioManager, AudioConfig
from .wake_word import WakeWordDetector
from .speech_to_text import SpeechToText
from .text_to_speech import TextToSpeech

__all__ = [
    "AudioManager",
    "AudioConfig",
    "WakeWordDetector",
    "SpeechToText",
    "TextToSpeech",
]
