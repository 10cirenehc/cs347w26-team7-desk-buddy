"""
Agent module for Desk Buddy.

Provides:
- LLM client (Llama 3.1 8B via llama.cpp or TensorRT-LLM)
- Agent core (query processing + prompt templates)
- Focus session manager (smart productivity timer)
- Alert engine (adaptive rules + desk/voice actions)
"""

from .llm_client import LLMClient, LLMConfig, LLMResponse
from .focus_session import FocusSessionManager, SessionPhase, SessionStats
from .agent_core import DeskBuddyAgent
from .alert_engine import AlertEngine, AlertRule, AlertAction

__all__ = [
    "LLMClient",
    "LLMConfig",
    "LLMResponse",
    "FocusSessionManager",
    "SessionPhase",
    "SessionStats",
    "DeskBuddyAgent",
    "AlertEngine",
    "AlertRule",
    "AlertAction",
]
