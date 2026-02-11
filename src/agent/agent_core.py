"""
Main agent logic for Desk Buddy.

Handles user queries, builds context from state history,
and generates responses using the LLM.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..perception.state_history import StateHistory
    from .llm_client import LLMClient
    from .focus_session import FocusSessionManager

logger = logging.getLogger(__name__)


# System prompt for the agent
SYSTEM_PROMPT = """You are Desk Buddy, a friendly AI assistant that helps users maintain good posture and focus while working.

Your personality:
- Friendly and supportive, not nagging or judgmental
- Concise - keep responses to 1-2 sentences for most queries
- Action-oriented - suggest specific things when appropriate
- Encouraging - celebrate good habits and progress

When discussing time, use natural language ("about 10 minutes" not "623 seconds").
When the user asks about their status, be specific but brief.
If you don't have data, acknowledge it and offer to help in other ways.

You can help with:
- Posture feedback and reminders
- Focus tracking and productivity tips
- Starting/managing focus sessions
- Break reminders
- General desk wellness advice
"""


@dataclass
class IntentMatch:
    """Result of intent matching."""
    intent: str
    confidence: float
    params: Dict[str, Any]


class DeskBuddyAgent:
    """
    Main agent for processing user queries.

    Combines intent matching for common commands with LLM for
    general conversation and complex queries.

    Usage:
        agent = DeskBuddyAgent(llm, history, session)

        # Process voice command
        response = agent.process_query("How's my posture?")
        tts.speak(response)
    """

    # Intent patterns (regex -> intent name)
    INTENT_PATTERNS = [
        # Focus session commands
        (r"start.*(?:focus|session|pomodoro|timer)", "start_focus"),
        (r"start.*(\d+).*(?:minute|min)", "start_focus_duration"),
        (r"(?:take|start|need).*break", "start_break"),
        (r"end.*(?:session|focus|timer)|stop.*(?:session|focus|timer)", "end_session"),
        (r"(?:skip|next)", "skip_session"),
        (r"(?:pause|hold)", "pause_session"),

        # Status queries
        (r"how.*(?:am i|my).*(?:doing|posture|focus)", "status_general"),
        (r"how.*long.*(?:been|sitting|standing|slouch|focus)", "status_duration"),
        (r"how.*(?:was|my).*(?:session|day|today)", "status_summary"),
        (r"what.*(?:time|remaining|left)", "status_time"),

        # Posture specific
        (r"(?:check|how).*posture", "check_posture"),
        (r"posture.*(?:tip|advice|help)", "posture_advice"),

        # Focus specific
        (r"(?:am i|being).*(?:focus|distract)", "check_focus"),
        (r"focus.*(?:tip|advice|help)", "focus_advice"),

        # Desk commands
        (r"(?:stand|standing).*(?:up|desk)", "desk_stand"),
        (r"(?:sit|sitting).*(?:down|desk)", "desk_sit"),

        # General
        (r"^(?:hi|hello|hey)(?:\s|$)", "greeting"),
        (r"(?:thank|thanks)", "thanks"),
        (r"(?:help|what can you)", "help"),
    ]

    def __init__(
        self,
        llm: 'LLMClient',
        history: 'StateHistory',
        session: Optional['FocusSessionManager'] = None,
        desk_callback: Optional[Callable] = None,
    ):
        """
        Initialize agent.

        Args:
            llm: LLM client for generation
            history: StateHistory for context
            session: FocusSessionManager (optional)
            desk_callback: Callback for desk commands (async)
        """
        self.llm = llm
        self.history = history
        self.session = session
        self.desk_callback = desk_callback

        # Compile intent patterns
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), intent)
            for pattern, intent in self.INTENT_PATTERNS
        ]

    def process_query(self, user_query: str) -> str:
        """
        Process a user voice query and generate response.

        Args:
            user_query: Transcribed voice command

        Returns:
            Response text to speak
        """
        logger.info(f"Processing query: {user_query}")

        # Try intent matching first
        intent = self._match_intent(user_query)

        if intent and intent.confidence > 0.7:
            response = self._handle_intent(intent, user_query)
            if response:
                return response

        # Fall back to LLM
        return self._generate_llm_response(user_query)

    def _match_intent(self, query: str) -> Optional[IntentMatch]:
        """Match query against intent patterns."""
        query_lower = query.lower().strip()

        for pattern, intent in self._compiled_patterns:
            match = pattern.search(query_lower)
            if match:
                # Extract any captured groups as params
                params = {}
                groups = match.groups()
                if groups:
                    # Try to parse numbers
                    for i, g in enumerate(groups):
                        if g and g.isdigit():
                            params[f"number_{i}"] = int(g)
                        elif g:
                            params[f"param_{i}"] = g

                return IntentMatch(
                    intent=intent,
                    confidence=0.9,
                    params=params,
                )

        return None

    def _handle_intent(self, intent: IntentMatch, original_query: str) -> Optional[str]:
        """Handle a matched intent."""
        handler = getattr(self, f"_intent_{intent.intent}", None)

        if handler:
            try:
                return handler(intent.params, original_query)
            except Exception as e:
                logger.error(f"Error handling intent {intent.intent}: {e}")
                return None

        return None

    # ----- Intent Handlers -----

    def _intent_start_focus(self, params: Dict, query: str) -> str:
        """Start a focus session."""
        if self.session:
            return self.session.start_focus()
        return "Focus sessions aren't enabled right now."

    def _intent_start_focus_duration(self, params: Dict, query: str) -> str:
        """Start focus session with specific duration."""
        duration = params.get("number_0", 25)
        if self.session:
            return self.session.start_focus(duration_min=duration)
        return f"Starting a {duration}-minute focus session."

    def _intent_start_break(self, params: Dict, query: str) -> str:
        """Start a break."""
        if self.session:
            return self.session.start_break()
        return "Taking a break! Stand up and stretch."

    def _intent_end_session(self, params: Dict, query: str) -> str:
        """End current session."""
        if self.session and self.session.is_active:
            stats = self.session.end()
            return f"Session ended. You were focused {stats.focus_ratio:.0%} of the time with {stats.posture_good_ratio:.0%} good posture."
        return "No active session to end."

    def _intent_skip_session(self, params: Dict, query: str) -> str:
        """Skip to next phase."""
        if self.session:
            return self.session.skip_to_next()
        return "No active session."

    def _intent_pause_session(self, params: Dict, query: str) -> str:
        """Pause session."""
        if self.session and self.session.is_active:
            return "Session paused. Say 'resume' when you're ready."
        return "No active session to pause."

    def _intent_status_general(self, params: Dict, query: str) -> str:
        """General status query."""
        summary = self.history.get_summary(window_seconds=300)

        if not summary.get("has_data"):
            return "I don't have enough data yet. Let me observe for a bit longer."

        current = summary.get("current", {})
        durations = summary.get("durations", {})

        parts = []

        # Current posture
        posture = current.get("posture", "unknown")
        if posture == "bad":
            bad_duration = durations.get("bad_posture_seconds", 0)
            if bad_duration > 60:
                parts.append(f"Your posture needs attention - you've been slouching for {self._format_duration(bad_duration)}")
            else:
                parts.append("Your posture could be better")
        elif posture == "good":
            good_duration = durations.get("good_posture_seconds", 0)
            if good_duration > 300:
                parts.append(f"Great posture! You've been sitting well for {self._format_duration(good_duration)}")
            else:
                parts.append("Your posture looks good")

        # Focus status
        focus = current.get("focus", "unknown")
        if focus == "distracted":
            parts.append("and you seem a bit distracted")
        elif focus == "focused":
            parts.append("and you're focused")

        if not parts:
            return "Everything looks okay! Keep up the good work."

        return ". ".join(parts) + "."

    def _intent_status_duration(self, params: Dict, query: str) -> str:
        """Duration-based status query."""
        query_lower = query.lower()

        if "slouch" in query_lower or "bad" in query_lower:
            duration = self.history.duration_in_state("posture", "bad")
            if duration > 0:
                return f"You've been slouching for {self._format_duration(duration)}."
            return "Your posture has been good recently!"

        if "sitting" in query_lower:
            duration = self.history.duration_in_state("presence", "seated")
            return f"You've been sitting for {self._format_duration(duration)}."

        if "standing" in query_lower:
            duration = self.history.duration_in_state("presence", "standing")
            if duration > 0:
                return f"You've been standing for {self._format_duration(duration)}."
            return "You're currently sitting."

        if "focus" in query_lower:
            focused = self.history.duration_in_state("focus", "focused")
            return f"You've been focused for {self._format_duration(focused)}."

        return "I'm tracking your session. What specifically would you like to know?"

    def _intent_status_summary(self, params: Dict, query: str) -> str:
        """Session/day summary."""
        # Use longer window for day summary
        summary = self.history.get_summary(window_seconds=3600)

        if not summary.get("has_data"):
            return "I haven't been tracking long enough for a summary yet."

        ratios = summary.get("ratios", {})
        good_posture = ratios.get("good_posture", 0)
        focused = ratios.get("focused", 0)

        parts = []
        parts.append(f"In the last hour, you had {good_posture:.0%} good posture")
        parts.append(f"and were focused {focused:.0%} of the time")

        if self.session:
            status = self.session.get_status()
            completed = status.get("sessions_completed", 0)
            if completed > 0:
                parts.append(f"You completed {completed} focus session{'s' if completed > 1 else ''}")

        return ". ".join(parts) + "."

    def _intent_status_time(self, params: Dict, query: str) -> str:
        """Time remaining query."""
        if self.session and self.session.is_active:
            return self.session.get_status_summary()
        return "No active session. Say 'start focus' to begin one."

    def _intent_check_posture(self, params: Dict, query: str) -> str:
        """Check current posture."""
        current = self.history.get_current()

        if not current:
            return "I can't see you right now. Make sure you're in view of the camera."

        posture = current.posture_state

        if posture == "good":
            return "Your posture looks good! Keep it up."
        elif posture == "bad":
            # Get more specific feedback if we have features
            if current.torso_pitch and current.torso_pitch > 15:
                return "You're leaning forward quite a bit. Try sitting back and straightening up."
            if current.forward_lean_z and current.forward_lean_z < -0.1:
                return "Your shoulders are hunched forward. Roll them back and sit up straight."
            return "Your posture could use some attention. Try sitting up straighter."
        else:
            return "I can't quite see your posture clearly. Try adjusting your position."

    def _intent_posture_advice(self, params: Dict, query: str) -> str:
        """Posture tips."""
        tips = [
            "Keep your feet flat on the floor and your back against the chair.",
            "Position your screen at eye level to avoid looking down.",
            "Take a break every 30 minutes to stand and stretch.",
            "Keep your shoulders relaxed and pulled back slightly.",
            "Make sure your keyboard and mouse are at elbow height.",
        ]
        import random
        return random.choice(tips)

    def _intent_check_focus(self, params: Dict, query: str) -> str:
        """Check focus state."""
        current = self.history.get_current()

        if not current:
            return "I need to observe you for a moment first."

        focus = current.focus_state
        factors = current.focus_factors

        if focus == "focused":
            return "You're focused! Keep up the great work."
        elif focus == "distracted":
            if "phone_in_hand" in factors:
                return "You seem distracted - I noticed you checking your phone."
            elif "looking_away" in factors:
                return "You seem distracted - you've been looking away from the screen."
            return "You seem a bit distracted. Try to refocus on your task."
        else:
            return "I'm not sure - are you at your desk?"

    def _intent_focus_advice(self, params: Dict, query: str) -> str:
        """Focus tips."""
        tips = [
            "Try the Pomodoro technique: 25 minutes of focus, then a 5-minute break.",
            "Put your phone out of reach to reduce distractions.",
            "Close unnecessary browser tabs and apps.",
            "Use noise-canceling headphones or ambient sounds.",
            "Set a clear goal for what you want to accomplish.",
        ]
        import random
        return random.choice(tips)

    def _intent_desk_stand(self, params: Dict, query: str) -> str:
        """Stand desk up."""
        if self.desk_callback:
            # This would be called async
            return "Moving desk to standing position."
        return "Desk control isn't connected right now."

    def _intent_desk_sit(self, params: Dict, query: str) -> str:
        """Lower desk."""
        if self.desk_callback:
            return "Moving desk to sitting position."
        return "Desk control isn't connected right now."

    def _intent_greeting(self, params: Dict, query: str) -> str:
        """Respond to greeting."""
        return "Hi! I'm Desk Buddy, your posture and focus assistant. How can I help?"

    def _intent_thanks(self, params: Dict, query: str) -> str:
        """Respond to thanks."""
        return "You're welcome! Let me know if you need anything else."

    def _intent_help(self, params: Dict, query: str) -> str:
        """Help message."""
        return ("I can help with posture and focus. Try asking: "
                "'How's my posture?', 'Start a focus session', "
                "'How long have I been sitting?', or 'Give me a posture tip'.")

    # ----- LLM Fallback -----

    def _generate_llm_response(self, query: str) -> str:
        """Generate response using LLM."""
        context = self._build_context()

        response = self.llm.generate(
            prompt=query,
            system_prompt=SYSTEM_PROMPT,
            context=context,
        )

        return response.text

    def _build_context(self) -> Dict[str, Any]:
        """Build context dict for LLM."""
        summary = self.history.get_summary(window_seconds=300)
        current = self.history.get_current()

        context = {
            "current_state": {},
            "durations": {},
            "session_stats": {},
        }

        if current:
            context["current_state"] = {
                "posture": current.posture_state,
                "focus": current.focus_state,
                "presence": current.presence_state,
                "face_detected": current.face_detected,
            }

        if summary.get("has_data"):
            context["durations"] = summary.get("durations", {})
            context["session_stats"] = {
                "good_posture_pct": summary.get("ratios", {}).get("good_posture", 0),
                "focused_pct": summary.get("ratios", {}).get("focused", 0),
            }

        if self.session:
            context["focus_session"] = self.session.get_status()

        return context

    def _format_duration(self, seconds: float) -> str:
        """Format seconds as human-readable duration."""
        if seconds < 60:
            return f"{int(seconds)} seconds"
        elif seconds < 3600:
            mins = int(seconds / 60)
            return f"about {mins} minute{'s' if mins > 1 else ''}"
        else:
            hours = int(seconds / 3600)
            mins = int((seconds % 3600) / 60)
            if mins > 0:
                return f"about {hours} hour{'s' if hours > 1 else ''} and {mins} minutes"
            return f"about {hours} hour{'s' if hours > 1 else ''}"
