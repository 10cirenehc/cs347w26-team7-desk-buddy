"""
LLM client for local inference.

Supports:
- llama.cpp (CPU/GPU inference)
- TensorRT-LLM (optimized for NVIDIA GPUs/Jetson)
- OpenAI-compatible API fallback
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import time

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM inference."""
    model: str = "llama-3.1-8b"
    model_path: Optional[str] = None  # Path to model file
    max_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "auto"  # "cpu", "cuda", "auto"
    n_ctx: int = 4096  # Context window
    n_gpu_layers: int = -1  # -1 = all layers on GPU


@dataclass
class LLMResponse:
    """Response from LLM generation."""
    text: str
    tokens_generated: int
    time_seconds: float
    finish_reason: str  # "stop", "length", "error"


class LLMClient:
    """
    LLM client for local model inference.

    Supports multiple backends:
    - llama-cpp-python: CPU/GPU inference with llama.cpp
    - TensorRT-LLM: Optimized for NVIDIA GPUs
    - API: OpenAI-compatible endpoints

    Usage:
        llm = LLMClient(model="llama-3.1-8b")

        response = llm.generate("How's my posture today?", context=state_summary)
        print(response.text)
    """

    def __init__(self, config: Optional[LLMConfig] = None, **kwargs):
        """
        Initialize LLM client.

        Args:
            config: LLMConfig instance
            **kwargs: Override config values
        """
        self.config = config or LLMConfig()

        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self._llm = None
        self._backend = None

        # Initialize backend
        self._init_backend()

    def _init_backend(self) -> bool:
        """Initialize LLM backend."""
        # Try llama-cpp-python
        if self._init_llama_cpp():
            return True

        logger.warning("No LLM backend available. Running in simulation mode.")
        logger.info("Install llama-cpp-python: pip install llama-cpp-python")
        self._backend = "simulation"
        return False

    def _init_llama_cpp(self) -> bool:
        """Initialize llama-cpp-python backend."""
        try:
            from llama_cpp import Llama

            model_path = self._find_model_path()
            if not model_path:
                logger.warning(f"Model not found: {self.config.model}")
                return False

            # Determine GPU layers
            n_gpu_layers = self.config.n_gpu_layers
            if n_gpu_layers == -1 and self.config.device == "cpu":
                n_gpu_layers = 0

            logger.info(f"Loading model with llama.cpp: {model_path}")
            self._llm = Llama(
                model_path=model_path,
                n_ctx=self.config.n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
            self._backend = "llama_cpp"
            logger.info("LLM loaded successfully")
            return True

        except ImportError:
            logger.info("llama-cpp-python not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to load llama.cpp model: {e}")
            return False

    def _find_model_path(self) -> Optional[str]:
        """Find model file path."""
        from pathlib import Path

        if self.config.model_path:
            path = Path(self.config.model_path)
            if path.exists():
                return str(path)

        model_name = self.config.model
        patterns = [
            f"*{model_name}*.gguf",
            f"*llama*8b*.gguf",
            f"*Meta-Llama*8B*.gguf",
        ]

        # Fast: check local ./models/ first (non-recursive)
        fast_paths = [Path.cwd() / "models", Path.cwd() / "data" / "models"]
        for base_path in fast_paths:
            if not base_path.exists():
                continue
            for pattern in patterns:
                matches = list(base_path.glob(pattern))
                if matches:
                    return str(matches[0])

        # Slow fallback: recursive search in broader locations
        logger.info("Model not in ./models/, searching other locations...")
        slow_paths = [
            Path.home() / "models",
            Path("/models"),
            Path.home() / ".cache" / "huggingface" / "hub",
        ]
        for base_path in slow_paths:
            if not base_path.exists():
                continue
            for pattern in patterns:
                matches = list(base_path.rglob(pattern))
                if matches:
                    return str(matches[0])

        return None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate text from prompt.

        Args:
            prompt: User prompt/query
            system_prompt: System prompt (optional)
            context: Additional context dict
            **kwargs: Override generation params

        Returns:
            LLMResponse with generated text
        """
        start_time = time.time()

        # Build full prompt
        full_prompt = self._build_prompt(prompt, system_prompt, context)

        # Get generation params
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)
        top_p = kwargs.get("top_p", self.config.top_p)

        try:
            if self._backend == "llama_cpp":
                return self._generate_llama_cpp(
                    full_prompt, max_tokens, temperature, top_p, start_time
                )
            else:
                return self._generate_simulation(prompt, start_time)

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return LLMResponse(
                text=f"I'm having trouble processing that. ({str(e)[:50]})",
                tokens_generated=0,
                time_seconds=time.time() - start_time,
                finish_reason="error",
            )

    def _build_prompt(
        self,
        user_prompt: str,
        system_prompt: Optional[str],
        context: Optional[Dict],
    ) -> str:
        """Build full prompt with system instructions and context."""
        parts = []

        # System prompt
        if system_prompt:
            parts.append(f"<|system|>\n{system_prompt}\n<|end|>")

        # Context
        if context:
            context_str = self._format_context(context)
            if context_str:
                parts.append(f"<|context|>\n{context_str}\n<|end|>")

        # User prompt
        parts.append(f"<|user|>\n{user_prompt}\n<|end|>")
        parts.append("<|assistant|>")

        return "\n".join(parts)

    def _format_context(self, context: Dict) -> str:
        """Format context dict as readable string."""
        lines = []

        if "current_state" in context:
            state = context["current_state"]
            lines.append(f"Current state: posture={state.get('posture', 'unknown')}, "
                        f"focus={state.get('focus', 'unknown')}")

        if "durations" in context:
            dur = context["durations"]
            if dur.get("bad_posture_seconds", 0) > 60:
                lines.append(f"Bad posture for {self._format_duration(dur['bad_posture_seconds'])}")
            if dur.get("distracted_seconds", 0) > 60:
                lines.append(f"Distracted for {self._format_duration(dur['distracted_seconds'])}")

        if "session_stats" in context:
            stats = context["session_stats"]
            lines.append(f"Session: {stats.get('good_posture_pct', 0):.0%} good posture, "
                        f"{stats.get('focused_pct', 0):.0%} focused")

        if "focus_session" in context:
            fs = context["focus_session"]
            if fs.get("active"):
                lines.append(f"Focus session: {fs.get('phase', 'unknown')}, "
                           f"{fs.get('remaining_min', 0):.0f} min remaining")

        return "\n".join(lines)

    def _format_duration(self, seconds: float) -> str:
        """Format seconds as human-readable duration."""
        if seconds < 60:
            return f"{int(seconds)} seconds"
        elif seconds < 3600:
            return f"{int(seconds / 60)} minutes"
        else:
            hours = int(seconds / 3600)
            mins = int((seconds % 3600) / 60)
            return f"{hours} hour{'s' if hours > 1 else ''} {mins} min"

    def _generate_llama_cpp(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        start_time: float,
    ) -> LLMResponse:
        """Generate using llama.cpp."""
        output = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["<|end|>", "<|user|>", "\n\n"],
        )

        text = output["choices"][0]["text"].strip()
        tokens = output["usage"]["completion_tokens"]
        finish_reason = output["choices"][0]["finish_reason"]

        return LLMResponse(
            text=text,
            tokens_generated=tokens,
            time_seconds=time.time() - start_time,
            finish_reason=finish_reason,
        )

    def _generate_simulation(self, prompt: str, start_time: float) -> LLMResponse:
        """Generate simulated response (for testing without model)."""
        prompt_lower = prompt.lower()

        # Simple rule-based responses for testing
        if "posture" in prompt_lower:
            response = "Based on what I can see, you should try sitting up straighter and keeping your shoulders back."
        elif "focus" in prompt_lower or "distract" in prompt_lower:
            response = "Let me check... It looks like you've been fairly focused. Keep up the good work!"
        elif "how long" in prompt_lower:
            response = "I've been tracking your session, but I don't have access to the full model right now."
        elif "start" in prompt_lower and "focus" in prompt_lower:
            response = "Starting a focus session. I'll track your posture and focus for the next 25 minutes."
        elif "break" in prompt_lower:
            response = "Taking a break is important! Stand up, stretch, and rest your eyes."
        elif "hello" in prompt_lower or "hi" in prompt_lower:
            response = "Hello! I'm Desk Buddy, your posture and focus assistant. How can I help you?"
        else:
            response = "I understand. Let me know if you need help with your posture or focus."

        return LLMResponse(
            text=response,
            tokens_generated=len(response.split()),
            time_seconds=time.time() - start_time,
            finish_reason="simulation",
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        """
        Chat with message history.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            **kwargs: Generation params

        Returns:
            LLMResponse with assistant's reply
        """
        # Convert to single prompt for llama.cpp backend
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(f"<|{role}|>\n{content}\n<|end|>")

        prompt_parts.append("<|assistant|>")
        full_prompt = "\n".join(prompt_parts)

        return self.generate(full_prompt, **kwargs)

    @property
    def is_available(self) -> bool:
        return self._backend is not None

    @property
    def backend(self) -> Optional[str]:
        return self._backend
