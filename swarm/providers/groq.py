"""Groq provider adapter."""

from typing import Any, Dict, Optional
from swarm.core.models import ChatCompletionRequest
from swarm.providers.base import BaseProvider


class GroqProvider(BaseProvider):
    """Groq API provider adapter."""

    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = DEFAULT_BASE_URL,
        enabled: bool = True,
        priority: int = 2,
        base_cooldown: float = 30.0,
        **kwargs: Any,
    ):
        super().__init__(
            name="groq",
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            enabled=enabled,
            priority=priority,
            base_cooldown=base_cooldown,
            **kwargs,
        )

    def format_payload(
        self, request: ChatCompletionRequest, model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        payload = super().format_payload(request, model_override)
        if not payload.get("model"):
            payload["model"] = "llama-3.3-70b-versatile"
        return payload
