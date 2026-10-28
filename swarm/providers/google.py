"""Google AI Studio (Gemini) provider adapter."""

from typing import Any, Dict, Optional
from swarm.core.models import ChatCompletionRequest
from swarm.providers.base import BaseProvider


class GoogleProvider(BaseProvider):
    """Google AI Studio adapter using Gemini's OpenAI compatibility endpoint."""

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = DEFAULT_BASE_URL,
        enabled: bool = True,
        priority: int = 1,
        base_cooldown: float = 30.0,
        **kwargs: Any,
    ):
        super().__init__(
            name="google",
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            enabled=enabled,
            priority=priority,
            base_cooldown=base_cooldown,
            **kwargs,
        )

    def get_headers(self) -> Dict[str, str]:
        headers = super().get_headers()
        # Ensure standard bearer auth
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def format_payload(
        self, request: ChatCompletionRequest, model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        payload = super().format_payload(request, model_override)
        # Default Gemini model if none provided
        if not payload.get("model"):
            payload["model"] = "gemini-2.5-flash"
        return payload
