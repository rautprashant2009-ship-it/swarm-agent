"""OpenRouter provider adapter."""

from typing import Any, Dict, Optional
from swarm.core.models import ChatCompletionRequest
from swarm.providers.base import BaseProvider


class OpenRouterProvider(BaseProvider):
    """OpenRouter API provider adapter with free-tier routing."""

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = DEFAULT_BASE_URL,
        enabled: bool = True,
        priority: int = 2,
        base_cooldown: float = 60.0,
        referer: str = "http://localhost:8000",
        app_title: str = "SwarmAgent",
        **kwargs: Any,
    ):
        super().__init__(
            name="openrouter",
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            enabled=enabled,
            priority=priority,
            base_cooldown=base_cooldown,
            **kwargs,
        )
        self.referer = referer
        self.app_title = app_title

    def get_headers(self) -> Dict[str, str]:
        headers = super().get_headers()
        headers["HTTP-Referer"] = self.referer
        headers["X-Title"] = self.app_title
        return headers

    def format_payload(
        self, request: ChatCompletionRequest, model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        payload = super().format_payload(request, model_override)
        if not payload.get("model"):
            payload["model"] = "meta-llama/llama-3.3-70b-instruct:free"
        return payload
