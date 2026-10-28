"""Moonshot AI (Kimi) provider adapter."""

from typing import Any, Dict, Optional
from swarm.core.models import ChatCompletionRequest
from swarm.providers.base import BaseProvider


class MoonshotProvider(BaseProvider):
    """Moonshot AI (Kimi) API provider adapter."""

    DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = DEFAULT_BASE_URL,
        enabled: bool = True,
        priority: int = 1,
        base_cooldown: float = 60.0,
        **kwargs: Any,
    ):
        super().__init__(
            name="moonshot",
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            enabled=enabled,
            priority=priority,
            base_cooldown=base_cooldown,
            **kwargs,
        )

    def is_quota_or_rate_limit(self, status_code: int, body_text: str) -> bool:
        body_lower = body_text.lower()
        if "insufficient_quota" in body_lower or "out of balance" in body_lower:
            return True
        return super().is_quota_or_rate_limit(status_code, body_text)

    def format_payload(
        self, request: ChatCompletionRequest, model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        payload = super().format_payload(request, model_override)
        if not payload.get("model"):
            payload["model"] = "moonshot-v1-32k"
        return payload
