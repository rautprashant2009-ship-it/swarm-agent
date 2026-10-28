"""Abstract base provider adapter and circuit breaker state machine."""

from __future__ import annotations
import asyncio
from enum import Enum
import json
import logging
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional
import httpx

from swarm.core.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ChunkChoice,
    Delta,
    Choice,
    ChatCompletionMessage,
    Usage,
)
from swarm.core.exceptions import (
    ProviderError,
    ProviderQuotaExhaustedError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    ProviderAuthError,
)

logger = logging.getLogger("swarm.providers.base")

QUOTA_ERROR_PATTERNS = [
    "quota",
    "rate_limit",
    "rate limit",
    "resource_exhausted",
    "resource has been exhausted",
    "insufficient_quota",
    "exceeded your current quota",
    "allocationexhausted",
    "throttling.ratequota",
    "too many requests",
    "capacity_exceeded",
    "model_overloaded",
    "tokens per minute",
    "requests per minute",
    "requests per day",
    "concurrency limit",
    "free tier",
    "balance",
    "recharge",
]


class ProviderState(str, Enum):
    HEALTHY = "HEALTHY"
    COOLING_DOWN = "COOLING_DOWN"
    HALF_OPEN = "HALF_OPEN"
    DISABLED = "DISABLED"


class BaseProvider:
    """Base class for all upstream AI providers in the pool."""

    def __init__(
        self,
        name: str,
        api_key: str = "",
        base_url: str = "",
        enabled: bool = True,
        priority: int = 1,
        base_cooldown: float = 30.0,
        timeout: float = 60.0,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self.name = name
        self.api_key = api_key.strip() if api_key else ""
        self.base_url = base_url.rstrip("/")
        self.enabled = enabled
        self.priority = priority
        self.base_cooldown = base_cooldown
        self.timeout = timeout
        self._shared_client = client

        # State machine
        if not self.enabled or not self.api_key:
            self.state = ProviderState.DISABLED
        else:
            self.state = ProviderState.HEALTHY

        self.consecutive_failures: int = 0
        self.cooldown_until: float = 0.0
        self._lock = asyncio.Lock()

    def get_client(self) -> httpx.AsyncClient:
        if self._shared_client is not None and not self._shared_client.is_closed:
            return self._shared_client
        return httpx.AsyncClient(timeout=self.timeout)

    def is_available(self) -> bool:
        """Check if provider is available for routing according to circuit breaker state."""
        if not self.enabled or self.state == ProviderState.DISABLED:
            return False

        if self.state == ProviderState.COOLING_DOWN:
            if time.time() >= self.cooldown_until:
                self.state = ProviderState.HALF_OPEN
                logger.info(f"[{self.name}] Cooldown elapsed -> transitioning to HALF_OPEN probe state.")
                return True
            return False

        if self.state in (ProviderState.HEALTHY, ProviderState.HALF_OPEN):
            return True

        return False

    def cooldown_remaining(self) -> float:
        """Seconds remaining in current cooldown."""
        return max(0.0, self.cooldown_until - time.time())

    def record_success(self) -> None:
        """Record successful invocation, resetting circuit breaker to HEALTHY."""
        if self.state != ProviderState.DISABLED:
            self.consecutive_failures = 0
            self.state = ProviderState.HEALTHY
            self.cooldown_until = 0.0

    def record_failure(
        self,
        status_code: Optional[int],
        error_body: str,
        retry_after: Optional[str] = None,
    ) -> float:
        """Record an upstream failure and update circuit breaker state and cooldown."""
        if status_code == 401:
            self.state = ProviderState.DISABLED
            logger.warning(f"[{self.name}] Received 401 Unauthorized -> Marked DISABLED.")
            return 0.0

        self.consecutive_failures += 1

        # Check for explicit Retry-After header
        cooldown = None
        if retry_after:
            try:
                cooldown = float(retry_after)
            except ValueError:
                pass

        # Check body text for retry duration: "Please try again in 5s" or "in 12 seconds"
        if cooldown is None and error_body:
            match = re.search(r"(?:try again in|retry after|wait)\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?", error_body, re.IGNORECASE)
            if match:
                try:
                    cooldown = float(match.group(1))
                except ValueError:
                    pass

        # Check for daily quota exhaustion
        if error_body and any(w in error_body.lower() for w in ["requests per day", "rpd", "daily limit"]):
            cooldown = 3600.0  # 1 hour backoff for daily limits

        if cooldown is not None:
            cooldown = min(max(cooldown, 5.0), 3600.0)
        else:
            # Exponential backoff: base_cooldown * 2^(failures - 1), capped at 1800s
            cooldown = min(self.base_cooldown * (2 ** (self.consecutive_failures - 1)), 1800.0)

        self.state = ProviderState.COOLING_DOWN
        self.cooldown_until = time.time() + cooldown
        logger.warning(
            f"[{self.name}] Failure #{self.consecutive_failures} (code={status_code}) -> "
            f"COOLING_DOWN for {cooldown:.1f}s until {time.strftime('%H:%M:%S', time.localtime(self.cooldown_until))}"
        )
        return cooldown

    def is_quota_or_rate_limit(self, status_code: int, body_text: str) -> bool:
        """Evaluate if an upstream error matches quota exhaustion or rate limits."""
        if status_code in (429, 403, 502, 503, 504):
            return True
        body_lower = body_text.lower()
        return any(pattern in body_lower for pattern in QUOTA_ERROR_PATTERNS)

    def get_endpoint_url(self) -> str:
        """Return the target completions URL."""
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def get_headers(self) -> Dict[str, str]:
        """Construct authentication and request headers."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def format_payload(
        self, request: ChatCompletionRequest, model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """Format request payload for upstream API."""
        payload = request.model_dump(exclude_none=True)
        if model_override:
            payload["model"] = model_override
        return payload

    async def chat_completion(
        self, request: ChatCompletionRequest, model_override: Optional[str] = None
    ) -> ChatCompletionResponse:
        """Execute non-streaming chat completion with error classification."""
        url = self.get_endpoint_url()
        headers = self.get_headers()
        payload = self.format_payload(request, model_override)

        client = self.get_client()
        should_close = client != self._shared_client

        try:
            response = await client.post(url, headers=headers, json=payload, timeout=self.timeout)
            status_code = response.status_code
            body_text = response.text

            if status_code == 200:
                self.record_success()
                data = response.json()
                # Ensure model matches requested model or candidate model
                if "model" not in data or not data["model"]:
                    data["model"] = model_override or request.model
                return ChatCompletionResponse.model_validate(data)

            # Error handling
            retry_after = response.headers.get("Retry-After")
            if status_code == 401:
                self.record_failure(401, body_text)
                raise ProviderAuthError(
                    message=f"Unauthorized: Invalid API key for provider {self.name}",
                    provider_name=self.name,
                    status_code=401,
                    response_body=body_text,
                )

            if self.is_quota_or_rate_limit(status_code, body_text):
                cooldown = self.record_failure(status_code, body_text, retry_after)
                raise ProviderQuotaExhaustedError(
                    message=f"Quota exhausted or rate limit hit on provider {self.name} (HTTP {status_code}): {body_text[:200]}",
                    provider_name=self.name,
                    status_code=status_code,
                    response_body=body_text,
                    retry_after=cooldown,
                )

            # Other provider errors
            self.record_failure(status_code, body_text, retry_after)
            raise ProviderError(
                message=f"Upstream provider error from {self.name} (HTTP {status_code}): {body_text[:200]}",
                provider_name=self.name,
                status_code=status_code,
                response_body=body_text,
            )

        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as net_err:
            self.record_failure(503, str(net_err))
            raise ProviderUnavailableError(
                message=f"Connection failure to provider {self.name}: {net_err}",
                provider_name=self.name,
                status_code=503,
            ) from net_err

        finally:
            if should_close:
                await client.aclose()

    async def stream_chat_completion(
        self, request: ChatCompletionRequest, model_override: Optional[str] = None
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Execute streaming chat completion with pre-stream failover detection."""
        url = self.get_endpoint_url()
        headers = self.get_headers()
        headers["Accept"] = "text/event-stream"
        payload = self.format_payload(request, model_override)
        payload["stream"] = True

        client = self.get_client()
        should_close = client != self._shared_client

        try:
            async with client.stream("POST", url, headers=headers, json=payload, timeout=self.timeout) as resp:
                status_code = resp.status_code

                if status_code != 200:
                    body_bytes = await resp.aread()
                    body_text = body_bytes.decode("utf-8", errors="replace")
                    retry_after = resp.headers.get("Retry-After")

                    if status_code == 401:
                        self.record_failure(401, body_text)
                        raise ProviderAuthError(
                            message=f"Unauthorized: Invalid API key for provider {self.name}",
                            provider_name=self.name,
                            status_code=401,
                            response_body=body_text,
                        )

                    if self.is_quota_or_rate_limit(status_code, body_text):
                        cooldown = self.record_failure(status_code, body_text, retry_after)
                        raise ProviderQuotaExhaustedError(
                            message=f"Quota exhausted on streaming request to {self.name} (HTTP {status_code}): {body_text[:200]}",
                            provider_name=self.name,
                            status_code=status_code,
                            response_body=body_text,
                            retry_after=cooldown,
                        )

                    self.record_failure(status_code, body_text, retry_after)
                    raise ProviderError(
                        message=f"Upstream streaming error from {self.name} (HTTP {status_code}): {body_text[:200]}",
                        provider_name=self.name,
                        status_code=status_code,
                        response_body=body_text,
                    )

                # HTTP 200 OK -> Stream connected successfully
                self.record_success()

                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(data_str)
                            chunk = ChatCompletionChunk.model_validate(chunk_data)
                            yield chunk
                        except Exception as e:
                            logger.debug(f"[{self.name}] Failed to parse SSE line: {line} ({e})")
                            continue

        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as net_err:
            self.record_failure(503, str(net_err))
            raise ProviderUnavailableError(
                message=f"Network error during streaming to {self.name}: {net_err}",
                provider_name=self.name,
                status_code=503,
            ) from net_err

        finally:
            if should_close:
                await client.aclose()
