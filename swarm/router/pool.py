"""Circuit breaker and 429 quota exhaustion transparent failover pool router."""

from __future__ import annotations
import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional
import httpx

from swarm.config import GatewayConfig, TargetCandidate, get_config
from swarm.core.exceptions import (
    ModelNotFoundError,
    ProviderAuthError,
    ProviderError,
    ProviderPoolExhaustedError,
    ProviderQuotaExhaustedError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from swarm.core.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChunkChoice,
    Delta,
    ModelCard,
    ModelListResponse,
)
from swarm.providers.base import BaseProvider, ProviderState
from swarm.providers.cerebras import CerebrasProvider
from swarm.providers.dashscope import DashScopeProvider
from swarm.providers.google import GoogleProvider
from swarm.providers.groq import GroqProvider
from swarm.providers.moonshot import MoonshotProvider
from swarm.providers.openrouter import OpenRouterProvider
from swarm.providers.siliconflow import SiliconFlowProvider
from swarm.providers.zhipu import ZhipuProvider
from swarm.router.aliases import AliasResolver, CandidateTarget

logger = logging.getLogger("swarm.router.pool")


class ProviderPool:
    """Manages the pool of 8 upstream providers, executes circuit breaker transitions,
    and performs transparent failover on 429 / quota exhaustion.
    """

    def __init__(
        self,
        config: Optional[GatewayConfig] = None,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self.config = config or get_config()
        self._shared_client = client or httpx.AsyncClient(timeout=60.0)

        # Initialize the 8 provider adapters
        self.providers: Dict[str, BaseProvider] = {}
        self._init_providers()

        # Initialize alias resolver
        alias_candidates: Dict[str, List[CandidateTarget]] = {}
        for alias, targets in self.config.alias_mappings.items():
            alias_candidates[alias] = [
                CandidateTarget(t.provider, t.model) for t in targets
            ]
        self.resolver = AliasResolver(alias_candidates)

        # Telemetry
        self.total_routed_requests: int = 0
        self.total_failovers: int = 0
        self._lock = asyncio.Lock()

    def _init_providers(self) -> None:
        provider_classes = {
            "google": GoogleProvider,
            "siliconflow": SiliconFlowProvider,
            "moonshot": MoonshotProvider,
            "zhipu": ZhipuProvider,
            "dashscope": DashScopeProvider,
            "openrouter": OpenRouterProvider,
            "groq": GroqProvider,
            "cerebras": CerebrasProvider,
        }

        for p_name, p_cls in provider_classes.items():
            p_cfg = self.config.providers.get(p_name)
            if p_cfg:
                self.providers[p_name] = p_cls(
                    api_key=p_cfg.api_key,
                    base_url=p_cfg.base_url,
                    enabled=p_cfg.enabled,
                    priority=p_cfg.priority,
                    base_cooldown=p_cfg.base_cooldown_seconds,
                    client=self._shared_client,
                )
            else:
                self.providers[p_name] = p_cls(
                    client=self._shared_client,
                )

    async def aclose(self) -> None:
        """Close shared HTTP client."""
        if self._shared_client and not self._shared_client.is_closed:
            await self._shared_client.aclose()

    async def route_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Route non-streaming request across healthy candidate providers with transparent failover."""
        candidates = self.resolver.resolve(request.model)
        if not candidates:
            raise ModelNotFoundError(request.model)

        candidate_errors: List[str] = []
        min_cooldown: float = float("inf")

        for candidate in candidates:
            provider = self.providers.get(candidate.provider)
            if not provider:
                continue

            if not provider.is_available():
                cooldown_left = provider.cooldown_remaining()
                if cooldown_left > 0:
                    min_cooldown = min(min_cooldown, cooldown_left)
                candidate_errors.append(f"{candidate.provider} ({provider.state.value})")
                continue

            try:
                logger.info(
                    f"Routing request for '{request.model}' to [{candidate.provider}] with model '{candidate.model}'"
                )
                response = await provider.chat_completion(request, model_override=candidate.model)
                async with self._lock:
                    self.total_routed_requests += 1
                return response

            except (
                ProviderQuotaExhaustedError,
                ProviderRateLimitError,
                ProviderUnavailableError,
            ) as e:
                async with self._lock:
                    self.total_failovers += 1
                cooldown_left = provider.cooldown_remaining()
                if cooldown_left > 0:
                    min_cooldown = min(min_cooldown, cooldown_left)
                candidate_errors.append(f"{candidate.provider}: HTTP {e.status_code} - {e.message[:80]}")
                logger.warning(
                    f"[FAILOVER] Provider '{candidate.provider}' failed ({e}). Attempting next candidate in pool..."
                )
                continue

            except ProviderAuthError as e:
                candidate_errors.append(f"{candidate.provider}: 401 Unauthorized")
                logger.error(f"[AUTH_ERROR] Provider '{candidate.provider}' disabled due to 401 Unauthorized.")
                continue

            except ProviderError as e:
                async with self._lock:
                    self.total_failovers += 1
                candidate_errors.append(f"{candidate.provider}: Error - {e.message[:80]}")
                logger.warning(f"[FAILOVER] Provider '{candidate.provider}' error ({e}). Failing over...")
                continue

        # If all candidates exhausted
        reset_time = min_cooldown if min_cooldown != float("inf") else 30.0
        err_msg = (
            f"All providers configured in pool for alias '{request.model}' are currently exhausted or cooling down. "
            f"Failures: [{'; '.join(candidate_errors)}]. Next reset in {int(reset_time)}s."
        )
        raise ProviderPoolExhaustedError(
            message=err_msg,
            model=request.model,
            candidate_errors=candidate_errors,
            reset_in=reset_time,
        )

    async def route_stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Route streaming request across healthy candidate providers with pre-stream transparent failover."""
        candidates = self.resolver.resolve(request.model)
        if not candidates:
            raise ModelNotFoundError(request.model)

        candidate_errors: List[str] = []
        min_cooldown: float = float("inf")

        for candidate in candidates:
            provider = self.providers.get(candidate.provider)
            if not provider:
                continue

            if not provider.is_available():
                cooldown_left = provider.cooldown_remaining()
                if cooldown_left > 0:
                    min_cooldown = min(min_cooldown, cooldown_left)
                candidate_errors.append(f"{candidate.provider} ({provider.state.value})")
                continue

            logger.info(
                f"Routing streaming request for '{request.model}' to [{candidate.provider}] with model '{candidate.model}'"
            )

            stream_iter = provider.stream_chat_completion(request, model_override=candidate.model)
            try:
                # Probe the first chunk before committing stream to client
                first_chunk = await stream_iter.__anext__()
            except StopAsyncIteration:
                # Clean empty stream
                async with self._lock:
                    self.total_routed_requests += 1
                return
            except (
                ProviderQuotaExhaustedError,
                ProviderRateLimitError,
                ProviderUnavailableError,
            ) as e:
                async with self._lock:
                    self.total_failovers += 1
                cooldown_left = provider.cooldown_remaining()
                if cooldown_left > 0:
                    min_cooldown = min(min_cooldown, cooldown_left)
                candidate_errors.append(f"{candidate.provider}: HTTP {e.status_code}")
                logger.warning(
                    f"[STREAM_FAILOVER] Provider '{candidate.provider}' failed on initial stream probe: {e}. Switching candidate..."
                )
                continue
            except ProviderAuthError:
                candidate_errors.append(f"{candidate.provider}: 401 Unauthorized")
                continue
            except Exception as e:
                async with self._lock:
                    self.total_failovers += 1
                candidate_errors.append(f"{candidate.provider}: {e}")
                logger.warning(f"[STREAM_FAILOVER] Provider '{candidate.provider}' probe error: {e}. Switching...")
                continue

            # Connection established! Yield first chunk and stream the rest
            async with self._lock:
                self.total_routed_requests += 1

            yield first_chunk
            try:
                async for chunk in stream_iter:
                    yield chunk
            except Exception as mid_stream_err:
                logger.error(f"[{candidate.provider}] Mid-stream disconnection: {mid_stream_err}")
                provider.record_failure(500, str(mid_stream_err))
                # Emit terminal error chunk
                yield ChatCompletionChunk(
                    model=request.model,
                    choices=[
                        ChunkChoice(
                            index=0,
                            delta=Delta(),
                            finish_reason="error",
                        )
                    ],
                )
            return

        # If all candidates exhausted
        reset_time = min_cooldown if min_cooldown != float("inf") else 30.0
        err_msg = (
            f"All providers configured in streaming pool for alias '{request.model}' are currently exhausted or cooling down. "
            f"Failures: [{'; '.join(candidate_errors)}]. Next reset in {int(reset_time)}s."
        )
        raise ProviderPoolExhaustedError(
            message=err_msg,
            model=request.model,
            candidate_errors=candidate_errors,
            reset_in=reset_time,
        )

    async def dispatch_role(
        self, role: str, messages: List[ChatMessage]
    ) -> ChatCompletionResponse:
        """Interface contract for swarm loops: dispatch directly by swarm role."""
        request = ChatCompletionRequest(model=role, messages=messages)
        return await self.route_chat_completion(request)

    def get_health(self) -> Dict[str, Any]:
        """Telemetry and health check diagnostics."""
        providers_status: Dict[str, Any] = {}
        any_healthy = False

        for name, provider in self.providers.items():
            providers_status[name] = {
                "state": provider.state.value,
                "failures": provider.consecutive_failures,
                "cooldown_remaining": int(provider.cooldown_remaining()),
            }
            if provider.state in (ProviderState.HEALTHY, ProviderState.HALF_OPEN):
                any_healthy = True

        return {
            "status": "healthy" if any_healthy else "degraded",
            "gateway": "swarm-router-v1",
            "providers": providers_status,
            "total_routed_requests": self.total_routed_requests,
            "total_failovers": self.total_failovers,
        }

    def get_models(self) -> ModelListResponse:
        """OpenAI-compatible models catalog listing."""
        catalog = self.resolver.get_catalog()
        model_cards = [
            ModelCard(id=model_id, created=1730000000, owned_by="swarm-router")
            for model_id in catalog
        ]
        return ModelListResponse(data=model_cards)


# Alias for interface contract in PROJECT.md
SwarmRouter = ProviderPool
