"""Standardized exception hierarchy for the Swarm Agent multi-provider gateway."""

from typing import Optional


class SwarmError(Exception):
    """Base exception for all swarm-agent errors."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ProviderError(SwarmError):
    """Base exception for upstream provider errors."""
    def __init__(
        self,
        message: str,
        provider_name: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
    ):
        super().__init__(message)
        self.provider_name = provider_name
        self.status_code = status_code
        self.response_body = response_body

    def __str__(self) -> str:
        code_str = f" [HTTP {self.status_code}]" if self.status_code else ""
        return f"[{self.provider_name}]{code_str}: {self.message}"


class ProviderQuotaExhaustedError(ProviderError):
    """Raised when an upstream provider has exhausted its rate limit, TPM/RPM, or free token quota."""
    def __init__(
        self,
        message: str,
        provider_name: str,
        status_code: Optional[int] = 429,
        response_body: Optional[str] = None,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message, provider_name, status_code, response_body)
        self.retry_after = retry_after


class ProviderRateLimitError(ProviderQuotaExhaustedError):
    """Alias for rate limiting (HTTP 429)."""
    pass


class ProviderUnavailableError(ProviderError):
    """Raised when an upstream provider experiences 502/503/504 errors or connection timeouts."""
    def __init__(
        self,
        message: str,
        provider_name: str,
        status_code: Optional[int] = 503,
        response_body: Optional[str] = None,
    ):
        super().__init__(message, provider_name, status_code, response_body)


class ProviderAuthError(ProviderError):
    """Raised when an upstream provider returns 401 Unauthorized (invalid or expired API key)."""
    def __init__(
        self,
        message: str,
        provider_name: str,
        status_code: Optional[int] = 401,
        response_body: Optional[str] = None,
    ):
        super().__init__(message, provider_name, status_code, response_body)


class ProviderPoolExhaustedError(SwarmError):
    """Raised when all candidate providers in the pool for a requested model/alias fail or are cooling down."""
    def __init__(
        self,
        message: str,
        model: str,
        candidate_errors: Optional[list] = None,
        reset_in: Optional[float] = None,
    ):
        super().__init__(message)
        self.model = model
        self.candidate_errors = candidate_errors or []
        self.reset_in = reset_in


class ModelNotFoundError(SwarmError):
    """Raised when a requested model or role cannot be resolved to any provider candidate."""
    def __init__(self, model: str):
        super().__init__(f"Model or role alias '{model}' could not be resolved.")
        self.model = model
