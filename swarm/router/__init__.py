"""Router package for Swarm Agent multi-provider pool."""

from swarm.router.aliases import (
    AliasResolver,
    CandidateTarget,
    DEFAULT_ALIAS_MAP,
    ROLE_TO_ALIAS,
    KNOWN_PROVIDER_MODELS,
)
from swarm.router.pool import ProviderPool, SwarmRouter

__all__ = [
    "AliasResolver",
    "CandidateTarget",
    "DEFAULT_ALIAS_MAP",
    "ROLE_TO_ALIAS",
    "KNOWN_PROVIDER_MODELS",
    "ProviderPool",
    "SwarmRouter",
]
