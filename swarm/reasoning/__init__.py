"""Swarm Ultra Hyper-Reasoning Package.
Provides multi-model consensus, test-time compute expansion, and adversarial verification.
"""

from swarm.reasoning.hyper_reasoner import (
    HyperReasoner,
    ReasoningCandidate,
    AdversarialCritique,
    ConsensusVerdict,
)

__all__ = [
    "HyperReasoner",
    "ReasoningCandidate",
    "AdversarialCritique",
    "ConsensusVerdict",
]
