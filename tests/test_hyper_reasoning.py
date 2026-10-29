"""Tests for the Ultra Hyper-Reasoning engine (TTC, Multi-Model Consensus, Invariant Extraction)."""

import pytest
from swarm.reasoning.hyper_reasoner import (
    HyperReasoner,
    ReasoningCandidate,
    AdversarialCritique,
)


@pytest.fixture
def reasoner():
    return HyperReasoner()


def test_invariant_extraction(reasoner):
    sample_text = (
        "Let S be the set of nodes.\n"
        "Theorem: For any graph G, sum of degrees == 2 * |E|.\n"
        "Invariant: Memory usage remains bounded by O(1).\n"
        "- The system state never drops below zero.\n"
    )
    invariants = reasoner.extract_invariants(sample_text)
    assert len(invariants) >= 2
    assert any("sum of degrees" in inv for inv in invariants)
    assert any("Memory usage" in inv for inv in invariants)


def test_adversarial_critique_passes_valid_candidate(reasoner):
    cand = ReasoningCandidate(
        candidate_id="c_valid",
        model="deepseek-v4-pro",
        persona="Deep Mathematical Reasoner",
        derivation=(
            "Theorem: The decoupling of inference compute from local state mutation maintains O(1) memory.\n"
            "Proof by Induction: Base case n=0 holds trivially with 0 allocated buffers.\n"
            "Inductive step: Each request is streamed via network socket and deallocated on stream close.\n"
            "Invariant: Local host memory bounded strictly at M_host < 100 MB."
        ),
        invariants=["M_host < 100 MB"],
    )
    critique = reasoner.audit_candidate(cand)
    assert critique.passed_audit is True
    assert critique.critique_score >= 80.0
    assert len(critique.weaknesses_detected) == 0


def test_adversarial_critique_flags_circular_logic_and_security(reasoner):
    cand_bad = ReasoningCandidate(
        candidate_id="c_bad",
        model="unverified-model",
        persona="Naive Coder",
        derivation="This is obviously true and trivially true. We can run eval(user_input) in a loop.",
        invariants=[],
    )
    critique = reasoner.audit_candidate(cand_bad)
    assert critique.passed_audit is False
    assert critique.critique_score < 70.0
    assert any("Security rule violation" in w for w in critique.weaknesses_detected)
    assert any("Circular justification" in w for w in critique.weaknesses_detected)


def test_consensus_calculation(reasoner):
    cand1 = ReasoningCandidate(
        candidate_id="c1",
        model="deepseek-v4-pro",
        persona="Reasoner",
        derivation="Valid formal proof with base case and inductive step.",
        invariants=["inv_1", "inv_2"],
    )
    cand2 = ReasoningCandidate(
        candidate_id="c2",
        model="qwen-3.8",
        persona="Coder",
        derivation="Valid algorithmic proof with base case and invariants.",
        invariants=["inv_1", "inv_3"],
    )
    crit1 = AdversarialCritique(candidate_id="c1", critique_score=95.0, passed_audit=True)
    crit2 = AdversarialCritique(candidate_id="c2", critique_score=90.0, passed_audit=True)

    score = reasoner.calculate_consensus([cand1, cand2], [crit1, crit2])
    assert 0.70 <= score <= 1.0


@pytest.mark.asyncio
async def test_full_hyper_reasoning_pipeline(reasoner):
    verdict = await reasoner.run_reasoning_pipeline(
        prompt="Prove that multi-provider failover prevents 429 exhaustion traps.",
        max_loops=2,
        min_consensus_threshold=0.75,
    )
    assert verdict.iterations_run >= 1
    assert verdict.consensus_score >= 0.70
    assert len(verdict.candidates) >= 3
    assert len(verdict.critiques) >= 3
    assert "Ultra Hyper-Reasoning Consensus Derivation" in verdict.primary_solution
    assert len(verdict.verified_invariants) > 0
