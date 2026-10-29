"""Tests for the Ultra Hyper-Research engine (Multi-Vector Decomposition, Neutrality, Triangulation)."""

import pytest
from swarm.research.hyper_researcher import (
    HyperResearcher,
    ResearchVector,
    TriangulatedSource,
)


@pytest.fixture
def researcher():
    return HyperResearcher(neutrality_enforced=True)


def test_topic_decomposition_creates_orthogonal_vectors(researcher):
    topic = "Asynchronous distributed consensus algorithms in zero-trust environments"
    vectors = researcher.decompose_topic(topic)
    assert len(vectors) == 4
    dimensions = [v.dimension for v in vectors]
    assert any("Theoretical Foundations" in d for d in dimensions)
    assert any("Empirical Benchmarks" in d for d in dimensions)
    assert any("Systems Architecture" in d for d in dimensions)
    assert any("Security" in d for d in dimensions)


def test_source_triangulation_enforces_regional_neutrality(researcher):
    topic = "Multi-provider token gateway architectures"
    vectors = researcher.decompose_topic(topic)
    evidence = researcher.triangulate_evidence(topic, vectors)

    assert len(evidence) >= 4
    for ev in evidence:
        assert ev.credibility_score >= 0.90
        assert len(ev.finding) > 20
        # Verify region-neutral categorization
        assert ev.region_neutral_category in [
            "Academic Open-Access",
            "Empirical Benchmarks",
            "Formal Engineering Standard",
            "Global Security Audits",
        ]


def test_contradiction_resolution(researcher):
    topic = "Local frontier model inference feasibility"
    vectors = researcher.decompose_topic(topic)
    evidence = researcher.triangulate_evidence(topic, vectors)
    contradictions = researcher.resolve_contradictions(evidence)

    assert len(contradictions) >= 1
    c = contradictions[0]
    assert len(c.conflict_dimension) > 0
    assert len(c.resolution) > 0
    assert len(c.reconciliation_proof) > 0


@pytest.mark.asyncio
async def test_full_hyper_research_pipeline(researcher):
    topic = "Autonomous multi-agent swarms versus monolithic LLM architectures"
    report = await researcher.execute_deep_research(topic)

    assert report.research_topic == topic
    assert report.epistemic_confidence >= 0.85
    assert report.regional_neutrality_verified is True
    assert len(report.vectors_explored) == 4
    assert len(report.triangulated_sources) >= 4
    assert len(report.contradictions_resolved) >= 1
    assert "Ultra Hyper-Research Global Technical Report" in report.executive_synthesis
    assert report.telemetry["neutrality_check"] == "PASSED (Zero regional preference)"
