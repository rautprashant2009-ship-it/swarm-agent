"""3-Tier Nested Feedback & Autonomous Convergence Engine (Milestone 5 / R5).

Architecture:
- Tier 1: Inner Local Test Loop (inner_test_loop.py)
  Pytest execution, traceback extraction, autonomous coder patching, 5-attempt budget.
- Tier 2: Middle Adversarial Audit Loop (middle_audit_loop.py)
  Independent GLM-5.3 Judge peer review, DeepSeek-V4-Pro Reasoner derivation,
  Qwen 3.8 Coder patch implementation before disk persistence.
- Tier 3: Outer Goal Convergence Loop (outer_goal_loop.py)
  Gemini 3.8 Flash High Chief Architect state machine with atomic checkpoint.json persistence,
  dependency resolution, and cold-boot crash recovery.
"""

from swarm.loops.inner_test_loop import (
    InnerLoopResult,
    InnerLoopRunner,
    InnerLoopState,
    InnerTestLoop,
)
from swarm.loops.middle_audit_loop import (
    AdversarialAuditEngine,
    AdversarialAuditLoop,
    AuditFinding,
    AuditFindingModel,
    AuditVerdict,
    AuditVerdictModel,
)
from swarm.loops.outer_goal_loop import (
    CheckpointData,
    CheckpointState,
    GoalConvergenceEngine,
    MilestoneState,
    OuterGoalConvergenceLoop,
)

__all__ = [
    # Tier 1
    "InnerTestLoop",
    "InnerLoopRunner",
    "InnerLoopResult",
    "InnerLoopState",
    # Tier 2
    "AdversarialAuditLoop",
    "AdversarialAuditEngine",
    "AuditFinding",
    "AuditFindingModel",
    "AuditVerdict",
    "AuditVerdictModel",
    # Tier 3
    "OuterGoalConvergenceLoop",
    "GoalConvergenceEngine",
    "CheckpointState",
    "CheckpointData",
    "MilestoneState",
]
