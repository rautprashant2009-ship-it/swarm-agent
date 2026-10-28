"""Tier 3: Outer Goal Convergence Loop (State Machine & Milestone Progression).

Maintains a durable, disk-checkpointed state machine led by the Chief Architect persona
(Gemini 3.8 Flash High / swarm-architect) that tracks global milestone progression (M1 through M6),
resolves inter-milestone dependencies, performs atomic checkpoint persistence to disk (checkpoint.json),
and guarantees crash recovery and cold-boot resumption across runtime restarts.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field

logger = logging.getLogger("swarm.loops.outer_goal_loop")


class MilestoneState(BaseModel):
    """Execution status and audit evidence for an individual project milestone."""

    status: Literal["PLANNED", "IN_PROGRESS", "VERIFIED", "FAILED"] = "PLANNED"
    verified_at: Optional[str] = None
    evidence: Optional[str] = None

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


class CheckpointState(BaseModel):
    """Durable state representation stored atomically in checkpoint.json."""

    session_id: str
    project_name: str = "swarm-agent"
    last_updated_utc: str
    current_milestone: str
    milestones: Dict[str, MilestoneState]
    acceptance_criteria: Dict[str, bool]
    global_convergence_achieved: bool = False

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


class OuterGoalConvergenceLoop:
    """Tier 3 Outer Goal Convergence Loop state machine with atomic checkpointing."""

    DEFAULT_MILESTONES: List[str] = [
        "M1_GATEWAY_ROUTER",
        "M2_COMPRESSION_ENGINE",
        "M3_SYSTEM_TOOLS",
        "M4_CLI_DISTRIBUTION",
        "M5_FEEDBACK_LOOPS",
        "M6_E2E_VERIFICATION",
    ]

    MILESTONE_DEPENDENCIES: Dict[str, List[str]] = {
        "M1_GATEWAY_ROUTER": [],
        "M2_COMPRESSION_ENGINE": [],
        "M3_SYSTEM_TOOLS": [],
        "M4_CLI_DISTRIBUTION": [
            "M1_GATEWAY_ROUTER",
            "M2_COMPRESSION_ENGINE",
            "M3_SYSTEM_TOOLS",
        ],
        "M5_FEEDBACK_LOOPS": [
            "M1_GATEWAY_ROUTER",
            "M2_COMPRESSION_ENGINE",
            "M3_SYSTEM_TOOLS",
        ],
        "M6_E2E_VERIFICATION": [
            "M1_GATEWAY_ROUTER",
            "M2_COMPRESSION_ENGINE",
            "M3_SYSTEM_TOOLS",
            "M4_CLI_DISTRIBUTION",
            "M5_FEEDBACK_LOOPS",
        ],
    }

    def __init__(
        self,
        checkpoint_path: str = "checkpoint.json",
        session_id: str = "sess-default-123",
        router: Optional[Any] = None,
    ) -> None:
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.session_id = session_id
        self.router = router
        self.state: CheckpointState = self._load_or_initialize()

    def _utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _load_or_initialize(self) -> CheckpointState:
        """Load existing checkpoint from disk or initialize fresh state."""
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                loaded = CheckpointState(**data)

                # Cold boot pointer resolution: advance current_milestone to next non-verified milestone
                for m in self.DEFAULT_MILESTONES:
                    if m in loaded.milestones and loaded.milestones[m].status != "VERIFIED":
                        loaded.current_milestone = m
                        if loaded.milestones[m].status == "PLANNED":
                            loaded.milestones[m].status = "IN_PROGRESS"
                        break
                else:
                    loaded.current_milestone = "COMPLETED"
                    loaded.global_convergence_achieved = True

                logger.info(
                    f"[Tier 3] Resumed checkpoint from disk: current_milestone='{loaded.current_milestone}'"
                )
                return loaded
            except Exception as e:
                logger.warning(
                    f"[Tier 3] Failed to load checkpoint at '{self.checkpoint_path}': {e}. Initializing fresh state."
                )

        # Fresh state initialization
        milestones = {m: MilestoneState(status="PLANNED") for m in self.DEFAULT_MILESTONES}
        milestones["M1_GATEWAY_ROUTER"].status = "IN_PROGRESS"

        fresh = CheckpointState(
            session_id=self.session_id,
            project_name="swarm-agent",
            last_updated_utc=self._utc_now_iso(),
            current_milestone="M1_GATEWAY_ROUTER",
            milestones=milestones,
            acceptance_criteria={
                "ac_test_gateway": False,
                "ac_test_compression": False,
                "ac_test_tools": False,
                "ac_test_convergence_loop": False,
                "ac_repo_packaging": False,
            },
            global_convergence_achieved=False,
        )
        logger.info(f"[Tier 3] Initialized fresh checkpoint state for session '{self.session_id}'.")
        return fresh

    def check_dependencies(self, milestone: str) -> Tuple[bool, List[str]]:
        """Verify whether all prerequisite milestones for a given milestone have been VERIFIED."""
        prereqs = self.MILESTONE_DEPENDENCIES.get(milestone, [])
        missing = [
            req
            for req in prereqs
            if req in self.state.milestones and self.state.milestones[req].status != "VERIFIED"
        ]
        return (len(missing) == 0, missing)

    def mark_milestone_complete(self, milestone: str, evidence: str) -> None:
        """Mark a milestone verified, record evidence, and advance the state machine pointer."""
        if milestone in self.state.milestones:
            self.state.milestones[milestone].status = "VERIFIED"
            self.state.milestones[milestone].verified_at = self._utc_now_iso()
            self.state.milestones[milestone].evidence = evidence
            logger.info(f"[Tier 3] Milestone '{milestone}' marked VERIFIED.")

        self.state.last_updated_utc = self._utc_now_iso()

        # Advance current milestone to next non-verified milestone
        for m in self.DEFAULT_MILESTONES:
            if self.state.milestones[m].status != "VERIFIED":
                self.state.current_milestone = m
                if self.state.milestones[m].status == "PLANNED":
                    self.state.milestones[m].status = "IN_PROGRESS"
                logger.info(f"[Tier 3] Current milestone advanced to '{m}'.")
                break
        else:
            self.state.current_milestone = "COMPLETED"
            self.state.global_convergence_achieved = True
            logger.info("[Tier 3] All milestones verified! Global convergence achieved.")

    def mark_milestone_failed(self, milestone: str, error: str) -> None:
        """Mark a milestone failed with error diagnostics."""
        if milestone in self.state.milestones:
            self.state.milestones[milestone].status = "FAILED"
            self.state.milestones[milestone].evidence = error
            logger.warning(f"[Tier 3] Milestone '{milestone}' marked FAILED: {error}")

        self.state.last_updated_utc = self._utc_now_iso()

    def set_acceptance_criterion(self, criterion: str, passed: bool) -> None:
        """Record acceptance criterion verification status."""
        self.state.acceptance_criteria[criterion] = passed
        self.state.last_updated_utc = self._utc_now_iso()
        logger.info(f"[Tier 3] Acceptance criterion '{criterion}' set to {passed}.")

    def persist_checkpoint(self) -> None:
        """Atomically persist checkpoint state to disk via tmp file and atomic rename."""
        parent_dir = os.path.dirname(self.checkpoint_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        tmp_path = f"{self.checkpoint_path}.tmp_{os.getpid()}_{time.time()}"
        data_json = self.state.model_dump_json(indent=2)

        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(data_json)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, self.checkpoint_path)
        logger.info(f"[Tier 3] Checkpoint persisted atomically to '{self.checkpoint_path}'.")

    def get_state(self) -> CheckpointState:
        """Return the current checkpoint state."""
        return self.state

    def run_convergence_loop(
        self,
        verifiers: Optional[Dict[str, Callable[[], Tuple[bool, str]]]] = None,
    ) -> CheckpointState:
        """Iteratively progress through milestones using verifiers until convergence or blocker."""
        verifiers = verifiers or {}

        for milestone in self.DEFAULT_MILESTONES:
            current_status = self.state.milestones[milestone].status
            if current_status == "VERIFIED":
                continue

            can_run, missing = self.check_dependencies(milestone)
            if not can_run:
                logger.warning(
                    f"[Tier 3] Cannot advance '{milestone}': missing dependencies {missing}."
                )
                break

            verifier = verifiers.get(milestone)
            if verifier is not None:
                logger.info(f"[Tier 3] Executing verifier for '{milestone}'...")
                success, evidence = verifier()
                if success:
                    self.mark_milestone_complete(milestone, evidence=evidence)
                    self.persist_checkpoint()
                else:
                    self.mark_milestone_failed(milestone, error=evidence)
                    self.persist_checkpoint()
                    break
            else:
                logger.info(f"[Tier 3] No verifier provided for '{milestone}'. Pausing loop.")
                break

        return self.state


# Aliases for interface contracts and reference harness compatibility
CheckpointData = CheckpointState
GoalConvergenceEngine = OuterGoalConvergenceLoop

__all__ = [
    "MilestoneState",
    "CheckpointState",
    "CheckpointData",
    "OuterGoalConvergenceLoop",
    "GoalConvergenceEngine",
]
