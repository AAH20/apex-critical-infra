"""
Apex Critical Infrastructure — Incident Response Module.

Automated incident response, forensics, and recovery
for defense and security operations.

Classes:
    AutomatedResponder: Playbook-driven automated response.
    ForensicsEngine: Digital forensics collection and analysis.
    RecoveryManager: System and service recovery orchestration.
    ResponsePlaybook: Executable response playbook definition.
    ForensicArtifact: Collected forensic evidence record.
    RecoveryPlan: System recovery procedure.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class PlaybookTrigger(Enum):
    """Conditions that trigger a response playbook."""

    THREAT_DETECTED = auto()
    INCIDENT_CREATED = auto()
    IOC_MATCH = auto()
    ANOMALY_DETECTED = auto()
    MANUAL = auto()
    SCHEDULED = auto()


class PlaybookStatus(Enum):
    """Execution status of a response playbook."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    PARTIAL = auto()
    CANCELLED = auto()


class ForensicArtifactType(Enum):
    """Types of forensic artifacts."""

    MEMORY_DUMP = auto()
    DISK_IMAGE = auto()
    NETWORK_CAPTURE = auto()
    LOG_FILE = auto()
    REGISTRY_HIVE = auto()
    FILE_SYSTEM = auto()
    PROCESS_LIST = auto()
    CONNECTION_TABLE = auto()


class RecoveryStatus(Enum):
    """Status of a recovery operation."""

    PLANNED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    ROLLED_BACK = auto()
    VERIFIED = auto()


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ResponsePlaybook:
    """Executable response playbook definition.

    Attributes:
        playbook_id: Unique playbook identifier.
        name: Human-readable playbook name.
        description: Detailed description.
        trigger: Condition that triggers this playbook.
        steps: Ordered list of response step callables.
        enabled: Whether the playbook is active.
        timeout_seconds: Maximum execution time.
        metadata: Additional playbook attributes.
    """

    playbook_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    trigger: PlaybookTrigger = PlaybookTrigger.MANUAL
    steps: list[Callable[[dict[str, Any]], bool]] = field(default_factory=list)
    enabled: bool = True
    timeout_seconds: float = 300.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ForensicArtifact:
    """Collected forensic evidence record.

    Attributes:
        artifact_id: Unique artifact identifier.
        artifact_type: Classification of the artifact.
        source: Origin system or source.
        collected_at: Collection timestamp.
        content_hash: SHA-256 hash of artifact content.
        size_bytes: Size of the artifact in bytes.
        metadata: Additional artifact attributes.
        chain_of_custody: Custody transfer log.
    """

    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    artifact_type: ForensicArtifactType = ForensicArtifactType.LOG_FILE
    source: str = "unknown"
    collected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    content_hash: str = ""
    size_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    chain_of_custody: list[dict[str, Any]] = field(default_factory=list)

    def compute_hash(self, content: bytes) -> str:
        """Compute and store the SHA-256 hash of artifact content.

        Args:
            content: Raw artifact bytes.

        Returns:
            The computed hex digest.
        """
        self.content_hash = hashlib.sha256(content).hexdigest()
        self.size_bytes = len(content)
        return self.content_hash

    def add_custody_entry(self, action: str, actor: str) -> None:
        """Add a chain of custody entry.

        Args:
            action: Custody action performed.
            actor: Entity performing the action.
        """
        self.chain_of_custody.append(
            {
                "action": action,
                "actor": actor,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )


@dataclass(slots=True)
class RecoveryPlan:
    """System recovery procedure.

    Attributes:
        plan_id: Unique plan identifier.
        name: Human-readable plan name.
        description: Detailed description.
        steps: Ordered recovery step callables.
        status: Current recovery status.
        created_at: Creation timestamp.
        started_at: Execution start timestamp.
        completed_at: Execution completion timestamp.
        verification_checks: Post-recovery verification callables.
        metadata: Additional plan attributes.
    """

    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    steps: list[Callable[[dict[str, Any]], bool]] = field(default_factory=list)
    status: RecoveryStatus = RecoveryStatus.PLANNED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    verification_checks: list[Callable[[], bool]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class EvidenceCollector(Protocol):
    """Protocol for forensic evidence collectors."""

    def collect(self, source: str, artifact_type: ForensicArtifactType) -> bytes:
        """Collect evidence from a source.

        Args:
            source: Target source identifier.
            artifact_type: Type of artifact to collect.

        Returns:
            Raw artifact bytes.
        """
        ...


# ---------------------------------------------------------------------------
# Automated Responder
# ---------------------------------------------------------------------------


class AutomatedResponder:
    """Playbook-driven automated response.

    Manages response playbooks, executes them based on triggers,
    and tracks execution outcomes.

    Attributes:
        playbooks: All registered playbooks by ID.
        _execution_log: Chronological execution history.
    """

    def __init__(self) -> None:
        """Initialize the automated responder."""
        self.playbooks: dict[str, ResponsePlaybook] = {}
        self._execution_log: list[dict[str, Any]] = []

    def register_playbook(self, playbook: ResponsePlaybook) -> None:
        """Register a response playbook.

        Args:
            playbook: The playbook to register.
        """
        self.playbooks[playbook.playbook_id] = playbook
        logger.info(
            "Registered playbook %s: %s", playbook.playbook_id, playbook.name
        )

    def find_by_trigger(self, trigger: PlaybookTrigger) -> list[ResponsePlaybook]:
        """Find all enabled playbooks for a trigger.

        Args:
            trigger: The trigger condition.

        Returns:
            List of matching playbooks.
        """
        return [
            pb
            for pb in self.playbooks.values()
            if pb.trigger == trigger and pb.enabled
        ]

    def execute(
        self, playbook_id: str, context: dict[str, Any] | None = None
    ) -> PlaybookExecutionResult:
        """Execute a response playbook.

        Args:
            playbook_id: The playbook to execute.
            context: Execution context data.

        Returns:
            PlaybookExecutionResult with step outcomes.

        Raises:
            KeyError: If the playbook is not found.
        """
        playbook = self.playbooks.get(playbook_id)
        if playbook is None:
            raise KeyError(f"Playbook not found: {playbook_id}")

        context = context or {}
        step_results: list[dict[str, Any]] = []
        start = time.monotonic()
        overall_success = True

        for i, step in enumerate(playbook.steps):
            step_start = time.monotonic()
            try:
                success = step(context)
            except Exception as exc:
                logger.exception(
                    "Playbook %s step %d failed: %s", playbook_id, i, exc
                )
                success = False

            elapsed = time.monotonic() - step_start
            step_results.append(
                {
                    "step_index": i,
                    "success": success,
                    "elapsed_seconds": elapsed,
                }
            )
            if not success:
                overall_success = False
                logger.warning(
                    "Playbook %s step %d failed", playbook_id, i
                )
                break

        total_elapsed = time.monotonic() - start
        status = (
            PlaybookStatus.COMPLETED
            if overall_success
            else PlaybookStatus.PARTIAL
        )

        result = PlaybookExecutionResult(
            playbook_id=playbook_id,
            status=status,
            step_results=step_results,
            total_elapsed_seconds=total_elapsed,
            context=context,
            timestamp=datetime.now(timezone.utc),
        )

        self._execution_log.append(
            {
                "playbook_id": playbook_id,
                "status": status.name,
                "elapsed_seconds": total_elapsed,
                "timestamp": result.timestamp.isoformat(),
            }
        )

        logger.info(
            "Playbook %s executed: %s in %.2fs",
            playbook_id,
            status.name,
            total_elapsed,
        )
        return result

    def execute_by_trigger(
        self, trigger: PlaybookTrigger, context: dict[str, Any] | None = None
    ) -> list[PlaybookExecutionResult]:
        """Execute all enabled playbooks for a trigger.

        Args:
            trigger: The trigger condition.
            context: Execution context data.

        Returns:
            List of execution results.
        """
        results: list[PlaybookExecutionResult] = []
        for playbook in self.find_by_trigger(trigger):
            result = self.execute(playbook.playbook_id, context)
            results.append(result)
        return results


@dataclass(frozen=True, slots=True)
class PlaybookExecutionResult:
    """Result of a playbook execution.

    Attributes:
        playbook_id: The executed playbook ID.
        status: Final execution status.
        step_results: Per-step execution results.
        total_elapsed_seconds: Total execution time.
        context: Execution context used.
        timestamp: Completion timestamp.
    """

    playbook_id: str
    status: PlaybookStatus
    step_results: list[dict[str, Any]]
    total_elapsed_seconds: float
    context: dict[str, Any]
    timestamp: datetime


# ---------------------------------------------------------------------------
# Forensics Engine
# ---------------------------------------------------------------------------


class ForensicsEngine:
    """Digital forensics collection and analysis.

    Collects forensic artifacts from multiple sources, maintains
    chain of custody, and provides analysis capabilities.

    Attributes:
        collectors: Registered evidence collectors by name.
        artifacts: All collected artifacts by ID.
    """

    def __init__(self) -> None:
        """Initialize the forensics engine."""
        self.collectors: dict[str, EvidenceCollector] = {}
        self.artifacts: dict[str, ForensicArtifact] = {}

    def register_collector(self, name: str, collector: EvidenceCollector) -> None:
        """Register an evidence collector.

        Args:
            name: Collector identifier.
            collector: Collector implementing EvidenceCollector.
        """
        self.collectors[name] = collector
        logger.debug("Registered forensics collector: %s", name)

    def collect(
        self,
        source: str,
        artifact_type: ForensicArtifactType,
        collector_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ForensicArtifact:
        """Collect a forensic artifact.

        Args:
            source: Target source identifier.
            artifact_type: Type of artifact to collect.
            collector_name: Specific collector to use (None for auto).
            metadata: Additional artifact metadata.

        Returns:
            The collected ForensicArtifact.

        Raises:
            ValueError: If no suitable collector is found.
        """
        if collector_name is not None:
            collector = self.collectors.get(collector_name)
            if collector is None:
                raise ValueError(f"Collector not found: {collector_name}")
        else:
            # Use first available collector
            if not self.collectors:
                raise ValueError("No evidence collectors registered")
            collector = next(iter(self.collectors.values()))

        content = collector.collect(source, artifact_type)

        artifact = ForensicArtifact(
            artifact_type=artifact_type,
            source=source,
            metadata=metadata or {},
        )
        artifact.compute_hash(content)
        artifact.add_custody_entry("collected", collector_name or "auto")

        self.artifacts[artifact.artifact_id] = artifact
        logger.info(
            "Collected %s artifact from %s (hash=%s)",
            artifact_type.name,
            source,
            artifact.content_hash[:16],
        )
        return artifact

    def get_artifacts_by_type(
        self, artifact_type: ForensicArtifactType
    ) -> list[ForensicArtifact]:
        """Get all artifacts of a specific type.

        Args:
            artifact_type: The type to filter by.

        Returns:
            List of matching artifacts.
        """
        return [
            a for a in self.artifacts.values() if a.artifact_type == artifact_type
        ]

    def get_artifacts_by_source(self, source: str) -> list[ForensicArtifact]:
        """Get all artifacts from a specific source.

        Args:
            source: The source to filter by.

        Returns:
            List of matching artifacts.
        """
        return [a for a in self.artifacts.values() if a.source == source]

    def verify_integrity(self, artifact_id: str, content: bytes) -> bool:
        """Verify artifact integrity against stored hash.

        Args:
            artifact_id: The artifact to verify.
            content: Raw content to check.

        Returns:
            True if the content hash matches the stored hash.

        Raises:
            KeyError: If the artifact is not found.
        """
        artifact = self.artifacts.get(artifact_id)
        if artifact is None:
            raise KeyError(f"Artifact not found: {artifact_id}")
        computed = hashlib.sha256(content).hexdigest()
        return computed == artifact.content_hash

    def generate_report(self) -> dict[str, Any]:
        """Generate a forensics collection report.

        Returns:
            Summary dictionary of all collected artifacts.
        """
        by_type: dict[str, int] = {}
        by_source: dict[str, int] = {}
        total_size = 0

        for artifact in self.artifacts.values():
            type_key = artifact.artifact_type.name
            by_type[type_key] = by_type.get(type_key, 0) + 1
            by_source[artifact.source] = by_source.get(artifact.source, 0) + 1
            total_size += artifact.size_bytes

        return {
            "total_artifacts": len(self.artifacts),
            "total_size_bytes": total_size,
            "by_type": by_type,
            "by_source": by_source,
            "artifacts": [
                {
                    "artifact_id": a.artifact_id,
                    "type": a.artifact_type.name,
                    "source": a.source,
                    "hash": a.content_hash,
                    "size": a.size_bytes,
                    "collected_at": a.collected_at.isoformat(),
                }
                for a in self.artifacts.values()
            ],
        }


# ---------------------------------------------------------------------------
# Recovery Manager
# ---------------------------------------------------------------------------


class RecoveryManager:
    """System and service recovery orchestration.

    Manages recovery plans, executes recovery steps, and
    verifies system restoration.

    Attributes:
        plans: All registered recovery plans by ID.
        _execution_log: Chronological recovery execution history.
    """

    def __init__(self) -> None:
        """Initialize the recovery manager."""
        self.plans: dict[str, RecoveryPlan] = {}
        self._execution_log: list[dict[str, Any]] = []

    def register_plan(self, plan: RecoveryPlan) -> None:
        """Register a recovery plan.

        Args:
            plan: The recovery plan to register.
        """
        self.plans[plan.plan_id] = plan
        logger.info("Registered recovery plan %s: %s", plan.plan_id, plan.name)

    def execute_plan(
        self, plan_id: str, context: dict[str, Any] | None = None
    ) -> RecoveryResult:
        """Execute a recovery plan.

        Args:
            plan_id: The plan to execute.
            context: Execution context data.

        Returns:
            RecoveryResult with step outcomes.

        Raises:
            KeyError: If the plan is not found.
        """
        plan = self.plans.get(plan_id)
        if plan is None:
            raise KeyError(f"Recovery plan not found: {plan_id}")

        context = context or {}
        plan.status = RecoveryStatus.IN_PROGRESS
        plan.started_at = datetime.now(timezone.utc)

        step_results: list[dict[str, Any]] = []
        overall_success = True

        for i, step in enumerate(plan.steps):
            step_start = time.monotonic()
            try:
                success = step(context)
            except Exception as exc:
                logger.exception(
                    "Recovery plan %s step %d failed: %s", plan_id, i, exc
                )
                success = False

            elapsed = time.monotonic() - step_start
            step_results.append(
                {
                    "step_index": i,
                    "success": success,
                    "elapsed_seconds": elapsed,
                }
            )
            if not success:
                overall_success = False
                plan.status = RecoveryStatus.FAILED
                logger.error(
                    "Recovery plan %s failed at step %d", plan_id, i
                )
                break

        if overall_success:
            # Run verification checks
            verification_results: list[bool] = []
            for check in plan.verification_checks:
                try:
                    verification_results.append(check())
                except Exception as exc:
                    logger.warning("Recovery verification check failed: %s", exc)
                    verification_results.append(False)

            if all(verification_results):
                plan.status = RecoveryStatus.VERIFIED
            else:
                plan.status = RecoveryStatus.COMPLETED
        else:
            plan.status = RecoveryStatus.FAILED

        plan.completed_at = datetime.now(timezone.utc)

        result = RecoveryResult(
            plan_id=plan_id,
            status=plan.status,
            step_results=step_results,
            verification_results=(
                verification_results if overall_success else []
            ),
            started_at=plan.started_at,
            completed_at=plan.completed_at,
        )

        self._execution_log.append(
            {
                "plan_id": plan_id,
                "status": plan.status.name,
                "started_at": plan.started_at.isoformat() if plan.started_at else None,
                "completed_at": (
                    plan.completed_at.isoformat() if plan.completed_at else None
                ),
            }
        )

        logger.info(
            "Recovery plan %s finished: %s", plan_id, plan.status.name
        )
        return result

    def get_plans_by_status(self, status: RecoveryStatus) -> list[RecoveryPlan]:
        """Get all plans in a given status.

        Args:
            status: The status to filter by.

        Returns:
            List of matching recovery plans.
        """
        return [p for p in self.plans.values() if p.status == status]

    def rollback(self, plan_id: str) -> bool:
        """Rollback a recovery plan.

        Args:
            plan_id: The plan to rollback.

        Returns:
            True if rollback succeeded.

        Raises:
            KeyError: If the plan is not found.
        """
        plan = self.plans.get(plan_id)
        if plan is None:
            raise KeyError(f"Recovery plan not found: {plan_id}")
        plan.status = RecoveryStatus.ROLLED_BACK
        logger.info("Recovery plan %s rolled back", plan_id)
        return True


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    """Result of a recovery plan execution.

    Attributes:
        plan_id: The executed plan ID.
        status: Final recovery status.
        step_results: Per-step execution results.
        verification_results: Post-recovery verification outcomes.
        started_at: Execution start timestamp.
        completed_at: Execution completion timestamp.
    """

    plan_id: str
    status: RecoveryStatus
    step_results: list[dict[str, Any]]
    verification_results: list[bool]
    started_at: datetime | None
    completed_at: datetime | None
