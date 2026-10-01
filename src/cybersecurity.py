"""
Apex Critical Infrastructure — Cybersecurity Module.

Threat detection, autonomous response, and incident management
for defense and security operations.

Classes:
    ThreatDetector: Real-time threat detection from security events.
    AutonomousResponseEngine: Automated response orchestration.
    IncidentManager: Full incident lifecycle management.
    SecurityEvent: Immutable security event record.
    Incident: Mutable incident record with status tracking.
    ResponseAction: Executable response action descriptor.
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


class ThreatLevel(Enum):
    """Severity classification for detected threats."""

    CRITICAL = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    INFO = auto()


class ThreatCategory(Enum):
    """Category of detected threat."""

    MALWARE = auto()
    INTRUSION = auto()
    DATA_EXFILTRATION = auto()
    PHISHING = auto()
    DDOS = auto()
    INSIDER_THREAT = auto()
    ZERO_DAY = auto()
    RANSOMWARE = auto()
    UNAUTHORIZED_ACCESS = auto()
    ANOMALY = auto()


class ResponseActionType(Enum):
    """Type of autonomous response action."""

    BLOCK_IP = auto()
    ISOLATE_HOST = auto()
    KILL_PROCESS = auto()
    QUARANTINE_FILE = auto()
    DISABLE_ACCOUNT = auto()
    SNAPSHOT_VM = auto()
    ALERT_SOC = auto()
    COLLECT_EVIDENCE = auto()
    ROLLBACK_CHANGE = auto()
    SCALE_RESOURCES = auto()


class IncidentStatus(Enum):
    """Lifecycle status of a security incident."""

    DETECTED = auto()
    TRIAGED = auto()
    CONTAINED = auto()
    ERADICATED = auto()
    RECOVERED = auto()
    CLOSED = auto()
    ESCALATED = auto()


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SecurityEvent:
    """Immutable security event record.

    Attributes:
        event_id: Unique event identifier.
        timestamp: UTC timestamp of event occurrence.
        source: Originating system or sensor.
        event_type: Classification of the event.
        raw_data: Original event payload.
        severity: Assessed severity level.
        metadata: Additional contextual key-value pairs.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "unknown"
    event_type: str = "generic"
    raw_data: dict[str, Any] = field(default_factory=dict)
    severity: ThreatLevel = ThreatLevel.INFO
    metadata: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 fingerprint of the event."""
        canonical = f"{self.source}:{self.event_type}:{sorted(self.raw_data.items())}"
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(slots=True)
class ResponseAction:
    """Executable response action descriptor.

    Attributes:
        action_type: Category of response action.
        target: Identifier of the target (IP, host, account, etc.).
        parameters: Action-specific parameters.
        priority: Execution priority (lower is higher).
        timeout_seconds: Maximum execution time.
        rollback_action: Optional rollback action type.
    """

    action_type: ResponseActionType
    target: str
    parameters: dict[str, Any] = field(default_factory=dict)
    priority: int = 100
    timeout_seconds: float = 30.0
    rollback_action: ResponseActionType | None = None


@dataclass(slots=True)
class Incident:
    """Mutable incident record with full lifecycle tracking.

    Attributes:
        incident_id: Unique incident identifier.
        title: Human-readable incident title.
        description: Detailed incident description.
        status: Current lifecycle status.
        severity: Current severity level.
        category: Threat category.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        assigned_to: Responsible analyst or team.
        related_events: Associated security event IDs.
        response_actions: Executed or planned response actions.
        notes: Free-form investigation notes.
    """

    incident_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    status: IncidentStatus = IncidentStatus.DETECTED
    severity: ThreatLevel = ThreatLevel.MEDIUM
    category: ThreatCategory = ThreatCategory.ANOMALY
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    assigned_to: str | None = None
    related_events: list[str] = field(default_factory=list)
    response_actions: list[ResponseAction] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def transition_to(self, new_status: IncidentStatus) -> None:
        """Transition incident to a new status, updating the timestamp.

        Args:
            new_status: Target lifecycle status.

        Raises:
            ValueError: If the transition is not valid.
        """
        valid_transitions: dict[IncidentStatus, set[IncidentStatus]] = {
            IncidentStatus.DETECTED: {
                IncidentStatus.TRIAGED,
                IncidentStatus.ESCALATED,
                IncidentStatus.CLOSED,
            },
            IncidentStatus.TRIAGED: {
                IncidentStatus.CONTAINED,
                IncidentStatus.ESCALATED,
                IncidentStatus.CLOSED,
            },
            IncidentStatus.CONTAINED: {
                IncidentStatus.ERADICATED,
                IncidentStatus.ESCALATED,
            },
            IncidentStatus.ERADICATED: {IncidentStatus.RECOVERED},
            IncidentStatus.RECOVERED: {IncidentStatus.CLOSED},
            IncidentStatus.ESCALATED: {
                IncidentStatus.TRIAGED,
                IncidentStatus.CONTAINED,
            },
            IncidentStatus.CLOSED: set(),
        }
        if new_status not in valid_transitions.get(self.status, set()):
            raise ValueError(
                f"Invalid transition: {self.status.name} -> {new_status.name}"
            )
        self.status = new_status
        self.updated_at = datetime.now(timezone.utc)
        logger.info(
            "Incident %s transitioned to %s", self.incident_id, new_status.name
        )

    def add_note(self, note: str) -> None:
        """Append an investigation note with timestamp.

        Args:
            note: Note text to append.
        """
        ts = datetime.now(timezone.utc).isoformat()
        self.notes.append(f"[{ts}] {note}")
        self.updated_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class DetectionRule(Protocol):
    """Protocol for pluggable threat detection rules."""

    def evaluate(self, event: SecurityEvent) -> ThreatLevel | None:
        """Evaluate a security event and return threat level if matched.

        Args:
            event: The security event to evaluate.

        Returns:
            ThreatLevel if the rule matches, None otherwise.
        """
        ...


class ResponseExecutor(Protocol):
    """Protocol for pluggable response action executors."""

    def execute(self, action: ResponseAction) -> bool:
        """Execute a response action.

        Args:
            action: The response action to execute.

        Returns:
            True if execution succeeded, False otherwise.
        """
        ...


# ---------------------------------------------------------------------------
# Threat Detector
# ---------------------------------------------------------------------------


class ThreatDetector:
    """Real-time threat detection engine.

    Evaluates security events against a set of detection rules and
    produces threat assessments with confidence scores.

    Attributes:
        rules: Registered detection rules.
        _event_history: Rolling window of recent events for correlation.
    """

    def __init__(self, max_history: int = 10_000) -> None:
        """Initialize the threat detector.

        Args:
            max_history: Maximum number of events to retain for correlation.
        """
        self.rules: list[DetectionRule] = []
        self._event_history: list[SecurityEvent] = []
        self._max_history = max_history

    def register_rule(self, rule: DetectionRule) -> None:
        """Register a detection rule.

        Args:
            rule: The detection rule to add.
        """
        self.rules.append(rule)
        logger.debug("Registered detection rule: %s", type(rule).__name__)

    def unregister_rule(self, rule: DetectionRule) -> None:
        """Remove a detection rule.

        Args:
            rule: The detection rule to remove.

        Raises:
            ValueError: If the rule is not registered.
        """
        try:
            self.rules.remove(rule)
        except ValueError as exc:
            raise ValueError("Rule is not registered") from exc

    def analyze(self, event: SecurityEvent) -> ThreatAssessment:
        """Analyze a security event for threats.

        Evaluates the event against all registered rules and performs
        temporal correlation with recent events.

        Args:
            event: The security event to analyze.

        Returns:
            ThreatAssessment with the highest detected threat level.
        """
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history :]

        highest_level = ThreatLevel.INFO
        matched_rules: list[str] = []

        for rule in self.rules:
            try:
                result = rule.evaluate(event)
            except Exception as exc:
                logger.warning(
                    "Detection rule %s failed: %s", type(rule).__name__, exc
                )
                continue
            if result is not None and result.value < highest_level.value:
                highest_level = result
                matched_rules.append(type(rule).__name__)

        # Temporal correlation: escalate if similar events cluster
        correlation_score = self._correlate(event)
        if correlation_score >= 5 and highest_level.value > ThreatLevel.HIGH.value:
            highest_level = ThreatLevel(
                max(ThreatLevel.CRITICAL.value, highest_level.value - 1)
            )

        return ThreatAssessment(
            event=event,
            threat_level=highest_level,
            matched_rules=matched_rules,
            correlation_score=correlation_score,
            timestamp=datetime.now(timezone.utc),
        )

    def _correlate(self, event: SecurityEvent) -> int:
        """Count similar events in the recent history window.

        Args:
            event: The event to correlate against history.

        Returns:
            Count of similar events in the rolling window.
        """
        return sum(
            1
            for e in self._event_history[-100:]
            if e.source == event.source and e.event_type == event.event_type
        )


@dataclass(frozen=True, slots=True)
class ThreatAssessment:
    """Result of threat analysis on a security event.

    Attributes:
        event: The analyzed security event.
        threat_level: Detected threat level.
        matched_rules: Names of rules that matched.
        correlation_score: Temporal correlation count.
        timestamp: Assessment timestamp.
    """

    event: SecurityEvent
    threat_level: ThreatLevel
    matched_rules: list[str]
    correlation_score: int
    timestamp: datetime


# ---------------------------------------------------------------------------
# Autonomous Response Engine
# ---------------------------------------------------------------------------


class AutonomousResponseEngine:
    """Automated response orchestration engine.

    Executes response actions based on threat assessments, with
    configurable approval gates and rollback support.

    Attributes:
        executors: Mapping of action types to executor callables.
        approval_required: Set of action types requiring human approval.
        action_log: Chronological log of executed actions.
    """

    def __init__(self) -> None:
        """Initialize the autonomous response engine."""
        self.executors: dict[ResponseActionType, Callable[[ResponseAction], bool]] = {}
        self.approval_required: set[ResponseActionType] = {
            ResponseActionType.ISOLATE_HOST,
            ResponseActionType.DISABLE_ACCOUNT,
            ResponseActionType.ROLLBACK_CHANGE,
        }
        self.action_log: list[dict[str, Any]] = []
        self._pending_approvals: dict[str, ResponseAction] = {}

    def register_executor(
        self,
        action_type: ResponseActionType,
        executor: Callable[[ResponseAction], bool],
    ) -> None:
        """Register an executor for a response action type.

        Args:
            action_type: The action type to handle.
            executor: Callable that executes the action.
        """
        self.executors[action_type] = executor
        logger.debug("Registered executor for %s", action_type.name)

    def set_approval_required(
        self, action_type: ResponseActionType, required: bool
    ) -> None:
        """Configure whether an action type requires human approval.

        Args:
            action_type: The action type to configure.
            required: True if approval is required.
        """
        if required:
            self.approval_required.add(action_type)
        else:
            self.approval_required.discard(action_type)

    def respond(
        self, assessment: ThreatAssessment, actions: list[ResponseAction]
    ) -> ResponseResult:
        """Execute response actions for a threat assessment.

        Actions are sorted by priority and executed sequentially.
        Actions requiring approval are queued for human review.

        Args:
            assessment: The threat assessment triggering response.
            actions: Planned response actions.

        Returns:
            ResponseResult with execution outcomes.
        """
        executed: list[ResponseAction] = []
        failed: list[tuple[ResponseAction, str]] = []
        pending: list[ResponseAction] = []

        sorted_actions = sorted(actions, key=lambda a: a.priority)

        for action in sorted_actions:
            if action.action_type in self.approval_required:
                approval_id = str(uuid.uuid4())
                self._pending_approvals[approval_id] = action
                pending.append(action)
                logger.info(
                    "Action %s queued for approval (id=%s)",
                    action.action_type.name,
                    approval_id,
                )
                continue

            success = self._execute_action(action)
            if success:
                executed.append(action)
            else:
                failed.append((action, "Execution failed"))

        return ResponseResult(
            assessment=assessment,
            executed=executed,
            failed=failed,
            pending_approval=pending,
            timestamp=datetime.now(timezone.utc),
        )

    def approve_action(self, approval_id: str) -> bool:
        """Approve a pending response action.

        Args:
            approval_id: The approval identifier.

        Returns:
            True if the action was found and executed.

        Raises:
            KeyError: If the approval_id is not found.
        """
        if approval_id not in self._pending_approvals:
            raise KeyError(f"Unknown approval ID: {approval_id}")
        action = self._pending_approvals.pop(approval_id)
        return self._execute_action(action)

    def _execute_action(self, action: ResponseAction) -> bool:
        """Execute a single response action.

        Args:
            action: The action to execute.

        Returns:
            True if execution succeeded.
        """
        executor = self.executors.get(action.action_type)
        if executor is None:
            logger.error(
                "No executor registered for %s", action.action_type.name
            )
            return False

        start = time.monotonic()
        try:
            success = executor(action)
        except Exception as exc:
            logger.exception(
                "Executor for %s raised: %s", action.action_type.name, exc
            )
            success = False

        elapsed = time.monotonic() - start
        self.action_log.append(
            {
                "action": action.action_type.name,
                "target": action.target,
                "success": success,
                "elapsed_seconds": elapsed,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        return success


@dataclass(frozen=True, slots=True)
class ResponseResult:
    """Result of an autonomous response execution.

    Attributes:
        assessment: The triggering threat assessment.
        executed: Successfully executed actions.
        failed: Failed actions with error messages.
        pending_approval: Actions awaiting human approval.
        timestamp: Result timestamp.
    """

    assessment: ThreatAssessment
    executed: list[ResponseAction]
    failed: list[tuple[ResponseAction, str]]
    pending_approval: list[ResponseAction]
    timestamp: datetime


# ---------------------------------------------------------------------------
# Incident Manager
# ---------------------------------------------------------------------------


class IncidentManager:
    """Full incident lifecycle management.

    Creates, tracks, and resolves security incidents with full
    audit trail and SLA monitoring.

    Attributes:
        incidents: All known incidents by ID.
        _escalation_callbacks: Callbacks invoked on escalation.
    """

    def __init__(self) -> None:
        """Initialize the incident manager."""
        self.incidents: dict[str, Incident] = {}
        self._escalation_callbacks: list[Callable[[Incident], None]] = []

    def register_escalation_callback(
        self, callback: Callable[[Incident], None]
    ) -> None:
        """Register a callback for incident escalation events.

        Args:
            callback: Callable invoked with the escalated incident.
        """
        self._escalation_callbacks.append(callback)

    def create_incident(
        self,
        title: str,
        description: str,
        severity: ThreatLevel,
        category: ThreatCategory,
        related_events: list[str] | None = None,
        assigned_to: str | None = None,
    ) -> Incident:
        """Create a new security incident.

        Args:
            title: Human-readable incident title.
            description: Detailed description.
            severity: Initial severity level.
            category: Threat category.
            related_events: Associated security event IDs.
            assigned_to: Responsible analyst or team.

        Returns:
            The newly created Incident.
        """
        incident = Incident(
            title=title,
            description=description,
            severity=severity,
            category=category,
            related_events=related_events or [],
            assigned_to=assigned_to,
        )
        self.incidents[incident.incident_id] = incident
        logger.info(
            "Created incident %s: %s (%s)",
            incident.incident_id,
            title,
            severity.name,
        )
        return incident

    def get_incident(self, incident_id: str) -> Incident:
        """Retrieve an incident by ID.

        Args:
            incident_id: The incident identifier.

        Returns:
            The matching Incident.

        Raises:
            KeyError: If the incident is not found.
        """
        try:
            return self.incidents[incident_id]
        except KeyError as exc:
            raise KeyError(f"Incident not found: {incident_id}") from exc

    def escalate(self, incident_id: str, reason: str) -> Incident:
        """Escalate an incident for senior review.

        Args:
            incident_id: The incident to escalate.
            reason: Escalation reason.

        Returns:
            The escalated Incident.

        Raises:
            KeyError: If the incident is not found.
        """
        incident = self.get_incident(incident_id)
        incident.transition_to(IncidentStatus.ESCALATED)
        incident.add_note(f"Escalated: {reason}")
        for callback in self._escalation_callbacks:
            try:
                callback(incident)
            except Exception as exc:
                logger.warning("Escalation callback failed: %s", exc)
        return incident

    def find_by_status(self, status: IncidentStatus) -> list[Incident]:
        """Find all incidents in a given status.

        Args:
            status: The status to filter by.

        Returns:
            List of matching incidents.
        """
        return [i for i in self.incidents.values() if i.status == status]

    def find_by_severity(self, severity: ThreatLevel) -> list[Incident]:
        """Find all incidents at or above a severity level.

        Args:
            severity: Minimum severity level.

        Returns:
            List of matching incidents.
        """
        return [
            i
            for i in self.incidents.values()
            if i.severity.value <= severity.value
        ]

    def open_incident_count(self) -> int:
        """Return the count of non-closed incidents.

        Returns:
            Number of active incidents.
        """
        return sum(
            1 for i in self.incidents.values() if i.status != IncidentStatus.CLOSED
        )
