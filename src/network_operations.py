"""Apex Critical Infrastructure — Network Operations.

Self-healing networks, predictive maintenance, and fault detection
for telecom and network infrastructure.

This module provides:
- NetworkNode: Represents a network element (base station, router, switch).
- NetworkTopology: Manages the graph of interconnected nodes.
- FaultDetector: Detects anomalies and faults from telemetry streams.
- PredictiveMaintenance: Predicts component failures before they occur.
- SelfHealingEngine: Automatically remediates detected faults.
- NetworkOperations: High-level orchestrator tying everything together.

All classes are fully typed, documented, and raise explicit exceptions
on invalid input or unrecoverable states.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import math
import random
import statistics
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Coroutine,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NetworkOperationsError(Exception):
    """Base exception for all network operations errors."""


class NodeNotFoundError(NetworkOperationsError):
    """Raised when a referenced node does not exist in the topology."""


class DuplicateNodeError(NetworkOperationsError):
    """Raised when attempting to add a node that already exists."""


class FaultDetectionError(NetworkOperationsError):
    """Raised when fault detection encounters an unrecoverable error."""


class SelfHealingError(NetworkOperationsError):
    """Raised when self-healing remediation fails."""


class InvalidTelemetryError(NetworkOperationsError):
    """Raised when telemetry data is malformed or out of expected range."""


class MaintenancePredictionError(NetworkOperationsError):
    """Raised when predictive maintenance cannot produce a valid prediction."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class NodeStatus(enum.Enum):
    """Operational status of a network node."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class NodeRole(enum.Enum):
    """Functional role of a node in the network."""

    BASE_STATION = "base_station"
    ROUTER = "router"
    SWITCH = "switch"
    GATEWAY = "gateway"
    EDGE_SERVER = "edge_server"
    CORE = "core"


class FaultSeverity(enum.Enum):
    """Severity classification for detected faults."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class RemediationAction(enum.Enum):
    """Possible self-healing remediation actions."""

    RESTART_SERVICE = "restart_service"
    FAILOVER = "failover"
    REROUTE_TRAFFIC = "reroute_traffic"
    SCALE_UP = "scale_up"
    ISOLATE_NODE = "isolate_node"
    REDUCE_LOAD = "reduce_load"
    NOTIFY_OPERATOR = "notify_operator"


class MaintenanceUrgency(enum.Enum):
    """Urgency level for predictive maintenance recommendations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TelemetryReading:
    """A single telemetry reading from a network node.

    Attributes:
        node_id: Identifier of the originating node.
        timestamp: Unix epoch timestamp of the reading.
        metric_name: Name of the measured metric (e.g. "cpu_utilization").
        value: Numeric value of the metric.
        unit: Optional unit string (e.g. "percent", "ms").
    """

    node_id: str
    timestamp: float
    metric_name: str
    value: float
    unit: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.node_id:
            raise InvalidTelemetryError("node_id must be non-empty")
        if self.timestamp < 0:
            raise InvalidTelemetryError("timestamp must be non-negative")
        if not self.metric_name:
            raise InvalidTelemetryError("metric_name must be non-empty")
        if math.isnan(self.value) or math.isinf(self.value):
            raise InvalidTelemetryError("value must be a finite number")


@dataclass(frozen=True, slots=True)
class FaultEvent:
    """A detected fault in the network.

    Attributes:
        fault_id: Unique identifier for this fault.
        node_id: Node where the fault was detected.
        severity: Severity level of the fault.
        description: Human-readable description.
        timestamp: When the fault was detected.
        metric_name: The metric that triggered the fault.
        observed_value: The value that was observed.
        threshold: The threshold that was breached.
        acknowledged: Whether an operator has acknowledged this fault.
    """

    fault_id: str
    node_id: str
    severity: FaultSeverity
    description: str
    timestamp: float
    metric_name: str
    observed_value: float
    threshold: float
    acknowledged: bool = False


@dataclass(frozen=True, slots=True)
class MaintenancePrediction:
    """A predictive maintenance recommendation.

    Attributes:
        node_id: Node the prediction applies to.
        component: Component name (e.g. "power_supply", "fan").
        failure_probability: Estimated probability of failure (0.0–1.0).
        estimated_time_to_failure_hours: Estimated hours until failure.
        urgency: Urgency level for maintenance.
        recommended_action: Suggested maintenance action.
        confidence: Model confidence in this prediction (0.0–1.0).
    """

    node_id: str
    component: str
    failure_probability: float
    estimated_time_to_failure_hours: float
    urgency: MaintenanceUrgency
    recommended_action: str
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.failure_probability <= 1.0:
            raise MaintenancePredictionError(
                f"failure_probability must be in [0, 1], got {self.failure_probability}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise MaintenancePredictionError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )
        if self.estimated_time_to_failure_hours < 0:
            raise MaintenancePredictionError(
                "estimated_time_to_failure_hours must be non-negative"
            )


@dataclass(frozen=True, slots=True)
class RemediationResult:
    """Result of a self-healing remediation action.

    Attributes:
        success: Whether the remediation succeeded.
        action: The action that was attempted.
        node_id: Target node.
        message: Human-readable result message.
        timestamp: When the remediation was executed.
    """

    success: bool
    action: RemediationAction
    node_id: str
    message: str
    timestamp: float


@dataclass(frozen=True, slots=True)
class HealthScore:
    """Composite health score for a network node.

    Attributes:
        node_id: The node this score applies to.
        score: Overall health score (0.0–100.0, higher is better).
        component_scores: Breakdown by component.
        timestamp: When the score was computed.
    """

    node_id: str
    score: float
    component_scores: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 100.0:
            raise ValueError(f"score must be in [0, 100], got {self.score}")


# ---------------------------------------------------------------------------
# NetworkNode
# ---------------------------------------------------------------------------


class NetworkNode:
    """Represents a single network element with telemetry and health tracking.

    A NetworkNode maintains a rolling window of recent telemetry readings,
    tracks its operational status, and records fault history.

    Attributes:
        node_id: Unique identifier for this node.
        role: Functional role in the network.
        status: Current operational status.
        max_telemetry_history: Maximum number of telemetry readings to retain.
    """

    def __init__(
        self,
        node_id: str,
        role: NodeRole = NodeRole.ROUTER,
        max_telemetry_history: int = 1000,
    ) -> None:
        """Initialize a NetworkNode.

        Args:
            node_id: Unique identifier (must be non-empty).
            role: Functional role of this node.
            max_telemetry_history: Cap on stored telemetry readings.

        Raises:
            ValueError: If node_id is empty or max_telemetry_history <= 0.
        """
        if not node_id:
            raise ValueError("node_id must be non-empty")
        if max_telemetry_history <= 0:
            raise ValueError("max_telemetry_history must be positive")

        self._node_id: str = node_id
        self._role: NodeRole = role
        self._status: NodeStatus = NodeStatus.HEALTHY
        self._max_history: int = max_telemetry_history
        self._telemetry: Deque[TelemetryReading] = deque(maxlen=max_telemetry_history)
        self._fault_history: List[FaultEvent] = []
        self._metadata: Dict[str, Any] = {}
        self._created_at: float = time.time()
        self._last_updated: float = time.time()

    @property
    def node_id(self) -> str:
        """Unique identifier for this node."""
        return self._node_id

    @property
    def role(self) -> NodeRole:
        """Functional role of this node."""
        return self._role

    @property
    def status(self) -> NodeStatus:
        """Current operational status."""
        return self._status

    @status.setter
    def status(self, value: NodeStatus) -> None:
        if not isinstance(value, NodeStatus):
            raise TypeError(f"status must be NodeStatus, got {type(value).__name__}")
        self._status = value
        self._last_updated = time.time()

    @property
    def metadata(self) -> Dict[str, Any]:
        """Mutable metadata dictionary for this node."""
        return self._metadata

    @property
    def created_at(self) -> float:
        """Creation timestamp."""
        return self._created_at

    @property
    def last_updated(self) -> float:
        """Last update timestamp."""
        return self._last_updated

    @property
    def fault_history(self) -> Tuple[FaultEvent, ...]:
        """Immutable view of fault history."""
        return tuple(self._fault_history)

    def add_telemetry(self, reading: TelemetryReading) -> None:
        """Add a telemetry reading to this node's history.

        Args:
            reading: The telemetry reading to add.

        Raises:
            InvalidTelemetryError: If reading.node_id does not match this node.
        """
        if reading.node_id != self._node_id:
            raise InvalidTelemetryError(
                f"Reading node_id '{reading.node_id}' does not match "
                f"this node '{self._node_id}'"
            )
        self._telemetry.append(reading)
        self._last_updated = time.time()

    def get_recent_telemetry(
        self,
        metric_name: Optional[str] = None,
        count: int = 100,
    ) -> List[TelemetryReading]:
        """Get recent telemetry readings, optionally filtered by metric.

        Args:
            metric_name: If provided, only return readings for this metric.
            count: Maximum number of readings to return.

        Returns:
            List of telemetry readings, most recent first.
        """
        readings = list(self._telemetry)
        if metric_name is not None:
            readings = [r for r in readings if r.metric_name == metric_name]
        return readings[-count:][::-1]

    def get_metric_statistics(self, metric_name: str) -> Dict[str, float]:
        """Compute statistics for a specific metric.

        Args:
            metric_name: The metric to analyze.

        Returns:
            Dictionary with keys: count, mean, stdev, min, max, latest.

        Raises:
            InvalidTelemetryError: If no readings exist for the metric.
        """
        values = [r.value for r in self._telemetry if r.metric_name == metric_name]
        if not values:
            raise InvalidTelemetryError(
                f"No telemetry readings for metric '{metric_name}' on node '{self._node_id}'"
            )
        return {
            "count": float(len(values)),
            "mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "latest": values[-1],
        }

    def record_fault(self, fault: FaultEvent) -> None:
        """Record a fault event in this node's history.

        Args:
            fault: The fault event to record.
        """
        self._fault_history.append(fault)
        if fault.severity in (FaultSeverity.CRITICAL, FaultSeverity.EMERGENCY):
            self._status = NodeStatus.UNHEALTHY
        elif fault.severity == FaultSeverity.WARNING:
            if self._status == NodeStatus.HEALTHY:
                self._status = NodeStatus.DEGRADED

    def clear_faults(self) -> None:
        """Clear all fault history and reset status to HEALTHY."""
        self._fault_history.clear()
        self._status = NodeStatus.HEALTHY
        self._last_updated = time.time()

    def uptime_seconds(self) -> float:
        """Return uptime in seconds since creation."""
        return time.time() - self._created_at

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this node to a dictionary."""
        return {
            "node_id": self._node_id,
            "role": self._role.value,
            "status": self._status.value,
            "created_at": self._created_at,
            "last_updated": self._last_updated,
            "uptime_seconds": self.uptime_seconds(),
            "telemetry_count": len(self._telemetry),
            "fault_count": len(self._fault_history),
            "metadata": dict(self._metadata),
        }

    def __repr__(self) -> str:
        return (
            f"NetworkNode(node_id={self._node_id!r}, role={self._role.value!r}, "
            f"status={self._status.value!r})"
        )


# ---------------------------------------------------------------------------
# NetworkTopology
# ---------------------------------------------------------------------------


class NetworkTopology:
    """Manages the graph of interconnected network nodes.

    Supports adding/removing nodes, establishing links, finding paths,
    and querying neighbors. The topology is undirected by default but
    can represent directed links as well.

    Attributes:
        nodes: Mapping of node_id to NetworkNode.
        links: Set of (node_id_a, node_id_b) tuples representing edges.
    """

    def __init__(self) -> None:
        """Initialize an empty network topology."""
        self._nodes: Dict[str, NetworkNode] = {}
        self._links: Set[Tuple[str, str]] = set()
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)

    @property
    def nodes(self) -> Dict[str, NetworkNode]:
        """Read-only view of nodes in the topology."""
        return dict(self._nodes)

    @property
    def links(self) -> Set[Tuple[str, str]]:
        """Read-only view of links in the topology."""
        return set(self._links)

    @property
    def node_count(self) -> int:
        """Number of nodes in the topology."""
        return len(self._nodes)

    @property
    def link_count(self) -> int:
        """Number of links in the topology."""
        return len(self._links)

    def add_node(self, node: NetworkNode) -> None:
        """Add a node to the topology.

        Args:
            node: The NetworkNode to add.

        Raises:
            DuplicateNodeError: If a node with the same ID already exists.
        """
        if node.node_id in self._nodes:
            raise DuplicateNodeError(
                f"Node '{node.node_id}' already exists in topology"
            )
        self._nodes[node.node_id] = node

    def remove_node(self, node_id: str) -> NetworkNode:
        """Remove a node and all its links from the topology.

        Args:
            node_id: The node to remove.

        Returns:
            The removed NetworkNode.

        Raises:
            NodeNotFoundError: If the node does not exist.
        """
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Node '{node_id}' not found in topology")
        # Remove all links involving this node
        neighbors = list(self._adjacency[node_id])
        for neighbor in neighbors:
            self.remove_link(node_id, neighbor)
        return self._nodes.pop(node_id)

    def get_node(self, node_id: str) -> NetworkNode:
        """Get a node by ID.

        Args:
            node_id: The node identifier.

        Returns:
            The NetworkNode.

        Raises:
            NodeNotFoundError: If the node does not exist.
        """
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Node '{node_id}' not found in topology")
        return self._nodes[node_id]

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the topology."""
        return node_id in self._nodes

    def add_link(self, node_id_a: str, node_id_b: str) -> None:
        """Add a bidirectional link between two nodes.

        Args:
            node_id_a: First node.
            node_id_b: Second node.

        Raises:
            NodeNotFoundError: If either node does not exist.
            ValueError: If attempting to link a node to itself.
        """
        if node_id_a == node_id_b:
            raise ValueError("Cannot link a node to itself")
        self._validate_nodes_exist(node_id_a, node_id_b)
        edge = (node_id_a, node_id_b)
        self._links.add(edge)
        self._adjacency[node_id_a].add(node_id_b)
        self._adjacency[node_id_b].add(node_id_a)

    def remove_link(self, node_id_a: str, node_id_b: str) -> None:
        """Remove a link between two nodes.

        Args:
            node_id_a: First node.
            node_id_b: Second node.

        Raises:
            NodeNotFoundError: If either node does not exist.
        """
        self._validate_nodes_exist(node_id_a, node_id_b)
        edge = (node_id_a, node_id_b)
        self._links.discard(edge)
        self._adjacency[node_id_a].discard(node_id_b)
        self._adjacency[node_id_b].discard(node_id_a)

    def get_neighbors(self, node_id: str) -> Set[str]:
        """Get the set of neighbor node IDs.

        Args:
            node_id: The node to query.

        Returns:
            Set of neighboring node IDs.

        Raises:
            NodeNotFoundError: If the node does not exist.
        """
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Node '{node_id}' not found in topology")
        return set(self._adjacency[node_id])

    def find_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find the shortest path between two nodes using BFS.

        Args:
            source: Starting node ID.
            target: Destination node ID.

        Returns:
            List of node IDs forming the path, or None if no path exists.

        Raises:
            NodeNotFoundError: If either node does not exist.
        """
        self._validate_nodes_exist(source, target)
        if source == target:
            return [source]

        visited: Set[str] = {source}
        queue: Deque[Tuple[str, List[str]]] = deque([(source, [source])])

        while queue:
            current, path = queue.popleft()
            for neighbor in self._adjacency[current]:
                if neighbor == target:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def get_connected_components(self) -> List[Set[str]]:
        """Find all connected components in the topology.

        Returns:
            List of sets, each containing node IDs in one component.
        """
        unvisited = set(self._nodes.keys())
        components: List[Set[str]] = []

        while unvisited:
            start = unvisited.pop()
            component = {start}
            queue: Deque[str] = deque([start])
            while queue:
                current = queue.popleft()
                for neighbor in self._adjacency[current]:
                    if neighbor in unvisited:
                        unvisited.remove(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
        return components

    def get_critical_nodes(self) -> List[str]:
        """Identify articulation points (nodes whose removal disconnects the graph).

        Uses Tarjan's algorithm for finding articulation points.

        Returns:
            List of node IDs that are articulation points.
        """
        discovery: Dict[str, int] = {}
        low: Dict[str, int] = {}
        parent: Dict[str, Optional[str]] = {}
        articulation_points: Set[str] = set()
        timer = 0

        def _dfs(u: str) -> None:
            nonlocal timer
            children = 0
            discovery[u] = low[u] = timer
            timer += 1
            for v in self._adjacency[u]:
                if v not in discovery:
                    children += 1
                    parent[v] = u
                    _dfs(v)
                    low[u] = min(low[u], low[v])
                    if parent[u] is None and children > 1:
                        articulation_points.add(u)
                    if parent[u] is not None and low[v] >= discovery[u]:
                        articulation_points.add(u)
                elif v != parent.get(u):
                    low[u] = min(low[u], discovery[v])

        for node_id in self._nodes:
            if node_id not in discovery:
                parent[node_id] = None
                _dfs(node_id)

        return list(articulation_points)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the topology to a dictionary."""
        return {
            "node_count": self.node_count,
            "link_count": self.link_count,
            "nodes": {nid: node.to_dict() for nid, node in self._nodes.items()},
            "links": [list(link) for link in self._links],
        }

    def _validate_nodes_exist(self, *node_ids: str) -> None:
        """Validate that all given node IDs exist in the topology."""
        for nid in node_ids:
            if nid not in self._nodes:
                raise NodeNotFoundError(f"Node '{nid}' not found in topology")

    def __repr__(self) -> str:
        return (
            f"NetworkTopology(nodes={self.node_count}, links={self.link_count})"
        )


# ---------------------------------------------------------------------------
# FaultDetector
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ThresholdRule:
    """A threshold-based fault detection rule.

    Attributes:
        metric_name: The metric to monitor.
        warning_threshold: Value above which a warning is raised.
        critical_threshold: Value above which a critical fault is raised.
        emergency_threshold: Value above which an emergency is raised.
        comparison: "greater" or "less" — direction of comparison.
        description: Human-readable description of the rule.
    """

    metric_name: str
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: float
    comparison: str = "greater"
    description: str = ""

    def __post_init__(self) -> None:
        if self.comparison not in ("greater", "less"):
            raise ValueError("comparison must be 'greater' or 'less'")
        if self.comparison == "greater":
            if not (
                self.warning_threshold
                <= self.critical_threshold
                <= self.emergency_threshold
            ):
                raise ValueError(
                    "For 'greater' comparison, thresholds must be ascending: "
                    "warning <= critical <= emergency"
                )
        else:
            if not (
                self.warning_threshold
                >= self.critical_threshold
                >= self.emergency_threshold
            ):
                raise ValueError(
                    "For 'less' comparison, thresholds must be descending: "
                    "warning >= critical >= emergency"
                )


class FaultDetector:
    """Detects faults from telemetry streams using configurable rules.

    Supports threshold-based rules and statistical anomaly detection
    (z-score based). Detected faults are emitted as FaultEvent objects.

    Attributes:
        rules: List of ThresholdRule objects for threshold-based detection.
        z_score_threshold: Z-score threshold for statistical anomaly detection.
        anomaly_window_size: Number of recent readings to use for anomaly detection.
    """

    def __init__(
        self,
        rules: Optional[List[ThresholdRule]] = None,
        z_score_threshold: float = 3.0,
        anomaly_window_size: int = 50,
    ) -> None:
        """Initialize the FaultDetector.

        Args:
            rules: Threshold rules for fault detection.
            z_score_threshold: Z-score above which a reading is anomalous.
            anomaly_window_size: Window size for statistical anomaly detection.

        Raises:
            ValueError: If z_score_threshold <= 0 or anomaly_window_size <= 0.
        """
        if z_score_threshold <= 0:
            raise ValueError("z_score_threshold must be positive")
        if anomaly_window_size <= 0:
            raise ValueError("anomaly_window_size must be positive")

        self._rules: List[ThresholdRule] = list(rules) if rules else []
        self._z_score_threshold: float = z_score_threshold
        self._anomaly_window_size: int = anomaly_window_size
        self._fault_counter: int = 0
        self._detected_faults: List[FaultEvent] = []
        self._handlers: List[Callable[[FaultEvent], None]] = []

    @property
    def rules(self) -> Tuple[ThresholdRule, ...]:
        """Current threshold rules."""
        return tuple(self._rules)

    @property
    def detected_faults(self) -> Tuple[FaultEvent, ...]:
        """All detected faults."""
        return tuple(self._detected_faults)

    def add_rule(self, rule: ThresholdRule) -> None:
        """Add a threshold rule.

        Args:
            rule: The rule to add.
        """
        self._rules.append(rule)

    def remove_rule(self, metric_name: str) -> bool:
        """Remove all rules for a given metric.

        Args:
            metric_name: The metric to remove rules for.

        Returns:
            True if any rules were removed.
        """
        original_len = len(self._rules)
        self._rules = [r for r in self._rules if r.metric_name != metric_name]
        return len(self._rules) < original_len

    def add_handler(self, handler: Callable[[FaultEvent], None]) -> None:
        """Register a callback for fault events.

        Args:
            handler: Callable that receives a FaultEvent.
        """
        self._handlers.append(handler)

    def remove_handler(self, handler: Callable[[FaultEvent], None]) -> bool:
        """Remove a previously registered handler.

        Args:
            handler: The handler to remove.

        Returns:
            True if the handler was found and removed.
        """
        if handler in self._handlers:
            self._handlers.remove(handler)
            return True
        return False

    def check_reading(self, reading: TelemetryReading) -> List[FaultEvent]:
        """Check a single telemetry reading against all rules.

        Args:
            reading: The telemetry reading to check.

        Returns:
            List of detected FaultEvent objects (may be empty).
        """
        faults: List[FaultEvent] = []
        for rule in self._rules:
            if rule.metric_name != reading.metric_name:
                continue
            severity = self._evaluate_threshold(rule, reading.value)
            if severity is not None:
                fault = self._create_fault(reading, rule, severity)
                faults.append(fault)
                self._detected_faults.append(fault)
                self._notify_handlers(fault)
        return faults

    def check_anomaly(self, reading: TelemetryReading) -> Optional[FaultEvent]:
        """Check if a reading is a statistical anomaly using z-score.

        Args:
            reading: The telemetry reading to check.

        Returns:
            A FaultEvent if the reading is anomalous, None otherwise.
        """
        # This requires historical context; the caller should provide it
        # via a node's telemetry history. For standalone use, we skip.
        return None

    def check_anomaly_with_history(
        self,
        reading: TelemetryReading,
        history: List[TelemetryReading],
    ) -> Optional[FaultEvent]:
        """Check if a reading is anomalous given historical context.

        Args:
            reading: The current reading.
            history: Previous readings for the same metric.

        Returns:
            A FaultEvent if anomalous, None otherwise.

        Raises:
            InvalidTelemetryError: If history is empty or contains different metrics.
        """
        if not history:
            raise InvalidTelemetryError("History cannot be empty")
        if any(r.metric_name != reading.metric_name for r in history):
            raise InvalidTelemetryError("All history readings must have the same metric_name")

        values = [r.value for r in history]
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0

        if stdev == 0.0:
            return None

        z_score = abs(reading.value - mean) / stdev
        if z_score > self._z_score_threshold:
            self._fault_counter += 1
            fault = FaultEvent(
                fault_id=f"FAULT-{self._fault_counter:06d}",
                node_id=reading.node_id,
                severity=FaultSeverity.WARNING,
                description=(
                    f"Statistical anomaly detected: z-score={z_score:.2f} "
                    f"(threshold={self._z_score_threshold:.2f})"
                ),
                timestamp=reading.timestamp,
                metric_name=reading.metric_name,
                observed_value=reading.value,
                threshold=mean + self._z_score_threshold * stdev,
            )
            self._detected_faults.append(fault)
            self._notify_handlers(fault)
            return fault
        return None

    def get_faults_by_severity(self, severity: FaultSeverity) -> List[FaultEvent]:
        """Get all detected faults of a given severity.

        Args:
            severity: The severity level to filter by.

        Returns:
            List of matching FaultEvent objects.
        """
        return [f for f in self._detected_faults if f.severity == severity]

    def get_faults_by_node(self, node_id: str) -> List[FaultEvent]:
        """Get all detected faults for a specific node.

        Args:
            node_id: The node to filter by.

        Returns:
            List of matching FaultEvent objects.
        """
        return [f for f in self._detected_faults if f.node_id == node_id]

    def acknowledge_fault(self, fault_id: str) -> bool:
        """Acknowledge a fault by ID.

        Args:
            fault_id: The fault to acknowledge.

        Returns:
            True if the fault was found and acknowledged.
        """
        for fault in self._detected_faults:
            if fault.fault_id == fault_id:
                # FaultEvent is frozen; create a new acknowledged version
                acknowledged = FaultEvent(
                    fault_id=fault.fault_id,
                    node_id=fault.node_id,
                    severity=fault.severity,
                    description=fault.description,
                    timestamp=fault.timestamp,
                    metric_name=fault.metric_name,
                    observed_value=fault.observed_value,
                    threshold=fault.threshold,
                    acknowledged=True,
                )
                idx = self._detected_faults.index(fault)
                self._detected_faults[idx] = acknowledged
                return True
        return False

    def clear_faults(self) -> None:
        """Clear all detected faults."""
        self._detected_faults.clear()

    def _evaluate_threshold(
        self, rule: ThresholdRule, value: float
    ) -> Optional[FaultSeverity]:
        """Evaluate a value against a threshold rule.

        Args:
            rule: The threshold rule.
            value: The value to check.

        Returns:
            The severity level if a threshold is breached, None otherwise.
        """
        if rule.comparison == "greater":
            if value >= rule.emergency_threshold:
                return FaultSeverity.EMERGENCY
            if value >= rule.critical_threshold:
                return FaultSeverity.CRITICAL
            if value >= rule.warning_threshold:
                return FaultSeverity.WARNING
        else:
            if value <= rule.emergency_threshold:
                return FaultSeverity.EMERGENCY
            if value <= rule.critical_threshold:
                return FaultSeverity.CRITICAL
            if value <= rule.warning_threshold:
                return FaultSeverity.WARNING
        return None

    def _create_fault(
        self,
        reading: TelemetryReading,
        rule: ThresholdRule,
        severity: FaultSeverity,
    ) -> FaultEvent:
        """Create a FaultEvent from a rule violation."""
        self._fault_counter += 1
        return FaultEvent(
            fault_id=f"FAULT-{self._fault_counter:06d}",
            node_id=reading.node_id,
            severity=severity,
            description=(
                f"Threshold breach on '{rule.metric_name}': "
                f"value={reading.value:.2f}, "
                f"threshold={self._get_threshold_for_severity(rule, severity):.2f}. "
                f"{rule.description}"
            ),
            timestamp=reading.timestamp,
            metric_name=rule.metric_name,
            observed_value=reading.value,
            threshold=self._get_threshold_for_severity(rule, severity),
        )

    @staticmethod
    def _get_threshold_for_severity(
        rule: ThresholdRule, severity: FaultSeverity
    ) -> float:
        """Get the threshold value for a given severity level."""
        mapping = {
            FaultSeverity.WARNING: rule.warning_threshold,
            FaultSeverity.CRITICAL: rule.critical_threshold,
            FaultSeverity.EMERGENCY: rule.emergency_threshold,
        }
        return mapping.get(severity, rule.warning_threshold)

    def _notify_handlers(self, fault: FaultEvent) -> None:
        """Notify all registered handlers of a new fault."""
        for handler in self._handlers:
            try:
                handler(fault)
            except Exception as exc:
                logger.warning("Fault handler %r failed: %s", handler, exc)

    def __repr__(self) -> str:
        return (
            f"FaultDetector(rules={len(self._rules)}, "
            f"faults_detected={len(self._detected_faults)})"
        )


# ---------------------------------------------------------------------------
# PredictiveMaintenance
# ---------------------------------------------------------------------------


class PredictiveMaintenance:
    """Predicts component failures using trend analysis and degradation models.

    Uses linear regression on telemetry trends to estimate time-to-failure
    and failure probability for critical components.

    Attributes:
        prediction_window_hours: How far ahead to predict.
        min_data_points: Minimum readings required for a prediction.
        degradation_models: Mapping of component names to degradation parameters.
    """

    def __init__(
        self,
        prediction_window_hours: float = 168.0,
        min_data_points: int = 10,
    ) -> None:
        """Initialize the PredictiveMaintenance engine.

        Args:
            prediction_window_hours: How far ahead to predict (default 1 week).
            min_data_points: Minimum readings needed for reliable prediction.

        Raises:
            ValueError: If parameters are non-positive.
        """
        if prediction_window_hours <= 0:
            raise ValueError("prediction_window_hours must be positive")
        if min_data_points < 2:
            raise ValueError("min_data_points must be at least 2")

        self._prediction_window: float = prediction_window_hours
        self._min_data_points: int = min_data_points
        self._degradation_models: Dict[str, Dict[str, float]] = {}
        self._predictions: List[MaintenancePrediction] = []

    @property
    def predictions(self) -> Tuple[MaintenancePrediction, ...]:
        """All generated predictions."""
        return tuple(self._predictions)

    def register_degradation_model(
        self,
        component: str,
        failure_threshold: float,
        degradation_rate: float,
        noise_factor: float = 0.1,
    ) -> None:
        """Register a degradation model for a component.

        The model assumes linear degradation: metric_value increases
        (or decreases) at `degradation_rate` per hour until it reaches
        `failure_threshold`.

        Args:
            component: Component name.
            failure_threshold: Value at which the component is considered failed.
            degradation_rate: Rate of degradation per hour.
            noise_factor: Expected noise as a fraction of the signal.
        """
        self._degradation_models[component] = {
            "failure_threshold": failure_threshold,
            "degradation_rate": degradation_rate,
            "noise_factor": noise_factor,
        }

    def predict(
        self,
        node_id: str,
        component: str,
        readings: List[TelemetryReading],
    ) -> Optional[MaintenancePrediction]:
        """Generate a maintenance prediction for a component.

        Uses linear regression on the readings to estimate the degradation
        trend and extrapolate time-to-failure.

        Args:
            node_id: The node the component belongs to.
            component: Component name (must have a registered model).
            readings: Historical telemetry readings for this component.

        Returns:
            A MaintenancePrediction, or None if insufficient data.

        Raises:
            MaintenancePredictionError: If no degradation model is registered
                for the component.
        """
        if component not in self._degradation_models:
            raise MaintenancePredictionError(
                f"No degradation model registered for component '{component}'"
            )
        if len(readings) < self._min_data_points:
            logger.debug(
                "Insufficient data for prediction: %d < %d",
                len(readings),
                self._min_data_points,
            )
            return None

        model = self._degradation_models[component]
        # Sort by timestamp
        sorted_readings = sorted(readings, key=lambda r: r.timestamp)
        values = [r.value for r in sorted_readings]
        timestamps = [r.timestamp for r in sorted_readings]

        # Normalize timestamps to hours from first reading
        t0 = timestamps[0]
        hours = [(t - t0) / 3600.0 for t in timestamps]

        # Simple linear regression
        slope, intercept = self._linear_regression(hours, values)
        current_value = values[-1]
        threshold = model["failure_threshold"]

        # Estimate time to failure
        if slope <= 0:
            # No degradation trend
            time_to_failure = float("inf")
            failure_prob = 0.0
        else:
            remaining = threshold - current_value
            time_to_failure = remaining / slope
            # Failure probability based on how close we are to the threshold
            failure_prob = min(1.0, max(0.0, 1.0 - (remaining / threshold)))

        # Determine urgency
        urgency = self._classify_urgency(time_to_failure, failure_prob)

        # Confidence based on data quality
        confidence = self._compute_confidence(values, slope)

        prediction = MaintenancePrediction(
            node_id=node_id,
            component=component,
            failure_probability=failure_prob,
            estimated_time_to_failure_hours=(
                time_to_failure if time_to_failure != float("inf") else 99999.0
            ),
            urgency=urgency,
            recommended_action=self._recommend_action(urgency, component),
            confidence=confidence,
        )
        self._predictions.append(prediction)
        return prediction

    def get_predictions_by_urgency(
        self, urgency: MaintenanceUrgency
    ) -> List[MaintenancePrediction]:
        """Get predictions filtered by urgency level.

        Args:
            urgency: The urgency level to filter by.

        Returns:
            List of matching predictions.
        """
        return [p for p in self._predictions if p.urgency == urgency]

    def get_predictions_by_node(self, node_id: str) -> List[MaintenancePrediction]:
        """Get predictions for a specific node.

        Args:
            node_id: The node to filter by.

        Returns:
            List of matching predictions.
        """
        return [p for p in self._predictions if p.node_id == node_id]

    def clear_predictions(self) -> None:
        """Clear all stored predictions."""
        self._predictions.clear()

    @staticmethod
    def _linear_regression(x: List[float], y: List[float]) -> Tuple[float, float]:
        """Compute simple linear regression (slope, intercept).

        Args:
            x: Independent variable values.
            y: Dependent variable values.

        Returns:
            Tuple of (slope, intercept).
        """
        n = len(x)
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denominator = sum((xi - mean_x) ** 2 for xi in x)
        if denominator == 0:
            return 0.0, mean_y
        slope = numerator / denominator
        intercept = mean_y - slope * mean_x
        return slope, intercept

    @staticmethod
    def _classify_urgency(
        time_to_failure: float, failure_prob: float
    ) -> MaintenanceUrgency:
        """Classify urgency based on time-to-failure and probability."""
        if failure_prob >= 0.8 or time_to_failure <= 24:
            return MaintenanceUrgency.CRITICAL
        if failure_prob >= 0.5 or time_to_failure <= 72:
            return MaintenanceUrgency.HIGH
        if failure_prob >= 0.2 or time_to_failure <= 168:
            return MaintenanceUrgency.MEDIUM
        return MaintenanceUrgency.LOW

    @staticmethod
    def _compute_confidence(values: List[float], slope: float) -> float:
        """Compute confidence score based on data quality.

        Higher variance relative to the trend reduces confidence.
        """
        if len(values) < 2:
            return 0.0
        mean_val = statistics.mean(values)
        if mean_val == 0:
            return 0.5
        try:
            stdev = statistics.stdev(values)
        except statistics.StatisticsError:
            return 0.5
        signal_to_noise = abs(slope) / (stdev / math.sqrt(len(values)) + 1e-9)
        confidence = min(1.0, signal_to_noise / 10.0)
        return max(0.0, confidence)

    @staticmethod
    def _recommend_action(urgency: MaintenanceUrgency, component: str) -> str:
        """Generate a recommended maintenance action."""
        actions = {
            MaintenanceUrgency.CRITICAL: (
                f"Schedule immediate maintenance for {component}. "
                "Consider temporary failover."
            ),
            MaintenanceUrgency.HIGH: (
                f"Schedule maintenance for {component} within 24-48 hours."
            ),
            MaintenanceUrgency.MEDIUM: (
                f"Plan maintenance for {component} within the next week."
            ),
            MaintenanceUrgency.LOW: (
                f"Monitor {component}. No immediate action required."
            ),
        }
        return actions[urgency]

    def __repr__(self) -> str:
        return (
            f"PredictiveMaintenance(models={len(self._degradation_models)}, "
            f"predictions={len(self._predictions)})"
        )


# ---------------------------------------------------------------------------
# SelfHealingEngine
# ---------------------------------------------------------------------------


class SelfHealingEngine:
    """Automatically remediates detected faults in the network.

    Uses a rule-based approach to select remediation actions based on
    fault severity, node role, and current network state. Actions are
    executed asynchronously and results are tracked.

    Attributes:
        enabled: Whether self-healing is currently active.
        max_retries: Maximum retry attempts for failed remediation.
        action_cooldown_seconds: Minimum time between actions on the same node.
    """

    def __init__(
        self,
        enabled: bool = True,
        max_retries: int = 3,
        action_cooldown_seconds: float = 60.0,
    ) -> None:
        """Initialize the SelfHealingEngine.

        Args:
            enabled: Whether self-healing is active.
            max_retries: Maximum retry attempts for a single remediation.
            action_cooldown_seconds: Cooldown between actions on the same node.

        Raises:
            ValueError: If max_retries < 0 or cooldown < 0.
        """
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if action_cooldown_seconds < 0:
            raise ValueError("action_cooldown_seconds must be non-negative")

        self._enabled: bool = enabled
        self._max_retries: int = max_retries
        self._cooldown: float = action_cooldown_seconds
        self._last_action_time: Dict[str, float] = {}
        self._action_history: List[RemediationResult] = []
        self._remediation_rules: List[
            Callable[[FaultEvent, NetworkTopology], Optional[RemediationAction]]
        ] = []
        self._setup_default_rules()

    @property
    def enabled(self) -> bool:
        """Whether self-healing is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def action_history(self) -> Tuple[RemediationResult, ...]:
        """History of all remediation actions."""
        return tuple(self._action_history)

    def add_remediation_rule(
        self,
        rule: Callable[[FaultEvent, NetworkTopology], Optional[RemediationAction]],
    ) -> None:
        """Add a custom remediation rule.

        Rules are evaluated in order. The first rule that returns a
        non-None action wins.

        Args:
            rule: Callable that takes (FaultEvent, NetworkTopology) and
                returns an optional RemediationAction.
        """
        self._remediation_rules.append(rule)

    async def handle_fault(
        self, fault: FaultEvent, topology: NetworkTopology
    ) -> Optional[RemediationResult]:
        """Handle a detected fault by selecting and executing a remediation.

        Args:
            fault: The fault to remediate.
            topology: Current network topology.

        Returns:
            RemediationResult if an action was taken, None otherwise.
        """
        if not self._enabled:
            logger.debug("Self-healing disabled, skipping fault %s", fault.fault_id)
            return None

        # Check cooldown
        last_time = self._last_action_time.get(fault.node_id, 0.0)
        if time.time() - last_time < self._cooldown:
            logger.debug(
                "Node %s in cooldown, skipping remediation", fault.node_id
            )
            return None

        # Select action
        action = self._select_action(fault, topology)
        if action is None:
            logger.debug("No remediation action for fault %s", fault.fault_id)
            return None

        # Execute with retries
        result = await self._execute_with_retries(action, fault, topology)
        self._action_history.append(result)
        self._last_action_time[fault.node_id] = time.time()
        return result

    async def handle_faults(
        self, faults: List[FaultEvent], topology: NetworkTopology
    ) -> List[RemediationResult]:
        """Handle multiple faults concurrently.

        Args:
            faults: List of faults to remediate.
            topology: Current network topology.

        Returns:
            List of RemediationResult objects.
        """
        tasks = [self.handle_fault(f, topology) for f in faults]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [
            r
            for r in results
            if isinstance(r, RemediationResult)
        ]

    def get_actions_by_node(self, node_id: str) -> List[RemediationResult]:
        """Get remediation history for a specific node.

        Args:
            node_id: The node to filter by.

        Returns:
            List of RemediationResult objects.
        """
        return [r for r in self._action_history if r.node_id == node_id]

    def get_success_rate(self) -> float:
        """Compute the success rate of remediation actions.

        Returns:
            Success rate in [0.0, 1.0]. Returns 0.0 if no actions taken.
        """
        if not self._action_history:
            return 0.0
        successes = sum(1 for r in self._action_history if r.success)
        return successes / len(self._action_history)

    def clear_history(self) -> None:
        """Clear all action history."""
        self._action_history.clear()
        self._last_action_time.clear()

    def _select_action(
        self, fault: FaultEvent, topology: NetworkTopology
    ) -> Optional[RemediationAction]:
        """Select a remediation action for a fault.

        Evaluates custom rules first, then falls back to default logic.
        """
        for rule in self._remediation_rules:
            try:
                action = rule(fault, topology)
                if action is not None:
                    return action
            except Exception as exc:
                logger.warning("Remediation rule failed: %s", exc)

        # Default logic based on severity
        return self._default_action(fault, topology)

    def _default_action(
        self, fault: FaultEvent, topology: NetworkTopology
    ) -> Optional[RemediationAction]:
        """Default remediation action selection logic."""
        if fault.severity == FaultSeverity.EMERGENCY:
            return RemediationAction.ISOLATE_NODE
        if fault.severity == FaultSeverity.CRITICAL:
            # Try failover if the node has neighbors
            try:
                neighbors = topology.get_neighbors(fault.node_id)
                if neighbors:
                    return RemediationAction.FAILOVER
            except NodeNotFoundError:
                pass
            return RemediationAction.RESTART_SERVICE
        if fault.severity == FaultSeverity.WARNING:
            return RemediationAction.REDUCE_LOAD
        return RemediationAction.NOTIFY_OPERATOR

    async def _execute_with_retries(
        self,
        action: RemediationAction,
        fault: FaultEvent,
        topology: NetworkTopology,
    ) -> RemediationResult:
        """Execute a remediation action with retry logic.

        Args:
            action: The action to execute.
            fault: The fault being remediated.
            topology: Current network topology.

        Returns:
            RemediationResult indicating success or failure.
        """
        last_message = ""
        for attempt in range(self._max_retries + 1):
            try:
                success, message = await self._execute_action(action, fault, topology)
                if success:
                    return RemediationResult(
                        success=True,
                        action=action,
                        node_id=fault.node_id,
                        message=message,
                        timestamp=time.time(),
                    )
                last_message = message
                if attempt < self._max_retries:
                    await asyncio.sleep(0.1 * (attempt + 1))
            except Exception as exc:
                last_message = str(exc)
                logger.warning(
                    "Remediation attempt %d failed for %s: %s",
                    attempt + 1,
                    action.value,
                    exc,
                )

        return RemediationResult(
            success=False,
            action=action,
            node_id=fault.node_id,
            message=f"Failed after {self._max_retries + 1} attempts: {last_message}",
            timestamp=time.time(),
        )

    async def _execute_action(
        self,
        action: RemediationAction,
        fault: FaultEvent,
        topology: NetworkTopology,
    ) -> Tuple[bool, str]:
        """Execute a single remediation action.

        Args:
            action: The action to execute.
            fault: The fault being remediated.
            topology: Current network topology.

        Returns:
            Tuple of (success, message).
        """
        # Simulate action execution with realistic delays
        delay = random.uniform(0.01, 0.1)
        await asyncio.sleep(delay)

        if action == RemediationAction.RESTART_SERVICE:
            return await self._restart_service(fault.node_id, topology)
        if action == RemediationAction.FAILOVER:
            return await self._failover(fault.node_id, topology)
        if action == RemediationAction.REROUTE_TRAFFIC:
            return await self._reroute_traffic(fault.node_id, topology)
        if action == RemediationAction.SCALE_UP:
            return await self._scale_up(fault.node_id, topology)
        if action == RemediationAction.ISOLATE_NODE:
            return await self._isolate_node(fault.node_id, topology)
        if action == RemediationAction.REDUCE_LOAD:
            return await self._reduce_load(fault.node_id, topology)
        if action == RemediationAction.NOTIFY_OPERATOR:
            return True, f"Operator notified about fault {fault.fault_id}"
        return False, f"Unknown action: {action.value}"

    async def _restart_service(
        self, node_id: str, topology: NetworkTopology
    ) -> Tuple[bool, str]:
        """Simulate restarting a service on a node."""
        try:
            node = topology.get_node(node_id)
            node.status = NodeStatus.MAINTENANCE
            await asyncio.sleep(0.05)
            node.status = NodeStatus.HEALTHY
            return True, f"Service restarted on node '{node_id}'"
        except NodeNotFoundError:
            return False, f"Node '{node_id}' not found"

    async def _failover(
        self, node_id: str, topology: NetworkTopology
    ) -> Tuple[bool, str]:
        """Simulate failing over to a neighbor node."""
        try:
            neighbors = topology.get_neighbors(node_id)
            if not neighbors:
                return False, f"No neighbors available for failover from '{node_id}'"
            target = random.choice(list(neighbors))
            return True, f"Failed over from '{node_id}' to '{target}'"
        except NodeNotFoundError:
            return False, f"Node '{node_id}' not found"

    async def _reroute_traffic(
        self, node_id: str, topology: NetworkTopology
    ) -> Tuple[bool, str]:
        """Simulate rerouting traffic around a node."""
        try:
            node = topology.get_node(node_id)
            node.status = NodeStatus.DEGRADED
            return True, f"Traffic rerouted around node '{node_id}'"
        except NodeNotFoundError:
            return False, f"Node '{node_id}' not found"

    async def _scale_up(
        self, node_id: str, topology: NetworkTopology
    ) -> Tuple[bool, str]:
        """Simulate scaling up resources on a node."""
        return True, f"Resources scaled up on node '{node_id}'"

    async def _isolate_node(
        self, node_id: str, topology: NetworkTopology
    ) -> Tuple[bool, str]:
        """Simulate isolating a node from the network."""
        try:
            node = topology.get_node(node_id)
            node.status = NodeStatus.OFFLINE
            return True, f"Node '{node_id}' isolated from network"
        except NodeNotFoundError:
            return False, f"Node '{node_id}' not found"

    async def _reduce_load(
        self, node_id: str, topology: NetworkTopology
    ) -> Tuple[bool, str]:
        """Simulate reducing load on a node."""
        return True, f"Load reduced on node '{node_id}'"

    def _setup_default_rules(self) -> None:
        """Set up default remediation rules."""
        # Rule: If a core node is critical, failover immediately
        def core_failover_rule(
            fault: FaultEvent, topology: NetworkTopology
        ) -> Optional[RemediationAction]:
            try:
                node = topology.get_node(fault.node_id)
                if node.role == NodeRole.CORE and fault.severity in (
                    FaultSeverity.CRITICAL,
                    FaultSeverity.EMERGENCY,
                ):
                    return RemediationAction.FAILOVER
            except NodeNotFoundError:
                pass
            return None

        self._remediation_rules.append(core_failover_rule)

    def __repr__(self) -> str:
        return (
            f"SelfHealingEngine(enabled={self._enabled}, "
            f"actions={len(self._action_history)}, "
            f"success_rate={self.get_success_rate():.2%})"
        )


# ---------------------------------------------------------------------------
# NetworkOperations (High-level orchestrator)
# ---------------------------------------------------------------------------


class NetworkOperations:
    """High-level orchestrator for network operations.

    Ties together topology management, fault detection, predictive
    maintenance, and self-healing into a unified interface.

    Attributes:
        topology: The managed network topology.
        fault_detector: The fault detection engine.
        predictive_maintenance: The predictive maintenance engine.
        self_healing: The self-healing engine.
    """

    def __init__(
        self,
        topology: Optional[NetworkTopology] = None,
        fault_detector: Optional[FaultDetector] = None,
        predictive_maintenance: Optional[PredictiveMaintenance] = None,
        self_healing: Optional[SelfHealingEngine] = None,
    ) -> None:
        """Initialize NetworkOperations.

        Args:
            topology: Network topology (creates a new one if None).
            fault_detector: Fault detector (creates a new one if None).
            predictive_maintenance: Predictive maintenance engine.
            self_healing: Self-healing engine.
        """
        self._topology = topology or NetworkTopology()
        self._fault_detector = fault_detector or FaultDetector()
        self._predictive_maintenance = predictive_maintenance or PredictiveMaintenance()
        self._self_healing = self_healing or SelfHealingEngine()

        # Wire up self-healing to fault detector
        self._fault_detector.add_handler(self._on_fault_detected)

    @property
    def topology(self) -> NetworkTopology:
        """The managed network topology."""
        return self._topology

    @property
    def fault_detector(self) -> FaultDetector:
        """The fault detection engine."""
        return self._fault_detector

    @property
    def predictive_maintenance(self) -> PredictiveMaintenance:
        """The predictive maintenance engine."""
        return self._predictive_maintenance

    @property
    def self_healing(self) -> SelfHealingEngine:
        """The self-healing engine."""
        return self._self_healing

    def add_node(self, node: NetworkNode) -> None:
        """Add a node to the network.

        Args:
            node: The node to add.
        """
        self._topology.add_node(node)

    def remove_node(self, node_id: str) -> NetworkNode:
        """Remove a node from the network.

        Args:
            node_id: The node to remove.

        Returns:
            The removed node.
        """
        return self._topology.remove_node(node_id)

    def connect(self, node_id_a: str, node_id_b: str) -> None:
        """Connect two nodes.

        Args:
            node_id_a: First node.
            node_id_b: Second node.
        """
        self._topology.add_link(node_id_a, node_id_b)

    def ingest_telemetry(self, reading: TelemetryReading) -> List[FaultEvent]:
        """Ingest a telemetry reading and check for faults.

        Args:
            reading: The telemetry reading to ingest.

        Returns:
            List of detected faults (may be empty).

        Raises:
            NodeNotFoundError: If the reading's node doesn't exist.
        """
        node = self._topology.get_node(reading.node_id)
        node.add_telemetry(reading)
        return self._fault_detector.check_reading(reading)

    def get_health_score(self, node_id: str) -> HealthScore:
        """Compute a composite health score for a node.

        Args:
            node_id: The node to score.

        Returns:
            HealthScore for the node.

        Raises:
            NodeNotFoundError: If the node doesn't exist.
        """
        node = self._topology.get_node(node_id)
        component_scores: Dict[str, float] = {}

        # Score based on status
        status_scores = {
            NodeStatus.HEALTHY: 100.0,
            NodeStatus.DEGRADED: 60.0,
            NodeStatus.UNHEALTHY: 20.0,
            NodeStatus.OFFLINE: 0.0,
            NodeStatus.MAINTENANCE: 40.0,
        }
        component_scores["status"] = status_scores.get(node.status, 50.0)

        # Score based on recent fault count
        recent_faults = len(node.fault_history)
        component_scores["faults"] = max(0.0, 100.0 - recent_faults * 10.0)

        # Score based on telemetry availability
        telemetry_count = len(node.get_recent_telemetry(count=1000))
        component_scores["telemetry"] = min(100.0, telemetry_count / 10.0)

        # Weighted average
        weights = {"status": 0.5, "faults": 0.3, "telemetry": 0.2}
        overall = sum(
            component_scores.get(k, 50.0) * w for k, w in weights.items()
        )

        return HealthScore(
            node_id=node_id,
            score=overall,
            component_scores=component_scores,
        )

    def get_network_health(self) -> Dict[str, HealthScore]:
        """Get health scores for all nodes in the network.

        Returns:
            Mapping of node_id to HealthScore.
        """
        return {
            node_id: self.get_health_score(node_id)
            for node_id in self._topology.nodes
        }

    def run_predictive_maintenance(self) -> List[MaintenancePrediction]:
        """Run predictive maintenance for all nodes.

        Returns:
            List of maintenance predictions.
        """
        predictions: List[MaintenancePrediction] = []
        for node in self._topology.nodes.values():
            for component in self._predictive_maintenance._degradation_models:
                readings = node.get_recent_telemetry(metric_name=component)
                if readings:
                    pred = self._predictive_maintenance.predict(
                        node.node_id, component, readings
                    )
                    if pred is not None:
                        predictions.append(pred)
        return predictions

    async def process_faults(self) -> List[RemediationResult]:
        """Process all unacknowledged faults through self-healing.

        Returns:
            List of remediation results.
        """
        unacknowledged = [
            f for f in self._fault_detector.detected_faults if not f.acknowledged
        ]
        return await self._self_healing.handle_faults(unacknowledged, self._topology)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current network state.

        Returns:
            Dictionary with network statistics.
        """
        health = self.get_network_health()
        return {
            "node_count": self._topology.node_count,
            "link_count": self._topology.link_count,
            "healthy_nodes": sum(
                1 for h in health.values() if h.score >= 80.0
            ),
            "degraded_nodes": sum(
                1 for h in health.values() if 40.0 <= h.score < 80.0
            ),
            "unhealthy_nodes": sum(
                1 for h in health.values() if h.score < 40.0
            ),
            "total_faults": len(self._fault_detector.detected_faults),
            "unacknowledged_faults": sum(
                1
                for f in self._fault_detector.detected_faults
                if not f.acknowledged
            ),
            "self_healing_success_rate": self._self_healing.get_success_rate(),
            "average_health": (
                statistics.mean(h.score for h in health.values())
                if health
                else 0.0
            ),
        }

    def _on_fault_detected(self, fault: FaultEvent) -> None:
        """Callback when a fault is detected.

        Records the fault on the node and triggers self-healing.
        """
        try:
            node = self._topology.get_node(fault.node_id)
            node.record_fault(fault)
        except NodeNotFoundError:
            logger.warning(
                "Fault detected on unknown node '%s'", fault.node_id
            )

    def __repr__(self) -> str:
        return (
            f"NetworkOperations(nodes={self._topology.node_count}, "
            f"faults={len(self._fault_detector.detected_faults)})"
        )
