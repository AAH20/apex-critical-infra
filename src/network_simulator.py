"""Apex Critical Infrastructure — Network Simulator.

Real-time network simulation with agentic AI decision-making for telecom
and network infrastructure testing, validation, and what-if analysis.

This module provides:
- SimulationConfig: Configuration for network simulations.
- NetworkEvent: Events that occur during simulation.
- AgentAction: Actions taken by the agentic AI agent.
- SimulationMetrics: Metrics collected during simulation.
- NetworkAgent: Agentic AI agent that monitors and acts on the network.
- NetworkSimulator: Real-time network simulation engine.
- SimulationRunner: High-level runner for simulation scenarios.

All classes are fully typed, documented, and raise explicit exceptions
on invalid input or unrecoverable states.
"""

from __future__ import annotations

import asyncio
import enum
import json
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


class SimulationError(Exception):
    """Base exception for all simulation errors."""


class SimulationConfigError(SimulationError):
    """Raised when simulation configuration is invalid."""


class SimulationRuntimeError(SimulationError):
    """Raised when simulation encounters a runtime error."""


class AgentActionError(SimulationError):
    """Raised when an agent action fails."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SimulationState(enum.Enum):
    """State of the network simulation."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    ERROR = "error"


class EventType(enum.Enum):
    """Types of events in the simulation."""

    NODE_FAILURE = "node_failure"
    NODE_RECOVERY = "node_recovery"
    TRAFFIC_SURGE = "traffic_surge"
    INTERFERENCE = "interference"
    FAULT_DETECTED = "fault_detected"
    REMEDIATION = "remediation"
    MAINTENANCE = "maintenance"
    SPECTRUM_REALLOCATION = "spectrum_reallocation"
    INFERENCE_COMPLETED = "inference_completed"
    AGENT_ACTION = "agent_action"
    METRICS_UPDATE = "metrics_update"


class AgentPolicy(enum.Enum):
    """Policies for the agentic AI agent."""

    REACTIVE = "reactive"
    PROACTIVE = "proactive"
    PREDICTIVE = "predictive"
    BALANCED = "balanced"


class NodeRole(enum.Enum):
    """Role of a network node."""

    ROUTER = "router"
    SWITCH = "switch"
    BASE_STATION = "base_station"
    GATEWAY = "gateway"
    EDGE_SERVER = "edge_server"


class NodeStatus(enum.Enum):
    """Operational status of a network node."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class FaultSeverity(enum.Enum):
    """Severity of a detected fault."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Configuration for a network simulation.

    Attributes:
        duration_seconds: Total simulation duration in seconds.
        time_step_seconds: Simulation time step in seconds.
        node_count: Number of network nodes to simulate.
        link_density: Probability of a link between two nodes (0.0-1.0).
        failure_rate: Probability of node failure per time step.
        traffic_surge_probability: Probability of a traffic surge event.
        interference_probability: Probability of interference per time step.
        enable_self_healing: Whether to enable self-healing.
        enable_predictive_maintenance: Whether to enable predictive maintenance.
        enable_spectrum_management: Whether to enable spectrum management.
        enable_edge_ai: Whether to enable edge AI.
        agent_policy: Policy for the agentic AI agent.
        random_seed: Random seed for reproducibility.
    """

    duration_seconds: float = 300.0
    time_step_seconds: float = 1.0
    node_count: int = 10
    link_density: float = 0.3
    failure_rate: float = 0.01
    traffic_surge_probability: float = 0.05
    interference_probability: float = 0.02
    enable_self_healing: bool = True
    enable_predictive_maintenance: bool = True
    enable_spectrum_management: bool = True
    enable_edge_ai: bool = True
    agent_policy: AgentPolicy = AgentPolicy.BALANCED
    random_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0:
            raise SimulationConfigError("duration_seconds must be positive")
        if self.time_step_seconds <= 0:
            raise SimulationConfigError("time_step_seconds must be positive")
        if self.node_count < 2:
            raise SimulationConfigError("node_count must be at least 2")
        if not 0.0 <= self.link_density <= 1.0:
            raise SimulationConfigError("link_density must be in [0, 1]")
        if not 0.0 <= self.failure_rate <= 1.0:
            raise SimulationConfigError("failure_rate must be in [0, 1]")
        if not 0.0 <= self.traffic_surge_probability <= 1.0:
            raise SimulationConfigError(
                "traffic_surge_probability must be in [0, 1]"
            )
        if not 0.0 <= self.interference_probability <= 1.0:
            raise SimulationConfigError(
                "interference_probability must be in [0, 1]"
            )


@dataclass(frozen=True, slots=True)
class NetworkEvent:
    """An event that occurs during simulation.

    Attributes:
        event_id: Unique identifier.
        event_type: Type of event.
        timestamp: Simulation time when the event occurred.
        node_id: Node associated with the event (if any).
        description: Human-readable description.
        metadata: Additional event data.
    """

    event_id: str
    event_type: EventType
    timestamp: float
    node_id: Optional[str] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentAction:
    """An action taken by the agentic AI agent.

    Attributes:
        action_id: Unique identifier.
        action_type: Type of action.
        timestamp: When the action was taken.
        target_node: Node the action targets (if any).
        parameters: Action parameters.
        result: Result of the action.
        success: Whether the action succeeded.
    """

    action_id: str
    action_type: str
    timestamp: float
    target_node: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: str = ""
    success: bool = True


@dataclass(frozen=True, slots=True)
class SimulationMetrics:
    """Metrics collected during simulation.

    Attributes:
        timestamp: When metrics were collected.
        total_nodes: Total number of nodes.
        healthy_nodes: Number of healthy nodes.
        degraded_nodes: Number of degraded nodes.
        unhealthy_nodes: Number of unhealthy nodes.
        offline_nodes: Number of offline nodes.
        total_faults: Total faults detected.
        total_remediations: Total remediation actions taken.
        successful_remediations: Number of successful remediations.
        average_health_score: Average health score across nodes.
        total_events: Total events generated.
        spectrum_utilization: Average spectrum utilization.
        interference_events: Total interference events.
        inference_count: Total inferences executed.
        average_inference_latency_ms: Average inference latency.
    """

    timestamp: float
    total_nodes: int = 0
    healthy_nodes: int = 0
    degraded_nodes: int = 0
    unhealthy_nodes: int = 0
    offline_nodes: int = 0
    total_faults: int = 0
    total_remediations: int = 0
    successful_remediations: int = 0
    average_health_score: float = 0.0
    total_events: int = 0
    spectrum_utilization: float = 0.0
    interference_events: int = 0
    inference_count: int = 0
    average_inference_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp,
            "total_nodes": self.total_nodes,
            "healthy_nodes": self.healthy_nodes,
            "degraded_nodes": self.degraded_nodes,
            "unhealthy_nodes": self.unhealthy_nodes,
            "offline_nodes": self.offline_nodes,
            "total_faults": self.total_faults,
            "total_remediations": self.total_remediations,
            "successful_remediations": self.successful_remediations,
            "average_health_score": self.average_health_score,
            "total_events": self.total_events,
            "spectrum_utilization": self.spectrum_utilization,
            "interference_events": self.interference_events,
            "inference_count": self.inference_count,
            "average_inference_latency_ms": self.average_inference_latency_ms,
        }


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Result of a completed simulation run.

    Attributes:
        config: The simulation configuration used.
        final_metrics: Metrics at the end of simulation.
        all_metrics: Time series of all metrics.
        events: All events generated during simulation.
        agent_actions: All actions taken by the agent.
        duration_seconds: Actual simulation duration.
        success: Whether the simulation completed successfully.
    """

    config: SimulationConfig
    final_metrics: SimulationMetrics
    all_metrics: List[SimulationMetrics]
    events: List[NetworkEvent]
    agent_actions: List[AgentAction]
    duration_seconds: float
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "config": {
                "duration_seconds": self.config.duration_seconds,
                "time_step_seconds": self.config.time_step_seconds,
                "node_count": self.config.node_count,
                "link_density": self.config.link_density,
                "failure_rate": self.config.failure_rate,
                "agent_policy": self.config.agent_policy.value,
                "random_seed": self.config.random_seed,
            },
            "final_metrics": self.final_metrics.to_dict(),
            "all_metrics": [m.to_dict() for m in self.all_metrics],
            "events": [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type.value,
                    "timestamp": e.timestamp,
                    "node_id": e.node_id,
                    "description": e.description,
                }
                for e in self.events
            ],
            "agent_actions": [
                {
                    "action_id": a.action_id,
                    "action_type": a.action_type,
                    "timestamp": a.timestamp,
                    "target_node": a.target_node,
                    "success": a.success,
                    "result": a.result,
                }
                for a in self.agent_actions
            ],
            "duration_seconds": self.duration_seconds,
            "success": self.success,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Network primitives
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class NetworkNode:
    """A node in the simulated network.

    Attributes:
        node_id: Unique identifier.
        role: Role of the node.
        status: Current operational status.
        health_score: Health score from 0-100.
        cpu_utilization: Current CPU utilization percentage.
        memory_utilization: Current memory utilization percentage.
        packet_loss_rate: Current packet loss rate.
        throughput_mbps: Current throughput in Mbps.
        latency_ms: Current latency in milliseconds.
        uptime_seconds: Total uptime in seconds.
        failure_count: Number of failures experienced.
        last_failure_time: Timestamp of last failure.
    """

    node_id: str
    role: NodeRole = NodeRole.ROUTER
    status: NodeStatus = NodeStatus.HEALTHY
    health_score: float = 100.0
    cpu_utilization: float = 30.0
    memory_utilization: float = 40.0
    packet_loss_rate: float = 0.1
    throughput_mbps: float = 100.0
    latency_ms: float = 5.0
    uptime_seconds: float = 0.0
    failure_count: int = 0
    last_failure_time: Optional[float] = None

    def update_health(self) -> None:
        """Recalculate health score based on current metrics."""
        score = 100.0
        score -= max(0, self.cpu_utilization - 50) * 0.5
        score -= max(0, self.memory_utilization - 60) * 0.3
        score -= self.packet_loss_rate * 5.0
        score -= max(0, self.latency_ms - 10) * 0.2
        self.health_score = max(0.0, min(100.0, score))

        if self.health_score >= 80:
            self.status = NodeStatus.HEALTHY
        elif self.health_score >= 50:
            self.status = NodeStatus.DEGRADED
        elif self.health_score >= 20:
            self.status = NodeStatus.UNHEALTHY
        else:
            self.status = NodeStatus.OFFLINE


@dataclass(slots=True)
class NetworkLink:
    """A link between two network nodes.

    Attributes:
        link_id: Unique identifier.
        node_a: First node ID.
        node_b: Second node ID.
        bandwidth_mbps: Link bandwidth in Mbps.
        latency_ms: Link latency in milliseconds.
        packet_loss_rate: Link packet loss rate.
        utilization: Current utilization ratio (0.0-1.0).
        is_active: Whether the link is active.
    """

    link_id: str
    node_a: str
    node_b: str
    bandwidth_mbps: float = 1000.0
    latency_ms: float = 1.0
    packet_loss_rate: float = 0.01
    utilization: float = 0.0
    is_active: bool = True


@dataclass(slots=True)
class NetworkTopology:
    """Network topology containing nodes and links.

    Attributes:
        nodes: Dictionary of nodes by ID.
        links: Dictionary of links by ID.
        adjacency: Adjacency list representation.
    """

    nodes: Dict[str, NetworkNode] = field(default_factory=dict)
    links: Dict[str, NetworkLink] = field(default_factory=dict)
    adjacency: Dict[str, Set[str]] = field(default_factory=dict)

    def add_node(self, node: NetworkNode) -> None:
        """Add a node to the topology.

        Args:
            node: The node to add.

        Raises:
            SimulationError: If node ID already exists.
        """
        if node.node_id in self.nodes:
            raise SimulationError(f"Node {node.node_id} already exists")
        self.nodes[node.node_id] = node
        self.adjacency[node.node_id] = set()

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its links.

        Args:
            node_id: ID of the node to remove.
        """
        if node_id not in self.nodes:
            return
        del self.nodes[node_id]
        del self.adjacency[node_id]
        # Remove links involving this node
        links_to_remove = [
            lid for lid, link in self.links.items()
            if link.node_a == node_id or link.node_b == node_id
        ]
        for lid in links_to_remove:
            del self.links[lid]
        # Clean adjacency
        for adj_set in self.adjacency.values():
            adj_set.discard(node_id)

    def add_link(self, node_a: str, node_b: str, **kwargs: Any) -> NetworkLink:
        """Add a link between two nodes.

        Args:
            node_a: First node ID.
            node_b: Second node ID.
            **kwargs: Additional link attributes.

        Returns:
            The created NetworkLink.

        Raises:
            SimulationError: If nodes don't exist or link already exists.
        """
        if node_a not in self.nodes:
            raise SimulationError(f"Node {node_a} not found")
        if node_b not in self.nodes:
            raise SimulationError(f"Node {node_b} not found")
        if node_a == node_b:
            raise SimulationError("Cannot link a node to itself")

        # Check if link already exists
        for link in self.links.values():
            if (link.node_a == node_a and link.node_b == node_b) or (
                link.node_a == node_b and link.node_b == node_a
            ):
                raise SimulationError(
                    f"Link between {node_a} and {node_b} already exists"
                )

        link_id = f"link-{node_a}-{node_b}"
        link = NetworkLink(link_id=link_id, node_a=node_a, node_b=node_b, **kwargs)
        self.links[link_id] = link
        self.adjacency[node_a].add(node_b)
        self.adjacency[node_b].add(node_a)
        return link

    def remove_link(self, link_id: str) -> None:
        """Remove a link from the topology.

        Args:
            link_id: ID of the link to remove.
        """
        if link_id not in self.links:
            return
        link = self.links[link_id]
        del self.links[link_id]
        self.adjacency[link.node_a].discard(link.node_b)
        self.adjacency[link.node_b].discard(link.node_a)

    def get_neighbors(self, node_id: str) -> Set[str]:
        """Get neighbors of a node.

        Args:
            node_id: The node ID.

        Returns:
            Set of neighbor node IDs.
        """
        return self.adjacency.get(node_id, set())

    def get_shortest_path(self, source: str, target: str) -> List[str]:
        """Find shortest path between two nodes using BFS.

        Args:
            source: Source node ID.
            target: Target node ID.

        Returns:
            List of node IDs forming the path, or empty list if no path.
        """
        if source == target:
            return [source]
        if source not in self.nodes or target not in self.nodes:
            return []

        visited: Set[str] = {source}
        queue: Deque[Tuple[str, List[str]]] = deque([(source, [source])])

        while queue:
            current, path = queue.popleft()
            for neighbor in self.adjacency.get(current, set()):
                if neighbor == target:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return []

    def get_connected_components(self) -> List[Set[str]]:
        """Get connected components of the topology.

        Returns:
            List of sets, each containing node IDs in a component.
        """
        visited: Set[str] = set()
        components: List[Set[str]] = []

        for node_id in self.nodes:
            if node_id in visited:
                continue
            component: Set[str] = set()
            queue: Deque[str] = deque([node_id])
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                for neighbor in self.adjacency.get(current, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)
            components.append(component)
        return components


# ---------------------------------------------------------------------------
# Traffic simulation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TrafficFlow:
    """A traffic flow between two nodes.

    Attributes:
        flow_id: Unique identifier.
        source: Source node ID.
        destination: Destination node ID.
        bandwidth_mbps: Current bandwidth usage.
        packet_rate: Packets per second.
        protocol: Protocol type (TCP, UDP, etc.).
        priority: Priority level (1-10).
        start_time: When the flow started.
        is_active: Whether the flow is active.
    """

    flow_id: str
    source: str
    destination: str
    bandwidth_mbps: float = 10.0
    packet_rate: float = 100.0
    protocol: str = "TCP"
    priority: int = 5
    start_time: float = 0.0
    is_active: bool = True


class TrafficSimulator:
    """Simulates network traffic flows.

    Attributes:
        flows: Dictionary of active traffic flows.
        total_bytes_transferred: Total bytes transferred.
        total_packets_dropped: Total packets dropped.
    """

    def __init__(self) -> None:
        """Initialize the traffic simulator."""
        self._flows: Dict[str, TrafficFlow] = {}
        self._total_bytes: int = 0
        self._total_packets_dropped: int = 0
        self._flow_counter: int = 0

    @property
    def flows(self) -> Tuple[TrafficFlow, ...]:
        """All active traffic flows."""
        return tuple(self._flows.values())

    @property
    def total_bytes_transferred(self) -> int:
        """Total bytes transferred."""
        return self._total_bytes

    @property
    def total_packets_dropped(self) -> int:
        """Total packets dropped."""
        return self._total_packets_dropped

    def create_flow(
        self,
        source: str,
        destination: str,
        bandwidth_mbps: float = 10.0,
        protocol: str = "TCP",
        priority: int = 5,
    ) -> TrafficFlow:
        """Create a new traffic flow.

        Args:
            source: Source node ID.
            destination: Destination node ID.
            bandwidth_mbps: Bandwidth in Mbps.
            protocol: Protocol type.
            priority: Priority level (1-10).

        Returns:
            The created TrafficFlow.
        """
        self._flow_counter += 1
        flow = TrafficFlow(
            flow_id=f"flow-{self._flow_counter:04d}",
            source=source,
            destination=destination,
            bandwidth_mbps=bandwidth_mbps,
            packet_rate=bandwidth_mbps * 10,
            protocol=protocol,
            priority=priority,
            start_time=time.time(),
        )
        self._flows[flow.flow_id] = flow
        return flow

    def remove_flow(self, flow_id: str) -> None:
        """Remove a traffic flow.

        Args:
            flow_id: ID of the flow to remove.
        """
        self._flows.pop(flow_id, None)

    def simulate_step(self, topology: NetworkTopology, time_step: float) -> None:
        """Simulate one time step of traffic.

        Args:
            topology: Current network topology.
            time_step: Time step in seconds.
        """
        for flow in list(self._flows.values()):
            if not flow.is_active:
                continue

            # Check if path exists
            path = topology.get_shortest_path(flow.source, flow.destination)
            if not path:
                flow.is_active = False
                self._total_packets_dropped += int(flow.packet_rate * time_step)
                continue

            # Calculate actual bandwidth based on path
            min_bandwidth = flow.bandwidth_mbps
            total_latency = 0.0
            for i in range(len(path) - 1):
                link = self._find_link(topology, path[i], path[i + 1])
                if link and link.is_active:
                    min_bandwidth = min(min_bandwidth, link.bandwidth_mbps)
                    total_latency += link.latency_ms
                else:
                    flow.is_active = False
                    break

            if flow.is_active:
                bytes_transferred = int(
                    min_bandwidth * 1_000_000 / 8 * time_step
                )
                self._total_bytes += bytes_transferred

    def _find_link(
        self, topology: NetworkTopology, node_a: str, node_b: str
    ) -> Optional[NetworkLink]:
        """Find link between two nodes.

        Args:
            topology: Network topology.
            node_a: First node.
            node_b: Second node.

        Returns:
            The NetworkLink or None.
        """
        for link in topology.links.values():
            if (link.node_a == node_a and link.node_b == node_b) or (
                link.node_a == node_b and link.node_b == node_a
            ):
                return link
        return None

    def get_flows_through_node(self, node_id: str) -> List[TrafficFlow]:
        """Get all flows passing through a node.

        Args:
            node_id: The node ID.

        Returns:
            List of TrafficFlow objects.
        """
        return [
            f for f in self._flows.values()
            if f.is_active and node_id in (f.source, f.destination)
        ]

    def clear(self) -> None:
        """Clear all flows."""
        self._flows.clear()
        self._total_bytes = 0
        self._total_packets_dropped = 0
        self._flow_counter = 0


# ---------------------------------------------------------------------------
# Failure injection
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FailureScenario:
    """A failure scenario to inject.

    Attributes:
        scenario_id: Unique identifier.
        target_node: Node to fail (None for random).
        failure_type: Type of failure.
        duration_seconds: How long the failure lasts.
        severity: Severity of the failure.
        probability: Probability of occurrence per step.
    """

    scenario_id: str
    target_node: Optional[str] = None
    failure_type: str = "complete"
    duration_seconds: float = 10.0
    severity: FaultSeverity = FaultSeverity.HIGH
    probability: float = 0.01


class FailureInjector:
    """Injects failures into the simulated network.

    Attributes:
        scenarios: List of failure scenarios.
        active_failures: Currently active failures.
        total_failures_injected: Total failures injected.
    """

    def __init__(self) -> None:
        """Initialize the failure injector."""
        self._scenarios: List[FailureScenario] = []
        self._active_failures: Dict[str, Tuple[FailureScenario, float]] = {}
        self._total_injected: int = 0
        self._scenario_counter: int = 0

    @property
    def scenarios(self) -> Tuple[FailureScenario, ...]:
        """All configured failure scenarios."""
        return tuple(self._scenarios)

    @property
    def active_failures(self) -> Dict[str, Tuple[FailureScenario, float]]:
        """Currently active failures with start times."""
        return dict(self._active_failures)

    @property
    def total_failures_injected(self) -> int:
        """Total failures injected."""
        return self._total_injected

    def add_scenario(self, scenario: FailureScenario) -> None:
        """Add a failure scenario.

        Args:
            scenario: The scenario to add.
        """
        self._scenarios.append(scenario)

    def inject_failure(
        self,
        topology: NetworkTopology,
        scenario: Optional[FailureScenario] = None,
    ) -> Optional[str]:
        """Inject a failure into the network.

        Args:
            topology: Network topology.
            scenario: Specific scenario to inject (random if None).

        Returns:
            ID of the injected failure, or None if no failure was injected.
        """
        if scenario is None:
            if not self._scenarios:
                return None
            scenario = random.choice(self._scenarios)

        # Select target node
        target = scenario.target_node
        if target is None:
            if not topology.nodes:
                return None
            target = random.choice(list(topology.nodes.keys()))

        if target not in topology.nodes:
            return None

        # Apply failure
        node = topology.nodes[target]
        node.status = NodeStatus.OFFLINE
        node.failure_count += 1
        node.last_failure_time = time.time()
        node.health_score = 0.0

        failure_id = f"failure-{self._total_injected:04d}"
        self._active_failures[failure_id] = (scenario, time.time())
        self._total_injected += 1

        return failure_id

    def recover_failure(self, topology: NetworkTopology, failure_id: str) -> bool:
        """Recover a failed node.

        Args:
            topology: Network topology.
            failure_id: ID of the failure to recover.

        Returns:
            True if recovery was successful.
        """
        if failure_id not in self._active_failures:
            return False

        scenario, _ = self._active_failures[failure_id]
        target = scenario.target_node
        if target and target in topology.nodes:
            node = topology.nodes[target]
            node.status = NodeStatus.HEALTHY
            node.health_score = 100.0
            node.cpu_utilization = 30.0
            node.memory_utilization = 40.0
            node.packet_loss_rate = 0.1

        del self._active_failures[failure_id]
        return True

    def step(self, topology: NetworkTopology, time_step: float) -> List[str]:
        """Process one time step of failure injection.

        Args:
            topology: Network topology.
            time_step: Time step in seconds.

        Returns:
            List of new failure IDs injected this step.
        """
        new_failures: List[str] = []

        # Random failures based on probability
        for scenario in self._scenarios:
            if random.random() < scenario.probability * time_step:
                failure_id = self.inject_failure(topology, scenario)
                if failure_id:
                    new_failures.append(failure_id)

        # Recover expired failures
        current_time = time.time()
        expired = [
            fid for fid, (_, start) in self._active_failures.items()
            if current_time - start > 10.0  # Default recovery time
        ]
        for fid in expired:
            self.recover_failure(topology, fid)

        return new_failures

    def clear(self) -> None:
        """Clear all scenarios and active failures."""
        self._scenarios.clear()
        self._active_failures.clear()
        self._total_injected = 0
        self._scenario_counter = 0


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PerformanceMetrics:
    """Detailed performance metrics for the network.

    Attributes:
        timestamp: When metrics were collected.
        throughput_mbps: Total network throughput.
        average_latency_ms: Average latency across all paths.
        packet_loss_rate: Overall packet loss rate.
        jitter_ms: Jitter in milliseconds.
        availability: Network availability percentage.
        mttr_seconds: Mean time to repair in seconds.
        mtbf_seconds: Mean time between failures in seconds.
        congestion_level: Congestion level (0.0-1.0).
    """

    timestamp: float
    throughput_mbps: float = 0.0
    average_latency_ms: float = 0.0
    packet_loss_rate: float = 0.0
    jitter_ms: float = 0.0
    availability: float = 100.0
    mttr_seconds: float = 0.0
    mtbf_seconds: float = 0.0
    congestion_level: float = 0.0


class MetricsCollector:
    """Collects and aggregates performance metrics.

    Attributes:
        metrics_history: History of collected performance metrics.
        custom_metrics: Custom metrics registered by the user.
    """

    def __init__(self) -> None:
        """Initialize the metrics collector."""
        self._history: List[PerformanceMetrics] = []
        self._custom_metrics: Dict[str, List[float]] = defaultdict(list)
        self._failure_times: List[float] = []
        self._repair_times: List[float] = []

    @property
    def metrics_history(self) -> Tuple[PerformanceMetrics, ...]:
        """History of performance metrics."""
        return tuple(self._history)

    def record_failure(self, timestamp: float) -> None:
        """Record a failure event.

        Args:
            timestamp: When the failure occurred.
        """
        self._failure_times.append(timestamp)

    def record_repair(self, timestamp: float) -> None:
        """Record a repair event.

        Args:
            timestamp: When the repair occurred.
        """
        self._repair_times.append(timestamp)

    def add_custom_metric(self, name: str, value: float) -> None:
        """Add a custom metric value.

        Args:
            name: Metric name.
            value: Metric value.
        """
        self._custom_metrics[name].append(value)

    def collect(
        self,
        topology: NetworkTopology,
        traffic_sim: TrafficSimulator,
    ) -> PerformanceMetrics:
        """Collect current performance metrics.

        Args:
            topology: Current network topology.
            traffic_sim: Traffic simulator.

        Returns:
            PerformanceMetrics for the current state.
        """
        total_throughput = 0.0
        total_latency = 0.0
        total_loss = 0.0
        active_nodes = 0
        link_count = 0

        for node in topology.nodes.values():
            if node.status != NodeStatus.OFFLINE:
                total_throughput += node.throughput_mbps
                total_latency += node.latency_ms
                total_loss += node.packet_loss_rate
                active_nodes += 1

        for link in topology.links.values():
            if link.is_active:
                total_latency += link.latency_ms
                total_loss += link.packet_loss_rate
                link_count += 1

        avg_latency = total_latency / max(1, active_nodes + link_count)
        avg_loss = total_loss / max(1, active_nodes + link_count)

        # Calculate availability
        total_nodes = len(topology.nodes)
        availability = (active_nodes / max(1, total_nodes)) * 100.0

        # Calculate MTTR and MTBF
        mttr = 0.0
        if len(self._repair_times) >= 2:
            repair_diffs = [
                self._repair_times[i] - self._repair_times[i - 1]
                for i in range(1, len(self._repair_times))
            ]
            mttr = statistics.mean(repair_diffs)

        mtbf = 0.0
        if len(self._failure_times) >= 2:
            failure_diffs = [
                self._failure_times[i] - self._failure_times[i - 1]
                for i in range(1, len(self._failure_times))
            ]
            mtbf = statistics.mean(failure_diffs)

        # Congestion estimation
        congestion = min(1.0, total_throughput / max(1, total_nodes * 100.0))

        metrics = PerformanceMetrics(
            timestamp=time.time(),
            throughput_mbps=total_throughput,
            average_latency_ms=avg_latency,
            packet_loss_rate=avg_loss,
            jitter_ms=random.uniform(0.1, 2.0),
            availability=availability,
            mttr_seconds=mttr,
            mtbf_seconds=mtbf,
            congestion_level=congestion,
        )
        self._history.append(metrics)
        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of collected metrics.

        Returns:
            Dictionary with summary statistics.
        """
        if not self._history:
            return {}

        throughputs = [m.throughput_mbps for m in self._history]
        latencies = [m.average_latency_ms for m in self._history]
        availabilities = [m.availability for m in self._history]

        return {
            "avg_throughput_mbps": statistics.mean(throughputs),
            "max_throughput_mbps": max(throughputs),
            "min_throughput_mbps": min(throughputs),
            "avg_latency_ms": statistics.mean(latencies),
            "max_latency_ms": max(latencies),
            "min_latency_ms": min(latencies),
            "avg_availability": statistics.mean(availabilities),
            "min_availability": min(availabilities),
            "total_samples": len(self._history),
            "custom_metrics": {
                name: {
                    "avg": statistics.mean(values),
                    "max": max(values),
                    "min": min(values),
                }
                for name, values in self._custom_metrics.items()
            },
        }

    def clear(self) -> None:
        """Clear all collected metrics."""
        self._history.clear()
        self._custom_metrics.clear()
        self._failure_times.clear()
        self._repair_times.clear()


# ---------------------------------------------------------------------------
# NetworkAgent
# ---------------------------------------------------------------------------


class NetworkAgent:
    """Agentic AI agent that monitors and acts on the simulated network.

    The agent observes network state, detects anomalies, and takes
    actions to maintain network health. It supports multiple policies:
    reactive (respond to faults), proactive (prevent issues),
    predictive (use predictions), and balanced (combination).

    Attributes:
        policy: The agent's decision-making policy.
        action_history: History of actions taken by the agent.
    """

    def __init__(
        self,
        policy: AgentPolicy = AgentPolicy.BALANCED,
        reaction_delay_seconds: float = 0.5,
    ) -> None:
        """Initialize the NetworkAgent.

        Args:
            policy: Decision-making policy.
            reaction_delay_seconds: Delay before reacting to events.

        Raises:
            ValueError: If reaction_delay_seconds < 0.
        """
        if reaction_delay_seconds < 0:
            raise ValueError("reaction_delay_seconds must be non-negative")

        self._policy: AgentPolicy = policy
        self._reaction_delay: float = reaction_delay_seconds
        self._action_history: List[AgentAction] = []
        self._action_counter: int = 0
        self._knowledge_base: Dict[str, Any] = {}
        self._event_queue: Deque[NetworkEvent] = deque()

    @property
    def policy(self) -> AgentPolicy:
        """Current agent policy."""
        return self._policy

    @policy.setter
    def policy(self, value: AgentPolicy) -> None:
        self._policy = value

    @property
    def action_history(self) -> Tuple[AgentAction, ...]:
        """History of agent actions."""
        return tuple(self._action_history)

    async def observe_and_act(
        self,
        topology: NetworkTopology,
        traffic_sim: TrafficSimulator,
        metrics_collector: MetricsCollector,
    ) -> List[AgentAction]:
        """Observe the network state and take appropriate actions.

        Args:
            topology: Current network topology.
            traffic_sim: Traffic simulator.
            metrics_collector: Metrics collector.

        Returns:
            List of actions taken.
        """
        actions: List[AgentAction] = []

        # Policy-specific behavior
        if self._policy == AgentPolicy.REACTIVE:
            actions.extend(
                await self._reactive_actions(topology, metrics_collector)
            )
        elif self._policy == AgentPolicy.PROACTIVE:
            actions.extend(
                await self._proactive_actions(topology, traffic_sim)
            )
        elif self._policy == AgentPolicy.PREDICTIVE:
            actions.extend(
                await self._predictive_actions(topology)
            )
        elif self._policy == AgentPolicy.BALANCED:
            actions.extend(
                await self._reactive_actions(topology, metrics_collector)
            )
            actions.extend(
                await self._proactive_actions(topology, traffic_sim)
            )
            actions.extend(
                await self._predictive_actions(topology)
            )

        self._action_history.extend(actions)
        return actions

    def record_event(self, event: NetworkEvent) -> None:
        """Record an event for the agent to process.

        Args:
            event: The event to record.
        """
        self._event_queue.append(event)

    def get_knowledge(self, key: str) -> Any:
        """Get a value from the agent's knowledge base.

        Args:
            key: The knowledge key.

        Returns:
            The stored value, or None.
        """
        return self._knowledge_base.get(key)

    def set_knowledge(self, key: str, value: Any) -> None:
        """Store a value in the agent's knowledge base.

        Args:
            key: The knowledge key.
            value: The value to store.
        """
        self._knowledge_base[key] = value

    def clear_history(self) -> None:
        """Clear action history and event queue."""
        self._action_history.clear()
        self._event_queue.clear()

    async def _reactive_actions(
        self,
        topology: NetworkTopology,
        metrics_collector: MetricsCollector,
    ) -> List[AgentAction]:
        """Take reactive actions based on current network state."""
        actions: List[AgentAction] = []

        # Find offline or unhealthy nodes
        for node in topology.nodes.values():
            if node.status == NodeStatus.OFFLINE:
                await asyncio.sleep(self._reaction_delay)
                self._action_counter += 1
                action = AgentAction(
                    action_id=f"AGENT-{self._action_counter:06d}",
                    action_type="node_recovery",
                    timestamp=time.time(),
                    target_node=node.node_id,
                    parameters={"previous_status": "offline"},
                    result=f"Attempting recovery of node {node.node_id}",
                    success=True,
                )
                actions.append(action)
                # Simulate recovery
                node.status = NodeStatus.HEALTHY
                node.health_score = 80.0
                metrics_collector.record_repair(time.time())

        return actions

    async def _proactive_actions(
        self,
        topology: NetworkTopology,
        traffic_sim: TrafficSimulator,
    ) -> List[AgentAction]:
        """Take proactive actions to prevent issues."""
        actions: List[AgentAction] = []

        # Check for degraded nodes and take preventive action
        for node in topology.nodes.values():
            if node.status == NodeStatus.DEGRADED:
                self._action_counter += 1
                action = AgentAction(
                    action_id=f"AGENT-{self._action_counter:06d}",
                    action_type="preventive_check",
                    timestamp=time.time(),
                    target_node=node.node_id,
                    parameters={"health_score": node.health_score},
                    result=(
                        f"Preventive check on node {node.node_id} "
                        f"(health: {node.health_score:.1f})"
                    ),
                    success=True,
                )
                actions.append(action)

        return actions

    async def _predictive_actions(
        self, topology: NetworkTopology
    ) -> List[AgentAction]:
        """Take predictive actions based on trend analysis."""
        actions: List[AgentAction] = []

        # Simple prediction: nodes with declining health trend
        for node in topology.nodes.values():
            if (
                node.status == NodeStatus.HEALTHY
                and node.health_score < 85.0
                and node.failure_count > 0
            ):
                self._action_counter += 1
                action = AgentAction(
                    action_id=f"AGENT-{self._action_counter:06d}",
                    action_type="predictive_maintenance",
                    timestamp=time.time(),
                    target_node=node.node_id,
                    parameters={
                        "health_score": node.health_score,
                        "failure_count": node.failure_count,
                    },
                    result=(
                        f"Predictive maintenance scheduled for {node.node_id}"
                    ),
                    success=True,
                )
                actions.append(action)

        return actions

    def __repr__(self) -> str:
        return (
            f"NetworkAgent(policy={self._policy.value!r}, "
            f"actions={len(self._action_history)})"
        )


# ---------------------------------------------------------------------------
# NetworkSimulator
# ---------------------------------------------------------------------------


class NetworkSimulator:
    """Real-time network simulation engine with agentic AI.

    Simulates a telecom network with nodes, links, faults, traffic,
    spectrum usage, and edge AI inference. An agentic AI agent
    monitors and manages the simulated network.

    Attributes:
        config: Simulation configuration.
        state: Current simulation state.
        topology: Current network topology.
        traffic_simulator: Traffic simulator instance.
        failure_injector: Failure injector instance.
        metrics_collector: Metrics collector instance.
        agent: The agentic AI agent.
    """

    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        """Initialize the NetworkSimulator.

        Args:
            config: Simulation configuration (uses defaults if None).
        """
        self._config = config or SimulationConfig()
        self._state = SimulationState.IDLE
        self._current_time: float = 0.0
        self._event_counter: int = 0
        self._events: List[NetworkEvent] = []
        self._metrics_history: List[SimulationMetrics] = []

        # Initialize subsystems
        self._topology = self._setup_topology()
        self._traffic_sim = TrafficSimulator()
        self._failure_injector = FailureInjector()
        self._metrics_collector = MetricsCollector()
        self._agent = NetworkAgent(policy=self._config.agent_policy)

        # Set random seed
        if self._config.random_seed is not None:
            random.seed(self._config.random_seed)

        # Create initial traffic flows
        self._create_initial_traffic()

    @property
    def config(self) -> SimulationConfig:
        """Simulation configuration."""
        return self._config

    @property
    def state(self) -> SimulationState:
        """Current simulation state."""
        return self._state

    @property
    def current_time(self) -> float:
        """Current simulation time."""
        return self._current_time

    @property
    def topology(self) -> NetworkTopology:
        """Current network topology."""
        return self._topology

    @property
    def traffic_simulator(self) -> TrafficSimulator:
        """Traffic simulator instance."""
        return self._traffic_sim

    @property
    def failure_injector(self) -> FailureInjector:
        """Failure injector instance."""
        return self._failure_injector

    @property
    def metrics_collector(self) -> MetricsCollector:
        """Metrics collector instance."""
        return self._metrics_collector

    @property
    def agent(self) -> NetworkAgent:
        """The agentic AI agent."""
        return self._agent

    @property
    def events(self) -> Tuple[NetworkEvent, ...]:
        """All simulation events."""
        return tuple(self._events)

    @property
    def metrics_history(self) -> Tuple[SimulationMetrics, ...]:
        """All collected metrics."""
        return tuple(self._metrics_history)

    async def run(self) -> SimulationResult:
        """Run the simulation to completion.

        Returns:
            SimulationResult with all metrics and events.

        Raises:
            SimulationRuntimeError: If the simulation encounters an error.
        """
        self._state = SimulationState.RUNNING
        start_time = time.time()
        logger.info(
            "Starting simulation: %d nodes, %.0fs duration",
            self._config.node_count,
            self._config.duration_seconds,
        )

        try:
            num_steps = int(
                self._config.duration_seconds / self._config.time_step_seconds
            )
            for step in range(num_steps):
                if self._state != SimulationState.RUNNING:
                    break

                self._current_time = step * self._config.time_step_seconds

                # Generate events
                await self._generate_events()

                # Simulate telemetry
                await self._simulate_telemetry()

                # Simulate traffic
                self._traffic_sim.simulate_step(
                    self._topology, self._config.time_step_seconds
                )

                # Run agent
                await self._run_agent()

                # Collect metrics
                metrics = self._collect_metrics()
                self._metrics_history.append(metrics)

                # Sleep for real-time simulation
                await asyncio.sleep(self._config.time_step_seconds)

            self._state = SimulationState.COMPLETED
            actual_duration = time.time() - start_time
            logger.info(
                "Simulation completed in %.1fs (simulated %.0fs)",
                actual_duration,
                self._current_time,
            )

            return SimulationResult(
                config=self._config,
                final_metrics=self._metrics_history[-1]
                if self._metrics_history
                else SimulationMetrics(timestamp=time.time()),
                all_metrics=list(self._metrics_history),
                events=list(self._events),
                agent_actions=list(self._agent.action_history),
                duration_seconds=actual_duration,
                success=True,
            )

        except Exception as exc:
            self._state = SimulationState.ERROR
            logger.error("Simulation failed: %s", exc)
            raise SimulationRuntimeError(f"Simulation failed: {exc}") from exc

    async def step(self) -> Optional[SimulationMetrics]:
        """Execute a single simulation step.

        Returns:
            SimulationMetrics after the step, or None if simulation is done.
        """
        if self._state not in (SimulationState.RUNNING, SimulationState.IDLE):
            return None

        if self._state == SimulationState.IDLE:
            self._state = SimulationState.RUNNING

        self._current_time += self._config.time_step_seconds

        await self._generate_events()
        await self._simulate_telemetry()
        self._traffic_sim.simulate_step(
            self._topology, self._config.time_step_seconds
        )
        await self._run_agent()

        metrics = self._collect_metrics()
        self._metrics_history.append(metrics)

        if self._current_time >= self._config.duration_seconds:
            self._state = SimulationState.COMPLETED

        return metrics

    def pause(self) -> None:
        """Pause the simulation."""
        if self._state == SimulationState.RUNNING:
            self._state = SimulationState.PAUSED

    def resume(self) -> None:
        """Resume a paused simulation."""
        if self._state == SimulationState.PAUSED:
            self._state = SimulationState.RUNNING

    def stop(self) -> None:
        """Stop the simulation."""
        self._state = SimulationState.STOPPED

    def reset(self) -> None:
        """Reset the simulation to initial state."""
        self._state = SimulationState.IDLE
        self._current_time = 0.0
        self._event_counter = 0
        self._events.clear()
        self._metrics_history.clear()
        self._agent.clear_history()
        self._topology = self._setup_topology()
        self._traffic_sim.clear()
        self._failure_injector.clear()
        self._metrics_collector.clear()
        self._create_initial_traffic()

    def _setup_topology(self) -> NetworkTopology:
        """Set up the simulated network topology."""
        topology = NetworkTopology()

        # Create nodes
        roles = [
            NodeRole.ROUTER,
            NodeRole.SWITCH,
            NodeRole.BASE_STATION,
            NodeRole.GATEWAY,
        ]
        for i in range(self._config.node_count):
            node = NetworkNode(
                node_id=f"node-{i:03d}",
                role=random.choice(roles),
            )
            topology.add_node(node)

        # Create links based on density
        node_ids = list(topology.nodes.keys())
        for i, node_a in enumerate(node_ids):
            for node_b in node_ids[i + 1:]:
                if random.random() < self._config.link_density:
                    try:
                        topology.add_link(node_a, node_b)
                    except SimulationError:
                        pass

        return topology

    def _create_initial_traffic(self) -> None:
        """Create initial traffic flows."""
        node_ids = list(self._topology.nodes.keys())
        if len(node_ids) < 2:
            return

        # Create a few random flows
        num_flows = max(1, len(node_ids) // 3)
        for _ in range(num_flows):
            source, destination = random.sample(node_ids, 2)
            self._traffic_sim.create_flow(
                source=source,
                destination=destination,
                bandwidth_mbps=random.uniform(5.0, 50.0),
                protocol=random.choice(["TCP", "UDP", "HTTP"]),
                priority=random.randint(1, 10),
            )

    async def _generate_events(self) -> None:
        """Generate random network events."""
        # Node failures
        if random.random() < self._config.failure_rate:
            node_ids = list(self._topology.nodes.keys())
            if node_ids:
                failed_node = random.choice(node_ids)
                self._emit_event(
                    EventType.NODE_FAILURE,
                    failed_node,
                    f"Node {failed_node} failed",
                )
                try:
                    node = self._topology.nodes[failed_node]
                    node.status = NodeStatus.OFFLINE
                    node.health_score = 0.0
                    self._metrics_collector.record_failure(time.time())
                except KeyError:
                    pass

        # Node recovery
        offline_nodes = [
            n for n in self._topology.nodes.values()
            if n.status == NodeStatus.OFFLINE
        ]
        if offline_nodes and random.random() < 0.3:
            recovered = random.choice(offline_nodes)
            self._emit_event(
                EventType.NODE_RECOVERY,
                recovered.node_id,
                f"Node {recovered.node_id} recovered",
            )
            recovered.status = NodeStatus.HEALTHY
            recovered.health_score = 100.0
            self._metrics_collector.record_repair(time.time())

        # Traffic surges
        if random.random() < self._config.traffic_surge_probability:
            node_ids = list(self._topology.nodes.keys())
            if node_ids:
                surge_node = random.choice(node_ids)
                self._emit_event(
                    EventType.TRAFFIC_SURGE,
                    surge_node,
                    f"Traffic surge on {surge_node}",
                    metadata={"multiplier": random.uniform(2.0, 5.0)},
                )

        # Interference events
        if random.random() < self._config.interference_probability:
            self._emit_event(
                EventType.INTERFERENCE,
                description="Interference detected",
                metadata={"severity": random.choice(["low", "medium", "high"])},
            )

    async def _simulate_telemetry(self) -> None:
        """Generate and ingest telemetry for all nodes."""
        for node in self._topology.nodes.values():
            if node.status == NodeStatus.OFFLINE:
                continue

            # Generate realistic telemetry
            base_cpu = 30.0
            base_mem = 40.0
            base_loss = 0.1

            # Add some nodes with high utilization
            if random.random() < 0.1:
                base_cpu += 50.0
            if random.random() < 0.05:
                base_mem += 40.0

            # Traffic surge effect
            for event in self._events[-10:]:
                if (
                    event.event_type == EventType.TRAFFIC_SURGE
                    and event.node_id == node.node_id
                ):
                    base_cpu += 20.0
                    base_mem += 15.0

            node.cpu_utilization = min(100.0, max(0.0, base_cpu + random.gauss(0, 10)))
            node.memory_utilization = min(100.0, max(0.0, base_mem + random.gauss(0, 5)))
            node.packet_loss_rate = max(0.0, base_loss + random.gauss(0, 0.5))
            node.latency_ms = max(0.1, 5.0 + random.gauss(0, 2.0))
            node.throughput_mbps = max(0.0, 100.0 + random.gauss(0, 20.0))
            node.update_health()

    async def _run_agent(self) -> None:
        """Run the agentic AI agent for one step."""
        actions = await self._agent.observe_and_act(
            self._topology,
            self._traffic_sim,
            self._metrics_collector,
        )
        for action in actions:
            self._emit_event(
                EventType.AGENT_ACTION,
                action.target_node,
                f"Agent action: {action.action_type}",
                metadata={"action_id": action.action_id, "success": action.success},
            )

    def _collect_metrics(self) -> SimulationMetrics:
        """Collect current simulation metrics."""
        # Count nodes by status
        healthy = sum(
            1 for n in self._topology.nodes.values()
            if n.status == NodeStatus.HEALTHY
        )
        degraded = sum(
            1 for n in self._topology.nodes.values()
            if n.status == NodeStatus.DEGRADED
        )
        unhealthy = sum(
            1 for n in self._topology.nodes.values()
            if n.status == NodeStatus.UNHEALTHY
        )
        offline = sum(
            1 for n in self._topology.nodes.values()
            if n.status == NodeStatus.OFFLINE
        )

        # Average health score
        health_scores = [
            n.health_score for n in self._topology.nodes.values()
        ]
        avg_health = (
            statistics.mean(health_scores) if health_scores else 0.0
        )

        # Performance metrics
        perf = self._metrics_collector.collect(
            self._topology, self._traffic_sim
        )

        return SimulationMetrics(
            timestamp=self._current_time,
            total_nodes=len(self._topology.nodes),
            healthy_nodes=healthy,
            degraded_nodes=degraded,
            unhealthy_nodes=unhealthy,
            offline_nodes=offline,
            total_faults=self._failure_injector.total_failures_injected,
            total_remediations=len(self._agent.action_history),
            successful_remediations=sum(
                1 for a in self._agent.action_history if a.success
            ),
            average_health_score=avg_health,
            total_events=len(self._events),
            spectrum_utilization=perf.congestion_level,
            interference_events=len([
                e for e in self._events
                if e.event_type == EventType.INTERFERENCE
            ]),
            inference_count=0,
            average_inference_latency_ms=perf.average_latency_ms,
        )

    def _emit_event(
        self,
        event_type: EventType,
        node_id: Optional[str] = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NetworkEvent:
        """Emit a simulation event."""
        self._event_counter += 1
        event = NetworkEvent(
            event_id=f"EVT-{self._event_counter:06d}",
            event_type=event_type,
            timestamp=self._current_time,
            node_id=node_id,
            description=description,
            metadata=metadata or {},
        )
        self._events.append(event)
        self._agent.record_event(event)
        return event

    def __repr__(self) -> str:
        return (
            f"NetworkSimulator(state={self._state.value!r}, "
            f"time={self._current_time:.1f}s, "
            f"events={len(self._events)})"
        )


# ---------------------------------------------------------------------------
# SimulationRunner
# ---------------------------------------------------------------------------


class SimulationRunner:
    """High-level runner for simulation scenarios.

    Provides a convenient interface for running simulations with
    different configurations and collecting results.

    Attributes:
        results: List of simulation results from previous runs.
    """

    def __init__(self) -> None:
        """Initialize the SimulationRunner."""
        self._results: List[SimulationResult] = []

    @property
    def results(self) -> Tuple[SimulationResult, ...]:
        """Results from all simulation runs."""
        return tuple(self._results)

    async def run_single(
        self, config: Optional[SimulationConfig] = None
    ) -> SimulationResult:
        """Run a single simulation.

        Args:
            config: Simulation configuration (uses defaults if None).

        Returns:
            SimulationResult.
        """
        simulator = NetworkSimulator(config)
        result = await simulator.run()
        self._results.append(result)
        return result

    async def run_batch(
        self,
        configs: List[SimulationConfig],
        max_concurrent: int = 5,
    ) -> List[SimulationResult]:
        """Run multiple simulations concurrently.

        Args:
            configs: List of simulation configurations.
            max_concurrent: Maximum number of concurrent simulations.

        Returns:
            List of SimulationResult objects.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _run_with_limit(
            config: SimulationConfig,
        ) -> SimulationResult:
            async with semaphore:
                simulator = NetworkSimulator(config)
                result = await simulator.run()
                self._results.append(result)
                return result

        tasks = [_run_with_limit(cfg) for cfg in configs]
        return await asyncio.gather(*tasks)

    async def run_parameter_sweep(
        self,
        base_config: SimulationConfig,
        parameter_name: str,
        values: List[Any],
    ) -> List[SimulationResult]:
        """Run simulations sweeping a parameter over a range of values.

        Args:
            base_config: Base configuration.
            parameter_name: Name of the parameter to sweep.
            values: Values to sweep over.

        Returns:
            List of SimulationResult objects.
        """
        configs = []
        for value in values:
            config_dict = {
                "duration_seconds": base_config.duration_seconds,
                "time_step_seconds": base_config.time_step_seconds,
                "node_count": base_config.node_count,
                "link_density": base_config.link_density,
                "failure_rate": base_config.failure_rate,
                "traffic_surge_probability": base_config.traffic_surge_probability,
                "interference_probability": base_config.interference_probability,
                "enable_self_healing": base_config.enable_self_healing,
                "enable_predictive_maintenance": base_config.enable_predictive_maintenance,
                "enable_spectrum_management": base_config.enable_spectrum_management,
                "enable_edge_ai": base_config.enable_edge_ai,
                "agent_policy": base_config.agent_policy,
                "random_seed": base_config.random_seed,
            }
            config_dict[parameter_name] = value
            configs.append(SimulationConfig(**config_dict))

        return await self.run_batch(configs)

    def get_aggregate_statistics(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all runs.

        Returns:
            Dictionary with aggregate statistics.
        """
        if not self._results:
            return {}

        durations = [r.duration_seconds for r in self._results]
        final_metrics = [r.final_metrics for r in self._results]

        return {
            "total_runs": len(self._results),
            "successful_runs": sum(1 for r in self._results if r.success),
            "average_duration_seconds": statistics.mean(durations),
            "average_faults": statistics.mean(
                m.total_faults for m in final_metrics
            ),
            "average_remediations": statistics.mean(
                m.total_remediations for m in final_metrics
            ),
            "average_health_score": statistics.mean(
                m.average_health_score for m in final_metrics
            ),
            "average_spectrum_utilization": statistics.mean(
                m.spectrum_utilization for m in final_metrics
            ),
            "average_inference_latency_ms": statistics.mean(
                m.average_inference_latency_ms for m in final_metrics
            ),
        }

    def export_results(self, filepath: str) -> None:
        """Export all results to a JSON file.

        Args:
            filepath: Path to the output file.
        """
        data = {
            "aggregate_statistics": self.get_aggregate_statistics(),
            "results": [r.to_dict() for r in self._results],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Exported %d results to %s", len(self._results), filepath)

    def clear_results(self) -> None:
        """Clear all stored results."""
        self._results.clear()

    def __repr__(self) -> str:
        return f"SimulationRunner(runs={len(self._results)})"
