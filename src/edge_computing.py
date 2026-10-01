"""Apex Critical Infrastructure — Edge Computing.

Distributed AI inference, low-latency edge deployment, and workload
orchestration for edge computing environments.

Integrates with Apex_ULL (ultra-low latency), ApexGraphSwarm (multi-agent
orchestration), and Data Center Commander (DC lifecycle).
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Generic,
    Protocol,
    TypeVar,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EDGE_LATENCY_SLA_MS = 10.0
DEFAULT_INFERENCE_TIMEOUT_MS = 500.0
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_REPLICAS = 10
DEFAULT_MIN_REPLICAS = 1
DEFAULT_SCALE_UP_THRESHOLD_PCT = 80.0
DEFAULT_SCALE_DOWN_THRESHOLD_PCT = 30.0
DEFAULT_COOLDOWN_SECONDS = 60.0
MAX_EDGE_NODES = 10_000
DEFAULT_HEARTBEAT_INTERVAL_S = 30.0
DEFAULT_FAILOVER_TIMEOUT_S = 5.0


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EdgeNodeState(Enum):
    """Operational state of an edge node."""

    PROVISIONING = auto()
    ACTIVE = auto()
    BUSY = auto()
    DEGRADED = auto()
    DRAINING = auto()
    OFFLINE = auto()
    MAINTENANCE = auto()


class InferenceEngine(Enum):
    """Supported inference engine types."""

    ONNX_RUNTIME = "onnx_runtime"
    TENSORRT = "tensorrt"
    TORCH_SERVE = "torch_serve"
    TRITON = "triton"
    OPENVINO = "openvino"
    CUSTOM = "custom"


class DeploymentStrategy(Enum):
    """Workload deployment strategy."""

    LATENCY_OPTIMIZED = "latency_optimized"
    THROUGHPUT_OPTIMIZED = "throughput_optimized"
    COST_OPTIMIZED = "cost_optimized"
    RELIABILITY_OPTIMIZED = "reliability_optimized"
    BALANCED = "balanced"


class WorkloadType(Enum):
    """Type of edge workload."""

    INFERENCE = "inference"
    PREPROCESSING = "preprocessing"
    POSTPROCESSING = "postprocessing"
    TRAINING = "training"
    FEDERATED_LEARNING = "federated_learning"


class ScalingPolicy(Enum):
    """Autoscaling policy type."""

    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    SCHEDULED = "scheduled"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EdgeComputingError(Exception):
    """Base exception for edge computing."""


class NodeUnavailableError(EdgeComputingError):
    """Raised when no edge node is available for a request."""


class InferenceTimeoutError(EdgeComputingError):
    """Raised when an inference request exceeds its timeout."""


class DeploymentError(EdgeComputingError):
    """Raised when a deployment operation fails."""


class OrchestrationError(EdgeComputingError):
    """Raised when workload orchestration fails."""


class ModelLoadError(EdgeComputingError):
    """Raised when a model fails to load on an edge node."""


class QuotaExceededError(EdgeComputingError):
    """Raised when a resource quota is exceeded."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GeoLocation:
    """Geographic location."""

    latitude: float
    longitude: float
    altitude_m: float = 0.0
    region: str = ""
    zone: str = ""

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError(f"latitude {self.latitude} out of range [-90, 90]")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError(f"longitude {self.longitude} out of range [-180, 180]")

    def distance_to(self, other: GeoLocation) -> float:
        """Calculate Haversine distance to another location in kilometers."""
        import math

        R = 6371.0  # Earth radius in km
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    """Resource capacity profile."""

    cpu_cores: int
    memory_gb: float
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    storage_gb: float = 100.0
    network_mbps: float = 1000.0

    def __post_init__(self) -> None:
        if self.cpu_cores <= 0:
            raise ValueError("cpu_cores must be positive")
        if self.memory_gb <= 0:
            raise ValueError("memory_gb must be positive")
        if self.gpu_count < 0:
            raise ValueError("gpu_count must be non-negative")
        if self.gpu_memory_gb < 0:
            raise ValueError("gpu_memory_gb must be non-negative")
        if self.storage_gb <= 0:
            raise ValueError("storage_gb must be positive")
        if self.network_mbps <= 0:
            raise ValueError("network_mbps must be positive")

    @property
    def total_compute_units(self) -> float:
        """Total compute units (CPU + GPU weighted)."""
        return float(self.cpu_cores) + (self.gpu_count * 8.0)


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """An AI model artifact for edge deployment."""

    model_id: str
    name: str
    version: str
    engine: InferenceEngine
    size_mb: float
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if self.size_mb <= 0:
            raise ValueError("size_mb must be positive")


@dataclass(frozen=True, slots=True)
class EdgeNode:
    """An edge computing node."""

    node_id: str
    location: GeoLocation
    resources: ResourceProfile
    state: EdgeNodeState = EdgeNodeState.PROVISIONING
    labels: dict[str, str] = field(default_factory=dict)
    current_load_pct: float = 0.0
    active_models: list[str] = field(default_factory=list)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must be non-empty")
        if not 0.0 <= self.current_load_pct <= 100.0:
            raise ValueError(
                f"current_load_pct {self.current_load_pct} out of range [0, 100]"
            )

    @property
    def is_available(self) -> bool:
        """Whether the node is available for new work."""
        return self.state in (EdgeNodeState.ACTIVE, EdgeNodeState.BUSY)

    @property
    def is_healthy(self) -> bool:
        """Whether the node is healthy (recent heartbeat)."""
        elapsed = (datetime.utcnow() - self.last_heartbeat).total_seconds()
        return elapsed < DEFAULT_HEARTBEAT_INTERVAL_S * 3


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """An inference request."""

    request_id: str
    model_id: str
    input_data: bytes
    priority: int = 5  # 1 (highest) to 10 (lowest)
    max_latency_ms: float = DEFAULT_INFERENCE_TIMEOUT_MS
    preferred_node: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if not 1 <= self.priority <= 10:
            raise ValueError("priority must be in range [1, 10]")
        if self.max_latency_ms <= 0:
            raise ValueError("max_latency_ms must be positive")


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Result of an inference request."""

    request_id: str
    model_id: str
    node_id: str
    output_data: bytes
    latency_ms: float
    confidence: float | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in range [0, 1]")


@dataclass(frozen=True, slots=True)
class DeploymentSpec:
    """Specification for deploying a model to edge nodes."""

    deployment_id: str
    model: ModelArtifact
    target_nodes: list[str]
    strategy: DeploymentStrategy = DeploymentStrategy.BALANCED
    replicas: int = 1
    resource_fraction: float = 0.5  # fraction of node resources to allocate
    auto_scaling: bool = True
    scaling_policy: ScalingPolicy = ScalingPolicy.REACTIVE
    min_replicas: int = DEFAULT_MIN_REPLICAS
    max_replicas: int = DEFAULT_MAX_REPLICAS
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.deployment_id:
            raise ValueError("deployment_id must be non-empty")
        if not self.target_nodes:
            raise ValueError("target_nodes must be non-empty")
        if self.replicas < 1:
            raise ValueError("replicas must be >= 1")
        if not 0.0 < self.resource_fraction <= 1.0:
            raise ValueError("resource_fraction must be in range (0, 1]")
        if self.min_replicas < 1:
            raise ValueError("min_replicas must be >= 1")
        if self.max_replicas < self.min_replicas:
            raise ValueError("max_replicas must be >= min_replicas")


@dataclass(frozen=True, slots=True)
class DeploymentStatus:
    """Status of a model deployment."""

    deployment_id: str
    model_id: str
    desired_replicas: int
    ready_replicas: int
    available_replicas: int
    node_assignments: dict[str, list[str]]  # node_id -> list of replica IDs
    avg_latency_ms: float
    total_requests: int
    error_count: int
    last_updated: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_fully_ready(self) -> bool:
        """Whether all desired replicas are ready."""
        return self.ready_replicas >= self.desired_replicas

    @property
    def error_rate(self) -> float:
        """Error rate as a fraction."""
        if self.total_requests == 0:
            return 0.0
        return self.error_count / self.total_requests


@dataclass(frozen=True, slots=True)
class WorkloadMetrics:
    """Metrics for a running workload."""

    workload_id: str
    requests_per_second: float
    avg_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    cpu_utilization_pct: float
    memory_utilization_pct: float
    gpu_utilization_pct: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class InferenceBackend(Protocol):
    """Protocol for inference execution backends."""

    def load_model(self, model: ModelArtifact, node_id: str) -> bool:
        """Load a model onto a node."""
        ...

    def unload_model(self, model_id: str, node_id: str) -> bool:
        """Unload a model from a node."""
        ...

    def infer(self, request: InferenceRequest, node_id: str) -> InferenceResult:
        """Execute an inference request."""
        ...

    def health_check(self, node_id: str) -> bool:
        """Check if a node is healthy."""
        ...


class ModelRegistry(Protocol):
    """Protocol for model registry."""

    def get_model(self, model_id: str) -> ModelArtifact | None:
        """Retrieve a model artifact by ID."""
        ...

    def list_models(self, engine: InferenceEngine | None = None) -> list[ModelArtifact]:
        """List available models, optionally filtered by engine."""
        ...


# ---------------------------------------------------------------------------
# Edge Node Manager
# ---------------------------------------------------------------------------


class EdgeNodeManager:
    """Manages the fleet of edge nodes.

    Handles node registration, health monitoring, load tracking, and
    node selection for workload placement.

    Parameters
    ----------
    heartbeat_interval_s:
        Expected heartbeat interval in seconds.
    failover_timeout_s:
        Timeout for node failover operations.
    """

    def __init__(
        self,
        *,
        heartbeat_interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S,
        failover_timeout_s: float = DEFAULT_FAILOVER_TIMEOUT_S,
    ) -> None:
        if heartbeat_interval_s <= 0:
            raise ValueError("heartbeat_interval_s must be positive")
        if failover_timeout_s <= 0:
            raise ValueError("failover_timeout_s must be positive")

        self._nodes: dict[str, EdgeNode] = {}
        self._heartbeat_interval_s = heartbeat_interval_s
        self._failover_timeout_s = failover_timeout_s
        self._load_history: dict[str, list[tuple[datetime, float]]] = {}

    @property
    def node_count(self) -> int:
        """Total number of registered nodes."""
        return len(self._nodes)

    @property
    def active_nodes(self) -> list[EdgeNode]:
        """Currently active nodes."""
        return [n for n in self._nodes.values() if n.is_available]

    def register_node(self, node: EdgeNode) -> None:
        """Register a new edge node.

        Parameters
        ----------
        node:
            The node to register.

        Raises
        ------
        ValueError:
            If a node with the same ID is already registered.
        """
        if node.node_id in self._nodes:
            raise ValueError(f"Node {node.node_id} already registered")
        self._nodes[node.node_id] = node
        self._load_history[node.node_id] = []
        logger.info("Edge node registered: %s (%s)", node.node_id, node.location.region)

    def deregister_node(self, node_id: str) -> EdgeNode | None:
        """Deregister an edge node.

        Returns
        -------
        EdgeNode | None
            The removed node, or None if not found.
        """
        node = self._nodes.pop(node_id, None)
        self._load_history.pop(node_id, None)
        if node:
            logger.info("Edge node deregistered: %s", node_id)
        return node

    def get_node(self, node_id: str) -> EdgeNode | None:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def update_node_state(self, node_id: str, state: EdgeNodeState) -> None:
        """Update the state of a node."""
        node = self._nodes.get(node_id)
        if node is None:
            raise ValueError(f"Unknown node: {node_id}")
        # Create updated node (dataclass is frozen)
        updated = EdgeNode(
            node_id=node.node_id,
            location=node.location,
            resources=node.resources,
            state=state,
            labels=node.labels,
            current_load_pct=node.current_load_pct,
            active_models=node.active_models,
            last_heartbeat=node.last_heartbeat,
        )
        self._nodes[node_id] = updated

    def update_node_load(self, node_id: str, load_pct: float) -> None:
        """Update the load percentage of a node."""
        if not 0.0 <= load_pct <= 100.0:
            raise ValueError(f"load_pct {load_pct} out of range [0, 100]")
        node = self._nodes.get(node_id)
        if node is None:
            raise ValueError(f"Unknown node: {node_id}")
        updated = EdgeNode(
            node_id=node.node_id,
            location=node.location,
            resources=node.resources,
            state=node.state,
            labels=node.labels,
            current_load_pct=load_pct,
            active_models=node.active_models,
            last_heartbeat=node.last_heartbeat,
        )
        self._nodes[node_id] = updated
        self._load_history.setdefault(node_id, []).append((datetime.utcnow(), load_pct))

    def heartbeat(self, node_id: str) -> None:
        """Record a heartbeat from a node."""
        node = self._nodes.get(node_id)
        if node is None:
            raise ValueError(f"Unknown node: {node_id}")
        updated = EdgeNode(
            node_id=node.node_id,
            location=node.location,
            resources=node.resources,
            state=node.state,
            labels=node.labels,
            current_load_pct=node.current_load_pct,
            active_models=node.active_models,
            last_heartbeat=datetime.utcnow(),
        )
        self._nodes[node_id] = updated

    def get_unhealthy_nodes(self) -> list[EdgeNode]:
        """Get nodes that have missed heartbeats."""
        return [n for n in self._nodes.values() if not n.is_healthy]

    def select_node(
        self,
        *,
        model_id: str | None = None,
        location: GeoLocation | None = None,
        max_latency_ms: float = DEFAULT_EDGE_LATENCY_SLA_MS,
        required_resources: ResourceProfile | None = None,
    ) -> EdgeNode | None:
        """Select the best available node for a request.

        Parameters
        ----------
        model_id:
            If specified, only consider nodes with this model loaded.
        location:
            If specified, prefer nodes closer to this location.
        max_latency_ms:
            Maximum acceptable latency.
        required_resources:
            Minimum resource requirements.

        Returns
        -------
        EdgeNode | None
            The selected node, or None if no suitable node is available.
        """
        candidates = self.active_nodes

        if not candidates:
            return None

        # Filter by model availability
        if model_id is not None:
            candidates = [n for n in candidates if model_id in n.active_models]
            if not candidates:
                return None

        # Filter by resource requirements
        if required_resources is not None:
            candidates = [
                n
                for n in candidates
                if n.resources.cpu_cores >= required_resources.cpu_cores
                and n.resources.memory_gb >= required_resources.memory_gb
                and n.resources.gpu_count >= required_resources.gpu_count
            ]
            if not candidates:
                return None

        # Score and rank
        scored: list[tuple[float, EdgeNode]] = []
        for node in candidates:
            score = 0.0

            # Prefer lower load
            score += (100.0 - node.current_load_pct) * 0.4

            # Prefer closer nodes
            if location is not None:
                dist = node.location.distance_to(location)
                score += max(0.0, 100.0 - dist) * 0.3

            # Prefer nodes with more headroom
            score += (100.0 - node.current_load_pct) * 0.3

            scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else None

    def get_fleet_summary(self) -> dict[str, Any]:
        """Get a summary of the entire fleet."""
        total = len(self._nodes)
        by_state: dict[str, int] = {}
        total_load = 0.0
        for node in self._nodes.values():
            by_state[node.state.name] = by_state.get(node.state.name, 0) + 1
            total_load += node.current_load_pct

        return {
            "total_nodes": total,
            "active_nodes": len(self.active_nodes),
            "unhealthy_nodes": len(self.get_unhealthy_nodes()),
            "by_state": by_state,
            "avg_load_pct": total_load / total if total > 0 else 0.0,
        }


# ---------------------------------------------------------------------------
# Distributed Inference Engine
# ---------------------------------------------------------------------------


class DistributedInferenceEngine:
    """Distributed AI inference across edge nodes.

    Routes inference requests to optimal edge nodes, handles failover,
    and provides latency-aware load balancing.

    Parameters
    ----------
    node_manager:
        The edge node manager.
    backend:
        The inference backend.
    """

    def __init__(
        self,
        node_manager: EdgeNodeManager,
        backend: InferenceBackend,
    ) -> None:
        self._node_manager = node_manager
        self._backend = backend
        self._request_count = 0
        self._error_count = 0
        self._total_latency_ms = 0.0

    @property
    def total_requests(self) -> int:
        """Total requests processed."""
        return self._request_count

    @property
    def error_count(self) -> int:
        """Total errors encountered."""
        return self._error_count

    @property
    def avg_latency_ms(self) -> float:
        """Average latency across all requests."""
        if self._request_count == 0:
            return 0.0
        return self._total_latency_ms / self._request_count

    def deploy_model(
        self,
        model: ModelArtifact,
        target_nodes: list[str],
        *,
        resource_fraction: float = 0.5,
    ) -> list[str]:
        """Deploy a model to target edge nodes.

        Parameters
        ----------
        model:
            The model artifact to deploy.
        target_nodes:
            Target node IDs.
        resource_fraction:
            Fraction of node resources to allocate.

        Returns
        -------
        list[str]
            Node IDs where the model was successfully deployed.

        Raises
        ------
        DeploymentError:
            If the deployment fails on all target nodes.
        """
        if not 0.0 < resource_fraction <= 1.0:
            raise ValueError("resource_fraction must be in range (0, 1]")

        deployed: list[str] = []
        errors: list[str] = []

        for node_id in target_nodes:
            try:
                node = self._node_manager.get_node(node_id)
                if node is None:
                    errors.append(f"Node {node_id} not found")
                    continue
                if not node.is_available:
                    errors.append(f"Node {node_id} not available (state={node.state.name})")
                    continue

                success = self._backend.load_model(model, node_id)
                if success:
                    # Update node's active models
                    new_models = list(node.active_models)
                    if model.model_id not in new_models:
                        new_models.append(model.model_id)
                    updated = EdgeNode(
                        node_id=node.node_id,
                        location=node.location,
                        resources=node.resources,
                        state=node.state,
                        labels=node.labels,
                        current_load_pct=node.current_load_pct,
                        active_models=new_models,
                        last_heartbeat=node.last_heartbeat,
                    )
                    self._node_manager._nodes[node_id] = updated
                    deployed.append(node_id)
                    logger.info("Model %s deployed to node %s", model.model_id, node_id)
                else:
                    errors.append(f"Failed to load model on node {node_id}")
            except Exception as exc:
                errors.append(f"Node {node_id}: {exc}")
                logger.exception("Deployment error on node %s", node_id)

        if not deployed and errors:
            raise DeploymentError(
                f"Model deployment failed on all target nodes: {'; '.join(errors)}"
            )

        return deployed

    def undeploy_model(self, model_id: str, target_nodes: list[str]) -> list[str]:
        """Remove a model from target edge nodes.

        Returns
        -------
        list[str]
            Node IDs where the model was successfully removed.
        """
        removed: list[str] = []
        for node_id in target_nodes:
            try:
                node = self._node_manager.get_node(node_id)
                if node is None:
                    continue
                success = self._backend.unload_model(model_id, node_id)
                if success:
                    new_models = [m for m in node.active_models if m != model_id]
                    updated = EdgeNode(
                        node_id=node.node_id,
                        location=node.location,
                        resources=node.resources,
                        state=node.state,
                        labels=node.labels,
                        current_load_pct=node.current_load_pct,
                        active_models=new_models,
                        last_heartbeat=node.last_heartbeat,
                    )
                    self._node_manager._nodes[node_id] = updated
                    removed.append(node_id)
            except Exception:
                logger.exception("Error undeploying model from node %s", node_id)

        return removed

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Execute an inference request.

        Parameters
        ----------
        request:
            The inference request.

        Returns
        -------
        InferenceResult
            The inference result.

        Raises
        ------
        NodeUnavailableError:
            If no suitable node is available.
        InferenceTimeoutError:
            If the inference exceeds the timeout.
        """
        self._request_count += 1
        start = time.monotonic()

        # Select target node
        node = self._node_manager.select_node(
            model_id=request.model_id,
            max_latency_ms=request.max_latency_ms,
        )

        if node is None:
            self._error_count += 1
            raise NodeUnavailableError(
                f"No available node for model {request.model_id}"
            )

        # Check preferred node
        if request.preferred_node:
            preferred = self._node_manager.get_node(request.preferred_node)
            if preferred and preferred.is_available:
                node = preferred

        try:
            result = self._backend.infer(request, node.node_id)
            elapsed_ms = (time.monotonic() - start) * 1000.0
            self._total_latency_ms += elapsed_ms

            if elapsed_ms > request.max_latency_ms:
                raise InferenceTimeoutError(
                    f"Inference took {elapsed_ms:.1f}ms, "
                    f"exceeding limit of {request.max_latency_ms}ms"
                )

            return result
        except InferenceTimeoutError:
            self._error_count += 1
            raise
        except Exception as exc:
            self._error_count += 1
            raise EdgeComputingError(f"Inference failed: {exc}") from exc

    def get_stats(self) -> dict[str, Any]:
        """Get inference engine statistics."""
        return {
            "total_requests": self._request_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / self._request_count if self._request_count > 0 else 0.0,
            "avg_latency_ms": self.avg_latency_ms,
        }


# ---------------------------------------------------------------------------
# Workload Orchestrator
# ---------------------------------------------------------------------------


class WorkloadOrchestrator:
    """Orchestrates workloads across the edge fleet.

    Handles deployment lifecycle, autoscaling, rolling updates, and
    workload migration.

    Parameters
    ----------
    node_manager:
        The edge node manager.
    inference_engine:
        The distributed inference engine.
    """

    def __init__(
        self,
        node_manager: EdgeNodeManager,
        inference_engine: DistributedInferenceEngine,
    ) -> None:
        self._node_manager = node_manager
        self._inference_engine = inference_engine
        self._deployments: dict[str, DeploymentStatus] = {}
        self._workload_metrics: dict[str, list[WorkloadMetrics]] = {}

    @property
    def deployments(self) -> dict[str, DeploymentStatus]:
        """Current deployments."""
        return dict(self._deployments)

    def deploy(self, spec: DeploymentSpec) -> DeploymentStatus:
        """Deploy a model according to a deployment specification.

        Parameters
        ----------
        spec:
            The deployment specification.

        Returns
        -------
        DeploymentStatus
            The initial deployment status.

        Raises
        ------
        DeploymentError:
            If the deployment fails.
        """
        if spec.deployment_id in self._deployments:
            raise DeploymentError(
                f"Deployment {spec.deployment_id} already exists"
            )

        # Deploy model to target nodes
        deployed_nodes = self._inference_engine.deploy_model(
            spec.model,
            spec.target_nodes,
            resource_fraction=spec.resource_fraction,
        )

        if not deployed_nodes:
            raise DeploymentError(
                f"Deployment {spec.deployment_id}: no nodes available"
            )

        # Create deployment status
        status = DeploymentStatus(
            deployment_id=spec.deployment_id,
            model_id=spec.model.model_id,
            desired_replicas=spec.replicas,
            ready_replicas=len(deployed_nodes),
            available_replicas=len(deployed_nodes),
            node_assignments={nid: [f"{spec.deployment_id}-r{i}"] for i, nid in enumerate(deployed_nodes)},
            avg_latency_ms=0.0,
            total_requests=0,
            error_count=0,
        )

        self._deployments[spec.deployment_id] = status
        logger.info(
            "Deployment %s created: model=%s, nodes=%d",
            spec.deployment_id,
            spec.model.model_id,
            len(deployed_nodes),
        )
        return status

    def undeploy(self, deployment_id: str) -> bool:
        """Remove a deployment.

        Returns
        -------
        bool
            True if the deployment was removed.
        """
        status = self._deployments.get(deployment_id)
        if status is None:
            return False

        all_nodes = [
            nid for nids in status.node_assignments.values() for nid in nids
        ]
        self._inference_engine.undeploy_model(status.model_id, all_nodes)
        del self._deployments[deployment_id]
        self._workload_metrics.pop(deployment_id, None)
        logger.info("Deployment %s removed", deployment_id)
        return True

    def scale(self, deployment_id: str, target_replicas: int) -> DeploymentStatus:
        """Scale a deployment to a target replica count.

        Parameters
        ----------
        deployment_id:
            The deployment to scale.
        target_replicas:
            Desired number of replicas.

        Returns
        -------
        DeploymentStatus
            Updated deployment status.

        Raises
        ------
        DeploymentError:
            If the deployment is not found or scaling fails.
        """
        status = self._deployments.get(deployment_id)
        if status is None:
            raise DeploymentError(f"Deployment {deployment_id} not found")

        if target_replicas < 1:
            raise ValueError("target_replicas must be >= 1")

        current = status.ready_replicas
        if target_replicas == current:
            return status

        if target_replicas > current:
            # Scale up: find additional nodes
            needed = target_replicas - current
            available = [
                n for n in self._node_manager.active_nodes
                if n.node_id not in status.node_assignments
            ]
            if len(available) < needed:
                raise DeploymentError(
                    f"Cannot scale to {target_replicas}: only {len(available)} nodes available"
                )
            new_nodes = [n.node_id for n in available[:needed]]
            self._inference_engine.deploy_model(
                ModelArtifact(
                    model_id=status.model_id,
                    name=status.model_id,
                    version="",
                    engine=InferenceEngine.CUSTOM,
                    size_mb=0,
                    input_shape=(),
                    output_shape=(),
                ),
                new_nodes,
            )
        else:
            # Scale down: remove excess replicas
            excess = current - target_replicas
            nodes_to_remove: list[str] = []
            for nid, replicas in list(status.node_assignments.items()):
                if excess <= 0:
                    break
                nodes_to_remove.append(nid)
                excess -= 1
            self._inference_engine.undeploy_model(status.model_id, nodes_to_remove)

        # Rebuild status
        new_assignments: dict[str, list[str]] = {}
        ready = 0
        for nid, replicas in status.node_assignments.items():
            if nid not in nodes_to_remove if target_replicas < current else True:
                new_assignments[nid] = replicas
                ready += 1

        updated = DeploymentStatus(
            deployment_id=status.deployment_id,
            model_id=status.model_id,
            desired_replicas=target_replicas,
            ready_replicas=ready,
            available_replicas=ready,
            node_assignments=new_assignments,
            avg_latency_ms=status.avg_latency_ms,
            total_requests=status.total_requests,
            error_count=status.error_count,
        )
        self._deployments[deployment_id] = updated
        return updated

    def record_metrics(self, metrics: WorkloadMetrics) -> None:
        """Record workload metrics for autoscaling decisions."""
        self._workload_metrics.setdefault(metrics.workload_id, []).append(metrics)
        # Keep last 1000 data points
        if len(self._workload_metrics[metrics.workload_id]) > 1000:
            self._workload_metrics[metrics.workload_id] = self._workload_metrics[metrics.workload_id][-1000:]

    def evaluate_autoscaling(self, deployment_id: str) -> int | None:
        """Evaluate whether a deployment should be scaled.

        Parameters
        ----------
        deployment_id:
            The deployment to evaluate.

        Returns
        -------
        int | None
            Target replica count if scaling is needed, None otherwise.
        """
        status = self._deployments.get(deployment_id)
        if status is None:
            return None

        metrics_list = self._workload_metrics.get(deployment_id, [])
        if not metrics_list:
            return None

        recent = metrics_list[-10:]
        avg_cpu = sum(m.cpu_utilization_pct for m in recent) / len(recent)
        avg_latency = sum(m.avg_latency_ms for m in recent) / len(recent)

        # Simple reactive autoscaling
        if avg_cpu > DEFAULT_SCALE_UP_THRESHOLD_PCT:
            return min(status.desired_replicas + 1, DEFAULT_MAX_REPLICAS)
        if avg_cpu < DEFAULT_SCALE_DOWN_THRESHOLD_PCT:
            return max(status.desired_replicas - 1, DEFAULT_MIN_REPLICAS)

        return None

    def get_deployment_summary(self) -> dict[str, Any]:
        """Get a summary of all deployments."""
        return {
            "total_deployments": len(self._deployments),
            "fully_ready": sum(1 for d in self._deployments.values() if d.is_fully_ready),
            "total_requests": sum(d.total_requests for d in self._deployments.values()),
            "total_errors": sum(d.error_count for d in self._deployments.values()),
            "deployments": {
                did: {
                    "model_id": d.model_id,
                    "desired": d.desired_replicas,
                    "ready": d.ready_replicas,
                    "error_rate": d.error_rate,
                }
                for did, d in self._deployments.items()
            },
        }


# ---------------------------------------------------------------------------
# Low-Latency Edge Deployer
# ---------------------------------------------------------------------------


class LowLatencyEdgeDeployer:
    """Deploys models to edge nodes optimized for ultra-low latency.

    Uses geographic proximity, network topology awareness, and
    predictive placement to minimize inference latency.

    Parameters
    ----------
    node_manager:
        The edge node manager.
    inference_engine:
        The distributed inference engine.
    """

    def __init__(
        self,
        node_manager: EdgeNodeManager,
        inference_engine: DistributedInferenceEngine,
    ) -> None:
        self._node_manager = node_manager
        self._inference_engine = inference_engine
        self._latency_sla_ms = DEFAULT_EDGE_LATENCY_SLA_MS

    @property
    def latency_sla_ms(self) -> float:
        """Current latency SLA in milliseconds."""
        return self._latency_sla_ms

    def set_latency_sla(self, sla_ms: float) -> None:
        """Set the latency SLA.

        Parameters
        ----------
        sla_ms:
            Target latency in milliseconds.
        """
        if sla_ms <= 0:
            raise ValueError("sla_ms must be positive")
        self._latency_sla_ms = sla_ms

    def deploy_for_latency(
        self,
        model: ModelArtifact,
        client_locations: list[GeoLocation],
        *,
        replicas_per_region: int = 2,
    ) -> dict[str, list[str]]:
        """Deploy a model optimized for low latency across client regions.

        Parameters
        ----------
        model:
            The model to deploy.
        client_locations:
            Client geographic locations to optimize for.
        replicas_per_region:
            Number of replicas per region.

        Returns
        -------
        dict[str, list[str]]
            Mapping of region names to deployed node IDs.
        """
        if not client_locations:
            raise ValueError("client_locations must be non-empty")

        # Group nodes by region
        nodes_by_region: dict[str, list[EdgeNode]] = {}
        for node in self._node_manager.active_nodes:
            region = node.location.region or "unknown"
            nodes_by_region.setdefault(region, []).append(node)

        # For each client location, find the closest region with capacity
        deployments: dict[str, list[str]] = {}
        for client_loc in client_locations:
            # Find closest region
            best_region: str | None = None
            best_dist = float("inf")
            for region, nodes in nodes_by_region.items():
                if not nodes:
                    continue
                dist = nodes[0].location.distance_to(client_loc)
                if dist < best_dist:
                    best_dist = dist
                    best_region = region

            if best_region is None:
                logger.warning("No available region for client at %s", client_loc)
                continue

            # Select nodes in the best region
            region_nodes = nodes_by_region[best_region]
            target = region_nodes[:replicas_per_region]
            target_ids = [n.node_id for n in target]

            deployed = self._inference_engine.deploy_model(model, target_ids)
            deployments[best_region] = deployed

        return deployments

    def route_request(
        self,
        request: InferenceRequest,
        client_location: GeoLocation,
    ) -> InferenceResult:
        """Route an inference request to the lowest-latency node.

        Parameters
        ----------
        request:
            The inference request.
        client_location:
            The client's geographic location.

        Returns
        -------
        InferenceResult
            The inference result.
        """
        node = self._node_manager.select_node(
            model_id=request.model_id,
            location=client_location,
            max_latency_ms=self._latency_sla_ms,
        )

        if node is None:
            raise NodeUnavailableError(
                f"No node available within {self._latency_sla_ms}ms SLA "
                f"for model {request.model_id}"
            )

        # Create a new request with the preferred node
        routed_request = InferenceRequest(
            request_id=request.request_id,
            model_id=request.model_id,
            input_data=request.input_data,
            priority=request.priority,
            max_latency_ms=request.max_latency_ms,
            preferred_node=node.node_id,
            metadata=request.metadata,
        )

        return self._inference_engine.infer(routed_request)


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    "EdgeComputingError",
    "EdgeNode",
    "EdgeNodeManager",
    "EdgeNodeState",
    "InferenceBackend",
    "InferenceEngine",
    "InferenceRequest",
    "InferenceResult",
    "InferenceTimeoutError",
    "DeploymentError",
    "DeploymentSpec",
    "DeploymentStatus",
    "DeploymentStrategy",
    "DistributedInferenceEngine",
    "GeoLocation",
    "LowLatencyEdgeDeployer",
    "ModelArtifact",
    "ModelLoadError",
    "ModelRegistry",
    "NodeUnavailableError",
    "OrchestrationError",
    "QuotaExceededError",
    "ResourceProfile",
    "ScalingPolicy",
    "WorkloadMetrics",
    "WorkloadOrchestrator",
    "WorkloadType",
]
