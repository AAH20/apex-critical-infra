"""Apex Critical Infrastructure — Edge AI.

Distributed AI at the edge, MEC (Multi-access Edge Computing) orchestration,
and low-latency inference for telecom and network infrastructure.

This module provides:
- EdgeNode: Represents an edge computing node with AI capabilities.
- MECOrchestrator: Orchestrates workloads across edge nodes.
- InferenceEngine: Manages AI model inference with latency optimization.
- ModelRegistry: Registry of AI models available for edge deployment.
- WorkloadScheduler: Schedules AI workloads across the edge infrastructure.
- EdgeAIManager: High-level orchestrator for edge AI operations.

All classes are fully typed, documented, and raise explicit exceptions
on invalid input or unrecoverable states.
"""

from __future__ import annotations

import asyncio
import enum
import hashlib
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


class EdgeAIError(Exception):
    """Base exception for all edge AI errors."""


class NodeNotFoundError(EdgeAIError):
    """Raised when a referenced edge node does not exist."""


class ModelNotFoundError(EdgeAIError):
    """Raised when a referenced AI model does not exist."""


class DeploymentError(EdgeAIError):
    """Raised when model deployment fails."""


class InferenceError(EdgeAIError):
    """Raised when inference execution fails."""


class SchedulingError(EdgeAIError):
    """Raised when workload scheduling fails."""


class ResourceExhaustedError(EdgeAIError):
    """Raised when edge node resources are insufficient."""


class LatencyBudgetExceededError(EdgeAIError):
    """Raised when inference latency exceeds the budget."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class NodeStatus(enum.Enum):
    """Operational status of an edge node."""

    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"


class NodeCapability(enum.Enum):
    """Hardware capabilities of an edge node."""

    GPU = "gpu"
    TPU = "tpu"
    FPGA = "fpga"
    CPU = "cpu"
    NPU = "npu"


class ModelFormat(enum.Enum):
    """Supported AI model formats."""

    ONNX = "onnx"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    TFLITE = "tflite"
    OPENVINO = "openvino"
    TENSORRT = "tensorrt"


class WorkloadPriority(enum.Enum):
    """Priority levels for AI workloads."""

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class SchedulingStrategy(enum.Enum):
    """Workload scheduling strategies."""

    LATENCY_OPTIMIZED = "latency_optimized"
    THROUGHPUT_OPTIMIZED = "throughput_optimized"
    ENERGY_EFFICIENT = "energy_efficient"
    COST_OPTIMIZED = "cost_optimized"
    FAIR_SHARING = "fair_sharing"
    CAPABILITY_MATCHED = "capability_matched"


class InferenceOptimization(enum.Enum):
    """Inference optimization techniques."""

    NONE = "none"
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    BATCHING = "batching"
    CACHING = "caching"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    """Resource capacity of an edge node.

    Attributes:
        cpu_cores: Number of CPU cores.
        memory_gb: Memory in gigabytes.
        gpu_memory_gb: GPU memory in gigabytes (0 if no GPU).
        compute_tflops: Compute capacity in TFLOPS.
        storage_gb: Storage in gigabytes.
        network_bandwidth_mbps: Network bandwidth in Mbps.
    """

    cpu_cores: int = 4
    memory_gb: float = 8.0
    gpu_memory_gb: float = 0.0
    compute_tflops: float = 1.0
    storage_gb: float = 64.0
    network_bandwidth_mbps: float = 100.0

    def __post_init__(self) -> None:
        if self.cpu_cores < 0:
            raise ValueError("cpu_cores must be non-negative")
        if self.memory_gb < 0:
            raise ValueError("memory_gb must be non-negative")
        if self.gpu_memory_gb < 0:
            raise ValueError("gpu_memory_gb must be non-negative")
        if self.compute_tflops < 0:
            raise ValueError("compute_tflops must be non-negative")
        if self.storage_gb < 0:
            raise ValueError("storage_gb must be non-negative")
        if self.network_bandwidth_mbps < 0:
            raise ValueError("network_bandwidth_mbps must be non-negative")

    @property
    def has_gpu(self) -> bool:
        """Whether this node has GPU acceleration."""
        return self.gpu_memory_gb > 0

    @property
    def has_tpu(self) -> bool:
        """Whether this node has TPU acceleration."""
        return self.compute_tflops > 10.0 and not self.has_gpu

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "gpu_memory_gb": self.gpu_memory_gb,
            "compute_tflops": self.compute_tflops,
            "storage_gb": self.storage_gb,
            "network_bandwidth_mbps": self.network_bandwidth_mbps,
            "has_gpu": self.has_gpu,
        }


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """Current resource usage on an edge node.

    Attributes:
        cpu_percent: CPU utilization percentage (0-100).
        memory_used_gb: Memory used in GB.
        gpu_memory_used_gb: GPU memory used in GB.
        gpu_util_percent: GPU utilization percentage (0-100).
        storage_used_gb: Storage used in GB.
        network_used_mbps: Network bandwidth used in Mbps.
        active_inferences: Number of active inference requests.
    """

    cpu_percent: float = 0.0
    memory_used_gb: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_util_percent: float = 0.0
    storage_used_gb: float = 0.0
    network_used_mbps: float = 0.0
    active_inferences: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.cpu_percent <= 100.0:
            raise ValueError(f"cpu_percent must be in [0, 100], got {self.cpu_percent}")
        if not 0.0 <= self.gpu_util_percent <= 100.0:
            raise ValueError(
                f"gpu_util_percent must be in [0, 100], got {self.gpu_util_percent}"
            )

    @property
    def is_overloaded(self) -> bool:
        """Whether the node is overloaded."""
        return self.cpu_percent > 90.0 or self.gpu_util_percent > 95.0

    @property
    def available_memory_gb(self) -> float:
        """Available memory (requires context of total)."""
        return max(0.0, self.memory_used_gb)  # Placeholder

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "cpu_percent": self.cpu_percent,
            "memory_used_gb": self.memory_used_gb,
            "gpu_memory_used_gb": self.gpu_memory_used_gb,
            "gpu_util_percent": self.gpu_util_percent,
            "storage_used_gb": self.storage_used_gb,
            "network_used_mbps": self.network_used_mbps,
            "active_inferences": self.active_inferences,
            "is_overloaded": self.is_overloaded,
        }


@dataclass(frozen=True, slots=True)
class AIModel:
    """An AI model available for edge deployment.

    Attributes:
        model_id: Unique identifier.
        name: Human-readable name.
        version: Model version string.
        format: Model format.
        size_mb: Model size in megabytes.
        required_compute_tflops: Minimum compute required in TFLOPS.
        required_memory_gb: Minimum memory required in GB.
        required_gpu: Whether GPU is required.
        latency_target_ms: Target inference latency in milliseconds.
        accuracy: Model accuracy metric (0.0-1.0).
        tags: Optional tags for categorization.
    """

    model_id: str
    name: str
    version: str = "1.0.0"
    format: ModelFormat = ModelFormat.ONNX
    size_mb: float = 10.0
    required_compute_tflops: float = 0.1
    required_memory_gb: float = 0.5
    required_gpu: bool = False
    latency_target_ms: float = 50.0
    accuracy: float = 0.95
    tags: FrozenSet[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if not self.name:
            raise ValueError("name must be non-empty")
        if self.size_mb <= 0:
            raise ValueError("size_mb must be positive")
        if self.required_compute_tflops < 0:
            raise ValueError("required_compute_tflops must be non-negative")
        if self.required_memory_gb < 0:
            raise ValueError("required_memory_gb must be non-negative")
        if self.latency_target_ms <= 0:
            raise ValueError("latency_target_ms must be positive")
        if not 0.0 <= self.accuracy <= 1.0:
            raise ValueError(f"accuracy must be in [0, 1], got {self.accuracy}")

    @property
    def fingerprint(self) -> str:
        """Generate a unique fingerprint for this model version."""
        data = f"{self.model_id}:{self.version}:{self.size_mb}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "format": self.format.value,
            "size_mb": self.size_mb,
            "required_compute_tflops": self.required_compute_tflops,
            "required_memory_gb": self.required_memory_gb,
            "required_gpu": self.required_gpu,
            "latency_target_ms": self.latency_target_ms,
            "accuracy": self.accuracy,
            "tags": list(self.tags),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """An inference request to be executed on an edge node.

    Attributes:
        request_id: Unique identifier.
        model_id: The model to run inference with.
        input_data_hash: Hash of the input data (for caching).
        priority: Workload priority.
        latency_budget_ms: Maximum acceptable latency in milliseconds.
        max_cost: Maximum acceptable cost (arbitrary units).
        metadata: Additional request metadata.
    """

    request_id: str
    model_id: str
    input_data_hash: str = ""
    priority: WorkloadPriority = WorkloadPriority.NORMAL
    latency_budget_ms: float = 100.0
    max_cost: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if self.latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be positive")
        if self.max_cost < 0:
            raise ValueError("max_cost must be non-negative")


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Result of an inference execution.

    Attributes:
        request_id: The original request ID.
        node_id: The node that executed the inference.
        model_id: The model used.
        latency_ms: Actual inference latency in milliseconds.
        output_hash: Hash of the output data.
        confidence: Confidence score of the result (0.0-1.0).
        cost: Cost of this inference.
        cached: Whether the result was served from cache.
        timestamp: When the inference completed.
    """

    request_id: str
    node_id: str
    model_id: str
    latency_ms: float
    output_hash: str = ""
    confidence: float = 1.0
    cost: float = 0.0
    cached: bool = False
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")

    @property
    def within_budget(self) -> bool:
        """Whether the latency was within budget."""
        # This is a placeholder; actual check requires the request's budget
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "request_id": self.request_id,
            "node_id": self.node_id,
            "model_id": self.model_id,
            "latency_ms": self.latency_ms,
            "output_hash": self.output_hash,
            "confidence": self.confidence,
            "cost": self.cost,
            "cached": self.cached,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True, slots=True)
class WorkloadPlacement:
    """A placement of a workload on an edge node.

    Attributes:
        placement_id: Unique identifier.
        workload_id: The workload being placed.
        node_id: The target edge node.
        model_id: The model to deploy.
        expected_latency_ms: Expected inference latency.
        expected_cost: Expected cost.
        placement_score: Quality score of this placement (0.0-1.0).
    """

    placement_id: str
    workload_id: str
    node_id: str
    model_id: str
    expected_latency_ms: float
    expected_cost: float
    placement_score: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.placement_score <= 1.0:
            raise ValueError(
                f"placement_score must be in [0, 1], got {self.placement_score}"
            )


# ---------------------------------------------------------------------------
# EdgeNode
# ---------------------------------------------------------------------------


class EdgeNode:
    """Represents an edge computing node with AI capabilities.

    An EdgeNode manages its local resources, deployed models, and
    inference execution. It tracks performance metrics and health.

    Attributes:
        node_id: Unique identifier.
        location: Geographic location (lat, lon).
        capabilities: Hardware capabilities.
        status: Current operational status.
    """

    def __init__(
        self,
        node_id: str,
        location: Tuple[float, float] = (0.0, 0.0),
        capabilities: Optional[ResourceProfile] = None,
        max_model_cache: int = 10,
    ) -> None:
        """Initialize an EdgeNode.

        Args:
            node_id: Unique identifier (must be non-empty).
            location: (latitude, longitude) tuple.
            capabilities: Hardware resource profile.
            max_model_cache: Maximum number of models to cache.

        Raises:
            ValueError: If node_id is empty or max_model_cache <= 0.
        """
        if not node_id:
            raise ValueError("node_id must be non-empty")
        if max_model_cache <= 0:
            raise ValueError("max_model_cache must be positive")

        self._node_id: str = node_id
        self._location: Tuple[float, float] = location
        self._capabilities: ResourceProfile = capabilities or ResourceProfile()
        self._status: NodeStatus = NodeStatus.ONLINE
        self._max_model_cache: int = max_model_cache
        self._deployed_models: Dict[str, AIModel] = {}
        self._model_access_times: Dict[str, float] = {}
        self._usage: ResourceUsage = ResourceUsage()
        self._inference_history: Deque[InferenceResult] = deque(maxlen=1000)
        self._total_inferences: int = 0
        self._total_latency_ms: float = 0.0
        self._created_at: float = time.time()
        self._metadata: Dict[str, Any] = {}

    @property
    def node_id(self) -> str:
        """Unique identifier."""
        return self._node_id

    @property
    def location(self) -> Tuple[float, float]:
        """Geographic location."""
        return self._location

    @property
    def capabilities(self) -> ResourceProfile:
        """Hardware capabilities."""
        return self._capabilities

    @property
    def status(self) -> NodeStatus:
        """Current operational status."""
        return self._status

    @status.setter
    def status(self, value: NodeStatus) -> None:
        if not isinstance(value, NodeStatus):
            raise TypeError(f"status must be NodeStatus, got {type(value).__name__}")
        self._status = value

    @property
    def usage(self) -> ResourceUsage:
        """Current resource usage."""
        return self._usage

    @usage.setter
    def usage(self, value: ResourceUsage) -> None:
        if not isinstance(value, ResourceUsage):
            raise TypeError(f"usage must be ResourceUsage, got {type(value).__name__}")
        self._usage = value

    @property
    def deployed_models(self) -> Dict[str, AIModel]:
        """Currently deployed models."""
        return dict(self._deployed_models)

    @property
    def total_inferences(self) -> int:
        """Total number of inferences executed."""
        return self._total_inferences

    @property
    def average_latency_ms(self) -> float:
        """Average inference latency in milliseconds."""
        if self._total_inferences == 0:
            return 0.0
        return self._total_latency_ms / self._total_inferences

    @property
    def metadata(self) -> Dict[str, Any]:
        """Mutable metadata dictionary."""
        return self._metadata

    @property
    def uptime_seconds(self) -> float:
        """Uptime in seconds."""
        return time.time() - self._created_at

    def can_deploy(self, model: AIModel) -> bool:
        """Check if a model can be deployed on this node.

        Args:
            model: The model to check.

        Returns:
            True if the model can be deployed.
        """
        if self._status != NodeStatus.ONLINE:
            return False
        if model.required_gpu and not self._capabilities.has_gpu:
            return False
        if model.required_compute_tflops > self._capabilities.compute_tflops:
            return False
        if model.required_memory_gb > self._capabilities.memory_gb:
            return False
        if model.size_mb > self._capabilities.storage_gb * 1024:
            return False
        return True

    def deploy_model(self, model: AIModel) -> bool:
        """Deploy a model on this node.

        Args:
            model: The model to deploy.

        Returns:
            True if deployment succeeded.

        Raises:
            DeploymentError: If the model cannot be deployed.
        """
        if not self.can_deploy(model):
            raise DeploymentError(
                f"Cannot deploy model '{model.model_id}' on node '{self._node_id}': "
                f"insufficient resources or incompatible capabilities"
            )

        # Evict LRU model if cache is full
        if len(self._deployed_models) >= self._max_model_cache:
            self._evict_lru_model()

        self._deployed_models[model.model_id] = model
        self._model_access_times[model.model_id] = time.time()
        logger.info(
            "Deployed model '%s' v%s on node '%s'",
            model.name,
            model.version,
            self._node_id,
        )
        return True

    def undeploy_model(self, model_id: str) -> Optional[AIModel]:
        """Remove a deployed model.

        Args:
            model_id: The model to remove.

        Returns:
            The removed model, or None if not found.
        """
        self._model_access_times.pop(model_id, None)
        return self._deployed_models.pop(model_id, None)

    def has_model(self, model_id: str) -> bool:
        """Check if a model is deployed."""
        return model_id in self._deployed_models

    async def run_inference(self, request: InferenceRequest) -> InferenceResult:
        """Execute an inference request.

        Args:
            request: The inference request.

        Returns:
            InferenceResult.

        Raises:
            InferenceError: If the model is not deployed or inference fails.
            LatencyBudgetExceededError: If latency exceeds the budget.
        """
        if request.model_id not in self._deployed_models:
            raise InferenceError(
                f"Model '{request.model_id}' not deployed on node '{self._node_id}'"
            )

        model = self._deployed_models[request.model_id]
        self._model_access_times[request.model_id] = time.time()

        # Simulate inference latency based on model and hardware
        base_latency = model.latency_target_ms
        # GPU acceleration reduces latency
        if self._capabilities.has_gpu and model.required_gpu:
            base_latency *= 0.3
        # Add some randomness
        latency = base_latency * random.uniform(0.8, 1.2)
        # Add queueing delay if busy
        if self._usage.active_inferences > 0:
            latency += self._usage.active_inferences * 2.0

        await asyncio.sleep(latency / 1000.0)  # Convert ms to seconds

        if latency > request.latency_budget_ms:
            raise LatencyBudgetExceededError(
                f"Inference latency {latency:.1f}ms exceeds budget "
                f"{request.latency_budget_ms:.1f}ms"
            )

        self._total_inferences += 1
        self._total_latency_ms += latency

        result = InferenceResult(
            request_id=request.request_id,
            node_id=self._node_id,
            model_id=request.model_id,
            latency_ms=latency,
            output_hash=hashlib.sha256(
                f"{request.request_id}:{time.time()}".encode()
            ).hexdigest()[:16],
            confidence=model.accuracy * random.uniform(0.95, 1.0),
            cost=self._compute_cost(model, latency),
        )
        self._inference_history.append(result)
        return result

    def get_inference_history(self, count: int = 100) -> List[InferenceResult]:
        """Get recent inference results.

        Args:
            count: Maximum number of results to return.

        Returns:
            List of InferenceResult objects, most recent first.
        """
        return list(self._inference_history)[-count:][::-1]

    def distance_to(self, location: Tuple[float, float]) -> float:
        """Compute Haversine distance to a location in kilometers.

        Args:
            location: (latitude, longitude) tuple.

        Returns:
            Distance in kilometers.
        """
        lat1, lon1 = self._location
        lat2, lon2 = location
        R = 6371.0  # Earth radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "node_id": self._node_id,
            "location": self._location,
            "capabilities": self._capabilities.to_dict(),
            "status": self._status.value,
            "usage": self._usage.to_dict(),
            "deployed_models": {
                mid: m.to_dict() for mid, m in self._deployed_models.items()
            },
            "total_inferences": self._total_inferences,
            "average_latency_ms": self.average_latency_ms,
            "uptime_seconds": self.uptime_seconds,
            "metadata": dict(self._metadata),
        }

    def _evict_lru_model(self) -> None:
        """Evict the least recently used model."""
        if not self._model_access_times:
            return
        lru_model = min(self._model_access_times, key=self._model_access_times.get)
        self.undeploy_model(lru_model)
        logger.debug("Evicted LRU model '%s' from node '%s'", lru_model, self._node_id)

    def _compute_cost(self, model: AIModel, latency_ms: float) -> float:
        """Compute the cost of an inference."""
        # Simple cost model: base cost + latency penalty
        base_cost = model.size_mb / 100.0
        latency_cost = latency_ms / 1000.0
        return base_cost + latency_cost

    def __repr__(self) -> str:
        return (
            f"EdgeNode(node_id={self._node_id!r}, status={self._status.value!r}, "
            f"models={len(self._deployed_models)})"
        )


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """Registry of AI models available for edge deployment.

    Supports registering, querying, and managing AI models with
    versioning and tagging.

    Attributes:
        models: Mapping of model_id to AIModel.
    """

    def __init__(self) -> None:
        """Initialize an empty model registry."""
        self._models: Dict[str, AIModel] = {}
        self._tags_index: Dict[str, Set[str]] = defaultdict(set)

    @property
    def models(self) -> Dict[str, AIModel]:
        """All registered models."""
        return dict(self._models)

    @property
    def model_count(self) -> int:
        """Number of registered models."""
        return len(self._models)

    def register(self, model: AIModel) -> None:
        """Register a model.

        Args:
            model: The model to register.

        Raises:
            ValueError: If a model with the same ID and version exists.
        """
        key = f"{model.model_id}:{model.version}"
        if key in self._models:
            raise ValueError(
                f"Model '{model.model_id}' v{model.version} already registered"
            )
        self._models[key] = model
        for tag in model.tags:
            self._tags_index[tag].add(key)

    def unregister(self, model_id: str, version: Optional[str] = None) -> bool:
        """Unregister a model.

        Args:
            model_id: The model ID.
            version: Specific version (None for all versions).

        Returns:
            True if any model was removed.
        """
        if version is not None:
            key = f"{model_id}:{version}"
            model = self._models.pop(key, None)
            if model:
                for tag in model.tags:
                    self._tags_index[tag].discard(key)
                return True
            return False

        # Remove all versions
        keys_to_remove = [
            k for k in self._models if k.startswith(f"{model_id}:")
        ]
        for key in keys_to_remove:
            model = self._models.pop(key)
            for tag in model.tags:
                self._tags_index[tag].discard(key)
        return len(keys_to_remove) > 0

    def get(self, model_id: str, version: Optional[str] = None) -> AIModel:
        """Get a model by ID and optional version.

        Args:
            model_id: The model ID.
            version: Specific version (None for latest).

        Returns:
            The AIModel.

        Raises:
            ModelNotFoundError: If the model doesn't exist.
        """
        if version is not None:
            key = f"{model_id}:{version}"
            if key not in self._models:
                raise ModelNotFoundError(
                    f"Model '{model_id}' v{version} not found"
                )
            return self._models[key]

        # Get latest version
        versions = [
            k for k in self._models if k.startswith(f"{model_id}:")
        ]
        if not versions:
            raise ModelNotFoundError(f"Model '{model_id}' not found")
        # Sort by version string (simple lexicographic)
        latest = sorted(versions)[-1]
        return self._models[latest]

    def find_by_tag(self, tag: str) -> List[AIModel]:
        """Find models by tag.

        Args:
            tag: The tag to search for.

        Returns:
            List of matching models.
        """
        keys = self._tags_index.get(tag, set())
        return [self._models[k] for k in keys if k in self._models]

    def find_by_capability(
        self,
        requires_gpu: bool = False,
        max_size_mb: Optional[float] = None,
    ) -> List[AIModel]:
        """Find models matching capability constraints.

        Args:
            requires_gpu: Whether the model requires GPU.
            max_size_mb: Maximum model size.

        Returns:
            List of matching models.
        """
        results = []
        for model in self._models.values():
            if requires_gpu and not model.required_gpu:
                continue
            if max_size_mb is not None and model.size_mb > max_size_mb:
                continue
            results.append(model)
        return results

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_count": self.model_count,
            "models": {k: m.to_dict() for k, m in self._models.items()},
        }

    def __repr__(self) -> str:
        return f"ModelRegistry(models={self.model_count})"


# ---------------------------------------------------------------------------
# WorkloadScheduler
# ---------------------------------------------------------------------------


class WorkloadScheduler:
    """Schedules AI workloads across edge nodes.

    Uses a pluggable scheduling strategy to assign workloads to
    edge nodes based on latency, cost, capability, and fairness.

    Attributes:
        strategy: The scheduling strategy.
        load_balancing_factor: How aggressively to balance load (0.0-1.0).
    """

    def __init__(
        self,
        strategy: SchedulingStrategy = SchedulingStrategy.LATENCY_OPTIMIZED,
        load_balancing_factor: float = 0.5,
    ) -> None:
        """Initialize the WorkloadScheduler.

        Args:
            strategy: Scheduling strategy.
            load_balancing_factor: Load balancing aggressiveness (0.0-1.0).

        Raises:
            ValueError: If load_balancing_factor is out of range.
        """
        if not 0.0 <= load_balancing_factor <= 1.0:
            raise ValueError("load_balancing_factor must be in [0, 1]")

        self._strategy: SchedulingStrategy = strategy
        self._load_balancing_factor: float = load_balancing_factor
        self._placement_history: List[WorkloadPlacement] = []
        self._node_load: Dict[str, int] = defaultdict(int)

    @property
    def strategy(self) -> SchedulingStrategy:
        """Current scheduling strategy."""
        return self._strategy

    @strategy.setter
    def strategy(self, value: SchedulingStrategy) -> None:
        self._strategy = value

    @property
    def placement_history(self) -> Tuple[WorkloadPlacement, ...]:
        """History of workload placements."""
        return tuple(self._placement_history)

    def schedule(
        self,
        request: InferenceRequest,
        nodes: List[EdgeNode],
        model_registry: ModelRegistry,
    ) -> Optional[WorkloadPlacement]:
        """Schedule a workload on the best available edge node.

        Args:
            request: The inference request.
            nodes: Available edge nodes.
            model_registry: Model registry for model lookup.

        Returns:
            WorkloadPlacement if a suitable node is found, None otherwise.

        Raises:
            SchedulingError: If no suitable node is found.
        """
        if not nodes:
            raise SchedulingError("No edge nodes available for scheduling")

        try:
            model = model_registry.get(request.model_id)
        except ModelNotFoundError:
            raise SchedulingError(
                f"Model '{request.model_id}' not found in registry"
            )

        # Filter nodes that can run this model
        candidates = [
            n for n in nodes
            if n.status == NodeStatus.ONLINE and n.can_deploy(model)
        ]

        if not candidates:
            raise SchedulingError(
                f"No edge node can run model '{request.model_id}'"
            )

        # Score each candidate
        scored: List[Tuple[float, EdgeNode]] = []
        for node in candidates:
            score = self._score_node(node, request, model)
            scored.append((score, node))

        # Select best node
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_node = scored[0]

        # Estimate latency and cost
        estimated_latency = self._estimate_latency(best_node, model, request)
        estimated_cost = self._estimate_cost(best_node, model)

        placement = WorkloadPlacement(
            placement_id=f"PLACE-{len(self._placement_history) + 1:06d}",
            workload_id=request.request_id,
            node_id=best_node.node_id,
            model_id=request.model_id,
            expected_latency_ms=estimated_latency,
            expected_cost=estimated_cost,
            placement_score=best_score,
        )
        self._placement_history.append(placement)
        self._node_load[best_node.node_id] += 1
        return placement

    def get_node_load(self, node_id: str) -> int:
        """Get the number of workloads placed on a node.

        Args:
            node_id: The node to query.

        Returns:
            Number of placements.
        """
        return self._node_load.get(node_id, 0)

    def get_load_distribution(self) -> Dict[str, int]:
        """Get the distribution of workloads across nodes.

        Returns:
            Mapping of node_id to placement count.
        """
        return dict(self._node_load)

    def clear_history(self) -> None:
        """Clear placement history."""
        self._placement_history.clear()
        self._node_load.clear()

    def _score_node(
        self, node: EdgeNode, request: InferenceRequest, model: AIModel
    ) -> float:
        """Score a node for a given request and model.

        Returns:
            Score in [0.0, 1.0], higher is better.
        """
        scores: List[float] = []

        # Latency score
        estimated_latency = self._estimate_latency(node, model, request)
        latency_score = max(0.0, 1.0 - estimated_latency / request.latency_budget_ms)
        scores.append(latency_score)

        # Resource availability score
        resource_score = 1.0 - max(
            node.usage.cpu_percent / 100.0,
            node.usage.gpu_util_percent / 100.0,
        )
        scores.append(resource_score)

        # Load balancing score (prefer less loaded nodes)
        node_load = self._node_load.get(node.node_id, 0)
        max_load = max(self._node_load.values(), default=0)
        if max_load > 0:
            load_score = 1.0 - (node_load / max_load)
        else:
            load_score = 1.0
        scores.append(load_score * self._load_balancing_factor)

        # Capability match score
        if model.required_gpu and node.capabilities.has_gpu:
            scores.append(1.0)
        elif not model.required_gpu:
            scores.append(0.8)
        else:
            scores.append(0.0)

        # Strategy-specific weighting
        if self._strategy == SchedulingStrategy.LATENCY_OPTIMIZED:
            weights = [0.5, 0.2, 0.1, 0.2]
        elif self._strategy == SchedulingStrategy.THROUGHPUT_OPTIMIZED:
            weights = [0.2, 0.3, 0.3, 0.2]
        elif self._strategy == SchedulingStrategy.ENERGY_EFFICIENT:
            weights = [0.2, 0.4, 0.2, 0.2]
        elif self._strategy == SchedulingStrategy.CAPABILITY_MATCHED:
            weights = [0.2, 0.2, 0.1, 0.5]
        else:
            weights = [0.25, 0.25, 0.25, 0.25]

        # Ensure weights match scores length
        weights = weights[: len(scores)]
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        weighted = sum(s * w for s, w in zip(scores, weights)) / total_weight
        return max(0.0, min(1.0, weighted))

    @staticmethod
    def _estimate_latency(
        node: EdgeNode, model: AIModel, request: InferenceRequest
    ) -> float:
        """Estimate inference latency on a node."""
        base = model.latency_target_ms
        if node.capabilities.has_gpu and model.required_gpu:
            base *= 0.3
        # Add queueing delay
        base += node.usage.active_inferences * 2.0
        return base

    @staticmethod
    def _estimate_cost(node: EdgeNode, model: AIModel) -> float:
        """Estimate inference cost on a node."""
        return model.size_mb / 100.0 + model.latency_target_ms / 1000.0

    def __repr__(self) -> str:
        return (
            f"WorkloadScheduler(strategy={self._strategy.value!r}, "
            f"placements={len(self._placement_history)})"
        )


# ---------------------------------------------------------------------------
# MECOrchestrator
# ---------------------------------------------------------------------------


class MECOrchestrator:
    """Orchestrates workloads across MEC edge nodes.

    Manages the lifecycle of AI workloads on edge infrastructure,
    including deployment, scheduling, execution, and monitoring.

    Attributes:
        nodes: Mapping of node_id to EdgeNode.
        model_registry: The model registry.
        scheduler: The workload scheduler.
    """

    def __init__(
        self,
        model_registry: Optional[ModelRegistry] = None,
        scheduler: Optional[WorkloadScheduler] = None,
    ) -> None:
        """Initialize the MECOrchestrator.

        Args:
            model_registry: Model registry.
            scheduler: Workload scheduler.
        """
        self._nodes: Dict[str, EdgeNode] = {}
        self._model_registry = model_registry or ModelRegistry()
        self._scheduler = schedulerheduler or WorkloadScheduler()
        self._active_placements: Dict[str, WorkloadPlacement] = {}
        self._placement_counter: int = 0

    @property
    def nodes(self) -> Dict[str, EdgeNode]:
        """Managed edge nodes."""
        return dict(self._nodes)

    @property
    def model_registry(self) -> ModelRegistry:
        """The model registry."""
        return self._model_registry

    @property
    def scheduler(self) -> WorkloadScheduler:
        """The workload scheduler."""
        return self._scheduler

    @property
    def active_placements(self) -> Dict[str, WorkloadPlacement]:
        """Currently active workload placements."""
        return dict(self._active_placements)

    def add_node(self, node: EdgeNode) -> None:
        """Add an edge node to the orchestrator.

        Args:
            node: The edge node to add.

        Raises:
            ValueError: If a node with the same ID already exists.
        """
        if node.node_id in self._nodes:
            raise ValueError(f"Node '{node.node_id}' already exists")
        self._nodes[node.node_id] = node

    def remove_node(self, node_id: str) -> Optional[EdgeNode]:
        """Remove an edge node.

        Args:
            node_id: The node to remove.

        Returns:
            The removed node, or None if not found.
        """
        return self._nodes.pop(node_id, None)

    def get_node(self, node_id: str) -> EdgeNode:
        """Get an edge node by ID.

        Args:
            node_id: The node identifier.

        Returns:
            The EdgeNode.

        Raises:
            NodeNotFoundError: If the node doesn't exist.
        """
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Node '{node_id}' not found")
        return self._nodes[node_id]

    def deploy_model_to_node(self, model_id: str, node_id: str) -> bool:
        """Deploy a model to a specific edge node.

        Args:
            model_id: The model to deploy.
            node_id: The target node.

        Returns:
            True if deployment succeeded.

        Raises:
            NodeNotFoundError: If the node doesn't exist.
            ModelNotFoundError: If the model doesn't exist.
            DeploymentError: If deployment fails.
        """
        node = self.get_node(node_id)
        model = self._model_registry.get(model_id)
        return node.deploy_model(model)

    def deploy_model_everywhere(self, model_id: str) -> Dict[str, bool]:
        """Deploy a model to all compatible nodes.

        Args:
            model_id: The model to deploy.

        Returns:
            Mapping of node_id to deployment success.
        """
        results: Dict[str, bool] = {}
        for node in self._nodes.values():
            try:
                results[node.node_id] = self.deploy_model_to_node(
                    model_id, node.node_id
                )
            except (DeploymentError, NodeNotFoundError, ModelNotFoundError) as exc:
                logger.warning(
                    "Failed to deploy model '%s' to node '%s': %s",
                    model_id,
                    node.node_id,
                    exc,
                )
                results[node.node_id] = False
        return results

    async def execute_inference(
        self, request: InferenceRequest
    ) -> InferenceResult:
        """Execute an inference request on the best available node.

        Args:
            request: The inference request.

        Returns:
            InferenceResult.

        Raises:
            SchedulingError: If no suitable node is found.
            InferenceError: If inference execution fails.
        """
        # Schedule the workload
        placement = self._scheduler.schedule(
            request, list(self._nodes.values()), self._model_registry
        )
        if placement is None:
            raise SchedulingError(
                f"Could not schedule workload '{request.request_id}'"
            )

        node = self._nodes[placement.node_id]

        # Deploy model if not already deployed
        if not node.has_model(request.model_id):
            self.deploy_model_to_node(request.model_id, node.node_id)

        # Execute inference
        result = await node.run_inference(request)
        self._active_placements[request.request_id] = placement
        return result

    async def execute_inference_batch(
        self, requests: List[InferenceRequest]
    ) -> List[InferenceResult]:
        """Execute a batch of inference requests concurrently.

        Args:
            requests: List of inference requests.

        Returns:
            List of InferenceResult objects.
        """
        tasks = [self.execute_inference(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [
            r for r in results
            if isinstance(r, InferenceResult)
        ]

    def get_cluster_metrics(self) -> Dict[str, Any]:
        """Get aggregate metrics for the edge cluster.

        Returns:
            Dictionary with cluster-wide metrics.
        """
        total_inferences = sum(n.total_inferences for n in self._nodes.values())
        latencies = [
            r.latency_ms
            for n in self._nodes.values()
            for r in n.get_inference_history(count=1000)
        ]
        avg_latency = statistics.mean(latencies) if latencies else 0.0
        p99_latency = (
            sorted(latencies)[int(len(latencies) * 0.99)]
            if len(latencies) >= 100
            else max(latencies, default=0.0)
        )

        return {
            "node_count": len(self._nodes),
            "online_nodes": sum(
                1 for n in self._nodes.values()
                if n.status == NodeStatus.ONLINE
            ),
            "total_inferences": total_inferences,
            "average_latency_ms": avg_latency,
            "p99_latency_ms": p99_latency,
            "active_placements": len(self._active_placements),
            "load_distribution": self._scheduler.get_load_distribution(),
        }

    def __repr__(self) -> str:
        return (
            f"MECOrchestrator(nodes={len(self._nodes)}, "
            f"models={self._model_registry.model_count}, "
            f"placements={len(self._active_placements)})"
        )


# ---------------------------------------------------------------------------
# EdgeAIManager (High-level orchestrator)
# ---------------------------------------------------------------------------


class EdgeAIManager:
    """High-level orchestrator for edge AI operations.

    Provides a unified interface for managing edge nodes, models,
    inference, and MEC orchestration.

    Attributes:
        orchestrator: The MEC orchestrator.
        model_registry: The model registry.
    """

    def __init__(
        self,
        orchestrator: Optional[MECOrchestrator] = None,
        model_registry: Optional[ModelRegistry] = None,
    ) -> None:
        """Initialize EdgeAIManager.

        Args:
            orchestrator: MEC orchestrator.
            model_registry: Model registry.
        """
        self._orchestrator = orchestrator or MECOrchestrator(
            model_registry=model_registry
        )
        self._model_registry = model_registry or self._orchestrator.model_registry

    @property
    def orchestrator(self) -> MECOrchestrator:
        """The MEC orchestrator."""
        return self._orchestrator

    @property
    def model_registry(self) -> ModelRegistry:
        """The model registry."""
        return self._model_registry

    def add_edge_node(self, node: EdgeNode) -> None:
        """Add an edge node.

        Args:
            node: The edge node to add.
        """
        self._orchestrator.add_node(node)

    def remove_edge_node(self, node_id: str) -> Optional[EdgeNode]:
        """Remove an edge node.

        Args:
            node_id: The node to remove.

        Returns:
            The removed node, or None.
        """
        return self._orchestrator.remove_node(node_id)

    def register_model(self, model: AIModel) -> None:
        """Register an AI model.

        Args:
            model: The model to register.
        """
        self._model_registry.register(model)

    def deploy_model(
        self, model_id: str, node_id: Optional[str] = None
    ) -> Any:
        """Deploy a model.

        Args:
            model_id: The model to deploy.
            node_id: Specific node (None for all compatible nodes).

        Returns:
            Deployment result.
        """
        if node_id is not None:
            return self._orchestrator.deploy_model_to_node(model_id, node_id)
        return self._orchestrator.deploy_model_everywhere(model_id)

    async def infer(self, request: InferenceRequest) -> InferenceResult:
        """Execute an inference request.

        Args:
            request: The inference request.

        Returns:
            InferenceResult.
        """
        return await self._orchestrator.execute_inference(request)

    async def infer_batch(
        self, requests: List[InferenceRequest]
    ) -> List[InferenceResult]:
        """Execute a batch of inference requests.

        Args:
            requests: List of inference requests.

        Returns:
            List of InferenceResult objects.
        """
        return await self._orchestrator.execute_inference_batch(requests)

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics.

        Returns:
            Dictionary with all metrics.
        """
        return {
            "cluster": self._orchestrator.get_cluster_metrics(),
            "models": self._model_registry.to_dict(),
            "nodes": {
                nid: node.to_dict()
                for nid, node in self._orchestrator.nodes.items()
            },
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the edge AI system.

        Returns:
            Dictionary with summary statistics.
        """
        metrics = self.get_metrics()
        cluster = metrics["cluster"]
        return {
            "node_count": cluster["node_count"],
            "online_nodes": cluster["online_nodes"],
            "total_models": self._model_registry.model_count,
            "total_inferences": cluster["total_inferences"],
            "average_latency_ms": cluster["average_latency_ms"],
            "p99_latency_ms": cluster["p99_latency_ms"],
            "active_placements": cluster["active_placements"],
        }

    def __repr__(self) -> str:
        return (
            f"EdgeAIManager(nodes={len(self._orchestrator.nodes)}, "
            f"models={self._model_registry.model_count})"
        )
