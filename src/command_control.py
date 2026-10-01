"""
Apex Critical Infrastructure — Command & Control Module.

Decision support, situational awareness, and resource allocation
for defense and security operations.

Classes:
    DecisionSupportEngine: Multi-criteria decision analysis for C2.
    SituationalAwareness: Real-time operational picture aggregation.
    ResourceAllocator: Dynamic resource allocation and optimization.
    DecisionOption: Evaluated decision alternative.
    Resource: Allocatable operational resource.
    TacticalPicture: Aggregated situational awareness snapshot.
"""

from __future__ import annotations

import heapq
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class DecisionPriority(Enum):
    """Priority levels for command decisions."""

    CRITICAL = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    ROUTINE = auto()


class ResourceType(Enum):
    """Types of allocatable resources."""

    COMPUTE = auto()
    NETWORK = auto()
    PERSONNEL = auto()
    SENSOR = auto()
    STORAGE = auto()
    BANDWIDTH = auto()


class ResourceStatus(Enum):
    """Operational status of a resource."""

    AVAILABLE = auto()
    ALLOCATED = auto()
    DEGRADED = auto()
    OFFLINE = auto()
    RESERVED = auto()


class AwarenessLevel(Enum):
    """Situational awareness completeness levels."""

    FULL = auto()
    PARTIAL = auto()
    LIMITED = auto()
    BLIND = auto()


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DecisionOption:
    """Evaluated decision alternative.

    Attributes:
        option_id: Unique option identifier.
        name: Human-readable option name.
        description: Detailed description.
        criteria_scores: Mapping of criterion names to scores (0-1).
        weights: Mapping of criterion names to weights.
        risk_score: Overall risk assessment (0-1, lower is better).
        estimated_impact: Estimated operational impact.
        resource_requirements: Required resources by type.
    """

    option_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    criteria_scores: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    risk_score: float = 0.5
    estimated_impact: float = 0.5
    resource_requirements: dict[ResourceType, float] = field(default_factory=dict)

    def weighted_score(self) -> float:
        """Calculate the weighted multi-criteria score.

        Returns:
            Weighted score between 0 and 1.
        """
        if not self.criteria_scores:
            return 0.0
        total_weight = sum(self.weights.values())
        if total_weight == 0:
            return 0.0
        score = sum(
            self.criteria_scores.get(c, 0.0) * self.weights.get(c, 0.0)
            for c in self.criteria_scores
        )
        return score / total_weight


@dataclass(slots=True)
class Resource:
    """Allocatable operational resource.

    Attributes:
        resource_id: Unique resource identifier.
        name: Human-readable resource name.
        resource_type: Category of resource.
        capacity: Total capacity units.
        allocated: Currently allocated units.
        status: Operational status.
        metadata: Additional resource attributes.
    """

    resource_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    resource_type: ResourceType = ResourceType.COMPUTE
    capacity: float = 1.0
    allocated: float = 0.0
    status: ResourceStatus = ResourceStatus.AVAILABLE
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> float:
        """Return remaining available capacity."""
        return max(0.0, self.capacity - self.allocated)

    @property
    def utilization(self) -> float:
        """Return utilization ratio (0-1)."""
        if self.capacity <= 0:
            return 0.0
        return self.allocated / self.capacity

    def allocate(self, amount: float) -> bool:
        """Attempt to allocate capacity.

        Args:
            amount: Units to allocate.

        Returns:
            True if allocation succeeded.
        """
        if amount < 0 or amount > self.available:
            return False
        self.allocated += amount
        if self.allocated >= self.capacity:
            self.status = ResourceStatus.ALLOCATED
        return True

    def release(self, amount: float) -> bool:
        """Release allocated capacity.

        Args:
            amount: Units to release.

        Returns:
            True if release succeeded.
        """
        if amount < 0 or amount > self.allocated:
            return False
        self.allocated -= amount
        if self.status == ResourceStatus.ALLOCATED and self.allocated < self.capacity:
            self.status = ResourceStatus.AVAILABLE
        return True


@dataclass(frozen=True, slots=True)
class TacticalPicture:
    """Aggregated situational awareness snapshot.

    Attributes:
        timestamp: Snapshot generation timestamp.
        awareness_level: Completeness of the picture.
        threat_indicators: Active threat indicators.
        resource_status: Current resource availability summary.
        operational_units: Status of operational units.
        intelligence_summary: Latest intelligence digest.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    awareness_level: AwarenessLevel = AwarenessLevel.PARTIAL
    threat_indicators: list[dict[str, Any]] = field(default_factory=list)
    resource_status: dict[str, Any] = field(default_factory=dict)
    operational_units: dict[str, Any] = field(default_factory=dict)
    intelligence_summary: str = ""


# ---------------------------------------------------------------------------
# Decision Support Engine
# ---------------------------------------------------------------------------


class DecisionSupportEngine:
    """Multi-criteria decision analysis engine for C2.

    Evaluates decision options against weighted criteria and
    produces ranked recommendations.

    Attributes:
        criteria_weights: Default criteria weights.
        _decision_history: Log of past decisions.
    """

    def __init__(self) -> None:
        """Initialize the decision support engine."""
        self.criteria_weights: dict[str, float] = {
            "effectiveness": 0.30,
            "speed": 0.25,
            "risk": 0.20,
            "cost": 0.15,
            "reversibility": 0.10,
        }
        self._decision_history: list[dict[str, Any]] = []

    def set_criterion_weight(self, criterion: str, weight: float) -> None:
        """Set the weight for a decision criterion.

        Args:
            criterion: Criterion name.
            weight: Weight value (0-1).

        Raises:
            ValueError: If weight is negative.
        """
        if weight < 0:
            raise ValueError("Weight must be non-negative")
        self.criteria_weights[criterion] = weight

    def evaluate(
        self, options: list[DecisionOption], context: dict[str, Any] | None = None
    ) -> list[DecisionOption]:
        """Evaluate and rank decision options.

        Args:
            options: Decision alternatives to evaluate.
            context: Optional operational context for adjustment.

        Returns:
            Options sorted by descending weighted score.
        """
        context = context or {}
        scored: list[tuple[float, DecisionOption]] = []

        for option in options:
            # Apply context adjustments
            adjusted_risk = option.risk_score
            if "risk_tolerance" in context:
                tolerance = context["risk_tolerance"]
                adjusted_risk = option.risk_score * (1.0 - tolerance * 0.5)

            # Composite score: weighted criteria minus risk penalty
            composite = option.weighted_score() * (1.0 - adjusted_risk * 0.3)
            scored.append((composite, option))

        scored.sort(key=lambda x: x[0], reverse=True)

        self._decision_history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "options_evaluated": len(options),
                "context": context,
                "top_choice": scored[0][1].name if scored else None,
            }
        )

        return [opt for _, opt in scored]

    def recommend(
        self, options: list[DecisionOption], context: dict[str, Any] | None = None
    ) -> DecisionOption | None:
        """Return the top-ranked decision option.

        Args:
            options: Decision alternatives.
            context: Optional operational context.

        Returns:
            Highest-scoring option, or None if no options provided.
        """
        ranked = self.evaluate(options, context)
        return ranked[0] if ranked else None


# ---------------------------------------------------------------------------
# Situational Awareness
# ---------------------------------------------------------------------------


class SituationalAwareness:
    """Real-time operational picture aggregation.

    Fuses data from multiple sources to maintain a comprehensive
    tactical picture with awareness level assessment.

    Attributes:
        sources: Registered data source callables.
        _picture_cache: Latest tactical picture.
        _source_health: Health status of each source.
    """

    def __init__(self) -> None:
        """Initialize situational awareness."""
        self.sources: dict[str, Callable[[], dict[str, Any]]] = {}
        self._picture_cache: TacticalPicture | None = None
        self._source_health: dict[str, bool] = {}

    def register_source(
        self, name: str, source: Callable[[], dict[str, Any]]
    ) -> None:
        """Register a data source.

        Args:
            name: Source identifier.
            source: Callable returning source data.
        """
        self.sources[name] = source
        self._source_health[name] = True
        logger.debug("Registered awareness source: %s", name)

    def update_picture(self) -> TacticalPicture:
        """Refresh the tactical picture from all sources.

        Returns:
            Updated TacticalPicture snapshot.
        """
        threat_indicators: list[dict[str, Any]] = []
        resource_status: dict[str, Any] = {}
        operational_units: dict[str, Any] = {}
        intelligence_parts: list[str] = []
        healthy_sources = 0

        for name, source in self.sources.items():
            try:
                data = source()
                self._source_health[name] = True
                healthy_sources += 1

                if "threats" in data:
                    threat_indicators.extend(data["threats"])
                if "resources" in data:
                    resource_status.update(data["resources"])
                if "units" in data:
                    operational_units.update(data["units"])
                if "intelligence" in data:
                    intelligence_parts.append(str(data["intelligence"]))
            except Exception as exc:
                logger.warning("Awareness source %s failed: %s", name, exc)
                self._source_health[name] = False

        # Determine awareness level based on source health
        total = len(self.sources)
        if total == 0:
            level = AwarenessLevel.BLIND
        elif healthy_sources == total:
            level = AwarenessLevel.FULL
        elif healthy_sources >= total * 0.6:
            level = AwarenessLevel.PARTIAL
        else:
            level = AwarenessLevel.LIMITED

        picture = TacticalPicture(
            awareness_level=level,
            threat_indicators=threat_indicators,
            resource_status=resource_status,
            operational_units=operational_units,
            intelligence_summary=" | ".join(intelligence_parts),
        )
        self._picture_cache = picture
        return picture

    def get_cached_picture(self) -> TacticalPicture | None:
        """Return the last cached tactical picture.

        Returns:
            Cached picture or None if never updated.
        """
        return self._picture_cache

    def source_health_summary(self) -> dict[str, bool]:
        """Return health status of all registered sources.

        Returns:
            Mapping of source names to health booleans.
        """
        return dict(self._source_health)


# ---------------------------------------------------------------------------
# Resource Allocator
# ---------------------------------------------------------------------------


class ResourceAllocator:
    """Dynamic resource allocation and optimization.

    Manages a pool of resources with priority-based allocation,
    preemption, and utilization tracking.

    Attributes:
        resources: All managed resources by ID.
        _allocation_log: Chronological allocation history.
    """

    def __init__(self) -> None:
        """Initialize the resource allocator."""
        self.resources: dict[str, Resource] = {}
        self._allocation_log: list[dict[str, Any]] = []

    def add_resource(self, resource: Resource) -> None:
        """Add a resource to the managed pool.

        Args:
            resource: The resource to add.
        """
        self.resources[resource.resource_id] = resource
        logger.debug(
            "Added resource %s (%s)", resource.name, resource.resource_type.name
        )

    def remove_resource(self, resource_id: str) -> Resource:
        """Remove a resource from the pool.

        Args:
            resource_id: The resource identifier.

        Returns:
            The removed Resource.

        Raises:
            KeyError: If the resource is not found.
        """
        try:
            return self.resources.pop(resource_id)
        except KeyError as exc:
            raise KeyError(f"Resource not found: {resource_id}") from exc

    def allocate(
        self,
        resource_type: ResourceType,
        amount: float,
        priority: DecisionPriority = DecisionPriority.MEDIUM,
    ) -> list[str]:
        """Allocate resources of a given type.

        Selects the best-fit available resources and allocates
        from them in priority order.

        Args:
            resource_type: Type of resource needed.
            amount: Total units required.
            priority: Allocation priority.

        Returns:
            List of allocated resource IDs.
        """
        candidates = [
            r
            for r in self.resources.values()
            if r.resource_type == resource_type
            and r.status in (ResourceStatus.AVAILABLE, ResourceStatus.RESERVED)
            and r.available > 0
        ]
        # Best-fit: sort by available capacity ascending
        candidates.sort(key=lambda r: r.available)

        allocated_ids: list[str] = []
        remaining = amount

        for resource in candidates:
            if remaining <= 0:
                break
            take = min(remaining, resource.available)
            if resource.allocate(take):
                allocated_ids.append(resource.resource_id)
                remaining -= take
                self._allocation_log.append(
                    {
                        "resource_id": resource.resource_id,
                        "amount": take,
                        "priority": priority.name,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

        if remaining > 0:
            logger.warning(
                "Partial allocation: %.2f of %.2f %s allocated",
                amount - remaining,
                amount,
                resource_type.name,
            )

        return allocated_ids

    def release(self, resource_id: str, amount: float | None = None) -> bool:
        """Release allocated capacity from a resource.

        Args:
            resource_id: The resource identifier.
            amount: Units to release (None for all).

        Returns:
            True if release succeeded.

        Raises:
            KeyError: If the resource is not found.
        """
        resource = self.resources.get(resource_id)
        if resource is None:
            raise KeyError(f"Resource not found: {resource_id}")
        release_amount = amount if amount is not None else resource.allocated
        return resource.release(release_amount)

    def utilization_by_type(self) -> dict[ResourceType, float]:
        """Calculate utilization ratio per resource type.

        Returns:
            Mapping of resource types to utilization ratios.
        """
        by_type: dict[ResourceType, list[Resource]] = {}
        for r in self.resources.values():
            by_type.setdefault(r.resource_type, []).append(r)

        result: dict[ResourceType, float] = {}
        for rtype, resources in by_type.items():
            total_capacity = sum(r.capacity for r in resources)
            total_allocated = sum(r.allocated for r in resources)
            result[rtype] = total_allocated / total_capacity if total_capacity > 0 else 0.0
        return result

    def find_underutilized(self, threshold: float = 0.2) -> list[Resource]:
        """Find resources below a utilization threshold.

        Args:
            threshold: Utilization ratio threshold.

        Returns:
            List of underutilized resources.
        """
        return [r for r in self.resources.values() if r.utilization < threshold]

    def find_overloaded(self, threshold: float = 0.9) -> list[Resource]:
        """Find resources above a utilization threshold.

        Args:
            threshold: Utilization ratio threshold.

        Returns:
            List of overloaded resources.
        """
        return [r for r in self.resources.values() if r.utilization > threshold]
