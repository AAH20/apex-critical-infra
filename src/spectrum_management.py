"""Apex Critical Infrastructure — Spectrum Management.

Dynamic spectrum allocation, interference detection, and spectrum
optimization for telecom and wireless network infrastructure.

This module provides:
- SpectrumBand: Represents a frequency band with configurable properties.
- SpectrumAllocation: Tracks allocation of spectrum resources to users/cells.
- InterferenceDetector: Detects and classifies interference patterns.
- SpectrumOptimizer: Optimizes spectrum allocation using constraint solving.
- DynamicSpectrumAllocator: Real-time dynamic spectrum allocation engine.
- SpectrumManager: High-level orchestrator for spectrum operations.

All classes are fully typed, documented, and raise explicit exceptions
on invalid input or unrecoverable states.
"""

from __future__ import annotations

import enum
import logging
import math
import random
import statistics
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SpectrumManagementError(Exception):
    """Base exception for all spectrum management errors."""


class InvalidFrequencyError(SpectrumManagementError):
    """Raised when a frequency value is out of valid range."""


class AllocationConflictError(SpectrumManagementError):
    """Raised when a spectrum allocation conflicts with existing allocations."""


class BandNotFoundError(SpectrumManagementError):
    """Raised when a referenced band does not exist."""


class InterferenceDetectionError(SpectrumManagementError):
    """Raised when interference detection encounters an error."""


class OptimizationError(SpectrumManagementError):
    """Raised when spectrum optimization fails to converge."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BandType(enum.Enum):
    """Type of spectrum band."""

    LICENSED = "licensed"
    UNLICENSED = "unlicensed"
    SHARED = "shared"
    DYNAMIC = "dynamic"


class AllocationStrategy(enum.Enum):
    """Strategy for dynamic spectrum allocation."""

    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    WORST_FIT = "worst_fit"
    RANDOM_FIT = "random_fit"
    FAIR_SHARING = "fair_sharing"
    PRIORITY_BASED = "priority_based"


class InterferenceType(enum.Enum):
    """Classification of interference types."""

    CO_CHANNEL = "co_channel"
    ADJACENT_CHANNEL = "adjacent_channel"
    INTERMODULATION = "intermodulation"
    ADJACENT_CHANNEL_LEAKAGE = "adjacent_channel_leakage"
    BLOCKING = "blocking"
    SPURIOUS = "spurious"


class InterferenceSeverity(enum.Enum):
    """Severity of detected interference."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    SEVERE = "severe"


class UserPriority(enum.Enum):
    """Priority level for spectrum allocation."""

    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BEST_EFFORT = 5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrequencyRange:
    """Represents a range of frequencies.

    Attributes:
        start_mhz: Start frequency in MHz (inclusive).
        end_mhz: End frequency in MHz (inclusive).
    """

    start_mhz: float
    end_mhz: float

    def __post_init__(self) -> None:
        if self.start_mhz < 0 or self.end_mhz < 0:
            raise InvalidFrequencyError("Frequencies must be non-negative")
        if self.start_mhz >= self.end_mhz:
            raise InvalidFrequencyError(
                f"start_mhz ({self.start_mhz}) must be less than "
                f"end_mhz ({self.end_mhz})"
            )

    @property
    def bandwidth_mhz(self) -> float:
        """Total bandwidth in MHz."""
        return self.end_mhz - self.start_mhz

    @property
    def center_mhz(self) -> float:
        """Center frequency in MHz."""
        return (self.start_mhz + self.end_mhz) / 2.0

    def contains(self, frequency_mhz: float) -> bool:
        """Check if a frequency is within this range."""
        return self.start_mhz <= frequency_mhz <= self.end_mhz

    def overlaps(self, other: FrequencyRange) -> bool:
        """Check if this range overlaps with another."""
        return not (self.end_mhz < other.start_mhz or self.start_mhz > other.end_mhz)

    def intersection(self, other: FrequencyRange) -> Optional[FrequencyRange]:
        """Compute the intersection of two ranges, or None if disjoint."""
        if not self.overlaps(other):
            return None
        return FrequencyRange(
            start_mhz=max(self.start_mhz, other.start_mhz),
            end_mhz=min(self.end_mhz, other.end_mhz),
        )

    def __repr__(self) -> str:
        return f"FrequencyRange({self.start_mhz:.1f}-{self.end_mhz:.1f} MHz)"


@dataclass(frozen=True, slots=True)
class SpectrumBand:
    """A configurable spectrum band.

    Attributes:
        band_id: Unique identifier.
        frequency_range: The frequency range of this band.
        band_type: Type of spectrum band.
        max_power_dbm: Maximum allowed transmit power in dBm.
        channel_width_mhz: Width of each channel in MHz.
        guard_band_mhz: Guard band between channels in MHz.
    """

    band_id: str
    frequency_range: FrequencyRange
    band_type: BandType = BandType.LICENSED
    max_power_dbm: float = 30.0
    channel_width_mhz: float = 5.0
    guard_band_mhz: float = 0.5

    def __post_init__(self) -> None:
        if not self.band_id:
            raise ValueError("band_id must be non-empty")
        if self.max_power_dbm < -100 or self.max_power_dbm > 100:
            raise ValueError("max_power_dbm must be in [-100, 100]")
        if self.channel_width_mhz <= 0:
            raise ValueError("channel_width_mhz must be positive")
        if self.guard_band_mhz < 0:
            raise ValueError("guard_band_mhz must be non-negative")

    @property
    def num_channels(self) -> int:
        """Number of channels that fit in this band."""
        effective_width = self.channel_width_mhz + self.guard_band_mhz
        return max(1, int(self.frequency_range.bandwidth_mhz / effective_width))

    def get_channel_frequency(self, channel_index: int) -> float:
        """Get the center frequency of a specific channel.

        Args:
            channel_index: Zero-based channel index.

        Returns:
            Center frequency in MHz.

        Raises:
            ValueError: If channel_index is out of range.
        """
        if channel_index < 0 or channel_index >= self.num_channels:
            raise ValueError(
                f"Channel index {channel_index} out of range [0, {self.num_channels})"
            )
        effective_width = self.channel_width_mhz + self.guard_band_mhz
        return self.frequency_range.start_mhz + effective_width * channel_index + self.channel_width_mhz / 2.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "band_id": self.band_id,
            "frequency_range": {
                "start_mhz": self.frequency_range.start_mhz,
                "end_mhz": self.frequency_range.end_mhz,
            },
            "band_type": self.band_type.value,
            "max_power_dbm": self.max_power_dbm,
            "channel_width_mhz": self.channel_width_mhz,
            "guard_band_mhz": self.guard_band_mhz,
            "num_channels": self.num_channels,
        }


@dataclass(frozen=True, slots=True)
class SpectrumAllocation:
    """An allocation of spectrum to a user or cell.

    Attributes:
        allocation_id: Unique identifier.
        band_id: The band being allocated from.
        channel_index: The channel index within the band.
        user_id: The user/cell this allocation is for.
        priority: Priority level of this allocation.
        power_dbm: Allocated transmit power in dBm.
        start_time: When the allocation starts (Unix timestamp).
        duration_seconds: How long the allocation lasts.
        metadata: Additional allocation metadata.
    """

    allocation_id: str
    band_id: str
    channel_index: int
    user_id: str
    priority: UserPriority = UserPriority.MEDIUM
    power_dbm: float = 20.0
    start_time: float = field(default_factory=time.time)
    duration_seconds: float = 3600.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.allocation_id:
            raise ValueError("allocation_id must be non-empty")
        if not self.band_id:
            raise ValueError("band_id must be non-empty")
        if not self.user_id:
            raise ValueError("user_id must be non-empty")
        if self.channel_index < 0:
            raise ValueError("channel_index must be non-negative")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")

    @property
    def end_time(self) -> float:
        """When this allocation ends."""
        return self.start_time + self.duration_seconds

    def is_active(self, current_time: Optional[float] = None) -> bool:
        """Check if this allocation is currently active.

        Args:
            current_time: Time to check at (defaults to now).

        Returns:
            True if the allocation is active.
        """
        if current_time is None:
            current_time = time.time()
        return self.start_time <= current_time < self.end_time

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "allocation_id": self.allocation_id,
            "band_id": self.band_id,
            "channel_index": self.channel_index,
            "user_id": self.user_id,
            "priority": self.priority.name,
            "power_dbm": self.power_dbm,
            "start_time": self.start_time,
            "duration_seconds": self.duration_seconds,
            "end_time": self.end_time,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class InterferenceEvent:
    """A detected interference event.

    Attributes:
        event_id: Unique identifier.
        source_band_id: Band where interference was detected.
        affected_band_id: Band affected by the interference.
        interference_type: Type of interference.
        severity: Severity level.
        frequency_mhz: Frequency where interference was observed.
        power_dbm: Measured interference power in dBm.
        timestamp: When detected.
        description: Human-readable description.
    """

    event_id: str
    source_band_id: str
    affected_band_id: str
    interference_type: InterferenceType
    severity: InterferenceSeverity
    frequency_mhz: float
    power_dbm: float
    timestamp: float = field(default_factory=time.time)
    description: str = ""


@dataclass(frozen=True, slots=True)
class SpectrumMetrics:
    """Metrics for spectrum utilization and quality.

    Attributes:
        band_id: The band these metrics apply to.
        utilization_ratio: Fraction of spectrum in use (0.0–1.0).
        average_snr_db: Average signal-to-noise ratio in dB.
        interference_ratio: Fraction of channels with interference (0.0–1.0).
        active_allocations: Number of active allocations.
        timestamp: When metrics were computed.
    """

    band_id: str
    utilization_ratio: float
    average_snr_db: float
    interference_ratio: float
    active_allocations: int
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not 0.0 <= self.utilization_ratio <= 1.0:
            raise ValueError(
                f"utilization_ratio must be in [0, 1], got {self.utilization_ratio}"
            )
        if not 0.0 <= self.interference_ratio <= 1.0:
            raise ValueError(
                f"interference_ratio must be in [0, 1], got {self.interference_ratio}"
            )


# ---------------------------------------------------------------------------
# InterferenceDetector
# ---------------------------------------------------------------------------


class InterferenceDetector:
    """Detects and classifies interference in spectrum bands.

    Uses spectral analysis and threshold-based detection to identify
    co-channel, adjacent-channel, and intermodulation interference.

    Attributes:
        snr_threshold_db: SNR below which interference is suspected.
        adjacent_channel_rejection_db: Required adjacent channel rejection.
        intermodulation_order: Order of intermodulation to check.
    """

    def __init__(
        self,
        snr_threshold_db: float = 15.0,
        adjacent_channel_rejection_db: float = 40.0,
        intermodulation_order: int = 3,
    ) -> None:
        """Initialize the InterferenceDetector.

        Args:
            snr_threshold_db: SNR threshold for interference detection.
            adjacent_channel_rejection_db: Required adjacent channel rejection.
            intermodulation_order: Order of intermodulation products to check.

        Raises:
            ValueError: If parameters are invalid.
        """
        if intermodulation_order < 2:
            raise ValueError("intermodulation_order must be >= 2")

        self._snr_threshold: float = snr_threshold_db
        self._adjacent_rejection: float = adjacent_channel_rejection_db
        self._im_order: int = intermodulation_order
        self._event_counter: int = 0
        self._detected_events: List[InterferenceEvent] = []

    @property
    def detected_events(self) -> Tuple[InterferenceEvent, ...]:
        """All detected interference events."""
        return tuple(self._detected_events)

    def detect_co_channel_interference(
        self,
        band: SpectrumBand,
        allocations: List[SpectrumAllocation],
        measurements: Dict[int, float],
    ) -> List[InterferenceEvent]:
        """Detect co-channel interference.

        Co-channel interference occurs when multiple users are allocated
        to the same channel in the same band.

        Args:
            band: The spectrum band to check.
            allocations: Current allocations in the band.
            measurements: Mapping of channel_index to measured power in dBm.

        Returns:
            List of detected InterferenceEvent objects.
        """
        events: List[InterferenceEvent] = []
        channel_users: Dict[int, List[SpectrumAllocation]] = defaultdict(list)

        for alloc in allocations:
            if alloc.band_id == band.band_id and alloc.is_active():
                channel_users[alloc.channel_index].append(alloc)

        for channel_idx, users in channel_users.items():
            if len(users) <= 1:
                continue

            # Multiple users on the same channel
            self._event_counter += 1
            measured_power = measurements.get(channel_idx, -100.0)
            severity = self._classify_severity(measured_power, band.max_power_dbm)

            event = InterferenceEvent(
                event_id=f"IF-{self._event_counter:06d}",
                source_band_id=band.band_id,
                affected_band_id=band.band_id,
                interference_type=InterferenceType.CO_CHANNEL,
                severity=severity,
                frequency_mhz=band.get_channel_frequency(channel_idx),
                power_dbm=measured_power,
                description=(
                    f"Co-channel interference: {len(users)} users on "
                    f"channel {channel_idx} ({band.band_id})"
                ),
            )
            events.append(event)
            self._detected_events.append(event)

        return events

    def detect_adjacent_channel_interference(
        self,
        band: SpectrumBand,
        allocations: List[SpectrumAllocation],
        measurements: Dict[int, float],
    ) -> List[InterferenceEvent]:
        """Detect adjacent-channel interference.

        Adjacent-channel interference occurs when power from one channel
        leaks into neighboring channels.

        Args:
            band: The spectrum band to check.
            allocations: Current allocations in the band.
            measurements: Mapping of channel_index to measured power in dBm.

        Returns:
            List of detected InterferenceEvent objects.
        """
        events: List[InterferenceEvent] = []
        active_channels = {
            a.channel_index
            for a in allocations
            if a.band_id == band.band_id and a.is_active()
        }

        for ch in active_channels:
            ch_power = measurements.get(ch, -100.0)
            # Check adjacent channels
            for adj_ch in (ch - 1, ch + 1):
                if adj_ch < 0 or adj_ch >= band.num_channels:
                    continue
                adj_power = measurements.get(adj_ch, -100.0)
                # If adjacent channel has significant power relative to main
                leakage = ch_power - adj_power
                if leakage < self._adjacent_rejection and adj_power > -80.0:
                    self._event_counter += 1
                    event = InterferenceEvent(
                        event_id=f"IF-{self._event_counter:06d}",
                        source_band_id=band.band_id,
                        affected_band_id=band.band_id,
                        interference_type=InterferenceType.ADJACENT_CHANNEL,
                        severity=InterferenceSeverity.MEDIUM,
                        frequency_mhz=band.get_channel_frequency(adj_ch),
                        power_dbm=adj_power,
                        description=(
                            f"Adjacent-channel interference from channel {ch} "
                            f"to channel {adj_ch} (leakage: {leakage:.1f} dB)"
                        ),
                    )
                    events.append(event)
                    self._detected_events.append(event)

        return events

    def detect_intermodulation(
        self,
        band: SpectrumBand,
        allocations: List[SpectrumAllocation],
    ) -> List[InterferenceEvent]:
        """Detect intermodulation interference.

        Intermodulation occurs when two or more signals mix in a
        non-linear device, creating spurious signals at new frequencies.

        Args:
            band: The spectrum band to check.
            allocations: Current allocations in the band.

        Returns:
            List of detected InterferenceEvent objects.
        """
        events: List[InterferenceEvent] = []
        active_allocs = [
            a for a in allocations
            if a.band_id == band.band_id and a.is_active()
        ]

        if len(active_allocs) < 2:
            return events

        # Check 3rd-order intermodulation: 2*f1 - f2 and 2*f2 - f1
        for i, alloc_a in enumerate(active_allocs):
            for alloc_b in active_allocs[i + 1:]:
                f1 = band.get_channel_frequency(alloc_a.channel_index)
                f2 = band.get_channel_frequency(alloc_b.channel_index)

                im_products = [
                    2 * f1 - f2,
                    2 * f2 - f1,
                    f1 + f2 - band.frequency_range.center_mhz,
                ]

                for im_freq in im_products:
                    if band.frequency_range.contains(im_freq):
                        self._event_counter += 1
                        event = InterferenceEvent(
                            event_id=f"IF-{self._event_counter:06d}",
                            source_band_id=band.band_id,
                            affected_band_id=band.band_id,
                            interference_type=InterferenceType.INTERMODULATION,
                            severity=InterferenceSeverity.HIGH,
                            frequency_mhz=im_freq,
                            power_dbm=min(alloc_a.power_dbm, alloc_b.power_dbm) - 20.0,
                            description=(
                                f"Intermodulation product at {im_freq:.1f} MHz "
                                f"from channels at {f1:.1f} and {f2:.1f} MHz"
                            ),
                        )
                        events.append(event)
                        self._detected_events.append(event)

        return events

    def detect_all(
        self,
        band: SpectrumBand,
        allocations: List[SpectrumAllocation],
        measurements: Dict[int, float],
    ) -> List[InterferenceEvent]:
        """Run all interference detection methods.

        Args:
            band: The spectrum band to check.
            allocations: Current allocations.
            measurements: Channel power measurements.

        Returns:
            Combined list of all detected interference events.
        """
        events: List[InterferenceEvent] = []
        events.extend(self.detect_co_channel_interference(band, allocations, measurements))
        events.extend(self.detect_adjacent_channel_interference(band, allocations, measurements))
        events.extend(self.detect_intermodulation(band, allocations))
        return events

    def get_events_by_severity(
        self, severity: InterferenceSeverity
    ) -> List[InterferenceEvent]:
        """Get events filtered by severity.

        Args:
            severity: The severity level to filter by.

        Returns:
            List of matching events.
        """
        return [e for e in self._detected_events if e.severity == severity]

    def get_events_by_type(
        self, interference_type: InterferenceType
    ) -> List[InterferenceEvent]:
        """Get events filtered by interference type.

        Args:
            interference_type: The type to filter by.

        Returns:
            List of matching events.
        """
        return [
            e for e in self._detected_events
            if e.interference_type == interference_type
        ]

    def clear_events(self) -> None:
        """Clear all detected events."""
        self._detected_events.clear()

    @staticmethod
    def _classify_severity(measured_power: float, max_power: float) -> InterferenceSeverity:
        """Classify interference severity based on power levels."""
        margin = max_power - measured_power
        if margin < 5:
            return InterferenceSeverity.SEVERE
        if margin < 15:
            return InterferenceSeverity.HIGH
        if margin < 30:
            return InterferenceSeverity.MEDIUM
        return InterferenceSeverity.LOW

    def __repr__(self) -> str:
        return (
            f"InterferenceDetector(events={len(self._detected_events)}, "
            f"snr_threshold={self._snr_threshold}dB)"
        )


# ---------------------------------------------------------------------------
# SpectrumOptimizer
# ---------------------------------------------------------------------------


class SpectrumOptimizer:
    """Optimizes spectrum allocation using constraint-based optimization.

    Supports multiple optimization objectives:
    - Maximize utilization
    - Minimize interference
    - Maximize fairness
    - Maximize throughput

    Uses a greedy heuristic approach with backtracking for
    constraint satisfaction.

    Attributes:
        optimization_objective: Primary optimization objective.
        fairness_weight: Weight for fairness in multi-objective optimization.
        interference_penalty: Penalty factor for interference in cost function.
    """

    def __init__(
        self,
        optimization_objective: str = "maximize_throughput",
        fairness_weight: float = 0.3,
        interference_penalty: float = 2.0,
    ) -> None:
        """Initialize the SpectrumOptimizer.

        Args:
            optimization_objective: One of "maximize_throughput",
                "minimize_interference", "maximize_fairness", "maximize_utilization".
            fairness_weight: Weight for fairness (0.0–1.0).
            interference_penalty: Penalty multiplier for interference.

        Raises:
            ValueError: If parameters are invalid.
        """
        valid_objectives = {
            "maximize_throughput",
            "minimize_interference",
            "maximize_fairness",
            "maximize_utilization",
        }
        if optimization_objective not in valid_objectives:
            raise ValueError(
                f"Invalid objective '{optimization_objective}'. "
                f"Must be one of: {valid_objectives}"
            )
        if not 0.0 <= fairness_weight <= 1.0:
            raise ValueError("fairness_weight must be in [0, 1]")
        if interference_penalty < 0:
            raise ValueError("interference_penalty must be non-negative")

        self._objective: str = optimization_objective
        self._fairness_weight: float = fairness_weight
        self._interference_penalty: float = interference_penalty

    def optimize(
        self,
        band: SpectrumBand,
        demands: List[Tuple[str, int, UserPriority]],
        existing_allocations: List[SpectrumAllocation],
    ) -> List[SpectrumAllocation]:
        """Optimize spectrum allocation for given demands.

        Args:
            band: The spectrum band to optimize.
            demands: List of (user_id, required_channels, priority) tuples.
            existing_allocations: Current allocations to consider.

        Returns:
            List of new optimized allocations.

        Raises:
            OptimizationError: If optimization fails.
        """
        # Sort demands by priority (lower number = higher priority)
        sorted_demands = sorted(demands, key=lambda d: d[2].value)

        # Track channel occupancy
        occupied: Dict[int, Optional[str]] = {}
        for alloc in existing_allocations:
            if alloc.band_id == band.band_id and alloc.is_active():
                occupied[alloc.channel_index] = alloc.user_id

        new_allocations: List[SpectrumAllocation] = []
        allocation_counter = 0

        for user_id, required_channels, priority in sorted_demands:
            channels_assigned = 0
            for ch in range(band.num_channels):
                if channels_assigned >= required_channels:
                    break
                if ch in occupied:
                    continue

                # Check for interference with adjacent channels
                if self._would_cause_interference(ch, occupied, band):
                    continue

                # Assign channel
                allocation_counter += 1
                alloc = SpectrumAllocation(
                    allocation_id=f"ALLOC-{allocation_counter:06d}",
                    band_id=band.band_id,
                    channel_index=ch,
                    user_id=user_id,
                    priority=priority,
                    power_dbm=self._compute_optimal_power(ch, occupied, band),
                    duration_seconds=3600.0,
                )
                new_allocations.append(alloc)
                occupied[ch] = user_id
                channels_assigned += 1

            if channels_assigned < required_channels:
                logger.warning(
                    "Could only assign %d/%d channels for user %s",
                    channels_assigned,
                    required_channels,
                    user_id,
                )

        return new_allocations

    def compute_utilization(
        self, band: SpectrumBand, allocations: List[SpectrumAllocation]
    ) -> float:
        """Compute spectrum utilization ratio.

        Args:
            band: The spectrum band.
            allocations: Current allocations.

        Returns:
            Utilization ratio in [0.0, 1.0].
        """
        active_channels = {
            a.channel_index
            for a in allocations
            if a.band_id == band.band_id and a.is_active()
        }
        if band.num_channels == 0:
            return 0.0
        return len(active_channels) / band.num_channels

    def compute_fairness_index(
        self, allocations: List[SpectrumAllocation]
    ) -> float:
        """Compute Jain's fairness index across users.

        Args:
            allocations: Current allocations.

        Returns:
            Fairness index in [0.0, 1.0]. 1.0 is perfectly fair.
        """
        user_channels: Dict[str, int] = defaultdict(int)
        for alloc in allocations:
            if alloc.is_active():
                user_channels[alloc.user_id] += 1

        if not user_channels:
            return 1.0

        values = list(user_channels.values())
        n = len(values)
        sum_values = sum(values)
        sum_squares = sum(v ** 2 for v in values)

        if sum_squares == 0:
            return 1.0
        return (sum_values ** 2) / (n * sum_squares)

    def _would_cause_interference(
        self,
        channel: int,
        occupied: Dict[int, Optional[str]],
        band: SpectrumBand,
    ) -> bool:
        """Check if assigning a channel would cause interference."""
        # Check adjacent channels
        for adj_ch in (channel - 1, channel + 1):
            if adj_ch in occupied and occupied[adj_ch] is not None:
                return True
        return False

    def _compute_optimal_power(
        self,
        channel: int,
        occupied: Dict[int, Optional[str]],
        band: SpectrumBand,
    ) -> float:
        """Compute optimal transmit power for a channel."""
        # Simple power control: reduce power if adjacent channels are occupied
        power = band.max_power_dbm
        for adj_ch in (channel - 1, channel + 1):
            if adj_ch in occupied and occupied[adj_ch] is not None:
                power = min(power, band.max_power_dbm - 10.0)
        return power

    def __repr__(self) -> str:
        return (
            f"SpectrumOptimizer(objective={self._objective!r}, "
            f"fairness_weight={self._fairness_weight})"
        )


# ---------------------------------------------------------------------------
# DynamicSpectrumAllocator
# ---------------------------------------------------------------------------


class DynamicSpectrumAllocator:
    """Real-time dynamic spectrum allocation engine.

    Manages spectrum bands, handles allocation requests, and
    continuously optimizes allocations based on changing conditions.

    Attributes:
        strategy: The allocation strategy to use.
        auto_optimize: Whether to run optimization periodically.
        optimization_interval_seconds: Interval between optimizations.
    """

    def __init__(
        self,
        strategy: AllocationStrategy = AllocationStrategy.BEST_FIT,
        auto_optimize: bool = True,
        optimization_interval_seconds: float = 60.0,
    ) -> None:
        """Initialize the DynamicSpectrumAllocator.

        Args:
            strategy: Allocation strategy.
            auto_optimize: Whether to auto-optimize.
            optimization_interval_seconds: Optimization interval.

        Raises:
            ValueError: If optimization_interval_seconds <= 0.
        """
        if optimization_interval_seconds <= 0:
            raise ValueError("optimization_interval_seconds must be positive")

        self._strategy: AllocationStrategy = strategy
        self._auto_optimize: bool = auto_optimize
        self._optimization_interval: float = optimization_interval_seconds
        self._bands: Dict[str, SpectrumBand] = {}
        self._allocations: Dict[str, SpectrumAllocation] = {}
        self._optimizer: SpectrumOptimizer = SpectrumOptimizer()
        self._interference_detector: InterferenceDetector = InterferenceDetector()
        self._allocation_counter: int = 0
        self._last_optimization: float = 0.0

    @property
    def strategy(self) -> AllocationStrategy:
        """Current allocation strategy."""
        return self._strategy

    @strategy.setter
    def strategy(self, value: AllocationStrategy) -> None:
        self._strategy = value

    @property
    def bands(self) -> Dict[str, SpectrumBand]:
        """Registered spectrum bands."""
        return dict(self._bands)

    @property
    def allocations(self) -> Dict[str, SpectrumAllocation]:
        """Current allocations."""
        return dict(self._allocations)

    def register_band(self, band: SpectrumBand) -> None:
        """Register a spectrum band for management.

        Args:
            band: The band to register.

        Raises:
            ValueError: If a band with the same ID already exists.
        """
        if band.band_id in self._bands:
            raise ValueError(f"Band '{band.band_id}' already registered")
        self._bands[band.band_id] = band

    def unregister_band(self, band_id: str) -> Optional[SpectrumBand]:
        """Unregister a band.

        Args:
            band_id: The band to remove.

        Returns:
            The removed band, or None if not found.
        """
        return self._bands.pop(band_id, None)

    def request_allocation(
        self,
        band_id: str,
        user_id: str,
        num_channels: int = 1,
        priority: UserPriority = UserPriority.MEDIUM,
        duration_seconds: float = 3600.0,
        power_dbm: Optional[float] = None,
    ) -> List[SpectrumAllocation]:
        """Request spectrum allocation.

        Args:
            band_id: The band to allocate from.
            user_id: The user requesting allocation.
            num_channels: Number of channels needed.
            priority: Priority level.
            duration_seconds: How long the allocation should last.
            power_dbm: Desired power level (None for auto).

        Returns:
            List of allocated SpectrumAllocation objects.

        Raises:
            BandNotFoundError: If the band doesn't exist.
            AllocationConflictError: If no channels are available.
        """
        if band_id not in self._bands:
            raise BandNotFoundError(f"Band '{band_id}' not found")

        band = self._bands[band_id]
        active_allocs = self._get_active_allocations(band_id)

        # Find available channels
        occupied = {a.channel_index for a in active_allocs}
        available = [
            ch for ch in range(band.num_channels) if ch not in occupied
        ]

        if len(available) < num_channels:
            raise AllocationConflictError(
                f"Insufficient channels in band '{band_id}': "
                f"need {num_channels}, have {len(available)}"
            )

        # Select channels based on strategy
        selected = self._select_channels(available, num_channels, band, active_allocs)

        # Create allocations
        new_allocations: List[SpectrumAllocation] = []
        for ch in selected:
            self._allocation_counter += 1
            alloc = SpectrumAllocation(
                allocation_id=f"ALLOC-{self._allocation_counter:06d}",
                band_id=band_id,
                channel_index=ch,
                user_id=user_id,
                priority=priority,
                power_dbm=power_dbm or band.max_power_dbm,
                duration_seconds=duration_seconds,
            )
            self._allocations[alloc.allocation_id] = alloc
            new_allocations.append(alloc)

        return new_allocations

    def release_allocation(self, allocation_id: str) -> Optional[SpectrumAllocation]:
        """Release an allocation.

        Args:
            allocation_id: The allocation to release.

        Returns:
            The released allocation, or None if not found.
        """
        return self._allocations.pop(allocation_id, None)

    def release_user_allocations(self, user_id: str) -> List[SpectrumAllocation]:
        """Release all allocations for a user.

        Args:
            user_id: The user whose allocations to release.

        Returns:
            List of released allocations.
        """
        to_release = [
            aid for aid, a in self._allocations.items()
            if a.user_id == user_id
        ]
        released = []
        for aid in to_release:
            alloc = self._allocations.pop(aid)
            released.append(alloc)
        return released

    def get_band_metrics(self, band_id: str) -> SpectrumMetrics:
        """Compute metrics for a band.

        Args:
            band_id: The band to analyze.

        Returns:
            SpectrumMetrics for the band.

        Raises:
            BandNotFoundError: If the band doesn't exist.
        """
        if band_id not in self._bands:
            raise BandNotFoundError(f"Band '{band_id}' not found")

        band = self._bands[band_id]
        active = self._get_active_allocations(band_id)
        utilization = self._optimizer.compute_utilization(band, active)

        # Estimate SNR (simplified model)
        snr_values = []
        for ch in range(band.num_channels):
            ch_power = -100.0
            for a in active:
                if a.channel_index == ch:
                    ch_power = a.power_dbm
                    break
            noise_floor = -100.0
            snr_values.append(ch_power - noise_floor)

        avg_snr = statistics.mean(snr_values) if snr_values else 0.0

        # Count interference events
        if_events = self._interference_detector.detect_all(band, active, {})
        interference_ratio = len(if_events) / max(1, band.num_channels)

        return SpectrumMetrics(
            band_id=band_id,
            utilization_ratio=utilization,
            average_snr_db=avg_snr,
            interference_ratio=min(1.0, interference_ratio),
            active_allocations=len(active),
        )

    def run_optimization(self) -> List[SpectrumAllocation]:
        """Run optimization across all bands.

        Returns:
            List of new optimized allocations.
        """
        new_allocations: List[SpectrumAllocation] = []
        for band in self._bands.values():
            active = self._get_active_allocations(band.band_id)
            demands: List[Tuple[str, int, UserPriority]] = []
            for alloc in active:
                demands.append((alloc.user_id, 1, alloc.priority))
            optimized = self._optimizer.optimize(band, demands, active)
            new_allocations.extend(optimized)
        self._last_optimization = time.time()
        return new_allocations

    def _get_active_allocations(self, band_id: str) -> List[SpectrumAllocation]:
        """Get all active allocations for a band."""
        return [
            a for a in self._allocations.values()
            if a.band_id == band_id and a.is_active()
        ]

    def _select_channels(
        self,
        available: List[int],
        num_channels: int,
        band: SpectrumBand,
        active_allocs: List[SpectrumAllocation],
    ) -> List[int]:
        """Select channels based on the current strategy."""
        if not available:
            return []

        if self._strategy == AllocationStrategy.FIRST_FIT:
            return available[:num_channels]

        if self._strategy == AllocationStrategy.BEST_FIT:
            # Choose channels that minimize fragmentation
            # Prefer channels adjacent to occupied ones
            occupied = {a.channel_index for a in active_allocs}
            scored = []
            for ch in available:
                score = 0
                if ch - 1 in occupied:
                    score += 1
                if ch + 1 in occupied:
                    score += 1
                scored.append((score, ch))
            scored.sort(reverse=True)
            return [ch for _, ch in scored[:num_channels]]

        if self._strategy == AllocationStrategy.WORST_FIT:
            # Choose channels that maximize fragmentation (spread out)
            if len(available) <= num_channels:
                return available
            step = len(available) / num_channels
            return [available[int(i * step)] for i in range(num_channels)]

        if self._strategy == AllocationStrategy.RANDOM_FIT:
            return random.sample(available, min(num_channels, len(available)))

        if self._strategy == AllocationStrategy.FAIR_SHARING:
            # Round-robin style: pick channels spread across the band
            if len(available) <= num_channels:
                return available
            step = len(available) / num_channels
            return [available[int(i * step)] for i in range(num_channels)]

        if self._strategy == AllocationStrategy.PRIORITY_BASED:
            # Same as first fit for now; priority is handled at allocation time
            return available[:num_channels]

        return available[:num_channels]

    def __repr__(self) -> str:
        return (
            f"DynamicSpectrumAllocator(bands={len(self._bands)}, "
            f"allocations={len(self._allocations)}, "
            f"strategy={self._strategy.value!r})"
        )


# ---------------------------------------------------------------------------
# SpectrumManager (High-level orchestrator)
# ---------------------------------------------------------------------------


class SpectrumManager:
    """High-level orchestrator for spectrum management.

    Ties together bands, allocations, interference detection,
    optimization, and dynamic allocation into a unified interface.

    Attributes:
        allocator: The dynamic spectrum allocator.
        interference_detector: The interference detector.
        optimizer: The spectrum optimizer.
    """

    def __init__(
        self,
        allocator: Optional[DynamicSpectrumAllocator] = None,
        interference_detector: Optional[InterferenceDetector] = None,
        optimizer: Optional[SpectrumOptimizer] = None,
    ) -> None:
        """Initialize SpectrumManager.

        Args:
            allocator: Dynamic spectrum allocator.
            interference_detector: Interference detector.
            optimizer: Spectrum optimizer.
        """
        self._allocator = allocator or DynamicSpectrumAllocator()
        self._interference_detector = interference_detector or InterferenceDetector()
        self._optimizer = optimizer or SpectrumOptimizer()

    @property
    def allocator(self) -> DynamicSpectrumAllocator:
        """The dynamic spectrum allocator."""
        return self._allocator

    @property
    def interference_detector(self) -> InterferenceDetector:
        """The interference detector."""
        return self._interference_detector

    @property
    def optimizer(self) -> SpectrumOptimizer:
        """The spectrum optimizer."""
        return self._optimizer

    def register_band(self, band: SpectrumBand) -> None:
        """Register a spectrum band.

        Args:
            band: The band to register.
        """
        self._allocator.register_band(band)

    def allocate(
        self,
        band_id: str,
        user_id: str,
        num_channels: int = 1,
        priority: UserPriority = UserPriority.MEDIUM,
        duration_seconds: float = 3600.0,
    ) -> List[SpectrumAllocation]:
        """Allocate spectrum to a user.

        Args:
            band_id: The band to allocate from.
            user_id: The user requesting allocation.
            num_channels: Number of channels needed.
            priority: Priority level.
            duration_seconds: Duration of allocation.

        Returns:
            List of allocations.
        """
        return self._allocator.request_allocation(
            band_id, user_id, num_channels, priority, duration_seconds
        )

    def deallocate(self, allocation_id: str) -> Optional[SpectrumAllocation]:
        """Deallocate spectrum.

        Args:
            allocation_id: The allocation to release.

        Returns:
            The released allocation, or None.
        """
        return self._allocator.release_allocation(allocation_id)

    def detect_interference(
        self, band_id: str, measurements: Dict[int, float]
    ) -> List[InterferenceEvent]:
        """Detect interference in a band.

        Args:
            band_id: The band to check.
            measurements: Channel power measurements.

        Returns:
            List of detected interference events.
        """
        if band_id not in self._allocator.bands:
            raise BandNotFoundError(f"Band '{band_id}' not found")
        band = self._allocator.bands[band_id]
        active = self._allocator._get_active_allocations(band_id)
        return self._interference_detector.detect_all(band, active, measurements)

    def optimize(self) -> List[SpectrumAllocation]:
        """Run spectrum optimization.

        Returns:
            List of new optimized allocations.
        """
        return self._allocator.run_optimization()

    def get_metrics(self, band_id: str) -> SpectrumMetrics:
        """Get metrics for a band.

        Args:
            band_id: The band to analyze.

        Returns:
            SpectrumMetrics.
        """
        return self._allocator.get_band_metrics(band_id)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of spectrum state.

        Returns:
            Dictionary with spectrum statistics.
        """
        bands = self._allocator.bands
        allocations = self._allocator.allocations
        active = [a for a in allocations.values() if a.is_active()]

        return {
            "band_count": len(bands),
            "total_allocations": len(allocations),
            "active_allocations": len(active),
            "total_interference_events": len(
                self._interference_detector.detected_events
            ),
            "bands": {bid: band.to_dict() for bid, band in bands.items()},
        }

    def __repr__(self) -> str:
        return (
            f"SpectrumManager(bands={len(self._allocator.bands)}, "
            f"allocations={len(self._allocator.allocations)})"
        )
