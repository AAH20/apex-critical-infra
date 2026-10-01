"""Apex Critical Infrastructure — Data Center Operations.

Autonomous DC operations engine providing cooling optimization, power
management, and capacity planning for production data center environments.

Integrates with Data Center Commander (DC lifecycle), Apex_ULL (ultra-low
latency), ApexGraphSwarm (multi-agent orchestration), and GRC_Claw
(governance).
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

DEFAULT_POWER_CAPACITY_KW = 10_000.0
DEFAULT_COOLING_CAPACITY_TONS = 3_000.0
DEFAULT_PUE_TARGET = 1.4
DEFAULT_TEMP_SETPOINT_C = 22.0
DEFAULT_TEMP_TOLERANCE_C = 2.0
CRITICAL_TEMP_THRESHOLD_C = 35.0
DEFAULT_CAPACITY_BUFFER_PCT = 20.0
MAX_POWER_DRAW_PCT = 90.0
MIN_UPTIME_SLA_PCT = 99.99


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DCOpsState(Enum):
    """Operational state of the data center."""

    INITIALIZING = auto()
    ONLINE = auto()
    DEGRADED = auto()
    MAINTENANCE = auto()
    OFFLINE = auto()
    EMERGENCY_SHUTDOWN = auto()


class CoolingMode(Enum):
    """Active cooling strategy."""

    MECHANICAL = auto()
    FREE_COOLING = auto()
    HYBRID = auto()
    ECONOMIZER = auto()
    EMERGENCY = auto()


class PowerSource(Enum):
    """Current power source."""

    GRID = auto()
    UPS = auto()
    GENERATOR = auto()
    RENEWABLE = auto()
    BATTERY = auto()


class CapacityAlertLevel(Enum):
    """Severity of capacity threshold breach."""

    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()
    EMERGENCY = auto()


class WorkloadPriority(Enum):
    """Priority classification for workload scheduling."""

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BEST_EFFORT = 5


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DCOpsError(Exception):
    """Base exception for data center operations."""


class CapacityExceededError(DCOpsError):
    """Raised when a capacity limit would be exceeded."""


class ThermalRunawayError(DCOpsError):
    """Raised when cooling cannot maintain safe temperatures."""


class PowerBudgetExceededError(DCOpsError):
    """Raised when power draw exceeds the configured budget."""


class InvalidStateTransitionError(DCOpsError):
    """Raised when an invalid state transition is attempted."""


class SensorReadError(DCOpsError):
    """Raised when a sensor reading is unavailable or invalid."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SensorReading:
    """A single sensor measurement."""

    sensor_id: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    zone: str = "default"

    def __post_init__(self) -> None:
        if not self.sensor_id:
            raise ValueError("sensor_id must be non-empty")
        if self.value != self.value:  # NaN check
            raise ValueError(f"sensor {self.sensor_id}: value is NaN")


@dataclass(frozen=True, slots=True)
class ThermalZone:
    """A thermal zone within the data center."""

    zone_id: str
    current_temp_c: float
    setpoint_temp_c: float = DEFAULT_TEMP_SETPOINT_C
    humidity_pct: float = 45.0
    airflow_cfm: float = 0.0
    cooling_active: bool = True

    def __post_init__(self) -> None:
        if not self.zone_id:
            raise ValueError("zone_id must be non-empty")
        if not 0.0 <= self.humidity_pct <= 100.0:
            raise ValueError(
                f"zone {self.zone_id}: humidity {self.humidity_pct}% out of range"
            )

    @property
    def temp_delta(self) -> float:
        """Temperature deviation from setpoint."""
        return self.current_temp_c - self.setpoint_temp_c

    @property
    def is_critical(self) -> bool:
        """Whether the zone is in a critical thermal state."""
        return self.current_temp_c >= CRITICAL_TEMP_THRESHOLD_C


@dataclass(frozen=True, slots=True)
class PowerReading:
    """A power measurement at a point in time."""

    total_draw_kw: float
    it_load_kw: float
    cooling_load_kw: float
    pue: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: PowerSource = PowerSource.GRID

    def __post_init__(self) -> None:
        if self.total_draw_kw < 0:
            raise ValueError("total_draw_kw must be non-negative")
        if self.pue < 1.0:
            raise ValueError(f"PUE {self.pue} is below theoretical minimum of 1.0")


@dataclass(frozen=True, slots=True)
class CapacitySnapshot:
    """Current capacity utilization snapshot."""

    compute_used_units: int
    compute_total_units: int
    storage_used_tb: float
    storage_total_tb: float
    network_used_gbps: float
    network_total_gbps: float
    power_used_kw: float
    power_total_kw: float
    cooling_used_tons: float
    cooling_total_tons: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def compute_utilization_pct(self) -> float:
        """Compute utilization as a percentage."""
        if self.compute_total_units == 0:
            return 0.0
        return (self.compute_used_units / self.compute_total_units) * 100.0

    @property
    def storage_utilization_pct(self) -> float:
        """Storage utilization as a percentage."""
        if self.storage_total_tb == 0:
            return 0.0
        return (self.storage_used_tb / self.storage_total_tb) * 100.0

    @property
    def network_utilization_pct(self) -> float:
        """Network utilization as a percentage."""
        if self.network_total_gbps == 0:
            return 0.0
        return (self.network_used_gbps / self.network_total_gbps) * 100.0

    @property
    def power_utilization_pct(self) -> float:
        """Power utilization as a percentage."""
        if self.power_total_kw == 0:
            return 0.0
        return (self.power_used_kw / self.power_total_kw) * 100.0

    @property
    def cooling_utilization_pct(self) -> float:
        """Cooling utilization as a percentage."""
        if self.cooling_total_tons == 0:
            return 0.0
        return (self.cooling_used_tons / self.cooling_total_tons) * 100.0

    @property
    def max_utilization_pct(self) -> float:
        """Highest utilization across all resource types."""
        return max(
            self.compute_utilization_pct,
            self.storage_utilization_pct,
            self.network_utilization_pct,
            self.power_utilization_pct,
            self.cooling_utilization_pct,
        )


@dataclass(frozen=True, slots=True)
class CapacityAlert:
    """A capacity threshold alert."""

    level: CapacityAlertLevel
    resource_type: str
    current_value: float
    threshold_value: float
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True, slots=True)
class Workload:
    """A schedulable workload."""

    workload_id: str
    name: str
    priority: WorkloadPriority
    compute_units: int
    memory_gb: float
    storage_gb: float
    network_mbps: float
    max_latency_ms: float
    preferred_zone: str | None = None
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.workload_id:
            raise ValueError("workload_id must be non-empty")
        if self.compute_units <= 0:
            raise ValueError("compute_units must be positive")
        if self.memory_gb <= 0:
            raise ValueError("memory_gb must be positive")
        if self.max_latency_ms <= 0:
            raise ValueError("max_latency_ms must be positive")


@dataclass(frozen=True, slots=True)
class WorkloadPlacement:
    """Result of a workload placement decision."""

    workload: Workload
    target_zone: str
    estimated_latency_ms: float
    confidence: float  # 0.0–1.0
    reason: str


@dataclass(frozen=True, slots=True)
class OpsAction:
    """An autonomous operations action."""

    action_id: str
    action_type: str
    target: str
    parameters: dict[str, Any] = field(default_factory=dict)
    executed_at: datetime = field(default_factory=datetime.utcnow)
    success: bool = True
    result: str = ""


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class SensorInterface(Protocol):
    """Protocol for sensor data providers."""

    def read_temperature(self, zone_id: str) -> SensorReading:
        """Read current temperature for a zone."""
        ...

    def read_humidity(self, zone_id: str) -> SensorReading:
        """Read current humidity for a zone."""
        ...

    def read_power(self) -> PowerReading:
        """Read current power metrics."""
        ...

    def read_all_zones(self) -> list[ThermalZone]:
        """Read all thermal zones."""
        ...


class ActuatorInterface(Protocol):
    """Protocol for control actuators."""

    def set_cooling_setpoint(self, zone_id: str, temp_c: float) -> bool:
        """Set cooling setpoint for a zone."""
        ...

    def set_fan_speed(self, zone_id: str, pct: float) -> bool:
        """Set fan speed percentage for a zone."""
        ...

    def set_power_limit(self, limit_kw: float) -> bool:
        """Set a power draw limit."""
        ...

    def enable_free_cooling(self, zone_id: str) -> bool:
        """Enable free cooling for a zone."""
        ...

    def disable_free_cooling(self, zone_id: str) -> bool:
        """Disable free cooling for a zone."""
        ...


class MetricsSink(Protocol):
    """Protocol for metrics emission."""

    def emit(self, metric_name: str, value: float, tags: dict[str, str]) -> None:
        """Emit a single metric."""
        ...


# ---------------------------------------------------------------------------
# Autonomous Operations Engine
# ---------------------------------------------------------------------------


class AutonomousDCOps:
    """Autonomous data center operations engine.

    Continuously monitors thermal, power, and capacity metrics; makes
    autonomous decisions for cooling optimization, power management,
    and capacity planning.

    Parameters
    ----------
    sensor_interface:
        Provider for sensor readings.
    actuator_interface:
        Provider for control actuators.
    power_capacity_kw:
        Total power capacity in kilowatts.
    cooling_capacity_tons:
        Total cooling capacity in refrigeration tons.
    pue_target:
        Target Power Usage Effectiveness.
    temp_setpoint_c:
        Target temperature setpoint in Celsius.
    temp_tolerance_c:
        Acceptable temperature deviation in Celsius.
    metrics_sink:
        Optional sink for emitting operational metrics.
    """

    _VALID_TRANSITIONS: dict[DCOpsState, set[DCOpsState]] = {
        DCOpsState.INITIALIZING: {
            DCOpsState.ONLINE,
            DCOpsState.OFFLINE,
            DCOpsState.EMERGENCY_SHUTDOWN,
        },
        DCOpsState.ONLINE: {
            DCOpsState.DEGRADED,
            DCOpsState.MAINTENANCE,
            DCOpsState.OFFLINE,
            DCOpsState.EMERGENCY_SHUTDOWN,
        },
        DCOpsState.DEGRADED: {
            DCOpsState.ONLINE,
            DCOpsState.MAINTENANCE,
            DCOpsState.OFFLINE,
            DCOpsState.EMERGENCY_SHUTDOWN,
        },
        DCOpsState.MAINTENANCE: {
            DCOpsState.ONLINE,
            DCOpsState.OFFLINE,
            DCOpsState.EMERGENCY_SHUTDOWN,
        },
        DCOpsState.OFFLINE: {
            DCOpsState.INITIALIZING,
            DCOpsState.EMERGENCY_SHUTDOWN,
        },
        DCOpsState.EMERGENCY_SHUTDOWN: {
            DCOpsState.INITIALIZING,
        },
    }

    def __init__(
        self,
        sensor_interface: SensorInterface,
        actuator_interface: ActuatorInterface,
        *,
        power_capacity_kw: float = DEFAULT_POWER_CAPACITY_KW,
        cooling_capacity_tons: float = DEFAULT_COOLING_CAPACITY_TONS,
        pue_target: float = DEFAULT_PUE_TARGET,
        temp_setpoint_c: float = DEFAULT_TEMP_SETPOINT_C,
        temp_tolerance_c: float = DEFAULT_TEMP_TOLERANCE_C,
        metrics_sink: MetricsSink | None = None,
    ) -> None:
        if power_capacity_kw <= 0:
            raise ValueError("power_capacity_kw must be positive")
        if cooling_capacity_tons <= 0:
            raise ValueError("cooling_capacity_tons must be positive")
        if pue_target < 1.0:
            raise ValueError("pue_target must be >= 1.0")
        if temp_tolerance_c < 0:
            raise ValueError("temp_tolerance_c must be non-negative")

        self._sensors = sensor_interface
        self._actuators = actuator_interface
        self._power_capacity_kw = power_capacity_kw
        self._cooling_capacity_tons = cooling_capacity_tons
        self._pue_target = pue_target
        self._temp_setpoint_c = temp_setpoint_c
        self._temp_tolerance_c = temp_tolerance_c
        self._metrics = metrics_sink

        self._state = DCOpsState.INITIALIZING
        self._cooling_mode = CoolingMode.MECHANICAL
        self._active_alerts: list[CapacityAlert] = []
        self._action_log: list[OpsAction] = []
        self._running = False

    @property
    def state(self) -> DCOpsState:
        """Current operational state."""
        return self._state

    @property
    def cooling_mode(self) -> CoolingMode:
        """Current cooling mode."""
        return self._cooling_mode

    @property
    def active_alerts(self) -> list[CapacityAlert]:
        """Currently active capacity alerts."""
        return list(self._active_alerts)

    @property
    def action_log(self) -> list[OpsAction]:
        """Log of executed operations actions."""
        return list(self._action_log)

    def transition_to(self, new_state: DCOpsState) -> None:
        """Transition to a new operational state.

        Parameters
        ----------
        new_state:
            Target state.

        Raises
        ------
        InvalidStateTransitionError:
            If the transition is not allowed from the current state.
        """
        if new_state == self._state:
            return
        if new_state not in self._VALID_TRANSITIONS.get(self._state, set()):
            raise InvalidStateTransitionError(
                f"Cannot transition from {self._state.name} to {new_state.name}"
            )
        old_state = self._state
        self._state = new_state
        logger.info("State transition: %s -> %s", old_state.name, new_state.name)

    def start(self) -> None:
        """Start the autonomous operations loop."""
        if self._running:
            logger.warning("DCOps already running")
            return
        self._running = True
        self.transition_to(DCOpsState.ONLINE)
        logger.info("AutonomousDCOps started")

    def stop(self) -> None:
        """Stop the autonomous operations loop."""
        self._running = False
        logger.info("AutonomousDCOps stopped")

    def run_cycle(self) -> list[OpsAction]:
        """Execute one full monitoring and control cycle.

        Returns
        -------
        list[OpsAction]
            Actions taken during this cycle.
        """
        if not self._running:
            raise DCOpsError("DCOps is not running; call start() first")

        actions: list[OpsAction] = []

        # 1. Thermal management
        thermal_actions = self._manage_thermal()
        actions.extend(thermal_actions)

        # 2. Power management
        power_actions = self._manage_power()
        actions.extend(power_actions)

        # 3. Capacity monitoring
        capacity_actions = self._monitor_capacity()
        actions.extend(capacity_actions)

        self._action_log.extend(actions)
        return actions

    def _manage_thermal(self) -> list[OpsAction]:
        """Manage thermal conditions across all zones."""
        actions: list[OpsAction] = []
        try:
            zones = self._sensors.read_all_zones()
        except Exception as exc:
            logger.error("Failed to read thermal zones: %s", exc)
            raise SensorReadError(f"Thermal sensor read failed: {exc}") from exc

        for zone in zones:
            if zone.is_critical:
                action = OpsAction(
                    action_id=f"thermal-emergency-{zone.zone_id}",
                    action_type="emergency_cooling",
                    target=zone.zone_id,
                    parameters={"temp_c": zone.current_temp_c},
                    result="Emergency cooling activated",
                )
                self._actuators.set_cooling_setpoint(zone.zone_id, self._temp_setpoint_c - 5.0)
                self._actuators.set_fan_speed(zone.zone_id, 100.0)
                actions.append(action)
                self._emit_metric(
                    "dcops.thermal.emergency",
                    zone.current_temp_c,
                    {"zone": zone.zone_id},
                )
            elif abs(zone.temp_delta) > self._temp_tolerance_c:
                new_setpoint = self._temp_setpoint_c + (
                    self._temp_tolerance_c * (1 if zone.temp_delta > 0 else -1)
                )
                success = self._actuators.set_cooling_setpoint(zone.zone_id, new_setpoint)
                action = OpsAction(
                    action_id=f"thermal-adjust-{zone.zone_id}",
                    action_type="adjust_cooling_setpoint",
                    target=zone.zone_id,
                    parameters={"new_setpoint_c": new_setpoint},
                    success=success,
                    result=f"Setpoint adjusted to {new_setpoint}C",
                )
                actions.append(action)
                self._emit_metric(
                    "dcops.thermal.setpoint_adjust",
                    new_setpoint,
                    {"zone": zone.zone_id},
                )

        return actions

    def _manage_power(self) -> list[OpsAction]:
        """Manage power draw and PUE."""
        actions: list[OpsAction] = []
        try:
            reading = self._sensors.read_power()
        except Exception as exc:
            logger.error("Failed to read power metrics: %s", exc)
            raise SensorReadError(f"Power sensor read failed: {exc}") from exc

        self._emit_metric("dcops.power.draw_kw", reading.total_draw_kw, {})
        self._emit_metric("dcops.power.pue", reading.pue, {})

        # PUE optimization
        if reading.pue > self._pue_target * 1.1:
            action = OpsAction(
                action_id="pue-optimize",
                action_type="pue_optimization",
                target="global",
                parameters={"current_pue": reading.pue, "target_pue": self._pue_target},
                result="PUE above target; investigating cooling efficiency",
            )
            actions.append(action)

        # Power budget enforcement
        draw_pct = (reading.total_draw_kw / self._power_capacity_kw) * 100.0
        if draw_pct > MAX_POWER_DRAW_PCT:
            new_limit = self._power_capacity_kw * 0.85
            success = self._actuators.set_power_limit(new_limit)
            action = OpsAction(
                action_id="power-limit",
                action_type="enforce_power_limit",
                target="global",
                parameters={"limit_kw": new_limit, "draw_pct": draw_pct},
                success=success,
                result=f"Power limited to {new_limit}kW ({draw_pct:.1f}% draw)",
            )
            actions.append(action)
            self._emit_metric("dcops.power.limit_enforced", new_limit, {})

        return actions

    def _monitor_capacity(self) -> list[OpsAction]:
        """Monitor capacity and generate alerts."""
        actions: list[OpsAction] = []
        # Capacity monitoring is delegated to CapacityPlanner
        return actions

    def _emit_metric(self, name: str, value: float, tags: dict[str, str]) -> None:
        """Emit a metric if a sink is configured."""
        if self._metrics is not None:
            try:
                self._metrics.emit(name, value, tags)
            except Exception:
                logger.exception("Failed to emit metric %s", name)


# ---------------------------------------------------------------------------
# Capacity Planner
# ---------------------------------------------------------------------------


class CapacityPlanner:
    """Plans and forecasts data center capacity.

    Tracks utilization trends, forecasts exhaustion, and generates
    capacity alerts with configurable thresholds.

    Parameters
    ----------
    power_capacity_kw:
        Total power capacity.
    cooling_capacity_tons:
        Total cooling capacity.
    compute_total_units:
        Total compute units (e.g., server slots).
    storage_total_tb:
        Total storage capacity in terabytes.
    network_total_gbps:
        Total network bandwidth in gigabits per second.
    buffer_pct:
        Target headroom percentage before alerting.
    """

    def __init__(
        self,
        *,
        power_capacity_kw: float = DEFAULT_POWER_CAPACITY_KW,
        cooling_capacity_tons: float = DEFAULT_COOLING_CAPACITY_TONS,
        compute_total_units: int = 1000,
        storage_total_tb: float = 500.0,
        network_total_gbps: float = 400.0,
        buffer_pct: float = DEFAULT_CAPACITY_BUFFER_PCT,
    ) -> None:
        if power_capacity_kw <= 0:
            raise ValueError("power_capacity_kw must be positive")
        if cooling_capacity_tons <= 0:
            raise ValueError("cooling_capacity_tons must be positive")
        if compute_total_units <= 0:
            raise ValueError("compute_total_units must be positive")
        if storage_total_tb <= 0:
            raise ValueError("storage_total_tb must be positive")
        if network_total_gbps <= 0:
            raise ValueError("network_total_gbps must be positive")
        if not 0 < buffer_pct < 100:
            raise ValueError("buffer_pct must be between 0 and 100")

        self._power_capacity_kw = power_capacity_kw
        self._cooling_capacity_tons = cooling_capacity_tons
        self._compute_total_units = compute_total_units
        self._storage_total_tb = storage_total_tb
        self._network_total_gbps = network_total_gbps
        self._buffer_pct = buffer_pct

        self._history: list[CapacitySnapshot] = []
        self._max_history = 10_000

    @property
    def buffer_pct(self) -> float:
        """Configured buffer percentage."""
        return self._buffer_pct

    def record_snapshot(self, snapshot: CapacitySnapshot) -> list[CapacityAlert]:
        """Record a capacity snapshot and evaluate thresholds.

        Parameters
        ----------
        snapshot:
            The current capacity snapshot.

        Returns
        -------
        list[CapacityAlert]
            Alerts generated from threshold evaluation.
        """
        self._history.append(snapshot)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        return self._evaluate_thresholds(snapshot)

    def _evaluate_thresholds(self, snap: CapacitySnapshot) -> list[CapacityAlert]:
        """Evaluate all capacity thresholds against a snapshot."""
        alerts: list[CapacityAlert] = []
        threshold = 100.0 - self._buffer_pct

        checks = [
            ("compute", snap.compute_utilization_pct, threshold),
            ("storage", snap.storage_utilization_pct, threshold),
            ("network", snap.network_utilization_pct, threshold),
            ("power", snap.power_utilization_pct, threshold),
            ("cooling", snap.cooling_utilization_pct, threshold),
        ]

        for resource, util, thresh in checks:
            if util >= 100.0:
                alerts.append(
                    CapacityAlert(
                        level=CapacityAlertLevel.EMERGENCY,
                        resource_type=resource,
                        current_value=util,
                        threshold_value=thresh,
                        message=f"{resource} capacity exhausted ({util:.1f}%)",
                    )
                )
            elif util >= threshold + 10:
                alerts.append(
                    CapacityAlert(
                        level=CapacityAlertLevel.CRITICAL,
                        resource_type=resource,
                        current_value=util,
                        threshold_value=thresh,
                        message=f"{resource} critically high ({util:.1f}%)",
                    )
                )
            elif util >= threshold:
                alerts.append(
                    CapacityAlert(
                        level=CapacityAlertLevel.WARNING,
                        resource_type=resource,
                        current_value=util,
                        threshold_value=thresh,
                        message=f"{resource} above threshold ({util:.1f}%)",
                    )
                )

        return alerts

    def forecast_exhaustion(
        self, resource_type: str, *, days_ahead: int = 30
    ) -> datetime | None:
        """Forecast when a resource will be exhausted.

        Uses linear regression on historical utilization data.

        Parameters
        ----------
        resource_type:
            One of: compute, storage, network, power, cooling.
        days_ahead:
            Maximum days to look ahead.

        Returns
        -------
        datetime | None
            Predicted exhaustion date, or None if not forecastable.
        """
        if len(self._history) < 2:
            return None

        # Extract utilization series
        getter = self._resource_getter(resource_type)
        if getter is None:
            raise ValueError(f"Unknown resource_type: {resource_type}")

        values = [getter(s) for s in self._history]
        n = len(values)

        # Simple linear regression: y = mx + b
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return None

        slope = numerator / denominator
        if slope <= 0:
            return None  # Not trending toward exhaustion

        # Find when utilization hits 100%
        current = values[-1]
        remaining = 100.0 - current
        periods_to_exhaust = remaining / slope

        if periods_to_exhaust > days_ahead:
            return None

        # Assume snapshots are roughly evenly spaced; use last timestamp
        last_ts = self._history[-1].timestamp
        return last_ts + timedelta(days=periods_to_exhaust)

    def _resource_getter(
        self, resource_type: str
    ) -> Callable[[CapacitySnapshot], float] | None:
        """Get the utilization getter for a resource type."""
        mapping: dict[str, Callable[[CapacitySnapshot], float]] = {
            "compute": lambda s: s.compute_utilization_pct,
            "storage": lambda s: s.storage_utilization_pct,
            "network": lambda s: s.network_utilization_pct,
            "power": lambda s: s.power_utilization_pct,
            "cooling": lambda s: s.cooling_utilization_pct,
        }
        return mapping.get(resource_type)

    def recommend_scaling(self) -> dict[str, Any]:
        """Generate scaling recommendations based on current utilization.

        Returns
        -------
        dict[str, Any]
            Scaling recommendations per resource type.
        """
        if not self._history:
            return {"status": "insufficient_data"}

        latest = self._history[-1]
        recommendations: dict[str, Any] = {}

        resources = {
            "compute": latest.compute_utilization_pct,
            "storage": latest.storage_utilization_pct,
            "network": latest.network_utilization_pct,
            "power": latest.power_utilization_pct,
            "cooling": latest.cooling_utilization_pct,
        }

        for resource, util in resources.items():
            if util >= 100.0:
                recommendations[resource] = {
                    "action": "emergency_scale",
                    "urgency": "immediate",
                    "detail": f"{resource} exhausted; immediate scaling required",
                }
            elif util >= 100.0 - self._buffer_pct:
                forecast = self.forecast_exhaustion(resource)
                recommendations[resource] = {
                    "action": "plan_scale",
                    "urgency": "high",
                    "detail": f"{resource} at {util:.1f}%; exhaustion forecast: {forecast}",
                }
            elif util >= 100.0 - self._buffer_pct * 2:
                recommendations[resource] = {
                    "action": "monitor",
                    "urgency": "medium",
                    "detail": f"{resource} at {util:.1f}%; monitor closely",
                }
            else:
                recommendations[resource] = {
                    "action": "none",
                    "urgency": "low",
                    "detail": f"{resource} at {util:.1f}%; healthy",
                }

        return recommendations


# ---------------------------------------------------------------------------
# Workload Scheduler
# ---------------------------------------------------------------------------


class WorkloadScheduler:
    """Schedules workloads across data center zones.

    Considers thermal conditions, power budgets, network latency, and
    workload priority when making placement decisions.

    Parameters
    ----------
    zones:
        Available thermal zones for placement.
    power_budget_kw:
        Maximum power budget for new placements.
    """

    def __init__(
        self,
        zones: list[ThermalZone],
        *,
        power_budget_kw: float = DEFAULT_POWER_CAPACITY_KW,
    ) -> None:
        if not zones:
            raise ValueError("At least one zone is required")
        if power_budget_kw <= 0:
            raise ValueError("power_budget_kw must be positive")

        self._zones = {z.zone_id: z for z in zones}
        self._power_budget_kw = power_budget_kw
        self._placements: dict[str, WorkloadPlacement] = {}
        self._zone_power: dict[str, float] = {zid: 0.0 for zid in self._zones}

    @property
    def zones(self) -> list[ThermalZone]:
        """Current zone configurations."""
        return list(self._zones.values())

    @property
    def placements(self) -> dict[str, WorkloadPlacement]:
        """Current workload placements keyed by workload_id."""
        return dict(self._placements)

    def update_zones(self, zones: list[ThermalZone]) -> None:
        """Update the zone configuration."""
        if not zones:
            raise ValueError("At least one zone is required")
        self._zones = {z.zone_id: z for z in zones}

    def place_workload(self, workload: Workload) -> WorkloadPlacement:
        """Determine optimal placement for a workload.

        Parameters
        ----------
        workload:
            The workload to place.

        Returns
        -------
        WorkloadPlacement
            The placement decision.

        Raises
        ------
        CapacityExceededError:
            If no zone can accommodate the workload.
        """
        candidates = self._rank_zones(workload)

        if not candidates:
            raise CapacityExceededError(
                f"No zone can accommodate workload {workload.workload_id} "
                f"(needs {workload.compute_units} compute units)"
            )

        best_zone_id, confidence, reason = candidates[0]
        zone = self._zones[best_zone_id]

        # Estimate latency based on zone thermal state and workload priority
        estimated_latency = self._estimate_latency(workload, zone)

        placement = WorkloadPlacement(
            workload=workload,
            target_zone=best_zone_id,
            estimated_latency_ms=estimated_latency,
            confidence=confidence,
            reason=reason,
        )

        self._placements[workload.workload_id] = placement
        self._zone_power[best_zone_id] += workload.compute_units * 0.3  # rough kW estimate

        return placement

    def _rank_zones(
        self, workload: Workload
    ) -> list[tuple[str, float, str]]:
        """Rank zones by suitability for a workload.

        Returns list of (zone_id, confidence, reason) tuples sorted by
        descending suitability.
        """
        scored: list[tuple[float, str, float, str]] = []

        for zone_id, zone in self._zones.items():
            score = 0.0
            reasons: list[str] = []

            # Thermal headroom (0–40 points)
            if zone.current_temp_c < zone.setpoint_temp_c:
                score += 40.0
                reasons.append("good_thermal_headroom")
            elif zone.current_temp_c < CRITICAL_TEMP_THRESHOLD_C:
                score += 20.0
                reasons.append("acceptable_thermal")
            else:
                score -= 50.0
                reasons.append("critical_thermal")

            # Preferred zone bonus (0–20 points)
            if workload.preferred_zone == zone_id:
                score += 20.0
                reasons.append("preferred_zone")

            # Power headroom (0–20 points)
            zone_power = self._zone_power.get(zone_id, 0.0)
            if zone_power < self._power_budget_kw * 0.5:
                score += 20.0
                reasons.append("good_power_headroom")
            elif zone_power < self._power_budget_kw * 0.8:
                score += 10.0
                reasons.append("moderate_power_headroom")
            else:
                score -= 30.0
                reasons.append("low_power_headroom")

            # Priority affinity (0–20 points)
            if workload.priority == WorkloadPriority.CRITICAL:
                # Critical workloads prefer cooler zones
                if zone.current_temp_c < DEFAULT_TEMP_SETPOINT_C:
                    score += 20.0
                    reasons.append("critical_cool_zone")
                else:
                    score += 5.0
            else:
                score += 10.0
                reasons.append("standard_placement")

            confidence = min(max(score / 100.0, 0.0), 1.0)
            scored.append((score, zone_id, confidence, ", ".join(reasons)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [(zid, conf, reason) for _, zid, conf, reason in scored]

    def _estimate_latency(self, workload: Workload, zone: ThermalZone) -> float:
        """Estimate latency for a workload in a given zone."""
        base_latency = 1.0  # ms
        thermal_penalty = max(0.0, zone.temp_delta) * 0.5
        priority_factor = 1.0 / workload.priority.value
        return base_latency + thermal_penalty + priority_factor

    def remove_workload(self, workload_id: str) -> WorkloadPlacement | None:
        """Remove a workload placement.

        Returns
        -------
        WorkloadPlacement | None
            The removed placement, or None if not found.
        """
        return self._placements.pop(workload_id, None)

    def get_zone_utilization(self) -> dict[str, dict[str, float]]:
        """Get utilization metrics per zone."""
        result: dict[str, dict[str, float]] = {}
        for zone_id, zone in self._zones.items():
            power = self._zone_power.get(zone_id, 0.0)
            result[zone_id] = {
                "temperature_c": zone.current_temp_c,
                "temp_delta": zone.temp_delta,
                "power_kw": power,
                "power_headroom_pct": (
                    (self._power_budget_kw - power) / self._power_capacity_kw * 100.0
                    if self._power_capacity_kw > 0
                    else 0.0
                ),
                "workload_count": sum(
                    1 for p in self._placements.values() if p.target_zone == zone_id
                ),
            }
        return result


# ---------------------------------------------------------------------------
# Cooling Optimizer
# ---------------------------------------------------------------------------


class CoolingOptimizer:
    """Optimizes cooling system operation for efficiency.

    Selects between mechanical cooling, free cooling, and hybrid modes
    based on ambient conditions and thermal load.

    Parameters
    ----------
    cooling_capacity_tons:
        Total cooling capacity in refrigeration tons.
    pue_target:
        Target PUE for optimization.
    free_cooling_threshold_c:
        Ambient temperature below which free cooling is viable.
    """

    def __init__(
        self,
        *,
        cooling_capacity_tons: float = DEFAULT_COOLING_CAPACITY_TONS,
        pue_target: float = DEFAULT_PUE_TARGET,
        free_cooling_threshold_c: float = 18.0,
    ) -> None:
        if cooling_capacity_tons <= 0:
            raise ValueError("cooling_capacity_tons must be positive")
        if pue_target < 1.0:
            raise ValueError("pue_target must be >= 1.0")
        if free_cooling_threshold_c < -50 or free_cooling_threshold_c > 50:
            raise ValueError("free_cooling_threshold_c out of reasonable range")

        self._cooling_capacity_tons = cooling_capacity_tons
        self._pue_target = pue_target
        self._free_cooling_threshold_c = free_cooling_threshold_c
        self._current_mode = CoolingMode.MECHANICAL
        mode_history: list[tuple[datetime, CoolingMode]] = []

    @property
    def current_mode(self) -> CoolingMode:
        """Current cooling mode."""
        return self._current_mode

    def select_mode(
        self,
        ambient_temp_c: float,
        internal_temp_c: float,
        cooling_load_tons: float,
    ) -> CoolingMode:
        """Select optimal cooling mode based on conditions.

        Parameters
        ----------
        ambient_temp_c:
            Outside air temperature.
        internal_temp_c:
            Current internal data center temperature.
        cooling_load_tons:
            Current cooling load in refrigeration tons.

        Returns
        -------
        CoolingMode
            The selected cooling mode.
        """
        load_ratio = cooling_load_tons / self._cooling_capacity_tons

        if internal_temp_c >= CRITICAL_TEMP_THRESHOLD_C:
            new_mode = CoolingMode.EMERGENCY
        elif ambient_temp_c <= self._free_cooling_threshold_c and load_ratio < 0.7:
            new_mode = CoolingMode.FREE_COOLING
        elif ambient_temp_c <= internal_temp_c - 5.0 and load_ratio < 0.8:
            new_mode = CoolingMode.ECONOMIZER
        elif ambient_temp_c <= internal_temp_c - 3.0:
            new_mode = CoolingMode.HYBRID
        else:
            new_mode = CoolingMode.MECHANICAL

        if new_mode != self._current_mode:
            logger.info(
                "Cooling mode transition: %s -> %s (ambient=%.1fC, internal=%.1fC, load=%.1f tons)",
                self._current_mode.name,
                new_mode.name,
                ambient_temp_c,
                internal_temp_c,
                cooling_load_tons,
            )
            self._current_mode = new_mode

        return new_mode

    def estimate_cooling_power(
        self, cooling_load_tons: float, mode: CoolingMode | None = None
    ) -> float:
        """Estimate power consumption for cooling in kW.

        Parameters
        ----------
        cooling_load_tons:
            Cooling load in refrigeration tons.
        mode:
            Cooling mode (defaults to current mode).

        Returns
        -------
        float
            Estimated cooling power draw in kW.
        """
        if cooling_load_tons < 0:
            raise ValueError("cooling_load_tons must be non-negative")

        m = mode or self._current_mode

        # COP (Coefficient of Performance) estimates by mode
        cop_map = {
            CoolingMode.MECHANICAL: 3.0,
            CoolingMode.FREE_COOLING: 15.0,
            CoolingMode.HYBRID: 8.0,
            CoolingMode.ECONOMIZER: 12.0,
            CoolingMode.EMERGENCY: 2.0,
        }
        cop = cop_map.get(m, 3.0)

        # 1 ton of refrigeration = 3.51685 kW
        return (cooling_load_tons * 3.51685) / cop

    def calculate_pue(
        self, it_load_kw: float, cooling_load_tons: float, mode: CoolingMode | None = None
    ) -> float:
        """Calculate Power Usage Effectiveness.

        Parameters
        ----------
        it_load_kw:
            IT equipment power draw in kW.
        cooling_load_tons:
            Cooling load in refrigeration tons.
        mode:
            Cooling mode (defaults to current mode).

        Returns
        -------
        float
            Calculated PUE.
        """
        if it_load_kw <= 0:
            raise ValueError("it_load_kw must be positive")

        cooling_power = self.estimate_cooling_power(cooling_load_tons, mode)
        # Add estimated overhead (lighting, misc) at 5% of IT load
        overhead = it_load_kw * 0.05
        total = it_load_kw + cooling_power + overhead
        return total / it_load_kw


# ---------------------------------------------------------------------------
# Power Manager
# ---------------------------------------------------------------------------


class PowerManager:
    """Manages power sources, UPS, generators, and renewable integration.

    Handles power source switching, load shedding, and renewable
    energy optimization.

    Parameters
    ----------
    grid_capacity_kw:
        Grid power capacity.
    ups_capacity_kw:
        UPS power capacity.
    generator_capacity_kw:
        Generator power capacity.
    renewable_capacity_kw:
        Renewable energy capacity.
    """

    def __init__(
        self,
        *,
        grid_capacity_kw: float = DEFAULT_POWER_CAPACITY_KW,
        ups_capacity_kw: float = 2_000.0,
        generator_capacity_kw: float = 5_000.0,
        renewable_capacity_kw: float = 1_000.0,
    ) -> None:
        if grid_capacity_kw <= 0:
            raise ValueError("grid_capacity_kw must be positive")
        if ups_capacity_kw < 0:
            raise ValueError("ups_capacity_kw must be non-negative")
        if generator_capacity_kw < 0:
            raise ValueError("generator_capacity_kw must be non-negative")
        if renewable_capacity_kw < 0:
            raise ValueError("renewable_capacity_kw must be non-negative")

        self._grid_capacity_kw = grid_capacity_kw
        self._ups_capacity_kw = ups_capacity_kw
        self._generator_capacity_kw = generator_capacity_kw
        self._renewable_capacity_kw = renewable_capacity_kw

        self._current_source = PowerSource.GRID
        self._ups_charge_pct = 100.0
        self._generator_running = False
        self._renewable_active = True
        self._load_shedding_active = False
        self._sheddable_loads: set[str] = set()

    @property
    def current_source(self) -> PowerSource:
        """Current active power source."""
        return self._current_source

    @property
    def ups_charge_pct(self) -> float:
        """Current UPS charge percentage."""
        return self._ups_charge_pct

    @property
    def generator_running(self) -> bool:
        """Whether the generator is running."""
        return self._generator_running

    @property
    def renewable_active(self) -> bool:
        """Whether renewable energy is active."""
        return self._renewable_active

    @property
    def available_capacity_kw(self) -> float:
        """Total available power capacity from all sources."""
        total = 0.0
        if self._current_source == PowerSource.GRID:
            total += self._grid_capacity_kw
        if self._ups_charge_pct > 10.0:
            total += self._ups_capacity_kw * (self._ups_charge_pct / 100.0)
        if self._generator_running:
            total += self._generator_capacity_kw
        if self._renewable_active:
            total += self._renewable_capacity_kw
        return total

    def switch_source(self, source: PowerSource) -> bool:
        """Switch the primary power source.

        Parameters
        ----------
        source:
            Target power source.

        Returns
        -------
        bool
            True if the switch was successful.
        """
        if source == self._current_source:
            return True

        if source == PowerSource.UPS and self._ups_charge_pct < 10.0:
            logger.warning("Cannot switch to UPS: charge too low (%.1f%%)", self._ups_charge_pct)
            return False

        if source == PowerSource.GENERATOR and not self._generator_running:
            self.start_generator()

        old_source = self._current_source
        self._current_source = source
        logger.info("Power source switched: %s -> %s", old_source.name, source.name)
        return True

    def start_generator(self) -> bool:
        """Start the backup generator.

        Returns
        -------
        bool
            True if the generator started successfully.
        """
        if self._generator_running:
            return True
        self._generator_running = True
        logger.info("Generator started")
        return True

    def stop_generator(self) -> bool:
        """Stop the backup generator.

        Returns
        -------
        bool
            True if the generator stopped successfully.
        """
        if not self._generator_running:
            return True
        self._generator_running = False
        logger.info("Generator stopped")
        return True

    def set_ups_charge(self, pct: float) -> None:
        """Set the UPS charge level.

        Parameters
        ----------
        pct:
            Charge percentage (0–100).
        """
        if not 0.0 <= pct <= 100.0:
            raise ValueError(f"UPS charge {pct}% out of range [0, 100]")
        self._ups_charge_pct = pct

    def enable_renewable(self) -> None:
        """Enable renewable energy integration."""
        self._renewable_active = True
        logger.info("Renewable energy enabled")

    def disable_renewable(self) -> None:
        """Disable renewable energy integration."""
        self._renewable_active = False
        logger.info("Renewable energy disabled")

    def register_sheddable_load(self, load_id: str) -> None:
        """Register a load that can be shed during power emergencies."""
        self._sheddable_loads.add(load_id)

    def shed_load(self, load_id: str) -> bool:
        """Shed a specific load.

        Returns
        -------
        bool
            True if the load was shed.
        """
        if load_id in self._sheddable_loads:
            self._sheddable_loads.discard(load_id)
            logger.info("Load shed: %s", load_id)
            return True
        return False

    def calculate_renewable_offset(
        self, total_demand_kw: float, renewable_output_kw: float
    ) -> float:
        """Calculate the renewable energy offset.

        Parameters
        ----------
        total_demand_kw:
            Total power demand.
        renewable_output_kw:
            Current renewable energy output.

        Returns
        -------
        float
            Offset ratio (0.0–1.0) of demand covered by renewables.
        """
        if total_demand_kw <= 0:
            return 0.0
        return min(renewable_output_kw / total_demand_kw, 1.0)

    def get_power_summary(self) -> dict[str, Any]:
        """Get a comprehensive power summary.

        Returns
        -------
        dict[str, Any]
            Power system status summary.
        """
        return {
            "current_source": self._current_source.name,
            "grid_capacity_kw": self._grid_capacity_kw,
            "ups_capacity_kw": self._ups_capacity_kw,
            "ups_charge_pct": self._ups_charge_pct,
            "generator_running": self._generator_running,
            "generator_capacity_kw": self._generator_capacity_kw,
            "renewable_active": self._renewable_active,
            "renewable_capacity_kw": self._renewable_capacity_kw,
            "available_capacity_kw": self.available_capacity_kw,
            "load_shedding_active": self._load_shedding_active,
            "sheddable_loads": list(self._sheddable_loads),
        }


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    "AutonomousDCOps",
    "CapacityAlert",
    "CapacityAlertLevel",
    "CapacityPlanner",
    "CapacitySnapshot",
    "CoolingMode",
    "CoolingOptimizer",
    "DCOpsError",
    "DCOpsState",
    "InvalidStateTransitionError",
    "OpsAction",
    "PowerBudgetExceededError",
    "PowerManager",
    "PowerReading",
    "PowerSource",
    "SensorInterface",
    "SensorReading",
    "SensorReadError",
    "ActuatorInterface",
    "MetricsSink",
    "ThermalRunawayError",
    "ThermalZone",
    "Workload",
    "WorkloadPlacement",
    "WorkloadPriority",
    "WorkloadScheduler",
    "CapacityExceededError",
]
