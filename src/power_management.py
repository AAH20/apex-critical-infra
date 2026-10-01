"""Apex Critical Infrastructure — Power Management.

UPS management, generator control, renewable integration, and PUE
optimization for data center power systems.

Integrates with Data Center Commander (DC lifecycle), Apex_ULL (ultra-low
latency), ApexGraphSwarm (multi-agent orchestration), and GRC_Claw
(governance).
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Protocol,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Power constants
DEFAULT_PUE_TARGET = 1.4
DEFAULT_PUE_WARNING = 1.6
DEFAULT_PUE_CRITICAL = 2.0
DEFAULT_UPS_CAPACITY_KW = 2_000.0
DEFAULT_UPS_RUNTIME_MIN = 15.0
DEFAULT_GENERATOR_CAPACITY_KW = 5_000.0
DEFAULT_GENERATOR_FUEL_CAPACITY_L = 10_000.0
DEFAULT_GENERATOR_FUEL_CONSUMPTION_L_H = 500.0
DEFAULT_RENEWABLE_CAPACITY_KW = 1_000.0
DEFAULT_BATTERY_CAPACITY_KWH = 500.0
DEFAULT_BATTERY_EFFICIENCY = 0.95
DEFAULT_POWER_FACTOR = 0.95
DEFAULT_VOLTAGE_V = 480.0
DEFAULT_FREQUENCY_HZ = 60.0

# Thresholds
DEFAULT_LOW_BATTERY_THRESHOLD_PCT = 20.0
DEFAULT_CRITICAL_BATTERY_THRESHOLD_PCT = 10.0
DEFAULT_LOW_FUEL_THRESHOLD_PCT = 25.0
DEFAULT_HIGH_PUE_THRESHOLD = 1.8
DEFAULT_POWER_BALANCE_TOLERANCE_PCT = 5.0

# Time constants
DEFAULT_RAMP_TIME_S = 30.0
DEFAULT_TRANSFER_TIME_S = 10.0
DEFAULT_STABILIZATION_TIME_S = 5.0


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PowerSourceState(Enum):
    """State of a power source."""

    OFFLINE = auto()
    STANDBY = auto()
    STARTING = auto()
    RUNNING = auto()
    DEGRADED = auto()
    FAULT = auto()
    MAINTENANCE = auto()


class UPSState(Enum):
    """State of a UPS system."""

    OFFLINE = auto()
    ONLINE = auto()
    ON_BATTERY = auto()
    BYPASS = auto()
    FAULT = auto()
    MAINTENANCE = auto()
    SELF_TEST = auto()


class GeneratorState(Enum):
    """State of a generator."""

    OFFLINE = auto()
    STANDBY = auto()
    CRANKING = auto()
    WARMING_UP = auto()
    RUNNING = auto()
    COOLING_DOWN = auto()
    FAULT = auto()
    MAINTENANCE = auto()
    FUEL_LOW = auto()


class RenewableSourceType(Enum):
    """Type of renewable energy source."""

    SOLAR = "solar"
    WIND = "wind"
    HYDRO = "hydro"
    GEOTHERMAL = "geothermal"
    FUEL_CELL = "fuel_cell"


class PowerQuality(Enum):
    """Power quality classification."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class LoadPriority(Enum):
    """Priority level for load shedding."""

    CRITICAL = 1
    ESSENTIAL = 2
    IMPORTANT = 3
    NORMAL = 4
    SHEDDABLE = 5


class PUEAlertLevel(Enum):
    """PUE alert severity."""

    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()
    EMERGENCY = auto()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PowerManagementError(Exception):
    """Base exception for power management."""


class UPSFaultError(PowerManagementError):
    """Raised when a UPS fault is detected."""


class GeneratorFaultError(PowerManagementError):
    """Raised when a generator fault is detected."""


class TransferSwitchError(PowerManagementError):
    """Raised when an ATS transfer fails."""


class PowerBudgetExceededError(PowerManagementError):
    """Raised when power draw exceeds the configured budget."""


class InsufficientCapacityError(PowerManagementError):
    """Raised when available capacity is insufficient."""


class RenewableIntegrationError(PowerManagementError):
    """Raised when renewable integration fails."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ElectricalMeasurement:
    """A single electrical measurement."""

    voltage_v: float
    current_a: float
    frequency_hz: float
    power_factor: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if self.voltage_v < 0:
            raise ValueError("voltage_v must be non-negative")
        if self.current_a < 0:
            raise ValueError("current_a must be non-negative")
        if self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        if not 0.0 < self.power_factor <= 1.0:
            raise ValueError("power_factor must be in range (0, 1]")

    @property
    def apparent_power_kva(self) -> float:
        """Apparent power in kVA."""
        return (self.voltage_v * self.current_a) / 1000.0

    @property
    def real_power_kw(self) -> float:
        """Real power in kW."""
        return self.apparent_power_kva * self.power_factor

    @property
    def reactive_power_kvar(self) -> float:
        """Reactive power in kVAR."""
        return self.apparent_power_kva * math.sqrt(1.0 - self.power_factor**2)


@dataclass(frozen=True, slots=True)
class UPSStatus:
    """Status of a UPS system."""

    ups_id: str
    state: UPSState
    charge_pct: float
    load_pct: float
    input_voltage_v: float
    output_voltage_v: float
    frequency_hz: float
    runtime_remaining_min: float
    temperature_c: float
    battery_health_pct: float
    last_self_test: datetime | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if not self.ups_id:
            raise ValueError("ups_id must be non-empty")
        if not 0.0 <= self.charge_pct <= 100.0:
            raise ValueError(f"charge_pct {self.charge_pct} out of range [0, 100]")
        if not 0.0 <= self.load_pct <= 100.0:
            raise ValueError(f"load_pct {self.load_pct} out of range [0, 100]")
        if not 0.0 <= self.battery_health_pct <= 100.0:
            raise ValueError(
                f"battery_health_pct {self.battery_health_pct} out of range [0, 100]"
            )

    @property
    def is_healthy(self) -> bool:
        """Whether the UPS is in a healthy state."""
        return (
            self.state in (UPSState.ONLINE, UPSState.ON_BATTERY)
            and self.charge_pct > 10.0
        )

    @property
    def is_on_battery(self) -> bool:
        """Whether the UPS is running on battery power."""
        return self.state == UPSState.ON_BATTERY


@dataclass(frozen=True, slots=True)
class GeneratorStatus:
    """Status of a generator."""

    generator_id: str
    state: GeneratorState
    output_kw: float
    fuel_level_pct: float
    fuel_remaining_l: float
    runtime_hours: float
    temperature_c: float
    oil_pressure_kpa: float
    last_maintenance: datetime | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if not self.generator_id:
            raise ValueError("generator_id must be non-empty")
        if self.output_kw < 0:
            raise ValueError("output_kw must be non-negative")
        if not 0.0 <= self.fuel_level_pct <= 100.0:
            raise ValueError(
                f"fuel_level_pct {self.fuel_level_pct} out of range [0, 100]"
            )
        if self.fuel_remaining_l < 0:
            raise ValueError("fuel_remaining_l must be non-negative")
        if self.runtime_hours < 0:
            raise ValueError("runtime_hours must be non-negative")

    @property
    def is_available(self) -> bool:
        """Whether the generator is available for use."""
        return self.state in (GeneratorState.STANDBY, GeneratorState.RUNNING)

    @property
    def estimated_runtime_hours(self) -> float:
        """Estimated remaining runtime based on fuel."""
        if self.fuel_remaining_l <= 0:
            return 0.0
        return self.fuel_remaining_l / DEFAULT_GENERATOR_FUEL_CONSUMPTION_L_H


@dataclass(frozen=True, slots=True)
class RenewableSource:
    """A renewable energy source."""

    source_id: str
    source_type: RenewableSourceType
    capacity_kw: float
    current_output_kw: float
    availability_pct: float
    location: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must be non-empty")
        if self.capacity_kw <= 0:
            raise ValueError("capacity_kw must be positive")
        if self.current_output_kw < 0:
            raise ValueError("current_output_kw must be non-negative")
        if self.current_output_kw > self.capacity_kw:
            raise ValueError("current_output_kw cannot exceed capacity_kw")
        if not 0.0 <= self.availability_pct <= 100.0:
            raise ValueError("availability_pct must be in range [0, 100]")

    @property
    def utilization_pct(self) -> float:
        """Current utilization percentage."""
        if self.capacity_kw == 0:
            return 0.0
        return (self.current_output_kw / self.capacity_kw) * 100.0


@dataclass(frozen=True, slots=True)
class PUEStatus:
    """Power Usage Effectiveness status."""

    pue_value: float
    it_load_kw: float
    total_facility_load_kw: float
    cooling_load_kw: float
    overhead_load_kw: float
    pue_target: float
    alert_level: PUEAlertLevel
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if self.pue_value < 1.0:
            raise ValueError(f"PUE {self.pue_value} is below theoretical minimum of 1.0")
        if self.it_load_kw <= 0:
            raise ValueError("it_load_kw must be positive")
        if self.total_facility_load_kw < self.it_load_kw:
            raise ValueError("total_facility_load_kw must be >= it_load_kw")

    @property
    def pue_delta(self) -> float:
        """Deviation from target PUE."""
        return self.pue_value - self.pue_target

    @property
    def efficiency_pct(self) -> float:
        """Efficiency as a percentage (inverse of PUE)."""
        return (1.0 / self.pue_value) * 100.0


@dataclass(frozen=True, slots=True)
class PowerBudget:
    """Power budget configuration."""

    total_capacity_kw: float
    it_load_budget_kw: float
    cooling_budget_kw: float
    overhead_budget_kw: float
    pue_target: float = DEFAULT_PUE_TARGET
    buffer_pct: float = 10.0

    def __post_init__(self) -> None:
        if self.total_capacity_kw <= 0:
            raise ValueError("total_capacity_kw must be positive")
        if self.it_load_budget_kw < 0:
            raise ValueError("it_load_budget_kw must be non-negative")
        if self.cooling_budget_kw < 0:
            raise ValueError("cooling_budget_kw must be non-negative")
        if self.overhead_budget_kw < 0:
            raise ValueError("overhead_budget_kw must be non-negative")
        if self.pue_target < 1.0:
            raise ValueError("pue_target must be >= 1.0")
        if not 0.0 <= self.buffer_pct <= 50.0:
            raise ValueError("buffer_pct must be in range [0, 50]")

    @property
    def effective_capacity_kw(self) -> float:
        """Effective capacity after buffer."""
        return self.total_capacity_kw * (1.0 - self.buffer_pct / 100.0)

    @property
    def total_budget_kw(self) -> float:
        """Sum of all budget components."""
        return self.it_load_budget_kw + self.cooling_budget_kw + self.overhead_budget_kw


@dataclass(frozen=True, slots=True)
class LoadShedEvent:
    """A load shedding event."""

    event_id: str
    load_id: str
    priority: LoadPriority
    shed_kw: float
    reason: str
    shed_at: datetime = field(default_factory=datetime.utcnow)
    restored_at: datetime | None = None

    @property
    def is_active(self) -> bool:
        """Whether the load is still shed."""
        return self.restored_at is None

    @property
    def duration_seconds(self) -> float:
        """Duration of the shed event in seconds."""
        end = self.restored_at or datetime.utcnow()
        return (end - self.shed_at).total_seconds()


@dataclass(frozen=True, slots=True)
class PowerAlert:
    """A power system alert."""

    level: PUEAlertLevel
    source: str
    message: str
    current_value: float
    threshold_value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class PowerMeter(Protocol):
    """Protocol for power metering."""

    def read(self) -> ElectricalMeasurement:
        """Read current electrical measurements."""
        ...


class UPSController(Protocol):
    """Protocol for UPS control."""

    def get_status(self) -> UPSStatus:
        """Get current UPS status."""
        ...

    def start_self_test(self) -> bool:
        """Initiate a self-test."""
        ...

    def set_bypass(self, enabled: bool) -> bool:
        """Enable or disable bypass mode."""
        ...


class GeneratorController(Protocol):
    """Protocol for generator control."""

    def get_status(self) -> GeneratorStatus:
        """Get current generator status."""
        ...

    def start(self) -> bool:
        """Start the generator."""
        ...

    def stop(self) -> bool:
        """Stop the generator."""
        ...

    def transfer_load(self, kw: float) -> bool:
        """Transfer load to the generator."""
        ...


# ---------------------------------------------------------------------------
# UPS Manager
# ---------------------------------------------------------------------------


class UPSManager:
    """Manages UPS systems for power continuity.

    Monitors battery health, manages charge cycles, and handles
    transitions between grid and battery power.

    Parameters
    ----------
    ups_id:
        Unique identifier for this UPS.
    capacity_kw:
        Rated power capacity in kilowatts.
    battery_capacity_kwh:
        Battery energy capacity in kilowatt-hours.
    """

    def __init__(
        self,
        *,
        ups_id: str,
        capacity_kw: float = DEFAULT_UPS_CAPACITY_KW,
        battery_capacity_kwh: float = DEFAULT_BATTERY_CAPACITY_KWH,
    ) -> None:
        if not ups_id:
            raise ValueError("ups_id must be non-empty")
        if capacity_kw <= 0:
            raise ValueError("capacity_kw must be positive")
        if battery_capacity_kwh <= 0:
            raise ValueError("battery_capacity_kwh must be positive")

        self._ups_id = ups_id
        self._capacity_kw = capacity_kw
        self._battery_capacity_kwh = battery_capacity_kwh
        self._state = UPSState.OFFLINE
        self._charge_pct = 100.0
        self._load_pct = 0.0
        self._battery_health_pct = 100.0
        self._temperature_c = 25.0
        self._last_self_test: datetime | None = None
        self._self_test_interval_days = 90
        self._charge_cycles = 0
        self._total_discharge_kwh = 0.0

    @property
    def ups_id(self) -> str:
        """UPS identifier."""
        return self._ups_id

    @property
    def state(self) -> UPSState:
        """Current UPS state."""
        return self._state

    @property
    def charge_pct(self) -> float:
        """Current battery charge percentage."""
        return self._charge_pct

    @property
    def is_healthy(self) -> bool:
        """Whether the UPS is healthy."""
        return (
            self._state in (UPSState.ONLINE, UPSState.ON_BATTERY)
            and self._charge_pct > DEFAULT_CRITICAL_BATTERY_THRESHOLD_PCT
            and self._battery_health_pct > 50.0
        )

    def get_status(self) -> UPSStatus:
        """Get current UPS status."""
        runtime = self.estimate_runtime_min()
        return UPSStatus(
            ups_id=self._ups_id,
            state=self._state,
            charge_pct=self._charge_pct,
            load_pct=self._load_pct,
            input_voltage_v=DEFAULT_VOLTAGE_V,
            output_voltage_v=DEFAULT_VOLTAGE_V,
            frequency_hz=DEFAULT_FREQUENCY_HZ,
            runtime_remaining_min=runtime,
            temperature_c=self._temperature_c,
            battery_health_pct=self._battery_health_pct,
            last_self_test=self._last_self_test,
        )

    def estimate_runtime_min(self) -> float:
        """Estimate remaining battery runtime in minutes.

        Returns
        -------
        float:
            Estimated runtime in minutes at current load.
        """
        if self._load_pct <= 0 or self._charge_pct <= 0:
            return 0.0

        load_kw = self._capacity_kw * (self._load_pct / 100.0)
        if load_kw <= 0:
            return 0.0

        available_kwh = self._battery_capacity_kwh * (self._charge_pct / 100.0)
        # Account for battery efficiency
        available_kwh *= DEFAULT_BATTERY_EFFICIENCY
        runtime_hours = available_kwh / load_kw
        return runtime_hours * 60.0

    def set_load(self, load_kw: float) -> None:
        """Set the current load on the UPS.

        Parameters
        ----------
        load_kw:
            Current load in kilowatts.
        """
        if load_kw < 0:
            raise ValueError("load_kw must be non-negative")
        self._load_pct = min(100.0, (load_kw / self._capacity_kw) * 100.0)

    def discharge(self, energy_kwh: float) -> None:
        """Record battery discharge.

        Parameters
        ----------
        energy_kwh:
            Energy discharged in kilowatt-hours.
        """
        if energy_kwh < 0:
            raise ValueError("energy_kwh must be non-negative")
        self._charge_pct = max(
            0.0, self._charge_pct - (energy_kwh / self._battery_capacity_kwh) * 100.0
        )
        self._total_discharge_kwh += energy_kwh

    def charge(self, energy_kwh: float) -> None:
        """Record battery charging.

        Parameters
        ----------
        energy_kwh:
            Energy charged in kilowatt-hours.
        """
        if energy_kwh < 0:
            raise ValueError("energy_kwh must be non-negative")
        self._charge_pct = min(
            100.0, self._charge_pct + (energy_kwh / self._battery_capacity_kwh) * 100.0
        )

    def start_self_test(self) -> bool:
        """Initiate a battery self-test.

        Returns
        -------
        bool:
            True if the self-test was initiated successfully.
        """
        if self._state == UPSState.FAULT:
            logger.error("UPS %s: cannot self-test in FAULT state", self._ups_id)
            return False
        self._state = UPSState.SELF_TEST
        self._last_self_test = datetime.utcnow()
        logger.info("UPS %s: self-test initiated", self._ups_id)
        return True

    def evaluate_self_test(self) -> dict[str, Any]:
        """Evaluate the results of the last self-test.

        Returns
        -------
        dict[str, Any]:
            Self-test results.
        """
        if self._last_self_test is None:
            return {"status": "never_tested"}

        days_since = (datetime.utcnow() - self._last_self_test).days
        passed = self._battery_health_pct > 70.0 and self._charge_pct > 50.0

        return {
            "status": "passed" if passed else "failed",
            "last_test": self._last_self_test.isoformat(),
            "days_since": days_since,
            "battery_health_pct": self._battery_health_pct,
            "charge_pct": self._charge_pct,
            "recommendation": (
                "Schedule replacement" if self._battery_health_pct < 70.0 else "OK"
            ),
        }

    def transition_to_battery(self) -> bool:
        """Transition to battery power.

        Returns
        -------
        bool:
            True if the transition was successful.
        """
        if self._charge_pct < DEFAULT_CRITICAL_BATTERY_THRESHOLD_PCT:
            logger.error(
                "UPS %s: cannot transition to battery, charge too low (%.1f%%)",
                self._ups_id,
                self._charge_pct,
            )
            return False
        self._state = UPSState.ON_BATTERY
        logger.info("UPS %s: transitioned to battery power", self._ups_id)
        return True

    def transition_to_online(self) -> bool:
        """Transition back to online (grid) power.

        Returns
        -------
        bool:
            True if the transition was successful.
        """
        self._state = UPSState.ONLINE
        logger.info("UPS %s: transitioned to online power", self._ups_id)
        return True

    def set_bypass(self, enabled: bool) -> bool:
        """Enable or disable bypass mode.

        Parameters
        ----------
        enabled:
            True to enable bypass, False to disable.

        Returns
        -------
        bool:
            True if the operation was successful.
        """
        if enabled:
            self._state = UPSState.BYPASS
            logger.info("UPS %s: bypass enabled", self._ups_id)
        else:
            self._state = UPSState.ONLINE
            logger.info("UPS %s: bypass disabled", self._ups_id)
        return True

    def get_health_report(self) -> dict[str, Any]:
        """Get a comprehensive health report.

        Returns
        -------
        dict[str, Any]:
            Health report.
        """
        return {
            "ups_id": self._ups_id,
            "state": self._state.name,
            "charge_pct": self._charge_pct,
            "battery_health_pct": self._battery_health_pct,
            "temperature_c": self._temperature_c,
            "charge_cycles": self._charge_cycles,
            "total_discharge_kwh": self._total_discharge_kwh,
            "runtime_remaining_min": self.estimate_runtime_min(),
            "self_test": self.evaluate_self_test(),
            "is_healthy": self.is_healthy,
        }


# ---------------------------------------------------------------------------
# Generator Controller
# ---------------------------------------------------------------------------


class GeneratorManager:
    """Manages backup generator systems.

    Handles automatic start/stop, load transfer, fuel monitoring,
    and maintenance scheduling.

    Parameters
    ----------
    generator_id:
        Unique identifier for this generator.
    capacity_kw:
        Rated power capacity in kilowatts.
    fuel_capacity_l:
        Fuel tank capacity in liters.
    """

    def __init__(
        self,
        *,
        generator_id: str,
        capacity_kw: float = DEFAULT_GENERATOR_CAPACITY_KW,
        fuel_capacity_l: float = DEFAULT_GENERATOR_FUEL_CAPACITY_L,
    ) -> None:
        if not generator_id:
            raise ValueError("generator_id must be non-empty")
        if capacity_kw <= 0:
            raise ValueError("capacity_kw must be positive")
        if fuel_capacity_l <= 0:
            raise ValueError("fuel_capacity_l must be positive")

        self._generator_id = generator_id
        self._capacity_kw = capacity_kw
        self._fuel_capacity_l = fuel_capacity_l
        self._state = GeneratorState.OFFLINE
        self._output_kw = 0.0
        self._fuel_level_pct = 100.0
        self._fuel_remaining_l = fuel_capacity_l
        self._runtime_hours = 0.0
        self._temperature_c = 25.0
        self._oil_pressure_kpa = 0.0
        self._last_maintenance: datetime | None = None
        self._start_time: datetime | None = None
        self._total_energy_produced_kwh = 0.0

    @property
    def generator_id(self) -> str:
        """Generator identifier."""
        return self._generator_id

    @property
    def state(self) -> GeneratorState:
        """Current generator state."""
        return self._state

    @property
    def is_available(self) -> bool:
        """Whether the generator is available."""
        return self._state in (GeneratorState.STANDBY, GeneratorState.RUNNING)

    @property
    def estimated_runtime_hours(self) -> float:
        """Estimated remaining runtime in hours."""
        if self._fuel_remaining_l <= 0:
            return 0.0
        return self._fuel_remaining_l / DEFAULT_GENERATOR_FUEL_CONSUMPTION_L_H

    def get_status(self) -> GeneratorStatus:
        """Get current generator status."""
        return GeneratorStatus(
            generator_id=self._generator_id,
            state=self._state,
            output_kw=self._output_kw,
            fuel_level_pct=self._fuel_level_pct,
            fuel_remaining_l=self._fuel_remaining_l,
            runtime_hours=self._runtime_hours,
            temperature_c=self._temperature_c,
            oil_pressure_kpa=self._oil_pressure_kpa,
            last_maintenance=self._last_maintenance,
        )

    def start(self) -> bool:
        """Start the generator.

        Returns
        -------
        bool:
            True if the generator started successfully.
        """
        if self._state == GeneratorState.FAULT:
            logger.error(
                "Generator %s: cannot start in FAULT state", self._generator_id
            )
            return False
        if self._fuel_level_pct < DEFAULT_LOW_FUEL_THRESHOLD_PCT:
            logger.error(
                "Generator %s: fuel too low (%.1f%%)",
                self._generator_id,
                self._fuel_level_pct,
            )
            return False

        self._state = GeneratorState.CRANKING
        self._start_time = datetime.utcnow()
        logger.info("Generator %s: cranking", self._generator_id)

        # Simulate cranking -> warming up -> running
        self._state = GeneratorState.WARMING_UP
        self._temperature_c = 40.0
        self._oil_pressure_kpa = 200.0
        logger.info("Generator %s: warming up", self._generator_id)

        self._state = GeneratorState.RUNNING
        self._temperature_c = 75.0
        self._oil_pressure_kpa = 400.0
        logger.info("Generator %s: running", self._generator_id)
        return True

    def stop(self) -> bool:
        """Stop the generator.

        Returns
        -------
        bool:
            True if the generator stopped successfully.
        """
        if self._state not in (GeneratorState.RUNNING, GeneratorState.WARMING_UP):
            return True

        self._state = GeneratorState.COOLING_DOWN
        self._output_kw = 0.0
        logger.info("Generator %s: cooling down", self._generator_id)

        self._state = GeneratorState.STANDBY
        self._temperature_c = 30.0
        self._oil_pressure_kpa = 0.0
        logger.info("Generator %s: standby", self._generator_id)
        return True

    def transfer_load(self, load_kw: float) -> bool:
        """Transfer electrical load to the generator.

        Parameters
        ----------
        load_kw:
            Load to transfer in kilowatts.

        Returns
        -------
        bool:
            True if the load was transferred successfully.
        """
        if self._state != GeneratorState.RUNNING:
            logger.error(
                "Generator %s: cannot transfer load, not running", self._generator_id
            )
            return False
        if load_kw > self._capacity_kw:
            logger.error(
                "Generator %s: load %.1f kW exceeds capacity %.1f kW",
                self._generator_id,
                load_kw,
                self._capacity_kw,
            )
            return False

        self._output_kw = load_kw
        logger.info(
            "Generator %s: load transferred (%.1f kW)", self._generator_id, load_kw
        )
        return True

    def update_fuel_consumption(self, hours: float) -> None:
        """Update fuel consumption based on runtime.

        Parameters
        ----------
        hours:
            Hours of operation to account for.
        """
        if hours < 0:
            raise ValueError("hours must be non-negative")
        consumption = DEFAULT_GENERATOR_FUEL_CONSUMPTION_L_H * hours
        self._fuel_remaining_l = max(0.0, self._fuel_remaining_l - consumption)
        self._fuel_level_pct = (self._fuel_remaining_l / self._fuel_capacity_l) * 100.0
        self._runtime_hours += hours
        self._total_energy_produced_kwh += self._output_kw * hours

        if self._fuel_level_pct < DEFAULT_LOW_FUEL_THRESHOLD_PCT:
            self._state = GeneratorState.FUEL_LOW
            logger.warning(
                "Generator %s: fuel low (%.1f%%)", self._generator_id, self._fuel_level_pct
            )

    def refuel(self, amount_l: float) -> None:
        """Refuel the generator.

        Parameters
        ----------
        amount_l:
            Amount of fuel to add in liters.
        """
        if amount_l < 0:
            raise ValueError("amount_l must be non-negative")
        self._fuel_remaining_l = min(
            self._fuel_capacity_l, self._fuel_remaining_l + amount_l
        )
        self._fuel_level_pct = (self._fuel_remaining_l / self._fuel_capacity_l) * 100.0
        if self._state == GeneratorState.FUEL_LOW and self._fuel_level_pct > DEFAULT_LOW_FUEL_THRESHOLD_PCT:
            self._state = GeneratorState.RUNNING if self._output_kw > 0 else GeneratorState.STANDBY
        logger.info(
            "Generator %s: refueled to %.1f%%", self._generator_id, self._fuel_level_pct
        )

    def get_health_report(self) -> dict[str, Any]:
        """Get a comprehensive health report.

        Returns
        -------
        dict[str, Any]:
            Health report.
        """
        return {
            "generator_id": self._generator_id,
            "state": self._state.name,
            "output_kw": self._output_kw,
            "capacity_kw": self._capacity_kw,
            "fuel_level_pct": self._fuel_level_pct,
            "fuel_remaining_l": self._fuel_remaining_l,
            "estimated_runtime_hours": self.estimated_runtime_hours,
            "runtime_hours": self._runtime_hours,
            "total_energy_produced_kwh": self._total_energy_produced_kwh,
            "temperature_c": self._temperature_c,
            "oil_pressure_kpa": self._oil_pressure_kpa,
            "is_available": self.is_available,
        }


# ---------------------------------------------------------------------------
# Renewable Energy Integrator
# ---------------------------------------------------------------------------


class RenewableEnergyIntegrator:
    """Integrates renewable energy sources into the power mix.

    Manages solar, wind, and other renewable sources to reduce
    grid dependence and carbon footprint.

    Parameters
    ----------
    sources:
        List of renewable energy sources to manage.
    """

    def __init__(self, sources: list[RenewableSource] | None = None) -> None:
        self._sources: dict[str, RenewableSource] = {}
        if sources:
            for source in sources:
                self.add_source(source)

    @property
    def total_capacity_kw(self) -> float:
        """Total renewable capacity in kW."""
        return sum(s.capacity_kw for s in self._sources.values())

    @property
    def total_output_kw(self) -> float:
        """Total current renewable output in kW."""
        return sum(s.current_output_kw for s in self._sources.values())

    @property
    def utilization_pct(self) -> float:
        """Overall utilization percentage."""
        if self.total_capacity_kw == 0:
            return 0.0
        return (self.total_output_kw / self.total_capacity_kw) * 100.0

    def add_source(self, source: RenewableSource) -> None:
        """Add a renewable energy source.

        Parameters
        ----------
        source:
            The source to add.
        """
        if source.source_id in self._sources:
            raise ValueError(f"Source {source.source_id} already exists")
        self._sources[source.source_id] = source

    def remove_source(self, source_id: str) -> RenewableSource | None:
        """Remove a renewable energy source.

        Returns
        -------
        RenewableSource | None:
            The removed source, or None if not found.
        """
        return self._sources.pop(source_id, None)

    def update_output(self, source_id: str, output_kw: float) -> None:
        """Update the output of a renewable source.

        Parameters
        ----------
        source_id:
            Source identifier.
        output_kw:
            New output in kilowatts.
        """
        source = self._sources.get(source_id)
        if source is None:
            raise ValueError(f"Unknown source: {source_id}")
        if output_kw < 0:
            raise ValueError("output_kw must be non-negative")
        if output_kw > source.capacity_kw:
            raise ValueError(
                f"output_kw {output_kw} exceeds capacity {source.capacity_kw}"
            )
        updated = RenewableSource(
            source_id=source.source_id,
            source_type=source.source_type,
            capacity_kw=source.capacity_kw,
            current_output_kw=output_kw,
            availability_pct=source.availability_pct,
            location=source.location,
            metadata=source.metadata,
        )
        self._sources[source_id] = updated

    def calculate_offset(self, total_demand_kw: float) -> dict[str, float]:
        """Calculate renewable energy offset.

        Parameters
        ----------
        total_demand_kw:
            Total power demand in kilowatts.

        Returns
        -------
        dict[str, float]:
            Offset metrics.
        """
        if total_demand_kw <= 0:
            return {
                "offset_kw": 0.0,
                "offset_pct": 0.0,
                "grid_demand_kw": 0.0,
            }

        offset_kw = min(self.total_output_kw, total_demand_kw)
        offset_pct = (offset_kw / total_demand_kw) * 100.0
        grid_demand_kw = total_demand_kw - offset_kw

        return {
            "offset_kw": offset_kw,
            "offset_pct": offset_pct,
            "grid_demand_kw": grid_demand_kw,
        }

    def forecast_output(
        self, source_type: RenewableSourceType, hours_ahead: int = 24
    ) -> list[tuple[datetime, float]]:
        """Forecast renewable energy output.

        Parameters
        ----------
        source_type:
            Type of renewable source to forecast.
        hours_ahead:
            Hours to forecast ahead.

        Returns
        -------
        list[tuple[datetime, float]]:
            Forecasted (timestamp, output_kw) pairs.
        """
        sources = [s for s in self._sources.values() if s.source_type == source_type]
        if not sources:
            return []

        total_capacity = sum(s.capacity_kw for s in sources)
        forecast: list[tuple[datetime, float]] = []
        now = datetime.utcnow()

        for hour in range(hours_ahead):
            timestamp = now + timedelta(hours=hour)
            # Simple diurnal model for solar
            if source_type == RenewableSourceType.SOLAR:
                hour_of_day = timestamp.hour
                if 6 <= hour_of_day <= 18:
                    # Parabolic solar curve
                    peak_factor = math.sin(math.pi * (hour_of_day - 6) / 12)
                    output = total_capacity * peak_factor * 0.8  # 80% capacity factor
                else:
                    output = 0.0
            elif source_type == RenewableSourceType.WIND:
                # Random wind with daily pattern
                import random
                random.seed(hour)
                output = total_capacity * random.uniform(0.2, 0.8)
            else:
                output = total_capacity * 0.7  # Baseload

            forecast.append((timestamp, round(output, 2)))

        return forecast

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all renewable sources."""
        by_type: dict[str, dict[str, float]] = {}
        for source in self._sources.values():
            type_name = source.source_type.value
            if type_name not in by_type:
                by_type[type_name] = {"capacity_kw": 0.0, "output_kw": 0.0, "count": 0}
            by_type[type_name]["capacity_kw"] += source.capacity_kw
            by_type[type_name]["output_kw"] += source.current_output_kw
            by_type[type_name]["count"] += 1

        return {
            "total_capacity_kw": self.total_capacity_kw,
            "total_output_kw": self.total_output_kw,
            "utilization_pct": self.utilization_pct,
            "source_count": len(self._sources),
            "by_type": by_type,
        }


# ---------------------------------------------------------------------------
# PUE Optimizer
# ---------------------------------------------------------------------------


class PUEOptimizer:
    """Optimizes Power Usage Effectiveness.

    Monitors PUE in real-time, identifies inefficiencies, and
    recommends corrective actions.

    Parameters
    ----------
    pue_target:
        Target PUE value.
    """

    def __init__(self, *, pue_target: float = DEFAULT_PUE_TARGET) -> None:
        if pue_target < 1.0:
            raise ValueError("pue_target must be >= 1.0")
        self._pue_target = pue_target
        self._history: list[PUEStatus] = []
        self._max_history = 10_000
        self._alerts: list[PowerAlert] = []

    @property
    def pue_target(self) -> float:
        """Target PUE."""
        return self._pue_target

    @property
    def current_pue(self) -> float | None:
        """Most recent PUE reading."""
        if not self._history:
            return None
        return self._history[-1].pue_value

    @property
    def average_pue(self) -> float | None:
        """Average PUE over the history window."""
        if not self._history:
            return None
        return sum(h.pue_value for h in self._history) / len(self._history)

    def calculate_pue(
        self,
        it_load_kw: float,
        cooling_load_kw: float,
        overhead_load_kw: float | None = None,
    ) -> PUEStatus:
        """Calculate PUE from component loads.

        Parameters
        ----------
        it_load_kw:
            IT equipment load in kilowatts.
        cooling_load_kw:
            Cooling system load in kilowatts.
        overhead_load_kw:
            Other overhead load (lighting, etc.). Estimated if None.

        Returns
        -------
        PUEStatus:
            Calculated PUE status.
        """
        if it_load_kw <= 0:
            raise ValueError("it_load_kw must be positive")
        if cooling_load_kw < 0:
            raise ValueError("cooling_load_kw must be non-negative")

        overhead = overhead_load_kw if overhead_load_kw is not None else it_load_kw * 0.05
        total = it_load_kw + cooling_load_kw + overhead
        pue = total / it_load_kw

        # Determine alert level
        if pue > DEFAULT_PUE_CRITICAL:
            alert_level = PUEAlertLevel.EMERGENCY
        elif pue > DEFAULT_HIGH_PUE_THRESHOLD:
            alert_level = PUEAlertLevel.CRITICAL
        elif pue > DEFAULT_PUE_WARNING:
            alert_level = PUEAlertLevel.WARNING
        elif pue > self._pue_target:
            alert_level = PUEAlertLevel.INFO
        else:
            alert_level = PUEAlertLevel.INFO

        status = PUEStatus(
            pue_value=round(pue, 3),
            it_load_kw=it_load_kw,
            total_facility_load_kw=total,
            cooling_load_kw=cooling_load_kw,
            overhead_load_kw=overhead,
            pue_target=self._pue_target,
            alert_level=alert_level,
        )

        self._history.append(status)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        if alert_level in (PUEAlertLevel.WARNING, PUEAlertLevel.CRITICAL, PUEAlertLevel.EMERGENCY):
            self._alerts.append(
                PowerAlert(
                    level=alert_level,
                    source="pue_optimizer",
                    message=f"PUE {pue:.3f} exceeds threshold (target: {self._pue_target})",
                    current_value=pue,
                    threshold_value=self._pue_target,
                )
            )

        return status

    def identify_inefficiencies(self) -> list[dict[str, Any]]:
        """Identify sources of PUE inefficiency.

        Returns
        -------
        list[dict[str, Any]]:
            List of identified inefficiencies.
        """
        if not self._history:
            return []

        latest = self._history[-1]
        inefficiencies: list[dict[str, Any]] = []

        # Cooling overhead analysis
        cooling_ratio = latest.cooling_load_kw / latest.it_load_kw
        if cooling_ratio > 0.5:
            inefficiencies.append({
                "source": "cooling",
                "severity": "high" if cooling_ratio > 0.7 else "medium",
                "detail": f"Cooling load is {cooling_ratio:.1%} of IT load",
                "recommendation": "Optimize airflow management and setpoints",
            })

        # Overhead analysis
        overhead_ratio = latest.overhead_load_kw / latest.it_load_kw
        if overhead_ratio > 0.1:
            inefficiencies.append({
                "source": "overhead",
                "severity": "medium",
                "detail": f"Overhead load is {overhead_ratio:.1%} of IT load",
                "recommendation": "Audit non-IT power consumers",
            })

        # Trend analysis
        if len(self._history) >= 10:
            recent = [h.pue_value for h in self._history[-10:]]
            trend = (recent[-1] - recent[0]) / len(recent)
            if trend > 0.01:
                inefficiencies.append({
                    "source": "trend",
                    "severity": "high",
                    "detail": f"PUE trending upward ({trend:+.4f} per reading)",
                    "recommendation": "Investigate recent changes in cooling or load",
                })

        return inefficiencies

    def recommend_actions(self) -> list[dict[str, Any]]:
        """Generate PUE optimization recommendations.

        Returns
        -------
        list[dict[str, Any]:
            Recommended actions.
        """
        recommendations: list[dict[str, Any]] = []
        inefficiencies = self.identify_inefficiencies()

        for ineff in inefficiencies:
            recommendations.append({
                "priority": ineff["severity"],
                "action": ineff["recommendation"],
                "category": ineff["source"],
                "expected_impact": "Reduce PUE by 0.05-0.15",
            })

        # Always include baseline recommendations
        if not recommendations:
            recommendations.append({
                "priority": "low",
                "action": "Continue monitoring PUE trends",
                "category": "monitoring",
                "expected_impact": "Maintain current efficiency",
            })

        return recommendations

    def get_trend(self, hours: int = 24) -> list[tuple[datetime, float]]:
        """Get PUE trend over a time window.

        Parameters
        ----------
        hours:
            Number of hours to look back.

        Returns
        -------
        list[tuple[datetime, float]]:
            PUE trend data.
        """
        if not self._history:
            return []
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [(h.timestamp, h.pue_value) for h in self._history if h.timestamp >= cutoff]

    def get_summary(self) -> dict[str, Any]:
        """Get PUE optimization summary."""
        return {
            "pue_target": self._pue_target,
            "current_pue": self.current_pue,
            "average_pue": self.average_pue,
            "readings_count": len(self._history),
            "active_alerts": len(self._alerts),
            "inefficiencies": self.identify_inefficiencies(),
            "recommendations": self.recommend_actions(),
        }


# ---------------------------------------------------------------------------
# Load Shedding Controller
# ---------------------------------------------------------------------------


class LoadSheddingController:
    """Manages load shedding during power emergencies.

    Prioritizes loads and sheds non-critical loads to maintain
    power for essential systems.

    Parameters
    ----------
    total_capacity_kw:
        Total available power capacity.
    """

    def __init__(self, *, total_capacity_kw: float) -> None:
        if total_capacity_kw <= 0:
            raise ValueError("total_capacity_kw must be positive")
        self._total_capacity_kw = total_capacity_kw
        self._loads: dict[str, dict[str, Any]] = {}
        self._shed_events: list[LoadShedEvent] = []
        self._shedding_active = False

    @property
    def shedding_active(self) -> bool:
        """Whether load shedding is currently active."""
        return self._shedding_active

    @property
    def total_shed_kw(self) -> float:
        """Total currently shed load in kW."""
        return sum(
            e.shed_kw for e in self._shed_events if e.is_active
        )

    def register_load(
        self,
        load_id: str,
        *,
        priority: LoadPriority,
        power_kw: float,
        description: str = "",
    ) -> None:
        """Register a manageable load.

        Parameters
        ----------
        load_id:
            Unique load identifier.
        priority:
            Load priority for shedding decisions.
        power_kw:
            Power consumption in kilowatts.
        description:
            Human-readable description.
        """
        if not load_id:
            raise ValueError("load_id must be non-empty")
        if power_kw < 0:
            raise ValueError("power_kw must be non-negative")
        self._loads[load_id] = {
            "priority": priority,
            "power_kw": power_kw,
            "description": description,
            "shed": False,
        }

    def unregister_load(self, load_id: str) -> bool:
        """Unregister a load.

        Returns
        -------
        bool:
            True if the load was found and removed.
        """
        return self._loads.pop(load_id, None) is not None

    def evaluate_shedding(self, current_demand_kw: float) -> list[LoadShedEvent]:
        """Evaluate and execute load shedding if needed.

        Parameters
        ----------
        current_demand_kw:
            Current total power demand in kilowatts.

        Returns
        -------
        list[LoadShedEvent]:
            New shed events created.
        """
        new_events: list[LoadShedEvent] = []
        available = self._total_capacity_kw - self.total_shed_kw

        if current_demand_kw <= available:
            self._shedding_active = False
            return new_events

        self._shedding_active = True
        deficit = current_demand_kw - available

        # Sort loads by priority (shed lowest priority first)
        sheddable = [
            (lid, info)
            for lid, info in self._loads.items()
            if not info["shed"] and info["priority"] != LoadPriority.CRITICAL
        ]
        sheddable.sort(key=lambda x: x[1]["priority"].value, reverse=True)

        for load_id, info in sheddable:
            if deficit <= 0:
                break
            event = LoadShedEvent(
                event_id=f"shed-{load_id}-{datetime.utcnow().timestamp()}",
                load_id=load_id,
                priority=info["priority"],
                shed_kw=info["power_kw"],
                reason=f"Power deficit: {deficit:.1f} kW",
            )
            self._shed_events.append(event)
            self._loads[load_id]["shed"] = True
            new_events.append(event)
            deficit -= info["power_kw"]

        return new_events

    def restore_load(self, load_id: str) -> bool:
        """Restore a shed load.

        Parameters
        ----------
        load_id:
            Load to restore.

        Returns
        -------
        bool:
            True if the load was restored.
        """
        for event in self._shed_events:
            if event.load_id == load_id and event.is_active:
                # Create a new event with restored_at set
                restored = LoadShedEvent(
                    event_id=event.event_id,
                    load_id=event.load_id,
                    priority=event.priority,
                    shed_kw=event.shed_kw,
                    reason=event.reason,
                    shed_at=event.shed_at,
                    restored_at=datetime.utcnow(),
                )
                idx = self._shed_events.index(event)
                self._shed_events[idx] = restored
                self._loads[load_id]["shed"] = False
                logger.info("Load restored: %s", load_id)
                return True
        return False

    def restore_all(self) -> int:
        """Restore all shed loads.

        Returns
        -------
        int:
            Number of loads restored.
        """
        count = 0
        for load_id in list(self._loads.keys()):
            if self.restore_load(load_id):
                count += 1
        self._shedding_active = False
        return count

    def get_status(self) -> dict[str, Any]:
        """Get load shedding status."""
        return {
            "shedding_active": self._shedding_active,
            "total_shed_kw": self.total_shed_kw,
            "registered_loads": len(self._loads),
            "shed_loads": sum(1 for info in self._loads.values() if info["shed"]),
            "total_events": len(self._shed_events),
            "active_events": sum(1 for e in self._shed_events if e.is_active),
        }


# ---------------------------------------------------------------------------
# Power Management Orchestrator
# ---------------------------------------------------------------------------


class PowerManagementOrchestrator:
    """Orchestrates all power management subsystems.

    Coordinates UPS, generators, renewables, PUE optimization, and
    load shedding for comprehensive power management.

    Parameters
    ----------
    ups_manager:
        UPS manager instance.
    generator_manager:
        Generator manager instance.
    renewable_integrator:
        Renewable energy integrator instance.
    pue_optimizer:
        PUE optimizer instance.
    load_shedding_controller:
        Load shedding controller instance.
    """

    def __init__(
        self,
        *,
        ups_manager: UPSManager,
        generator_manager: GeneratorManager,
        renewable_integrator: RenewableEnergyIntegrator,
        pue_optimizer: PUEOptimizer,
        load_shedding_controller: LoadSheddingController,
    ) -> None:
        self._ups = ups_manager
        self._generator = generator_manager
        self._renewables = renewable_integrator
        self._pue = pue_optimizer
        self._shedding = load_shedding_controller
        self._alerts: list[PowerAlert] = []

    @property
    def alerts(self) -> list[PowerAlert]:
        """Active power alerts."""
        return list(self._alerts)

    def run_control_cycle(
        self,
        it_load_kw: float,
        cooling_load_kw: float,
        grid_available: bool = True,
    ) -> dict[str, Any]:
        """Execute one power management control cycle.

        Parameters
        ----------
        it_load_kw:
            Current IT load in kilowatts.
        cooling_load_kw:
            Current cooling load in kilowatts.
        grid_available:
            Whether grid power is available.

        Returns
        -------
        dict[str, Any]:
            Control decisions and system status.
        """
        # 1. Calculate PUE
        pue_status = self._pue.calculate_pue(it_load_kw, cooling_load_kw)

        # 2. Calculate renewable offset
        total_demand = it_load_kw + cooling_load_kw
        renewable_offset = self._renewables.calculate_offset(total_demand)

        # 3. Handle grid failure
        if not grid_available:
            self._handle_grid_failure(total_demand)

        # 4. Evaluate load shedding
        shed_events = self._shedding.evaluate_shedding(total_demand)

        # 5. Check UPS status
        ups_status = self._ups.get_status()
        if ups_status.charge_pct < DEFAULT_LOW_BATTERY_THRESHOLD_PCT:
            self._alerts.append(
                PowerAlert(
                    level=PUEAlertLevel.WARNING,
                    source="ups",
                    message=f"UPS charge low ({ups_status.charge_pct:.1f}%)",
                    current_value=ups_status.charge_pct,
                    threshold_value=DEFAULT_LOW_BATTERY_THRESHOLD_PCT,
                )
            )

        return {
            "pue": pue_status.pue_value,
            "pue_target": self._pue.pue_target,
            "renewable_offset_pct": renewable_offset["offset_pct"],
            "grid_demand_kw": renewable_offset["grid_demand_kw"],
            "ups_state": ups_status.state.name,
            "ups_charge_pct": ups_status.charge_pct,
            "generator_state": self._generator.state.name,
            "shedding_active": self._shedding.shedding_active,
            "shed_events": len(shed_events),
            "alerts": len(self._alerts),
        }

    def _handle_grid_failure(self, total_demand_kw: float) -> None:
        """Handle a grid power failure."""
        logger.warning("Grid failure detected! Initiating failover...")

        # 1. Transfer to UPS
        if self._ups.state == UPSState.ONLINE:
            self._ups.transition_to_battery()

        # 2. Start generator if UPS runtime is insufficient
        ups_runtime_min = self._ups.estimate_runtime_min()
        if ups_runtime_min < DEFAULT_UPS_RUNTIME_MIN:
            if not self._generator.is_available:
                self._generator.start()
            self._generator.transfer_load(total_demand_kw)

        # 3. Shed non-critical loads if needed
        available = (
            self._ups._capacity_kw * (self._ups.charge_pct / 100.0)
            + (self._generator._capacity_kw if self._generator.state == GeneratorState.RUNNING else 0.0)
            + self._renewables.total_output_kw
        )
        if total_demand_kw > available:
            self._shedding.evaluate_shedding(total_demand_kw)

    def get_system_summary(self) -> dict[str, Any]:
        """Get a comprehensive power system summary."""
        return {
            "ups": self._ups.get_health_report(),
            "generator": self._generator.get_health_report(),
            "renewables": self._renewables.get_summary(),
            "pue": self._pue.get_summary(),
            "load_shedding": self._shedding.get_status(),
            "active_alerts": len(self._alerts),
        }


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    "ElectricalMeasurement",
    "GeneratorFaultError",
    "GeneratorManager",
    "GeneratorState",
    "GeneratorStatus",
    "InsufficientCapacityError",
    "LoadPriority",
    "LoadShedEvent",
    "LoadSheddingController",
    "PowerAlert",
    "PowerBudget",
    "PowerBudgetExceededError",
    "PowerManagementError",
    "PowerManagementOrchestrator",
    "PowerQuality",
    "PowerSourceState",
    "PUEAlertLevel",
    "PUEOptimizer",
    "PUEStatus",
    "RenewableEnergyIntegrator",
    "RenewableIntegrationError",
    "RenewableSource",
    "RenewableSourceType",
    "TransferSwitchError",
    "UPSFaultError",
    "UPSManager",
    "UPSState",
    "UPSStatus",
]
