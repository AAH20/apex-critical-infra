"""Apex Critical Infrastructure — Cooling Optimization.

HVAC optimization, liquid cooling control, free cooling, and thermal
modeling for data center environments.

Integrates with Data Center Commander (DC lifecycle), Apex_ULL (ultra-low
latency), and ApexGraphSwarm (multi-agent orchestration).
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

# Thermal constants
ABSOLUTE_ZERO_C = -273.15
STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴)
AIR_DENSITY_KG_M3 = 1.225  # at sea level, 15°C
AIR_SPECIFIC_HEAT_J_KG_K = 1005.0  # J/(kg·K)
WATER_SPECIFIC_HEAT_J_KG_K = 4186.0  # J/(kg·K)
TON_OF_REFRIGERATION_KW = 3.51685  # kW per ton

# Default operational parameters
DEFAULT_SUPPLY_TEMP_C = 18.0
DEFAULT_RETURN_TEMP_C = 27.0
DEFAULT_AIRFLOW_CFM_PER_TON = 400.0
DEFAULT_CHILLED_WATER_SUPPLY_C = 7.0
DEFAULT_CHILLED_WATER_RETURN_C = 12.0
DEFAULT_COOLING_TOWER_APPROACH_C = 5.0
DEFAULT_FREE_COOLING_THRESHOLD_C = 15.0
DEFAULT_WET_BULB_THRESHOLD_C = 10.0
DEFAULT_MAX_HUMIDITY_PCT = 80.0
DEFAULT_MIN_HUMIDITY_PCT = 20.0
DEFAULT_THERMAL_MASS_KJ_M2_K = 50.0
DEFAULT_HEAT_TRANSFER_COEFF_W_M2_K = 10.0

# Control parameters
DEFAULT_PID_KP = 1.0
DEFAULT_PID_KI = 0.1
DEFAULT_PID_KD = 0.05
DEFAULT_DEADBAND_C = 0.5
DEFAULT_RAMP_RATE_PCT_PER_MIN = 10.0


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CoolingSystemType(Enum):
    """Type of cooling system."""

    AIR_COOLED = "air_cooled"
    WATER_COOLED = "water_cooled"
    DIRECT_LIQUID = "direct_liquid"
    IMMERSION = "immersion"
    TWO_PHASE = "two_phase"
    EVAPORATIVE = "evaporative"
    HYBRID = "hybrid"


class CoolingState(Enum):
    """Operational state of a cooling system."""

    OFF = auto()
    STANDBY = auto()
    STARTING = auto()
    RUNNING = auto()
    DEGRADED = auto()
    MAINTENANCE = auto()
    FAULT = auto()
    EMERGENCY = auto()


class ValveState(Enum):
    """State of a control valve."""

    CLOSED = auto()
    OPEN = auto()
    MODULATING = auto()
    FAULT = auto()


class PumpState(Enum):
    """State of a pump."""

    OFF = auto()
    RUNNING = auto()
    DEGRADED = auto()
    FAULT = auto()


class FreeCoolingMode(Enum):
    """Free cooling operating mode."""

    DISABLED = auto()
    ECONOMIZER = auto()
    EVAPORATIVE = auto()
    THERMOSIPHON = auto()
    COMPRESSOR_ASSIST = auto()


class ThermalAlertLevel(Enum):
    """Severity of thermal alerts."""

    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()
    EMERGENCY = auto()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CoolingError(Exception):
    """Base exception for cooling optimization."""


class ThermalModelError(CoolingError):
    """Raised when a thermal model calculation fails."""


class ControlLoopError(CoolingError):
    """Raised when a control loop encounters an error."""


class SensorCalibrationError(CoolingError):
    """Raised when a sensor calibration issue is detected."""


class SafetyInterlockError(CoolingError):
    """Raised when a safety interlock prevents an operation."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ThermalPoint:
    """A point in 3D thermal space."""

    x_m: float
    y_m: float
    z_m: float
    temperature_c: float

    def __post_init__(self) -> None:
        if self.temperature_c < ABSOLUTE_ZERO_C:
            raise ValueError(
                f"temperature {self.temperature_c}C below absolute zero"
            )


@dataclass(frozen=True, slots=True)
class AirProperties:
    """Thermodynamic properties of air."""

    dry_bulb_temp_c: float
    relative_humidity_pct: float
    altitude_m: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.relative_humidity_pct <= 100.0:
            raise ValueError(
                f"relative_humidity_pct {self.relative_humidity_pct} out of range [0, 100]"
            )

    @property
    def wet_bulb_temp_c(self) -> float:
        """Estimate wet-bulb temperature using Stull formula."""
        T = self.dry_bulb_temp_c
        RH = self.relative_humidity_pct
        # Stull (2011) approximation
        tw = (
            T * math.atan(0.151977 * math.sqrt(RH + 8.313659))
            + math.atan(T + RH)
            - math.atan(RH - 1.676331)
            + 0.00391838 * RH ** (3 / 2) * math.atan(0.023101 * RH)
            - 4.686035
        )
        return tw

    @property
    def dew_point_c(self) -> float:
        """Calculate dew point temperature."""
        T = self.dry_bulb_temp_c
        RH = self.relative_humidity_pct
        # Magnus formula
        a = 17.625
        b = 243.04
        alpha = math.log(RH / 100.0) + (a * T) / (b + T)
        return (b * alpha) / (a - alpha)

    @property
    def density_kg_m3(self) -> float:
        """Calculate air density."""
        T_k = self.dry_bulb_temp_c - ABSOLUTE_ZERO_C
        # Barometric formula for pressure at altitude
        P0 = 101325.0  # Pa at sea level
        P = P0 * math.exp(-self.altitude_m / 8434.0)
        # Ideal gas law for humid air (approximation)
        return P / (287.05 * T_k)

    @property
    def enthalpy_kj_kg(self) -> float:
        """Calculate specific enthalpy of moist air."""
        T = self.dry_bulb_temp_c
        RH = self.relative_humidity_pct
        # Humidity ratio (kg water / kg dry air)
        p_ws = 610.94 * math.exp(17.625 * T / (T + 243.04))  # saturation pressure
        p_w = (RH / 100.0) * p_ws
        W = 0.62198 * p_w / (101325.0 - p_w)
        # Enthalpy
        return (1.006 * T + W * (2501.0 + 1.86 * T))  # kJ/kg


@dataclass(frozen=True, slots=True)
class LiquidCoolingState:
    """State of a liquid cooling loop."""

    supply_temp_c: float
    return_temp_c: float
    flow_rate_lpm: float  # liters per minute
    pressure_kpa: float
    coolant_type: str = "water"

    def __post_init__(self) -> None:
        if self.flow_rate_lpm < 0:
            raise ValueError("flow_rate_lpm must be non-negative")
        if self.pressure_kpa < 0:
            raise ValueError("pressure_kpa must be non-negative")

    @property
    def delta_t_c(self) -> float:
        """Temperature differential across the loop."""
        return self.return_temp_c - self.supply_temp_c

    def heat_rejection_kw(self) -> float:
        """Calculate heat rejection rate in kW."""
        if self.flow_rate_lpm <= 0:
            return 0.0
        # Q = m_dot * cp * delta_T
        # mass flow (kg/s) = flow_rate (L/min) * density (kg/L) / 60
        density = 1.0 if self.coolant_type == "water" else 0.8  # kg/L
        mass_flow_kg_s = self.flow_rate_lpm * density / 60.0
        cp = WATER_SPECIFIC_HEAT_J_KG_K if self.coolant_type == "water" else 2000.0
        return mass_flow_kg_s * cp * self.delta_t_c / 1000.0  # kW


@dataclass(frozen=True, slots=True)
class ThermalLoad:
    """A thermal load source."""

    load_id: str
    heat_output_kw: float
    location: ThermalPoint
    duty_cycle_pct: float = 100.0
    name: str = ""

    def __post_init__(self) -> None:
        if not self.load_id:
            raise ValueError("load_id must be non-empty")
        if self.heat_output_kw < 0:
            raise ValueError("heat_output_kw must be non-negative")
        if not 0.0 <= self.duty_cycle_pct <= 100.0:
            raise ValueError("duty_cycle_pct must be in range [0, 100]")

    @property
    def effective_heat_kw(self) -> float:
        """Effective heat output considering duty cycle."""
        return self.heat_output_kw * (self.duty_cycle_pct / 100.0)


@dataclass(frozen=True, slots=True)
class CoolingCapacity:
    """Available cooling capacity."""

    total_tons: float
    available_tons: float
    supply_temp_c: float
    ambient_temp_c: float
    system_type: CoolingSystemType
    efficiency_cop: float = 3.0

    def __post_init__(self) -> None:
        if self.total_tons <= 0:
            raise ValueError("total_tons must be positive")
        if self.available_tons < 0:
            raise ValueError("available_tons must be non-negative")
        if self.available_tons > self.total_tons:
            raise ValueError("available_tons cannot exceed total_tons")
        if self.efficiency_cop <= 0:
            raise ValueError("efficiency_cop must be positive")

    @property
    def utilization_pct(self) -> float:
        """Current utilization percentage."""
        if self.total_tons == 0:
            return 0.0
        return ((self.total_tons - self.available_tons) / self.total_tons) * 100.0

    @property
    def available_kw(self) -> float:
        """Available cooling capacity in kW."""
        return self.available_tons * TON_OF_REFRIGERATION_KW


@dataclass(frozen=True, slots=True)
class ThermalAlert:
    """A thermal system alert."""

    level: ThermalAlertLevel
    source: str
    message: str
    current_value: float
    threshold_value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True, slots=True)
class PIDControlOutput:
    """Output from a PID controller."""

    setpoint: float
    process_variable: float
    error: float
    output_pct: float
    integral: float
    derivative: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class TemperatureSensor(Protocol):
    """Protocol for temperature sensors."""

    def read(self, location: ThermalPoint) -> float:
        """Read temperature at a location."""
        ...


class ActuatorController(Protocol):
    """Protocol for cooling system actuators."""

    def set_damper_position(self, pct: float) -> bool:
        """Set damper position (0–100%)."""
        ...

    def set_valve_position(self, pct: float) -> bool:
        """Set valve position (0–100%)."""
        ...

    def set_pump_speed(self, pct: float) -> bool:
        """Set pump speed (0–100%)."""
        ...

    def set_fan_speed(self, pct: float) -> bool:
        """Set fan speed (0–100%)."""
        ...


# ---------------------------------------------------------------------------
# Thermal Model
# ---------------------------------------------------------------------------


class ThermalModel:
    """Physics-based thermal model for data center environments.

    Models heat transfer, airflow, and thermal mass to predict
    temperature distribution and cooling requirements.

    Parameters
    ----------
    room_length_m:
        Room length in meters.
    room_width_m:
        Room width in meters.
    room_height_m:
        Room height in meters.
    thermal_mass_kj_m2_k:
        Thermal mass per unit area.
    """

    def __init__(
        self,
        *,
        room_length_m: float = 50.0,
        room_width_m: float = 30.0,
        room_height_m: float = 4.0,
        thermal_mass_kj_m2_k: float = DEFAULT_THERMAL_MASS_KJ_M2_K,
    ) -> None:
        if room_length_m <= 0 or room_width_m <= 0 or room_height_m <= 0:
            raise ValueError("Room dimensions must be positive")
        if thermal_mass_kj_m2_k <= 0:
            raise ValueError("thermal_mass_kj_m2_k must be positive")

        self._length_m = room_length_m
        self._width_m = room_width_m
        self._height_m = room_height_m
        self._thermal_mass = thermal_mass_kj_m2_k
        self._loads: list[ThermalLoad] = []
        self._ambient_temp_c = 20.0
        self._current_temp_c = 22.0

    @property
    def volume_m3(self) -> float:
        """Room volume in cubic meters."""
        return self._length_m * self._width_m * self._height_m

    @property
    def floor_area_m2(self) -> float:
        """Floor area in square meters."""
        return self._length_m * self._width_m

    @property
    def total_thermal_mass_kj_k(self) -> float:
        """Total thermal mass in kJ/K."""
        return self._thermal_mass * self.floor_area_m2

    @property
    def current_temp_c(self) -> float:
        """Current modeled temperature."""
        return self._current_temp_c

    @property
    def ambient_temp_c(self) -> float:
        """Ambient temperature."""
        return self._ambient_temp_c

    def set_ambient_temp(self, temp_c: float) -> None:
        """Set the ambient temperature."""
        self._ambient_temp_c = temp_c

    def add_load(self, load: ThermalLoad) -> None:
        """Add a thermal load to the model."""
        self._loads.append(load)

    def remove_load(self, load_id: str) -> bool:
        """Remove a thermal load by ID.

        Returns
        -------
        bool
            True if the load was found and removed.
        """
        for i, load in enumerate(self._loads):
            if load.load_id == load_id:
                self._loads.pop(i)
                return True
        return False

    @property
    def total_heat_load_kw(self) -> float:
        """Total effective heat load in kW."""
        return sum(load.effective_heat_kw for load in self._loads)

    def predict_temperature(
        self,
        cooling_capacity_kw: float,
        time_horizon_s: float = 3600.0,
        time_step_s: float = 60.0,
    ) -> list[tuple[datetime, float]]:
        """Predict temperature trajectory over a time horizon.

        Uses a simple thermal mass model:
            dT/dt = (Q_load - Q_cooling - Q_loss) / C_thermal

        Parameters
        ----------
        cooling_capacity_kw:
            Available cooling capacity in kW.
        time_horizon_s:
            Prediction horizon in seconds.
        time_step_s:
            Simulation time step in seconds.

        Returns
        -------
        list[tuple[datetime, float]]
            Predicted (timestamp, temperature) pairs.
        """
        if time_horizon_s <= 0:
            raise ValueError("time_horizon_s must be positive")
        if time_step_s <= 0:
            raise ValueError("time_step_s must be positive")

        results: list[tuple[datetime, float]] = []
        temp = self._current_temp_c
        now = datetime.utcnow()
        steps = int(time_horizon_s / time_step_s)

        # Heat loss coefficient (simplified)
        u_value = 0.5  # W/(m²·K) — typical for insulated walls
        envelope_area = 2 * (
            self._length_m * self._height_m + self._width_m * self._height_m
        ) + self.floor_area_m2
        heat_loss_coeff = u_value * envelope_area  # W/K

        for step in range(steps):
            q_load_w = self.total_heat_load_kw * 1000.0
            q_cooling_w = cooling_capacity_kw * 1000.0
            q_loss_w = heat_loss_coeff * (temp - self._ambient_temp_c)

            net_heat_w = q_load_w - q_cooling_w - q_loss_w
            delta_t = (net_heat_w * time_step_s) / (self.total_thermal_mass_kj_k * 1000.0)
            temp += delta_t

            timestamp = now + timedelta(seconds=(step + 1) * time_step_s)
            results.append((timestamp, temp))

        return results

    def calculate_cooling_requirement(
        self,
        target_temp_c: float,
        safety_margin_pct: float = 10.0,
    ) -> float:
        """Calculate required cooling capacity to maintain target temperature.

        Parameters
        ----------
        target_temp_c:
            Desired internal temperature.
        safety_margin_pct:
            Safety margin percentage.

        Returns
        -------
        float
            Required cooling capacity in tons.
        """
        if safety_margin_pct < 0:
            raise ValueError("safety_margin_pct must be non-negative")

        # Steady-state: cooling must handle total heat load plus envelope gain
        u_value = 0.5
        envelope_area = 2 * (
            self._length_m * self._height_m + self._width_m * self._height_m
        ) + self.floor_area_m2
        envelope_gain_w = u_value * envelope_area * (self._ambient_temp_c - target_temp_c)

        total_heat_w = self.total_heat_load_kw * 1000.0 + max(0.0, envelope_gain_w)
        total_heat_kw = total_heat_w / 1000.0

        # Apply safety margin
        total_heat_kw *= (1.0 + safety_margin_pct / 100.0)

        # Convert to tons
        return total_heat_kw / TON_OF_REFRIGERATION_KW

    def calculate_airflow_requirement(
        self, cooling_load_tons: float, supply_temp_c: float = DEFAULT_SUPPLY_TEMP_C
    ) -> float:
        """Calculate required airflow in CFM.

        Parameters
        ----------
        cooling_load_tons:
            Cooling load in tons.
        supply_temp_c:
            Supply air temperature.

        Returns
        -------
        float
            Required airflow in cubic feet per minute.
        """
        if cooling_load_tons <= 0:
            return 0.0
        if supply_temp_c >= self._current_temp_c:
            raise ValueError(
                f"supply_temp_c ({supply_temp_c}) must be below "
                f"current temp ({self._current_temp_c})"
            )

        # Q = 1.08 * CFM * delta_T (standard formula)
        delta_t = self._current_temp_c - supply_temp_c
        cooling_btu_hr = cooling_load_tons * 12000.0
        cfm = cooling_btu_hr / (1.08 * delta_t)
        return cfm

    def get_thermal_map(self, resolution_m: float = 2.0) -> list[ThermalPoint]:
        """Generate a thermal map of the space.

        Parameters
        ----------
        resolution_m:
            Grid resolution in meters.

        Returns
        -------
        list[ThermalPoint]
            Thermal points across the space.
        """
        if resolution_m <= 0:
            raise ValueError("resolution_m must be positive")

        points: list[ThermalPoint] = []
        nx = max(1, int(self._length_m / resolution_m))
        ny = max(1, int(self._width_m / resolution_m))
        nz = max(1, int(self._height_m / resolution_m))

        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    x = (ix + 0.5) * self._length_m / nx
                    y = (iy + 0.5) * self._width_m / ny
                    z = (iz + 0.5) * self._height_m / nz

                    # Simple model: temperature increases with height
                    # and near heat sources
                    temp = self._current_temp_c + (z / self._height_m) * 2.0

                    # Add heat source influence
                    for load in self._loads:
                        dist = math.sqrt(
                            (x - load.location.x_m) ** 2
                            + (y - load.location.y_m) ** 2
                            + (z - load.location.z_m) ** 2
                        )
                        influence = load.effective_heat_kw / (1.0 + dist)
                        temp += influence * 0.1

                    points.append(
                        ThermalPoint(
                            x_m=x, y_m=y, z_m=z, temperature_c=round(temp, 2)
                        )
                    )

        return points


# ---------------------------------------------------------------------------
# HVAC Optimizer
# ---------------------------------------------------------------------------


class HVACOptimizer:
    """Optimizes HVAC system operation for efficiency.

    Implements PID-based control, setpoint optimization, and
    demand-based ventilation.

    Parameters
    ----------
    cooling_capacity:
        Available cooling capacity.
    supply_temp_setpoint_c:
        Target supply air temperature.
    """

    def __init__(
        self,
        *,
        cooling_capacity: CoolingCapacity,
        supply_temp_setpoint_c: float = DEFAULT_SUPPLY_TEMP_C,
    ) -> None:
        self._capacity = cooling_capacity
        self._supply_temp_setpoint_c = supply_temp_setpoint_c
        self._state = CoolingState.STANDBY
        self._current_load_pct = 0.0
        self._pid_state: dict[str, float] = {"integral": 0.0, "last_error": 0.0}

    @property
    def state(self) -> CoolingState:
        """Current HVAC state."""
        return self._state

    @property
    def current_load_pct(self) -> float:
        """Current load percentage."""
        return self._current_load_pct

    def start(self) -> None:
        """Start the HVAC system."""
        if self._state == CoolingState.FAULT:
            raise SafetyInterlockError("Cannot start HVAC in FAULT state")
        self._state = CoolingState.RUNNING
        logger.info("HVAC started")

    def stop(self) -> None:
        """Stop the HVAC system."""
        self._state = CoolingState.OFF
        self._current_load_pct = 0.0
        logger.info("HVAC stopped")

    def emergency_stop(self) -> None:
        """Emergency stop the HVAC system."""
        self._state = CoolingState.EMERGENCY
        self._current_load_pct = 0.0
        logger.warning("HVAC emergency stop activated")

    def compute_pid(
        self,
        process_variable: float,
        setpoint: float,
        kp: float = DEFAULT_PID_KP,
        ki: float = DEFAULT_PID_KI,
        kd: float = DEFAULT_PID_KD,
        dt_s: float = 1.0,
        output_min: float = 0.0,
        output_max: float = 100.0,
    ) -> PIDControlOutput:
        """Compute PID control output.

        Parameters
        ----------
        process_variable:
            Current measured value.
        setpoint:
            Target setpoint.
        kp:
            Proportional gain.
        ki:
            Integral gain.
        kd:
            Derivative gain.
        dt_s:
            Time step in seconds.
        output_min:
            Minimum output value.
        output_max:
            Maximum output value.

        Returns
        -------
        PIDControlOutput
            The PID control output.
        """
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")

        error = setpoint - process_variable

        # Proportional
        p_term = kp * error

        # Integral with anti-windup
        self._pid_state["integral"] += error * dt_s
        self._pid_state["integral"] = max(
            output_min / ki if ki > 0 else -1e6,
            min(output_max / ki if ki > 0 else 1e6, self._pid_state["integral"]),
        )
        i_term = ki * self._pid_state["integral"]

        # Derivative
        d_term = kd * (error - self._pid_state["last_error"]) / dt_s
        self._pid_state["last_error"] = error

        # Total output
        output = p_term + i_term + d_term
        output = max(output_min, min(output_max, output))

        return PIDControlOutput(
            setpoint=setpoint,
            process_variable=process_variable,
            error=error,
            output_pct=output,
            integral=self._pid_state["integral"],
            derivative=d_term,
        )

    def optimize_setpoint(
        self,
        current_temp_c: float,
        ambient_temp_c: float,
        humidity_pct: float,
        occupancy_count: int = 0,
    ) -> float:
        """Optimize the temperature setpoint for efficiency.

        Adjusts setpoint based on ambient conditions, humidity, and
        occupancy to minimize energy consumption while maintaining
        comfort/safety.

        Parameters
        ----------
        current_temp_c:
            Current internal temperature.
        ambient_temp_c:
            Outside air temperature.
        humidity_pct:
            Current relative humidity.
        occupancy_count:
            Number of occupants (for demand-controlled ventilation).

        Returns
        -------
        float
            Optimized setpoint temperature.
        """
        base_setpoint = self._supply_temp_setpoint_c

        # Adjust for ambient conditions
        if ambient_temp_c < current_temp_c - 10:
            # Can use more free cooling
            base_setpoint -= 1.0
        elif ambient_temp_c > current_temp_c + 5:
            # Need more mechanical cooling
            base_setpoint += 0.5

        # Adjust for humidity
        if humidity_pct > DEFAULT_MAX_HUMIDITY_PCT:
            base_setpoint -= 1.0  # Dehumidify
        elif humidity_pct < DEFAULT_MIN_HUMIDITY_PCT:
            base_setpoint += 0.5  # Reduce over-dehumidification

        # Adjust for occupancy
        if occupancy_count > 10:
            base_setpoint -= 0.5

        # Clamp to safe range
        return max(15.0, min(25.0, base_setpoint))

    def calculate_cooling_power(
        self, load_tons: float, cop: float | None = None
    ) -> float:
        """Calculate cooling power consumption.

        Parameters
        ----------
        load_tons:
            Cooling load in tons.
        cop:
            Coefficient of Performance (defaults to capacity COP).

        Returns
        -------
        float
            Power consumption in kW.
        """
        if load_tons < 0:
            raise ValueError("load_tons must be non-negative")
        effective_cop = cop or self._capacity.efficiency_cop
        if effective_cop <= 0:
            raise ValueError("COP must be positive")
        return (load_tons * TON_OF_REFRIGERATION_KW) / effective_cop

    def get_efficiency_metrics(self) -> dict[str, Any]:
        """Get current efficiency metrics."""
        return {
            "state": self._state.name,
            "load_pct": self._current_load_pct,
            "supply_temp_setpoint_c": self._supply_temp_setpoint_c,
            "capacity_tons": self._capacity.total_tons,
            "available_tons": self._capacity.available_tons,
            "utilization_pct": self._capacity.utilization_pct,
        }


# ---------------------------------------------------------------------------
# Liquid Cooling Controller
# ---------------------------------------------------------------------------


class LiquidCoolingController:
    """Controls liquid cooling systems for high-density computing.

    Manages coolant flow, temperature differential, and heat rejection
    for direct-to-chip and immersion cooling.

    Parameters
    ----------
    system_type:
        Type of liquid cooling system.
    max_flow_rate_lpm:
        Maximum coolant flow rate in liters per minute.
    """

    def __init__(
        self,
        *,
        system_type: CoolingSystemType = CoolingSystemType.DIRECT_LIQUID,
        max_flow_rate_lpm: float = 100.0,
    ) -> None:
        if max_flow_rate_lpm <= 0:
            raise ValueError("max_flow_rate_lpm must be positive")

        self._system_type = system_type
        self._max_flow_rate_lpm = max_flow_rate_lpm
        self._state = CoolingState.STANDBY
        self._current_flow_lpm = 0.0
        self._supply_temp_c = DEFAULT_CHILLED_WATER_SUPPLY_C
        self._return_temp_c = DEFAULT_CHILLED_WATER_RETURN_C
        self._pump_state = PumpState.OFF
        self._valve_state = ValveState.CLOSED
        self._target_delta_t_c = 5.0

    @property
    def state(self) -> CoolingState:
        """Current system state."""
        return self._state

    @property
    def current_flow_lpm(self) -> float:
        """Current flow rate in liters per minute."""
        return self._current_flow_lpm

    @property
    def current_heat_rejection_kw(self) -> float:
        """Current heat rejection rate in kW."""
        loop_state = LiquidCoolingState(
            supply_temp_c=self._supply_temp_c,
            return_temp_c=self._return_temp_c,
            flow_rate_lpm=self._current_flow_lpm,
            pressure_kpa=200.0,
        )
        return loop_state.heat_rejection_kw()

    def start(self) -> None:
        """Start the liquid cooling system."""
        if self._state == CoolingState.FAULT:
            raise SafetyInterlockError("Cannot start liquid cooling in FAULT state")
        self._state = CoolingState.RUNNING
        self._pump_state = PumpState.RUNNING
        self._valve_state = ValveState.OPEN
        logger.info("Liquid cooling started (%s)", self._system_type.value)

    def stop(self) -> None:
        """Stop the liquid cooling system."""
        self._state = CoolingState.OFF
        self._pump_state = PumpState.OFF
        self._valve_state = ValveState.CLOSED
        self._current_flow_lpm = 0.0
        logger.info("Liquid cooling stopped")

    def set_flow_rate(self, flow_lpm: float) -> bool:
        """Set the coolant flow rate.

        Parameters
        ----------
        flow_lpm:
            Target flow rate in liters per minute.

        Returns
        -------
        bool
            True if the flow rate was set successfully.
        """
        if flow_lpm < 0:
            raise ValueError("flow_lpm must be non-negative")
        if flow_lpm > self._max_flow_rate_lpm:
            logger.warning(
                "Requested flow %.1f LPM exceeds max %.1f LPM",
                flow_lpm,
                self._max_flow_rate_lpm,
            )
            flow_lpm = self._max_flow_rate_lpm

        self._current_flow_lpm = flow_lpm
        return True

    def modulate_for_load(
        self, heat_load_kw: float, target_supply_temp_c: float | None = None
    ) -> dict[str, Any]:
        """Modulate the cooling system for a given heat load.

        Parameters
        ----------
        heat_load_kw:
            Current heat load in kW.
        target_supply_temp_c:
            Target supply temperature (optional).

        Returns
        -------
        dict[str, Any]
            Control decisions.
        """
        if heat_load_kw < 0:
            raise ValueError("heat_load_kw must be non-negative")

        supply_temp = target_supply_temp_c or self._supply_temp_c

        # Calculate required flow rate
        # Q = m_dot * cp * delta_T => m_dot = Q / (cp * delta_T)
        if self._target_delta_t_c > 0:
            required_flow = (heat_load_kw * 60.0) / (
                WATER_SPECIFIC_HEAT_J_KG_K * self._target_delta_t_c / 1000.0
            )
        else:
            required_flow = self._max_flow_rate_lpm

        # Clamp to system limits
        actual_flow = min(required_flow, self._max_flow_rate_lpm)
        self._current_flow_lpm = actual_flow

        # Calculate resulting return temperature
        if actual_flow > 0:
            actual_delta_t = (heat_load_kw * 60.0) / (
                WATER_SPECIFIC_HEAT_J_KG_K * actual_flow / 1000.0
            )
            self._return_temp_c = supply_temp + actual_delta_t
        else:
            self._return_temp_c = supply_temp

        return {
            "flow_rate_lpm": actual_flow,
            "supply_temp_c": supply_temp,
            "return_temp_c": self._return_temp_c,
            "delta_t_c": self._return_temp_c - supply_temp,
            "heat_rejection_kw": self.current_heat_rejection_kw,
            "pump_speed_pct": (actual_flow / self._max_flow_rate_lpm) * 100.0,
        }

    def detect_leak(self, pressure_kpa: float, flow_rate_lpm: float) -> bool:
        """Detect potential coolant leaks.

        Parameters
        ----------
        pressure_kpa:
            Current loop pressure.
        flow_rate_lpm:
            Current flow rate.

        Returns
        -------
        bool
            True if a leak is suspected.
        """
        # Simple heuristic: pressure drop with constant flow suggests leak
        expected_pressure = 200.0 + (flow_rate_lpm / self._max_flow_rate_lpm) * 100.0
        pressure_drop = expected_pressure - pressure_kpa
        return pressure_drop > 50.0  # kPa threshold

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "system_type": self._system_type.value,
            "state": self._state.name,
            "flow_rate_lpm": self._current_flow_lpm,
            "max_flow_rate_lpm": self._max_flow_rate_lpm,
            "supply_temp_c": self._supply_temp_c,
            "return_temp_c": self._return_temp_c,
            "delta_t_c": self._return_temp_c - self._supply_temp_c,
            "heat_rejection_kw": self.current_heat_rejection_kw,
            "pump_state": self._pump_state.name,
            "valve_state": self._valve_state.name,
        }


# ---------------------------------------------------------------------------
# Free Cooling Controller
# ---------------------------------------------------------------------------


class FreeCoolingController:
    """Controls free cooling systems (economizers, evaporative cooling).

    Determines when free cooling is available and manages the transition
    between free and mechanical cooling.

    Parameters
    ----------
    free_cooling_threshold_c:
        Ambient temperature below which free cooling is viable.
    wet_bulb_threshold_c:
        Wet-bulb temperature threshold for evaporative cooling.
    """

    def __init__(
        self,
        *,
        free_cooling_threshold_c: float = DEFAULT_FREE_COOLING_THRESHOLD_C,
        wet_bulb_threshold_c: float = DEFAULT_WET_BULB_THRESHOLD_C,
    ) -> None:
        self._threshold_c = free_cooling_threshold_c
        self._wet_bulb_threshold_c = wet_bulb_threshold_c
        self._mode = FreeCoolingMode.DISABLED
        self._active = False
        self._economizer_damper_pct = 0.0
        self._evaporative_pump_on = False

    @property
    def mode(self) -> FreeCoolingMode:
        """Current free cooling mode."""
        return self._mode

    @property
    def active(self) -> bool:
        """Whether free cooling is currently active."""
        return self._active

    def evaluate_free_cooling(
        self,
        ambient_air: AirProperties,
        internal_temp_c: float,
        cooling_load_tons: float,
    ) -> FreeCoolingMode:
        """Evaluate and select free cooling mode.

        Parameters
        ----------
        ambient_air:
            Current ambient air properties.
        internal_temp_c:
            Current internal temperature.
        cooling_load_tons:
            Current cooling load in tons.

        Returns
        -------
        FreeCoolingMode
            Selected free cooling mode.
        """
        if ambient_air.dry_bulb_temp_c > internal_temp_c - 3.0:
            # Ambient too warm for free cooling
            self._mode = FreeCoolingMode.DISABLED
            self._active = False
            self._economizer_damper_pct = 0.0
            self._evaporative_pump_on = False
        elif ambient_air.dry_bulb_temp_c <= self._threshold_c:
            # Direct economizer mode
            self._mode = FreeCoolingMode.ECONOMIZER
            self._active = True
            self._economizer_damper_pct = min(
                100.0, (internal_temp_c - ambient_air.dry_bulb_temp_c) * 10.0
            )
            self._evaporative_pump_on = False
        elif ambient_air.wet_bulb_temp_c <= self._wet_bulb_threshold_c:
            # Evaporative cooling possible
            self._mode = FreeCoolingMode.EVAPORATIVE
            self._active = True
            self._economizer_damper_pct = 50.0
            self._evaporative_pump_on = True
        else:
            # Compressor assist mode
            self._mode = FreeCoolingMode.COMPRESSOR_ASSIST
            self._active = True
            self._economizer_damper_pct = 30.0
            self._evaporative_pump_on = False

        return self._mode

    def estimate_free_cooling_capacity(
        self,
        ambient_air: AirProperties,
        internal_temp_c: float,
        max_airflow_cfm: float,
    ) -> float:
        """Estimate available free cooling capacity in tons.

        Parameters
        ----------
        ambient_air:
            Current ambient air properties.
        internal_temp_c:
            Current internal temperature.
        max_airflow_cfm:
            Maximum available airflow in CFM.

        Returns
        -------
        float:
            Estimated free cooling capacity in tons.
        """
        if not self._active or max_airflow_cfm <= 0:
            return 0.0

        delta_t = internal_temp_c - ambient_air.dry_bulb_temp_c
        if delta_t <= 0:
            return 0.0

        # Q = 1.08 * CFM * delta_T (BTU/hr)
        cooling_btu_hr = 1.08 * max_airflow_cfm * delta_t
        return cooling_btu_hr / 12000.0  # Convert to tons

    def calculate_energy_savings(
        self,
        mechanical_cooling_power_kw: float,
        free_cooling_hours: float,
    ) -> dict[str, float]:
        """Calculate energy savings from free cooling.

        Parameters
        ----------
        mechanical_cooling_power_kw:
            Power that would have been used by mechanical cooling.
        free_cooling_hours:
            Hours of free cooling operation.

        Returns
        -------
        dict[str, float]
            Energy savings metrics.
        """
        if mechanical_cooling_power_kw < 0:
            raise ValueError("mechanical_cooling_power_kw must be non-negative")
        if free_cooling_hours < 0:
            raise ValueError("free_cooling_hours must be non-negative")

        saved_kwh = mechanical_cooling_power_kw * free_cooling_hours
        # Assume grid electricity at $0.10/kWh and 0.4 kg CO2/kWh
        cost_savings = saved_kwh * 0.10
        co2_reduction_kg = saved_kwh * 0.4

        return {
            "saved_kwh": saved_kwh,
            "cost_savings_usd": cost_savings,
            "co2_reduction_kg": co2_reduction_kg,
        }

    def get_status(self) -> dict[str, Any]:
        """Get free cooling system status."""
        return {
            "mode": self._mode.name,
            "active": self._active,
            "economizer_damper_pct": self._economizer_damper_pct,
            "evaporative_pump_on": self._evaporative_pump_on,
            "threshold_c": self._threshold_c,
            "wet_bulb_threshold_c": self._wet_bulb_threshold_c,
        }


# ---------------------------------------------------------------------------
# Cooling System Orchestrator
# ---------------------------------------------------------------------------


class CoolingOrchestrator:
    """Orchestrates all cooling systems for optimal efficiency.

    Coordinates HVAC, liquid cooling, and free cooling systems to
    minimize energy consumption while maintaining safe operating
    temperatures.

    Parameters
    ----------
    hvac_optimizer:
        The HVAC optimizer.
    liquid_controller:
        The liquid cooling controller.
    free_cooling_controller:
        The free cooling controller.
    thermal_model:
        The thermal model.
    """

    def __init__(
        self,
        *,
        hvac_optimizer: HVACOptimizer,
        liquid_controller: LiquidCoolingController,
        free_cooling_controller: FreeCoolingController,
        thermal_model: ThermalModel,
    ) -> None:
        self._hvac = hvac_optimizer
        self._liquid = liquid_controller
        self._free_cooling = free_cooling_controller
        self._thermal_model = thermal_model
        self._alerts: list[ThermalAlert] = []

    @property
    def alerts(self) -> list[ThermalAlert]:
        """Active thermal alerts."""
        return list(self._alerts)

    def run_control_cycle(
        self,
        ambient_air: AirProperties,
        internal_temp_c: float,
        cooling_load_tons: float,
    ) -> dict[str, Any]:
        """Execute one control cycle across all cooling systems.

        Parameters
        ----------
        ambient_air:
            Current ambient air properties.
        internal_temp_c:
            Current internal temperature.
        cooling_load_tons:
            Current cooling load in tons.

        Returns
        -------
        dict[str, Any]
            Control decisions and system status.
        """
        # 1. Evaluate free cooling opportunity
        fc_mode = self._free_cooling.evaluate_free_cooling(
            ambient_air, internal_temp_c, cooling_load_tons
        )

        # 2. Calculate free cooling capacity
        free_cooling_tons = self._free_cooling.estimate_free_cooling_capacity(
            ambient_air, internal_temp_c, 10000.0
        )

        # 3. Determine mechanical cooling requirement
        remaining_load = max(0.0, cooling_load_tons - free_cooling_tons)

        # 4. Modulate liquid cooling
        liquid_kw = remaining_load * TON_OF_REFRIGERATION_KW
        liquid_control = self._liquid.modulate_for_load(liquid_kw)

        # 5. Optimize HVAC setpoint
        optimized_setpoint = self._hvac.optimize_setpoint(
            internal_temp_c,
            ambient_air.dry_bulb_temp_c,
            ambient_air.relative_humidity_pct,
        )

        # 6. Check for thermal alerts
        self._check_alerts(internal_temp_c, cooling_load_tons)

        return {
            "free_cooling_mode": fc_mode.name,
            "free_cooling_active": self._free_cooling.active,
            "free_cooling_tons": free_cooling_tons,
            "mechanical_cooling_tons": remaining_load,
            "liquid_cooling": liquid_control,
            "hvac_setpoint_c": optimized_setpoint,
            "total_load_tons": cooling_load_tons,
            "alerts": len(self._alerts),
        }

    def _check_alerts(self, internal_temp_c: float, cooling_load_tons: float) -> None:
        """Check for thermal threshold violations."""
        if internal_temp_c > 35.0:
            self._alerts.append(
                ThermalAlert(
                    level=ThermalAlertLevel.EMERGENCY,
                    source="thermal_model",
                    message=f"Internal temperature {internal_temp_c}C exceeds emergency threshold",
                    current_value=internal_temp_c,
                    threshold_value=35.0,
                )
            )
        elif internal_temp_c > 30.0:
            self._alerts.append(
                ThermalAlert(
                    level=ThermalAlertLevel.CRITICAL,
                    source="thermal_model",
                    message=f"Internal temperature {internal_temp_c}C exceeds critical threshold",
                    current_value=internal_temp_c,
                    threshold_value=30.0,
                )
            )

    def get_system_summary(self) -> dict[str, Any]:
        """Get a comprehensive cooling system summary."""
        return {
            "hvac": self._hvac.get_efficiency_metrics(),
            "liquid_cooling": self._liquid.get_status(),
            "free_cooling": self._free_cooling.get_status(),
            "thermal_model": {
                "current_temp_c": self._thermal_model.current_temp_c,
                "ambient_temp_c": self._thermal_model.ambient_temp_c,
                "total_heat_load_kw": self._thermal_model.total_heat_load_kw,
                "volume_m3": self._thermal_model.volume_m3,
            },
            "active_alerts": len(self._alerts),
        }


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    "AirProperties",
    "CoolingCapacity",
    "CoolingError",
    "CoolingOrchestrator",
    "CoolingState",
    "CoolingSystemType",
    "ControlLoopError",
    "FreeCoolingController",
    "FreeCoolingMode",
    "HVACOptimizer",
    "LiquidCoolingController",
    "LiquidCoolingState",
    "PIDControlOutput",
    "PumpState",
    "SafetyInterlockError",
    "SensorCalibrationError",
    "ThermalAlert",
    "ThermalAlertLevel",
    "ThermalLoad",
    "ThermalModel",
    "ThermalModelError",
    "ThermalPoint",
    "ValveState",
]
