"""Smart Grid module for Apex Critical Infrastructure.

Provides autonomous grid balancing, demand forecasting, frequency regulation,
and voltage control for agentic AI decision-making in power systems.
"""

from __future__ import annotations

import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class GridStateError(Exception):
    """Raised when the grid state is invalid or inconsistent."""


class RegulationError(Exception):
    """Raised when frequency or voltage regulation fails."""


class ForecastError(Exception):
    """Raised when demand forecasting encounters an error."""


@dataclass(frozen=True)
class Bus:
    """Represents a bus (node) in the power grid.

    Attributes:
        bus_id: Unique identifier for the bus.
        voltage_pu: Per-unit voltage magnitude (1.0 = nominal).
        voltage_angle_deg: Voltage phase angle in degrees.
        active_power_mw: Net active power injection in MW (positive = generation).
        reactive_power_mvar: Net reactive power injection in MVAr.
        bus_type: One of 'slack', 'pv', 'pq'.
    """

    bus_id: str
    voltage_pu: float = 1.0
    voltage_angle_deg: float = 0.0
    active_power_mw: float = 0.0
    reactive_power_mvar: float = 0.0
    bus_type: str = "pq"

    def __post_init__(self) -> None:
        if self.bus_type not in ("slack", "pv", "pq"):
            raise GridStateError(f"Invalid bus_type '{self.bus_type}' for bus {self.bus_id}")
        if self.voltage_pu <= 0:
            raise GridStateError(f"Voltage must be positive for bus {self.bus_id}")


@dataclass(frozen=True)
class Line:
    """Represents a transmission line between two buses.

    Attributes:
        line_id: Unique identifier.
        from_bus: Source bus ID.
        to_bus: Destination bus ID.
        resistance_pu: Per-unit resistance.
        reactance_pu: Per-unit reactance.
        susceptance_pu: Per-unit shunt susceptance.
        rating_mva: Thermal rating in MVA.
    """

    line_id: str
    from_bus: str
    to_bus: str
    resistance_pu: float = 0.01
    reactance_pu: float = 0.05
    susceptance_pu: float = 0.0
    rating_mva: float = 100.0

    def __post_init__(self) -> None:
        if self.from_bus == self.to_bus:
            raise GridStateError(f"Line {self.line_id} connects a bus to itself")
        if self.rating_mva <= 0:
            raise GridStateError(f"Line {self.line_id} has non-positive rating")


@dataclass
class Generator:
    """Represents a controllable generator.

    Attributes:
        gen_id: Unique identifier.
        bus_id: Bus where the generator is connected.
        p_min_mw: Minimum active power output in MW.
        p_max_mw: Maximum active power output in MW.
        q_min_mvar: Minimum reactive power output in MVAr.
        q_max_mvar: Maximum reactive power output in MVAr.
        ramp_rate_mw_per_min: Maximum ramp rate in MW/min.
        current_output_mw: Current active power output in MW.
        is_online: Whether the generator is online.
        marginal_cost_per_mwh: Marginal cost in $/MWh for economic dispatch.
    """

    gen_id: str
    bus_id: str
    p_min_mw: float = 0.0
    p_max_mw: float = 100.0
    q_min_mvar: float = -50.0
    q_max_mvar: float = 50.0
    ramp_rate_mw_per_min: float = 10.0
    current_output_mw: float = 0.0
    is_online: bool = True
    marginal_cost_per_mwh: float = 50.0

    def __post_init__(self) -> None:
        if self.p_min_mw > self.p_max_mw:
            raise GridStateError(
                f"Generator {self.gen_id}: p_min ({self.p_min_mw}) > p_max ({self.p_max_mw})"
            )
        if self.q_min_mvar > self.q_max_mvar:
            raise GridStateError(
                f"Generator {self.gen_id}: q_min ({self.q_min_mvar}) > q_max ({self.q_max_mvar})"
            )
        if self.ramp_rate_mw_per_min <= 0:
            raise GridStateError(
                f"Generator {self.gen_id}: ramp_rate must be positive"
            )
        self.current_output_mw = max(self.p_min_mw, min(self.p_max_mw, self.current_output_mw))


@dataclass
class Load:
    """Represents a controllable or uncontrollable load.

    Attributes:
        load_id: Unique identifier.
        bus_id: Bus where the load is connected.
        active_power_mw: Active power demand in MW.
        reactive_power_mvar: Reactive power demand in MVAr.
        is_controllable: Whether this load can be shed or shifted.
        priority: Priority level (1 = highest, 10 = lowest).
    """

    load_id: str
    bus_id: str
    active_power_mw: float = 10.0
    reactive_power_mvar: float = 2.0
    is_controllable: bool = False
    priority: int = 5

    def __post_init__(self) -> None:
        if self.active_power_mw < 0:
            raise GridStateError(f"Load {self.load_id}: active_power_mw must be non-negative")
        if not 1 <= self.priority <= 10:
            raise GridStateError(f"Load {self.load_id}: priority must be in [1, 10]")


@dataclass
class GridSnapshot:
    """A point-in-time snapshot of the grid state.

    Attributes:
        timestamp: When the snapshot was taken.
        buses: Dictionary of bus_id -> Bus.
        lines: Dictionary of line_id -> Line.
        generators: Dictionary of gen_id -> Generator.
        loads: Dictionary of load_id -> Load.
        frequency_hz: System frequency in Hz.
        nominal_frequency_hz: Nominal system frequency in Hz.
    """

    timestamp: datetime
    buses: Dict[str, Bus] = field(default_factory=dict)
    lines: Dict[str, Line] = field(default_factory=dict)
    generators: Dict[str, Generator] = field(default_factory=dict)
    loads: Dict[str, Load] = field(default_factory=dict)
    frequency_hz: float = 50.0
    nominal_frequency_hz: float = 50.0

    @property
    def total_generation_mw(self) -> float:
        """Total active power generation in MW."""
        return sum(
            g.current_output_mw for g in self.generators.values() if g.is_online
        )

    @property
    def total_load_mw(self) -> float:
        """Total active power demand in MW."""
        return sum(load.active_power_mw for load in self.loads.values())

    @property
    def power_imbalance_mw(self) -> float:
        """Power imbalance (generation - load) in MW. Positive = surplus."""
        return self.total_generation_mw - self.total_load_mw

    @property
    def frequency_deviation_hz(self) -> float:
        """Frequency deviation from nominal in Hz."""
        return self.frequency_hz - self.nominal_frequency_hz

    def validate(self) -> None:
        """Validate the grid snapshot for consistency.

        Raises:
            GridStateError: If the snapshot is inconsistent.
        """
        if self.frequency_hz <= 0:
            raise GridStateError("Frequency must be positive")
        if self.nominal_frequency_hz <= 0:
            raise GridStateError("Nominal frequency must be positive")

        # Check that all generator bus IDs exist
        for gen in self.generators.values():
            if gen.bus_id not in self.buses:
                raise GridStateError(
                    f"Generator {gen.gen_id} references unknown bus {gen.bus_id}"
                )

        # Check that all load bus IDs exist
        for load in self.loads.values():
            if load.bus_id not in self.buses:
                raise GridStateError(
                    f"Load {load.load_id} references unknown bus {load.bus_id}"
                )

        # Check that all line bus IDs exist
        for line in self.lines.values():
            if line.from_bus not in self.buses:
                raise GridStateError(
                    f"Line {line.line_id} references unknown from_bus {line.from_bus}"
                )
            if line.to_bus not in self.buses:
                raise GridStateError(
                    f"Line {line.line_id} references unknown to_bus {line.to_bus}"
                )


# ---------------------------------------------------------------------------
# Demand Forecasting
# ---------------------------------------------------------------------------

class ForecastHorizon(Enum):
    """Forecast horizon categories."""

    VERY_SHORT_TERM = "very_short_term"  # Minutes ahead
    SHORT_TERM = "short_term"            # Hours ahead
    MEDIUM_TERM = "medium_term"          # Days ahead
    LONG_TERM = "long_term"              # Weeks/months ahead


@dataclass
class DemandForecast:
    """A demand forecast result.

    Attributes:
        timestamps: List of forecast timestamps.
        predicted_load_mw: Predicted load in MW for each timestamp.
        confidence_lower_mw: Lower confidence bound in MW.
        confidence_upper_mw: Upper confidence bound in MW.
        horizon: Forecast horizon category.
        model_name: Name of the forecasting model used.
    """

    timestamps: List[datetime]
    predicted_load_mw: List[float]
    confidence_lower_mw: List[float]
    confidence_upper_mw: List[float]
    horizon: ForecastHorizon
    model_name: str

    def __post_init__(self) -> None:
        n = len(self.timestamps)
        if not (n == len(self.predicted_load_mw) == len(self.confidence_lower_mw) == len(self.confidence_upper_mw)):
            raise ForecastError("All forecast arrays must have the same length")
        if n == 0:
            raise ForecastError("Forecast must contain at least one data point")
        for i in range(n):
            if self.confidence_lower_mw[i] > self.predicted_load_mw[i]:
                raise ForecastError(f"Lower bound > predicted at index {i}")
            if self.predicted_load_mw[i] > self.confidence_upper_mw[i]:
                raise ForecastError(f"Predicted > upper bound at index {i}")


class DemandForecaster(ABC):
    """Abstract base class for demand forecasting models."""

    @abstractmethod
    def forecast(
        self,
        historical_load_mw: List[float],
        historical_timestamps: List[datetime],
        horizon: ForecastHorizon,
        steps_ahead: int,
    ) -> DemandForecast:
        """Generate a demand forecast.

        Args:
            historical_load_mw: Historical load data in MW.
            historical_timestamps: Timestamps for historical data.
            horizon: Forecast horizon category.
            steps_ahead: Number of steps to forecast ahead.

        Returns:
            A DemandForecast instance.

        Raises:
            ForecastError: If forecasting fails.
        """
        ...

    @abstractmethod
    def update(self, actual_load_mw: float, timestamp: datetime) -> None:
        """Update the model with actual observed data.

        Args:
            actual_load_mw: Actual load in MW.
            timestamp: Timestamp of the observation.
        """
        ...


class ExponentialSmoothingForecaster(DemandForecaster):
    """Holt-Winters exponential smoothing forecaster for demand prediction.

    Uses double exponential smoothing (level + trend) for short-term
    demand forecasting. Suitable for real-time grid balancing.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.1,
        confidence_level: float = 0.95,
    ) -> None:
        """Initialize the forecaster.

        Args:
            alpha: Level smoothing factor in (0, 1).
            beta: Trend smoothing factor in (0, 1).
            confidence_level: Confidence level for prediction intervals.

        Raises:
            ValueError: If parameters are out of range.
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not 0 < beta < 1:
            raise ValueError(f"beta must be in (0, 1), got {beta}")
        if not 0 < confidence_level < 1:
            raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

        self._alpha = alpha
        self._beta = beta
        self._confidence_level = confidence_level
        self._level: Optional[float] = None
        self._trend: float = 0.0
        self._residuals: List[float] = []
        self._last_timestamp: Optional[datetime] = None
        self._mean_interval_minutes: float = 60.0

    def _estimate_interval(self, timestamps: List[datetime]) -> float:
        """Estimate the mean time interval between observations in minutes."""
        if len(timestamps) < 2:
            return 60.0
        intervals = [
            (timestamps[i + 1] - timestamps[i]).total_seconds() / 60.0
            for i in range(len(timestamps) - 1)
        ]
        return max(1.0, sum(intervals) / len(intervals))

    def forecast(
        self,
        historical_load_mw: List[float],
        historical_timestamps: List[datetime],
        horizon: ForecastHorizon,
        steps_ahead: int,
    ) -> DemandForecast:
        """Generate demand forecast using double exponential smoothing.

        Args:
            historical_load_mw: Historical load data in MW.
            historical_timestamps: Timestamps for historical data.
            horizon: Forecast horizon category.
            steps_ahead: Number of steps to forecast ahead.

        Returns:
            A DemandForecast instance.

        Raises:
            ForecastError: If insufficient data or invalid parameters.
        """
        if len(historical_load_mw) < 2:
            raise ForecastError("Need at least 2 historical data points")
        if len(historical_load_mw) != len(historical_timestamps):
            raise ForecastError("Load and timestamp arrays must have the same length")
        if steps_ahead < 1:
            raise ForecastError("steps_ahead must be >= 1")

        self._mean_interval_minutes = self._estimate_interval(historical_timestamps)

        # Initialize level and trend
        self._level = historical_load_mw[0]
        self._trend = historical_load_mw[1] - historical_load_mw[0]
        self._residuals.clear()

        # Fit on historical data
        for i in range(1, len(historical_load_mw)):
            observed = historical_load_mw[i]
            predicted = self._level + self._trend
            residual = observed - predicted
            self._residuals.append(residual)

            old_level = self._level
            self._level = self._alpha * observed + (1 - self._alpha) * (self._level + self._trend)
            self._trend = self._beta * (self._level - old_level) + (1 - self._beta) * self._trend

        if self._level is None:
            raise ForecastError("Failed to initialize level")

        # Compute prediction interval width from residuals
        if self._residuals:
            std_residual = math.sqrt(sum(r ** 2 for r in self._residuals) / len(self._residuals))
        else:
            std_residual = 0.0

        # Z-score for confidence level (approximate)
        z_score = 1.96 if self._confidence_level >= 0.95 else 1.645

        # Generate forecasts
        last_ts = historical_timestamps[-1]
        timestamps: List[datetime] = []
        predicted: List[float] = []
        lower: List[float] = []
        upper: List[float] = []

        for step in range(1, steps_ahead + 1):
            ts = last_ts + timedelta(minutes=self._mean_interval_minutes * step)
            forecast_value = self._level + step * self._trend
            forecast_value = max(0.0, forecast_value)  # Load can't be negative

            # Widen interval with horizon
            interval_width = z_score * std_residual * math.sqrt(step)

            timestamps.append(ts)
            predicted.append(forecast_value)
            lower.append(max(0.0, forecast_value - interval_width))
            upper.append(forecast_value + interval_width)

        return DemandForecast(
            timestamps=timestamps,
            predicted_load_mw=predicted,
            confidence_lower_mw=lower,
            confidence_upper_mw=upper,
            horizon=horizon,
            model_name="ExponentialSmoothing",
        )

    def update(self, actual_load_mw: float, timestamp: datetime) -> None:
        """Update the model with actual observed data.

        Args:
            actual_load_mw: Actual load in MW.
            timestamp: Timestamp of the observation.
        """
        if self._level is None:
            self._level = actual_load_mw
            return

        predicted = self._level + self._trend
        residual = actual_load_mw - predicted
        self._residuals.append(residual)
        # Keep only last 100 residuals for efficiency
        if len(self._residuals) > 100:
            self._residuals = self._residuals[-100:]

        old_level = self._level
        self._level = self._alpha * actual_load_mw + (1 - self._alpha) * (self._level + self._trend)
        self._trend = self._beta * (self._level - old_level) + (1 - self._beta) * self._trend
        self._last_timestamp = timestamp


class MovingAverageForecaster(DemandForecaster):
    """Simple moving average forecaster for baseline demand prediction."""

    def __init__(self, window_size: int = 24, confidence_level: float = 0.95) -> None:
        """Initialize the moving average forecaster.

        Args:
            window_size: Number of past observations to average.
            confidence_level: Confidence level for prediction intervals.
        """
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self._window_size = window_size
        self._confidence_level = confidence_level
        self._history: List[float] = []
        self._last_timestamp: Optional[datetime] = None
        self._mean_interval_minutes: float = 60.0

    def forecast(
        self,
        historical_load_mw: List[float],
        historical_timestamps: List[datetime],
        horizon: ForecastHorizon,
        steps_ahead: int,
    ) -> DemandForecast:
        """Generate demand forecast using moving average.

        Args:
            historical_load_mw: Historical load data in MW.
            historical_timestamps: Timestamps for historical data.
            horizon: Forecast horizon category.
            steps_ahead: Number of steps to forecast ahead.

        Returns:
            A DemandForecast instance.
        """
        if len(historical_load_mw) < self._window_size:
            raise ForecastError(
                f"Need at least {self._window_size} historical data points, "
                f"got {len(historical_load_mw)}"
            )
        if steps_ahead < 1:
            raise ForecastError("steps_ahead must be >= 1")

        # Estimate interval
        if len(historical_timestamps) >= 2:
            intervals = [
                (historical_timestamps[i + 1] - historical_timestamps[i]).total_seconds() / 60.0
                for i in range(len(historical_timestamps) - 1)
            ]
            self._mean_interval_minutes = max(1.0, sum(intervals) / len(intervals))

        self._history = list(historical_load_mw[-self._window_size:])

        # Compute mean and std from window
        mean_val = sum(self._history) / len(self._history)
        variance = sum((x - mean_val) ** 2 for x in self._history) / len(self._history)
        std_val = math.sqrt(variance)

        z_score = 1.96 if self._confidence_level >= 0.95 else 1.645

        last_ts = historical_timestamps[-1]
        timestamps: List[datetime] = []
        predicted: List[float] = []
        lower: List[float] = []
        upper: List[float] = []

        for step in range(1, steps_ahead + 1):
            ts = last_ts + timedelta(minutes=self._mean_interval_minutes * step)
            interval_width = z_score * std_val * math.sqrt(step)

            timestamps.append(ts)
            predicted.append(max(0.0, mean_val))
            lower.append(max(0.0, mean_val - interval_width))
            upper.append(mean_val + interval_width)

        return DemandForecast(
            timestamps=timestamps,
            predicted_load_mw=predicted,
            confidence_lower_mw=lower,
            confidence_upper_mw=upper,
            horizon=horizon,
            model_name="MovingAverage",
        )

    def update(self, actual_load_mw: float, timestamp: datetime) -> None:
        """Update history with actual observation.

        Args:
            actual_load_mw: Actual load in MW.
            timestamp: Timestamp of the observation.
        """
        self._history.append(actual_load_mw)
        if len(self._history) > self._window_size:
            self._history = self._history[-self._window_size:]
        self._last_timestamp = timestamp


# ---------------------------------------------------------------------------
# Frequency Regulation
# ---------------------------------------------------------------------------

@dataclass
class FrequencyRegulationAction:
    """A frequency regulation control action.

    Attributes:
        timestamp: When the action was computed.
        frequency_hz: Current system frequency in Hz.
        nominal_frequency_hz: Nominal frequency in Hz.
        deviation_hz: Frequency deviation in Hz.
        required_correction_mw: Required power correction in MW.
        generator_adjustments: Dict of gen_id -> power adjustment in MW.
        load_shed_mw: Amount of load to shed in MW (if deficit).
        is_stable: Whether the frequency is within acceptable bounds.
    """

    timestamp: datetime
    frequency_hz: float
    nominal_frequency_hz: float
    deviation_hz: float
    required_correction_mw: float
    generator_adjustments: Dict[str, float]
    load_shed_mw: float
    is_stable: bool


class FrequencyRegulator:
    """Automatic Generation Control (AGC) for frequency regulation.

    Implements a PI controller to maintain system frequency at nominal
    by adjusting generator setpoints.
    """

    def __init__(
        self,
        kp: float = 0.5,
        ki: float = 0.1,
        deadband_hz: float = 0.02,
        max_correction_mw: float = 100.0,
    ) -> None:
        """Initialize the frequency regulator.

        Args:
            kp: Proportional gain.
            ki: Integral gain.
            deadband_hz: Frequency deadband in Hz (no action within this range).
            max_correction_mw: Maximum total correction in MW.

        Raises:
            ValueError: If parameters are invalid.
        """
        if kp < 0:
            raise ValueError("kp must be non-negative")
        if ki < 0:
            raise ValueError("ki must be non-negative")
        if deadband_hz < 0:
            raise ValueError("deadband_hz must be non-negative")
        if max_correction_mw <= 0:
            raise ValueError("max_correction_mw must be positive")

        self._kp = kp
        self._ki = ki
        self._deadband_hz = deadband_hz
        self._max_correction_mw = max_correction_mw
        self._integral_error: float = 0.0
        self._last_timestamp: Optional[datetime] = None

    def reset(self) -> None:
        """Reset the integral error accumulator."""
        self._integral_error = 0.0
        self._last_timestamp = None

    def compute_action(
        self,
        snapshot: GridSnapshot,
        timestamp: Optional[datetime] = None,
    ) -> FrequencyRegulationAction:
        """Compute frequency regulation action.

        Args:
            snapshot: Current grid state snapshot.
            timestamp: Optional timestamp (defaults to snapshot.timestamp).

        Returns:
            A FrequencyRegulationAction with generator adjustments.

        Raises:
            RegulationError: If the snapshot is invalid.
        """
        snapshot.validate()
        ts = timestamp or snapshot.timestamp

        deviation = snapshot.frequency_deviation_hz

        # Check deadband
        if abs(deviation) <= self._deadband_hz:
            return FrequencyRegulationAction(
                timestamp=ts,
                frequency_hz=snapshot.frequency_hz,
                nominal_frequency_hz=snapshot.nominal_frequency_hz,
                deviation_hz=deviation,
                required_correction_mw=0.0,
                generator_adjustments={},
                load_shed_mw=0.0,
                is_stable=True,
            )

        # Update integral error with anti-windup
        if self._last_timestamp is not None:
            dt_minutes = (ts - self._last_timestamp).total_seconds() / 60.0
            if dt_minutes > 0:
                self._integral_error += deviation * dt_minutes
                # Anti-windup: clamp integral
                max_integral = self._max_correction_mw / max(self._ki, 1e-6)
                self._integral_error = max(-max_integral, min(max_integral, self._integral_error))

        self._last_timestamp = ts

        # PI control law
        correction = -(self._kp * deviation + self._ki * self._integral_error)
        correction = max(-self._max_correction_mw, min(self._max_correction_mw, correction))

        # Distribute correction among online generators
        adjustments = self._distribute_correction(snapshot, correction)

        # If deficit and generators can't compensate, shed load
        load_shed = 0.0
        if correction < 0 and abs(correction) > sum(abs(v) for v in adjustments.values()):
            load_shed = min(abs(correction) - sum(abs(v) for v in adjustments.values()), snapshot.total_load_mw * 0.1)

        return FrequencyRegulationAction(
            timestamp=ts,
            frequency_hz=snapshot.frequency_hz,
            nominal_frequency_hz=snapshot.nominal_frequency_hz,
            deviation_hz=deviation,
            required_correction_mw=correction,
            generator_adjustments=adjustments,
            load_shed_mw=load_shed,
            is_stable=False,
        )

    def _distribute_correction(
        self,
        snapshot: GridSnapshot,
        correction_mw: float,
    ) -> Dict[str, float]:
        """Distribute power correction among available generators.

        Args:
            snapshot: Current grid state.
            correction_mw: Total correction needed in MW (positive = increase gen).

        Returns:
            Dict of gen_id -> power adjustment in MW.
        """
        online_gens = [g for g in snapshot.generators.values() if g.is_online]
        if not online_gens:
            return {}

        # Calculate available headroom for each generator
        headrooms: List[Tuple[str, float]] = []
        total_headroom = 0.0

        for gen in online_gens:
            if correction_mw > 0:
                # Need to increase generation
                headroom = gen.p_max_mw - gen.current_output_mw
            else:
                # Need to decrease generation
                headroom = gen.current_output_mw - gen.p_min_mw
            headroom = max(0.0, headroom)
            headrooms.append((gen.gen_id, headroom))
            total_headroom += headroom

        if total_headroom <= 0:
            return {}

        # Distribute proportionally to headroom
        adjustments: Dict[str, float] = {}
        for gen_id, headroom in headrooms:
            share = (headroom / total_headroom) * correction_mw
            adjustments[gen_id] = share

        return adjustments


# ---------------------------------------------------------------------------
# Voltage Control
# ---------------------------------------------------------------------------

@dataclass
class VoltageControlAction:
    """A voltage control action.

    Attributes:
        timestamp: When the action was computed.
        bus_voltages: Dict of bus_id -> voltage in p.u.
        violations: List of (bus_id, voltage_pu, limit_type) tuples.
        generator_q_adjustments: Dict of gen_id -> reactive power adjustment in MVAr.
        capacitor_bank_actions: Dict of capacitor_bank_id -> switch action (True=on, False=off).
        transformer_tap_actions: Dict of transformer_id -> tap change (+1 or -1).
        is_stable: Whether all voltages are within bounds.
    """

    timestamp: datetime
    bus_voltages: Dict[str, float]
    violations: List[Tuple[str, float, str]]
    generator_q_adjustments: Dict[str, float]
    capacitor_bank_actions: Dict[str, bool]
    transformer_tap_actions: Dict[str, int]
    is_stable: bool


@dataclass
class CapacitorBank:
    """A switched capacitor bank for reactive power support.

    Attributes:
        bank_id: Unique identifier.
        bus_id: Bus where the bank is connected.
        capacity_mvar: Reactive power capacity in MVAr when switched on.
        is_on: Whether the bank is currently switched on.
    """

    bank_id: str
    bus_id: str
    capacity_mvar: float = 10.0
    is_on: bool = False


@dataclass
class TransformerTap:
    """A tap-changing transformer.

    Attributes:
        transformer_id: Unique identifier.
        current_tap: Current tap position.
        min_tap: Minimum tap position.
        max_tap: Maximum tap position.
        tap_step_pu: Voltage change per tap in p.u.
    """

    transformer_id: str
    current_tap: int = 0
    min_tap: int = -16
    max_tap: int = 16
    tap_step_pu: float = 0.00625


class VoltageController:
    """Voltage control using reactive power compensation.

    Maintains bus voltages within acceptable bounds by adjusting
    generator reactive output, switching capacitor banks, and
    changing transformer taps.
    """

    def __init__(
        self,
        v_min_pu: float = 0.95,
        v_max_pu: float = 1.05,
        control_deadband_pu: float = 0.01,
    ) -> None:
        """Initialize the voltage controller.

        Args:
            v_min_pu: Minimum acceptable voltage in p.u.
            v_max_pu: Maximum acceptable voltage in p.u.
            control_deadband_pu: Deadband around nominal in p.u.

        Raises:
            ValueError: If voltage limits are invalid.
        """
        if v_min_pu >= v_max_pu:
            raise ValueError(f"v_min_pu ({v_min_pu}) must be < v_max_pu ({v_max_pu})")
        if v_min_pu <= 0:
            raise ValueError("v_min_pu must be positive")
        if control_deadband_pu < 0:
            raise ValueError("control_deadband_pu must be non-negative")

        self._v_min = v_min_pu
        self._v_max = v_max_pu
        self._deadband = control_deadband_pu

    def compute_action(
        self,
        snapshot: GridSnapshot,
        capacitor_banks: Optional[Dict[str, CapacitorBank]] = None,
        transformers: Optional[Dict[str, TransformerTap]] = None,
        timestamp: Optional[datetime] = None,
    ) -> VoltageControlAction:
        """Compute voltage control actions.

        Args:
            snapshot: Current grid state snapshot.
            capacitor_banks: Optional capacitor banks for switching.
            transformers: Optional tap-changing transformers.
            timestamp: Optional timestamp.

        Returns:
            A VoltageControlAction with control decisions.

        Raises:
            RegulationError: If the snapshot is invalid.
        """
        snapshot.validate()
        ts = timestamp or snapshot.timestamp
        capacitor_banks = capacitor_banks or {}
        transformers = transformers or {}

        bus_voltages = {bus_id: bus.voltage_pu for bus_id, bus in snapshot.buses.items()}
        violations: List[Tuple[str, float, str]] = []

        # Identify violations
        for bus_id, v in bus_voltages.items():
            if v < self._v_min:
                violations.append((bus_id, v, "undervoltage"))
            elif v > self._v_max:
                violations.append((bus_id, v, "overvoltage"))

        gen_q_adjustments: Dict[str, float] = {}
        cap_actions: Dict[str, bool] = {}
        tap_actions: Dict[str, int] = {}

        if not violations:
            return VoltageControlAction(
                timestamp=ts,
                bus_voltages=bus_voltages,
                violations=[],
                generator_q_adjustments={},
                capacitor_bank_actions={},
                transformer_tap_actions={},
                is_stable=True,
            )

        # Sort violations by severity (distance from bounds)
        violations.sort(key=lambda x: abs(x[1] - 1.0), reverse=True)

        for bus_id, v, vtype in violations:
            deviation = 1.0 - v  # Positive for undervoltage, negative for overvoltage

            if abs(deviation) <= self._deadband:
                continue

            # Try generator reactive power adjustment first
            gens_at_bus = [
                g for g in snapshot.generators.values()
                if g.bus_id == bus_id and g.is_online
            ]

            for gen in gens_at_bus:
                if deviation > 0:
                    # Undervoltage: inject reactive power
                    q_headroom = gen.q_max_mvar - 0  # Simplified: assume current Q = 0
                    q_adjust = min(deviation * 50.0, q_headroom)  # Gain factor
                    if q_adjust > 0.1:
                        gen_q_adjustments[gen.gen_id] = q_adjust
                        break
                else:
                    # Overvoltage: absorb reactive power
                    q_headroom = 0 - gen.q_min_mvar
                    q_adjust = max(deviation * 50.0, -q_headroom)
                    if abs(q_adjust) > 0.1:
                        gen_q_adjustments[gen.gen_id] = q_adjust
                        break

            # If no generator available, try capacitor banks
            if bus_id not in [g.bus_id for g in snapshot.generators.values() if g.gen_id in gen_q_adjustments]:
                banks_at_bus = [
                    cb for cb in capacitor_banks.values() if cb.bus_id == bus_id
                ]
                for bank in banks_at_bus:
                    if deviation > 0 and not bank.is_on:
                        cap_actions[bank.bank_id] = True
                        break
                    elif deviation < 0 and bank.is_on:
                        cap_actions[bank.bank_id] = False
                        break

            # Try transformer tap changes
            for tx in transformers.values():
                if deviation > 0 and tx.current_tap < tx.max_tap:
                    tap_actions[tx.transformer_id] = 1
                    break
                elif deviation < 0 and tx.current_tap > tx.min_tap:
                    tap_actions[tx.transformer_id] = -1
                    break

        return VoltageControlAction(
            timestamp=ts,
            bus_voltages=bus_voltages,
            violations=violations,
            generator_q_adjustments=gen_q_adjustments,
            capacitor_bank_actions=cap_actions,
            transformer_tap_actions=tap_actions,
            is_stable=len(violations) == 0,
        )


# ---------------------------------------------------------------------------
# Autonomous Grid Balancing
# ---------------------------------------------------------------------------

@dataclass
class BalancingAction:
    """A grid balancing decision.

    Attributes:
        timestamp: When the decision was made.
        imbalance_mw: Detected power imbalance in MW.
        generator_setpoints: Dict of gen_id -> new setpoint in MW.
        load_shed_decisions: List of (load_id, amount_mw) to shed.
        load_shift_decisions: List of (load_id, amount_mw, target_timestamp) to shift.
        storage_charge_mw: Power to charge storage in MW (negative = discharge).
        estimated_cost_per_hour: Estimated hourly cost in $.
        is_balanced: Whether the grid is balanced after actions.
    """

    timestamp: datetime
    imbalance_mw: float
    generator_setpoints: Dict[str, float]
    load_shed_decisions: List[Tuple[str, float]]
    load_shift_decisions: List[Tuple[str, float, datetime]]
    storage_charge_mw: float
    estimated_cost_per_hour: float
    is_balanced: bool


class GridBalancer:
    """Autonomous grid balancing using economic dispatch.

    Balances generation and load using a merit-order approach
    with optional storage and demand response.
    """

    def __init__(
        self,
        storage_capacity_mwh: float = 100.0,
        storage_max_power_mw: float = 50.0,
        storage_efficiency: float = 0.9,
        reserve_margin: float = 0.1,
    ) -> None:
        """Initialize the grid balancer.

        Args:
            storage_capacity_mwh: Storage energy capacity in MWh.
            storage_max_power_mw: Storage max charge/discharge power in MW.
            storage_efficiency: Round-trip efficiency in (0, 1].
            reserve_margin: Required reserve margin as fraction of load.

        Raises:
            ValueError: If parameters are invalid.
        """
        if storage_capacity_mwh <= 0:
            raise ValueError("storage_capacity_mwh must be positive")
        if storage_max_power_mw <= 0:
            raise ValueError("storage_max_power_mw must be positive")
        if not 0 < storage_efficiency <= 1:
            raise ValueError("storage_efficiency must be in (0, 1]")
        if not 0 <= reserve_margin < 1:
            raise ValueError("reserve_margin must be in [0, 1)")

        self._storage_capacity_mwh = storage_capacity_mwh
        self._storage_max_power_mw = storage_max_power_mw
        self._storage_efficiency = storage_efficiency
        self._reserve_margin = reserve_margin
        self._storage_soc_mwh: float = storage_capacity_mwh * 0.5  # Start at 50% SOC

    @property
    def storage_soc_mwh(self) -> float:
        """Current storage state of charge in MWh."""
        return self._storage_soc_mwh

    def set_storage_soc(self, soc_mwh: float) -> None:
        """Set the storage state of charge.

        Args:
            soc_mwh: State of charge in MWh.

        Raises:
            ValueError: If SOC is out of bounds.
        """
        if not 0 <= soc_mwh <= self._storage_capacity_mwh:
            raise ValueError(
                f"SOC must be in [0, {self._storage_capacity_mwh}], got {soc_mwh}"
            )
        self._storage_soc_mwh = soc_mwh

    def compute_balancing_action(
        self,
        snapshot: GridSnapshot,
        forecast: Optional[DemandForecast] = None,
        timestamp: Optional[datetime] = None,
    ) -> BalancingAction:
        """Compute grid balancing action using economic dispatch.

        Args:
            snapshot: Current grid state snapshot.
            forecast: Optional demand forecast for look-ahead.
            timestamp: Optional timestamp.

        Returns:
            A BalancingAction with generator setpoints and control decisions.

        Raises:
            GridStateError: If the snapshot is invalid.
        """
        snapshot.validate()
        ts = timestamp or snapshot.timestamp

        imbalance = snapshot.power_imbalance_mw
        total_load = snapshot.total_load_mw
        required_reserve = total_load * self._reserve_margin

        # Determine target generation
        if forecast and forecast.predicted_load_mw:
            # Use forecast for look-ahead
            target_load = forecast.predicted_load_mw[0]
        else:
            target_load = total_load

        target_generation = target_load + required_reserve

        # Economic dispatch: merit order
        setpoints = self._economic_dispatch(snapshot, target_generation)

        # Calculate total dispatched generation
        total_dispatched = sum(setpoints.values())

        # Storage action
        storage_action = 0.0
        if total_dispatched < target_load:
            # Deficit: discharge storage
            deficit = target_load - total_dispatched
            max_discharge = min(
                self._storage_max_power_mw,
                self._storage_soc_mwh * self._storage_efficiency,
            )
            storage_action = -min(deficit, max_discharge)
            self._storage_soc_mwh += storage_action / self._storage_efficiency
            self._storage_soc_mwh = max(0.0, self._storage_soc_mwh)
        elif total_dispatched > target_load * 1.05:
            # Surplus: charge storage
            surplus = total_dispatched - target_load
            max_charge = min(
                self._storage_max_power_mw,
                (self._storage_capacity_mwh - self._storage_soc_mwh),
            )
            storage_action = min(surplus, max_charge)
            self._storage_soc_mwh += storage_action * self._storage_efficiency
            self._storage_soc_mwh = min(self._storage_capacity_mwh, self._storage_soc_mwh)

        # Load shedding (last resort)
        load_shed: List[Tuple[str, float]] = []
        remaining_deficit = target_load - total_dispatched + storage_action
        if remaining_deficit > 0:
            controllable_loads = sorted(
                [l for l in snapshot.loads.values() if l.is_controllable],
                key=lambda l: l.priority,
                reverse=True,  # Shed lowest priority first
            )
            for load in controllable_loads:
                if remaining_deficit <= 0:
                    break
                shed_amount = min(load.active_power_mw, remaining_deficit)
                load_shed.append((load.load_id, shed_amount))
                remaining_deficit -= shed_amount

        # Load shifting (for surplus)
        load_shift: List[Tuple[str, float, datetime]] = []
        surplus = total_dispatched + storage_action - target_load
        if surplus > 0:
            shiftable_loads = [l for l in snapshot.loads.values() if l.is_controllable]
            for load in shiftable_loads:
                if surplus <= 0:
                    break
                shift_amount = min(load.active_power_mw * 0.5, surplus)
                load_shift.append((load.load_id, shift_amount, ts + timedelta(hours=1)))
                surplus -= shift_amount

        # Estimate cost
        estimated_cost = self._estimate_cost(snapshot, setpoints)

        is_balanced = abs(remaining_deficit) < total_load * 0.01 if total_load > 0 else True

        return BalancingAction(
            timestamp=ts,
            imbalance_mw=imbalance,
            generator_setpoints=setpoints,
            load_shed_decisions=load_shed,
            load_shift_decisions=load_shift,
            storage_charge_mw=storage_action,
            estimated_cost_per_hour=estimated_cost,
            is_balanced=is_balanced,
        )

    def _economic_dispatch(
        self,
        snapshot: GridSnapshot,
        target_generation_mw: float,
    ) -> Dict[str, float]:
        """Compute economic dispatch using merit order.

        Args:
            snapshot: Current grid state.
            target_generation_mw: Target total generation in MW.

        Returns:
            Dict of gen_id -> setpoint in MW.
        """
        online_gens = [g for g in snapshot.generators.values() if g.is_online]
        if not online_gens:
            return {}

        # Sort by marginal cost (merit order)
        sorted_gens = sorted(online_gens, key=lambda g: g.marginal_cost_per_mwh)

        setpoints: Dict[str, float] = {}
        remaining = target_generation_mw

        for gen in sorted_gens:
            if remaining <= 0:
                setpoints[gen.gen_id] = gen.p_min_mw
                continue

            # Allocate up to max capacity
            allocation = min(gen.p_max_mw, remaining)
            allocation = max(gen.p_min_mw, allocation)
            setpoints[gen.gen_id] = allocation
            remaining -= allocation

        return setpoints

    def _estimate_cost(
        self,
        snapshot: GridSnapshot,
        setpoints: Dict[str, float],
    ) -> float:
        """Estimate hourly operating cost.

        Args:
            snapshot: Current grid state.
            setpoints: Generator setpoints.

        Returns:
            Estimated cost in $/hour.
        """
        cost = 0.0
        for gen_id, setpoint in setpoints.items():
            if gen_id in snapshot.generators:
                gen = snapshot.generators[gen_id]
                cost += setpoint * gen.marginal_cost_per_mwh
        return cost


# ---------------------------------------------------------------------------
# Grid Controller (orchestrator)
# ---------------------------------------------------------------------------

class SmartGridController:
    """High-level smart grid controller orchestrating all subsystems.

    Coordinates frequency regulation, voltage control, demand forecasting,
    and autonomous balancing into a unified control loop.
    """

    def __init__(
        self,
        frequency_regulator: Optional[FrequencyRegulator] = None,
        voltage_controller: Optional[VoltageController] = None,
        grid_balancer: Optional[GridBalancer] = None,
        forecaster: Optional[DemandForecaster] = None,
    ) -> None:
        """Initialize the smart grid controller.

        Args:
            frequency_regulator: Frequency regulation subsystem.
            voltage_controller: Voltage control subsystem.
            grid_balancer: Grid balancing subsystem.
            forecaster: Demand forecasting subsystem.
        """
        self._freq_regulator = frequency_regulator or FrequencyRegulator()
        self._voltage_controller = voltage_controller or VoltageController()
        self._grid_balancer = grid_balancer or GridBalancer()
        self._forecaster = forecaster or ExponentialSmoothingForecaster()
        self._capacitor_banks: Dict[str, CapacitorBank] = {}
        self._transformers: Dict[str, TransformerTap] = {}
        self._historical_load: List[float] = []
        self._historical_timestamps: List[datetime] = []
        self._last_snapshot: Optional[GridSnapshot] = None

    @property
    def capacitor_banks(self) -> Dict[str, CapacitorBank]:
        """Capacitor banks under control."""
        return self._capacitor_banks

    @property
    def transformers(self) -> Dict[str, TransformerTap]:
        """Transformers under control."""
        return self._transformers

    def add_capacitor_bank(self, bank: CapacitorBank) -> None:
        """Add a capacitor bank for voltage control.

        Args:
            bank: The capacitor bank to add.
        """
        self._capacitor_banks[bank.bank_id] = bank

    def add_transformer(self, transformer: TransformerTap) -> None:
        """Add a tap-changing transformer for voltage control.

        Args:
            transformer: The transformer to add.
        """
        self._transformers[transformer.transformer_id] = transformer

    def process_snapshot(self, snapshot: GridSnapshot) -> Dict[str, Any]:
        """Process a grid snapshot through all control subsystems.

        Args:
            snapshot: Current grid state snapshot.

        Returns:
            Dict with keys 'frequency_action', 'voltage_action',
            'balancing_action', and 'forecast'.

        Raises:
            GridStateError: If the snapshot is invalid.
        """
        snapshot.validate()
        self._last_snapshot = snapshot

        # Update historical data for forecasting
        self._historical_load.append(snapshot.total_load_mw)
        self._historical_timestamps.append(snapshot.timestamp)
        if len(self._historical_load) > 1000:
            self._historical_load = self._historical_load[-1000:]
            self._historical_timestamps = self._historical_timestamps[-1000:]

        # Frequency regulation
        freq_action = self._freq_regulator.compute_action(snapshot)

        # Voltage control
        voltage_action = self._voltage_controller.compute_action(
            snapshot, self._capacitor_banks, self._transformers
        )

        # Demand forecast
        forecast: Optional[DemandForecast] = None
        if len(self._historical_load) >= 2:
            try:
                forecast = self._forecaster.forecast(
                    self._historical_load,
                    self._historical_timestamps,
                    ForecastHorizon.SHORT_TERM,
                    steps_ahead=12,
                )
            except ForecastError as e:
                logger.warning("Forecast failed: %s", e)

        # Grid balancing
        balancing_action = self._grid_balancer.compute_balancing_action(
            snapshot, forecast
        )

        return {
            "frequency_action": freq_action,
            "voltage_action": voltage_action,
            "balancing_action": balancing_action,
            "forecast": forecast,
        }

    def get_historical_data(self) -> Tuple[List[float], List[datetime]]:
        """Get historical load data.

        Returns:
            Tuple of (historical_load_mw, historical_timestamps).
        """
        return list(self._historical_load), list(self._historical_timestamps)
