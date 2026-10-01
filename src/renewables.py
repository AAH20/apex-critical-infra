"""Renewables module for Apex Critical Infrastructure.

Provides solar/wind forecasting, grid storage optimization, and curtailment
management for agentic AI decision-making in renewable energy systems.
"""

from __future__ import annotations

import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class RenewablesError(Exception):
    """Base exception for renewables module."""


class ForecastError(RenewablesError):
    """Raised when renewable forecasting fails."""


class StorageError(RenewablesError):
    """Raised when storage optimization fails."""


class CurtailmentError(RenewablesError):
    """Raised when curtailment management fails."""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class RenewableType(Enum):
    """Types of renewable energy sources."""

    SOLAR = "solar"
    WIND = "wind"
    HYDRO = "hydro"
    GEOTHERMAL = "geothermal"


class StorageType(Enum):
    """Types of grid storage systems."""

    LITHIUM_ION = "lithium_ion"
    FLOW_BATTERY = "flow_battery"
    COMPRESSED_AIR = "compressed_air"
    PUMPED_HYDRO = "pumped_hydro"
    HYDROGEN = "hydrogen"


class CurtailmentReason(Enum):
    """Reasons for curtailment."""

    GRID_CONSTRAINT = "grid_constraint"
    OVERSUPPLY = "oversupply"
    FORECAST_ERROR = "forecast_error"
    MAINTENANCE = "maintenance"
    MARKET_SIGNAL = "market_signal"


@dataclass(frozen=True)
class WeatherConditions:
    """Weather conditions relevant to renewable generation.

    Attributes:
        solar_irradiance_wm2: Solar irradiance in W/m².
        wind_speed_ms: Wind speed in m/s.
        wind_direction_deg: Wind direction in degrees.
        temperature_c: Temperature in Celsius.
        cloud_cover_fraction: Cloud cover fraction in [0, 1].
        humidity_fraction: Relative humidity in [0, 1].
        pressure_hpa: Atmospheric pressure in hPa.
    """

    solar_irradiance_wm2: float = 0.0
    wind_speed_ms: float = 0.0
    wind_direction_deg: float = 0.0
    temperature_c: float = 20.0
    cloud_cover_fraction: float = 0.0
    humidity_fraction: float = 0.5
    pressure_hpa: float = 1013.25

    def __post_init__(self) -> None:
        if self.solar_irradiance_wm2 < 0:
            raise RenewablesError("Solar irradiance must be non-negative")
        if self.wind_speed_ms < 0:
            raise RenewablesError("Wind speed must be non-negative")
        if not 0 <= self.cloud_cover_fraction <= 1:
            raise RenewablesError("Cloud cover must be in [0, 1]")
        if not 0 <= self.humidity_fraction <= 1:
            raise RenewablesError("Humidity must be in [0, 1]")


@dataclass
class RenewableAsset:
    """A renewable energy generation asset.

    Attributes:
        asset_id: Unique identifier.
        asset_type: Type of renewable source.
        capacity_mw: Nameplate capacity in MW.
        current_output_mw: Current output in MW.
        location_lat: Latitude in degrees.
        location_lon: Longitude in degrees.
        elevation_m: Elevation in meters.
        is_online: Whether the asset is operational.
        availability_factor: Availability factor in [0, 1].
    """

    asset_id: str
    asset_type: RenewableType
    capacity_mw: float = 100.0
    current_output_mw: float = 0.0
    location_lat: float = 0.0
    location_lon: float = 0.0
    elevation_m: float = 0.0
    is_online: bool = True
    availability_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.capacity_mw <= 0:
            raise RenewablesError(f"Asset {self.asset_id}: capacity_mw must be positive")
        if not 0 <= self.availability_factor <= 1:
            raise RenewablesError(f"Asset {self.asset_id}: availability_factor must be in [0, 1]")
        self.current_output_mw = max(0.0, min(self.capacity_mw, self.current_output_mw))

    @property
    def capacity_factor(self) -> float:
        """Current capacity factor (output / capacity)."""
        if self.capacity_mw <= 0 or not self.is_online:
            return 0.0
        return self.current_output_mw / self.capacity_mw


@dataclass
class StorageSystem:
    """A grid-scale energy storage system.

    Attributes:
        storage_id: Unique identifier.
        storage_type: Type of storage technology.
        capacity_mwh: Energy capacity in MWh.
        max_power_mw: Maximum charge/discharge power in MW.
        round_trip_efficiency: Round-trip efficiency in (0, 1].
        current_soc_mwh: Current state of charge in MWh.
        min_soc_mwh: Minimum allowed state of charge in MWh.
        max_soc_mwh: Maximum allowed state of charge in MWh.
        is_online: Whether the system is operational.
        degradation_factor: Degradation factor in [0, 1] (1 = new).
    """

    storage_id: str
    storage_type: StorageType
    capacity_mwh: float = 100.0
    max_power_mw: float = 50.0
    round_trip_efficiency: float = 0.9
    current_soc_mwh: float = 50.0
    min_soc_mwh: float = 5.0
    max_soc_mwh: float = 95.0
    is_online: bool = True
    degradation_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.capacity_mwh <= 0:
            raise StorageError(f"Storage {self.storage_id}: capacity_mwh must be positive")
        if self.max_power_mw <= 0:
            raise StorageError(f"Storage {self.storage_id}: max_power_mw must be positive")
        if not 0 < self.round_trip_efficiency <= 1:
            raise StorageError(f"Storage {self.storage_id}: efficiency must be in (0, 1]")
        if not 0 <= self.min_soc_mwh < self.max_soc_mwh <= self.capacity_mwh:
            raise StorageError(f"Storage {self.storage_id}: invalid SOC bounds")
        if not 0 < self.degradation_factor <= 1:
            raise StorageError(f"Storage {self.storage_id}: degradation_factor must be in (0, 1]")
        self.current_soc_mwh = max(self.min_soc_mwh, min(self.max_soc_mwh, self.current_soc_mwh))

    @property
    def soc_fraction(self) -> float:
        """State of charge as a fraction of capacity."""
        return self.current_soc_mwh / self.capacity_mwh if self.capacity_mwh > 0 else 0.0

    @property
    def available_energy_mwh(self) -> float:
        """Available energy for discharge in MWh."""
        return max(0.0, self.current_soc_mwh - self.min_soc_mwh)

    @property
    def available_capacity_mwh(self) -> float:
        """Available capacity for charging in MWh."""
        return max(0.0, self.max_soc_mwh - self.current_soc_mwh)

    def charge(self, energy_mwh: float) -> float:
        """Charge the storage system.

        Args:
            energy_mwh: Energy to add in MWh (before efficiency loss).

        Returns:
            Actual energy stored in MWh.

        Raises:
            StorageError: If the storage is offline or energy is invalid.
        """
        if not self.is_online:
            raise StorageError(f"Storage {self.storage_id} is offline")
        if energy_mwh < 0:
            raise StorageError("Charge energy must be non-negative")

        effective_energy = energy_mwh * self.round_trip_efficiency
        actual = min(effective_energy, self.available_capacity_mwh)
        self.current_soc_mwh += actual
        return actual

    def discharge(self, energy_mwh: float) -> float:
        """Discharge the storage system.

        Args:
            energy_mwh: Energy to remove in MWh (before efficiency loss).

        Returns:
            Actual energy delivered in MWh.

        Raises:
            StorageError: If the storage is offline or energy is invalid.
        """
        if not self.is_online:
            raise StorageError(f"Storage {self.storage_id} is offline")
        if energy_mwh < 0:
            raise StorageError("Discharge energy must be non-negative")

        requested = energy_mwh / self.round_trip_efficiency
        actual = min(requested, self.available_energy_mwh)
        self.current_soc_mwh -= actual
        return actual * self.round_trip_efficiency


@dataclass
class RenewableForecast:
    """A renewable generation forecast.

    Attributes:
        asset_id: ID of the renewable asset.
        timestamps: List of forecast timestamps.
        predicted_output_mw: Predicted output in MW for each timestamp.
        confidence_lower_mw: Lower confidence bound in MW.
        confidence_upper_mw: Upper confidence bound in MW.
        weather: Predicted weather conditions.
        model_name: Name of the forecasting model used.
    """

    asset_id: str
    timestamps: List[datetime]
    predicted_output_mw: List[float]
    confidence_lower_mw: List[float]
    confidence_upper_mw: List[float]
    weather: List[WeatherConditions]
    model_name: str

    def __post_init__(self) -> None:
        n = len(self.timestamps)
        if not (n == len(self.predicted_output_mw) == len(self.confidence_lower_mw) == len(self.confidence_upper_mw) == len(self.weather)):
            raise ForecastError("All forecast arrays must have the same length")
        if n == 0:
            raise ForecastError("Forecast must contain at least one data point")
        for i in range(n):
            if self.confidence_lower_mw[i] > self.predicted_output_mw[i]:
                raise ForecastError(f"Lower bound > predicted at index {i}")
            if self.predicted_output_mw[i] > self.confidence_upper_mw[i]:
                raise ForecastError(f"Predicted > upper bound at index {i}")


@dataclass
class CurtailmentAction:
    """A curtailment decision.

    Attributes:
        timestamp: When the decision was made.
        asset_id: ID of the asset to curtail.
        current_output_mw: Current output in MW.
        target_output_mw: Target output after curtailment in MW.
        curtailed_mw: Amount curtailed in MW.
        reason: Reason for curtailment.
        duration_minutes: Expected duration in minutes.
        estimated_lost_revenue: Estimated lost revenue in $.
    """

    timestamp: datetime
    asset_id: str
    current_output_mw: float
    target_output_mw: float
    curtailed_mw: float
    reason: CurtailmentReason
    duration_minutes: float
    estimated_lost_revenue: float

    def __post_init__(self) -> None:
        if self.target_output_mw < 0:
            raise CurtailmentError("Target output must be non-negative")
        if self.target_output_mw > self.current_output_mw:
            raise CurtailmentError("Target output cannot exceed current output")
        self.curtailed_mw = self.current_output_mw - self.target_output_mw


# ---------------------------------------------------------------------------
# Solar Forecasting
# ---------------------------------------------------------------------------

class SolarForecaster(ABC):
    """Abstract base class for solar power forecasting."""

    @abstractmethod
    def forecast(
        self,
        asset: RenewableAsset,
        weather_forecast: List[WeatherConditions],
        timestamps: List[datetime],
    ) -> RenewableForecast:
        """Generate solar power forecast.

        Args:
            asset: The solar asset to forecast.
            weather_forecast: Predicted weather conditions.
            timestamps: Forecast timestamps.

        Returns:
            A RenewableForecast instance.

        Raises:
            ForecastError: If forecasting fails.
        """
        ...

    @abstractmethod
    def estimate_clear_sky_irradiance(
        self,
        timestamp: datetime,
        latitude: float,
        longitude: float,
    ) -> float:
        """Estimate clear-sky solar irradiance.

        Args:
            timestamp: Time of day.
            latitude: Latitude in degrees.
            longitude: Longitude in degrees.

        Returns:
            Clear-sky irradiance in W/m².
        """
        ...


class ClearSkySolarForecaster(SolarForecaster):
    """Solar forecaster using clear-sky model with cloud cover adjustment.

    Implements a simplified clear-sky solar model based on solar geometry
    and cloud cover attenuation.
    """

    SOLAR_CONSTANT_WM2: float = 1361.0  # Solar constant in W/m²

    def __init__(self, confidence_level: float = 0.9) -> None:
        """Initialize the solar forecaster.

        Args:
            confidence_level: Confidence level for prediction intervals.
        """
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        self._confidence_level = confidence_level

    def estimate_clear_sky_irradiance(
        self,
        timestamp: datetime,
        latitude: float,
        longitude: float,
    ) -> float:
        """Estimate clear-sky solar irradiance using solar geometry.

        Args:
            timestamp: Time of day.
            latitude: Latitude in degrees.
            longitude: Longitude in degrees.

        Returns:
            Clear-sky irradiance in W/m².
        """
        # Day of year
        day_of_year = timestamp.timetuple().tm_yday

        # Solar declination angle (approximate)
        declination = math.radians(23.45 * math.sin(math.radians((360 / 365) * (day_of_year - 81))))

        # Hour angle
        hour = timestamp.hour + timestamp.minute / 60.0
        hour_angle = math.radians(15 * (hour - 12))

        # Solar elevation
        lat_rad = math.radians(latitude)
        sin_elevation = (
            math.sin(lat_rad) * math.sin(declination)
            + math.cos(lat_rad) * math.cos(declination) * math.cos(hour_angle)
        )
        sin_elevation = max(0.0, sin_elevation)

        # Air mass (simplified)
        if sin_elevation > 0:
            air_mass = 1.0 / sin_elevation
        else:
            return 0.0

        # Clear-sky irradiance (simplified Bird model)
        irradiance = self.SOLAR_CONSTANT_WM2 * sin_elevation * math.exp(-0.000117 * air_mass * 1000)

        return max(0.0, irradiance)

    def forecast(
        self,
        asset: RenewableAsset,
        weather_forecast: List[WeatherConditions],
        timestamps: List[datetime],
    ) -> RenewableForecast:
        """Generate solar power forecast.

        Args:
            asset: The solar asset to forecast.
            weather_forecast: Predicted weather conditions.
            timestamps: Forecast timestamps.

        Returns:
            A RenewableForecast instance.
        """
        if not weather_forecast:
            raise ForecastError("Weather forecast cannot be empty")
        if len(weather_forecast) != len(timestamps):
            raise ForecastError("Weather forecast and timestamps must have the same length")
        if asset.asset_type != RenewableType.SOLAR:
            raise ForecastError(f"Asset {asset.asset_id} is not a solar asset")

        predicted: List[float] = []
        lower: List[float] = []
        upper: List[float] = []

        # Panel efficiency and temperature coefficient
        panel_efficiency = 0.20
        temp_coefficient = -0.004  # per degree C

        for i, (weather, ts) in enumerate(zip(weather_forecast, timestamps)):
            # Clear-sky irradiance
            clear_sky = self.estimate_clear_sky_irradiance(
                ts, asset.location_lat, asset.location_lon
            )

            # Cloud cover attenuation
            cloud_factor = 1.0 - 0.75 * weather.cloud_cover_fraction

            # Effective irradiance
            effective_irradiance = clear_sky * cloud_factor

            # Temperature derating
            temp_diff = weather.temperature_c - 25.0  # STC reference
            temp_factor = 1.0 + temp_coefficient * temp_diff

            # Predicted output
            output = (
                asset.capacity_mw
                * panel_efficiency
                * (effective_irradiance / 1000.0)
                * temp_factor
                * asset.availability_factor
            )
            output = max(0.0, min(asset.capacity_mw, output))

            # Confidence interval widens with horizon
            uncertainty = 0.1 + 0.02 * i  # 10% base + 2% per step
            z_score = 1.645 if self._confidence_level >= 0.9 else 1.28

            predicted.append(output)
            lower.append(max(0.0, output - z_score * uncertainty * output))
            upper.append(min(asset.capacity_mw, output + z_score * uncertainty * output))

        return RenewableForecast(
            asset_id=asset.asset_id,
            timestamps=timestamps,
            predicted_output_mw=predicted,
            confidence_lower_mw=lower,
            confidence_upper_mw=upper,
            weather=weather_forecast,
            model_name="ClearSkySolar",
        )


# ---------------------------------------------------------------------------
# Wind Forecasting
# ---------------------------------------------------------------------------

class WindForecaster(ABC):
    """Abstract base class for wind power forecasting."""

    @abstractmethod
    def forecast(
        self,
        asset: RenewableAsset,
        weather_forecast: List[WeatherConditions],
        timestamps: List[datetime],
    ) -> RenewableForecast:
        """Generate wind power forecast.

        Args:
            asset: The wind asset to forecast.
            weather_forecast: Predicted weather conditions.
            timestamps: Forecast timestamps.

        Returns:
            A RenewableForecast instance.

        Raises:
            ForecastError: If forecasting fails.
        """
        ...


class PowerCurveWindForecaster(WindForecaster):
    """Wind forecaster using a simplified power curve model.

    Maps wind speed to power output using a typical wind turbine
    power curve with cut-in, rated, and cut-out speeds.
    """

    def __init__(
        self,
        cut_in_speed_ms: float = 3.0,
        rated_speed_ms: float = 12.0,
        cut_out_speed_ms: float = 25.0,
        confidence_level: float = 0.9,
    ) -> None:
        """Initialize the wind forecaster.

        Args:
            cut_in_speed_ms: Cut-in wind speed in m/s.
            rated_speed_ms: Rated wind speed in m/s.
            cut_out_speed_ms: Cut-out wind speed in m/s.
            confidence_level: Confidence level for prediction intervals.
        """
        if not 0 < cut_in_speed_ms < rated_speed_ms < cut_out_speed_ms:
            raise ValueError("Wind speeds must satisfy: 0 < cut_in < rated < cut_out")
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")

        self._cut_in = cut_in_speed_ms
        self._rated = rated_speed_ms
        self._cut_out = cut_out_speed_ms
        self._confidence_level = confidence_level

    def _power_curve(self, wind_speed_ms: float) -> float:
        """Compute power output fraction from wind speed.

        Args:
            wind_speed_ms: Wind speed in m/s.

        Returns:
            Power output as fraction of rated capacity in [0, 1].
        """
        if wind_speed_ms < self._cut_in or wind_speed_ms >= self._cut_out:
            return 0.0
        if wind_speed_ms >= self._rated:
            return 1.0

        # Cubic region between cut-in and rated
        fraction = (
            (wind_speed_ms ** 3 - self._cut_in ** 3)
            / (self._rated ** 3 - self._cut_in ** 3)
        )
        return max(0.0, min(1.0, fraction))

    def forecast(
        self,
        asset: RenewableAsset,
        weather_forecast: List[WeatherConditions],
        timestamps: List[datetime],
    ) -> RenewableForecast:
        """Generate wind power forecast.

        Args:
            asset: The wind asset to forecast.
            weather_forecast: Predicted weather conditions.
            timestamps: Forecast timestamps.

        Returns:
            A RenewableForecast instance.
        """
        if not weather_forecast:
            raise ForecastError("Weather forecast cannot be empty")
        if len(weather_forecast) != len(timestamps):
            raise ForecastError("Weather forecast and timestamps must have the same length")
        if asset.asset_type != RenewableType.WIND:
            raise ForecastError(f"Asset {asset.asset_id} is not a wind asset")

        predicted: List[float] = []
        lower: List[float] = []
        upper: List[float] = []

        for i, (weather, ts) in enumerate(zip(weather_forecast, timestamps)):
            # Power curve lookup
            power_fraction = self._power_curve(weather.wind_speed_ms)

            # Air density correction (simplified)
            temp_k = weather.temperature_c + 273.15
            air_density = (weather.pressure_hpa * 100) / (287.05 * temp_k)
            density_correction = air_density / 1.225  # Standard conditions

            output = (
                asset.capacity_mw
                * power_fraction
                * density_correction
                * asset.availability_factor
            )
            output = max(0.0, min(asset.capacity_mw, output))

            # Confidence interval widens with horizon
            uncertainty = 0.15 + 0.03 * i  # 15% base + 3% per step
            z_score = 1.645 if self._confidence_level >= 0.9 else 1.28

            predicted.append(output)
            lower.append(max(0.0, output - z_score * uncertainty * output))
            upper.append(min(asset.capacity_mw, output + z_score * uncertainty * output))

        return RenewableForecast(
            asset_id=asset.asset_id,
            timestamps=timestamps,
            predicted_output_mw=predicted,
            confidence_lower_mw=lower,
            confidence_upper_mw=upper,
            weather=weather_forecast,
            model_name="PowerCurveWind",
        )


# ---------------------------------------------------------------------------
# Grid Storage Optimization
# ---------------------------------------------------------------------------

@dataclass
class StorageAction:
    """A storage control action.

    Attributes:
        timestamp: When the action was computed.
        storage_id: ID of the storage system.
        power_mw: Power setpoint in MW (positive = charge, negative = discharge).
        duration_minutes: Duration of the action in minutes.
        expected_soc_mwh: Expected state of action in MWh.
        reason: Reason for the action.
    """

    timestamp: datetime
    storage_id: str
    power_mw: float
    duration_minutes: float
    expected_soc_mwh: float
    reason: str


class StorageOptimizer:
    """Optimizes grid storage operation for economic and grid benefits.

    Implements price arbitrage, renewable smoothing, and frequency
    regulation support.
    """

    def __init__(
        self,
        electricity_prices_per_mwh: Optional[List[float]] = None,
        price_timestamps: Optional[List[datetime]] = None,
    ) -> None:
        """Initialize the storage optimizer.

        Args:
            electricity_prices_per_mwh: Electricity prices in $/MWh.
            price_timestamps: Timestamps for prices.
        """
        self._prices = electricity_prices_per_mwh or []
        self._price_timestamps = price_timestamps or []

    def set_prices(
        self,
        prices_per_mwh: List[float],
        timestamps: List[datetime],
    ) -> None:
        """Set electricity prices for optimization.

        Args:
            prices_per_mwh: Prices in $/MWh.
            timestamps: Timestamps for prices.

        Raises:
            ValueError: If prices and timestamps don't match.
        """
        if len(prices_per_mwh) != len(timestamps):
            raise ValueError("Prices and timestamps must have the same length")
        self._prices = list(prices_per_mwh)
        self._price_timestamps = list(timestamps)

    def optimize_arbitrage(
        self,
        storage: StorageSystem,
        timestamps: List[datetime],
        prices: List[float],
    ) -> List[StorageAction]:
        """Optimize storage for price arbitrage.

        Charges during low-price periods and discharges during high-price periods.

        Args:
            storage: The storage system to optimize.
            timestamps: Optimization timestamps.
            prices: Electricity prices in $/MWh.

        Returns:
            List of StorageAction instances.

        Raises:
            StorageError: If optimization fails.
        """
        if not storage.is_online:
            raise StorageError(f"Storage {storage.storage_id} is offline")
        if len(timestamps) != len(prices):
            raise StorageError("Timestamps and prices must have the same length")
        if len(timestamps) < 2:
            raise StorageError("Need at least 2 timestamps for arbitrage")

        actions: List[StorageAction] = []
        soc = storage.current_soc_mwh

        # Find price thresholds
        avg_price = sum(prices) / len(prices)
        price_range = max(prices) - min(prices)
        low_threshold = avg_price - 0.3 * price_range
        high_threshold = avg_price + 0.3 * price_range

        for i, (ts, price) in enumerate(zip(timestamps, prices)):
            if price <= low_threshold and soc < storage.max_soc_mwh:
                # Charge during low prices
                energy_needed = storage.max_soc_mwh - soc
                power = min(storage.max_power_mw, energy_needed)
                duration = 60.0  # 1 hour default
                actual_energy = power * (duration / 60.0) * storage.round_trip_efficiency
                soc = min(storage.max_soc_mwh, soc + actual_energy)

                actions.append(StorageAction(
                    timestamp=ts,
                    storage_id=storage.storage_id,
                    power_mw=power,
                    duration_minutes=duration,
                    expected_soc_mwh=soc,
                    reason="price_arbitrage_charge",
                ))
            elif price >= high_threshold and soc > storage.min_soc_mwh:
                # Discharge during high prices
                energy_available = soc - storage.min_soc_mwh
                power = min(storage.max_power_mw, energy_available)
                duration = 60.0
                actual_energy = power * (duration / 60.0) / storage.round_trip_efficiency
                soc = max(storage.min_soc_mwh, soc - actual_energy)

                actions.append(StorageAction(
                    timestamp=ts,
                    storage_id=storage.storage_id,
                    power_mw=-power,
                    duration_minutes=duration,
                    expected_soc_mwh=soc,
                    reason="price_arbitrage_discharge",
                ))

        return actions

    def optimize_renewable_smoothing(
        self,
        storage: StorageSystem,
        renewable_forecast: RenewableForecast,
        target_output_mw: float,
    ) -> List[StorageAction]:
        """Optimize storage to smooth renewable output.

        Charges when renewable output exceeds target, discharges when below.

        Args:
            storage: The storage system to optimize.
            renewable_forecast: Forecast of renewable generation.
            target_output_mw: Target smoothed output in MW.

        Returns:
            List of StorageAction instances.
        """
        if not storage.is_online:
            raise StorageError(f"Storage {storage.storage_id} is offline")

        actions: List[StorageAction] = []
        soc = storage.current_soc_mwh

        for i, (ts, predicted) in enumerate(
            zip(renewable_forecast.timestamps, renewable_forecast.predicted_output_mw)
        ):
            deviation = predicted - target_output_mw

            if deviation > 0 and soc < storage.max_soc_mwh:
                # Surplus: charge
                energy = min(deviation, storage.max_power_mw)
                actual = min(energy, storage.available_capacity_mwh / storage.round_trip_efficiency)
                soc += actual * storage.round_trip_efficiency

                actions.append(StorageAction(
                    timestamp=ts,
                    storage_id=storage.storage_id,
                    power_mw=actual,
                    duration_minutes=60.0,
                    expected_soc_mwh=soc,
                    reason="renewable_smoothing_charge",
                ))
            elif deviation < 0 and soc > storage.min_soc_mwh:
                # Deficit: discharge
                energy = min(abs(deviation), storage.max_power_mw)
                actual = min(energy, storage.available_energy_mwh * storage.round_trip_efficiency)
                soc -= actual / storage.round_trip_efficiency

                actions.append(StorageAction(
                    timestamp=ts,
                    storage_id=storage.storage_id,
                    power_mw=-actual,
                    duration_minutes=60.0,
                    expected_soc_mwh=soc,
                    reason="renewable_smoothing_discharge",
                ))

        return actions

    def optimize_frequency_regulation(
        self,
        storage: StorageSystem,
        frequency_deviations_hz: List[float],
        timestamps: List[datetime],
    ) -> List[StorageAction]:
        """Optimize storage for frequency regulation.

        Provides fast frequency response by charging/discharging based on
        frequency deviations.

        Args:
            storage: The storage system to optimize.
            frequency_deviations_hz: Frequency deviations in Hz.
            timestamps: Timestamps for deviations.

        Returns:
            List of StorageAction instances.
        """
        if not storage.is_online:
            raise StorageError(f"Storage {storage.storage_id} is offline")
        if len(frequency_deviations_hz) != len(timestamps):
            raise StorageError("Deviations and timestamps must have the same length")

        actions: List[StorageAction] = []
        soc = storage.current_soc_mw
        deadband = 0.02  # Hz

        for ts, deviation in zip(timestamps, frequency_deviations_hz):
            if abs(deviation) <= deadband:
                continue

            # Discharge on low frequency, charge on high frequency
            if deviation < 0:
                # Low frequency: discharge to support grid
                power = min(storage.max_power_mw, abs(deviation) * 500)  # Gain factor
                actual = min(power, storage.available_energy_mwh * storage.round_trip_efficiency)
                soc -= actual / storage.round_trip_efficiency

                actions.append(StorageAction(
                    timestamp=ts,
                    storage_id=storage.storage_id,
                    power_mw=-actual,
                    duration_minutes=1.0,
                    expected_soc_mwh=soc,
                    reason="frequency_regulation_discharge",
                ))
            else:
                # High frequency: charge to absorb excess
                power = min(storage.max_power_mw, deviation * 500)
                actual = min(power, storage.available_capacity_mwh / storage.round_trip_efficiency)
                soc += actual * storage.round_trip_efficiency

                actions.append(StorageAction(
                    timestamp=ts,
                    storage_id=storage.storage_id,
                    power_mw=actual,
                    duration_minutes=1.0,
                    expected_soc_mwh=soc,
                    reason="frequency_regulation_charge",
                ))

        return actions


# ---------------------------------------------------------------------------
# Curtailment Management
# ---------------------------------------------------------------------------

@dataclass
class CurtailmentSchedule:
    """A curtailment schedule for a renewable asset.

    Attributes:
        asset_id: ID of the asset.
        schedule: List of (timestamp, target_output_mw) tuples.
        total_curtailed_mwh: Total energy curtailed in MWh.
        reason: Primary reason for curtailment.
    """

    asset_id: str
    schedule: List[Tuple[datetime, float]]
    total_curtailed_mwh: float
    reason: CurtailmentReason


class CurtailmentManager:
    """Manages renewable curtailment decisions.

    Determines when and how much to curtail renewable generation based on
    grid constraints, oversupply, and market signals.
    """

    def __init__(
        self,
        min_curtailment_mw: float = 1.0,
        max_curtailment_fraction: float = 0.8,
        ramp_rate_mw_per_min: float = 10.0,
    ) -> None:
        """Initialize the curtailment manager.

        Args:
            min_curtailment_mw: Minimum curtailment in MW.
            max_curtailment_fraction: Maximum fraction of output to curtail.
            ramp_rate_mw_per_min: Maximum ramp rate in MW/min.
        """
        if min_curtailment_mw < 0:
            raise ValueError("min_curtailment_mw must be non-negative")
        if not 0 < max_curtailment_fraction <= 1:
            raise ValueError("max_curtailment_fraction must be in (0, 1]")
        if ramp_rate_mw_per_min <= 0:
            raise ValueError("ramp_rate_mw_per_min must be positive")

        self._min_curtailment = min_curtailment_mw
        self._max_fraction = max_curtailment_fraction
        self._ramp_rate = ramp_rate_mw_per_min

    def compute_curtailment(
        self,
        asset: RenewableAsset,
        forecast: RenewableForecast,
        grid_capacity_mw: float,
        current_renewable_total_mw: float,
        timestamp: Optional[datetime] = None,
    ) -> Optional[CurtailmentAction]:
        """Compute curtailment action for a renewable asset.

        Args:
            asset: The renewable asset.
            forecast: Generation forecast for the asset.
            grid_capacity_mw: Available grid capacity in MW.
            current_renewable_total_mw: Total current renewable output in MW.
            timestamp: Optional timestamp.

        Returns:
            A CurtailmentAction if curtailment is needed, None otherwise.
        """
        ts = timestamp or datetime.now()

        if not asset.is_online or asset.current_output_mw <= 0:
            return None

        # Check if curtailment is needed
        if current_renewable_total_mw <= grid_capacity_mw:
            return None

        # Calculate required curtailment
        excess = current_renewable_total_mw - grid_capacity_mw
        max_curtail = asset.current_output_mw * self._max_fraction
        curtailment = min(excess, max_curtail)

        if curtailment < self._min_curtailment:
            return None

        target_output = asset.current_output_mw - curtailment

        # Estimate lost revenue (simplified)
        lost_revenue = curtailment * 50.0  # Assume $50/MWh

        return CurtailmentAction(
            timestamp=ts,
            asset_id=asset.asset_id,
            current_output_mw=asset.current_output_mw,
            target_output_mw=target_output,
            curtailed_mw=curtailment,
            reason=CurtailmentReason.GRID_CONSTRAINT,
            duration_minutes=60.0,
            estimated_lost_revenue=lost_revenue,
        )

    def compute_oversupply_curtailment(
        self,
        assets: List[RenewableAsset],
        total_demand_mw: float,
        must_run_generation_mw: float,
        timestamp: Optional[datetime] = None,
    ) -> List[CurtailmentAction]:
        """Compute curtailment actions for oversupply conditions.

        Args:
            assets: List of renewable assets.
            total_demand_mw: Total demand in MW.
            must_run_generation_mw: Must-run generation (nuclear, etc.) in MW.
            timestamp: Optional timestamp.

        Returns:
            List of CurtailmentAction instances.
        """
        ts = timestamp or datetime.now()
        actions: List[CurtailmentAction] = []

        total_renewable = sum(a.current_output_mw for a in assets if a.is_online)
        total_generation = total_renewable + must_run_generation_mw

        if total_generation <= total_demand_mw:
            return actions

        excess = total_generation - total_demand_mw

        # Sort assets by curtailment cost (lowest cost first)
        sorted_assets = sorted(
            [a for a in assets if a.is_online and a.current_output_mw > 0],
            key=lambda a: a.current_output_mw,
            reverse=True,
        )

        for asset in sorted_assets:
            if excess <= 0:
                break

            max_curtail = asset.current_output_mw * self._max_fraction
            curtailment = min(excess, max_curtail)

            if curtailment < self._min_curtailment:
                continue

            target_output = asset.current_output_mw - curtailment
            lost_revenue = curtailment * 40.0  # Assume $40/MWh for oversupply

            actions.append(CurtailmentAction(
                timestamp=ts,
                asset_id=asset.asset_id,
                current_output_mw=asset.current_output_mw,
                target_output_mw=target_output,
                curtailed_mw=curtailment,
                reason=CurtailmentReason.OVERSUPPLY,
                duration_minutes=60.0,
                estimated_lost_revenue=lost_revenue,
            ))

            excess -= curtailment

        return actions

    def create_curtailment_schedule(
        self,
        asset: RenewableAsset,
        forecast: RenewableForecast,
        grid_capacity_mw: float,
        reason: CurtailmentReason,
    ) -> CurtailmentSchedule:
        """Create a curtailment schedule based on forecast.

        Args:
            asset: The renewable asset.
            forecast: Generation forecast.
            grid_capacity_mw: Grid capacity in MW.
            reason: Reason for curtailment.

        Returns:
            A CurtailmentSchedule instance.
        """
        schedule: List[Tuple[datetime, float]] = []
        total_curtailed = 0.0

        for ts, predicted in zip(forecast.timestamps, forecast.predicted_output_mw):
            if predicted > grid_capacity_mw:
                target = grid_capacity_mw
                curtailed = predicted - target
                total_curtailed += curtailed
            else:
                target = predicted

            schedule.append((ts, target))

        return CurtailmentSchedule(
            asset_id=asset.asset_id,
            schedule=schedule,
            total_curtailed_mwh=total_curtailed,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# Renewables Controller (orchestrator)
# ---------------------------------------------------------------------------

class RenewablesController:
    """High-level renewables controller orchestrating all subsystems.

    Coordinates solar/wind forecasting, storage optimization, and
    curtailment management into a unified control loop.
    """

    def __init__(
        self,
        solar_forecaster: Optional[SolarForecaster] = None,
        wind_forecaster: Optional[WindForecaster] = None,
        storage_optimizer: Optional[StorageOptimizer] = None,
        curtailment_manager: Optional[CurtailmentManager] = None,
    ) -> None:
        """Initialize the renewables controller.

        Args:
            solar_forecaster: Solar forecasting subsystem.
            wind_forecaster: Wind forecasting subsystem.
            storage_optimizer: Storage optimization subsystem.
            curtailment_manager: Curtailment management subsystem.
        """
        self._solar_forecaster = solar_forecaster or ClearSkySolarForecaster()
        self._wind_forecaster = wind_forecaster or PowerCurveWindForecaster()
        self._storage_optimizer = storage_optimizer or StorageOptimizer()
        self._curtailment_manager = curtailment_manager or CurtailmentManager()
        self._assets: Dict[str, RenewableAsset] = {}
        self._storage_systems: Dict[str, StorageSystem] = {}

    @property
    def assets(self) -> Dict[str, RenewableAsset]:
        """Renewable assets under control."""
        return self._assets

    @property
    def storage_systems(self) -> Dict[str, StorageSystem]:
        """Storage systems under control."""
        return self._storage_systems

    def add_asset(self, asset: RenewableAsset) -> None:
        """Add a renewable asset.

        Args:
            asset: The renewable asset to add.
        """
        self._assets[asset.asset_id] = asset

    def add_storage(self, storage: StorageSystem) -> None:
        """Add a storage system.

        Args:
            storage: The storage system to add.
        """
        self._storage_systems[storage.storage_id] = storage

    def forecast_asset(
        self,
        asset_id: str,
        weather_forecast: List[WeatherConditions],
        timestamps: List[datetime],
    ) -> RenewableForecast:
        """Generate forecast for a specific asset.

        Args:
            asset_id: ID of the asset to forecast.
            weather_forecast: Predicted weather conditions.
            timestamps: Forecast timestamps.

        Returns:
            A RenewableForecast instance.

        Raises:
            RenewablesError: If the asset is not found or type is unsupported.
        """
        if asset_id not in self._assets:
            raise RenewablesError(f"Asset {asset_id} not found")

        asset = self._assets[asset_id]

        if asset.asset_type == RenewableType.SOLAR:
            return self._solar_forecaster.forecast(asset, weather_forecast, timestamps)
        elif asset.asset_type == RenewableType.WIND:
            return self._wind_forecaster.forecast(asset, weather_forecast, timestamps)
        else:
            raise RenewablesError(f"Unsupported asset type: {asset.asset_type}")

    def optimize_storage(
        self,
        storage_id: str,
        mode: str,
        **kwargs: Any,
    ) -> List[StorageAction]:
        """Optimize a storage system.

        Args:
            storage_id: ID of the storage system.
            mode: Optimization mode ('arbitrage', 'renewable_smoothing', 'frequency_regulation').
            **kwargs: Mode-specific arguments.

        Returns:
            List of StorageAction instances.

        Raises:
            RenewablesError: If the storage is not found or mode is invalid.
        """
        if storage_id not in self._storage_systems:
            raise RenewablesError(f"Storage {storage_id} not found")

        storage = self._storage_systems[storage_id]

        if mode == "arbitrage":
            return self._storage_optimizer.optimize_arbitrage(
                storage,
                kwargs["timestamps"],
                kwargs["prices"],
            )
        elif mode == "renewable_smoothing":
            return self._storage_optimizer.optimize_renewable_smoothing(
                storage,
                kwargs["forecast"],
                kwargs["target_output_mw"],
            )
        elif mode == "frequency_regulation":
            return self._storage_optimizer.optimize_frequency_regulation(
                storage,
                kwargs["frequency_deviations_hz"],
                kwargs["timestamps"],
            )
        else:
            raise RenewablesError(f"Unknown optimization mode: {mode}")

    def compute_curtailment(
        self,
        grid_capacity_mw: float,
        total_demand_mw: float,
        must_run_generation_mw: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> List[CurtailmentAction]:
        """Compute curtailment actions for all assets.

        Args:
            grid_capacity_mw: Available grid capacity in MW.
            total_demand_mw: Total demand in MW.
            must_run_generation_mw: Must-run generation in MW.
            timestamp: Optional timestamp.

        Returns:
            List of CurtailmentAction instances.
        """
        ts = timestamp or datetime.now()
        actions: List[CurtailmentAction] = []

        # Check oversupply
        oversupply_actions = self._curtailment_manager.compute_oversupply_curtailment(
            list(self._assets.values()),
            total_demand_mw,
            must_run_generation_mw,
            ts,
        )
        actions.extend(oversupply_actions)

        # Check individual asset constraints
        total_renewable = sum(a.current_output_mw for a in self._assets.values() if a.is_online)
        for asset in self._assets.values():
            if not asset.is_online:
                continue

            # Create a simple forecast for curtailment check
            forecast = RenewableForecast(
                asset_id=asset.asset_id,
                timestamps=[ts],
                predicted_output_mw=[asset.current_output_mw],
                confidence_lower_mw=[asset.current_output_mw * 0.9],
                confidence_upper_mw=[asset.current_output_mw * 1.1],
                weather=[WeatherConditions()],
                model_name="Current",
            )

            action = self._curtailment_manager.compute_curtailment(
                asset,
                forecast,
                grid_capacity_mw,
                total_renewable,
                ts,
            )
            if action:
                actions.append(action)

        return actions

    def get_total_renewable_output(self) -> float:
        """Get total renewable output in MW.

        Returns:
            Total renewable generation in MW.
        """
        return sum(a.current_output_mw for a in self._assets.values() if a.is_online)

    def get_total_storage_soc_mwh(self) -> float:
        """Get total storage state of charge in MWh.

        Returns:
            Total storage SOC in MWh.
        """
        return sum(s.current_soc_mwh for s in self._storage_systems.values() if s.is_online)
