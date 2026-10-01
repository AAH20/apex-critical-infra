"""Grid Simulator module for Apex Critical Infrastructure.

Provides real-time grid simulation with agentic AI decision-making,
integrating smart grid, renewables, and predictive maintenance systems.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from smart_grid import (
    Bus,
    CapacitorBank,
    FrequencyRegulator,
    Generator,
    GridBalancer,
    GridSnapshot,
    GridStateError,
    Line,
    Load,
    SmartGridController,
    TransformerTap,
    VoltageController,
)
from predictive_maintenance import (
    Equipment,
    EquipmentType,
    EquipmentStatus,
    MaintenanceScheduler,
    MaintenanceTask,
    MaintenanceType,
    MaintenancePriority,
    PredictiveMaintenanceController,
    SensorReading,
    StatisticalAnomalyDetector,
    WeibullFailurePredictor,
)
from renewables import (
    ClearSkySolarForecaster,
    CurtailmentManager,
    PowerCurveWindForecaster,
    RenewableAsset,
    RenewableForecast,
    RenewableType,
    RenewablesController,
    StorageOptimizer,
    StorageSystem,
    StorageType,
    WeatherConditions,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SimulationError(Exception):
    """Raised when grid simulation encounters an error."""


class ConfigurationError(SimulationError):
    """Raised when simulation configuration is invalid."""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class SimulationStatus(Enum):
    """Simulation execution status."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class SimulationConfig:
    """Configuration for grid simulation.

    Attributes:
        duration_hours: Total simulation duration in hours.
        time_step_seconds: Simulation time step in seconds.
        random_seed: Random seed for reproducibility.
        enable_forecasting: Whether to enable demand forecasting.
        enable_storage: Whether to enable storage optimization.
        enable_curtailment: Whether to enable curtailment management.
        enable_predictive_maintenance: Whether to enable predictive maintenance.
        disturbance_probability: Probability of random disturbances per step.
        load_variability: Load variability as fraction of mean.
        renewable_variability: Renewable variability as fraction of mean.
    """

    duration_hours: float = 24.0
    time_step_seconds: float = 60.0
    random_seed: int = 42
    enable_forecasting: bool = True
    enable_storage: bool = True
    enable_curtailment: bool = True
    enable_predictive_maintenance: bool = True
    disturbance_probability: float = 0.01
    load_variability: float = 0.05
    renewable_variability: float = 0.1

    def __post_init__(self) -> None:
        if self.duration_hours <= 0:
            raise ConfigurationError("duration_hours must be positive")
        if self.time_step_seconds <= 0:
            raise ConfigurationError("time_step_seconds must be positive")
        if not 0 <= self.disturbance_probability <= 1:
            raise ConfigurationError("disturbance_probability must be in [0, 1]")
        if self.load_variability < 0:
            raise ConfigurationError("load_variability must be non-negative")
        if self.renewable_variability < 0:
            raise ConfigurationError("renewable_variability must be non-negative")


@dataclass
class SimulationStep:
    """A single simulation step result.

    Attributes:
        step_number: Step index.
        timestamp: Simulation timestamp.
        snapshot: Grid state snapshot.
        frequency_hz: System frequency in Hz.
        total_generation_mw: Total generation in MW.
        total_load_mw: Total load in MW.
        total_renewable_mw: Total renewable generation in MW.
        total_storage_soc_mwh: Total storage SOC in MWh.
        power_imbalance_mw: Power imbalance in MW.
        frequency_stable: Whether frequency is stable.
        voltage_stable: Whether voltage is stable.
        curtailment_mw: Total curtailment in MW.
        maintenance_alerts: List of maintenance alerts.
        decisions: Dict of AI decisions made.
    """

    step_number: int
    timestamp: datetime
    snapshot: GridSnapshot
    frequency_hz: float
    total_generation_mw: float
    total_load_mw: float
    total_renewable_mw: float
    total_storage_soc_mwh: float
    power_imbalance_mw: float
    frequency_stable: bool
    voltage_stable: bool
    curtailment_mw: float
    maintenance_alerts: List[str]
    decisions: Dict[str, Any]


@dataclass
class SimulationResult:
    """Complete simulation result.

    Attributes:
        config: Simulation configuration used.
        steps: List of simulation steps.
        start_time: Simulation start time.
        end_time: Simulation end time.
        status: Final simulation status.
        summary: Summary statistics.
    """

    config: SimulationConfig
    steps: List[SimulationStep]
    start_time: datetime
    end_time: datetime
    status: SimulationStatus
    summary: Dict[str, Any]


# ---------------------------------------------------------------------------
# Grid Builder
# ---------------------------------------------------------------------------

class GridBuilder:
    """Builder for creating grid simulation scenarios."""

    @staticmethod
    def create_simple_grid() -> Tuple[Dict[str, Bus], Dict[str, Line], Dict[str, Generator], Dict[str, Load]]:
        """Create a simple 3-bus test grid.

        Returns:
            Tuple of (buses, lines, generators, loads) dicts.
        """
        buses = {
            "bus1": Bus(bus_id="bus1", voltage_pu=1.0, bus_type="slack"),
            "bus2": Bus(bus_id="bus2", voltage_pu=0.98, bus_type="pv"),
            "bus3": Bus(bus_id="bus3", voltage_pu=0.95, bus_type="pq"),
        }

        lines = {
            "line12": Line(line_id="line12", from_bus="bus1", to_bus="bus2", rating_mva=200.0),
            "line23": Line(line_id="line23", from_bus="bus2", to_bus="bus3", rating_mva=150.0),
            "line13": Line(line_id="line13", from_bus="bus1", to_bus="bus3", rating_mva=100.0),
        }

        generators = {
            "gen1": Generator(
                gen_id="gen1", bus_id="bus1",
                p_min_mw=50.0, p_max_mw=300.0,
                current_output_mw=150.0,
                marginal_cost_per_mwh=40.0,
            ),
            "gen2": Generator(
                gen_id="gen2", bus_id="bus2",
                p_min_mw=20.0, p_max_mw=150.0,
                current_output_mw=80.0,
                marginal_cost_per_mwh=60.0,
            ),
        }

        loads = {
            "load1": Load(load_id="load1", bus_id="bus2", active_power_mw=60.0),
            "load2": Load(load_id="load2", bus_id="bus3", active_power_mw=90.0, is_controllable=True, priority=5),
        }

        return buses, lines, generators, loads

    @staticmethod
    def create_renewable_grid() -> Tuple[Dict[str, Bus], Dict[str, Line], Dict[str, Generator], Dict[str, Load]]:
        """Create a grid with high renewable penetration.

        Returns:
            Tuple of (buses, lines, generators, loads) dicts.
        """
        buses = {
            "bus1": Bus(bus_id="bus1", voltage_pu=1.0, bus_type="slack"),
            "bus2": Bus(bus_id="bus2", voltage_pu=0.99, bus_type="pv"),
            "bus3": Bus(bus_id="bus3", voltage_pu=0.97, bus_type="pq"),
            "bus4": Bus(bus_id="bus4", voltage_pu=0.96, bus_type="pq"),
        }

        lines = {
            "line12": Line(line_id="line12", from_bus="bus1", to_bus="bus2", rating_mva=300.0),
            "line23": Line(line_id="line23", from_bus="bus2", to_bus="bus3", rating_mva=200.0),
            "line34": Line(line_id="line34", from_bus="bus3", to_bus="bus4", rating_mva=150.0),
            "line14": Line(line_id="line14", from_bus="bus1", to_bus="bus4", rating_mva=100.0),
        }

        generators = {
            "gen1": Generator(
                gen_id="gen1", bus_id="bus1",
                p_min_mw=30.0, p_max_mw=200.0,
                current_output_mw=100.0,
                marginal_cost_per_mwh=35.0,
            ),
            "gen2": Generator(
                gen_id="gen2", bus_id="bus2",
                p_min_mw=10.0, p_max_mw=100.0,
                current_output_mw=50.0,
                marginal_cost_per_mwh=55.0,
            ),
        }

        loads = {
            "load1": Load(load_id="load1", bus_id="bus2", active_power_mw=80.0),
            "load2": Load(load_id="load2", bus_id="bus3", active_power_mw=120.0, is_controllable=True, priority=4),
            "load3": Load(load_id="load3", bus_id="bus4", active_power_mw=60.0, is_controllable=True, priority=6),
        }

        return buses, lines, generators, loads


# ---------------------------------------------------------------------------
# Grid Simulator
# ---------------------------------------------------------------------------

class GridSimulator:
    """Real-time grid simulator with agentic AI decision-making.

    Simulates grid operation over time, integrating smart grid control,
    renewable energy management, and predictive maintenance.
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        smart_grid_controller: Optional[SmartGridController] = None,
        renewables_controller: Optional[RenewablesController] = None,
        maintenance_controller: Optional[PredictiveMaintenanceController] = None,
    ) -> None:
        """Initialize the grid simulator.

        Args:
            config: Simulation configuration.
            smart_grid_controller: Smart grid controller instance.
            renewables_controller: Renewables controller instance.
            maintenance_controller: Predictive maintenance controller instance.
        """
        self._config = config or SimulationConfig()
        self._smart_grid = smart_grid_controller or SmartGridController()
        self._renewables = renewables_controller or RenewablesController()
        self._maintenance = maintenance_controller or PredictiveMaintenanceController()

        self._status = SimulationStatus.IDLE
        self._current_step = 0
        self._steps: List[SimulationStep] = []
        self._start_time: Optional[datetime] = None
        self._current_time: Optional[datetime] = None

        # Grid state
        self._buses: Dict[str, Bus] = {}
        self._lines: Dict[str, Line] = {}
        self._generators: Dict[str, Generator] = {}
        self._loads: Dict[str, Load] = {}

        # Set random seed
        random.seed(self._config.random_seed)

    @property
    def status(self) -> SimulationStatus:
        """Current simulation status."""
        return self._status

    @property
    def current_step(self) -> int:
        """Current simulation step."""
        return self._current_step

    @property
    def steps(self) -> List[SimulationStep]:
        """Simulation steps completed so far."""
        return self._steps

    def setup_scenario(
        self,
        scenario: str = "simple",
    ) -> None:
        """Set up a simulation scenario.

        Args:
            scenario: Scenario name ('simple' or 'renewable').

        Raises:
            ConfigurationError: If scenario is unknown.
        """
        if scenario == "simple":
            self._buses, self._lines, self._generators, self._loads = GridBuilder.create_simple_grid()
        elif scenario == "renewable":
            self._buses, self._lines, self._generators, self._loads = GridBuilder.create_renewable_grid()
        else:
            raise ConfigurationError(f"Unknown scenario: {scenario}")

        # Add renewable assets for renewable scenario
        if scenario == "renewable":
            solar = RenewableAsset(
                asset_id="solar_farm_1",
                asset_type=RenewableType.SOLAR,
                capacity_mw=80.0,
                current_output_mw=40.0,
                location_lat=40.0,
                location_lon=-105.0,
            )
            wind = RenewableAsset(
                asset_id="wind_farm_1",
                asset_type=RenewableType.WIND,
                capacity_mw=120.0,
                current_output_mw=60.0,
                location_lat=41.0,
                location_lon=-104.0,
            )
            self._renewables.add_asset(solar)
            self._renewables.add_asset(wind)

            # Add storage
            storage = StorageSystem(
                storage_id="battery_1",
                storage_type=StorageType.LITHIUM_ION,
                capacity_mwh=200.0,
                max_power_mw=50.0,
                round_trip_efficiency=0.92,
                current_soc_mwh=100.0,
            )
            self._renewables.add_storage(storage)

        # Add equipment for predictive maintenance
        if self._config.enable_predictive_maintenance:
            transformer = Equipment(
                equipment_id="tx_1",
                equipment_type=EquipmentType.TRANSFORMER,
                name="Main Transformer",
                installation_date=datetime.now() - timedelta(days=365 * 10),
                expected_lifetime_years=30.0,
                health_index=0.75,
                criticality=0.9,
            )
            generator_eq = Equipment(
                equipment_id="gen_1",
                equipment_type=EquipmentType.GENERATOR,
                name="Main Generator",
                installation_date=datetime.now() - timedelta(days=365 * 5),
                expected_lifetime_years=25.0,
                health_index=0.85,
                criticality=0.8,
            )
            self._maintenance.add_equipment(transformer)
            self._maintenance.add_equipment(generator_eq)

        # Add capacitor banks and transformers for voltage control
        cap_bank = CapacitorBank(
            bank_id="cb_1",
            bus_id="bus3",
            capacity_mvar=20.0,
        )
        self._smart_grid.add_capacitor_bank(cap_bank)

        tx_tap = TransformerTap(
            transformer_id="tx_tap_1",
            current_tap=0,
        )
        self._smart_grid.add_transformer(tx_tap)

        self._status = SimulationStatus.IDLE
        self._current_step = 0
        self._steps = []

    def run(self) -> SimulationResult:
        """Run the complete simulation.

        Returns:
            A SimulationResult instance.

        Raises:
            SimulationError: If simulation fails.
        """
        if not self._buses:
            raise SimulationError("No scenario configured. Call setup_scenario() first.")

        self._status = SimulationStatus.RUNNING
        self._start_time = datetime.now()
        self._current_time = self._start_time
        self._steps = []
        self._current_step = 0

        total_steps = int(self._config.duration_hours * 3600 / self._config.time_step_seconds)

        logger.info(
            "Starting simulation: %d steps over %.1f hours",
            total_steps, self._config.duration_hours,
        )

        try:
            for step in range(total_steps):
                if self._status == SimulationStatus.PAUSED:
                    break

                step_result = self._run_step(step)
                self._steps.append(step_result)
                self._current_step = step + 1
                self._current_time = self._start_time + timedelta(
                    seconds=(step + 1) * self._config.time_step_seconds
                )

                # Log progress
                if step % max(1, total_steps // 10) == 0:
                    logger.info(
                        "Simulation progress: %d/%d steps (%.0f%%)",
                        step + 1, total_steps, 100 * (step + 1) / total_steps,
                    )

        except Exception as e:
            self._status = SimulationStatus.ERROR
            logger.error("Simulation failed at step %d: %s", self._current_step, e)
            raise SimulationError(f"Simulation failed: {e}") from e

        self._status = SimulationStatus.COMPLETED
        end_time = datetime.now()

        summary = self._compute_summary()

        logger.info("Simulation completed: %d steps", len(self._steps))

        return SimulationResult(
            config=self._config,
            steps=self._steps,
            start_time=self._start_time,
            end_time=end_time,
            status=self._status,
            summary=summary,
        )

    def _run_step(self, step_number: int) -> SimulationStep:
        """Run a single simulation step.

        Args:
            step_number: Step index.

        Returns:
            A SimulationStep instance.
        """
        ts = self._start_time + timedelta(seconds=step_number * self._config.time_step_seconds)

        # Update loads with variability
        self._update_loads(ts)

        # Update renewable generation
        self._update_renewables(ts)

        # Create grid snapshot
        snapshot = GridSnapshot(
            timestamp=ts,
            buses=dict(self._buses),
            lines=dict(self._lines),
            generators=dict(self._generators),
            loads=dict(self._loads),
            frequency_hz=self._compute_frequency(),
        )

        # Smart grid control
        grid_results = self._smart_grid.process_snapshot(snapshot)

        # Apply frequency regulation
        freq_action = grid_results["frequency_action"]
        self._apply_frequency_action(freq_action)

        # Apply voltage control
        voltage_action = grid_results["voltage_action"]
        self._apply_voltage_action(voltage_action)

        # Apply balancing action
        balancing_action = grid_results["balancing_action"]
        self._apply_balancing_action(balancing_action)

        # Renewables management
        curtailment_mw = 0.0
        if self._config.enable_curtailment:
            curtailment_actions = self._renewables.compute_curtailment(
                grid_capacity_mw=500.0,
                total_demand_mw=snapshot.total_load_mw,
            )
            curtailment_mw = sum(a.curtailed_mw for a in curtailment_actions)

        # Predictive maintenance
        maintenance_alerts: List[str] = []
        if self._config.enable_predictive_maintenance:
            maintenance_alerts = self._run_maintenance_check(ts)

        # Random disturbances
        if random.random() < self._config.disturbance_probability:
            self._apply_disturbance()

        # Compute stability metrics
        freq_stable = abs(snapshot.frequency_deviation_hz) < 0.05
        voltage_stable = voltage_action.is_stable

        # Collect decisions
        decisions: Dict[str, Any] = {
            "frequency_action": freq_action,
            "voltage_action": voltage_action,
            "balancing_action": balancing_action,
            "curtailment_mw": curtailment_mw,
        }

        return SimulationStep(
            step_number=step_number,
            timestamp=ts,
            snapshot=snapshot,
            frequency_hz=snapshot.frequency_hz,
            total_generation_mw=snapshot.total_generation_mw,
            total_load_mw=snapshot.total_load_mw,
            total_renewable_mw=self._renewables.get_total_renewable_output(),
            total_storage_soc_mwh=self._renewables.get_total_storage_soc_mwh(),
            power_imbalance_mw=snapshot.power_imbalance_mw,
            frequency_stable=freq_stable,
            voltage_stable=voltage_stable,
            curtailment_mw=curtailment_mw,
            maintenance_alerts=maintenance_alerts,
            decisions=decisions,
        )

    def _update_loads(self, timestamp: datetime) -> None:
        """Update load values with daily profile and variability.

        Args:
            timestamp: Current simulation time.
        """
        hour = timestamp.hour + timestamp.minute / 60.0

        # Daily load profile (simplified)
        if 0 <= hour < 6:
            load_factor = 0.6
        elif 6 <= hour < 9:
            load_factor = 0.8
        elif 9 <= hour < 17:
            load_factor = 1.0
        elif 17 <= hour < 21:
            load_factor = 1.1
        else:
            load_factor = 0.7

        for load in self._loads.values():
            base_load = load.active_power_mw
            variability = random.gauss(0, self._config.load_variability)
            new_load = base_load * load_factor * (1.0 + variability)
            load.active_power_mw = max(0.0, new_load)

    def _update_renewables(self, timestamp: datetime) -> None:
        """Update renewable generation with weather variability.

        Args:
            timestamp: Current simulation time.
        """
        hour = timestamp.hour + timestamp.minute / 60.0

        for asset in self._renewables.assets.values():
            if not asset.is_online:
                continue

            if asset.asset_type == RenewableType.SOLAR:
                # Solar profile: peak at noon
                if 6 <= hour <= 20:
                    solar_factor = math.sin(math.pi * (hour - 6) / 14)
                else:
                    solar_factor = 0.0

                variability = random.gauss(0, self._config.renewable_variability)
                output = asset.capacity_mw * solar_factor * (1.0 + variability)
                asset.current_output_mw = max(0.0, min(asset.capacity_mw, output))

            elif asset.asset_type == RenewableType.WIND:
                # Wind: more variable, higher at night
                wind_base = 0.5 + 0.3 * math.sin(2 * math.pi * hour / 24 + math.pi)
                variability = random.gauss(0, self._config.renewable_variability * 1.5)
                output = asset.capacity_mw * wind_base * (1.0 + variability)
                asset.current_output_mw = max(0.0, min(asset.capacity_mw, output))

    def _compute_frequency(self) -> float:
        """Compute system frequency based on power imbalance.

        Returns:
            System frequency in Hz.
        """
        total_gen = sum(g.current_output_mw for g in self._generators.values() if g.is_online)
        total_load = sum(l.active_power_mw for l in self._loads.values())
        total_renewable = self._renewables.get_total_renewable_output()

        # Include renewable in generation
        total_gen += total_renewable

        imbalance = total_gen - total_load

        # Frequency response: 50 Hz nominal, droop of 0.05 Hz per 100 MW
        nominal = 50.0
        droop = 0.0005  # Hz per MW
        frequency = nominal - imbalance * droop

        # Add small noise
        frequency += random.gauss(0, 0.005)

        return max(49.0, min(51.0, frequency))

    def _apply_frequency_action(self, action: Any) -> None:
        """Apply frequency regulation action.

        Args:
            action: FrequencyRegulationAction to apply.
        """
        for gen_id, adjustment in action.generator_adjustments.items():
            if gen_id in self._generators:
                gen = self._generators[gen_id]
                new_output = gen.current_output_mw + adjustment
                gen.current_output_mw = max(gen.p_min_mw, min(gen.p_max_mw, new_output))

    def _apply_voltage_action(self, action: Any) -> None:
        """Apply voltage control action.

        Args:
            action: VoltageControlAction to apply.
        """
        # Update bus voltages based on control actions
        for bus_id, voltage in action.bus_voltages.items():
            if bus_id in self._buses:
                bus = self._buses[bus_id]
                # Apply small voltage adjustment
                adjustment = 0.0
                if bus.voltage_pu < 0.95:
                    adjustment = 0.01
                elif bus.voltage_pu > 1.05:
                    adjustment = -0.01

                new_voltage = bus.voltage_pu + adjustment
                self._buses[bus_id] = Bus(
                    bus_id=bus.bus_id,
                    voltage_pu=new_voltage,
                    voltage_angle_deg=bus.voltage_angle_deg,
                    active_power_mw=bus.active_power_mw,
                    reactive_power_mvar=bus.reactive_power_mvar,
                    bus_type=bus.bus_type,
                )

    def _apply_balancing_action(self, action: Any) -> None:
        """Apply grid balancing action.

        Args:
            action: BalancingAction to apply.
        """
        for gen_id, setpoint in action.generator_setpoints.items():
            if gen_id in self._generators:
                gen = self._generators[gen_id]
                gen.current_output_mw = max(gen.p_min_mw, min(gen.p_max_mw, setpoint))

    def _run_maintenance_check(self, timestamp: datetime) -> List[str]:
        """Run predictive maintenance check.

        Args:
            timestamp: Current simulation time.

        Returns:
            List of maintenance alert messages.
        """
        alerts: List[str] = []

        for eq_id, eq in self._maintenance.equipment.items():
            # Generate synthetic sensor readings
            readings = self._generate_sensor_readings(eq, timestamp)
            for reading in readings:
                self._maintenance.add_sensor_reading(reading)

            # Analyze equipment
            try:
                result = self._maintenance.analyze_equipment(eq_id)
                prediction = result["prediction"]

                if prediction.failure_probability > 0.6:
                    alerts.append(
                        f"HIGH failure risk for {eq.name}: "
                        f"{prediction.failure_probability:.0%} - "
                        f"{prediction.recommended_action}"
                    )
                elif prediction.failure_probability > 0.3:
                    alerts.append(
                        f"MEDIUM failure risk for {eq.name}: "
                        f"{prediction.failure_probability:.0%}"
                    )
            except Exception as e:
                logger.debug("Maintenance analysis failed for %s: %s", eq_id, e)

        return alerts

    def _generate_sensor_readings(
        self,
        equipment: Equipment,
        timestamp: datetime,
    ) -> List[SensorReading]:
        """Generate synthetic sensor readings for equipment.

        Args:
            equipment: The equipment to generate readings for.
            timestamp: Current time.

        Returns:
            List of SensorReading instances.
        """
        readings: List[SensorReading] = []

        # Temperature sensor
        base_temp = 60.0 + (1.0 - equipment.health_index) * 30.0
        temp = base_temp + random.gauss(0, 2.0)
        readings.append(SensorReading(
            sensor_id=f"{equipment.equipment_id}_temp",
            equipment_id=equipment.equipment_id,
            timestamp=timestamp,
            value=temp,
            unit="C",
        ))

        # Vibration sensor
        base_vib = 2.0 + (1.0 - equipment.health_index) * 5.0
        vib = max(0.0, base_vib + random.gauss(0, 0.5))
        readings.append(SensorReading(
            sensor_id=f"{equipment.equipment_id}_vib",
            equipment_id=equipment.equipment_id,
            timestamp=timestamp,
            value=vib,
            unit="mm/s",
        ))

        # Current sensor
        current = 100.0 + random.gauss(0, 10.0)
        readings.append(SensorReading(
            sensor_id=f"{equipment.equipment_id}_current",
            equipment_id=equipment.equipment_id,
            timestamp=timestamp,
            value=current,
            unit="A",
        ))

        return readings

    def _apply_disturbance(self) -> None:
        """Apply a random disturbance to the grid."""
        disturbance_type = random.choice(["load_spike", "gen_trip", "line_fault"])

        if disturbance_type == "load_spike":
            # Random load spike
            load = random.choice(list(self._loads.values()))
            load.active_power_mw *= 1.2
            logger.debug("Disturbance: load spike on %s", load.load_id)

        elif disturbance_type == "gen_trip":
            # Generator trip
            online_gens = [g for g in self._generators.values() if g.is_online]
            if online_gens:
                gen = random.choice(online_gens)
                gen.is_online = False
                logger.debug("Disturbance: generator %s tripped", gen.gen_id)

        elif disturbance_type == "line_fault":
            # Line fault (simplified: reduce voltage)
            bus = random.choice(list(self._buses.values()))
            self._buses[bus.bus_id] = Bus(
                bus_id=bus.bus_id,
                voltage_pu=bus.voltage_pu * 0.95,
                voltage_angle_deg=bus.voltage_angle_deg,
                active_power_mw=bus.active_power_mw,
                reactive_power_mvar=bus.reactive_power_mvar,
                bus_type=bus.bus_type,
            )
            logger.debug("Disturbance: line fault near %s", bus.bus_id)

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute simulation summary statistics.

        Returns:
            Dict of summary statistics.
        """
        if not self._steps:
            return {}

        freq_stable_count = sum(1 for s in self._steps if s.frequency_stable)
        voltage_stable_count = sum(1 for s in self._steps if s.voltage_stable)
        total_curtailment = sum(s.curtailment_mw for s in self._steps)
        all_alerts = [alert for s in self._steps for alert in s.maintenance_alerts]

        avg_generation = sum(s.total_generation_mw for s in self._steps) / len(self._steps)
        avg_load = sum(s.total_load_mw for s in self._steps) / len(self._steps)
        avg_renewable = sum(s.total_renewable_mw for s in self._steps) / len(self._steps)

        return {
            "total_steps": len(self._steps),
            "frequency_stable_fraction": freq_stable_count / len(self._steps),
            "voltage_stable_fraction": voltage_stable_count / len(self._steps),
            "total_curtailment_mw": total_curtailment,
            "average_generation_mw": avg_generation,
            "average_load_mw": avg_load,
            "average_renewable_mw": avg_renewable,
            "maintenance_alert_count": len(all_alerts),
            "unique_maintenance_alerts": len(set(all_alerts)),
        }

    def pause(self) -> None:
        """Pause the simulation."""
        if self._status == SimulationStatus.RUNNING:
            self._status = SimulationStatus.PAUSED

    def resume(self) -> None:
        """Resume the simulation."""
        if self._status == SimulationStatus.PAUSED:
            self._status = SimulationStatus.RUNNING

    def reset(self) -> None:
        """Reset the simulation to initial state."""
        self._status = SimulationStatus.IDLE
        self._current_step = 0
        self._steps = []
        self._start_time = None
        self._current_time = None
