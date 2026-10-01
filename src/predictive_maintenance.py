"""Predictive Maintenance module for Apex Critical Infrastructure.

Provides equipment failure prediction, anomaly detection, and maintenance
scheduling for agentic AI decision-making in critical infrastructure.
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

class PredictiveMaintenanceError(Exception):
    """Base exception for predictive maintenance module."""


class FailurePredictionError(PredictiveMaintenanceError):
    """Raised when failure prediction fails."""


class AnomalyDetectionError(PredictiveMaintenanceError):
    """Raised when anomaly detection fails."""


class SchedulingError(PredictiveMaintenanceError):
    """Raised when maintenance scheduling fails."""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class EquipmentType(Enum):
    """Types of equipment in critical infrastructure."""

    TRANSFORMER = "transformer"
    TURBINE = "turbine"
    GENERATOR = "generator"
    CIRCUIT_BREAKER = "circuit_breaker"
    CABLE = "cable"
    SWITCHGEAR = "switchgear"
    BATTERY = "battery"
    SOLAR_INVERTER = "solar_inverter"
    WIND_TURBINE = "wind_turbine"
    PUMP = "pump"
    COMPRESSOR = "compressor"
    MOTOR = "motor"


class EquipmentStatus(Enum):
    """Operational status of equipment."""

    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    MAINTENANCE_REQUIRED = "maintenance_required"
    CRITICAL = "critical"
    OFFLINE = "offline"


class MaintenanceType(Enum):
    """Types of maintenance activities."""

    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    CONDITION_BASED = "condition_based"
    EMERGENCY = "emergency"


class MaintenancePriority(Enum):
    """Priority levels for maintenance tasks."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass(frozen=True)
class SensorReading:
    """A single sensor reading from equipment.

    Attributes:
        sensor_id: Unique sensor identifier.
        equipment_id: ID of the equipment being monitored.
        timestamp: When the reading was taken.
        value: Sensor value.
        unit: Unit of measurement.
        is_valid: Whether the reading passed validation.
    """

    sensor_id: str
    equipment_id: str
    timestamp: datetime
    value: float
    unit: str
    is_valid: bool = True

    def __post_init__(self) -> None:
        if not self.is_valid and math.isnan(self.value):
            raise PredictiveMaintenanceError(
                f"Invalid reading from sensor {self.sensor_id}: NaN value"
            )


@dataclass
class Equipment:
    """A piece of equipment in the infrastructure.

    Attributes:
        equipment_id: Unique identifier.
        equipment_type: Type of equipment.
        name: Human-readable name.
        installation_date: When the equipment was installed.
        expected_lifetime_years: Expected lifetime in years.
        current_status: Current operational status.
        health_index: Health index in [0, 1] (1 = perfect health).
        criticality: Criticality score in [0, 1] (1 = most critical).
        location: Physical location description.
        maintenance_history: List of past maintenance timestamps.
        sensor_ids: List of associated sensor IDs.
    """

    equipment_id: str
    equipment_type: EquipmentType
    name: str
    installation_date: datetime
    expected_lifetime_years: float = 20.0
    current_status: EquipmentStatus = EquipmentStatus.OPERATIONAL
    health_index: float = 1.0
    criticality: float = 0.5
    location: str = ""
    maintenance_history: List[datetime] = field(default_factory=list)
    sensor_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.expected_lifetime_years <= 0:
            raise PredictiveMaintenanceError(
                f"Equipment {self.equipment_id}: expected_lifetime_years must be positive"
            )
        self.health_index = max(0.0, min(1.0, self.health_index))
        self.criticality = max(0.0, min(1.0, self.criticality))

    @property
    def age_years(self) -> float:
        """Current age of the equipment in years."""
        return (datetime.now() - self.installation_date).days / 365.25

    @property
    def remaining_life_fraction(self) -> float:
        """Remaining life as a fraction of expected lifetime."""
        return max(0.0, 1.0 - self.age_years / self.expected_lifetime_years)


@dataclass
class FailurePrediction:
    """A failure prediction result.

    Attributes:
        equipment_id: ID of the equipment.
        timestamp: When the prediction was made.
        failure_probability: Probability of failure in [0, 1].
        predicted_failure_time: Predicted time of failure.
        confidence_interval_days: Confidence interval in days.
        contributing_factors: List of factors contributing to the prediction.
        recommended_action: Recommended maintenance action.
        model_name: Name of the prediction model used.
    """

    equipment_id: str
    timestamp: datetime
    failure_probability: float
    predicted_failure_time: Optional[datetime]
    confidence_interval_days: float
    contributing_factors: List[str]
    recommended_action: str
    model_name: str

    def __post_init__(self) -> None:
        self.failure_probability = max(0.0, min(1.0, self.failure_probability))
        if self.confidence_interval_days < 0:
            raise FailurePredictionError("confidence_interval_days must be non-negative")


@dataclass
class AnomalyEvent:
    """A detected anomaly.

    Attributes:
        event_id: Unique anomaly event identifier.
        equipment_id: ID of the equipment.
        timestamp: When the anomaly was detected.
        sensor_id: ID of the sensor that triggered the anomaly.
        anomaly_score: Anomaly severity score in [0, 1].
        description: Human-readable description.
        is_confirmed: Whether the anomaly has been confirmed.
        root_cause: Identified root cause (if any).
    """

    event_id: str
    equipment_id: str
    timestamp: datetime
    sensor_id: str
    anomaly_score: float
    description: str
    is_confirmed: bool = False
    root_cause: Optional[str] = None

    def __post_init__(self) -> None:
        self.anomaly_score = max(0.0, min(1.0, self.anomaly_score))


@dataclass
class MaintenanceTask:
    """A maintenance task.

    Attributes:
        task_id: Unique task identifier.
        equipment_id: ID of the equipment to maintain.
        maintenance_type: Type of maintenance.
        priority: Priority level.
        description: Task description.
        scheduled_start: Scheduled start time.
        scheduled_end: Scheduled end time.
        estimated_duration_hours: Estimated duration in hours.
        required_skills: List of required skills.
        required_parts: List of required parts.
        is_completed: Whether the task is completed.
        actual_cost: Actual cost in $.
    """

    task_id: str
    equipment_id: str
    maintenance_type: MaintenanceType
    priority: MaintenancePriority
    description: str
    scheduled_start: datetime
    scheduled_end: datetime
    estimated_duration_hours: float = 4.0
    required_skills: List[str] = field(default_factory=list)
    required_parts: List[str] = field(default_factory=list)
    is_completed: bool = False
    actual_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.scheduled_end <= self.scheduled_start:
            raise SchedulingError(
                f"Task {self.task_id}: scheduled_end must be after scheduled_start"
            )
        if self.estimated_duration_hours <= 0:
            raise SchedulingError(
                f"Task {self.task_id}: estimated_duration_hours must be positive"
            )


# ---------------------------------------------------------------------------
# Failure Prediction
# ---------------------------------------------------------------------------

class FailurePredictor(ABC):
    """Abstract base class for equipment failure prediction models."""

    @abstractmethod
    def predict(
        self,
        equipment: Equipment,
        sensor_history: List[SensorReading],
    ) -> FailurePrediction:
        """Predict equipment failure probability.

        Args:
            equipment: The equipment to analyze.
            sensor_history: Historical sensor readings.

        Returns:
            A FailurePrediction instance.

        Raises:
            FailurePredictionError: If prediction fails.
        """
        ...

    @abstractmethod
    def update(self, equipment: Equipment, reading: SensorReading) -> None:
        """Update the model with new sensor data.

        Args:
            equipment: The equipment being monitored.
            reading: New sensor reading.
        """
        ...


class WeibullFailurePredictor(FailurePredictor):
    """Weibull distribution-based failure predictor.

    Uses Weibull analysis to model equipment failure probability
    based on age, usage, and condition indicators.
    """

    def __init__(
        self,
        shape_parameter: float = 2.0,
        scale_parameter: float = 10.0,
        confidence_level: float = 0.9,
    ) -> None:
        """Initialize the Weibull failure predictor.

        Args:
            shape_parameter: Weibull shape parameter (beta).
            scale_parameter: Weibull scale parameter (eta) in years.
            confidence_level: Confidence level for predictions.
        """
        if shape_parameter <= 0:
            raise ValueError("shape_parameter must be positive")
        if scale_parameter <= 0:
            raise ValueError("scale_parameter must be positive")
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")

        self._shape = shape_parameter
        self._scale = scale_parameter
        self._confidence_level = confidence_level
        self._sensor_baselines: Dict[str, Tuple[float, float]] = {}  # sensor_id -> (mean, std)

    def _weibull_cdf(self, t: float) -> float:
        """Compute Weibull CDF.

        Args:
            t: Time in years.

        Returns:
            Cumulative failure probability.
        """
        if t <= 0:
            return 0.0
        return 1.0 - math.exp(-((t / self._scale) ** self._shape))

    def _weibull_reliability(self, t: float) -> float:
        """Compute Weibull reliability function.

        Args:
            t: Time in years.

        Returns:
            Reliability (survival) probability.
        """
        if t <= 0:
            return 1.0
        return math.exp(-((t / self._scale) ** self._shape))

    def predict(
        self,
        equipment: Equipment,
        sensor_history: List[SensorReading],
    ) -> FailurePrediction:
        """Predict equipment failure using Weibull analysis.

        Args:
            equipment: The equipment to analyze.
            sensor_history: Historical sensor readings.

        Returns:
            A FailurePrediction instance.
        """
        if not sensor_history:
            raise FailurePredictionError("Sensor history cannot be empty")

        # Update baselines from sensor history
        self._update_baselines(sensor_history)

        # Age-based failure probability
        age = equipment.age_years
        age_factor = self._weibull_cdf(age)

        # Condition-based adjustment
        condition_factor = self._compute_condition_factor(sensor_history)

        # Combined failure probability
        failure_prob = age_factor * 0.3 + condition_factor * 0.7
        failure_prob = max(0.0, min(1.0, failure_prob))

        # Predict failure time
        if failure_prob > 0.5:
            # Find time when reliability drops to 1 - failure_prob
            target_reliability = 1.0 - failure_prob
            # Inverse Weibull: t = scale * (-ln(R))^(1/shape)
            predicted_years = self._scale * (-math.log(target_reliability)) ** (1.0 / self._shape)
            predicted_time = equipment.installation_date + timedelta(days=predicted_years * 365.25)
        else:
            predicted_time = None

        # Confidence interval
        z_score = 1.645 if self._confidence_level >= 0.9 else 1.28
        ci_days = z_score * 30.0 * (1.0 + age / self._scale)  # Simplified

        # Contributing factors
        factors = self._identify_contributing_factors(equipment, sensor_history)

        # Recommended action
        action = self._recommend_action(failure_prob, factors)

        return FailurePrediction(
            equipment_id=equipment.equipment_id,
            timestamp=datetime.now(),
            failure_probability=failure_prob,
            predicted_failure_time=predicted_time,
            confidence_interval_days=ci_days,
            contributing_factors=factors,
            recommended_action=action,
            model_name="Weibull",
        )

    def _update_baselines(self, readings: List[SensorReading]) -> None:
        """Update sensor baselines from readings.

        Args:
            readings: Sensor readings to process.
        """
        sensor_values: Dict[str, List[float]] = {}
        for r in readings:
            if r.is_valid:
                sensor_values.setdefault(r.sensor_id, []).append(r.value)

        for sensor_id, values in sensor_values.items():
            if len(values) >= 2:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                std = math.sqrt(variance)
                self._sensor_baselines[sensor_id] = (mean, std)

    def _compute_condition_factor(self, readings: List[SensorReading]) -> float:
        """Compute condition factor from sensor deviations.

        Args:
            readings: Sensor readings to analyze.

        Returns:
            Condition factor in [0, 1].
        """
        if not readings or not self._sensor_baselines:
            return 0.0

        deviations: List[float] = []
        for r in readings:
            if r.sensor_id in self._sensor_baselines and r.is_valid:
                mean, std = self._sensor_baselines[r.sensor_id]
                if std > 0:
                    deviation = abs(r.value - mean) / std
                    deviations.append(deviation)

        if not deviations:
            return 0.0

        avg_deviation = sum(deviations) / len(deviations)
        # Map deviation to [0, 1] using sigmoid-like function
        condition_factor = 1.0 - math.exp(-avg_deviation / 3.0)
        return max(0.0, min(1.0, condition_factor))

    def _identify_contributing_factors(
        self,
        equipment: Equipment,
        readings: List[SensorReading],
    ) -> List[str]:
        """Identify factors contributing to failure risk.

        Args:
            equipment: The equipment.
            readings: Sensor readings.

        Returns:
            List of contributing factor descriptions.
        """
        factors: List[str] = []

        # Age factor
        if equipment.age_years > equipment.expected_lifetime_years * 0.7:
            factors.append(f"Advanced age: {equipment.age_years:.1f} years")

        # Health index
        if equipment.health_index < 0.5:
            factors.append(f"Low health index: {equipment.health_index:.2f}")

        # Sensor anomalies
        recent_readings = readings[-10:] if len(readings) > 10 else readings
        for r in recent_readings:
            if r.sensor_id in self._sensor_baselines and r.is_valid:
                mean, std = self._sensor_baselines[r.sensor_id]
                if std > 0 and abs(r.value - mean) > 2 * std:
                    factors.append(f"Sensor {r.sensor_id} deviation: {r.value:.2f} {r.unit}")

        # Maintenance history
        if not equipment.maintenance_history:
            factors.append("No maintenance history")

        return factors

    def _recommend_action(self, failure_prob: float, factors: List[str]) -> str:
        """Recommend maintenance action based on failure probability.

        Args:
            failure_prob: Failure probability in [0, 1].
            factors: Contributing factors.

        Returns:
            Recommended action description.
        """
        if failure_prob > 0.8:
            return "Immediate inspection and potential replacement required"
        elif failure_prob > 0.6:
            return "Schedule maintenance within 2 weeks"
        elif failure_prob > 0.4:
            return "Schedule maintenance within 1 month"
        elif failure_prob > 0.2:
            return "Monitor closely and plan maintenance"
        else:
            return "Continue normal operation with routine monitoring"

    def update(self, equipment: Equipment, reading: SensorReading) -> None:
        """Update the model with new sensor data.

        Args:
            equipment: The equipment being monitored.
            reading: New sensor reading.
        """
        # Baselines are updated during predict()
        pass


class ProportionalHazardsPredictor(FailurePredictor):
    """Proportional hazards model for failure prediction.

    Uses a Cox proportional hazards approach with covariates
    from sensor data and equipment characteristics.
    """

    def __init__(self, baseline_hazard: float = 0.01) -> None:
        """Initialize the proportional hazards predictor.

        Args:
            baseline_hazard: Baseline hazard rate.
        """
        if baseline_hazard <= 0:
            raise ValueError("baseline_hazard must be positive")
        self._baseline_hazard = baseline_hazard
        self._covariate_effects: Dict[str, float] = {}

    def predict(
        self,
        equipment: Equipment,
        sensor_history: List[SensorReading],
    ) -> FailurePrediction:
        """Predict failure using proportional hazards model.

        Args:
            equipment: The equipment to analyze.
            sensor_history: Historical sensor readings.

        Returns:
            A FailurePrediction instance.
        """
        if not sensor_history:
            raise FailurePredictionError("Sensor history cannot be empty")

        # Compute risk score from covariates
        risk_score = self._compute_risk_score(equipment, sensor_history)

        # Survival probability
        age = equipment.age_years
        survival_prob = math.exp(-self._baseline_hazard * risk_score * age)
        failure_prob = 1.0 - survival_prob

        # Predicted failure time
        if failure_prob > 0.3:
            predicted_years = -math.log(1.0 - failure_prob) / (self._baseline_hazard * risk_score)
            predicted_time = equipment.installation_date + timedelta(days=predicted_years * 365.25)
        else:
            predicted_time = None

        factors = self._identify_risk_factors(equipment, sensor_history)
        action = self._recommend_action(failure_prob)

        return FailurePrediction(
            equipment_id=equipment.equipment_id,
            timestamp=datetime.now(),
            failure_probability=failure_prob,
            predicted_failure_time=predicted_time,
            confidence_interval_days=60.0,
            contributing_factors=factors,
            recommended_action=action,
            model_name="ProportionalHazards",
        )

    def _compute_risk_score(
        self,
        equipment: Equipment,
        readings: List[SensorReading],
    ) -> float:
        """Compute risk score from covariates.

        Args:
            equipment: The equipment.
            readings: Sensor readings.

        Returns:
            Risk score multiplier.
        """
        score = 1.0

        # Age effect
        age_ratio = equipment.age_years / equipment.expected_lifetime_years
        score *= 1.0 + age_ratio ** 2

        # Health index effect
        score *= 1.0 + (1.0 - equipment.health_index) * 2.0

        # Criticality effect
        score *= 1.0 + equipment.criticality

        # Sensor anomaly effect
        if readings:
            recent = readings[-5:]
            anomalies = sum(1 for r in recent if not r.is_valid)
            score *= 1.0 + anomalies * 0.2

        return score

    def _identify_risk_factors(
        self,
        equipment: Equipment,
        readings: List[SensorReading],
    ) -> List[str]:
        """Identify risk factors.

        Args:
            equipment: The equipment.
            readings: Sensor readings.

        Returns:
            List of risk factor descriptions.
        """
        factors: List[str] = []
        age_ratio = equipment.age_years / equipment.expected_lifetime_years

        if age_ratio > 0.7:
            factors.append(f"Age ratio: {age_ratio:.2f}")
        if equipment.health_index < 0.6:
            factors.append(f"Health index: {equipment.health_index:.2f}")
        if equipment.criticality > 0.7:
            factors.append(f"High criticality: {equipment.criticality:.2f}")

        invalid_count = sum(1 for r in readings if not r.is_valid)
        if invalid_count > 0:
            factors.append(f"Invalid readings: {invalid_count}")

        return factors

    def _recommend_action(self, failure_prob: float) -> str:
        """Recommend action based on failure probability.

        Args:
            failure_prob: Failure probability.

        Returns:
            Recommended action.
        """
        if failure_prob > 0.7:
            return "Urgent maintenance required"
        elif failure_prob > 0.4:
            return "Schedule maintenance soon"
        else:
            return "Continue monitoring"

    def update(self, equipment: Equipment, reading: SensorReading) -> None:
        """Update the model with new sensor data.

        Args:
            equipment: The equipment being monitored.
            reading: New sensor reading.
        """
        pass


# ---------------------------------------------------------------------------
# Anomaly Detection
# ---------------------------------------------------------------------------

class AnomalyDetector(ABC):
    """Abstract base class for anomaly detection algorithms."""

    @abstractmethod
    def detect(
        self,
        equipment: Equipment,
        readings: List[SensorReading],
    ) -> List[AnomalyEvent]:
        """Detect anomalies in sensor readings.

        Args:
            equipment: The equipment being monitored.
            readings: Sensor readings to analyze.

        Returns:
            List of detected AnomalyEvent instances.

        Raises:
            AnomalyDetectionError: If detection fails.
        """
        ...

    @abstractmethod
    def update_baseline(self, readings: List[SensorReading]) -> None:
        """Update the baseline model with new data.

        Args:
            readings: Sensor readings for baseline update.
        """
        ...


class StatisticalAnomalyDetector(AnomalyDetector):
    """Statistical anomaly detection using z-score and moving average.

    Detects anomalies based on statistical deviations from
    historical baselines.
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        window_size: int = 20,
        min_readings: int = 5,
    ) -> None:
        """Initialize the statistical anomaly detector.

        Args:
            z_threshold: Z-score threshold for anomaly detection.
            window_size: Moving average window size.
            min_readings: Minimum readings required for detection.
        """
        if z_threshold <= 0:
            raise ValueError("z_threshold must be positive")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if min_readings < 1:
            raise ValueError("min_readings must be >= 1")

        self._z_threshold = z_threshold
        self._window_size = window_size
        self._min_readings = min_readings
        self._baselines: Dict[str, Tuple[float, float]] = {}  # sensor_id -> (mean, std)
        self._history: Dict[str, List[float]] = {}  # sensor_id -> values

    def update_baseline(self, readings: List[SensorReading]) -> None:
        """Update baselines from readings.

        Args:
            readings: Sensor readings for baseline update.
        """
        for r in readings:
            if not r.is_valid:
                continue
            self._history.setdefault(r.sensor_id, []).append(r.value)
            if len(self._history[r.sensor_id]) > 1000:
                self._history[r.sensor_id] = self._history[r.sensor_id][-1000:]

        for sensor_id, values in self._history.items():
            if len(values) >= self._min_readings:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                std = math.sqrt(variance)
                self._baselines[sensor_id] = (mean, std)

    def detect(
        self,
        equipment: Equipment,
        readings: List[SensorReading],
    ) -> List[AnomalyEvent]:
        """Detect anomalies in sensor readings.

        Args:
            equipment: The equipment being monitored.
            readings: Sensor readings to analyze.

        Returns:
            List of detected AnomalyEvent instances.
        """
        if not readings:
            raise AnomalyDetectionError("No readings provided")

        # Update baselines
        self.update_baseline(readings)

        anomalies: List[AnomalyEvent] = []

        for r in readings:
            if not r.is_valid:
                # Invalid reading is an anomaly
                anomalies.append(AnomalyEvent(
                    event_id=f"anom_{r.sensor_id}_{r.timestamp.isoformat()}",
                    equipment_id=equipment.equipment_id,
                    timestamp=r.timestamp,
                    sensor_id=r.sensor_id,
                    anomaly_score=1.0,
                    description=f"Invalid reading from sensor {r.sensor_id}",
                    is_confirmed=True,
                    root_cause="Sensor malfunction",
                ))
                continue

            if r.sensor_id not in self._baselines:
                continue

            mean, std = self._baselines[r.sensor_id]
            if std <= 0:
                continue

            z_score = abs(r.value - mean) / std

            if z_score > self._z_threshold:
                # Anomaly detected
                anomaly_score = min(1.0, z_score / (self._z_threshold * 2))
                direction = "high" if r.value > mean else "low"

                anomalies.append(AnomalyEvent(
                    event_id=f"anom_{r.sensor_id}_{r.timestamp.isoformat()}",
                    equipment_id=equipment.equipment_id,
                    timestamp=r.timestamp,
                    sensor_id=r.sensor_id,
                    anomaly_score=anomaly_score,
                    description=(
                        f"Anomalous {direction} reading from {r.sensor_id}: "
                        f"{r.value:.2f} {r.unit} (z={z_score:.1f})"
                    ),
                    is_confirmed=False,
                ))

        return anomalies


class IsolationForestAnomalyDetector(AnomalyDetector):
    """Simplified isolation forest anomaly detector.

    Uses random partitioning to isolate anomalies. This is a simplified
    implementation suitable for real-time detection.
    """

    def __init__(
        self,
        n_trees: int = 10,
        sample_size: int = 256,
        contamination: float = 0.1,
    ) -> None:
        """Initialize the isolation forest detector.

        Args:
            n_trees: Number of isolation trees.
            sample_size: Sample size for each tree.
            contamination: Expected contamination fraction.
        """
        if n_trees < 1:
            raise ValueError("n_trees must be >= 1")
        if sample_size < 2:
            raise ValueError("sample_size must be >= 2")
        if not 0 < contamination < 0.5:
            raise ValueError("contamination must be in (0, 0.5)")

        self._n_trees = n_trees
        self._sample_size = sample_size
        self._contamination = contamination
        self._baseline_data: Dict[str, List[float]] = {}

    def update_baseline(self, readings: List[SensorReading]) -> None:
        """Update baseline data.

        Args:
            readings: Sensor readings for baseline update.
        """
        for r in readings:
            if r.is_valid:
                self._baseline_data.setdefault(r.sensor_id, []).append(r.value)
                if len(self._baseline_data[r.sensor_id]) > 1000:
                    self._baseline_data[r.sensor_id] = self._baseline_data[r.sensor_id][-1000:]

    def detect(
        self,
        equipment: Equipment,
        readings: List[SensorReading],
    ) -> List[AnomalyEvent]:
        """Detect anomalies using isolation-based scoring.

        Args:
            equipment: The equipment being monitored.
            readings: Sensor readings to analyze.

        Returns:
            List of detected AnomalyEvent instances.
        """
        if not readings:
            raise AnomalyDetectionError("No readings provided")

        self.update_baseline(readings)
        anomalies: List[AnomalyEvent] = []

        for r in readings:
            if not r.is_valid:
                anomalies.append(AnomalyEvent(
                    event_id=f"anom_{r.sensor_id}_{r.timestamp.isoformat()}",
                    equipment_id=equipment.equipment_id,
                    timestamp=r.timestamp,
                    sensor_id=r.sensor_id,
                    anomaly_score=1.0,
                    description=f"Invalid reading from sensor {r.sensor_id}",
                    is_confirmed=True,
                    root_cause="Sensor malfunction",
                ))
                continue

            if r.sensor_id not in self._baseline_data:
                continue

            baseline = self._baseline_data[r.sensor_id]
            if len(baseline) < self._min_samples():
                continue

            # Simplified isolation score: distance from k-nearest neighbors
            score = self._isolation_score(r.value, baseline)

            if score > 1.0 - self._contamination:
                anomalies.append(AnomalyEvent(
                    event_id=f"anom_{r.sensor_id}_{r.timestamp.isoformat()}",
                    equipment_id=equipment.equipment_id,
                    timestamp=r.timestamp,
                    sensor_id=r.sensor_id,
                    anomaly_score=min(1.0, score),
                    description=(
                        f"Isolation anomaly from {r.sensor_id}: "
                        f"{r.value:.2f} {r.unit} (score={score:.2f})"
                    ),
                ))

        return anomalies

    def _min_samples(self) -> int:
        """Minimum samples required for detection."""
        return max(5, self._sample_size // 4)

    def _isolation_score(self, value: float, baseline: List[float]) -> float:
        """Compute isolation score for a value.

        Args:
            value: The value to score.
            baseline: Baseline data.

        Returns:
            Isolation score in [0, 1].
        """
        # Simplified: use percentile-based scoring
        sorted_baseline = sorted(baseline)
        n = len(sorted_baseline)

        # Find position in sorted array
        pos = 0
        for i, v in enumerate(sorted_baseline):
            if value <= v:
                pos = i
                break
        else:
            pos = n

        # Score based on distance from median
        median = sorted_baseline[n // 2]
        max_dist = max(abs(sorted_baseline[0] - median), abs(sorted_baseline[-1] - median))

        if max_dist <= 0:
            return 0.0

        distance = abs(value - median) / max_dist
        return min(1.0, distance)


# ---------------------------------------------------------------------------
# Maintenance Scheduling
# ---------------------------------------------------------------------------

@dataclass
class MaintenanceSchedule:
    """A maintenance schedule.

    Attributes:
        schedule_id: Unique schedule identifier.
        tasks: List of scheduled maintenance tasks.
        start_date: Schedule start date.
        end_date: Schedule end date.
        total_estimated_cost: Total estimated cost in $.
        resource_utilization: Resource utilization fraction in [0, 1].
    """

    schedule_id: str
    tasks: List[MaintenanceTask]
    start_date: datetime
    end_date: datetime
    total_estimated_cost: float
    resource_utilization: float


class MaintenanceScheduler:
    """Schedules maintenance tasks based on predictions and constraints.

    Optimizes maintenance scheduling to minimize downtime and cost
    while respecting resource constraints.
    """

    def __init__(
        self,
        max_concurrent_tasks: int = 3,
        working_hours_per_day: float = 8.0,
        planning_horizon_days: int = 30,
    ) -> None:
        """Initialize the maintenance scheduler.

        Args:
            max_concurrent_tasks: Maximum concurrent maintenance tasks.
            working_hours_per_day: Working hours per day.
            planning_horizon_days: Planning horizon in days.
        """
        if max_concurrent_tasks < 1:
            raise ValueError("max_concurrent_tasks must be >= 1")
        if working_hours_per_day <= 0:
            raise ValueError("working_hours_per_day must be positive")
        if planning_horizon_days < 1:
            raise ValueError("planning_horizon_days must be >= 1")

        self._max_concurrent = max_concurrent_tasks
        self._working_hours = working_hours_per_day
        self._horizon_days = planning_horizon_days

    def create_schedule(
        self,
        equipment_list: List[Equipment],
        predictions: Dict[str, FailurePrediction],
        anomalies: Dict[str, List[AnomalyEvent]],
        start_date: Optional[datetime] = None,
    ) -> MaintenanceSchedule:
        """Create a maintenance schedule.

        Args:
            equipment_list: List of equipment to schedule maintenance for.
            predictions: Failure predictions by equipment_id.
            anomalies: Anomaly events by equipment_id.
            start_date: Optional schedule start date.

        Returns:
            A MaintenanceSchedule instance.

        Raises:
            SchedulingError: If scheduling fails.
        """
        start = start_date or datetime.now()
        tasks: List[MaintenanceTask] = []

        # Priority scoring for each equipment
        scored_equipment: List[Tuple[float, Equipment]] = []
        for eq in equipment_list:
            score = self._compute_priority_score(eq, predictions.get(eq.equipment_id), anomalies.get(eq.equipment_id, []))
            scored_equipment.append((score, eq))

        # Sort by priority score (highest first)
        scored_equipment.sort(key=lambda x: x[0], reverse=True)

        # Create tasks
        current_time = start
        for score, eq in scored_equipment:
            if score <= 0:
                continue

            pred = predictions.get(eq.equipment_id)
            pred_anomalies = anomalies.get(eq.equipment_id, [])

            # Determine maintenance type and priority
            if pred and pred.failure_probability > 0.7:
                maint_type = MaintenanceType.PREDICTIVE
                priority = MaintenancePriority.CRITICAL
            elif pred_anomalies:
                maint_type = MaintenanceType.CONDITION_BASED
                priority = MaintenancePriority.HIGH
            elif eq.health_index < 0.5:
                maint_type = MaintenanceType.PREVENTIVE
                priority = MaintenancePriority.MEDIUM
            else:
                maint_type = MaintenanceType.PREVENTIVE
                priority = MaintenancePriority.LOW

            # Estimate duration based on equipment type and condition
            duration = self._estimate_duration(eq, maint_type)

            # Schedule task
            task_start = current_time
            task_end = task_start + timedelta(hours=duration)

            task = MaintenanceTask(
                task_id=f"maint_{eq.equipment_id}_{task_start.isoformat()}",
                equipment_id=eq.equipment_id,
                maintenance_type=maint_type,
                priority=priority,
                description=f"{maint_type.value} maintenance for {eq.name}",
                scheduled_start=task_start,
                scheduled_end=task_end,
                estimated_duration_hours=duration,
                required_skills=self._get_required_skills(eq.equipment_type),
                required_parts=self._get_required_parts(eq.equipment_type),
            )
            tasks.append(task)

            # Advance time (respect concurrent limit)
            if len(tasks) % self._max_concurrent == 0:
                current_time = task_end

        if not tasks:
            raise SchedulingError("No maintenance tasks could be scheduled")

        end_date = max(t.scheduled_end for t in tasks)
        total_cost = sum(self._estimate_cost(t) for t in tasks)
        utilization = self._compute_utilization(tasks, start, end_date)

        return MaintenanceSchedule(
            schedule_id=f"schedule_{start.isoformat()}",
            tasks=tasks,
            start_date=start,
            end_date=end_date,
            total_estimated_cost=total_cost,
            resource_utilization=utilization,
        )

    def _compute_priority_score(
        self,
        equipment: Equipment,
        prediction: Optional[FailurePrediction],
        anomalies: List[AnomalyEvent],
    ) -> float:
        """Compute priority score for maintenance scheduling.

        Args:
            equipment: The equipment.
            prediction: Failure prediction (if any).
            anomalies: Anomaly events.

        Returns:
            Priority score in [0, 1].
        """
        score = 0.0

        # Failure probability contribution
        if prediction:
            score += prediction.failure_probability * 0.4

        # Anomaly contribution
        if anomalies:
            max_anomaly = max(a.anomaly_score for a in anomalies)
            score += max_anomaly * 0.3

        # Health index contribution
        score += (1.0 - equipment.health_index) * 0.2

        # Criticality contribution
        score += equipment.criticality * 0.1

        return min(1.0, score)

    def _estimate_duration(self, equipment: Equipment, maint_type: MaintenanceType) -> float:
        """Estimate maintenance duration in hours.

        Args:
            equipment: The equipment.
            maint_type: Type of maintenance.

        Returns:
            Estimated duration in hours.
        """
        base_duration = {
            EquipmentType.TRANSFORMER: 8.0,
            EquipmentType.TURBINE: 16.0,
            EquipmentType.GENERATOR: 12.0,
            EquipmentType.CIRCUIT_BREAKER: 4.0,
            EquipmentType.CABLE: 6.0,
            EquipmentType.SWITCHGEAR: 6.0,
            EquipmentType.BATTERY: 4.0,
            EquipmentType.SOLAR_INVERTER: 2.0,
            EquipmentType.WIND_TURBINE: 8.0,
            EquipmentType.PUMP: 4.0,
            EquipmentType.COMPRESSOR: 6.0,
            EquipmentType.MOTOR: 3.0,
        }

        duration = base_duration.get(equipment.equipment_type, 4.0)

        # Adjust for maintenance type
        if maint_type == MaintenanceType.EMERGENCY:
            duration *= 0.5  # Faster for emergencies
        elif maint_type == MaintenanceType.PREDICTIVE:
            duration *= 1.2  # More thorough

        # Adjust for equipment condition
        if equipment.health_index < 0.3:
            duration *= 1.5

        return duration

    def _get_required_skills(self, equipment_type: EquipmentType) -> List[str]:
        """Get required skills for maintenance.

        Args:
            equipment_type: Type of equipment.

        Returns:
            List of required skills.
        """
        skill_map = {
            EquipmentType.TRANSFORMER: ["electrical", "oil_analysis"],
            EquipmentType.TURBINE: ["mechanical", "vibration_analysis"],
            EquipmentType.GENERATOR: ["electrical", "mechanical"],
            EquipmentType.CIRCUIT_BREAKER: ["electrical"],
            EquipmentType.CABLE: ["electrical", "cable_termination"],
            EquipmentType.SWITCHGEAR: ["electrical"],
            EquipmentType.BATTERY: ["electrical", "chemical"],
            EquipmentType.SOLAR_INVERTER: ["electrical", "electronics"],
            EquipmentType.WIND_TURBINE: ["mechanical", "electrical", "climbing"],
            EquipmentType.PUMP: ["mechanical"],
            EquipmentType.COMPRESSOR: ["mechanical", "pneumatics"],
            EquipmentType.MOTOR: ["electrical", "mechanical"],
        }
        return skill_map.get(equipment_type, ["general"])

    def _get_required_parts(self, equipment_type: EquipmentType) -> List[str]:
        """Get likely required parts for maintenance.

        Args:
            equipment_type: Type of equipment.

        Returns:
            List of likely required parts.
        """
        parts_map = {
            EquipmentType.TRANSFORMER: ["insulating_oil", "gaskets", "bushings"],
            EquipmentType.TURBINE: ["blades", "bearings", "seals"],
            EquipmentType.GENERATOR: ["brushes", "bearings", "windings"],
            EquipmentType.CIRCUIT_BREAKER: ["contacts", "springs", "gas"],
            EquipmentType.CABLE: ["joints", "terminations"],
            EquipmentType.SWITCHGEAR: ["contacts", "insulators"],
            EquipmentType.BATTERY: ["cells", "electrolyte", "connectors"],
            EquipmentType.SOLAR_INVERTER: ["capacitors", "igbt_modules", "fans"],
            EquipmentType.WIND_TURBINE: ["blades", "gearbox", "generator"],
            EquipmentType.PUMP: ["seals", "impeller", "bearings"],
            EquipmentType.COMPRESSOR: ["valves", "pistons", "filters"],
            EquipmentType.MOTOR: ["bearings", "windings", "brushes"],
        }
        return parts_map.get(equipment_type, ["general_parts"])

    def _estimate_cost(self, task: MaintenanceTask) -> float:
        """Estimate maintenance task cost.

        Args:
            task: The maintenance task.

        Returns:
            Estimated cost in $.
        """
        # Base cost by priority
        base_cost = {
            MaintenancePriority.LOW: 1000.0,
            MaintenancePriority.MEDIUM: 2500.0,
            MaintenancePriority.HIGH: 5000.0,
            MaintenancePriority.CRITICAL: 10000.0,
            MaintenancePriority.EMERGENCY: 20000.0,
        }

        cost = base_cost.get(task.priority, 2000.0)
        cost += task.estimated_duration_hours * 150.0  # Labor cost
        cost += len(task.required_parts) * 500.0  # Parts cost

        return cost

    def _compute_utilization(
        self,
        tasks: List[MaintenanceTask],
        start: datetime,
        end: datetime,
    ) -> float:
        """Compute resource utilization.

        Args:
            tasks: Scheduled tasks.
            start: Schedule start.
            end: Schedule end.

        Returns:
            Utilization fraction in [0, 1].
        """
        total_hours = (end - start).total_seconds() / 3600.0
        if total_hours <= 0:
            return 0.0

        task_hours = sum(t.estimated_duration_hours for t in tasks)
        available_hours = total_hours * self._max_concurrent

        return min(1.0, task_hours / available_hours) if available_hours > 0 else 0.0


# ---------------------------------------------------------------------------
# Predictive Maintenance Controller (orchestrator)
# ---------------------------------------------------------------------------

class PredictiveMaintenanceController:
    """High-level predictive maintenance controller.

    Coordinates failure prediction, anomaly detection, and maintenance
    scheduling into a unified system.
    """

    def __init__(
        self,
        failure_predictor: Optional[FailurePredictor] = None,
        anomaly_detector: Optional[AnomalyDetector] = None,
        maintenance_scheduler: Optional[MaintenanceScheduler] = None,
    ) -> None:
        """Initialize the predictive maintenance controller.

        Args:
            failure_predictor: Failure prediction subsystem.
            anomaly_detector: Anomaly detection subsystem.
            maintenance_scheduler: Maintenance scheduling subsystem.
        """
        self._failure_predictor = failure_predictor or WeibullFailurePredictor()
        self._anomaly_detector = anomaly_detector or StatisticalAnomalyDetector()
        self._maintenance_scheduler = maintenance_scheduler or MaintenanceScheduler()
        self._equipment: Dict[str, Equipment] = {}
        self._sensor_history: Dict[str, List[SensorReading]] = {}
        self._anomaly_history: Dict[str, List[AnomalyEvent]] = {}

    @property
    def equipment(self) -> Dict[str, Equipment]:
        """Equipment under monitoring."""
        return self._equipment

    def add_equipment(self, equipment: Equipment) -> None:
        """Add equipment for monitoring.

        Args:
            equipment: The equipment to add.
        """
        self._equipment[equipment.equipment_id] = equipment
        self._sensor_history.setdefault(equipment.equipment_id, [])
        self._anomaly_history.setdefault(equipment.equipment_id, [])

    def add_sensor_reading(self, reading: SensorReading) -> None:
        """Add a sensor reading.

        Args:
            reading: The sensor reading to add.
        """
        self._sensor_history.setdefault(reading.equipment_id, []).append(reading)
        # Keep last 1000 readings per equipment
        if len(self._sensor_history[reading.equipment_id]) > 1000:
            self._sensor_history[reading.equipment_id] = self._sensor_history[reading.equipment_id][-1000:]

    def analyze_equipment(self, equipment_id: str) -> Dict[str, Any]:
        """Run full analysis on a piece of equipment.

        Args:
            equipment_id: ID of the equipment to analyze.

        Returns:
            Dict with keys 'prediction' and 'anomalies'.

        Raises:
            PredictiveMaintenanceError: If equipment not found.
        """
        if equipment_id not in self._equipment:
            raise PredictiveMaintenanceError(f"Equipment {equipment_id} not found")

        equipment = self._equipment[equipment_id]
        readings = self._sensor_history.get(equipment_id, [])

        if not readings:
            raise PredictiveMaintenanceError(f"No sensor readings for {equipment_id}")

        # Failure prediction
        prediction = self._failure_predictor.predict(equipment, readings)

        # Anomaly detection
        anomalies = self._anomaly_detector.detect(equipment, readings)
        self._anomaly_history[equipment_id].extend(anomalies)

        return {
            "prediction": prediction,
            "anomalies": anomalies,
        }

    def create_maintenance_schedule(
        self,
        equipment_ids: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
    ) -> MaintenanceSchedule:
        """Create a maintenance schedule.

        Args:
            equipment_ids: Optional list of equipment IDs to schedule.
            start_date: Optional schedule start date.

        Returns:
            A MaintenanceSchedule instance.
        """
        if equipment_ids:
            equipment_list = [self._equipment[eid] for eid in equipment_ids if eid in self._equipment]
        else:
            equipment_list = list(self._equipment.values())

        if not equipment_list:
            raise SchedulingError("No equipment available for scheduling")

        # Run analysis on all equipment
        predictions: Dict[str, FailurePrediction] = {}
        anomalies: Dict[str, List[AnomalyEvent]] = {}

        for eq in equipment_list:
            try:
                result = self.analyze_equipment(eq.equipment_id)
                predictions[eq.equipment_id] = result["prediction"]
                anomalies[eq.equipment_id] = self._anomaly_history.get(eq.equipment_id, [])
            except PredictiveMaintenanceError as e:
                logger.warning("Analysis failed for %s: %s", eq.equipment_id, e)

        return self._maintenance_scheduler.create_schedule(
            equipment_list, predictions, anomalies, start_date
        )

    def get_equipment_health_report(self) -> Dict[str, Dict[str, Any]]:
        """Get health report for all equipment.

        Returns:
            Dict of equipment_id -> health report.
        """
        report: Dict[str, Dict[str, Any]] = {}

        for eq_id, eq in self._equipment.items():
            readings = self._sensor_history.get(eq_id, [])
            anomaly_count = len(self._anomaly_history.get(eq_id, []))

            report[eq_id] = {
                "equipment_id": eq_id,
                "name": eq.name,
                "type": eq.equipment_type.value,
                "status": eq.current_status.value,
                "health_index": eq.health_index,
                "criticality": eq.criticality,
                "age_years": eq.age_years,
                "remaining_life_fraction": eq.remaining_life_fraction,
                "sensor_reading_count": len(readings),
                "anomaly_count": anomaly_count,
                "last_maintenance": eq.maintenance_history[-1] if eq.maintenance_history else None,
            }

        return report
