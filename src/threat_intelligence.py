"""
Apex Critical Infrastructure — Threat Intelligence Module.

Threat hunting, IOC management, and attack surface analysis
for defense and security operations.

Classes:
    ThreatHunter: Proactive threat hunting across telemetry.
    IOCManager: Indicator of Compromise lifecycle management.
    AttackSurfaceAnalyzer: Continuous attack surface assessment.
    IOC: Indicator of Compromise record.
    ThreatHypothesis: Active threat hunting hypothesis.
    AttackSurfaceReport: Aggregated attack surface findings.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class IOCType(Enum):
    """Types of Indicators of Compromise."""

    IP = auto()
    DOMAIN = auto()
    URL = auto()
    FILE_HASH = auto()
    EMAIL = auto()
    MUTEX = auto()
    REGISTRY_KEY = auto()
    YARA_RULE = auto()
    SNORT_SIGNATURE = auto()
    CVE = auto()


class IOCConfidence(Enum):
    """Confidence level for an IOC."""

    CONFIRMED = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    UNVERIFIED = auto()


class ThreatHuntStatus(Enum):
    """Status of a threat hunting hypothesis."""

    ACTIVE = auto()
    CONFIRMED = auto()
    DISPROVEN = auto()
    EXPIRED = auto()


class ExposureLevel(Enum):
    """Attack surface exposure classification."""

    CRITICAL = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    NONE = auto()


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class IOC:
    """Indicator of Compromise record.

    Attributes:
        ioc_id: Unique IOC identifier.
        value: The indicator value (IP, hash, domain, etc.).
        ioc_type: Classification of the indicator.
        confidence: Confidence level.
        source: Origin of the IOC.
        created_at: Creation timestamp.
        expires_at: Optional expiration timestamp.
        metadata: Additional contextual information.
        tags: Classification tags.
    """

    ioc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    value: str = ""
    ioc_type: IOCType = IOCType.IP
    confidence: IOCConfidence = IOCConfidence.UNVERIFIED
    source: str = "unknown"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def is_expired(self) -> bool:
        """Check if the IOC has expired.

        Returns:
            True if the IOC has an expiration date in the past.
        """
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 fingerprint of the IOC.

        Returns:
            Hex digest of the IOC value and type.
        """
        canonical = f"{self.ioc_type.name}:{self.value}"
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(slots=True)
class ThreatHypothesis:
    """Active threat hunting hypothesis.

    Attributes:
        hypothesis_id: Unique hypothesis identifier.
        description: Human-readable hypothesis description.
        query: Hunt query or detection logic.
        status: Current hunt status.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        matches: Number of matches found.
        confidence: Current confidence in the hypothesis.
        tags: Classification tags.
    """

    hypothesis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    query: str = ""
    status: ThreatHuntStatus = ThreatHuntStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    matches: int = 0
    confidence: float = 0.0
    tags: list[str] = field(default_factory=list)

    def update_status(self, status: ThreatHuntStatus) -> None:
        """Update the hypothesis status.

        Args:
            status: New hunt status.
        """
        self.status = status
        self.updated_at = datetime.now(timezone.utc)

    def record_matches(self, count: int) -> None:
        """Record the number of matches found.

        Args:
            count: Number of matches.
        """
        self.matches = count
        self.updated_at = datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class AttackSurfaceReport:
    """Aggregated attack surface findings.

    Attributes:
        timestamp: Report generation timestamp.
        total_exposed_services: Count of exposed services.
        critical_findings: List of critical exposure findings.
        high_findings: List of high exposure findings.
        medium_findings: List of medium exposure findings.
        low_findings: List of low exposure findings.
        recommendations: Prioritized remediation recommendations.
        overall_risk_score: Aggregate risk score (0-1).
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_exposed_services: int = 0
    critical_findings: list[dict[str, Any]] = field(default_factory=list)
    high_findings: list[dict[str, Any]] = field(default_factory=list)
    medium_findings: list[dict[str, Any]] = field(default_factory=list)
    low_findings: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    overall_risk_score: float = 0.0


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class HuntDataSource(Protocol):
    """Protocol for threat hunting data sources."""

    def search(self, query: str) -> list[dict[str, Any]]:
        """Search the data source with a query.

        Args:
            query: Hunt query string.

        Returns:
            List of matching records.
        """
        ...


# ---------------------------------------------------------------------------
# Threat Hunter
# ---------------------------------------------------------------------------


class ThreatHunter:
    """Proactive threat hunting across telemetry.

    Manages hunting hypotheses, executes queries against data
    sources, and tracks hunt outcomes.

    Attributes:
        data_sources: Registered hunt data sources.
        hypotheses: All known hunting hypotheses.
    """

    def __init__(self) -> None:
        """Initialize the threat hunter."""
        self.data_sources: dict[str, HuntDataSource] = {}
        self.hypotheses: dict[str, ThreatHypothesis] = {}

    def register_data_source(self, name: str, source: HuntDataSource) -> None:
        """Register a hunt data source.

        Args:
            name: Source identifier.
            source: Data source implementing HuntDataSource.
        """
        self.data_sources[name] = source
        logger.debug("Registered hunt data source: %s", name)

    def create_hypothesis(
        self, description: str, query: str, tags: list[str] | None = None
    ) -> ThreatHypothesis:
        """Create a new threat hunting hypothesis.

        Args:
            description: Human-readable hypothesis description.
            query: Hunt query or detection logic.
            tags: Classification tags.

        Returns:
            The newly created ThreatHypothesis.
        """
        hypothesis = ThreatHypothesis(
            description=description, query=query, tags=tags or []
        )
        self.hypotheses[hypothesis.hypothesis_id] = hypothesis
        logger.info(
            "Created threat hypothesis %s: %s",
            hypothesis.hypothesis_id,
            description,
        )
        return hypothesis

    def execute_hunt(self, hypothesis_id: str) -> ThreatHypothesis:
        """Execute a hunt hypothesis against all data sources.

        Args:
            hypothesis_id: The hypothesis to execute.

        Returns:
            Updated ThreatHypothesis with match results.

        Raises:
            KeyError: If the hypothesis is not found.
        """
        hypothesis = self.hypotheses.get(hypothesis_id)
        if hypothesis is None:
            raise KeyError(f"Hypothesis not found: {hypothesis_id}")

        total_matches = 0
        for name, source in self.data_sources.items():
            try:
                results = source.search(hypothesis.query)
                total_matches += len(results)
                logger.debug(
                    "Hunt %s on %s: %d matches",
                    hypothesis_id,
                    name,
                    len(results),
                )
            except Exception as exc:
                logger.warning(
                    "Hunt source %s failed for %s: %s", name, hypothesis_id, exc
                )

        hypothesis.record_matches(total_matches)

        # Auto-update status based on matches
        if total_matches > 0 and hypothesis.status == ThreatHuntStatus.ACTIVE:
            hypothesis.confidence = min(1.0, total_matches / 100.0)
        elif total_matches == 0 and hypothesis.status == ThreatHuntStatus.ACTIVE:
            hypothesis.confidence = max(0.0, hypothesis.confidence - 0.1)

        return hypothesis

    def get_active_hypotheses(self) -> list[ThreatHypothesis]:
        """Return all active hunting hypotheses.

        Returns:
            List of active hypotheses.
        """
        return [
            h for h in self.hypotheses.values() if h.status == ThreatHuntStatus.ACTIVE
        ]

    def expire_hypothesis(self, hypothesis_id: str) -> ThreatHypothesis:
        """Mark a hypothesis as expired.

        Args:
            hypothesis_id: The hypothesis to expire.

        Returns:
            The expired ThreatHypothesis.

        Raises:
            KeyError: If the hypothesis is not found.
        """
        hypothesis = self.hypotheses.get(hypothesis_id)
        if hypothesis is None:
            raise KeyError(f"Hypothesis not found: {hypothesis_id}")
        hypothesis.update_status(ThreatHuntStatus.EXPIRED)
        return hypothesis


# ---------------------------------------------------------------------------
# IOC Manager
# ---------------------------------------------------------------------------


class IOCManager:
    """Indicator of Compromise lifecycle management.

    Stores, deduplicates, matches, and expires IOCs with
    full provenance tracking.

    Attributes:
        iocs: All known IOCs by ID.
        _value_index: Index of IOC values for fast lookup.
    """

    def __init__(self) -> None:
        """Initialize the IOC manager."""
        self.iocs: dict[str, IOC] = {}
        self._value_index: dict[str, str] = {}  # fingerprint -> ioc_id

    def add_ioc(
        self,
        value: str,
        ioc_type: IOCType,
        confidence: IOCConfidence = IOCConfidence.UNVERIFIED,
        source: str = "unknown",
        expires_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> IOC:
        """Add an IOC, deduplicating by value and type.

        Args:
            value: The indicator value.
            ioc_type: Classification of the indicator.
            confidence: Confidence level.
            source: Origin of the IOC.
            expires_at: Optional expiration timestamp.
            metadata: Additional contextual information.
            tags: Classification tags.

        Returns:
            The added or existing IOC.
        """
        canonical = f"{ioc_type.name}:{value}"
        fingerprint = hashlib.sha256(canonical.encode()).hexdigest()

        if fingerprint in self._value_index:
            existing_id = self._value_index[fingerprint]
            existing = self.iocs[existing_id]
            # Upgrade confidence if new source is more confident
            if confidence.value < existing.confidence.value:
                existing.confidence = confidence
            existing.tags = list(set(existing.tags + (tags or [])))
            logger.debug("Deduplicated IOC: %s", value)
            return existing

        ioc = IOC(
            value=value,
            ioc_type=ioc_type,
            confidence=confidence,
            source=source,
            expires_at=expires_at,
            metadata=metadata or {},
            tags=tags or [],
        )
        self.iocs[ioc.ioc_id] = ioc
        self._value_index[fingerprint] = ioc.ioc_id
        logger.info("Added IOC %s (%s): %s", ioc.ioc_id, ioc_type.name, value)
        return ioc

    def match(self, value: str, ioc_type: IOCType | None = None) -> list[IOC]:
        """Find IOCs matching a value.

        Args:
            value: The value to match.
            ioc_type: Optional type filter.

        Returns:
            List of matching IOCs.
        """
        results: list[IOC] = []
        for ioc in self.iocs.values():
            if ioc.value == value:
                if ioc_type is None or ioc.ioc_type == ioc_type:
                    if not ioc.is_expired():
                        results.append(ioc)
        return results

    def bulk_add(self, iocs: list[dict[str, Any]]) -> list[IOC]:
        """Add multiple IOCs from raw dictionaries.

        Args:
            iocs: List of IOC dictionaries with keys matching add_ioc params.

        Returns:
            List of added or existing IOCs.
        """
        results: list[IOC] = []
        for ioc_data in iocs:
            ioc = self.add_ioc(**ioc_data)
            results.append(ioc)
        return results

    def get_by_type(self, ioc_type: IOCType) -> list[IOC]:
        """Get all IOCs of a specific type.

        Args:
            ioc_type: The type to filter by.

        Returns:
            List of matching IOCs.
        """
        return [ioc for ioc in self.iocs.values() if ioc.ioc_type == ioc_type]

    def get_by_confidence(self, confidence: IOCConfidence) -> list[IOC]:
        """Get all IOCs at or above a confidence level.

        Args:
            confidence: Minimum confidence level.

        Returns:
            List of matching IOCs.
        """
        return [
            ioc
            for ioc in self.iocs.values()
            if ioc.confidence.value <= confidence.value
        ]

    def expire_old_iocs(self) -> int:
        """Remove all expired IOCs from the manager.

        Returns:
            Number of IOCs removed.
        """
        expired = [ioc_id for ioc_id, ioc in self.iocs.items() if ioc.is_expired()]
        for ioc_id in expired:
            ioc = self.iocs.pop(ioc_id)
            canonical = f"{ioc.ioc_type.name}:{ioc.value}"
            fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
            self._value_index.pop(fingerprint, None)
        if expired:
            logger.info("Expired %d IOCs", len(expired))
        return len(expired)

    def stats(self) -> dict[str, int]:
        """Return IOC statistics by type.

        Returns:
            Mapping of IOC type names to counts.
        """
        stats: dict[str, int] = {}
        for ioc in self.iocs.values():
            key = ioc.ioc_type.name
            stats[key] = stats.get(key, 0) + 1
        return stats


# ---------------------------------------------------------------------------
# Attack Surface Analyzer
# ---------------------------------------------------------------------------


class AttackSurfaceAnalyzer:
    """Continuous attack surface assessment.

    Discovers and classifies exposed services, identifies
    vulnerabilities, and produces prioritized recommendations.

    Attributes:
        _findings: Current attack surface findings.
        _scan_history: Historical scan results.
    """

    def __init__(self) -> None:
        """Initialize the attack surface analyzer."""
        self._findings: list[dict[str, Any]] = []
        self._scan_history: list[dict[str, Any]] = []

    def scan(
        self, targets: list[str], ports: list[int] | None = None
    ) -> AttackSurfaceReport:
        """Perform an attack surface scan.

        Args:
            targets: Target hosts or networks to scan.
            ports: Optional list of ports to check.

        Returns:
            AttackSurfaceReport with findings.
        """
        ports = ports or [21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 993, 995, 3306, 3389, 5432, 8080, 8443]
        findings: list[dict[str, Any]] = []

        for target in targets:
            for port in ports:
                finding = self._probe(target, port)
                if finding is not None:
                    findings.append(finding)

        self._findings = findings
        self._scan_history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "targets": targets,
                "findings_count": len(findings),
            }
        )

        return self._generate_report(findings)

    def _probe(self, target: str, port: int) -> dict[str, Any] | None:
        """Probe a target port for exposure.

        Args:
            target: Target host.
            port: Port to probe.

        Returns:
            Finding dictionary if exposed, None otherwise.
        """
        # In production, this would perform actual network probing
        # For now, return a simulated finding structure
        risk_map: dict[int, ExposureLevel] = {
            21: ExposureLevel.HIGH,
            23: ExposureLevel.CRITICAL,
            25: ExposureLevel.MEDIUM,
            53: ExposureLevel.MEDIUM,
            80: ExposureLevel.MEDIUM,
            443: ExposureLevel.LOW,
            445: ExposureLevel.HIGH,
            3389: ExposureLevel.HIGH,
            3306: ExposureLevel.HIGH,
            5432: ExposureLevel.HIGH,
            8080: ExposureLevel.MEDIUM,
        }
        exposure = risk_map.get(port, ExposureLevel.LOW)
        if exposure == ExposureLevel.NONE:
            return None
        return {
            "target": target,
            "port": port,
            "exposure": exposure,
            "service": self._guess_service(port),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _guess_service(self, port: int) -> str:
        """Guess service name from port number.

        Args:
            port: Port number.

        Returns:
            Service name string.
        """
        services: dict[int, str] = {
            21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
            53: "dns", 80: "http", 110: "pop3", 143: "imap",
            443: "https", 445: "smb", 993: "imaps", 995: "pop3s",
            3306: "mysql", 3389: "rdp", 5432: "postgresql",
            8080: "http-proxy", 8443: "https-alt",
        }
        return services.get(port, "unknown")

    def _generate_report(self, findings: list[dict[str, Any]]) -> AttackSurfaceReport:
        """Generate an attack surface report from findings.

        Args:
            findings: Raw scan findings.

        Returns:
            AttackSurfaceReport with classified findings.
        """
        critical = [f for f in findings if f["exposure"] == ExposureLevel.CRITICAL]
        high = [f for f in findings if f["exposure"] == ExposureLevel.HIGH]
        medium = [f for f in findings if f["exposure"] == ExposureLevel.MEDIUM]
        low = [f for f in findings if f["exposure"] == ExposureLevel.LOW]

        # Calculate overall risk score
        weights = {
            ExposureLevel.CRITICAL: 1.0,
            ExposureLevel.HIGH: 0.7,
            ExposureLevel.MEDIUM: 0.4,
            ExposureLevel.LOW: 0.1,
        }
        total_weight = sum(weights[f["exposure"]] for f in findings)
        max_possible = len(findings) * 1.0 if findings else 1.0
        risk_score = min(1.0, total_weight / max_possible)

        recommendations = self._generate_recommendations(critical, high, medium)

        return AttackSurfaceReport(
            total_exposed_services=len(findings),
            critical_findings=critical,
            high_findings=high,
            medium_findings=medium,
            low_findings=low,
            recommendations=recommendations,
            overall_risk_score=risk_score,
        )

    def _generate_recommendations(
        self,
        critical: list[dict[str, Any]],
        high: list[dict[str, Any]],
        medium: list[dict[str, Any]],
    ) -> list[str]:
        """Generate prioritized remediation recommendations.

        Args:
            critical: Critical findings.
            high: High findings.
            medium: Medium findings.

        Returns:
            Prioritized recommendation strings.
        """
        recommendations: list[str] = []
        for f in critical:
            recommendations.append(
                f"CRITICAL: Close or secure {f['service']} on {f['target']}:{f['port']} immediately"
            )
        for f in high:
            recommendations.append(
                f"HIGH: Review access controls for {f['service']} on {f['target']}:{f['port']}"
            )
        for f in medium:
            recommendations.append(
                f"MEDIUM: Harden configuration of {f['service']} on {f['target']}:{f['port']}"
            )
        return recommendations

    def get_current_findings(self) -> list[dict[str, Any]]:
        """Return the most recent scan findings.

        Returns:
            List of finding dictionaries.
        """
        return list(self._findings)

    def scan_history(self) -> list[dict[str, Any]]:
        """Return the scan history.

        Returns:
            List of historical scan summaries.
        """
        return list(self._scan_history)
