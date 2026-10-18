# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Backend-neutral orchestration for staged autotuning candidates.

The legacy :class:`tilelang.autotuner.AutoTuner` owns CUDA/HIP tensor
generation, validation, and timing.  Those assumptions do not hold for every
backend.  This module keeps configuration enumeration in TileLang while
delegating build, validation, timing, timeout enforcement, and device safety
to a backend-specific runner.

The orchestrator is deliberately sequential.  A runner may parallelize a
stage when its toolchain is known to be safe, but the common layer must not
concurrently enter a device runtime by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Callable, Dict, Iterable, Literal, Mapping, Optional, Protocol, Tuple

CandidateStatus = Literal[
    "success",
    "build_failed",
    "validation_failed",
    "measurement_failed",
]


@dataclass(frozen=True)
class ValidationResult:
    """Result returned by a runner after checking one built candidate."""

    passed: bool
    metrics: Mapping[str, Any] = field(default_factory=dict)
    message: Optional[str] = None


@dataclass(frozen=True)
class TimingResult:
    """One candidate's measured latency and optional raw samples."""

    latency_ms: float
    samples_ms: Tuple[float, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("latency_ms must be a finite non-negative number")
        if any(not math.isfinite(sample) or sample < 0 for sample in self.samples_ms):
            raise ValueError("samples_ms must contain only finite non-negative numbers")


class CandidateRunner(Protocol):
    """Backend contract used by :func:`run_candidates`.

    Implementations are responsible for enforcing ``timeout_s``.  A TPU
    implementation should normally satisfy that contract with an isolated
    subprocess and process-group termination rather than a Python thread.
    Validation cases, target identity, working directories, and device locks
    belong to the runner instance so they cannot vary silently between
    candidates.
    """

    def build(
        self,
        kernel_factory: Callable[..., Any],
        config: Mapping[str, Any],
        *,
        timeout_s: float,
    ) -> Any:
        """Build one candidate and return an opaque backend-owned handle."""

    def validate(self, candidate: Any, *, timeout_s: float) -> ValidationResult:
        """Check numerical and runtime validity for one built candidate."""

    def measure(
        self,
        candidate: Any,
        *,
        warmup: int,
        rep: int,
        timeout_s: float,
    ) -> TimingResult:
        """Measure one valid candidate using backend-appropriate synchronization."""

    def close(self) -> None:
        """Release runner-owned resources after every candidate was attempted."""


@dataclass(frozen=True)
class CandidateRecord:
    """Complete outcome of one requested configuration."""

    config: Mapping[str, Any]
    status: CandidateStatus
    build_seconds: float
    validation_seconds: float
    measurement_seconds: float
    validation: Optional[ValidationResult] = None
    timing: Optional[TimingResult] = None
    candidate: Any = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class StagedAutotuneResult:
    """All candidate outcomes and the fastest valid candidate, if one exists."""

    records: Tuple[CandidateRecord, ...]
    best_config: Optional[Mapping[str, Any]]
    best_latency_ms: Optional[float]
    best_candidate: Any = None
    runner_close_error: Optional[str] = None

    @property
    def successful_candidates(self) -> int:
        return sum(record.status == "success" for record in self.records)

    def require_best(self) -> "StagedAutotuneResult":
        """Raise after preserving records when selection is not safe to use."""

        if self.runner_close_error is not None:
            raise RuntimeError(f"Autotuning runner cleanup failed: {self.runner_close_error}")
        if self.best_config is None:
            counts: Dict[str, int] = {}
            for record in self.records:
                counts[record.status] = counts.get(record.status, 0) + 1
            raise RuntimeError(f"No autotuning candidate succeeded; outcomes={counts}")
        return self


def _elapsed(start: float) -> float:
    return time.monotonic() - start


def _error_fields(error: BaseException) -> Tuple[str, str]:
    return type(error).__name__, str(error)


def run_candidates(
    kernel_factory: Callable[..., Any],
    configs: Iterable[Mapping[str, Any]],
    runner: CandidateRunner,
    *,
    warmup: int = 25,
    rep: int = 100,
    timeout_s: float = 100.0,
) -> StagedAutotuneResult:
    """Evaluate every configuration through a backend-specific runner.

    Failed configurations remain in ``records`` and therefore remain visible
    in the tuning budget.  Validation failures never proceed to measurement.
    The runner is closed even if a stage raises.
    """

    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if isinstance(rep, bool) or not isinstance(rep, int) or rep <= 0:
        raise ValueError("rep must be a positive integer")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or \
            not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be a finite positive number")

    records = []
    best_record: Optional[CandidateRecord] = None
    close_error = None

    try:
        for original_config in configs:
            if not isinstance(original_config, Mapping):
                raise TypeError("each autotuning configuration must be a mapping")
            config = dict(original_config)
            build_seconds = 0.0
            validation_seconds = 0.0
            measurement_seconds = 0.0

            start = time.monotonic()
            try:
                candidate = runner.build(kernel_factory, config, timeout_s=float(timeout_s))
                build_seconds = _elapsed(start)
            except Exception as error:  # backend errors are candidate outcomes
                build_seconds = _elapsed(start)
                error_type, error_message = _error_fields(error)
                records.append(
                    CandidateRecord(
                        config=config,
                        status="build_failed",
                        build_seconds=build_seconds,
                        validation_seconds=0.0,
                        measurement_seconds=0.0,
                        error_type=error_type,
                        error_message=error_message,
                    ))
                continue

            start = time.monotonic()
            try:
                validation = runner.validate(candidate, timeout_s=float(timeout_s))
                validation_seconds = _elapsed(start)
                if not isinstance(validation, ValidationResult):
                    raise TypeError("runner.validate must return ValidationResult")
            except Exception as error:
                validation_seconds = _elapsed(start)
                error_type, error_message = _error_fields(error)
                records.append(
                    CandidateRecord(
                        config=config,
                        status="validation_failed",
                        build_seconds=build_seconds,
                        validation_seconds=validation_seconds,
                        measurement_seconds=0.0,
                        candidate=candidate,
                        error_type=error_type,
                        error_message=error_message,
                    ))
                continue

            if not validation.passed:
                records.append(
                    CandidateRecord(
                        config=config,
                        status="validation_failed",
                        build_seconds=build_seconds,
                        validation_seconds=validation_seconds,
                        measurement_seconds=0.0,
                        validation=validation,
                        candidate=candidate,
                        error_message=validation.message,
                    ))
                continue

            start = time.monotonic()
            try:
                timing = runner.measure(
                    candidate,
                    warmup=warmup,
                    rep=rep,
                    timeout_s=float(timeout_s),
                )
                measurement_seconds = _elapsed(start)
                if not isinstance(timing, TimingResult):
                    raise TypeError("runner.measure must return TimingResult")
            except Exception as error:
                measurement_seconds = _elapsed(start)
                error_type, error_message = _error_fields(error)
                records.append(
                    CandidateRecord(
                        config=config,
                        status="measurement_failed",
                        build_seconds=build_seconds,
                        validation_seconds=validation_seconds,
                        measurement_seconds=measurement_seconds,
                        validation=validation,
                        candidate=candidate,
                        error_type=error_type,
                        error_message=error_message,
                    ))
                continue

            record = CandidateRecord(
                config=config,
                status="success",
                build_seconds=build_seconds,
                validation_seconds=validation_seconds,
                measurement_seconds=measurement_seconds,
                validation=validation,
                timing=timing,
                candidate=candidate,
            )
            records.append(record)
            if best_record is None or timing.latency_ms < best_record.timing.latency_ms:
                best_record = record
    finally:
        try:
            runner.close()
        except Exception as error:  # preserve completed candidate records for diagnosis
            close_error = f"{type(error).__name__}: {error}"

    return StagedAutotuneResult(
        records=tuple(records),
        best_config=None if best_record is None else best_record.config,
        best_latency_ms=None if best_record is None else best_record.timing.latency_ms,
        best_candidate=None if best_record is None else best_record.candidate,
        runner_close_error=close_error,
    )
