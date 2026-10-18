# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import pytest

from tilelang.autotuner import AutoTuner
from tilelang.autotuner.candidate_runner import (
    TimingResult,
    ValidationResult,
    run_candidates,
)


class _FakeRunner:

    def __init__(self):
        self.calls = []
        self.closed = False

    def build(self, kernel_factory, config, *, timeout_s):
        self.calls.append(("build", config["case"], timeout_s))
        if config["case"] == "build_error":
            raise RuntimeError("synthetic build failure")
        return {"config": dict(config), "kernel": kernel_factory(**config)}

    def validate(self, candidate, *, timeout_s):
        case = candidate["config"]["case"]
        self.calls.append(("validate", case, timeout_s))
        if case == "validation_error":
            raise TimeoutError("synthetic validation timeout")
        return ValidationResult(
            passed=case != "wrong_answer",
            metrics={"checked": 4},
            message="mismatch" if case == "wrong_answer" else None,
        )

    def measure(self, candidate, *, warmup, rep, timeout_s):
        case = candidate["config"]["case"]
        self.calls.append(("measure", case, warmup, rep, timeout_s))
        if case == "measurement_error":
            raise RuntimeError("synthetic measurement failure")
        latency = {"slow": 2.0, "fast": 1.0}[case]
        return TimingResult(latency_ms=latency, samples_ms=(latency, latency))

    def close(self):
        self.closed = True


class _CloseFailureRunner(_FakeRunner):

    def close(self):
        raise RuntimeError("synthetic cleanup failure")


def _kernel_factory(**config):
    return tuple(sorted(config.items()))


def test_run_candidates_retains_failures_and_selects_fastest():
    configs = [{
        "case": case
    } for case in (
        "build_error",
        "validation_error",
        "wrong_answer",
        "measurement_error",
        "slow",
        "fast",
    )]
    runner = _FakeRunner()

    result = run_candidates(
        _kernel_factory,
        configs,
        runner,
        warmup=3,
        rep=7,
        timeout_s=11,
    )

    assert [record.status for record in result.records] == [
        "build_failed",
        "validation_failed",
        "validation_failed",
        "measurement_failed",
        "success",
        "success",
    ]
    assert result.best_config == {"case": "fast"}
    assert result.best_latency_ms == 1.0
    assert result.successful_candidates == 2
    assert result.records[2].validation.metrics == {"checked": 4}
    assert result.records[2].error_message == "mismatch"
    assert not any(call[0] == "measure" and call[1] == "wrong_answer" for call in runner.calls)
    assert runner.closed


def test_run_candidates_preserves_all_failed_result_for_diagnosis():
    runner = _FakeRunner()
    result = run_candidates(
        _kernel_factory,
        [{
            "case": "build_error"
        }, {
            "case": "wrong_answer"
        }],
        runner,
    )

    assert result.best_config is None
    assert len(result.records) == 2
    with pytest.raises(RuntimeError, match="No autotuning candidate succeeded"):
        result.require_best()


def test_auto_tuner_exposes_backend_runner_without_profiler_assumptions():
    runner = _FakeRunner()
    result = AutoTuner.from_kernel(
        _kernel_factory,
        configs=[{
            "case": "slow"
        }, {
            "case": "fast"
        }],
    ).run_with_runner(
        runner, warmup=2, rep=5, timeout=13)

    assert result.best_config == {"case": "fast"}
    assert ("measure", "fast", 2, 5, 13.0) in runner.calls


def test_runner_cleanup_failure_is_preserved_and_rejects_selection():
    result = run_candidates(
        _kernel_factory,
        [{
            "case": "fast"
        }],
        _CloseFailureRunner(),
    )

    assert result.best_config == {"case": "fast"}
    assert result.runner_close_error == "RuntimeError: synthetic cleanup failure"
    with pytest.raises(RuntimeError, match="runner cleanup failed"):
        result.require_best()


@pytest.mark.parametrize(
    "kwargs",
    (
        {
            "warmup": -1
        },
        {
            "rep": 0
        },
        {
            "timeout_s": 0
        },
        {
            "timeout_s": float("nan")
        },
    ),
)
def test_run_candidates_rejects_invalid_measurement_settings(kwargs):
    with pytest.raises(ValueError):
        run_candidates(_kernel_factory, [], _FakeRunner(), **kwargs)


def test_timing_result_rejects_invalid_latencies():
    with pytest.raises(ValueError):
        TimingResult(latency_ms=float("inf"))
    with pytest.raises(ValueError):
        TimingResult(latency_ms=1.0, samples_ms=(1.0, -1.0))
