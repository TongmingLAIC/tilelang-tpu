# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Shared contracts for import-safe TPU demos.

The helpers in this module deliberately keep compile-time target selection
separate from host runtime selection.  PCIe is opt-in and requires an explicit
device id; automated board runs add an external process-tree watchdog as well.
"""

from __future__ import annotations

import os
import math
import time
from typing import Any, Mapping, Optional, Sequence

import torch

from tpu_demo.cases import (CHIP_CORE_COUNTS, CHIPS, PROGRAMMING_MODELS, RUNTIME_MODES,
                            kernel_variant)


class DemoNumericalMismatch(AssertionError):
    """Numerical contract failure with JSON-serializable diagnostics."""

    def __init__(self, message: str, metrics: Mapping[str, Any]):
        super().__init__(message)
        self.metrics = dict(metrics)


def torch_dtype(dtype: str) -> torch.dtype:
    try:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "e4m3_float8": torch.float8_e4m3fn,
            "e5m2_float8": torch.float8_e5m2,
        }[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported demo dtype: {dtype!r}") from error


def validate_dimensions(operation: str, **dimensions: int) -> None:
    """Reject dynamic, boolean, zero, and negative demo dimensions early."""
    for name, value in dimensions.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{operation} requires positive integer {name}, got {value!r}")


def validate_exact_tiling(operation: str, *axes: tuple[str, int, int]) -> None:
    """Require full tiles because these demos intentionally have no tail path."""
    for name, extent, tile in axes:
        if extent % tile:
            raise ValueError(f"{operation} requires {name}={extent} to be divisible by tile {tile}")


def validate_positive_scalar(operation: str, name: str, value: float) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value <= 0):
        raise ValueError(f"{operation} requires finite positive {name}, got {value!r}")


def validate_selection(
    *,
    chip: str,
    programming_model: str,
    runtime_mode: str,
    supports_rv: bool,
    allow_pcie: bool,
    device_id: Optional[int],
) -> None:
    if chip not in CHIPS:
        raise ValueError(f"unsupported TPU chip: {chip!r}")
    if programming_model not in PROGRAMMING_MODELS:
        raise ValueError(f"unsupported programming model: {programming_model!r}")
    if runtime_mode not in RUNTIME_MODES:
        raise ValueError(f"unsupported runtime mode: {runtime_mode!r}")
    if chip == "bm1690" and programming_model == "rv":
        raise ValueError("BM1690 does not implement the RV Tensor programming model")
    if programming_model == "rv" and not supports_rv:
        raise ValueError("this demo requires TPU-Kernel semantic operations that RV Tensor "
                         "does not yet expose")
    if runtime_mode == "pcie":
        expected_profile = {
            "TILELANG_TPU_PROFILE_SESSION": "1",
            "TILELANG_TPU_PROFILE_RUNTIME_MODE": "pcie",
            "TILELANG_TPU_PROFILE_CHIP": chip,
            "TILELANG_TPU_PROFILE_PROGRAMMING_MODEL": programming_model,
        }
        for name, expected in expected_profile.items():
            if os.environ.get(name) != expected:
                raise ValueError("PCIe demos may run only in the supervised demo matrix; "
                                 f"missing runner-owned {name}={expected!r}")
    configure_runtime(runtime_mode, allow_pcie, device_id, chip=chip)


def configure_runtime(runtime_mode: str, allow_pcie: bool, device_id: Optional[int], *,
                      chip: str) -> None:
    if runtime_mode == "cmodel":
        if allow_pcie or device_id is not None:
            raise ValueError("PCIe acknowledgement/device id are invalid in CModel mode")
        os.environ["TPU_RT_CORE_NUM"] = str(CHIP_CORE_COUNTS[chip])
        for name in ("TILELANG_TPU_ALLOW_PCIE_LOAD", "TILELANG_TPU_ALLOW_PCIE_PROFILE",
                     "TILELANG_TPU_DEVICE_ID", "BMLIB_ENABLE_ALL_PROFILE"):
            os.environ.pop(name, None)
        return
    if not allow_pcie:
        raise ValueError("PCIe execution is disabled by default; use the supervised demo matrix")
    if (device_id is None or isinstance(device_id, bool) or not isinstance(device_id, int) or
            not 0 <= device_id <= 2**31 - 1):
        raise ValueError("PCIe execution requires an explicit non-negative 32-bit device id")
    requested_id = str(device_id)
    existing_gate = os.environ.get("TILELANG_TPU_ALLOW_PCIE_LOAD")
    existing_id = os.environ.get("TILELANG_TPU_DEVICE_ID")
    if existing_gate not in (None, "1"):
        raise ValueError("TILELANG_TPU_ALLOW_PCIE_LOAD must be unset or equal to 1")
    if existing_id is not None and existing_id != requested_id:
        raise ValueError("device id conflicts with TILELANG_TPU_DEVICE_ID")
    if os.environ.get("TILELANG_TPU_ALLOW_PCIE_PROFILE") != "1":
        raise ValueError("supervised PCIe demos require TILELANG_TPU_ALLOW_PCIE_PROFILE=1")
    if os.environ.get("BMLIB_ENABLE_ALL_PROFILE") != "1":
        raise ValueError("supervised PCIe demos require BMLIB_ENABLE_ALL_PROFILE=1")
    if "TPU_RT_CORE_NUM" in os.environ:
        raise ValueError("PCIe demos must not inherit the CModel TPU_RT_CORE_NUM setting")
    os.environ["TILELANG_TPU_ALLOW_PCIE_LOAD"] = "1"
    os.environ["TILELANG_TPU_DEVICE_ID"] = requested_id


def target_string(chip: str, programming_model: str) -> str:
    return f"tpu -mcpu={chip} -tpu-programming-model={programming_model}"


def compile_and_launch(
    program: Any,
    arguments: Sequence[torch.Tensor],
    *,
    chip: str,
    programming_model: str,
    runtime_mode: str,
) -> Mapping[str, float]:
    # Import after target/runtime validation so an invalid request cannot load
    # the vendor runtime as a side effect.
    import tilelang

    begin = time.monotonic()
    kernel = tilelang.compile(
        program,
        out_idx=-1,
        target=target_string(chip, programming_model),
        runtime_mode=runtime_mode,
    )
    compiled = time.monotonic()
    kernel(*arguments)
    finished = time.monotonic()
    return {
        "compile_seconds": compiled - begin,
        "launch_seconds": finished - compiled,
    }


def comparison(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if tuple(actual.shape) != tuple(expected.shape):
        metrics = {
            "passed": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
        }
        raise DemoNumericalMismatch(
            f"shape mismatch: actual={tuple(actual.shape)}, expected={tuple(expected.shape)}",
            metrics,
        )
    if actual.dtype != expected.dtype:
        metrics = {
            "passed": False,
            "actual_dtype": str(actual.dtype),
            "expected_dtype": str(expected.dtype),
        }
        raise DemoNumericalMismatch(
            f"dtype mismatch: actual={actual.dtype}, expected={expected.dtype}", metrics)
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    finite = bool(torch.isfinite(actual_f32).all() and torch.isfinite(expected_f32).all())
    difference = torch.abs(actual_f32 - expected_f32)
    close = torch.isclose(actual_f32, expected_f32, atol=atol, rtol=rtol)
    passed = finite and bool(torch.all(close))
    max_error = float(torch.max(difference))
    mean_error = float(torch.mean(difference))
    max_error_json = max_error if math.isfinite(max_error) else None
    mean_error_json = mean_error if math.isfinite(mean_error) else None
    metrics = {
        "passed": passed,
        "finite": finite,
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": max_error_json,
        "mean_abs_error": mean_error_json,
        "mismatched_elements": int(torch.count_nonzero(~close)),
        "element_count": actual.numel(),
    }
    if not passed:

        def json_sample(tensor: torch.Tensor) -> list[Any]:
            return [
                repr(value) if isinstance(value, float) and not math.isfinite(value) else value
                for value in tensor.reshape(-1)[:8].tolist()
            ]

        metrics["actual_sample"] = json_sample(actual)
        metrics["expected_sample"] = json_sample(expected)
        raise DemoNumericalMismatch(
            "numerical mismatch: "
            f"max_abs_error={metrics['max_abs_error']!r}, "
            f"mismatched={metrics['mismatched_elements']}/{metrics['element_count']}",
            metrics,
        )
    return metrics


def tolerance(dtype: str, family: str) -> tuple[float, float]:
    if family == "elementwise":
        return {
            "float16": (5e-3, 5e-3),
            "bfloat16": (2e-2, 2e-2),
            "float32": (1e-5, 1e-5),
            "e4m3_float8": (2.5e-1, 1e-1),
            "e5m2_float8": (5e-1, 2e-1),
        }[dtype]
    if family == "elementwise-div":
        return {
            "float16": (1e-2, 1e-2),
            "bfloat16": (3e-2, 3e-2),
            "float32": (1e-5, 1e-5),
        }[dtype]
    if family == "matmul":
        return {
            "float16": (1e-2, 1e-2),
            "bfloat16": (2e-2, 2e-2),
            "float32": (1e-2, 1e-2),
            "e4m3_float8": (2.5e-1, 1.25e-1),
            "e5m2_float8": (7.5e-1, 2.5e-1),
        }[dtype]
    if family == "matmul-native-fp32":
        if dtype != "float32":
            raise ValueError("native FP32 matmul tolerance requires float32")
        return 5e-5, 5e-5
    if family in ("rmsnorm", "swiglu"):
        return {
            "float16": (1e-2, 1e-2),
            "bfloat16": (3e-2, 3e-2),
            "float32": (1e-2, 1e-2),
            "e4m3_float8": (2.5e-1, 1.25e-1),
            "e5m2_float8": (7.5e-1, 2.5e-1),
        }[dtype]
    if family == "rope":
        return {
            "float16": (5e-3, 5e-3),
            "bfloat16": (2e-2, 2e-2),
            "float32": (1e-5, 1e-5),
            "e4m3_float8": (2.5e-1, 1.25e-1),
            "e5m2_float8": (5e-1, 2.5e-1),
        }[dtype]
    if family == "flashattn":
        # In addition to reduced-precision GEMMs, the vendor polynomial exp
        # primitive has a characterized slope error on weighted logits.  The
        # weighted-keys case keeps this bound tied to an ideal softmax oracle.
        return {
            "float16": (2e-2, 2e-2),
            "bfloat16": (2e-2, 2e-2),
            "float32": (2e-2, 2e-2),
            "e4m3_float8": (2.5e-1, 1.5e-1),
            "e5m2_float8": (7.5e-1, 3e-1),
        }[dtype]
    raise ValueError(f"unknown tolerance family: {family!r}")


def result_payload(
    *,
    operation: str,
    dtype: str,
    chip: str,
    programming_model: str,
    runtime_mode: str,
    metrics: Mapping[str, Any],
    timing: Mapping[str, float],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    result_parameters = dict(parameters)
    result_parameters["kernel_variant"] = kernel_variant(operation, dtype, programming_model)
    return {
        "status": "passed",
        "operation": operation,
        "dtype": dtype,
        "chip": chip,
        "programming_model": programming_model,
        "runtime_mode": runtime_mode,
        "parameters": result_parameters,
        "metrics": dict(metrics),
        "timing": dict(timing),
    }
