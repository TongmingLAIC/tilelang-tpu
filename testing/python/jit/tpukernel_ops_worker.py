# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Isolated numerical worker for the TPU-Kernel operation matrix.

The matrix runner imports only the declarative case registry from this module.
Torch, TileLang, and the vendor runtime are deliberately imported only after a
worker has validated the runner-created session.  Consequently the matrix
parent never loads a TPU runtime, and every case gets a fresh process-global
CModel or PCIe runtime.

This file is an executable worker, not a pytest test.  Use
``tpukernel_ops_matrix.py`` to invoke it with timeout and process-group
supervision.
"""

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
import os
import time
import traceback
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import zlib

_CHIPS = ("sg2260e", "bm1690")
_CORE_COUNTS = {"sg2260e": 4, "bm1690": 8}
_RUNTIME_MODES = ("cmodel", "pcie")
_FLOAT_DTYPES = ("float16", "bfloat16", "float32")
_REDUCTION_WIDTHS = (15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65)
_RESULT_PREFIX = "TPUKERNEL_NUMERIC_RESULT="


@dataclass(frozen=True)
class CaseSpec:
    """One independently compilable and runnable numerical contract probe."""

    case_id: str
    operation: str
    dtype: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    supported_chips: Tuple[str, ...] = _CHIPS

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


def build_case_specs() -> Tuple[CaseSpec, ...]:
    """Return the stable, non-FP8 TPU-Kernel numerical matrix.

    The registry intentionally contains only dtypes accepted explicitly by the
    current frontend/codegen contracts.  In particular, absence of FP8 here is
    not evidence about hardware capability; FP8 requires separate CModel
    characterization before it can be added to this conformance matrix.
    """

    cases: List[CaseSpec] = []

    def add(operation: str,
            dtype: str,
            variant: str = "dense",
            *,
            supported_chips: Tuple[str, ...] = _CHIPS,
            **parameters: Any) -> None:
        suffix = f".{variant}" if variant else ""
        case_id = f"{operation}.{dtype}{suffix}"
        cases.append(CaseSpec(case_id, operation, dtype, parameters, supported_chips))

    for dtype in _FLOAT_DTYPES:
        add(
            "copy",
            dtype,
            "local-roundtrip",
            src_dtype=dtype,
            dst_dtype=dtype,
            direct_global=False,
        )
        add(
            "copy",
            dtype,
            "global-to-global",
            src_dtype=dtype,
            dst_dtype=dtype,
            direct_global=True,
        )
    # One exact FP32 probe covers the full rank-3 [N,C,W] DMA descriptor path
    # without multiplying every dtype combination.  Non-unit N/C and a
    # non-EU-aligned W make N-stride and lane-distributed C mistakes visible.
    add(
        "copy",
        "float32",
        "rank3-local-roundtrip",
        src_dtype="float32",
        dst_dtype="float32",
        direct_global=False,
        shape=(2, 3, 17),
    )
    for src_dtype, dst_dtype in (("float16", "float32"), ("bfloat16", "float32"),
                                 ("float32", "float16"), ("float32", "bfloat16")):
        add(
            "copy",
            dst_dtype,
            f"{src_dtype}-to-{dst_dtype}",
            src_dtype=src_dtype,
            dst_dtype=dst_dtype,
            direct_global=False,
        )
    # The generic copy selector maps all six integer storage formats.  Keep
    # these as exact, byte-preserving probes instead of inferring support from
    # the top-k index path, which exercises a different vendor primitive.
    for dtype in ("int8", "uint8", "int16", "uint16", "int32", "uint32"):
        add(
            "copy",
            dtype,
            "local-roundtrip",
            src_dtype=dtype,
            dst_dtype=dtype,
            direct_global=False,
        )
        add(
            "copy",
            dtype,
            "global-to-global",
            src_dtype=dtype,
            dst_dtype=dtype,
            direct_global=True,
        )

    for dtype in _FLOAT_DTYPES:
        add("fill", dtype, value=1.25)

    for dtype in ("float16", "bfloat16"):
        add("gemm", dtype, "overwrite", accumulate=False, transpose_b=False)
        add("gemm", dtype, "accumulate", accumulate=True, transpose_b=False)
        add("gemm", dtype, "transpose-b", accumulate=False, transpose_b=True)
    # BM1690 and SG2260E both expose the native FP32 matrix path.  Keep its
    # storage contract distinct because these operands use local.matrix rather
    # than the lane-distributed local layout used by lower-precision GEMM.
    add("gemm", "float32", "overwrite", accumulate=False, transpose_b=False)
    add("gemm", "float32", "accumulate", accumulate=True, transpose_b=False)

    for operation in ("add", "sub", "mul", "div", "max"):
        for dtype in _FLOAT_DTYPES:
            add(operation, dtype)
        # W-broadcast is a distinct lowering path (zero rhs W-stride).
        add(operation, "float32", "broadcast", broadcast_rhs=True)
    # Exercise the online-softmax identity used by FlashAttention explicitly:
    # max(-inf, x), max(x, -inf), and max(-inf, -inf) must all preserve the
    # exact FP32 operand value/encoding selected by the semantic operation.
    add("max", "float32", "negative-infinity", negative_infinity_sentinel=True)

    for operation, value in (("add-scalar", -0.25), ("mul-scalar", 0.75)):
        for dtype in _FLOAT_DTYPES:
            add(operation, dtype, value=value)

    for dtype in _FLOAT_DTYPES:
        add("exp", dtype)

    for operation in ("reduce-sum", "reduce-max"):
        for dtype in _FLOAT_DTYPES:
            for width in _REDUCTION_WIDTHS:
                add(operation, dtype, f"w{width}", width=width)

    for dtype in _FLOAT_DTYPES:
        add("rsqrt", dtype)

    for dtype in _FLOAT_DTYPES:
        add("gather", dtype)

    # The SG2260E PPL 1.7 CModel terminates inside this HAU primitive.  Keep
    # the probes in the shared registry but schedule them only for BM1690.
    add("topk", "float32", "descending", supported_chips=("bm1690",), descended=True)
    add("topk", "float32", "ascending", supported_chips=("bm1690",), descended=False)
    add("topk", "int32", "descending", supported_chips=("bm1690",), descended=True)
    add("topk", "uint32", "descending", supported_chips=("bm1690",), descended=True)
    add("topk", "int32", "ascending", supported_chips=("bm1690",), descended=False)
    add("topk", "uint32", "ascending", supported_chips=("bm1690",), descended=False)

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise AssertionError("duplicate TPU-Kernel numerical case id")
    return tuple(cases)


_CASE_BY_ID = {case.case_id: case for case in build_case_specs()}


class NumericalMismatch(RuntimeError):
    """A numerical failure carrying JSON-serializable comparison details."""

    def __init__(self, message: str, details: Mapping[str, Any]):
        super().__init__(message)
        self.details = dict(details)


def _json_sample(tensor: Any) -> List[Any]:
    """Return a short JSON-safe sample, spelling non-finite values as text."""

    values = tensor.reshape(-1)[:8].tolist()
    return [
        repr(value) if isinstance(value, float) and not math.isfinite(value) else value
        for value in values
    ]


def _json_max(tensor: Any) -> Optional[float]:
    if tensor.numel() == 0:
        return 0.0
    if not bool(tensor.isfinite().all().item()):
        return None
    return float(tensor.max().item())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", choices=tuple(_CASE_BY_ID))
    parser.add_argument("--chip", choices=_CHIPS)
    parser.add_argument("--runtime-mode", choices=_RUNTIME_MODES)
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args()
    if args.list_cases:
        if args.case_id is not None or args.chip is not None or args.runtime_mode is not None:
            parser.error("--list-cases cannot be combined with worker execution arguments")
    elif args.case_id is None or args.chip is None or args.runtime_mode is None:
        parser.error("--case-id, --chip, and --runtime-mode are required")
    return args


def _validate_worker_session(args: argparse.Namespace) -> None:
    """Reject direct/ambiguous execution before importing any TPU runtime."""

    expected = {
        "TILELANG_TPU_NUMERIC_SESSION": "1",
        "TILELANG_TPU_NUMERIC_CASE": args.case_id,
        "TILELANG_TPU_NUMERIC_CHIP": args.chip,
        "TILELANG_TPU_NUMERIC_RUNTIME_MODE": args.runtime_mode,
    }
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"numerical worker requires runner-owned {name}={value!r}")
    if os.environ.get("TILELANG_TPU_BENCHMARK_RUNS") != "0":
        raise RuntimeError("numerical worker requires exactly one TileLang launch")

    if args.runtime_mode == "pcie":
        if os.environ.get("TILELANG_TPU_NUMERIC_ALLOW_PCIE") != "1":
            raise RuntimeError("PCIe numerical worker is missing its runner safety gate")
        if os.environ.get("TILELANG_TPU_ALLOW_PCIE_LOAD") != "1":
            raise RuntimeError("PCIe numerical worker requires TILELANG_TPU_ALLOW_PCIE_LOAD=1")
        device_id = os.environ.get("TILELANG_TPU_DEVICE_ID", "")
        if not device_id.isdecimal() or int(device_id) > 2**31 - 1:
            raise RuntimeError("PCIe numerical worker requires a valid numeric device id")
        if "TPU_RT_CORE_NUM" in os.environ:
            raise RuntimeError("PCIe numerical worker inherited CModel core topology")
    else:
        for name in ("TILELANG_TPU_NUMERIC_ALLOW_PCIE", "TILELANG_TPU_ALLOW_PCIE_LOAD",
                     "TILELANG_TPU_DEVICE_ID"):
            if name in os.environ:
                raise RuntimeError(f"CModel numerical worker inherited forbidden {name}")
        expected_cores = str(_CORE_COUNTS[args.chip])
        if os.environ.get("TPU_RT_CORE_NUM") != expected_cores:
            raise RuntimeError(f"CModel numerical worker requires TPU_RT_CORE_NUM={expected_cores}")


def _seed_for(case_id: str) -> int:
    return zlib.crc32(case_id.encode("utf-8")) & 0x7FFF_FFFF


def _torch_dtype(torch: Any, dtype: str) -> Any:
    try:
        return getattr(torch, dtype)
    except AttributeError as error:
        raise RuntimeError(f"this PyTorch does not expose dtype {dtype!r}") from error


def _random_float(torch: Any,
                  shape: Tuple[int, ...],
                  dtype: str,
                  generator: Any,
                  *,
                  positive: bool = False) -> Any:
    if positive:
        base = torch.rand(shape, generator=generator, dtype=torch.float32) * 1.5 + 0.5
    else:
        base = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.5
    return base.to(_torch_dtype(torch, dtype)).contiguous()


def _comparison(actual: Any,
                expected: Any,
                *,
                atol: float,
                rtol: float,
                exact: bool = False) -> Dict[str, Any]:
    if tuple(actual.shape) != tuple(expected.shape):
        raise NumericalMismatch(
            f"shape mismatch: actual={tuple(actual.shape)}, expected={tuple(expected.shape)}",
            {
                "actual_shape": list(actual.shape),
                "expected_shape": list(expected.shape)
            },
        )
    if actual.dtype != expected.dtype:
        raise NumericalMismatch(
            f"dtype mismatch: actual={actual.dtype}, expected={expected.dtype}",
            {
                "actual_dtype": str(actual.dtype),
                "expected_dtype": str(expected.dtype)
            },
        )

    if exact:
        equal = actual == expected
        mismatch_count = int((~equal).sum().item())
        metrics: Dict[str, Any] = {
            "exact": True,
            "atol": 0.0,
            "rtol": 0.0,
            "element_count": actual.numel(),
            "mismatch_count": mismatch_count,
        }
        if actual.is_floating_point():
            error = (actual.double() - expected.double()).abs()
            metrics["max_abs_error"] = _json_max(error)
        if mismatch_count:
            metrics["actual_sample"] = _json_sample(actual)
            metrics["expected_sample"] = _json_sample(expected)
            raise NumericalMismatch(
                f"exact comparison failed for {mismatch_count}/{actual.numel()} elements",
                metrics,
            )
        return metrics

    torch = __import__("torch")
    actual_f64 = actual.to(dtype=torch.float64)
    expected_f64 = expected.to(dtype=torch.float64)
    finite = torch.isfinite(actual_f64) & torch.isfinite(expected_f64)
    error = (actual_f64 - expected_f64).abs()
    allowed = atol + rtol * expected_f64.abs()
    close = finite & (error <= allowed)
    mismatch_count = int((~close).sum().item())
    nonfinite_count = int((~finite).sum().item())
    max_abs_error = _json_max(error)
    denominator = torch.clamp(expected_f64.abs(), min=1.0e-12)
    max_rel_error = _json_max(error / denominator)
    metrics = {
        "exact": False,
        "atol": float(atol),
        "rtol": float(rtol),
        "element_count": actual.numel(),
        "mismatch_count": mismatch_count,
        "nonfinite_count": nonfinite_count,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
    }
    if mismatch_count:
        metrics["actual_sample"] = _json_sample(actual)
        metrics["expected_sample"] = _json_sample(expected)
        raise NumericalMismatch(
            f"allclose failed for {mismatch_count}/{actual.numel()} elements; "
            f"max_abs_error={max_abs_error!r}, max_rel_error={max_rel_error!r}",
            metrics,
        )
    return metrics


def _tolerance(dtype: str, operation: str) -> Tuple[float, float]:
    base = {
        "float32": (1.0e-5, 1.0e-5),
        "float16": (5.0e-3, 5.0e-3),
        "bfloat16": (3.0e-2, 3.0e-2),
    }[dtype]
    if operation == "div":
        return {
            "float32": (1.0e-5, 1.0e-5),
            "float16": (1.0e-2, 1.0e-2),
            "bfloat16": (6.0e-2, 6.0e-2),
        }[dtype]
    if operation == "exp":
        return {
            "float32": (1.0e-2, 1.0e-2),
            "float16": (2.0e-2, 2.0e-2),
            "bfloat16": (8.0e-2, 8.0e-2),
        }[dtype]
    if operation == "reduce-sum":
        return {
            "float32": (2.0e-4, 2.0e-4),
            "float16": (3.0e-2, 3.0e-2),
            "bfloat16": (2.0e-1, 5.0e-2),
        }[dtype]
    return base


def _compile_and_launch(tilelang: Any,
                        prim_func: Any,
                        arguments: Tuple[Any, ...],
                        chip: str,
                        runtime_mode: str,
                        out_idx: Union[int, List[int]] = -1) -> Dict[str, float]:
    target = f"tpu -mcpu={chip} -tpu-programming-model=tpukernel"
    compile_begin = time.monotonic()
    kernel = tilelang.compile(
        prim_func,
        out_idx=out_idx,
        target=target,
        runtime_mode=runtime_mode,
    )
    compile_seconds = time.monotonic() - compile_begin
    run_begin = time.monotonic()
    kernel(*arguments)
    run_seconds = time.monotonic() - run_begin
    return {
        "compile_seconds": compile_seconds,
        "run_seconds": run_seconds,
    }


def _run_copy(spec: CaseSpec, chip: str, runtime_mode: str, tilelang: Any, T: Any, torch: Any,
              generator: Any) -> Tuple[Dict[str, Any], Dict[str, float]]:
    shape = tuple(int(extent) for extent in spec.parameters.get("shape", (4, 32)))
    src_dtype = str(spec.parameters["src_dtype"])
    dst_dtype = str(spec.parameters["dst_dtype"])
    direct_global = bool(spec.parameters["direct_global"])

    @T.prim_func
    def kernel(src: T.Tensor(shape, src_dtype), dst: T.Tensor(shape, dst_dtype)):
        with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
            if direct_global:
                T.ppl_copy(src, dst)
            else:
                src_local = T.alloc_shared(shape, src_dtype)
                dst_local = T.alloc_shared(shape, dst_dtype)
                T.ppl_copy(src, src_local)
                T.ppl_copy(src_local, dst_local)
                T.ppl_copy(dst_local, dst)

    if src_dtype.startswith(("int", "uint")):
        info = torch.iinfo(_torch_dtype(torch, src_dtype))
        src = torch.randint(
            info.min,
            info.max + 1,
            shape,
            dtype=_torch_dtype(torch, src_dtype),
            generator=generator,
        )
    else:
        src = _random_float(torch, shape, src_dtype, generator)
    dst = torch.zeros(shape, dtype=_torch_dtype(torch, dst_dtype))
    timing = _compile_and_launch(tilelang, kernel, (src, dst), chip, runtime_mode)
    expected = src.to(_torch_dtype(torch, dst_dtype))
    return _comparison(dst, expected, atol=0.0, rtol=0.0, exact=True), timing


def _run_fill(spec: CaseSpec, chip: str, runtime_mode: str, tilelang: Any, T: Any, torch: Any,
              _generator: Any) -> Tuple[Dict[str, Any], Dict[str, float]]:
    shape = (4, 32)
    dtype = spec.dtype
    value = float(spec.parameters["value"])

    @T.prim_func
    def kernel(dst: T.Tensor(shape, dtype)):
        with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
            local = T.alloc_shared(shape, dtype)
            T.ppl_fill(local, T.float32(value))
            T.ppl_copy(local, dst)

    dst = torch.zeros(shape, dtype=_torch_dtype(torch, dtype))
    timing = _compile_and_launch(tilelang, kernel, (dst,), chip, runtime_mode)
    expected = torch.full(shape, value, dtype=_torch_dtype(torch, dtype))
    return _comparison(dst, expected, atol=0.0, rtol=0.0, exact=True), timing


def _run_elementwise(spec: CaseSpec, chip: str, runtime_mode: str, tilelang: Any, T: Any,
                     torch: Any, generator: Any) -> Tuple[Dict[str, Any], Dict[str, float]]:
    shape = (4, 32)
    rhs_shape = (4, 1) if spec.parameters.get("broadcast_rhs") else shape
    dtype = spec.dtype
    operation = spec.operation

    @T.prim_func
    def kernel(lhs: T.Tensor(shape, dtype), rhs: T.Tensor(rhs_shape, dtype),
               dst: T.Tensor(shape, dtype)):
        with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
            lhs_local = T.alloc_shared(shape, dtype)
            rhs_local = T.alloc_shared(rhs_shape, dtype)
            dst_local = T.alloc_shared(shape, dtype)
            T.ppl_copy(lhs, lhs_local)
            T.ppl_copy(rhs, rhs_local)
            if operation == "add":
                T.ppl_add(dst_local, lhs_local, rhs_local)
            elif operation == "sub":
                T.ppl_subtract(dst_local, lhs_local, rhs_local)
            elif operation == "mul":
                T.ppl_mul(dst_local, lhs_local, rhs_local)
            elif operation == "div":
                T.ppl_div(dst_local, lhs_local, rhs_local)
            elif operation == "max":
                T.ppl_max(dst_local, lhs_local, rhs_local)
            T.ppl_copy(dst_local, dst)

    if operation == "max" and spec.parameters.get("negative_infinity_sentinel"):
        if dtype != "float32" or rhs_shape != shape:
            raise AssertionError("the max negative-infinity sentinel is a dense FP32 probe")
        finite = torch.linspace(
            -8.0, 8.0, steps=shape[0] * shape[1], dtype=torch.float32).reshape(shape)
        lhs = finite.clone()
        rhs = torch.flip(finite, dims=(1,)).clone()
        lhs[:, 0::4] = -float("inf")
        rhs[:, 1::4] = -float("inf")
        lhs[:, 2::4] = -float("inf")
        rhs[:, 2::4] = -float("inf")
    else:
        lhs = _random_float(torch, shape, dtype, generator)
        rhs = _random_float(torch, rhs_shape, dtype, generator, positive=operation == "div")
    dst = torch.zeros(shape, dtype=_torch_dtype(torch, dtype))
    timing = _compile_and_launch(tilelang, kernel, (lhs, rhs, dst), chip, runtime_mode)
    lhs_f32 = lhs.float()
    rhs_f32 = rhs.float()
    if operation == "add":
        expected_f32 = lhs_f32 + rhs_f32
    elif operation == "sub":
        expected_f32 = lhs_f32 - rhs_f32
    elif operation == "mul":
        expected_f32 = lhs_f32 * rhs_f32
    elif operation == "div":
        expected_f32 = lhs_f32 / rhs_f32
    else:
        expected_f32 = torch.maximum(lhs_f32, rhs_f32)
    expected = expected_f32.to(_torch_dtype(torch, dtype))
    if operation == "max":
        return _comparison(dst, expected, atol=0.0, rtol=0.0, exact=True), timing
    atol, rtol = _tolerance(dtype, operation)
    return _comparison(dst, expected, atol=atol, rtol=rtol), timing


def _run_scalar(spec: CaseSpec, chip: str, runtime_mode: str, tilelang: Any, T: Any, torch: Any,
                generator: Any) -> Tuple[Dict[str, Any], Dict[str, float]]:
    shape = (4, 32)
    dtype = spec.dtype
    value = float(spec.parameters["value"])
    operation = spec.operation

    @T.prim_func
    def kernel(src: T.Tensor(shape, dtype), dst: T.Tensor(shape, dtype)):
        with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
            src_local = T.alloc_shared(shape, dtype)
            dst_local = T.alloc_shared(shape, dtype)
            T.ppl_copy(src, src_local)
            if operation == "add-scalar":
                T.ppl_add_C(dst_local, src_local, T.float32(value))
            else:
                T.ppl_mul_C(dst_local, src_local, T.float32(value))
            T.ppl_copy(dst_local, dst)

    src = _random_float(torch, shape, dtype, generator)
    dst = torch.zeros_like(src)
    timing = _compile_and_launch(tilelang, kernel, (src, dst), chip, runtime_mode)
    expected_f32 = src.float() + value if operation == "add-scalar" else src.float() * value
    expected = expected_f32.to(_torch_dtype(torch, dtype))
    atol, rtol = _tolerance(dtype, operation)
    return _comparison(dst, expected, atol=atol, rtol=rtol), timing


def _run_gemm(spec: CaseSpec, chip: str, runtime_mode: str, tilelang: Any, T: Any, torch: Any,
              generator: Any) -> Tuple[Dict[str, Any], Dict[str, float]]:
    m, n, k = 16, 16, 16
    dtype = spec.dtype
    accumulate = bool(spec.parameters["accumulate"])
    transpose_b = bool(spec.parameters["transpose_b"])
    b_shape = (n, k) if transpose_b else (k, n)
    c_dtype = "float32" if accumulate else dtype
    local_scope = "local.matrix" if dtype == "float32" else "shared"
    initial = 0.5

    @T.prim_func
    def kernel(a: T.Tensor((m, k), dtype), b: T.Tensor(b_shape, dtype), c: T.Tensor((m, n),
                                                                                    c_dtype)):
        with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
            a_local = T.alloc_shared((m, k), dtype, scope=local_scope)
            b_local = T.alloc_shared(b_shape, dtype, scope=local_scope)
            c_local = T.alloc_shared((m, n), c_dtype, scope=local_scope)
            T.ppl_copy(a, a_local)
            T.ppl_copy(b, b_local)
            if accumulate:
                # local.matrix is intentionally reserved for FP32 matrix
                # operands, so seed accumulation through its global tensor.
                if dtype == "float32":
                    T.ppl_copy(c, c_local)
                else:
                    T.ppl_fill(c_local, T.float32(initial))
            T.ppl_gemm(
                a_local,
                b_local,
                c_local,
                transpose_B=transpose_b,
                accumulate=accumulate,
            )
            T.ppl_copy(c_local, c)

    a = _random_float(torch, (m, k), dtype, generator)
    b = _random_float(torch, b_shape, dtype, generator)
    c = torch.full((m, n),
                   initial if accumulate and dtype == "float32" else 0.0,
                   dtype=_torch_dtype(torch, c_dtype))
    timing = _compile_and_launch(tilelang, kernel, (a, b, c), chip, runtime_mode)
    b_ref = b.float().transpose(0, 1) if transpose_b else b.float()
    expected_f32 = torch.matmul(a.float(), b_ref)
    if accumulate:
        expected_f32 = expected_f32 + initial
    expected = expected_f32.to(_torch_dtype(torch, c_dtype))
    if dtype == "float32":
        atol, rtol = (2.0e-5, 2.0e-5)
    elif dtype == "bfloat16":
        atol, rtol = (1.5e-1, 3.0e-2)
    else:
        atol, rtol = (3.0e-2, 2.0e-2)
    return _comparison(c, expected, atol=atol, rtol=rtol), timing


def _run_exp(spec: CaseSpec, chip: str, runtime_mode: str, tilelang: Any, T: Any, torch: Any,
             generator: Any) -> Tuple[Dict[str, Any], Dict[str, float]]:
    shape = (4, 32)
    dtype = spec.dtype

    @T.prim_func
    def kernel(src: T.Tensor(shape, dtype), dst: T.Tensor(shape, dtype)):
        with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
            value = T.alloc_shared(shape, dtype)
            work0 = T.alloc_shared(shape, dtype)
            work1 = T.alloc_shared(shape, dtype)
            coeff = T.alloc_shared((64, 32), dtype)
            T.ppl_copy(src, value)
            T.ppl_exp(value, work0, work1, coeff)
            T.ppl_copy(value, dst)

    src = torch.clamp(
        _random_float(torch, shape, dtype, generator).float(),
        -2.0,
        2.0,
    ).to(_torch_dtype(torch, dtype))
    dst = torch.zeros_like(src)
    timing = _compile_and_launch(tilelang, kernel, (src, dst), chip, runtime_mode)
    expected = torch.exp(src.float()).to(_torch_dtype(torch, dtype))
    atol, rtol = _tolerance(dtype, "exp")
    return _comparison(dst, expected, atol=atol, rtol=rtol), timing


def _run_reduction(spec: CaseSpec, chip: str, runtime_mode: str, tilelang: Any, T: Any, torch: Any,
                   generator: Any) -> Tuple[Dict[str, Any], Dict[str, float]]:
    rows = 65
    width = int(spec.parameters["width"])
    dtype = spec.dtype
    operation = spec.operation

    @T.prim_func
    def kernel(src: T.Tensor((rows, width), dtype), dst: T.Tensor((rows, 1), dtype)):
        with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
            src_local = T.alloc_shared((rows, width), dtype)
            dst_local = T.alloc_shared((rows, 1), dtype)
            T.ppl_copy(src, src_local)
            if operation == "reduce-sum":
                T.ppl_reduce_sum(src_local, dst_local, dim=1)
            else:
                T.ppl_reduce_max(src_local, dst_local, dim=1)
            T.ppl_copy(dst_local, dst)

    # A row count above the 64-lane boundary jointly checks the local N-stride
    # while the case matrix sweeps each requested EU-width boundary.
    src = _random_float(torch, (rows, width), dtype, generator)
    dst = torch.zeros((rows, 1), dtype=_torch_dtype(torch, dtype))
    timing = _compile_and_launch(tilelang, kernel, (src, dst), chip, runtime_mode)
    if operation == "reduce-sum":
        expected_f32 = torch.sum(src.float(), dim=1, keepdim=True)
        expected = expected_f32.to(_torch_dtype(torch, dtype))
        atol, rtol = _tolerance(dtype, operation)
        metrics = _comparison(dst, expected, atol=atol, rtol=rtol)
    else:
        expected = torch.max(src, dim=1, keepdim=True).values
        metrics = _comparison(dst, expected, atol=0.0, rtol=0.0, exact=True)
    metrics["reduction_width"] = width
    metrics["reduction_rows"] = rows
    return metrics, timing


def _run_rsqrt(spec: CaseSpec, chip: str, runtime_mode: str, tilelang: Any, T: Any, torch: Any,
               generator: Any) -> Tuple[Dict[str, Any], Dict[str, float]]:
    shape = (4, 32)
    dtype = spec.dtype

    @T.prim_func
    def kernel(src: T.Tensor(shape, dtype), dst: T.Tensor(shape, dtype)):
        with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
            src_local = T.alloc_shared(shape, dtype)
            dst_local = T.alloc_shared(shape, dtype)
            T.ppl_copy(src, src_local)
            T.ppl_rsqrt(dst_local, src_local)
            T.ppl_copy(dst_local, dst)

    src = _random_float(torch, shape, dtype, generator, positive=True)
    dst = torch.zeros_like(src)
    timing = _compile_and_launch(tilelang, kernel, (src, dst), chip, runtime_mode)
    expected = torch.rsqrt(src.float()).to(_torch_dtype(torch, dtype))
    atol, rtol = _tolerance(dtype, "rsqrt")
    return _comparison(dst, expected, atol=atol, rtol=rtol), timing


def _run_gather(spec: CaseSpec, chip: str, runtime_mode: str, tilelang: Any, T: Any, torch: Any,
                generator: Any) -> Tuple[Dict[str, Any], Dict[str, float]]:
    rows, width, count = 17, 32, 7
    dtype = spec.dtype

    @T.prim_func
    def kernel(Param: T.Tensor((rows, width), dtype), Index: T.Tensor((count, 1), "uint32"),
               Output: T.Tensor((count, width), dtype)):
        with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
            T.ppl_gather(Output, Param, Index, rows)

    param = _random_float(torch, (rows, width), dtype, generator)
    index_i32 = torch.tensor([16, 0, 8, 3, 12, 1, 15], dtype=torch.int32).view(count, 1)
    index_u32 = index_i32.view(torch.uint32)
    dst = torch.zeros((count, width), dtype=_torch_dtype(torch, dtype))
    timing = _compile_and_launch(tilelang, kernel, (param, index_u32, dst), chip, runtime_mode)
    expected = param[index_i32.long().reshape(-1)]
    return _comparison(dst, expected, atol=0.0, rtol=0.0, exact=True), timing


def _run_topk(spec: CaseSpec, chip: str, runtime_mode: str, tilelang: Any, T: Any, torch: Any,
              generator: Any) -> Tuple[Dict[str, Any], Dict[str, float]]:
    length, k = 257, 11
    dtype = spec.dtype
    descended = bool(spec.parameters["descended"])

    @T.prim_func
    def kernel(src: T.Tensor((length,), dtype), dst_data: T.Tensor((k,), dtype), dst_idx: T.Tensor(
        (k,), "int32")):
        with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
            T.ppl_topk(dst_data, dst_idx, src, k, descended, length)

    # Deliberately include repeated values.  ``sort_natural_index`` promises a
    # stable ordering, so equal keys must retain their increasing natural
    # source index instead of returning an arbitrary top-k subset.
    del generator
    repeated = torch.arange(length, dtype=torch.int64) % 17
    if dtype == "float32":
        src = (repeated.float() * 0.125 - 1.0).contiguous()
        reference_values = src.double()
    elif dtype == "int32":
        src = (repeated - 8).to(torch.int32).contiguous()
        reference_values = src.to(torch.int64)
    else:
        src = repeated.to(torch.uint32).contiguous()
        reference_values = src.to(torch.int64)
    dst_data = torch.zeros(k, dtype=_torch_dtype(torch, dtype))
    dst_idx = torch.zeros(k, dtype=torch.int32)
    timing = _compile_and_launch(
        tilelang, kernel, (src, dst_data, dst_idx), chip, runtime_mode, out_idx=[1, 2])
    values = reference_values.tolist()
    ordered_indices = sorted(
        range(length),
        key=lambda index: (-values[index] if descended else values[index], index),
    )
    expected_idx64 = torch.tensor(ordered_indices[:k], dtype=torch.int64)
    expected_idx = expected_idx64.to(torch.int32)
    expected_data = src[expected_idx64]
    data_metrics = _comparison(dst_data, expected_data, atol=0.0, rtol=0.0, exact=True)
    index_metrics = _comparison(dst_idx, expected_idx, atol=0.0, rtol=0.0, exact=True)
    return {
        "valid_prefix_length": k,
        "stable_ties_checked": True,
        "data": data_metrics,
        "indices": index_metrics,
    }, timing


def _run_case(spec: CaseSpec, chip: str, runtime_mode: str) -> Dict[str, Any]:
    # Imports are intentionally delayed until all PCIe/session gates pass.
    import torch
    import tilelang
    import tilelang.language as T

    torch.set_num_threads(1)
    seed = _seed_for(spec.case_id)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    dispatch = {
        "copy": _run_copy,
        "fill": _run_fill,
        "gemm": _run_gemm,
        "add": _run_elementwise,
        "sub": _run_elementwise,
        "mul": _run_elementwise,
        "div": _run_elementwise,
        "max": _run_elementwise,
        "add-scalar": _run_scalar,
        "mul-scalar": _run_scalar,
        "exp": _run_exp,
        "reduce-sum": _run_reduction,
        "reduce-max": _run_reduction,
        "rsqrt": _run_rsqrt,
        "gather": _run_gather,
        "topk": _run_topk,
    }
    runner = dispatch[spec.operation]
    metrics, timing = runner(spec, chip, runtime_mode, tilelang, T, torch, generator)
    return {
        "status": "passed",
        "case": spec.to_json(),
        "chip": chip,
        "programming_model": "tpukernel",
        "runtime_mode": runtime_mode,
        "seed": seed,
        "timing": timing,
        "metrics": metrics,
        "torch_version": str(torch.__version__),
    }


def _emit_result(payload: Mapping[str, Any]) -> None:
    print(
        _RESULT_PREFIX +
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
        flush=True,
    )


def main() -> int:
    args = _parse_args()
    if args.list_cases:
        print(
            json.dumps(
                [case.to_json() for case in build_case_specs()],
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ))
        return 0

    assert args.case_id is not None
    assert args.chip is not None
    assert args.runtime_mode is not None
    started = time.monotonic()
    try:
        _validate_worker_session(args)
        payload = _run_case(_CASE_BY_ID[args.case_id], args.chip, args.runtime_mode)
        payload["total_seconds"] = time.monotonic() - started
        _emit_result(payload)
        return 0
    except BaseException as error:  # Preserve a machine-readable first failure.
        payload: Dict[str, Any] = {
            "status": "failed",
            "case": _CASE_BY_ID[args.case_id].to_json(),
            "chip": args.chip,
            "programming_model": "tpukernel",
            "runtime_mode": args.runtime_mode,
            "error_type": type(error).__name__,
            "error": str(error),
            "total_seconds": time.monotonic() - started,
        }
        if isinstance(error, NumericalMismatch):
            payload["metrics"] = error.details
        _emit_result(payload)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
