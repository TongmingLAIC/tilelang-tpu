# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Isolated numerical worker for the TPU-Kernel/RV FP8 capability contract."""

import argparse
import json
import os
from typing import Tuple

import torch

import tilelang
import tilelang.language as T

_DTYPES = {
    "e4m3": ("e4m3_float8", torch.float8_e4m3fn),
    "e5m2": ("e5m2_float8", torch.float8_e5m2),
}
_RESULT_PREFIX = "TPU_FP8_NUMERIC_RESULT="
_RESULT_SCHEMA_VERSION = 1


def _emit_result(payload) -> None:
    print(
        _RESULT_PREFIX +
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
        flush=True,
    )


_PCIE_GATE_VARIABLES = (
    "TILELANG_TPU_ALLOW_PCIE_LOAD",
    "TILELANG_TPU_ALLOW_PCIE_PROFILE",
    "TILELANG_TPU_DEVICE_ID",
    "BMLIB_ENABLE_ALL_PROFILE",
    "PROFILE_RECORD_SIZE",
    "PROFILE_BOOK_KEEPING",
)


def _profile_selection() -> Tuple[str, str, str]:
    if os.environ.get("TILELANG_TPU_PROFILE_SESSION") != "1":
        raise RuntimeError("FP8 worker must run through TPUInstructionProfiler")
    chip = os.environ.get("TILELANG_TPU_PROFILE_CHIP", "")
    programming_model = os.environ.get("TILELANG_TPU_PROFILE_PROGRAMMING_MODEL", "")
    runtime_mode = os.environ.get("TILELANG_TPU_PROFILE_RUNTIME_MODE", "")
    if programming_model not in ("tpukernel", "rv"):
        raise RuntimeError("FP8 capability worker requires programming_model=tpukernel or rv")
    if chip not in ("bm1690", "sg2260e"):
        raise RuntimeError("FP8 capability worker requires a supported TPU chip")
    if runtime_mode not in ("cmodel", "pcie"):
        raise RuntimeError("FP8 capability worker requires runtime_mode=cmodel or pcie")
    if programming_model == "rv" and chip != "sg2260e":
        raise RuntimeError("RV Tensor FP8 validation requires chip=sg2260e")
    if os.environ.get("TILELANG_TPU_BENCHMARK_RUNS") != "0":
        raise RuntimeError("FP8 profiling requires exactly one kernel launch")
    if runtime_mode == "cmodel":
        inherited_gates = tuple(name for name in _PCIE_GATE_VARIABLES if name in os.environ)
        if inherited_gates:
            raise RuntimeError("CModel FP8 worker refuses PCIe recorder/device state: " +
                               ", ".join(inherited_gates))
    else:
        if chip != "sg2260e":
            raise RuntimeError("PCIe FP8 worker accepts only chip=sg2260e")
        required = {
            "TILELANG_TPU_ALLOW_PCIE_LOAD": "1",
            "TILELANG_TPU_ALLOW_PCIE_PROFILE": "1",
            "BMLIB_ENABLE_ALL_PROFILE": "1",
        }
        invalid = tuple(
            name for name, expected in required.items() if os.environ.get(name) != expected)
        if invalid:
            raise RuntimeError("PCIe FP8 worker requires explicit load/profile/recorder gates: " +
                               ", ".join(invalid))
        device_id = os.environ.get("TILELANG_TPU_DEVICE_ID", "")
        if device_id != "0":
            raise RuntimeError("PCIe FP8 worker accepts only numeric device id 0")
    return chip, programming_model, runtime_mode


def _target(chip: str) -> str:
    programming_model = os.environ.get("TILELANG_TPU_PROFILE_PROGRAMMING_MODEL", "")
    return f"tpu -mcpu={chip} -tpu-programming-model={programming_model}"


def _elementwise_operation(case: str) -> str:
    suffix = "-broadcast"
    if case.endswith(suffix):
        return case[:-len(suffix)]
    return case


def _compile(kernel, chip: str, runtime_mode: str):
    return tilelang.compile(
        kernel,
        out_idx=-1,
        target=_target(chip),
        runtime_mode=runtime_mode,
    )


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, *, atol=0.0, rtol=0.0) -> None:
    if not torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol):
        max_abs = float(torch.max(torch.abs(actual.float() - expected.float())))
        raise RuntimeError(f"FP8 numerical mismatch; max_abs={max_abs}")


def _run_copy(dtype: str, torch_dtype: torch.dtype, chip: str, runtime_mode: str, *,
              direct_global: bool) -> None:

    @T.prim_func
    def copy_kernel(A: T.Tensor((1, 64), dtype), B: T.Tensor((1, 64), dtype)):
        with T.Kernel(1, is_cpu=True) as _:
            if direct_global:
                T.ppl_copy(A, B)
            else:
                local = T.alloc_shared((1, 64), dtype)
                T.ppl_copy(A, local)
                T.ppl_copy(local, B)

    values = torch.linspace(-4.0, 4.0, 64, dtype=torch.float32).to(torch_dtype)
    source = values.reshape(1, 64)
    output = torch.zeros_like(source)
    _compile(copy_kernel, chip, runtime_mode)(source, output)
    if not torch.equal(source.view(torch.uint8), output.view(torch.uint8)):
        raise RuntimeError("same-format FP8 copy changed the encoded bytes")


def _run_fill(dtype: str, torch_dtype: torch.dtype, chip: str, runtime_mode: str,
              value: float) -> None:

    @T.prim_func
    def fill_kernel(B: T.Tensor((1, 64), dtype)):
        with T.Kernel(1, is_cpu=True) as _:
            local = T.alloc_shared((1, 64), dtype)
            T.ppl_fill(local, T.float32(value))
            T.ppl_copy(local, B)

    output = torch.full((1, 64), -2.0, dtype=torch.float32).to(torch_dtype)
    _compile(fill_kernel, chip, runtime_mode)(output)
    expected = torch.full((1, 64), value, dtype=torch.float32).to(torch_dtype)
    if not torch.equal(output.view(torch.uint8), expected.view(torch.uint8)):
        raise RuntimeError(f"FP8 fill({value}) disagrees with the framework encoding")


def _run_cast(case: str, dtype: str, torch_dtype: torch.dtype, chip: str,
              runtime_mode: str) -> None:
    if case == "cast-to-fp8":
        peer_dtype, peer_torch_dtype, to_fp8 = "float32", torch.float32, True
    elif case == "cast-from-fp8":
        peer_dtype, peer_torch_dtype, to_fp8 = "float32", torch.float32, False
    else:
        peer_name = "bfloat16" if "bf16" in case else "float16"
        peer_dtype = peer_name
        peer_torch_dtype = torch.bfloat16 if peer_name == "bfloat16" else torch.float16
        to_fp8 = case.endswith("to-fp8")

    if to_fp8:

        @T.prim_func
        def cast_kernel(A: T.Tensor((1, 64), peer_dtype), B: T.Tensor((1, 64), dtype)):
            with T.Kernel(1, is_cpu=True) as _:
                src = T.alloc_shared((1, 64), peer_dtype)
                dst = T.alloc_shared((1, 64), dtype)
                T.ppl_copy(A, src)
                T.ppl_copy(src, dst)
                T.ppl_copy(dst, B)

        source = torch.linspace(
            -3.0, 3.0, 64, dtype=torch.float32).to(peer_torch_dtype).reshape(1, 64)
        output = torch.zeros((1, 64), dtype=torch.float32).to(torch_dtype)
        expected = source.to(torch_dtype)
    else:

        @T.prim_func
        def cast_kernel(A: T.Tensor((1, 64), dtype), B: T.Tensor((1, 64), peer_dtype)):
            with T.Kernel(1, is_cpu=True) as _:
                src = T.alloc_shared((1, 64), dtype)
                dst = T.alloc_shared((1, 64), peer_dtype)
                T.ppl_copy(A, src)
                T.ppl_copy(src, dst)
                T.ppl_copy(dst, B)

        source = torch.linspace(-3.0, 3.0, 64, dtype=torch.float32).to(torch_dtype).reshape(1, 64)
        output = torch.zeros((1, 64), dtype=peer_torch_dtype)
        expected = source.to(peer_torch_dtype)

    _compile(cast_kernel, chip, runtime_mode)(source, output)
    if to_fp8:
        if not torch.equal(output.view(torch.uint8), expected.view(torch.uint8)):
            raise RuntimeError("FP32 to FP8 cast disagrees with the framework encoding")
    else:
        _assert_close(output, expected)


def _run_elementwise(operation: str, dtype: str, torch_dtype: torch.dtype, chip: str,
                     runtime_mode: str, *, broadcast_rhs: bool) -> None:
    if operation not in ("add", "sub", "mul", "max"):
        raise ValueError(f"unsupported FP8 elementwise operation: {operation}")
    rhs_shape = (1, 1) if broadcast_rhs else (1, 64)

    @T.prim_func
    def elementwise(A: T.Tensor((1, 64), dtype), B: T.Tensor(rhs_shape, dtype), C: T.Tensor((1, 64),
                                                                                            dtype)):
        with T.Kernel(1, is_cpu=True) as _:
            lhs = T.alloc_shared((1, 64), dtype)
            rhs = T.alloc_shared(rhs_shape, dtype)
            out = T.alloc_shared((1, 64), dtype)
            T.ppl_copy(A, lhs)
            T.ppl_copy(B, rhs)
            if operation == "add":
                T.ppl_add(out, lhs, rhs)
            elif operation == "sub":
                T.ppl_subtract(out, lhs, rhs)
            elif operation == "mul":
                T.ppl_mul(out, lhs, rhs)
            else:
                T.ppl_max(out, lhs, rhs)
            T.ppl_copy(out, C)

    lhs_pattern = torch.tensor(
        [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0],
        dtype=torch.float32,
    ).repeat(8)
    if broadcast_rhs:
        rhs_pattern = torch.tensor([[0.5]], dtype=torch.float32)
    else:
        rhs_pattern = torch.tensor(
            [0.5, 1.0, 2.0, -1.0, -0.5, 2.0, -1.0, 0.5],
            dtype=torch.float32,
        ).repeat(8)
    lhs = lhs_pattern.to(torch_dtype).reshape(1, 64)
    rhs = rhs_pattern.to(torch_dtype).reshape(rhs_shape)
    output = torch.zeros_like(lhs)
    _compile(elementwise, chip, runtime_mode)(lhs, rhs, output)
    expected_fp32 = {
        "add": torch.add,
        "sub": torch.sub,
        "mul": torch.mul,
        "max": torch.maximum,
    }[operation](lhs.float(), rhs.float())
    expected = expected_fp32.to(torch_dtype)
    if operation == "max":
        if not torch.equal(output.view(torch.uint8), expected.view(torch.uint8)):
            raise RuntimeError("FP8 max changed the selected operand encoding")
        return
    if not torch.equal(output.view(torch.uint8), expected.view(torch.uint8)):
        _assert_close(output, expected, atol=0.25, rtol=0.0)


def _run_scalar(operation: str, dtype: str, torch_dtype: torch.dtype, chip: str,
                runtime_mode: str) -> None:
    value = -0.25 if operation == "add" else 0.75

    @T.prim_func
    def scalar_kernel(A: T.Tensor((1, 64), dtype), C: T.Tensor((1, 64), dtype)):
        with T.Kernel(1, is_cpu=True) as _:
            src = T.alloc_shared((1, 64), dtype)
            out = T.alloc_shared((1, 64), dtype)
            T.ppl_copy(A, src)
            if operation == "add":
                T.ppl_add_C(out, src, T.float32(value))
            else:
                T.ppl_mul_C(out, src, T.float32(value))
            T.ppl_copy(out, C)

    source = torch.tensor(
        [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        dtype=torch.float32,
    ).repeat(8).to(torch_dtype).reshape(1, 64)
    output = torch.zeros_like(source)
    _compile(scalar_kernel, chip, runtime_mode)(source, output)
    expected_fp32 = source.float() + value if operation == "add" else source.float() * value
    expected = expected_fp32.to(torch_dtype)
    if not torch.equal(output.view(torch.uint8), expected.view(torch.uint8)):
        _assert_close(output, expected, atol=0.25, rtol=0.0)


def _run_embedding(dtype: str, torch_dtype: torch.dtype, chip: str, runtime_mode: str) -> None:
    rows, width, count = 17, 32, 7

    @T.prim_func
    def embedding_kernel(param_buffer: T.Tensor((rows, width), dtype), index_buffer: T.Tensor(
        (count, 1), "uint32"), output_buffer: T.Tensor((count, width), dtype)):
        with T.Kernel(1, is_cpu=True) as _:
            T.ppl_embedding(output_buffer, param_buffer, index_buffer)

    param = (torch.arange(rows * width, dtype=torch.float32).reshape(rows, width) % 19 -
             9.0).to(torch_dtype)
    index_i32 = torch.tensor([16, 0, 8, 3, 12, 1, 15], dtype=torch.int32).reshape(count, 1)
    index_u32 = index_i32.view(torch.uint32)
    output = torch.zeros((count, width), dtype=torch_dtype)
    _compile(embedding_kernel, chip, runtime_mode)(param, index_u32, output)
    expected = param[index_i32.long().reshape(-1)]
    if not torch.equal(output.view(torch.uint8), expected.view(torch.uint8)):
        raise RuntimeError("FP8 embedding changed the selected encoded bytes")


def _run_reduction(operation: str, dtype: str, torch_dtype: torch.dtype, chip: str,
                   runtime_mode: str) -> None:
    rows, width = 4, 65

    @T.prim_func
    def reduction_kernel(A: T.Tensor((rows, width), dtype), C: T.Tensor((rows, 1), dtype)):
        with T.Kernel(1, is_cpu=True) as _:
            src = T.alloc_shared((rows, width), dtype)
            out = T.alloc_shared((rows, 1), dtype)
            T.ppl_copy(A, src)
            if operation == "sum":
                T.ppl_reduce_sum(src, out, dim=1)
            else:
                T.ppl_reduce_max(src, out, dim=1)
            T.ppl_copy(out, C)

    pattern = torch.linspace(-0.25, 0.25, rows * width, dtype=torch.float32)
    source = pattern.to(torch_dtype).reshape(rows, width)
    output = torch.zeros((rows, 1), dtype=torch_dtype)
    _compile(reduction_kernel, chip, runtime_mode)(source, output)
    if operation == "sum":
        # RV maps the reduction to a deterministic chain of same-dtype adds;
        # model the required FP8 rounding after each instruction.
        expected = torch.zeros((rows, 1), dtype=torch_dtype)
        for column in range(width):
            expected = (expected.float() + source[:, column:column + 1].float()).to(torch_dtype)
    else:
        expected = torch.max(source.float(), dim=1, keepdim=True).values.to(torch_dtype)
    if operation == "max":
        if not torch.equal(output.view(torch.uint8), expected.view(torch.uint8)):
            raise RuntimeError("FP8 reduce-max changed the selected operand encoding: "
                               f"actual={output.float().reshape(-1).tolist()}, "
                               f"expected={expected.float().reshape(-1).tolist()}")
    else:
        _assert_close(output, expected, atol=0.25, rtol=0.125)


def _run_gemm(case: str, dtype: str, torch_dtype: torch.dtype, chip: str,
              runtime_mode: str) -> None:
    transpose_b = case in ("gemm-nt-overwrite", "gemm-nt-accumulate")
    accumulate = case in ("gemm-nn-accumulate", "gemm-nt-accumulate")
    b_shape = (16, 64) if transpose_b else (64, 16)

    @T.prim_func
    def gemm(A: T.Tensor((16, 64), dtype), B: T.Tensor(b_shape, dtype), C: T.Tensor((16, 16),
                                                                                    "float32")):
        with T.Kernel(1, is_cpu=True) as _:
            lhs = T.alloc_shared((16, 64), dtype)
            rhs = T.alloc_shared(b_shape, dtype)
            out = T.alloc_shared((16, 16), "float32")
            T.ppl_copy(A, lhs)
            T.ppl_copy(B, rhs)
            if accumulate:
                T.ppl_fill(out, T.float32(1))
            T.ppl_gemm(
                lhs,
                rhs,
                out,
                transpose_B=transpose_b,
                accumulate=accumulate,
            )
            T.ppl_copy(out, C)

    torch.manual_seed(7)
    lhs = (torch.randn(16, 64, dtype=torch.float32) * 0.5).to(torch_dtype)
    rhs = (torch.randn(*b_shape, dtype=torch.float32) * 0.5).to(torch_dtype)
    output = torch.zeros((16, 16), dtype=torch.float32)
    _compile(gemm, chip, runtime_mode)(lhs, rhs, output)
    expected = lhs.float() @ (rhs.float().T if transpose_b else rhs.float())
    if accumulate:
        expected += 1.0
    _assert_close(output, expected, atol=0.12, rtol=0.02)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=tuple(_DTYPES), required=True)
    parser.add_argument(
        "--case",
        choices=(
            "copy",
            "copy-global-to-global",
            "fill-zero",
            "fill-nonzero",
            "cast-to-fp8",
            "cast-from-fp8",
            "cast-fp16-to-fp8",
            "cast-fp8-to-fp16",
            "cast-bf16-to-fp8",
            "cast-fp8-to-bf16",
            "add",
            "sub",
            "mul",
            "max",
            "add-broadcast",
            "sub-broadcast",
            "mul-broadcast",
            "max-broadcast",
            "add-scalar",
            "mul-scalar",
            "embedding",
            "reduce-sum",
            "reduce-max",
            "gemm-nn-overwrite",
            "gemm-nn-accumulate",
            "gemm-nt-overwrite",
            "gemm-nt-accumulate",
        ),
        required=True,
    )
    args = parser.parse_args()
    chip = os.environ.get("TILELANG_TPU_PROFILE_CHIP", "")
    programming_model = os.environ.get("TILELANG_TPU_PROFILE_PROGRAMMING_MODEL", "")
    runtime_mode = os.environ.get("TILELANG_TPU_PROFILE_RUNTIME_MODE", "")
    try:
        chip, programming_model, runtime_mode = _profile_selection()
        dtype, torch_dtype = _DTYPES[args.dtype]
        if args.case in ("copy", "copy-global-to-global"):
            _run_copy(
                dtype,
                torch_dtype,
                chip,
                runtime_mode,
                direct_global=args.case == "copy-global-to-global",
            )
        elif args.case in ("fill-zero", "fill-nonzero"):
            _run_fill(
                dtype,
                torch_dtype,
                chip,
                runtime_mode,
                0.0 if args.case == "fill-zero" else 1.25,
            )
        elif args.case.startswith("cast-"):
            _run_cast(args.case, dtype, torch_dtype, chip, runtime_mode)
        elif args.case in ("add", "sub", "mul", "max", "add-broadcast", "sub-broadcast",
                           "mul-broadcast", "max-broadcast"):
            operation = _elementwise_operation(args.case)
            _run_elementwise(
                operation,
                dtype,
                torch_dtype,
                chip,
                runtime_mode,
                broadcast_rhs=args.case.endswith("-broadcast"),
            )
        elif args.case in ("add-scalar", "mul-scalar"):
            _run_scalar(
                args.case[:-len("-scalar")],
                dtype,
                torch_dtype,
                chip,
                runtime_mode,
            )
        elif args.case == "embedding":
            _run_embedding(dtype, torch_dtype, chip, runtime_mode)
        elif args.case in ("reduce-sum", "reduce-max"):
            _run_reduction(
                args.case.removeprefix("reduce-"), dtype, torch_dtype, chip, runtime_mode)
        else:
            _run_gemm(args.case, dtype, torch_dtype, chip, runtime_mode)
    except BaseException as error:
        _emit_result({
            "schema_version": _RESULT_SCHEMA_VERSION,
            "status": "failed",
            "chip": chip,
            "programming_model": programming_model,
            "runtime_mode": runtime_mode,
            "dtype": args.dtype,
            "case": args.case,
            "error_type": type(error).__name__,
            "error": str(error),
        })
        raise
    _emit_result({
        "schema_version": _RESULT_SCHEMA_VERSION,
        "status": "passed",
        "chip": chip,
        "programming_model": programming_model,
        "runtime_mode": runtime_mode,
        "dtype": args.dtype,
        "case": args.case,
        "metrics": {
            "passed": True
        },
    })


if __name__ == "__main__":
    main()
