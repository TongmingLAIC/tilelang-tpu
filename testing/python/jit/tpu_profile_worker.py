# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Fresh-compile workers used by the TPU instruction-profile tests.

This file intentionally is not named ``test_*.py``: it is invoked by
``TPUInstructionProfiler`` in a new process and owns the complete JIT
compile/load/dispatch lifecycle.  It never loads a prebuilt TileLang TPU shared
object; PCIe is accepted only when the profiler supplies every safety gate.
"""

import argparse
import json
import os

import torch

import tilelang
import tilelang.language as T

_RESULT_PREFIX = "TPU_CORE_NUMERIC_RESULT="
_RESULT_SCHEMA_VERSION = 1

_COPY_CASES = {
    "copy-fp32-local-roundtrip": ("float32", "local"),
    "copy-fp32-global-to-global": ("float32", "global"),
    "copy-fp16-local-roundtrip": ("float16", "local"),
    "copy-fp16-global-to-global": ("float16", "global"),
}
_CONVERSION_CASES = {
    "copy-fp16-to-bf16": ("float16", "bfloat16"),
    "copy-bf16-to-fp16": ("bfloat16", "float16"),
}

_MAX_CASES = {
    "elementwise-max-fp16-dense": ("float16", "dense"),
    "elementwise-max-bf16-dense": ("bfloat16", "dense"),
    "elementwise-max-fp32-dense": ("float32", "dense"),
    "elementwise-max-fp16-broadcast": ("float16", "broadcast"),
    "elementwise-max-bf16-broadcast": ("bfloat16", "broadcast"),
    "elementwise-max-fp32-broadcast": ("float32", "broadcast"),
    "elementwise-max-fp32-negative-infinity": ("float32", "negative-infinity"),
}

_BROADCAST_CASES = {
    f"elementwise-{operation}-{dtype}-broadcast": (operation, dtype)
    for operation in ("add", "sub", "mul", "div") for dtype in ("fp16", "bf16", "fp32")
}
_FP32_TRANSPOSE_A_CASES = {
    "matmul-fp32-transpose-a-overwrite": False,
    "matmul-fp32-transpose-a-accumulate": True,
}


def _profile_selection():
    """Read the profiler's explicit chip/programming-model/runtime contract."""

    if os.environ.get("TILELANG_TPU_PROFILE_SESSION") != "1":
        raise RuntimeError("tpu_profile_worker must run through TPUInstructionProfiler.")
    chip = os.environ.get("TILELANG_TPU_PROFILE_CHIP")
    programming_model = os.environ.get("TILELANG_TPU_PROFILE_PROGRAMMING_MODEL")
    runtime_mode = os.environ.get("TILELANG_TPU_PROFILE_RUNTIME_MODE")
    if chip not in ("bm1690", "sg2260e"):
        raise RuntimeError("missing or invalid TileLang TPU chip selection.")
    if programming_model not in ("tpukernel", "rv"):
        raise RuntimeError("missing or invalid TileLang TPU programming-model selection.")
    if chip == "bm1690" and programming_model == "rv":
        raise RuntimeError("BM1690 does not implement the RV Tensor programming model.")
    if runtime_mode not in ("cmodel", "pcie"):
        raise RuntimeError("missing or invalid TileLang TPU profile selection.")
    if os.environ.get("TILELANG_TPU_BENCHMARK_RUNS") != "0":
        raise RuntimeError("instruction profiling requires exactly one TileLang launch.")
    if runtime_mode == "cmodel":
        for name in ("TILELANG_TPU_ALLOW_PCIE_LOAD", "TILELANG_TPU_ALLOW_PCIE_PROFILE",
                     "TILELANG_TPU_DEVICE_ID"):
            if name in os.environ:
                raise RuntimeError(
                    f"CModel profile worker inherited forbidden PCIe setting {name}.")
    else:
        if chip != "sg2260e":
            raise RuntimeError("PCIe core-op worker accepts only chip=sg2260e.")
        for name in ("TILELANG_TPU_ALLOW_PCIE_LOAD", "TILELANG_TPU_ALLOW_PCIE_PROFILE",
                     "BMLIB_ENABLE_ALL_PROFILE"):
            if os.environ.get(name) != "1":
                raise RuntimeError(f"PCIe profile worker is missing safety gate {name}=1.")
        device_id = os.environ.get("TILELANG_TPU_DEVICE_ID", "")
        if device_id != "0":
            raise RuntimeError("PCIe core-op worker accepts only numeric device id 0.")
    return chip, programming_model, runtime_mode


def _emit_result(payload) -> None:
    print(
        _RESULT_PREFIX +
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
        flush=True,
    )


@T.prim_func
def _copy_fp32_global_program(
        source: T.Tensor((4, 32), "float32"),
        destination: T.Tensor((4, 32), "float32"),
):
    with T.Kernel(1, is_cpu=True):
        T.ppl_copy(source, destination)


@T.prim_func
def _copy_fp32_local_program(
        source: T.Tensor((4, 32), "float32"),
        destination: T.Tensor((4, 32), "float32"),
):
    with T.Kernel(1, is_cpu=True):
        source_local = T.alloc_shared((4, 32), "float32")
        destination_local = T.alloc_shared((4, 32), "float32")
        T.ppl_copy(source, source_local)
        T.ppl_copy(source_local, destination_local)
        T.ppl_copy(destination_local, destination)


@T.prim_func
def _copy_fp16_global_program(
        source: T.Tensor((4, 32), "float16"),
        destination: T.Tensor((4, 32), "float16"),
):
    with T.Kernel(1, is_cpu=True):
        T.ppl_copy(source, destination)


@T.prim_func
def _copy_fp16_local_program(
        source: T.Tensor((4, 32), "float16"),
        destination: T.Tensor((4, 32), "float16"),
):
    with T.Kernel(1, is_cpu=True):
        source_local = T.alloc_shared((4, 32), "float16")
        destination_local = T.alloc_shared((4, 32), "float16")
        T.ppl_copy(source, source_local)
        T.ppl_copy(source_local, destination_local)
        T.ppl_copy(destination_local, destination)


_COPY_PROGRAMS = {
    ("float32", "global"): _copy_fp32_global_program,
    ("float32", "local"): _copy_fp32_local_program,
    ("float16", "global"): _copy_fp16_global_program,
    ("float16", "local"): _copy_fp16_local_program,
}


@T.prim_func
def _copy_fp16_to_bf16_program(
        source: T.Tensor((4, 32), "float16"),
        destination: T.Tensor((4, 32), "bfloat16"),
):
    with T.Kernel(1, is_cpu=True):
        source_local = T.alloc_shared((4, 32), "float16")
        destination_local = T.alloc_shared((4, 32), "bfloat16")
        T.ppl_copy(source, source_local)
        T.ppl_copy(source_local, destination_local)
        T.ppl_copy(destination_local, destination)


@T.prim_func
def _copy_bf16_to_fp16_program(
        source: T.Tensor((4, 32), "bfloat16"),
        destination: T.Tensor((4, 32), "float16"),
):
    with T.Kernel(1, is_cpu=True):
        source_local = T.alloc_shared((4, 32), "bfloat16")
        destination_local = T.alloc_shared((4, 32), "float16")
        T.ppl_copy(source, source_local)
        T.ppl_copy(source_local, destination_local)
        T.ppl_copy(destination_local, destination)


_CONVERSION_PROGRAMS = {
    ("float16", "bfloat16"): _copy_fp16_to_bf16_program,
    ("bfloat16", "float16"): _copy_bf16_to_fp16_program,
}


def _matmul(chip: str, programming_model: str, runtime_mode: str) -> None:

    @T.prim_func
    def matmul(
            A: T.Tensor((64, 64), "float16"),
            B: T.Tensor((64, 64), "float16"),
            C: T.Tensor((64, 64), "float16"),
    ):
        with T.Kernel(2, 2, is_cpu=True) as (bx, by):
            A_shared = T.alloc_shared((32, 32), "float16")
            B_shared = T.alloc_shared((32, 32), "float16")
            C_acc = T.alloc_shared((32, 32), "float32")
            C_out = T.alloc_shared((32, 32), "float16")

            T.ppl_fill(C_acc, T.float32(0))
            # Keep the core-op correctness matrix dependency-ordered.  TPU
            # software-pipeline overlap needs a separate hazard-aware contract;
            # num_stages=1 must not make a load race its immediate GEMM user.
            for k in T.serial(2):
                T.ppl_copy(A[by * 32, k * 32], A_shared)
                T.ppl_copy(B[k * 32, bx * 32], B_shared)
                T.ppl_gemm(A_shared, B_shared, C_acc, accumulate=True)
            T.ppl_copy(C_acc, C_out)
            T.ppl_copy(C_out, C[by * 32, bx * 32])

    torch.manual_seed(0)
    kernel = tilelang.compile(
        matmul,
        out_idx=-1,
        target=(f"tpu -mcpu={chip} "
                f"-tpu-programming-model={programming_model}"),
        runtime_mode=runtime_mode,
    )
    a = torch.randn(64, 64).half()
    b = torch.randn(64, 64).half()
    c = torch.zeros(64, 64).half()
    kernel(a, b, c)
    reference = torch.matmul(a, b).half()
    if not torch.allclose(c, reference, atol=1e-2, rtol=1e-2):
        difference = float(torch.max(torch.abs(reference - c)))
        raise RuntimeError(f"{programming_model} matmul mismatch; max abs difference={difference}")


def _matmul_fp32_transpose_a(accumulate: bool, chip: str, programming_model: str,
                             runtime_mode: str) -> None:
    m, n, k = 16, 16, 32

    @T.prim_func
    def matmul(
            A: T.Tensor((k, m), "float32"),
            B: T.Tensor((k, n), "float32"),
            C: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(1, is_cpu=True):
            A_local = T.alloc_shared((k, m), "float32", scope="local.matrix")
            B_local = T.alloc_shared((k, n), "float32", scope="local.matrix")
            C_local = T.alloc_shared((m, n), "float32", scope="local.matrix")
            T.ppl_copy(A, A_local)
            T.ppl_copy(B, B_local)
            if accumulate:
                T.ppl_copy(C, C_local)
            T.ppl_gemm(A_local, B_local, C_local, transpose_A=True, accumulate=accumulate)
            T.ppl_copy(C_local, C)

    generator = torch.Generator().manual_seed(17 if accumulate else 13)
    a = torch.randn((k, m), generator=generator, dtype=torch.float32) * 0.25
    b = torch.randn((k, n), generator=generator, dtype=torch.float32) * 0.25
    initial = 0.5 if accumulate else 0.0
    c = torch.full((m, n), initial, dtype=torch.float32)
    kernel = tilelang.compile(
        matmul,
        out_idx=-1,
        target=(f"tpu -mcpu={chip} "
                f"-tpu-programming-model={programming_model}"),
        runtime_mode=runtime_mode,
    )
    kernel(a, b, c)
    reference = torch.matmul(a.transpose(0, 1), b) + initial
    if not torch.allclose(c, reference, atol=5e-5, rtol=5e-5):
        difference = float(torch.max(torch.abs(reference - c)))
        raise RuntimeError(f"{programming_model} FP32 transpose-A matmul mismatch; "
                           f"accumulate={accumulate}; max abs difference={difference}")


def _elementwise(operation: str,
                 chip: str,
                 programming_model: str,
                 runtime_mode: str,
                 *,
                 dtype: str = "float32",
                 variant: str = "dense") -> None:
    if dtype not in ("float16", "bfloat16", "float32"):
        raise ValueError(f"unsupported elementwise dtype {dtype!r}")
    if variant not in ("dense", "broadcast", "negative-infinity"):
        raise ValueError(f"unsupported elementwise variant {variant!r}")
    if variant == "negative-infinity" and (operation != "max" or dtype != "float32"):
        raise ValueError("the negative-infinity sentinel is an FP32 max probe")

    shape = (64, 64)
    rhs_shape = (64, 1) if variant == "broadcast" else shape
    tile_shape = (32, 32)
    rhs_tile_shape = (32, 1) if variant == "broadcast" else tile_shape

    @T.prim_func
    def elementwise(
            A: T.Tensor(shape, dtype),
            B: T.Tensor(rhs_shape, dtype),
            C: T.Tensor(shape, dtype),
    ):
        with T.Kernel(2, 2, is_cpu=True) as (bx, by):
            A_shared = T.alloc_shared(tile_shape, dtype)
            B_shared = T.alloc_shared(rhs_tile_shape, dtype)
            C_shared = T.alloc_shared(tile_shape, dtype)
            T.ppl_copy(A[by * 32, bx * 32], A_shared)
            if variant == "broadcast":
                T.ppl_copy(B[by * 32, 0], B_shared)
            else:
                T.ppl_copy(B[by * 32, bx * 32], B_shared)
            if operation == "add":
                T.ppl_add(C_shared, A_shared, B_shared)
            elif operation == "sub":
                T.ppl_subtract(C_shared, A_shared, B_shared)
            elif operation == "mul":
                T.ppl_mul(C_shared, A_shared, B_shared)
            elif operation == "div":
                T.ppl_div(C_shared, A_shared, B_shared)
            elif operation == "max":
                T.ppl_max(C_shared, A_shared, B_shared)
            T.ppl_copy(C_shared, C[by * 32, bx * 32])

    torch.manual_seed(0)
    kernel = tilelang.compile(
        elementwise,
        out_idx=-1,
        target=(f"tpu -mcpu={chip} "
                f"-tpu-programming-model={programming_model}"),
        runtime_mode=runtime_mode,
    )
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    if variant == "negative-infinity":
        finite = torch.linspace(
            -8.0, 8.0, steps=shape[0] * shape[1], dtype=torch.float32).reshape(shape)
        a = finite.clone()
        b = torch.flip(finite, dims=(1,)).clone()
        a[:, 0::4] = -float("inf")
        b[:, 1::4] = -float("inf")
        a[:, 2::4] = -float("inf")
        b[:, 2::4] = -float("inf")
    else:
        a = torch.randn(shape, dtype=torch.float32).to(torch_dtype)
    if operation == "div":
        # Keep the divisor bounded away from zero; rvt_fdiv is a documented
        # reciprocal-sqrt approximation rather than strict IEEE division.
        b = (torch.rand(rhs_shape, dtype=torch.float32) * 1.5 + 0.5).to(torch_dtype)
    elif variant != "negative-infinity":
        b = torch.randn(rhs_shape, dtype=torch.float32).to(torch_dtype)
    c = torch.zeros(shape, dtype=torch_dtype)
    kernel(a, b, c)
    reference = {
        "add": torch.add,
        "sub": torch.sub,
        "mul": torch.mul,
        "div": torch.div,
        "max": torch.maximum,
    }[operation](a, b)
    if operation == "max":
        if not torch.equal(c, reference):
            mismatches = int(torch.count_nonzero(c != reference))
            raise RuntimeError(
                f"{programming_model} elementwise-max {dtype} {variant} mismatch; "
                f"unequal elements={mismatches}; actual={c.reshape(-1)[:8].tolist()}; "
                f"expected={reference.reshape(-1)[:8].tolist()}")
        return
    if operation == "div":
        atol, rtol = {
            "float16": (1e-2, 1e-2),
            "bfloat16": (3e-2, 3e-2),
            "float32": (1e-5, 1e-5),
        }[dtype]
    else:
        atol, rtol = {
            "float16": (5e-3, 5e-3),
            "bfloat16": (2e-2, 2e-2),
            "float32": (1e-5, 1e-5),
        }[dtype]
    if not torch.allclose(c, reference, atol=atol, rtol=rtol):
        difference = float(torch.max(torch.abs(reference - c)))
        raise RuntimeError(f"{programming_model} elementwise-{operation} mismatch; "
                           f"max abs difference={difference}")


def _copy(dtype: str, transfer: str, chip: str, programming_model: str, runtime_mode: str) -> None:
    shape = (4, 32)
    try:
        copy_kernel = _COPY_PROGRAMS[(dtype, transfer)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported copy combination dtype={dtype!r}, transfer={transfer!r}") from exc

    torch_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    # Quarter-integers in this range have exact FP16 and FP32 encodings.  An
    # exact comparison therefore checks copy semantics without introducing a
    # numerical tolerance that could conceal a missing or partial transfer.
    source = (torch.arange(128, dtype=torch.float32).reshape(shape) * 0.25 - 16.0).to(torch_dtype)
    destination = torch.full(shape, 113.0, dtype=torch_dtype)
    kernel = tilelang.compile(
        copy_kernel,
        out_idx=-1,
        target=(f"tpu -mcpu={chip} "
                f"-tpu-programming-model={programming_model}"),
        runtime_mode=runtime_mode,
    )
    kernel(source, destination)
    if not torch.equal(destination, source):
        mismatches = int(torch.count_nonzero(destination != source))
        raise RuntimeError(f"{programming_model} {transfer}-to-{transfer} {dtype} copy "
                           f"mismatch; unequal elements={mismatches}")


def _copy_conversion(source_dtype: str, destination_dtype: str, chip: str, programming_model: str,
                     runtime_mode: str) -> None:
    shape = (4, 32)
    torch_dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    source = torch.linspace(
        -3.0, 3.0, shape[0] * shape[1],
        dtype=torch.float32).reshape(shape).to(torch_dtypes[source_dtype])
    destination = torch.zeros(shape, dtype=torch_dtypes[destination_dtype])
    kernel = tilelang.compile(
        _CONVERSION_PROGRAMS[(source_dtype, destination_dtype)],
        out_idx=-1,
        target=(f"tpu -mcpu={chip} "
                f"-tpu-programming-model={programming_model}"),
        runtime_mode=runtime_mode,
    )
    kernel(source, destination)
    expected = source.to(torch_dtypes[destination_dtype])
    if not torch.equal(destination.view(torch.int16), expected.view(torch.int16)):
        difference = float(torch.max(torch.abs(destination.float() - expected.float())))
        raise RuntimeError(f"{programming_model} {source_dtype}-to-{destination_dtype} conversion "
                           f"mismatch; max abs difference={difference}")


def _rv_control(chip: str, programming_model: str, runtime_mode: str) -> None:
    if programming_model != "rv":
        raise RuntimeError("rv-control requires programming_model='rv'.")

    @T.prim_func
    def control(A: T.Tensor((1,), "float32")):
        T.func_attr({"global_symbol": "profile_rv_control", "tir.noalias": T.bool(True)})
        T.rvt_kernel_start()
        T.rvt_sync_all()

    kernel = tilelang.compile(
        control,
        out_idx=[],
        target=(f"tpu -mcpu={chip} "
                f"-tpu-programming-model={programming_model}"),
        runtime_mode=runtime_mode,
    )
    kernel(torch.zeros(1, dtype=torch.float32))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=(
            "matmul",
            "tpukernel-matmul",
            *_FP32_TRANSPOSE_A_CASES,
            "elementwise-add",
            "elementwise-sub",
            "elementwise-mul",
            "elementwise-div",
            *_BROADCAST_CASES,
            *_MAX_CASES,
            *_COPY_CASES,
            *_CONVERSION_CASES,
            "rv-control",
        ),
        required=True,
    )
    args = parser.parse_args()
    chip = os.environ.get("TILELANG_TPU_PROFILE_CHIP", "")
    programming_model = os.environ.get("TILELANG_TPU_PROFILE_PROGRAMMING_MODEL", "")
    runtime_mode = os.environ.get("TILELANG_TPU_PROFILE_RUNTIME_MODE", "")
    try:
        chip, programming_model, runtime_mode = _profile_selection()
        if args.case in ("matmul", "tpukernel-matmul"):
            _matmul(chip, programming_model, runtime_mode)
        elif args.case in _FP32_TRANSPOSE_A_CASES:
            _matmul_fp32_transpose_a(_FP32_TRANSPOSE_A_CASES[args.case], chip, programming_model,
                                     runtime_mode)
        elif args.case in _MAX_CASES:
            dtype, variant = _MAX_CASES[args.case]
            _elementwise("max", chip, programming_model, runtime_mode, dtype=dtype, variant=variant)
        elif args.case in _BROADCAST_CASES:
            operation, dtype_token = _BROADCAST_CASES[args.case]
            dtype = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}[dtype_token]
            _elementwise(
                operation, chip, programming_model, runtime_mode, dtype=dtype, variant="broadcast")
        elif args.case.startswith("elementwise-"):
            _elementwise(args.case[len("elementwise-"):], chip, programming_model, runtime_mode)
        elif args.case in _COPY_CASES:
            dtype, transfer = _COPY_CASES[args.case]
            _copy(dtype, transfer, chip, programming_model, runtime_mode)
        elif args.case in _CONVERSION_CASES:
            source_dtype, destination_dtype = _CONVERSION_CASES[args.case]
            _copy_conversion(source_dtype, destination_dtype, chip, programming_model, runtime_mode)
        else:
            _rv_control(chip, programming_model, runtime_mode)
    except BaseException as error:
        _emit_result({
            "schema_version": _RESULT_SCHEMA_VERSION,
            "status": "failed",
            "case": args.case,
            "chip": chip,
            "programming_model": programming_model,
            "runtime_mode": runtime_mode,
            "error_type": type(error).__name__,
            "error": str(error),
        })
        raise
    _emit_result({
        "schema_version": _RESULT_SCHEMA_VERSION,
        "status": "passed",
        "case": args.case,
        "chip": chip,
        "programming_model": programming_model,
        "runtime_mode": runtime_mode,
        "metrics": {
            "passed": True
        },
    })


if __name__ == "__main__":
    main()
