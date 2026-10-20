# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""W8A16: dequantize an fp8 block-scaled weight, then multiply.

This is the weight path behind torch-tpu's `mm_w8a16_dq_forward` -- the linear
layer DeepSeek-V3 uses for `q_a_proj`, `kv_a_proj_with_mqa` and `o_proj`.  The
reference is the vendor's own `MmW8A16Dq_ppl.pl`, whose three steps are

    w_deq[n, k] = bf16(w_fp8[n, k]) * scale[n // 128, k // 128]

    cast   E4M3 -> compute dtype          T.ppl_copy
    scale  per 128x128 block              T.ppl_mul
    gemm   compute dtype x compute dtype  T.ppl_gemm

Note that the fp8 never reaches the matrix unit.  The vendor casts and scales
on the vector unit first and only a 16-bit operand enters the MMA, which is
what makes this operator portable where the packed-integer GPTQ path is not:
it needs no dequantize instruction, only a cast, an elementwise multiply and a
GEMM, all three of which the RV contract already exposes.

Layouts, taken from the reference:

    left    (M, K)       FP16/BF16 activations
    right   (N, K)       E4M3; the `nn.Linear` native layout, not transposed
    scale   (N, K//128)  FP16/BF16
    result  (M, N)       FP16/BF16

Why the scale arrives pre-expanded on N
    The block scale is indexed `scale[n // 128, k // 128]`, but `T.ppl_mul`
    accepts only an equal shape or the rank-2 `(M, 1)` W-broadcast; there is no
    way to express the vendor's 4-D stride broadcast.  Tiling K by 128 makes
    the k block constant inside a tile, so all that remains is one scale per
    output row -- exactly the `(block_n, 1)` column the contract allows.
    Expanding only the N axis turns a `(N//128, K//128)` table into an
    `(N, K//128)` one that a plain `ppl_copy` reads as that column.  It costs
    128x on one axis instead of 128*128x on both.

    Verified on SG2260E: this form is bit-identical to a baseline that expands
    the scale fully to `(N, K)` and does a plain elementwise multiply.

Verdicts
    `passed`      compiled, ran, and matched the host oracle
    `unsupported` the target's toolchain refused to lower the fp8 chain;
                  `reason` and `evidence` carry the toolchain diagnostic
    (raises)      compiled and ran but disagreed with the oracle

    Unlike GPTQ this operator has no known missing primitive on any target, so
    the `unsupported` branch is defensive: it only triggers when the toolchain
    diagnostic actually names fp8, and any other failure is re-raised rather
    than being misreported as absence of support.
"""

import time
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tpu_demo.common import (comparison, result_payload, target_string, tolerance, torch_dtype,
                             unsupported_payload, validate_dimensions, validate_exact_tiling,
                             validate_selection)

OPERATION = "w8a16-dq"

# 128 is the DeepSeek checkpoint's `weight_block_size`.  N spans two blocks so
# the demo exercises the block boundary rather than a single constant scale.
DEFAULT_M = 32
DEFAULT_N = 256
DEFAULT_K = 256
DEFAULT_GROUP_SIZE = 128
DEFAULT_BLOCK_M = 32
DEFAULT_BLOCK_N = 64

# Substrings that identify "this target has no lowering for the fp8 chain".
# Deliberately narrow: the codegen diagnostic names the format, not the op.
_MISSING_LOWERING_MARKERS = ("e4m3", "e5m2", "fp8")


def _reports_missing_lowering(error: BaseException) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in _MISSING_LOWERING_MARKERS)


def _summarize_toolchain_error(message: str) -> str:
    """Keep the diagnostic line and drop the TVM backtrace around it.

    The evidence field is meant to be read by whoever decides whether the
    target gained the lowering, so the failure reason matters and the frames
    do not.
    """
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    for line in reversed(lines):
        if "Check failed" in line or "Error:" in line:
            return line
    return lines[-1] if lines else message


def _chunk_count(k: int, group_size: int) -> int:
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if k % group_size:
        raise ValueError(f"group_size must divide K: group_size={group_size}, K={k}")
    return k // group_size


def build_w8a16_dq(*,
                   m: int = DEFAULT_M,
                   n: int = DEFAULT_N,
                   k: int = DEFAULT_K,
                   group_size: int = DEFAULT_GROUP_SIZE,
                   block_m: int = DEFAULT_BLOCK_M,
                   block_n: int = DEFAULT_BLOCK_N,
                   dtype: str = "float16",
                   programming_model: str = "rv"):
    """Build the dequantize-then-GEMM program.

    Each block owns `(block_m, block_n)` of the result and walks K in
    128-wide chunks: cast that chunk of the weight to the compute dtype, scale
    it by the matching `(block_n, 1)` column, and accumulate it into the
    result.
    """
    if programming_model not in ("tpukernel", "rv"):
        raise ValueError(f"unsupported TPU programming model: {programming_model!r}")
    torch_dtype(dtype)
    validate_dimensions(
        OPERATION, m=m, n=n, k=k, block_m=block_m, block_n=block_n, group_size=group_size)
    validate_exact_tiling(OPERATION, ("m", m, block_m), ("n", n, block_n))
    chunks = _chunk_count(k, group_size)
    if n % group_size:
        raise ValueError(f"group_size must divide N so the block table is exact: "
                         f"group_size={group_size}, N={n}")

    @T.prim_func
    def w8a16_dq(left: T.Tensor((m, k), dtype), right: T.Tensor((n, k), "e4m3_float8"),
                 scale: T.Tensor((n, chunks), dtype), result: T.Tensor((m, n), dtype)):
        with T.Kernel(T.ceildiv(m, block_m), T.ceildiv(n, block_n), is_cpu=True) as (bx, by):
            left_compute = T.alloc_shared((block_m, group_size), dtype)
            right_fp8 = T.alloc_shared((block_n, group_size), "e4m3_float8")
            weight_compute = T.alloc_shared((block_n, group_size), dtype)
            scale_column = T.alloc_shared((block_n, 1), dtype)
            accumulate = T.alloc_shared((block_m, block_n), "float32")
            output_compute = T.alloc_shared((block_m, block_n), dtype)

            T.ppl_fill(accumulate, T.float32(0.0))
            for chunk in range(chunks):
                # cast: fp8 -> compute dtype
                T.ppl_copy(right[by * block_n, chunk * group_size], right_fp8)
                T.ppl_copy(right_fp8, weight_compute)
                # scale: one value per output row, broadcast across the chunk
                T.ppl_copy(scale[by * block_n, chunk], scale_column)
                T.ppl_mul(weight_compute, weight_compute, scale_column)
                # gemm: accumulate this k chunk onto the zeroed accumulator
                T.ppl_copy(left[bx * block_m, chunk * group_size], left_compute)
                T.ppl_gemm(left_compute, weight_compute, accumulate,
                           transpose_B=True, accumulate=True)

            T.ppl_copy(accumulate, output_compute)
            T.ppl_copy(output_compute, result[bx * block_m, by * block_n])

    return w8a16_dq


def host_inputs(*, m: int, n: int, k: int, group_size: int, dtype: str,
                seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build W8A16-shaped host tensors: an fp8 weight and its block-scale table.

    The scale is drawn independently of the weight so the oracle exercises the
    dequantize arithmetic rather than any particular quantizer, and it is
    expanded along N here -- see the module docstring for why the kernel wants
    that layout.
    """
    chunks = _chunk_count(k, group_size)
    compute = torch_dtype(dtype)
    generator = torch.Generator().manual_seed(seed)

    left = (torch.randn((m, k), generator=generator) * 0.5).to(compute)
    # Keep the weights inside e4m3's range so the cast, not saturation, is what
    # the comparison measures.
    right = (torch.randn((n, k), generator=generator) * 0.25).to(torch.float8_e4m3fn)

    blocks = torch.rand((n // group_size, chunks), generator=generator) * 0.5 + 0.75
    scale = blocks.repeat_interleave(group_size, dim=0).to(compute)

    return left, right, scale


def expand_scale(scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """The kernel's `(N, K//group)` table -> the full `(N, K)` table."""
    return scale.to(torch.float32).repeat_interleave(group_size, dim=1)


def torch_reference(left: torch.Tensor, right: torch.Tensor, scale: torch.Tensor, *,
                    group_size: int, dtype: str) -> torch.Tensor:
    """Host oracle: dequantize the fp8 weight, then multiply.

    The dequantized weight lives in a compute-dtype tile on the device, so the
    product is rounded there before the GEMM runs.  An oracle that skips that
    rounding measures ideal arithmetic rather than the operator and disagrees
    with a perfectly correct kernel by ~2e-2 on this data, so it is mirrored
    here.
    """
    compute = torch_dtype(dtype)
    dequant = right.to(torch.float32) * expand_scale(scale, group_size)
    dequant = dequant.to(compute).to(torch.float32)
    reference = left.to(torch.float32) @ dequant.T
    return reference.to(compute)


def run(*,
        dtype: str,
        chip: str,
        programming_model: str,
        runtime_mode: str,
        allow_pcie: bool = False,
        device_id: Optional[int] = None,
        seed: int = 0) -> dict:
    torch_dtype(dtype)
    validate_selection(
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode,
        supports_rv=True,
        allow_pcie=allow_pcie,
        device_id=device_id,
    )

    m, n, k, group_size = DEFAULT_M, DEFAULT_N, DEFAULT_K, DEFAULT_GROUP_SIZE
    parameters = {
        "m": m,
        "n": n,
        "k": k,
        "group_size": group_size,
        "block_m": DEFAULT_BLOCK_M,
        "block_n": DEFAULT_BLOCK_N,
        "seed": seed,
    }

    left, right, scale = host_inputs(
        m=m, n=n, k=k, group_size=group_size, dtype=dtype, seed=seed)
    result = torch.zeros((m, n), dtype=torch_dtype(dtype))
    program = build_w8a16_dq(
        m=m,
        n=n,
        k=k,
        group_size=group_size,
        block_m=DEFAULT_BLOCK_M,
        block_n=DEFAULT_BLOCK_N,
        dtype=dtype,
        programming_model=programming_model)

    begin = time.monotonic()
    try:
        kernel = tilelang.compile(
            program,
            out_idx=-1,
            target=target_string(chip, programming_model),
            runtime_mode=runtime_mode,
        )
    except Exception as error:  # noqa: BLE001 - classified immediately below
        if _reports_missing_lowering(error):
            return unsupported_payload(
                operation=OPERATION,
                dtype=dtype,
                chip=chip,
                programming_model=programming_model,
                runtime_mode=runtime_mode,
                reason=("this target has no lowering for the fp8 cast/GEMM chain "
                        "that W8A16 weights require"),
                evidence={
                    "exception": type(error).__name__,
                    "message": _summarize_toolchain_error(str(error)),
                },
                parameters=parameters)
        raise
    compiled = time.monotonic()

    kernel(left, right, scale, result)
    finished = time.monotonic()

    expected = torch_reference(
        left, right, scale, group_size=group_size, dtype=dtype).to(result.dtype)
    atol, rtol = tolerance(dtype, "matmul")
    metrics = comparison(result, expected, atol=atol, rtol=rtol)
    return result_payload(
        operation=OPERATION,
        dtype=dtype,
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode,
        metrics=metrics,
        timing={
            "compile_seconds": compiled - begin,
            "launch_seconds": finished - compiled,
        },
        parameters=parameters)
