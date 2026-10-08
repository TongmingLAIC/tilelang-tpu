# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""GPTQ / W4A16: dequantize packed 4-bit weights, then multiply.

This is the weight path behind torch-tpu's `matmul_gptq_forward` and
`llama_mlp_gptq_forward`.  The reference is the vendor's own
`examples/cxx/matmul/w4a16_matmul_dq2.pl`, which expresses the same two steps:
dequantize the packed weight, then GEMM.

Composition (no fused primitive, per the OP_MAPPING policy):

    weight  = dq2(right_packed, offset_scale, group_size)   # T.ppl_dq2
    result  = left @ weight^T                               # T.ppl_gemm

Layouts, taken from the reference:

    left      (M, K)      FP16/BF16 activations
    right     (N, K/2)    UINT8; two 4-bit weights per byte, low nibble first
    offset_scale (N, K/G) UINT32; low 16 bits = offset, high 16 bits = scale,
                          both FP16 ("compact layout" in the PPL manual)
    result    (M, N)      FP16/BF16

Why this demo exists
    The group-wise dequantize is the one primitive in the Llama/DeepSeek set
    that SG2260E does not expose a working lowering for.  Rather than record
    that as prose, this demo *measures* it: it builds the operator, compiles it
    for the requested target, and reports one of three verdicts.

        passed       compiled, ran, and matched the host oracle
        unsupported  the target's toolchain refused to lower the dequantize;
                     `reason` and `evidence` carry the toolchain diagnostic
        (raises)     compiled and ran but disagreed with the oracle

    So a target that gains the missing lowering starts reporting `passed` with
    no change to this file, and a target that does not keeps a reproducible,
    machine-readable record of why.

Verdict detection
    `unsupported` is decided by whether the toolchain error names the missing
    dequantize (`dq2`).  Any other compilation failure is re-raised rather than
    being misreported as absence of support.

Safety
    On current SG2260E firmware the TPU-Kernel lowering emits
    `tpu_bdc_f16_group_dequant`, which the vendor ships as a weak stub; the RV
    lowering is refused at compile time.  A board run that reaches the launch
    step can therefore wedge the device, which needs a driver reload to
    recover.  Automated board runs should use the supervised matrix runner and
    its process-tree watchdog; `runtime_mode="cmodel"` avoids the device
    entirely.
"""

import time
from typing import Optional

import numpy as np
import tilelang
import tilelang.language as T
import torch

from tpu_demo.common import (comparison, result_payload, target_string, tolerance, torch_dtype,
                             unsupported_payload, validate_dimensions, validate_exact_tiling,
                             validate_selection)

OPERATION = "gptq-w4a16"

# Defaults chosen to exercise more than one quantization group while keeping
# every dimension a multiple of the 32-element EU width.
DEFAULT_M = 32
DEFAULT_N = 64
DEFAULT_K = 256
DEFAULT_GROUP_SIZE = 128
DEFAULT_BLOCK_M = 32
DEFAULT_BLOCK_N = 64

# Substrings that identify "this target has no lowering for the dequantize".
# The codegen diagnostic names the op as `tl.tpu.dq2`.
_MISSING_DEQUANT_MARKERS = ("dq2",)


def _reports_missing_dequant(error: BaseException) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in _MISSING_DEQUANT_MARKERS)


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


def _group_count(k: int, group_size: int) -> int:
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if k % group_size:
        raise ValueError(f"group_size must divide K: group_size={group_size}, K={k}")
    return k // group_size


def build_gptq_w4a16(*,
                     m: int = DEFAULT_M,
                     n: int = DEFAULT_N,
                     k: int = DEFAULT_K,
                     group_size: int = DEFAULT_GROUP_SIZE,
                     block_m: int = DEFAULT_BLOCK_M,
                     block_n: int = DEFAULT_BLOCK_N,
                     dtype: str = "float16",
                     programming_model: str = "rv"):
    """Build the dequantize-then-GEMM program.

    Each block owns `(block_m, block_n)` of the result: it dequantizes that
    column block of the weight into FP16/BF16 local memory, then multiplies the
    matching activation rows against it.
    """
    if programming_model not in ("tpukernel", "rv"):
        raise ValueError(f"unsupported TPU programming model: {programming_model!r}")
    torch_dtype(dtype)
    validate_dimensions(
        OPERATION, m=m, n=n, k=k, block_m=block_m, block_n=block_n, group_size=group_size)
    validate_exact_tiling(OPERATION, ("m", m, block_m), ("n", n, block_n))
    groups = _group_count(k, group_size)
    if k % 32:
        raise ValueError(f"K must be a multiple of the 32-element EU width, got {k}")

    @T.prim_func
    def gptq_w4a16(left: T.Tensor((m, k), dtype), right: T.Tensor((n, k // 2), "uint8"),
                   offset_scale: T.Tensor((n, groups), "uint32"), result: T.Tensor((m, n), dtype)):
        with T.Kernel(T.ceildiv(m, block_m), T.ceildiv(n, block_n), is_cpu=True) as (bx, by):
            left_compute = T.alloc_shared((block_m, k), dtype)
            packed_compute = T.alloc_shared((block_n, k // 2), "uint8")
            offset_scale_compute = T.alloc_shared((block_n, groups), "uint32")
            weight_compute = T.alloc_shared((block_n, k), dtype)
            accumulate = T.alloc_shared((block_m, block_n), "float32")
            output_compute = T.alloc_shared((block_m, block_n), dtype)

            # Weight column block, dequantized in place.
            T.ppl_copy(right[by * block_n, 0], packed_compute)
            T.ppl_copy(offset_scale[by * block_n, 0], offset_scale_compute)
            T.ppl_dq2(weight_compute, packed_compute, offset_scale_compute, group_size)

            # Matching activation rows.
            T.ppl_copy(left[bx * block_m, 0], left_compute)

            T.ppl_fill(accumulate, T.float32(0.0))
            T.ppl_gemm(left_compute, weight_compute, accumulate,
                       transpose_B=True, accumulate=False)
            T.ppl_copy(accumulate, output_compute)
            T.ppl_copy(output_compute, result[bx * block_m, by * block_n])

    return gptq_w4a16


def host_inputs(*, m: int, n: int, k: int, group_size: int, dtype: str,
                seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build GPTQ-shaped host tensors: packed 4-bit codes plus per-group offset/scale.

    The codes are drawn independently of the offset/scale so the oracle
    exercises the dequantize arithmetic rather than any particular quantizer.
    """
    groups = _group_count(k, group_size)
    generator = torch.Generator().manual_seed(seed)

    left = (torch.randn((m, k), generator=generator) * 0.25).to(torch_dtype(dtype))

    # Two 4-bit codes per byte, low nibble first (input_reorder in the reference).
    codes = torch.randint(0, 16, (n, k), generator=generator, dtype=torch.int64)
    right = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)

    # One (offset, scale) pair per group, packed low-half offset / high-half
    # scale, both FP16 (scale_zp_reorder in the reference).
    offset = (torch.rand((n, groups), generator=generator) * 2.0).to(torch.float16)
    scale = (torch.rand((n, groups), generator=generator) * 0.05 + 0.001).to(torch.float16)
    offset_bits = offset.view(torch.int16).numpy().view(np.uint16).astype(np.uint32)
    scale_bits = scale.view(torch.int16).numpy().view(np.uint16).astype(np.uint32)
    packed = offset_bits | (scale_bits << np.uint32(16))
    offset_scale = torch.from_numpy(packed).to(torch.uint32)

    return left, right, offset_scale


def torch_reference(left: torch.Tensor, right: torch.Tensor, offset_scale: torch.Tensor, *,
                    n: int, k: int, group_size: int) -> torch.Tensor:
    """Host oracle: unpack the 4-bit weights, dequantize per group, then GEMM.

    Mirrors `input_reorder` and `scale_zp_reorder` from
    `examples/cxx/matmul/w4a16_matmul_dq2.py` so the oracle and the device
    kernel agree on the packing, leaving the dequantize arithmetic as the only
    thing under test.
    """
    groups = _group_count(k, group_size)

    packed = right.to(torch.int64)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).reshape(n, k).to(torch.float32)

    halves = offset_scale.view(torch.uint16).reshape(n, groups, 2)
    offset = halves[..., 0].view(torch.float16).to(torch.float32)
    scale = halves[..., 1].view(torch.float16).to(torch.float32)
    offset = offset.repeat_interleave(group_size, dim=1)
    scale = scale.repeat_interleave(group_size, dim=1)

    weight = (codes - offset) * scale
    return torch.nn.functional.linear(left.float(), weight)


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

    left, right, offset_scale = host_inputs(
        m=m, n=n, k=k, group_size=group_size, dtype=dtype, seed=seed)
    result = torch.zeros((m, n), dtype=torch_dtype(dtype))
    program = build_gptq_w4a16(
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
        if _reports_missing_dequant(error):
            return unsupported_payload(
                operation=OPERATION,
                dtype=dtype,
                chip=chip,
                programming_model=programming_model,
                runtime_mode=runtime_mode,
                reason=("this target has no lowering for the group-wise 4-bit "
                        "dequantize that GPTQ weights require"),
                evidence={
                    "exception": type(error).__name__,
                    "message": _summarize_toolchain_error(str(error)),
                },
                parameters=parameters)
        raise
    compiled = time.monotonic()

    kernel(left, right, offset_scale, result)
    finished = time.monotonic()

    expected = torch_reference(
        left, right, offset_scale, n=n, k=k, group_size=group_size).to(result.dtype)
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
