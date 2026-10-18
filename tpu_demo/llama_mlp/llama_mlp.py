# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Llama MLP: gate/up projections, SwiGLU, down projection.

    gate = x @ gate_weight^T
    up   = x @ up_weight^T
    out  = (silu(gate) * up) @ down_weight^T

Weight layouts follow `soph_llama.py` in the vLLM port: up and gate are
`(intermediate, hidden)`, down is `(hidden, intermediate)`.

Tiling
    The grid covers output rows only.  The intermediate dimension is walked
    serially, and each tile's contribution to the down projection is
    accumulated in FP32, so the intermediate activation never round-trips
    through global memory.

    `silu(g) = g / (1 + exp(-g))` is evaluated in FP32 with a materialised
    ones-tile rather than `ppl_add_C`; both that and the division form match
    the reference kernel this was ported from.

Dtype note
    `ppl_exp` is FP32-only on RV, and its coefficient buffer has a fixed
    `(64, 32)` shape on both backends.  The coefficient values are unused on RV
    but the buffer must still be allocated and passed.
"""

from typing import Optional

import tilelang.language as T
import torch

from tpu_demo.common import (comparison, compile_and_launch, result_payload, tolerance, torch_dtype,
                             validate_dimensions, validate_exact_tiling, validate_selection)

OPERATION = "llama-mlp"

DEFAULT_BATCH_SEQ = 32
DEFAULT_HIDDEN = 64
DEFAULT_INTERMEDIATE = 64
ACCUM_DTYPE = "float32"


def build_llama_mlp(*,
                    batch_seq: int = DEFAULT_BATCH_SEQ,
                    hidden: int = DEFAULT_HIDDEN,
                    intermediate: int = DEFAULT_INTERMEDIATE,
                    block_bs: int = 32,
                    block_h: int = 32,
                    block_i: int = 32,
                    dtype: str = "float16",
                    programming_model: str = "rv"):
    """Build the Llama MLP program.

    Every dimension must be divisible by its block size; there is no tail
    handling.
    """
    if programming_model not in ("tpukernel", "rv"):
        raise ValueError(f"unsupported TPU programming model: {programming_model!r}")
    torch_dtype(dtype)
    validate_dimensions(
        OPERATION,
        batch_seq=batch_seq,
        hidden=hidden,
        intermediate=intermediate,
        block_bs=block_bs,
        block_h=block_h,
        block_i=block_i)
    validate_exact_tiling(
        OPERATION, ("batch_seq", batch_seq, block_bs), ("hidden", hidden, block_h),
        ("intermediate", intermediate, block_i))

    @T.macro
    def silu(gate_in, silu_out, x_neg, ones, x_neg_exp_1, work0, work1, coeff):
        T.ppl_mul_C(x_neg, gate_in, T.float32(-1.0))
        T.ppl_exp(x_neg, work0, work1, coeff)
        T.ppl_add(x_neg_exp_1, x_neg, ones)
        T.ppl_div(silu_out, gate_in, x_neg_exp_1)

    @T.prim_func
    def llama_mlp(x: T.Tensor((batch_seq, hidden), dtype),
                  up_weight: T.Tensor((intermediate, hidden), dtype),
                  gate_weight: T.Tensor((intermediate, hidden), dtype),
                  down_weight: T.Tensor((hidden, intermediate), dtype),
                  output: T.Tensor((batch_seq, hidden), dtype)):
        with T.Kernel(batch_seq // block_bs, hidden // block_h, is_cpu=True) as (bx, by):
            x_block = T.alloc_shared((block_bs, block_h), dtype)
            gate_w_block = T.alloc_shared((block_i, block_h), dtype)
            up_w_block = T.alloc_shared((block_i, block_h), dtype)
            down_w_block = T.alloc_shared((block_h, block_i), dtype)
            gated_up_block = T.alloc_shared((block_bs, block_i), dtype)

            down_out_acc = T.alloc_shared((block_bs, block_h), ACCUM_DTYPE)
            down_out_block = T.alloc_shared((block_bs, block_h), dtype)

            gate_out_fp32 = T.alloc_shared((block_bs, block_i), ACCUM_DTYPE)
            up_out_fp32 = T.alloc_shared((block_bs, block_i), ACCUM_DTYPE)
            proj_part_fp32 = T.alloc_shared((block_bs, block_i), ACCUM_DTYPE)
            down_part_fp32 = T.alloc_shared((block_bs, block_h), ACCUM_DTYPE)

            silu_out = T.alloc_shared((block_bs, block_i), ACCUM_DTYPE)
            gated_up_fp32 = T.alloc_shared((block_bs, block_i), ACCUM_DTYPE)

            x_neg = T.alloc_shared((block_bs, block_i), ACCUM_DTYPE)
            ones = T.alloc_shared((block_bs, block_i), ACCUM_DTYPE)
            x_neg_exp_1 = T.alloc_shared((block_bs, block_i), ACCUM_DTYPE)
            exp_work0 = T.alloc_shared((block_bs, block_i), ACCUM_DTYPE)
            exp_work1 = T.alloc_shared((block_bs, block_i), ACCUM_DTYPE)
            # ppl_exp requires a (64, 32) coefficient buffer; unused on RV.
            exp_coeff = T.alloc_shared((64, 32), ACCUM_DTYPE)

            T.ppl_fill(ones, T.float32(1.0))
            T.ppl_fill(down_out_acc, T.float32(0.0))

            for bz in range(intermediate // block_i):
                T.ppl_fill(gate_out_fp32, T.float32(0.0))
                T.ppl_fill(up_out_fp32, T.float32(0.0))

                # Both projections, each with a full reduction over `hidden`.
                for kk in range(hidden // block_h):
                    T.ppl_copy(x[bx * block_bs, kk * block_h], x_block)
                    T.ppl_copy(gate_weight[bz * block_i, kk * block_h], gate_w_block)
                    T.ppl_copy(up_weight[bz * block_i, kk * block_h], up_w_block)

                    T.ppl_fill(proj_part_fp32, T.float32(0.0))
                    T.ppl_gemm(x_block, gate_w_block, proj_part_fp32,
                               transpose_B=True, accumulate=False)
                    T.ppl_add(gate_out_fp32, gate_out_fp32, proj_part_fp32)

                    T.ppl_fill(proj_part_fp32, T.float32(0.0))
                    T.ppl_gemm(x_block, up_w_block, proj_part_fp32,
                               transpose_B=True, accumulate=False)
                    T.ppl_add(up_out_fp32, up_out_fp32, proj_part_fp32)

                silu(gate_out_fp32, silu_out, x_neg, ones, x_neg_exp_1, exp_work0, exp_work1,
                     exp_coeff)
                T.ppl_mul(gated_up_fp32, silu_out, up_out_fp32)
                T.ppl_copy(gated_up_fp32, gated_up_block)

                T.ppl_copy(down_weight[by * block_h, bz * block_i], down_w_block)
                T.ppl_fill(down_part_fp32, T.float32(0.0))
                T.ppl_gemm(gated_up_block, down_w_block, down_part_fp32,
                           transpose_B=True, accumulate=False)
                T.ppl_add(down_out_acc, down_out_acc, down_part_fp32)

            T.ppl_copy(down_out_acc, down_out_block)
            T.ppl_copy(down_out_block, output[bx * block_bs, by * block_h])

    return llama_mlp


def torch_reference(x: torch.Tensor, up_weight: torch.Tensor, gate_weight: torch.Tensor,
                    down_weight: torch.Tensor) -> torch.Tensor:
    """Host oracle: SwiGLU MLP evaluated in FP32."""
    xf = x.float()
    gate = torch.nn.functional.linear(xf, gate_weight.float())
    up = torch.nn.functional.linear(xf, up_weight.float())
    return torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, down_weight.float())


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

    batch_seq, hidden, intermediate = DEFAULT_BATCH_SEQ, DEFAULT_HIDDEN, DEFAULT_INTERMEDIATE
    blocks = {"block_bs": 32, "block_h": 32, "block_i": 32}
    parameters = {
        "batch_seq": batch_seq,
        "hidden": hidden,
        "intermediate": intermediate,
        "seed": seed,
        **blocks,
    }

    host_dtype = torch_dtype(dtype)
    generator = torch.Generator().manual_seed(seed)
    # Bounded inputs keep the exp argument in range and the accumulation well
    # inside the reduced-precision representation.
    x = (torch.randn((batch_seq, hidden), generator=generator) * 0.25).to(host_dtype)
    up_weight = (torch.randn((intermediate, hidden), generator=generator) * 0.25).to(host_dtype)
    gate_weight = (torch.randn((intermediate, hidden), generator=generator) * 0.25).to(host_dtype)
    down_weight = (torch.randn((hidden, intermediate), generator=generator) * 0.25).to(host_dtype)
    destination = torch.zeros((batch_seq, hidden), dtype=host_dtype)

    timing = compile_and_launch(
        build_llama_mlp(
            batch_seq=batch_seq,
            hidden=hidden,
            intermediate=intermediate,
            dtype=dtype,
            programming_model=programming_model,
            **blocks),
        (x, up_weight, gate_weight, down_weight, destination),
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode,
    )

    expected = torch_reference(x, up_weight, gate_weight, down_weight).to(host_dtype)
    atol, rtol = tolerance(dtype, "swiglu")
    metrics = comparison(destination, expected, atol=atol, rtol=rtol)
    return result_payload(
        operation=OPERATION,
        dtype=dtype,
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode,
        metrics=metrics,
        timing=timing,
        parameters=parameters)
