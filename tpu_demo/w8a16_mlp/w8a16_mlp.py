# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""W8A16 MLP: fused SwiGLU MLP over fp8 block-scaled weights.

This is the operator behind torch-tpu's `mlp_w8a16_dq_forward` -- the dense
MLP and shared-expert path in DeepSeek-V3.  It is `w8a16_dq` applied three
times around a SwiGLU:

    gate = x @ dequant(gate_weight)^T
    up   = x @ dequant(up_weight)^T
    out  = (silu(gate) * up) @ dequant(down_weight)^T

Each weight is E4M3 with a `weight_scale_inv` block table, dequantized exactly
as in `w8a16_dq`: cast to the compute dtype, then multiply by one scale per
output row (see that module for why the table arrives pre-expanded on the row
axis).

Argument order
    Note the operator's own argument order is `(gate, up, down)`, which is the
    reverse of `llama_mlp_forward`'s `(up, gate)`.  This demo names every
    tensor, so the distinction only matters when comparing against the
    vendor's call sites.

Layouts
    All three weights use the checkpoint's native `nn.Linear` layout -- `[out,
    in]` -- so the same `(rows, cols)` / `(rows, cols//128)` convention covers
    all of them:

        x             (batch, hidden)
        gate_weight   (intermediate, hidden)   gate_scale   (intermediate, hidden//128)
        up_weight     (intermediate, hidden)   up_scale     (intermediate, hidden//128)
        down_weight   (hidden, intermediate)   down_scale   (hidden, intermediate//128)
        output        (batch, hidden)

    The vendor transposes `down_proj` at load time so its kernel can use an NN
    GEMM; this demo keeps the native layout and transposes in the GEMM instead.
    Same arithmetic, one fewer layout convention to remember.

Tiling
    The grid covers output rows.  The intermediate dimension is walked
    serially and each chunk's contribution to the down projection accumulates
    in FP32, so the intermediate activation never round-trips through global
    memory.
"""

from typing import Optional

import tilelang.language as T
import torch

from tpu_demo.common import (comparison, compile_and_launch, result_payload, tolerance, torch_dtype,
                             validate_dimensions, validate_exact_tiling, validate_selection)

OPERATION = "w8a16-mlp"

# Sized so every weight spans more than one 128x128 block; a smaller demo would
# make the block table a single constant and stop testing the block boundary.
DEFAULT_BATCH_SEQ = 32
DEFAULT_HIDDEN = 256
DEFAULT_INTERMEDIATE = 256
DEFAULT_GROUP_SIZE = 128
DEFAULT_BLOCK_BS = 32
DEFAULT_BLOCK_H = 64
DEFAULT_BLOCK_I = 64
ACCUM_DTYPE = "float32"


def _chunk_count(length: int, group_size: int) -> int:
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if length % group_size:
        raise ValueError(f"group_size must divide {length}: group_size={group_size}")
    return length // group_size


def build_w8a16_mlp(*,
                    batch_seq: int = DEFAULT_BATCH_SEQ,
                    hidden: int = DEFAULT_HIDDEN,
                    intermediate: int = DEFAULT_INTERMEDIATE,
                    group_size: int = DEFAULT_GROUP_SIZE,
                    block_bs: int = DEFAULT_BLOCK_BS,
                    block_h: int = DEFAULT_BLOCK_H,
                    block_i: int = DEFAULT_BLOCK_I,
                    dtype: str = "float16",
                    programming_model: str = "rv"):
    """Build the W8A16 SwiGLU MLP program.

    Every dimension must be divisible by its block size; there is no tail
    handling.  The block sizes must also divide the quantization group so that
    a weight tile never straddles a scale block.
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
        block_i=block_i,
        group_size=group_size)
    validate_exact_tiling(
        OPERATION, ("batch_seq", batch_seq, block_bs), ("hidden", hidden, block_h),
        ("intermediate", intermediate, block_i))
    for name, tile in (("block_h", block_h), ("block_i", block_i)):
        if group_size % tile:
            raise ValueError(f"{name} must divide group_size so a weight tile stays "
                             f"inside one scale block: {name}={tile}, group_size={group_size}")
    hidden_chunks = _chunk_count(hidden, group_size)
    inter_chunks = _chunk_count(intermediate, group_size)

    @T.macro
    def silu(gate_in, silu_out, x_neg, ones, x_neg_exp_1, work0, work1, coeff):
        T.ppl_mul_C(x_neg, gate_in, T.float32(-1.0))
        T.ppl_exp(x_neg, work0, work1, coeff)
        T.ppl_add(x_neg_exp_1, x_neg, ones)
        T.ppl_div(silu_out, gate_in, x_neg_exp_1)

    @T.macro
    def dequantize(raw_fp8, raw_staging, raw_scale, scale_column, weight_out):
        """Cast an fp8 weight tile and scale it, one row-column at a time.

        The global -> local hop must stay same-dtype: DMA never converts, so
        the fp8 bytes land in an fp8 staging tile first and the cast happens
        local-to-local.  Same two-step shape as `w8a16_dq`.
        """
        T.ppl_copy(raw_fp8, raw_staging)
        T.ppl_copy(raw_staging, weight_out)
        T.ppl_copy(raw_scale, scale_column)
        T.ppl_mul(weight_out, weight_out, scale_column)

    @T.prim_func
    def w8a16_mlp(x: T.Tensor((batch_seq, hidden), dtype),
                  gate_weight: T.Tensor((intermediate, hidden), "e4m3_float8"),
                  gate_scale: T.Tensor((intermediate, hidden_chunks), dtype),
                  up_weight: T.Tensor((intermediate, hidden), "e4m3_float8"),
                  up_scale: T.Tensor((intermediate, hidden_chunks), dtype),
                  down_weight: T.Tensor((hidden, intermediate), "e4m3_float8"),
                  down_scale: T.Tensor((hidden, inter_chunks), dtype),
                  output: T.Tensor((batch_seq, hidden), dtype)):
        with T.Kernel(batch_seq // block_bs, hidden // block_h, is_cpu=True) as (bx, by):
            x_block = T.alloc_shared((block_bs, block_h), dtype)

            gate_raw = T.alloc_shared((block_i, block_h), "e4m3_float8")
            up_raw = T.alloc_shared((block_i, block_h), "e4m3_float8")
            down_raw = T.alloc_shared((block_h, block_i), "e4m3_float8")
            gate_w_block = T.alloc_shared((block_i, block_h), dtype)
            up_w_block = T.alloc_shared((block_i, block_h), dtype)
            down_w_block = T.alloc_shared((block_h, block_i), dtype)
            gate_scale_col = T.alloc_shared((block_i, 1), dtype)
            up_scale_col = T.alloc_shared((block_i, 1), dtype)
            down_scale_col = T.alloc_shared((block_h, 1), dtype)

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

                    dequantize(gate_weight[bz * block_i, kk * block_h], gate_raw,
                               gate_scale[bz * block_i, kk * block_h // group_size],
                               gate_scale_col, gate_w_block)
                    dequantize(up_weight[bz * block_i, kk * block_h], up_raw,
                               up_scale[bz * block_i, kk * block_h // group_size],
                               up_scale_col, up_w_block)

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

                dequantize(down_weight[by * block_h, bz * block_i], down_raw,
                           down_scale[by * block_h, bz * block_i // group_size],
                           down_scale_col, down_w_block)
                T.ppl_fill(down_part_fp32, T.float32(0.0))
                T.ppl_gemm(gated_up_block, down_w_block, down_part_fp32,
                           transpose_B=True, accumulate=False)
                T.ppl_add(down_out_acc, down_out_acc, down_part_fp32)

            T.ppl_copy(down_out_acc, down_out_block)
            T.ppl_copy(down_out_block, output[bx * block_bs, by * block_h])

    return w8a16_mlp


def host_inputs(*, batch_seq: int, hidden: int, intermediate: int, group_size: int,
                dtype: str, seed: int):
    """Build an activation plus three fp8 weights and their block-scale tables.

    The scales are drawn independently of the weights so the oracle exercises
    the dequantize arithmetic rather than any particular quantizer, and each
    table is expanded along its row axis -- the layout the kernel reads.
    """
    hidden_chunks = _chunk_count(hidden, group_size)
    inter_chunks = _chunk_count(intermediate, group_size)
    compute = torch_dtype(dtype)
    generator = torch.Generator().manual_seed(seed)

    def fp8(rows: int, cols: int) -> torch.Tensor:
        return (torch.randn((rows, cols), generator=generator) * 0.25).to(torch.float8_e4m3fn)

    def scale_table(rows: int, chunks: int) -> torch.Tensor:
        blocks = torch.rand((rows // group_size, chunks), generator=generator) * 0.5 + 0.75
        return blocks.repeat_interleave(group_size, dim=0).to(compute)

    x = (torch.randn((batch_seq, hidden), generator=generator) * 0.5).to(compute)
    return (
        x,
        fp8(intermediate, hidden), scale_table(intermediate, hidden_chunks),
        fp8(intermediate, hidden), scale_table(intermediate, hidden_chunks),
        fp8(hidden, intermediate), scale_table(hidden, inter_chunks),
    )


def torch_reference(x, gate_weight, gate_scale, up_weight, up_scale, down_weight, down_scale, *,
                    group_size: int, dtype: str) -> torch.Tensor:
    """Host oracle: SwiGLU MLP over dequantized weights, rounded like the device.

    The device holds each dequantized weight in a compute-dtype tile, so the
    product is rounded there before the GEMM; it also round-trips the gated
    activation through a compute-dtype tile before the down projection.  Both
    roundings are mirrored here, because an oracle that skips them measures
    ideal arithmetic rather than the operator.
    """
    compute = torch_dtype(dtype)

    def dequantize(raw: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        dequant = raw.to(torch.float32) * scale.to(torch.float32).repeat_interleave(
            group_size, dim=1)
        return dequant.to(compute).to(torch.float32)

    xf = x.to(torch.float32)
    gate = xf @ dequantize(gate_weight, gate_scale).T
    up = xf @ dequantize(up_weight, up_scale).T

    gated = up * (gate / (1.0 + torch.exp(-gate)))
    gated = gated.to(compute).to(torch.float32)
    return (gated @ dequantize(down_weight, down_scale).T).to(compute)


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

    batch_seq = DEFAULT_BATCH_SEQ
    hidden, intermediate = DEFAULT_HIDDEN, DEFAULT_INTERMEDIATE
    group_size = DEFAULT_GROUP_SIZE
    parameters = {
        "batch_seq": batch_seq,
        "hidden": hidden,
        "intermediate": intermediate,
        "group_size": group_size,
        "block_bs": DEFAULT_BLOCK_BS,
        "block_h": DEFAULT_BLOCK_H,
        "block_i": DEFAULT_BLOCK_I,
        "seed": seed,
    }

    host = host_inputs(
        batch_seq=batch_seq, hidden=hidden, intermediate=intermediate,
        group_size=group_size, dtype=dtype, seed=seed)
    output = torch.zeros((batch_seq, hidden), dtype=torch_dtype(dtype))
    program = build_w8a16_mlp(
        batch_seq=batch_seq,
        hidden=hidden,
        intermediate=intermediate,
        group_size=group_size,
        block_bs=DEFAULT_BLOCK_BS,
        block_h=DEFAULT_BLOCK_H,
        block_i=DEFAULT_BLOCK_I,
        dtype=dtype,
        programming_model=programming_model)

    timing = compile_and_launch(
        program,
        (*host, output),
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode)

    expected = torch_reference(*host, group_size=group_size, dtype=dtype).to(output.dtype)
    atol, rtol = tolerance(dtype, "swiglu")
    metrics = comparison(output, expected, atol=atol, rtol=rtol)
    return result_payload(
        operation=OPERATION,
        dtype=dtype,
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode,
        metrics=metrics,
        timing=timing,
        parameters=parameters)
