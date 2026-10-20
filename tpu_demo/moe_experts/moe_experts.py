# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Fused MoE: routed experts as one grouped W8A16 SwiGLU MLP.

This is the operator behind torch-tpu's `fused_moe_fused_experts`, the routed
half of DeepSeek-V3's `DeepseekV3MoE`.  The reference is the vendor's
`fusedMoE.pl`.

Scope: static routing
    A real MoE sorts its tokens by expert before the kernel runs, so the
    per-expert work is a contiguous slice rather than a scatter-gather driven
    by per-token counts.  This demo fixes that sorted assignment at build
    time, which is what lets every index in the kernel be a compile-time
    constant:

      * expert `e` owns the contiguous output rows `[e*cap, (e+1)*cap)`
      * `token_index` names the token each of those rows came from, so the
        gather is a single `ppl_embedding` over a static index table

    The dynamic part a scheduler really needs -- counting tokens per expert,
    and scattering results back to their original positions -- is *not*
    modelled.  On the current SG2260E RV firmware `ppl_scatter` writes nothing
    (see the `mla-decode` demo and the survey note), so a per-row write-back is
    not expressible here anyway.  Working with a sorted assignment sidesteps it
    honestly rather than papering over it.

Operator boundary
    The kernel returns the per-(token, slot) expert outputs, exactly as
    `fused_moe_fused_experts` does.  Applying the routing weights is the
    caller's job -- the vendor kernel takes `routing_weights` and never reads
    them, and `DeepseekV3MoE.forward` does the weighted sum itself with a
    `torch.bmm` afterwards.  So the routing *weights* are not part of this
    demo; the routing *assignment* is, and it is what `token_index` encodes.

Layouts
    Weights are stacked along the expert axis and read with a static row
    offset, so each expert looks exactly like one `w8a16_mlp`:

        x             (tokens, hidden)                fp16/bf16
        token_index   (experts * cap, 1)              uint32
        gate_weight   (experts * inter, hidden)       e4m3
        gate_scale    (experts * inter, 1)            fp16/bf16
        up_weight     (experts * inter, hidden)       e4m3
        up_scale      (experts * inter, 1)            fp16/bf16
        down_weight   (experts * hidden, inter)       e4m3
        down_scale    (experts * hidden, 1)           fp16/bf16
        gathered      (experts * cap, hidden)         fp16/bf16

    Every dimension stays inside one 128-wide quantization block, since
    `w8a16_dq` is where the block boundary is exercised.
"""

from typing import Optional, Sequence

import tilelang.language as T
import torch

from tpu_demo.common import (comparison, compile_and_launch, result_payload, tolerance, torch_dtype,
                             validate_dimensions, validate_selection)

OPERATION = "moe-experts"

TOKENS = 8
HIDDEN = 64
INTERMEDIATE = 32
NUM_EXPERTS = 4
TOP_K = 2
# Every expert takes exactly this many of the `tokens * top_k` routed slots.
CAPACITY = TOKENS * TOP_K // NUM_EXPERTS

ACCUM_DTYPE = "float32"


def routed_experts(token: int) -> tuple[int, int]:
    """The two experts a token routes to.

    Fixed rather than learned: the demo's subject is the grouped expert MLP,
    not the gate.
    """
    return token % NUM_EXPERTS, (token + 1) % NUM_EXPERTS


def token_order() -> tuple[int, ...]:
    """Tokens sorted by expert: entry `e*CAPACITY + s` is expert `e`'s s-th token.

    This is the permutation a real MoE's sorter produces, and it is what lets
    the kernel walk each expert's rows as one contiguous slice.
    """
    per_expert: list[list[int]] = [[] for _ in range(NUM_EXPERTS)]
    for token in range(TOKENS):
        for expert in routed_experts(token):
            per_expert[expert].append(token)
    order: list[int] = []
    for expert in range(NUM_EXPERTS):
        if len(per_expert[expert]) != CAPACITY:
            raise ValueError(f"expert {expert} got {len(per_expert[expert])} tokens, "
                             f"expected {CAPACITY}")
        order.extend(per_expert[expert])
    return tuple(order)


def build_moe_experts(*,
                      tokens: int = TOKENS,
                      hidden: int = HIDDEN,
                      intermediate: int = INTERMEDIATE,
                      num_experts: int = NUM_EXPERTS,
                      top_k: int = TOP_K,
                      dtype: str = "float16",
                      programming_model: str = "rv"):
    """Build the grouped expert program.

    One core walks the experts serially.  Each expert dequantizes its slice of
    the stacked weights, runs a SwiGLU MLP over its `capacity` gathered rows,
    and writes its contiguous block of the output.
    """
    if programming_model not in ("tpukernel", "rv"):
        raise ValueError(f"unsupported TPU programming model: {programming_model!r}")
    torch_dtype(dtype)
    capacity = tokens * top_k // num_experts
    if tokens * top_k % num_experts:
        raise ValueError(f"tokens * top_k must divide evenly over experts: "
                         f"{tokens} * {top_k} / {num_experts}")
    validate_dimensions(
        OPERATION, tokens=tokens, hidden=hidden, intermediate=intermediate,
        num_experts=num_experts, top_k=top_k, capacity=capacity)
    staged = num_experts * capacity

    @T.macro
    def silu(gate_in, silu_out, x_neg, ones, x_neg_exp_1, work0, work1, coeff):
        T.ppl_mul_C(x_neg, gate_in, T.float32(-1.0))
        T.ppl_exp(x_neg, work0, work1, coeff)
        T.ppl_add(x_neg_exp_1, x_neg, ones)
        T.ppl_div(silu_out, gate_in, x_neg_exp_1)

    @T.prim_func
    def moe_experts(x: T.Tensor((tokens, hidden), dtype),
                    token_index: T.Tensor((staged, 1), "uint32"),
                    gate_weight: T.Tensor((num_experts * intermediate, hidden), "e4m3_float8"),
                    gate_scale: T.Tensor((num_experts * intermediate, 1), dtype),
                    up_weight: T.Tensor((num_experts * intermediate, hidden), "e4m3_float8"),
                    up_scale: T.Tensor((num_experts * intermediate, 1), dtype),
                    down_weight: T.Tensor((num_experts * hidden, intermediate), "e4m3_float8"),
                    down_scale: T.Tensor((num_experts * hidden, 1), dtype),
                    x_stage: T.Tensor((staged, hidden), dtype),
                    gathered: T.Tensor((staged, hidden), dtype)):
        with T.Kernel(1, is_cpu=True) as (bx,):
            # One gather for the whole sorted batch, exactly as the routed
            # experts of a real MoE are fed.
            T.ppl_embedding(x_stage, x, token_index)

            x_local = T.alloc_shared((capacity, hidden), dtype)

            gate_raw = T.alloc_shared((intermediate, hidden), "e4m3_float8")
            up_raw = T.alloc_shared((intermediate, hidden), "e4m3_float8")
            down_raw = T.alloc_shared((hidden, intermediate), "e4m3_float8")
            gate_tile = T.alloc_shared((intermediate, hidden), dtype)
            up_tile = T.alloc_shared((intermediate, hidden), dtype)
            down_tile = T.alloc_shared((hidden, intermediate), dtype)
            gate_col = T.alloc_shared((intermediate, 1), dtype)
            up_col = T.alloc_shared((intermediate, 1), dtype)
            down_col = T.alloc_shared((hidden, 1), dtype)

            gate_out = T.alloc_shared((capacity, intermediate), ACCUM_DTYPE)
            up_out = T.alloc_shared((capacity, intermediate), ACCUM_DTYPE)
            silu_out = T.alloc_shared((capacity, intermediate), ACCUM_DTYPE)
            gated = T.alloc_shared((capacity, intermediate), ACCUM_DTYPE)
            gated_h = T.alloc_shared((capacity, intermediate), dtype)
            down_out = T.alloc_shared((capacity, hidden), ACCUM_DTYPE)
            out_tile = T.alloc_shared((capacity, hidden), dtype)

            x_neg = T.alloc_shared((capacity, intermediate), ACCUM_DTYPE)
            ones = T.alloc_shared((capacity, intermediate), ACCUM_DTYPE)
            x_neg_exp_1 = T.alloc_shared((capacity, intermediate), ACCUM_DTYPE)
            exp_work0 = T.alloc_shared((capacity, intermediate), ACCUM_DTYPE)
            exp_work1 = T.alloc_shared((capacity, intermediate), ACCUM_DTYPE)
            # ppl_exp requires a (64, 32) coefficient buffer; unused on RV.
            exp_coeff = T.alloc_shared((64, 32), ACCUM_DTYPE)

            T.ppl_fill(ones, T.float32(1.0))

            for expert in range(num_experts):
                # This expert's slice of the stacked weights, dequantized here.
                T.ppl_copy(gate_weight[expert * intermediate, 0], gate_raw)
                T.ppl_copy(gate_raw, gate_tile)
                T.ppl_copy(gate_scale[expert * intermediate, 0], gate_col)
                T.ppl_mul(gate_tile, gate_tile, gate_col)

                T.ppl_copy(up_weight[expert * intermediate, 0], up_raw)
                T.ppl_copy(up_raw, up_tile)
                T.ppl_copy(up_scale[expert * intermediate, 0], up_col)
                T.ppl_mul(up_tile, up_tile, up_col)

                T.ppl_copy(down_weight[expert * hidden, 0], down_raw)
                T.ppl_copy(down_raw, down_tile)
                T.ppl_copy(down_scale[expert * hidden, 0], down_col)
                T.ppl_mul(down_tile, down_tile, down_col)

                # This expert's tokens, already contiguous thanks to the sort.
                T.ppl_copy(x_stage[expert * capacity, 0], x_local)

                T.ppl_fill(gate_out, T.float32(0.0))
                T.ppl_gemm(x_local, gate_tile, gate_out,
                           transpose_B=True, accumulate=False)
                T.ppl_fill(up_out, T.float32(0.0))
                T.ppl_gemm(x_local, up_tile, up_out,
                           transpose_B=True, accumulate=False)

                silu(gate_out, silu_out, x_neg, ones, x_neg_exp_1, exp_work0, exp_work1,
                     exp_coeff)
                T.ppl_mul(gated, silu_out, up_out)
                T.ppl_copy(gated, gated_h)

                T.ppl_fill(down_out, T.float32(0.0))
                T.ppl_gemm(gated_h, down_tile, down_out,
                           transpose_B=True, accumulate=False)
                T.ppl_copy(down_out, out_tile)

                T.ppl_copy(out_tile, gathered[expert * capacity, 0])

    return moe_experts


def host_inputs(*, tokens: int, hidden: int, intermediate: int, num_experts: int,
                dtype: str, seed: int):
    """Build the activation plus each expert's stacked fp8 weights and scales."""
    compute = torch_dtype(dtype)
    generator = torch.Generator().manual_seed(seed)
    order = token_order()

    def fp8(rows: int, cols: int) -> torch.Tensor:
        return (torch.randn((rows, cols), generator=generator) * 0.25).to(torch.float8_e4m3fn)

    def scale_col(rows: int) -> torch.Tensor:
        # Every dimension here is inside a single 128-wide quantization block.
        return (torch.rand((rows, 1), generator=generator) * 0.5 + 0.75).to(compute)

    return {
        "x": (torch.randn((tokens, hidden), generator=generator) * 0.5).to(compute),
        "token_index": torch.tensor(order, dtype=torch.uint32).reshape(len(order), 1),
        "gate_weight": fp8(num_experts * intermediate, hidden),
        "gate_scale": scale_col(num_experts * intermediate),
        "up_weight": fp8(num_experts * intermediate, hidden),
        "up_scale": scale_col(num_experts * intermediate),
        "down_weight": fp8(num_experts * hidden, intermediate),
        "down_scale": scale_col(num_experts * hidden),
        "x_stage": torch.zeros((num_experts * CAPACITY, hidden), dtype=compute),
        "gathered": torch.zeros((num_experts * CAPACITY, hidden), dtype=compute),
    }


def torch_reference(inputs, *, hidden: int, intermediate: int, num_experts: int,
                    dtype: str) -> torch.Tensor:
    """Host oracle: one SwiGLU MLP per expert over its sorted rows.

    The dequantized weights and the gated activation are rounded to the compute
    dtype where the device rounds them, so this is an oracle for the operator
    rather than for ideal arithmetic.
    """
    compute = torch_dtype(dtype)
    capacity = CAPACITY

    def dequantize(raw: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return (raw.to(torch.float32) * scale.to(torch.float32)).to(compute).to(torch.float32)

    def rounded(value: torch.Tensor) -> torch.Tensor:
        return value.to(compute).to(torch.float32)

    x = inputs["x"].to(torch.float32)
    order = inputs["token_index"].reshape(-1).to(torch.int64)
    rows = []

    for expert in range(num_experts):
        index = order[expert * capacity:(expert + 1) * capacity]
        x_e = x[index]

        gate_w = dequantize(inputs["gate_weight"][expert * intermediate:(expert + 1) * intermediate],
                            inputs["gate_scale"][expert * intermediate:(expert + 1) * intermediate])
        up_w = dequantize(inputs["up_weight"][expert * intermediate:(expert + 1) * intermediate],
                          inputs["up_scale"][expert * intermediate:(expert + 1) * intermediate])
        down_w = dequantize(inputs["down_weight"][expert * hidden:(expert + 1) * hidden],
                            inputs["down_scale"][expert * hidden:(expert + 1) * hidden])

        gate = x_e @ gate_w.T
        up = x_e @ up_w.T
        gated = rounded(up * (gate / (1.0 + torch.exp(-gate))))
        rows.append(rounded(gated @ down_w.T))

    return torch.cat(rows, dim=0).to(compute)


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

    parameters = {
        "tokens": TOKENS,
        "hidden": HIDDEN,
        "intermediate": INTERMEDIATE,
        "num_experts": NUM_EXPERTS,
        "top_k": TOP_K,
        "capacity": CAPACITY,
        "token_order": list(token_order()),
        "seed": seed,
    }

    inputs = host_inputs(
        tokens=TOKENS, hidden=HIDDEN, intermediate=INTERMEDIATE,
        num_experts=NUM_EXPERTS, dtype=dtype, seed=seed)

    program = build_moe_experts(
        tokens=TOKENS,
        hidden=HIDDEN,
        intermediate=INTERMEDIATE,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        dtype=dtype,
        programming_model=programming_model)

    argument_order = ("x", "token_index", "gate_weight", "gate_scale", "up_weight",
                      "up_scale", "down_weight", "down_scale", "x_stage", "gathered")
    timing = compile_and_launch(
        program,
        [inputs[name] for name in argument_order],
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode)

    expected = torch_reference(
        inputs, hidden=HIDDEN, intermediate=INTERMEDIATE,
        num_experts=NUM_EXPERTS, dtype=dtype).to(inputs["gathered"].dtype)
    atol, rtol = tolerance(dtype, "swiglu")
    metrics = comparison(inputs["gathered"], expected, atol=atol, rtol=rtol)
    return result_payload(
        operation=OPERATION,
        dtype=dtype,
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode,
        metrics=metrics,
        timing=timing,
        parameters=parameters)
