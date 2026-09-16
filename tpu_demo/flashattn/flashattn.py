# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Tiled online-softmax attention for TPU-Kernel and RV Tensor.

The public tensors retain TileLang's ``[batch, sequence, heads, head_dim]``
layout. Explicit singleton slices preserve the four-dimensional DMA descriptor
while a sequence/head-dimension tile is copied to GEMM-local storage. FP16 and
BF16 use their native matrix-engine type. FP32 is an explicit
mixed-precision contract: Q/K/V are converted to BF16 for matrix multiplication,
while scores, online-softmax state, and output accumulation remain FP32. The
probability tile is cast to the matrix-engine type for the probability/value
GEMM, whereas the normalization denominator remains FP32. Validation uses an
ideal semantic softmax oracle; dtype-specific tolerances cover that explicit
probability cast rather than duplicating implementation rounding in the oracle.
"""

import math
from typing import Optional

import tilelang.language as T
import torch

from tpu_demo.common import (comparison, compile_and_launch, result_payload, tolerance, torch_dtype,
                             validate_dimensions, validate_exact_tiling, validate_selection)


def build_flashattn(*,
                    batch: int = 1,
                    heads: int = 1,
                    sequence: int = 32,
                    head_dim: int = 16,
                    block_m: int = 16,
                    block_n: int = 16,
                    dtype: str = "float16"):
    torch_dtype(dtype)
    validate_dimensions(
        "flashattn",
        batch=batch,
        heads=heads,
        sequence=sequence,
        head_dim=head_dim,
        block_m=block_m,
        block_n=block_n)
    validate_exact_tiling("flashattn", ("sequence/block_m", sequence, block_m),
                          ("sequence/block_n", sequence, block_n))
    scale = 1.0 / math.sqrt(head_dim)

    if dtype != "float32":

        @T.prim_func
        def flashattn_low_precision(Q: T.Tensor(
            (batch, sequence, heads, head_dim), dtype), K: T.Tensor(
                (batch, sequence, heads, head_dim), dtype), V: T.Tensor(
                    (batch, sequence, heads, head_dim), dtype), Mask: T.Tensor(
                        (sequence, sequence), "float32"), Output: T.Tensor(
                            (batch, sequence, heads, head_dim), dtype)):
            with T.Kernel(T.ceildiv(sequence, block_m), heads, batch, is_cpu=True) as (bx, by, bz):
                q_compute = T.alloc_shared((block_m, head_dim), dtype)
                k_compute = T.alloc_shared((block_n, head_dim), dtype)
                v_compute = T.alloc_shared((block_n, head_dim), dtype)
                output_local = T.alloc_shared((block_m, head_dim), dtype)
                scores = T.alloc_shared((block_m, block_n), "float32")
                mask = T.alloc_shared((block_m, block_n), "float32")
                scores_compute = T.alloc_shared((block_m, block_n), dtype)
                accumulator = T.alloc_shared((block_m, head_dim), "float32")
                normalized = T.alloc_shared((block_m, head_dim), "float32")
                row_max = T.alloc_shared((block_m, 1), "float32")
                current_max = T.alloc_shared((block_m, 1), "float32")
                previous_max = T.alloc_shared((block_m, 1), "float32")
                previous_scale = T.alloc_shared((block_m, 1), "float32")
                chunk_sum = T.alloc_shared((block_m, 1), "float32")
                row_sum = T.alloc_shared((block_m, 1), "float32")
                scale_work0 = T.alloc_shared((block_m, 1), "float32")
                scale_work1 = T.alloc_shared((block_m, 1), "float32")
                exp_coeff = T.alloc_shared((64, 32), "float32")
                score_work0 = T.alloc_shared((block_m, block_n), "float32")
                score_work1 = T.alloc_shared((block_m, block_n), "float32")

                T.ppl_copy(Q[bz:bz + 1, bx * block_m:(bx + 1) * block_m, by:by + 1, 0:head_dim],
                           q_compute)
                T.ppl_fill(accumulator, T.float32(0))
                T.ppl_fill(row_sum, T.float32(0))
                T.ppl_fill(row_max, -T.infinity("float32"))

                for ko in T.serial(T.ceildiv(sequence, block_n)):
                    T.ppl_copy(K[bz:bz + 1, ko * block_n:(ko + 1) * block_n, by:by + 1, 0:head_dim],
                               k_compute)
                    T.ppl_copy(V[bz:bz + 1, ko * block_n:(ko + 1) * block_n, by:by + 1, 0:head_dim],
                               v_compute)

                    T.ppl_gemm(q_compute, k_compute, scores, transpose_B=True, accumulate=False)
                    T.ppl_mul_C(scores, scores, T.float32(scale))
                    T.ppl_copy(Mask[bx * block_m, ko * block_n], mask)
                    T.ppl_add(scores, scores, mask)
                    T.ppl_copy(row_max, previous_max)
                    T.ppl_reduce_max(scores, current_max, dim=1)
                    T.ppl_max(row_max, previous_max, current_max)
                    T.ppl_subtract(previous_scale, previous_max, row_max)
                    T.ppl_exp(previous_scale, scale_work0, scale_work1, exp_coeff)

                    T.ppl_subtract(scores, scores, row_max)
                    T.ppl_exp(scores, score_work0, score_work1, exp_coeff)
                    T.ppl_reduce_sum(scores, chunk_sum, dim=1)

                    T.ppl_mul(row_sum, row_sum, previous_scale)
                    T.ppl_add(row_sum, row_sum, chunk_sum)
                    T.ppl_mul(accumulator, accumulator, previous_scale)
                    T.ppl_copy(scores, scores_compute)
                    T.ppl_gemm(scores_compute, v_compute, accumulator, accumulate=True)

                T.ppl_div(normalized, accumulator, row_sum)
                T.ppl_copy(normalized, output_local)
                T.ppl_copy(
                    output_local, Output[bz:bz + 1, bx * block_m:(bx + 1) * block_m, by:by + 1,
                                         0:head_dim])

        return flashattn_low_precision

    @T.prim_func
    def flashattn_fp32(Q: T.Tensor((batch, sequence, heads, head_dim), "float32"), K: T.Tensor(
        (batch, sequence, heads, head_dim), "float32"), V: T.Tensor(
            (batch, sequence, heads, head_dim), "float32"), Mask: T.Tensor(
                (sequence, sequence), "float32"), Output: T.Tensor(
                    (batch, sequence, heads, head_dim), "float32")):
        with T.Kernel(T.ceildiv(sequence, block_m), heads, batch, is_cpu=True) as (bx, by, bz):
            q_input = T.alloc_shared((block_m, head_dim), "float32")
            k_input = T.alloc_shared((block_n, head_dim), "float32")
            v_input = T.alloc_shared((block_n, head_dim), "float32")
            q_compute = T.alloc_shared((block_m, head_dim), "bfloat16")
            k_compute = T.alloc_shared((block_n, head_dim), "bfloat16")
            v_compute = T.alloc_shared((block_n, head_dim), "bfloat16")
            scores = T.alloc_shared((block_m, block_n), "float32")
            mask = T.alloc_shared((block_m, block_n), "float32")
            scores_compute = T.alloc_shared((block_m, block_n), "bfloat16")
            accumulator = T.alloc_shared((block_m, head_dim), "float32")
            normalized = T.alloc_shared((block_m, head_dim), "float32")
            row_max = T.alloc_shared((block_m, 1), "float32")
            current_max = T.alloc_shared((block_m, 1), "float32")
            previous_max = T.alloc_shared((block_m, 1), "float32")
            previous_scale = T.alloc_shared((block_m, 1), "float32")
            chunk_sum = T.alloc_shared((block_m, 1), "float32")
            row_sum = T.alloc_shared((block_m, 1), "float32")
            scale_work0 = T.alloc_shared((block_m, 1), "float32")
            scale_work1 = T.alloc_shared((block_m, 1), "float32")
            exp_coeff = T.alloc_shared((64, 32), "float32")
            score_work0 = T.alloc_shared((block_m, block_n), "float32")
            score_work1 = T.alloc_shared((block_m, block_n), "float32")

            T.ppl_copy(Q[bz:bz + 1, bx * block_m:(bx + 1) * block_m, by:by + 1, 0:head_dim],
                       q_input)
            T.ppl_copy(q_input, q_compute)
            T.ppl_fill(accumulator, T.float32(0))
            T.ppl_fill(row_sum, T.float32(0))
            T.ppl_fill(row_max, -T.infinity("float32"))

            for ko in T.serial(T.ceildiv(sequence, block_n)):
                T.ppl_copy(K[bz:bz + 1, ko * block_n:(ko + 1) * block_n, by:by + 1, 0:head_dim],
                           k_input)
                T.ppl_copy(V[bz:bz + 1, ko * block_n:(ko + 1) * block_n, by:by + 1, 0:head_dim],
                           v_input)
                T.ppl_copy(k_input, k_compute)
                T.ppl_copy(v_input, v_compute)

                T.ppl_gemm(q_compute, k_compute, scores, transpose_B=True, accumulate=False)
                T.ppl_mul_C(scores, scores, T.float32(scale))
                T.ppl_copy(Mask[bx * block_m, ko * block_n], mask)
                T.ppl_add(scores, scores, mask)
                T.ppl_copy(row_max, previous_max)
                T.ppl_reduce_max(scores, current_max, dim=1)
                T.ppl_max(row_max, previous_max, current_max)
                T.ppl_subtract(previous_scale, previous_max, row_max)
                T.ppl_exp(previous_scale, scale_work0, scale_work1, exp_coeff)

                T.ppl_subtract(scores, scores, row_max)
                T.ppl_exp(scores, score_work0, score_work1, exp_coeff)
                T.ppl_reduce_sum(scores, chunk_sum, dim=1)

                T.ppl_mul(row_sum, row_sum, previous_scale)
                T.ppl_add(row_sum, row_sum, chunk_sum)
                T.ppl_mul(accumulator, accumulator, previous_scale)
                T.ppl_copy(scores, scores_compute)
                T.ppl_gemm(scores_compute, v_compute, accumulator, accumulate=True)

            T.ppl_div(normalized, accumulator, row_sum)
            T.ppl_copy(normalized, Output[bz:bz + 1, bx * block_m:(bx + 1) * block_m, by:by + 1,
                                          0:head_dim])

    return flashattn_fp32


def _reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor,
               dtype: str) -> torch.Tensor:
    if dtype == "float32":
        q_compute = q.to(torch.bfloat16).float()
        k_compute = k.to(torch.bfloat16).float()
        v_compute = v.to(torch.bfloat16).float()
    else:
        q_compute, k_compute, v_compute = q.float(), k.float(), v.float()
    scores = torch.einsum("bqhd,bkhd->bhqk", q_compute, k_compute)
    probabilities = torch.softmax(scores / math.sqrt(q.shape[-1]) + mask, dim=-1)
    output = torch.einsum("bhqk,bkhd->bqhd", probabilities, v_compute)
    return output.to(q.dtype)


def _validation_inputs(dtype: str, variant: str, seed: int):
    """Build deterministic inputs that expose distinct attention failure modes."""

    if variant not in ("balanced", "descending-max", "weighted-keys"):
        raise ValueError(f"unsupported FlashAttention validation variant: {variant!r}")
    batch, sequence, heads, head_dim = 1, 32, 1, 16
    generator = torch.Generator().manual_seed(seed)
    host_dtype = torch_dtype(dtype)
    tensor_shape = (batch, sequence, heads, head_dim)
    if variant == "descending-max":
        # The second K tile has a dramatically lower row maximum. The former
        # broken merge evaluated exp((m_old-m_new)/sqrt(d)) near exp(800).
        q = torch.full(tensor_shape, 10.0, dtype=host_dtype)
        first_k = torch.full((batch, 16, heads, head_dim), 10.0, dtype=host_dtype)
        second_k = torch.full((batch, 16, heads, head_dim), -10.0, dtype=host_dtype)
        k = torch.cat((first_k, second_k), dim=1)
        v = (torch.randn(tensor_shape, generator=generator) * 0.5).to(host_dtype)
    elif variant == "weighted-keys":
        # Monotonic key logits and key-dependent values make uniform attention,
        # within-tile weight loss, and argmax-only approximations observably wrong.
        query_scale = torch.linspace(0.75, 1.5, sequence).reshape(1, sequence, 1, 1)
        key_scale = torch.linspace(-1.0, 1.0, sequence).reshape(1, sequence, 1, 1)
        channel_scale = torch.linspace(-0.5, 0.5, head_dim).reshape(1, 1, 1, head_dim)
        q = (0.5 * query_scale).expand(tensor_shape).to(host_dtype).contiguous()
        k = (0.5 * key_scale).expand(tensor_shape).to(host_dtype).contiguous()
        v = (0.5 * key_scale +
             0.25 * channel_scale).expand(tensor_shape).to(host_dtype).contiguous()
    else:
        # Bounded random logits cover both K tiles. Distinct positive V tiles
        # expose a dropped numerator tile, stale denominator, and no-op output.
        q = (torch.randn(tensor_shape, generator=generator) * 0.25).to(host_dtype)
        k = (torch.randn(tensor_shape, generator=generator) * 0.25).to(host_dtype)
        first_v = torch.full((batch, 16, heads, head_dim), 0.25, dtype=host_dtype)
        second_v = torch.full((batch, 16, heads, head_dim), 0.75, dtype=host_dtype)
        v = torch.cat((first_v, second_v), dim=1)
    return q, k, v


def _attention_mask(sequence: int, is_causal: bool) -> torch.Tensor:
    if not isinstance(is_causal, bool):
        raise TypeError(f"flashattn requires boolean is_causal, got {is_causal!r}")
    if is_causal:
        return torch.triu(
            torch.full((sequence, sequence), float("-inf"), dtype=torch.float32), diagonal=1)
    return torch.zeros((sequence, sequence), dtype=torch.float32)


def run(*,
        dtype: str,
        chip: str,
        programming_model: str,
        runtime_mode: str,
        variant: str = "balanced",
        is_causal: bool = False,
        allow_pcie: bool = False,
        device_id: Optional[int] = None,
        seed: int = 0) -> dict:
    torch_dtype(dtype)
    if not isinstance(is_causal, bool):
        raise TypeError(f"flashattn requires boolean is_causal, got {is_causal!r}")
    validate_selection(
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode,
        supports_rv=True,
        allow_pcie=allow_pcie,
        device_id=device_id)
    batch, sequence, heads, head_dim = 1, 32, 1, 16
    tensors = _validation_inputs(dtype, variant, seed)
    q, k, v = tensors
    mask = _attention_mask(sequence, is_causal)
    output = torch.zeros_like(q)
    timing = compile_and_launch(
        build_flashattn(
            batch=batch, heads=heads, sequence=sequence, head_dim=head_dim, dtype=dtype),
        (*tensors, mask, output),
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode)
    expected = _reference(*tensors, mask, dtype)
    atol, rtol = tolerance(dtype, "flashattn")
    metrics = comparison(output, expected, atol=atol, rtol=rtol)
    return result_payload(
        operation="flashattn",
        dtype=dtype,
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode,
        metrics=metrics,
        timing=timing,
        parameters={
            "batch": batch,
            "heads": heads,
            "sequence": sequence,
            "head_dim": head_dim,
            "block_m": 16,
            "block_n": 16,
            "is_causal": is_causal,
            "fp32_compute_dtype": ("bfloat16" if dtype == "float32" else None),
            "variant": variant,
            "seed": seed
        })
