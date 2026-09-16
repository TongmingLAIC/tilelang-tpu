# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""RMSNorm and memory-bounded split-K RMSNorm for the portable TPU backends."""

from typing import Optional

import tilelang.language as T
import torch

from tpu_demo.common import (comparison, compile_and_launch, result_payload, tolerance, torch_dtype,
                             validate_dimensions, validate_exact_tiling, validate_positive_scalar,
                             validate_selection)


def build_rmsnorm(*,
                  rows: int = 8,
                  width: int = 64,
                  block_rows: int = 4,
                  dtype: str = "float16",
                  epsilon: float = 1e-12):
    torch_dtype(dtype)
    validate_positive_scalar("rmsnorm", "epsilon", epsilon)
    validate_dimensions("rmsnorm", rows=rows, width=width, block_rows=block_rows)
    validate_exact_tiling("rmsnorm", ("rows", rows, block_rows))

    if dtype != "float32":

        @T.prim_func
        def rmsnorm_low_precision(source: T.Tensor((rows, width), dtype), weight: T.Tensor(
            (rows, width), dtype), destination: T.Tensor((rows, width), dtype)):
            with T.Kernel(T.ceildiv(rows, block_rows), is_cpu=True) as (bx,):
                input_local = T.alloc_shared((block_rows, width), dtype)
                weight_local = T.alloc_shared((block_rows, width), dtype)
                output_local = T.alloc_shared((block_rows, width), dtype)
                value = T.alloc_shared((block_rows, width), "float32")
                square = T.alloc_shared((block_rows, width), "float32")
                sum_square = T.alloc_shared((block_rows, 1), "float32")
                variance = T.alloc_shared((block_rows, 1), "float32")
                inverse_rms = T.alloc_shared((block_rows, 1), "float32")
                normalized = T.alloc_shared((block_rows, width), "float32")
                T.ppl_copy(source[bx * block_rows, 0], input_local)
                T.ppl_copy(input_local, value)
                T.ppl_mul(square, value, value)
                T.ppl_reduce_sum(square, sum_square, dim=1)
                T.ppl_mul_C(variance, sum_square, T.float32(1.0 / width))
                T.ppl_add_C(variance, variance, T.float32(epsilon))
                T.ppl_rsqrt(inverse_rms, variance)
                T.ppl_mul(normalized, value, inverse_rms)
                T.ppl_copy(weight[bx * block_rows, 0], weight_local)
                T.ppl_copy(normalized, output_local)
                T.ppl_mul(output_local, output_local, weight_local)
                T.ppl_copy(output_local, destination[bx * block_rows, 0])

        return rmsnorm_low_precision

    @T.prim_func
    def rmsnorm_fp32(source: T.Tensor((rows, width), "float32"), weight: T.Tensor(
        (rows, width), "float32"), destination: T.Tensor((rows, width), "float32")):
        with T.Kernel(T.ceildiv(rows, block_rows), is_cpu=True) as (bx,):
            weight_local = T.alloc_shared((block_rows, width), "float32")
            value = T.alloc_shared((block_rows, width), "float32")
            square = T.alloc_shared((block_rows, width), "float32")
            sum_square = T.alloc_shared((block_rows, 1), "float32")
            variance = T.alloc_shared((block_rows, 1), "float32")
            inverse_rms = T.alloc_shared((block_rows, 1), "float32")
            normalized = T.alloc_shared((block_rows, width), "float32")
            T.ppl_copy(source[bx * block_rows, 0], value)
            T.ppl_mul(square, value, value)
            T.ppl_reduce_sum(square, sum_square, dim=1)
            T.ppl_mul_C(variance, sum_square, T.float32(1.0 / width))
            T.ppl_add_C(variance, variance, T.float32(epsilon))
            T.ppl_rsqrt(inverse_rms, variance)
            T.ppl_mul(normalized, value, inverse_rms)
            T.ppl_copy(weight[bx * block_rows, 0], weight_local)
            T.ppl_mul(normalized, normalized, weight_local)
            T.ppl_copy(normalized, destination[bx * block_rows, 0])

    return rmsnorm_fp32


def build_rmsnorm_splitk(*,
                         rows: int = 8,
                         width: int = 128,
                         block_rows: int = 4,
                         block_k: int = 32,
                         dtype: str = "float16",
                         epsilon: float = 1e-12):
    torch_dtype(dtype)
    validate_positive_scalar("rmsnorm-splitk", "epsilon", epsilon)
    validate_dimensions(
        "rmsnorm-splitk", rows=rows, width=width, block_rows=block_rows, block_k=block_k)
    validate_exact_tiling("rmsnorm-splitk", ("rows", rows, block_rows), ("width", width, block_k))

    if dtype != "float32":

        @T.prim_func
        def rmsnorm_splitk_low_precision(source: T.Tensor((rows, width), dtype), weight: T.Tensor(
            (rows, width), dtype), destination: T.Tensor((rows, width), dtype)):
            with T.Kernel(T.ceildiv(rows, block_rows), is_cpu=True) as (bx,):
                input_local = T.alloc_shared((block_rows, block_k), dtype)
                weight_local = T.alloc_shared((block_rows, block_k), dtype)
                output_local = T.alloc_shared((block_rows, block_k), dtype)
                value = T.alloc_shared((block_rows, block_k), "float32")
                square = T.alloc_shared((block_rows, block_k), "float32")
                chunk_sum = T.alloc_shared((block_rows, 1), "float32")
                sum_square = T.alloc_shared((block_rows, 1), "float32")
                variance = T.alloc_shared((block_rows, 1), "float32")
                inverse_rms = T.alloc_shared((block_rows, 1), "float32")
                normalized = T.alloc_shared((block_rows, block_k), "float32")
                T.ppl_fill(sum_square, T.float32(0))
                steps = T.ceildiv(width, block_k)
                # Serial order is part of the correctness contract until TPU
                # producer/consumer hazards are represented by a backend pass.
                for ko in T.serial(steps):
                    T.ppl_copy(source[bx * block_rows, ko * block_k], input_local)
                    T.ppl_copy(input_local, value)
                    T.ppl_mul(square, value, value)
                    T.ppl_reduce_sum(square, chunk_sum, dim=1)
                    T.ppl_add(sum_square, sum_square, chunk_sum)
                T.ppl_mul_C(variance, sum_square, T.float32(1.0 / width))
                T.ppl_add_C(variance, variance, T.float32(epsilon))
                T.ppl_rsqrt(inverse_rms, variance)
                for ko in T.serial(steps):
                    reverse_ko = steps - 1 - ko
                    T.ppl_copy(source[bx * block_rows, reverse_ko * block_k], input_local)
                    T.ppl_copy(input_local, value)
                    T.ppl_mul(normalized, value, inverse_rms)
                    T.ppl_copy(weight[bx * block_rows, reverse_ko * block_k], weight_local)
                    T.ppl_copy(normalized, output_local)
                    T.ppl_mul(output_local, output_local, weight_local)
                    T.ppl_copy(output_local, destination[bx * block_rows, reverse_ko * block_k])

        return rmsnorm_splitk_low_precision

    @T.prim_func
    def rmsnorm_splitk_fp32(source: T.Tensor((rows, width), "float32"), weight: T.Tensor(
        (rows, width), "float32"), destination: T.Tensor((rows, width), "float32")):
        with T.Kernel(T.ceildiv(rows, block_rows), is_cpu=True) as (bx,):
            weight_local = T.alloc_shared((block_rows, block_k), "float32")
            value = T.alloc_shared((block_rows, block_k), "float32")
            square = T.alloc_shared((block_rows, block_k), "float32")
            chunk_sum = T.alloc_shared((block_rows, 1), "float32")
            sum_square = T.alloc_shared((block_rows, 1), "float32")
            variance = T.alloc_shared((block_rows, 1), "float32")
            inverse_rms = T.alloc_shared((block_rows, 1), "float32")
            normalized = T.alloc_shared((block_rows, block_k), "float32")
            T.ppl_fill(sum_square, T.float32(0))
            steps = T.ceildiv(width, block_k)
            # Serial order is part of the correctness contract until TPU
            # producer/consumer hazards are represented by a backend pass.
            for ko in T.serial(steps):
                T.ppl_copy(source[bx * block_rows, ko * block_k], value)
                T.ppl_mul(square, value, value)
                T.ppl_reduce_sum(square, chunk_sum, dim=1)
                T.ppl_add(sum_square, sum_square, chunk_sum)
            T.ppl_mul_C(variance, sum_square, T.float32(1.0 / width))
            T.ppl_add_C(variance, variance, T.float32(epsilon))
            T.ppl_rsqrt(inverse_rms, variance)
            for ko in T.serial(steps):
                reverse_ko = steps - 1 - ko
                T.ppl_copy(source[bx * block_rows, reverse_ko * block_k], value)
                T.ppl_mul(normalized, value, inverse_rms)
                T.ppl_copy(weight[bx * block_rows, reverse_ko * block_k], weight_local)
                T.ppl_mul(normalized, normalized, weight_local)
                T.ppl_copy(normalized, destination[bx * block_rows, reverse_ko * block_k])

    return rmsnorm_splitk_fp32


def run(*,
        dtype: str,
        chip: str,
        programming_model: str,
        runtime_mode: str,
        split_k: bool = False,
        allow_pcie: bool = False,
        device_id: Optional[int] = None,
        seed: int = 0) -> dict:
    torch_dtype(dtype)
    if not isinstance(split_k, bool):
        raise ValueError(f"split_k must be bool, got {type(split_k).__name__}")
    validate_selection(
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode,
        supports_rv=True,
        allow_pcie=allow_pcie,
        device_id=device_id,
    )
    rows, width = (8, 128) if split_k else (8, 64)
    generator = torch.Generator().manual_seed(seed)
    host_dtype = torch_dtype(dtype)
    source = torch.randn((rows, width), generator=generator).to(host_dtype)
    weight = (torch.randn((rows, width), generator=generator) * 0.25 + 1.0).to(host_dtype)
    destination = torch.zeros_like(source)
    program = (
        build_rmsnorm_splitk(rows=rows, width=width, dtype=dtype) if split_k else build_rmsnorm(
            rows=rows, width=width, dtype=dtype))
    timing = compile_and_launch(
        program, (source, weight, destination),
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode)
    normalized = (
        source.float() *
        torch.rsqrt(torch.mean(source.float().square(), dim=1, keepdim=True) + 1e-12))
    # The low-precision kernel rounds the normalized value before applying
    # the low-precision weight.  Express that boundary explicitly because
    # PyTorch CPU does not implement arithmetic directly on FP8 tensors.
    expected = (normalized.to(host_dtype).float() * weight.float()).to(host_dtype)
    atol, rtol = tolerance(dtype, "rmsnorm")
    metrics = comparison(destination, expected, atol=atol, rtol=rtol)
    return result_payload(
        operation="rmsnorm-splitk" if split_k else "rmsnorm",
        dtype=dtype,
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode,
        metrics=metrics,
        timing=timing,
        parameters={
            "rows": rows,
            "width": width,
            "block_rows": 4,
            "block_k": 32 if split_k else None,
            "epsilon": 1e-12,
            "seed": seed
        })
