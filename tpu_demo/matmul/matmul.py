# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Tiled matrix multiplication for TPU-Kernel and RV Tensor."""

from typing import Optional

import tilelang.language as T
import torch

from tpu_demo.common import (comparison, compile_and_launch, result_payload, tolerance, torch_dtype,
                             validate_dimensions, validate_exact_tiling, validate_selection)


def build_matmul(*,
                 m: int = 32,
                 n: int = 32,
                 k: int = 32,
                 block_m: int = 16,
                 block_n: int = 16,
                 block_k: int = 16,
                 dtype: str = "float16",
                 programming_model: str = "tpukernel"):
    if programming_model not in ("tpukernel", "rv"):
        raise ValueError(f"unsupported TPU programming model: {programming_model!r}")
    torch_dtype(dtype)
    validate_dimensions("matmul", m=m, n=n, k=k, block_m=block_m, block_n=block_n, block_k=block_k)
    validate_exact_tiling("matmul", ("m", m, block_m), ("n", n, block_n), ("k", k, block_k))

    if dtype != "float32":

        @T.prim_func
        def matmul_low_precision(A: T.Tensor((m, k), dtype), B: T.Tensor((k, n), dtype),
                                 C: T.Tensor((m, n), dtype)):
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), is_cpu=True) as (bx, by):
                A_compute = T.alloc_shared((block_m, block_k), dtype)
                B_compute = T.alloc_shared((block_k, block_n), dtype)
                C_acc = T.alloc_shared((block_m, block_n), "float32")
                C_output = T.alloc_shared((block_m, block_n), dtype)
                T.ppl_fill(C_acc, T.float32(0))
                for ko in T.serial(T.ceildiv(k, block_k)):
                    T.ppl_copy(A[by * block_m, ko * block_k], A_compute)
                    T.ppl_copy(B[ko * block_k, bx * block_n], B_compute)
                    T.ppl_gemm(A_compute, B_compute, C_acc, accumulate=True)
                T.ppl_copy(C_acc, C_output)
                T.ppl_copy(C_output, C[by * block_m, bx * block_n])

        return matmul_low_precision

    @T.prim_func
    def matmul_fp32(A: T.Tensor((m, k), "float32"), B: T.Tensor((k, n), "float32"), C: T.Tensor(
        (m, n), "float32")):
        with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), is_cpu=True) as (bx, by):
            A_compute = T.alloc_shared((block_m, block_k), "float32", scope="local.matrix")
            B_compute = T.alloc_shared((block_k, block_n), "float32", scope="local.matrix")
            C_acc = T.alloc_shared((block_m, block_n), "float32", scope="local.matrix")
            steps = T.ceildiv(k, block_k)
            T.ppl_copy(A[by * block_m, 0], A_compute)
            T.ppl_copy(B[0, bx * block_n], B_compute)
            T.ppl_gemm(A_compute, B_compute, C_acc, accumulate=False)
            for ko in T.serial(steps - 1):
                next_ko = ko + 1
                T.ppl_copy(A[by * block_m, next_ko * block_k], A_compute)
                T.ppl_copy(B[next_ko * block_k, bx * block_n], B_compute)
                T.ppl_gemm(A_compute, B_compute, C_acc, accumulate=True)
            T.ppl_copy(C_acc, C[by * block_m, bx * block_n])

    return matmul_fp32


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
    shape = (32, 32, 32)
    generator = torch.Generator().manual_seed(seed)
    host_dtype = torch_dtype(dtype)
    # Bounded inputs exercise accumulation without making reduced-precision
    # TPU-Kernel conversion dominate the comparison tolerance.
    a = (torch.randn((shape[0], shape[2]), generator=generator) * 0.25).to(host_dtype)
    b = (torch.randn((shape[2], shape[1]), generator=generator) * 0.25).to(host_dtype)
    dst = torch.zeros((shape[0], shape[1]), dtype=host_dtype)
    timing = compile_and_launch(
        build_matmul(
            m=shape[0], n=shape[1], k=shape[2], dtype=dtype, programming_model=programming_model),
        (a, b, dst),
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode,
    )
    expected = torch.matmul(a.float(), b.float()).to(host_dtype)
    tolerance_family = "matmul-native-fp32" if dtype == "float32" else "matmul"
    atol, rtol = tolerance(dtype, tolerance_family)
    metrics = comparison(dst, expected, atol=atol, rtol=rtol)
    return result_payload(
        operation="matmul",
        dtype=dtype,
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode,
        metrics=metrics,
        timing=timing,
        parameters={
            "m": shape[0],
            "n": shape[1],
            "k": shape[2],
            "block": 16,
            "fp32_compute_dtype": "float32" if dtype == "float32" else None,
            "seed": seed
        })
