# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Explicit two-buffer TPU-Kernel GEMM pipeline over the reduction dimension."""

import tilelang.language as T


def build_pipelined_matmul(*, m: int, n: int, k: int, block_k: int):
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in (m, n, k, block_k)):
        raise ValueError("matrix dimensions and reduction tile must be positive integers")
    if k % block_k or k // block_k < 2:
        raise ValueError("pipelined GEMM requires at least two full reduction tiles")
    if max(m, n, k) > 65535:
        raise ValueError("TPU-Kernel tensor descriptor dimensions must not exceed 65535")
    steps = k // block_k

    @T.prim_func
    def pipelined_matmul(A: T.Tensor((m, k), "float16"),
                         B: T.Tensor((k, n), "float16"),
                         C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, 1, is_cpu=True):
            a0 = T.alloc_shared((m, block_k), "float16")
            a1 = T.alloc_shared((m, block_k), "float16")
            b0 = T.alloc_shared((block_k, n), "float16")
            b1 = T.alloc_shared((block_k, n), "float16")
            accumulator = T.alloc_shared((m, n), "float32")
            output = T.alloc_shared((m, n), "float16")
            T.ppl_fill(accumulator, T.float32(0))
            T.ppl_copy(A[0, 0], a0)
            T.ppl_copy(B[0, 0], b0)
            for j in T.serial(steps - 1):
                T.tpukernel_parallel_start()
                if j % 2 == 0:
                    T.ppl_gemm(a0, b0, accumulator, accumulate=True)
                    T.ppl_copy(A[0, (j + 1) * block_k], a1)
                    T.ppl_copy(B[(j + 1) * block_k, 0], b1)
                else:
                    T.ppl_gemm(a1, b1, accumulator, accumulate=True)
                    T.ppl_copy(A[0, (j + 1) * block_k], a0)
                    T.ppl_copy(B[(j + 1) * block_k, 0], b0)
                T.tpukernel_parallel_end()
            if (steps - 1) % 2 == 0:
                T.ppl_gemm(a0, b0, accumulator, accumulate=True)
            else:
                T.ppl_gemm(a1, b1, accumulator, accumulate=True)
            T.ppl_copy(accumulator, output)
            T.ppl_copy(output, C)

    return pipelined_matmul
