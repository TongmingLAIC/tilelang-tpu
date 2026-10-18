# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Comparable serial and manually pipelined TPU-Kernel tiled scalar-add kernels.

The pipelined form follows the vendor's load/compute/store command-group
pattern.  It is intentionally explicit so that address allocation sees the
four physical buffers and the result can be checked independently before an
automatic TPU software-pipeline pass is considered.
"""

import tilelang.language as T


def build_pipeline_add(*, tiles: int, tile_width: int, dtype: str = "float16",
                       pipelined: bool = False):
    if isinstance(tiles, bool) or not isinstance(tiles, int) or tiles < 4:
        raise ValueError("pipeline_add requires at least four static tiles")
    if isinstance(tile_width, bool) or not isinstance(tile_width, int) or tile_width <= 0:
        raise ValueError("pipeline_add requires a positive static tile_width")
    if dtype not in ("float16", "bfloat16", "float32"):
        raise ValueError("pipeline_add supports float16, bfloat16, and float32")
    width = tiles * tile_width

    if not pipelined:

        @T.prim_func
        def serial_add(A: T.Tensor((1, width), dtype), C: T.Tensor((1, width), dtype)):
            with T.Kernel(1, 1, is_cpu=True):
                a = T.alloc_shared((1, tile_width), dtype)
                r = T.alloc_shared((1, tile_width), dtype)
                for j in T.serial(tiles):
                    T.ppl_copy(A[0, j * tile_width], a)
                    T.ppl_add_C(r, a, T.float32(0.25))
                    T.ppl_copy(r, C[0, j * tile_width])

        return serial_add

    @T.prim_func
    def pipelined_add(A: T.Tensor((1, width), dtype), C: T.Tensor((1, width), dtype)):
        with T.Kernel(1, 1, is_cpu=True):
            a0 = T.alloc_shared((1, tile_width), dtype)
            a1 = T.alloc_shared((1, tile_width), dtype)
            r0 = T.alloc_shared((1, tile_width), dtype)
            r1 = T.alloc_shared((1, tile_width), dtype)

            T.ppl_copy(A[0, 0], a0)
            T.tpukernel_parallel_start()
            T.ppl_add_C(r0, a0, T.float32(0.25))
            T.ppl_copy(A[0, tile_width], a1)
            T.tpukernel_parallel_end()

            for j in T.serial(tiles - 2):
                T.tpukernel_parallel_start()
                if j % 2 == 0:
                    T.ppl_copy(r0, C[0, j * tile_width])
                    T.ppl_add_C(r1, a1, T.float32(0.25))
                    T.ppl_copy(A[0, (j + 2) * tile_width], a0)
                else:
                    T.ppl_copy(r1, C[0, j * tile_width])
                    T.ppl_add_C(r0, a0, T.float32(0.25))
                    T.ppl_copy(A[0, (j + 2) * tile_width], a1)
                T.tpukernel_parallel_end()

            T.tpukernel_parallel_start()
            if (tiles - 2) % 2 == 0:
                T.ppl_copy(r0, C[0, (tiles - 2) * tile_width])
                T.ppl_add_C(r1, a1, T.float32(0.25))
            else:
                T.ppl_copy(r1, C[0, (tiles - 2) * tile_width])
                T.ppl_add_C(r0, a0, T.float32(0.25))
            T.tpukernel_parallel_end()
            if (tiles - 2) % 2 == 0:
                T.ppl_copy(r1, C[0, (tiles - 1) * tile_width])
            else:
                T.ppl_copy(r0, C[0, (tiles - 1) * tile_width])

    return pipelined_add
