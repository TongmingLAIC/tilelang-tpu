# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""RoPE adapters splitting only the global row dimension into legal dim4 axes.

Local tiles, arithmetic and intermediate dtypes are copied from the original
and PPL-matched builders. Flat input/output bytes and tile visit order agree.
"""
import tilelang.language as T
from tpu_demo.common import torch_dtype, validate_dimensions, validate_exact_tiling


def build_split_global_rope(*,
                            rows: int = 8,
                            width: int = 32,
                            block_rows: int = 4,
                            block_width: int = 32,
                            dtype: str = "float16"):
    if rows % 32768:
        raise ValueError("large RoPE requires rows divisible by 32768")
    torch_dtype(dtype)
    validate_dimensions(
        "rope", rows=rows, width=width, block_rows=block_rows, block_width=block_width)
    validate_exact_tiling("rope", ("rows", rows, block_rows), ("width", width, block_width))
    if width % 2 or block_width % 2:
        raise ValueError("RoPE width and block width must both be even")

    @T.prim_func
    def rope(source: T.Tensor((rows // 32768, 32768, 1, width), dtype), cosine: T.Tensor(
        (rows // 32768, 32768, 1, width // 2), "float32"), sine: T.Tensor(
            (rows // 32768, 32768, 1, width // 2), "float32"), destination: T.Tensor(
                (rows // 32768, 32768, 1, width), dtype)):
        with T.Kernel(
                T.ceildiv(rows, block_rows), T.ceildiv(width, block_width),
                is_cpu=True) as (bx, by):
            shape = (block_rows, block_width)
            x = T.alloc_shared(shape, dtype)
            output = T.alloc_shared(shape, dtype)
            table_shape = (block_rows, block_width // 2)
            cos = T.alloc_shared(table_shape, "float32")
            sin = T.alloc_shared(table_shape, "float32")
            even_value = T.alloc_shared((block_rows, 1), dtype)
            odd_value = T.alloc_shared((block_rows, 1), dtype)
            even_f32 = T.alloc_shared((block_rows, 1), "float32")
            odd_f32 = T.alloc_shared((block_rows, 1), "float32")
            cos_f32 = T.alloc_shared((block_rows, 1), "float32")
            sin_f32 = T.alloc_shared((block_rows, 1), "float32")
            even_cos = T.alloc_shared((block_rows, 1), "float32")
            odd_sin = T.alloc_shared((block_rows, 1), "float32")
            odd_cos = T.alloc_shared((block_rows, 1), "float32")
            even_sin = T.alloc_shared((block_rows, 1), "float32")
            T.ppl_copy(
                source[bx * block_rows // 32768, bx * block_rows % 32768, 0, by * block_width], x)
            T.ppl_copy(
                cosine[bx * block_rows // 32768, bx * block_rows % 32768, 0,
                       by * (block_width // 2)], cos)
            T.ppl_copy(
                sine[bx * block_rows // 32768, bx * block_rows % 32768, 0, by * (block_width // 2)],
                sin)
            for pair in T.serial(block_width // 2):
                T.ppl_copy(x[0, 2 * pair], even_value)
                T.ppl_copy(x[0, 2 * pair + 1], odd_value)
                T.ppl_copy(even_value, even_f32)
                T.ppl_copy(odd_value, odd_f32)
                T.ppl_copy(cos[0, pair], cos_f32)
                T.ppl_copy(sin[0, pair], sin_f32)
                T.ppl_mul(even_cos, even_f32, cos_f32)
                T.ppl_mul(odd_sin, odd_f32, sin_f32)
                T.ppl_mul(odd_cos, odd_f32, cos_f32)
                T.ppl_mul(even_sin, even_f32, sin_f32)
                T.ppl_subtract(even_cos, even_cos, odd_sin)
                T.ppl_add(odd_cos, odd_cos, even_sin)
                T.ppl_copy(even_cos, even_value)
                T.ppl_copy(odd_cos, odd_value)
                T.ppl_copy(even_value, output[0, 2 * pair])
                T.ppl_copy(odd_value, output[0, 2 * pair + 1])
            T.ppl_copy(
                output, destination[bx * block_rows // 32768, bx * block_rows % 32768, 0,
                                    by * block_width])

    return rope


def build_split_global_matched_rope(rows, width, br, bw, dtype):

    if rows % 32768:
        raise ValueError("large RoPE requires rows divisible by 32768")

    @T.prim_func
    def kernel(x: T.Tensor((rows // 32768, 32768, 1, width), dtype), cos: T.Tensor(
        (rows // 32768, 32768, 1, width), dtype), sin: T.Tensor(
            (rows // 32768, 32768, 1, width), dtype), out: T.Tensor(
                (rows // 32768, 32768, 1, width), dtype)):
        with T.Kernel(rows // br, width // bw, is_cpu=True) as (r, w):
            xx = T.alloc_shared((br, bw), dtype)
            cc = T.alloc_shared((br, bw), dtype)
            ss = T.alloc_shared((br, bw), dtype)
            xc = T.alloc_shared((br, bw), dtype)
            xs = T.alloc_shared((br, bw), dtype)
            negative = T.alloc_shared((br, bw), dtype)
            ns = T.alloc_shared((br, bw), dtype)
            result = T.alloc_shared((br, bw), dtype)
            a = T.alloc_shared((br, 1), dtype)
            b = T.alloc_shared((br, 1), dtype)
            c = T.alloc_shared((br, 1), dtype)
            d = T.alloc_shared((br, 1), dtype)
            T.ppl_copy(x[r * br // 32768, r * br % 32768, 0, w * bw], xx)
            T.ppl_copy(cos[r * br // 32768, r * br % 32768, 0, w * bw], cc)
            T.ppl_copy(sin[r * br // 32768, r * br % 32768, 0, w * bw], ss)
            T.ppl_mul(xc, xx, cc)
            T.ppl_mul(xs, xx, ss)
            T.ppl_mul_C(negative, xx, -1.0)
            T.ppl_mul(ns, negative, ss)
            for j in T.serial(bw // 2):
                T.ppl_copy(xc[0, j * 2], a)
                T.ppl_copy(ns[0, j * 2 + 1], b)
                T.ppl_add(a, a, b)
                T.ppl_copy(xc[0, j * 2 + 1], c)
                T.ppl_copy(xs[0, j * 2], d)
                T.ppl_add(c, c, d)
                T.ppl_copy(a, result[0, j * 2])
                T.ppl_copy(c, result[0, j * 2 + 1])
            T.ppl_copy(result, out[r * br // 32768, r * br % 32768, 0, w * bw])

    return kernel
