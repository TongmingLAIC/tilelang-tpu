# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Interleaved rotary-position embedding for TPU-Kernel and RV Tensor."""

from typing import Optional, Tuple

import tilelang.language as T
import torch

from tpu_demo.common import (comparison, compile_and_launch, result_payload, tolerance, torch_dtype,
                             validate_dimensions, validate_exact_tiling, validate_selection)


def build_rope(*,
               rows: int = 8,
               width: int = 32,
               block_rows: int = 4,
               block_width: int = 32,
               dtype: str = "float16"):
    torch_dtype(dtype)
    validate_dimensions(
        "rope", rows=rows, width=width, block_rows=block_rows, block_width=block_width)
    validate_exact_tiling("rope", ("rows", rows, block_rows), ("width", width, block_width))
    if width % 2 or block_width % 2:
        raise ValueError("RoPE width and block width must both be even")

    @T.prim_func
    def rope(source: T.Tensor((rows, width), dtype), cosine: T.Tensor(
        (rows, width // 2), "float32"), sine: T.Tensor((rows, width // 2), "float32"),
             destination: T.Tensor((rows, width), dtype)):
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
            T.ppl_copy(source[bx * block_rows, by * block_width], x)
            T.ppl_copy(cosine[bx * block_rows, by * (block_width // 2)], cos)
            T.ppl_copy(sine[bx * block_rows, by * (block_width // 2)], sin)
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
            T.ppl_copy(output, destination[bx * block_rows, by * block_width])

    return rope


def _cosine_sine(rows: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(rows, dtype=torch.float32).unsqueeze(1)
    frequencies = 10000.0**(-2 * torch.arange(width // 2, dtype=torch.float32) / width)
    theta = positions * frequencies.unsqueeze(0)
    return torch.cos(theta).contiguous(), torch.sin(theta).contiguous()


def _reference(source: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
    source_f32, cosine_f32, sine_f32 = source.float(), cosine.float(), sine.float()
    output = torch.empty_like(source_f32)
    output[:, 0::2] = (source_f32[:, 0::2] * cosine_f32 - source_f32[:, 1::2] * sine_f32)
    output[:, 1::2] = (source_f32[:, 1::2] * cosine_f32 + source_f32[:, 0::2] * sine_f32)
    return output.to(source.dtype)


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
        device_id=device_id)
    rows, width = 8, 32
    generator = torch.Generator().manual_seed(seed)
    host_dtype = torch_dtype(dtype)
    source = torch.randn((rows, width), generator=generator).to(host_dtype)
    cosine, sine = _cosine_sine(rows, width)
    destination = torch.zeros_like(source)
    timing = compile_and_launch(
        build_rope(rows=rows, width=width, dtype=dtype), (source, cosine, sine, destination),
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode)
    expected = _reference(source, cosine, sine)
    atol, rtol = tolerance(dtype, "rope")
    metrics = comparison(destination, expected, atol=atol, rtol=rtol)
    return result_payload(
        operation="rope",
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
            "block_width": 32,
            "seed": seed
        })
