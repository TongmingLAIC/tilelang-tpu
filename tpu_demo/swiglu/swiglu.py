# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""SwiGLU using FP32 intermediates and portable TPU semantic operations."""

from typing import Optional

import tilelang.language as T
import torch

from tpu_demo.common import (comparison, compile_and_launch, result_payload, tolerance, torch_dtype,
                             validate_dimensions, validate_exact_tiling, validate_selection)


def build_swiglu(*,
                 rows: int = 8,
                 width: int = 32,
                 block_rows: int = 4,
                 block_width: int = 32,
                 dtype: str = "float16"):
    torch_dtype(dtype)
    validate_dimensions(
        "swiglu", rows=rows, width=width, block_rows=block_rows, block_width=block_width)
    validate_exact_tiling("swiglu", ("rows", rows, block_rows), ("width", width, block_width))

    if dtype != "float32":

        @T.prim_func
        def swiglu_low_precision(gate: T.Tensor((rows, width), dtype), up: T.Tensor(
            (rows, width), dtype), destination: T.Tensor((rows, width), dtype)):
            with T.Kernel(
                    T.ceildiv(rows, block_rows), T.ceildiv(width, block_width),
                    is_cpu=True) as (bx, by):
                shape = (block_rows, block_width)
                gate_input = T.alloc_shared(shape, dtype)
                up_input = T.alloc_shared(shape, dtype)
                output_local = T.alloc_shared(shape, dtype)
                gate_f32 = T.alloc_shared(shape, "float32")
                up_f32 = T.alloc_shared(shape, "float32")
                negative_gate = T.alloc_shared(shape, "float32")
                denominator = T.alloc_shared(shape, "float32")
                one = T.alloc_shared(shape, "float32")
                sigmoid = T.alloc_shared(shape, "float32")
                silu = T.alloc_shared(shape, "float32")
                output_f32 = T.alloc_shared(shape, "float32")
                work0 = T.alloc_shared(shape, "float32")
                work1 = T.alloc_shared(shape, "float32")
                coeff = T.alloc_shared((64, 32), "float32")
                T.ppl_copy(gate[bx * block_rows, by * block_width], gate_input)
                T.ppl_copy(up[bx * block_rows, by * block_width], up_input)
                T.ppl_copy(gate_input, gate_f32)
                T.ppl_copy(up_input, up_f32)
                T.ppl_mul_C(negative_gate, gate_f32, T.float32(-1.0))
                T.ppl_exp(negative_gate, work0, work1, coeff)
                T.ppl_add_C(denominator, negative_gate, T.float32(1.0))
                T.ppl_fill(one, T.float32(1.0))
                T.ppl_div(sigmoid, one, denominator)
                T.ppl_mul(silu, gate_f32, sigmoid)
                T.ppl_mul(output_f32, silu, up_f32)
                T.ppl_copy(output_f32, output_local)
                T.ppl_copy(output_local, destination[bx * block_rows, by * block_width])

        return swiglu_low_precision

    @T.prim_func
    def swiglu_fp32(gate: T.Tensor((rows, width), "float32"), up: T.Tensor(
        (rows, width), "float32"), destination: T.Tensor((rows, width), "float32")):
        with T.Kernel(
                T.ceildiv(rows, block_rows), T.ceildiv(width, block_width),
                is_cpu=True) as (bx, by):
            shape = (block_rows, block_width)
            gate_f32 = T.alloc_shared(shape, "float32")
            up_f32 = T.alloc_shared(shape, "float32")
            negative_gate = T.alloc_shared(shape, "float32")
            denominator = T.alloc_shared(shape, "float32")
            one = T.alloc_shared(shape, "float32")
            sigmoid = T.alloc_shared(shape, "float32")
            silu = T.alloc_shared(shape, "float32")
            output_f32 = T.alloc_shared(shape, "float32")
            work0 = T.alloc_shared(shape, "float32")
            work1 = T.alloc_shared(shape, "float32")
            coeff = T.alloc_shared((64, 32), "float32")
            T.ppl_copy(gate[bx * block_rows, by * block_width], gate_f32)
            T.ppl_copy(up[bx * block_rows, by * block_width], up_f32)
            T.ppl_mul_C(negative_gate, gate_f32, T.float32(-1.0))
            T.ppl_exp(negative_gate, work0, work1, coeff)
            T.ppl_add_C(denominator, negative_gate, T.float32(1.0))
            T.ppl_fill(one, T.float32(1.0))
            T.ppl_div(sigmoid, one, denominator)
            T.ppl_mul(silu, gate_f32, sigmoid)
            T.ppl_mul(output_f32, silu, up_f32)
            T.ppl_copy(output_f32, destination[bx * block_rows, by * block_width])

    return swiglu_fp32


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
    shape = (8, 32)
    generator = torch.Generator().manual_seed(seed)
    host_dtype = torch_dtype(dtype)
    gate = torch.clamp(torch.randn(shape, generator=generator), -3.0, 3.0).to(host_dtype)
    up = torch.randn(shape, generator=generator).to(host_dtype)
    destination = torch.zeros_like(gate)
    timing = compile_and_launch(
        build_swiglu(rows=shape[0], width=shape[1], dtype=dtype), (gate, up, destination),
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode)
    expected = (up.float() * torch.nn.functional.silu(gate.float())).to(host_dtype)
    atol, rtol = tolerance(dtype, "swiglu")
    metrics = comparison(destination, expected, atol=atol, rtol=rtol)
    return result_payload(
        operation="swiglu",
        dtype=dtype,
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode,
        metrics=metrics,
        timing=timing,
        parameters={
            "shape": list(shape),
            "block_rows": 4,
            "block_width": 32,
            "accumulation_dtype": "float32",
            "seed": seed
        })
