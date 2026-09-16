# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Elementwise add/subtract/multiply/divide for TPU-Kernel and RV Tensor."""

from typing import Optional

import tilelang.language as T
import torch

from tpu_demo.common import (comparison, compile_and_launch, result_payload, tolerance, torch_dtype,
                             validate_dimensions, validate_selection)

OPERATIONS = ("add", "sub", "mul", "div")


def build_elementwise(operation: str, *, rows: int = 4, width: int = 32, dtype: str = "float32"):
    if operation not in OPERATIONS:
        raise ValueError(f"unsupported elementwise operation: {operation!r}")
    torch_dtype(dtype)
    validate_dimensions("elementwise", rows=rows, width=width)
    shape = (rows, width)

    if operation == "add":

        @T.prim_func
        def elementwise_add(lhs: T.Tensor(shape, dtype), rhs: T.Tensor(shape, dtype),
                            dst: T.Tensor(shape, dtype)):
            with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
                lhs_local = T.alloc_shared(shape, dtype)
                rhs_local = T.alloc_shared(shape, dtype)
                dst_local = T.alloc_shared(shape, dtype)
                T.ppl_copy(lhs, lhs_local)
                T.ppl_copy(rhs, rhs_local)
                T.ppl_add(dst_local, lhs_local, rhs_local)
                T.ppl_copy(dst_local, dst)

        return elementwise_add

    if operation == "sub":

        @T.prim_func
        def elementwise_sub(lhs: T.Tensor(shape, dtype), rhs: T.Tensor(shape, dtype),
                            dst: T.Tensor(shape, dtype)):
            with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
                lhs_local = T.alloc_shared(shape, dtype)
                rhs_local = T.alloc_shared(shape, dtype)
                dst_local = T.alloc_shared(shape, dtype)
                T.ppl_copy(lhs, lhs_local)
                T.ppl_copy(rhs, rhs_local)
                T.ppl_subtract(dst_local, lhs_local, rhs_local)
                T.ppl_copy(dst_local, dst)

        return elementwise_sub

    if operation == "mul":

        @T.prim_func
        def elementwise_mul(lhs: T.Tensor(shape, dtype), rhs: T.Tensor(shape, dtype),
                            dst: T.Tensor(shape, dtype)):
            with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
                lhs_local = T.alloc_shared(shape, dtype)
                rhs_local = T.alloc_shared(shape, dtype)
                dst_local = T.alloc_shared(shape, dtype)
                T.ppl_copy(lhs, lhs_local)
                T.ppl_copy(rhs, rhs_local)
                T.ppl_mul(dst_local, lhs_local, rhs_local)
                T.ppl_copy(dst_local, dst)

        return elementwise_mul

    @T.prim_func
    def elementwise_div(lhs: T.Tensor(shape, dtype), rhs: T.Tensor(shape, dtype),
                        dst: T.Tensor(shape, dtype)):
        with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
            lhs_local = T.alloc_shared(shape, dtype)
            rhs_local = T.alloc_shared(shape, dtype)
            dst_local = T.alloc_shared(shape, dtype)
            T.ppl_copy(lhs, lhs_local)
            T.ppl_copy(rhs, rhs_local)
            T.ppl_div(dst_local, lhs_local, rhs_local)
            T.ppl_copy(dst_local, dst)

    return elementwise_div


def run(*,
        operation: str,
        dtype: str,
        chip: str,
        programming_model: str,
        runtime_mode: str,
        allow_pcie: bool = False,
        device_id: Optional[int] = None,
        seed: int = 0) -> dict:
    if operation not in OPERATIONS:
        raise ValueError(f"unsupported elementwise operation: {operation!r}")
    torch_dtype(dtype)
    validate_selection(
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode,
        supports_rv=True,
        allow_pcie=allow_pcie,
        device_id=device_id,
    )
    shape = (4, 32)
    generator = torch.Generator().manual_seed(seed)
    host_dtype = torch_dtype(dtype)
    lhs = torch.randn(shape, generator=generator, dtype=torch.float32).to(host_dtype)
    if operation == "div":
        rhs = (torch.rand(shape, generator=generator) * 1.5 + 0.5).to(host_dtype)
    else:
        rhs = torch.randn(shape, generator=generator, dtype=torch.float32).to(host_dtype)
    dst = torch.zeros(shape, dtype=host_dtype)
    timing = compile_and_launch(
        build_elementwise(operation, dtype=dtype),
        (lhs, rhs, dst),
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode,
    )
    expected_f32 = {
        "add": torch.add,
        "sub": torch.sub,
        "mul": torch.mul,
        "div": torch.div,
    }[operation](lhs.float(), rhs.float())
    family = "elementwise-div" if operation == "div" else "elementwise"
    atol, rtol = tolerance(dtype, family)
    metrics = comparison(dst, expected_f32.to(host_dtype), atol=atol, rtol=rtol)
    return result_payload(
        operation=f"elementwise-{operation}",
        dtype=dtype,
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode,
        metrics=metrics,
        timing=timing,
        parameters={
            "shape": list(shape),
            "seed": seed
        })
