# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""RV normalization/activation regression, through the public JIT adapter.

Pytest performs source-only checks. Execute ONE numerical case in a fresh
process with --case NAME --runtime cmodel|pcie. Run the whole CModel matrix
before PCIe. The caller must stop on any error/timeout and preserve the log.
"""
import argparse
import json
from pathlib import Path

import pytest
import torch
import tilelang
import tilelang.language as T

TARGET = "tpu -mcpu=sg2260e -tpu-programming-model=rv"
DTYPES = ("float16", "bfloat16", "float32")
CASES = tuple(f"{op}.{dtype}" for op in ("fill", "scalar", "rsqrt", "sum", "max")
              for dtype in DTYPES) + ("sum-wide.float32", "max-wide.float32", "exp.float32",
                                      "exp-extremes.float32", "sigmoid.float32", "rmsnorm.float32",
                                      "softmax.float32", "swiglu.float32") + tuple(
                                          f"{op}.{dtype}"
                                          for op in ("demo-rmsnorm", "demo-splitk", "demo-swiglu")
                                          for dtype in DTYPES)


def make_kernel(operation, dtype="float32", rows=65, width=33):
    if operation.startswith("demo-"):
        from tpu_demo.rmsnorm.rmsnorm import build_rmsnorm, build_rmsnorm_splitk
        from tpu_demo.swiglu.swiglu import build_swiglu
        return {
            "demo-rmsnorm": build_rmsnorm,
            "demo-splitk": build_rmsnorm_splitk,
            "demo-swiglu": build_swiglu
        }[operation](
            dtype=dtype)
    reduce = operation in ("sum", "max", "sum-wide", "max-wide")
    out_width = 1 if reduce else width

    @T.prim_func
    def kernel(A: T.Tensor((rows, width), dtype), O: T.Tensor((rows, out_width), dtype)):
        with T.Kernel(1, is_cpu=True):
            x = T.alloc_shared((rows, width), dtype)
            y = T.alloc_shared((rows, width), dtype)
            T.ppl_copy(A, x)
            if operation == "fill":
                T.ppl_fill(y, T.float32(-2.5))
                T.ppl_copy(y, O)
            elif operation == "scalar":
                T.ppl_mul_C(y, x, T.float32(0.5))
                T.ppl_add_C(y, y, T.float32(-1.25))
                T.ppl_mul_C(y, y, T.float32(-2.0))
                T.ppl_copy(y, O)
            elif operation == "rsqrt":
                T.ppl_rsqrt(y, x)
                T.ppl_copy(y, O)
            elif operation in ("sum", "max", "sum-wide", "max-wide"):
                r = T.alloc_shared((rows, 1), dtype)
                # The public max contract overwrites this positive sentinel.
                T.ppl_fill(r, T.float32(42.0))
                if operation in ("sum", "sum-wide"):
                    T.ppl_reduce_sum(x, r, dim=1)
                else:
                    T.ppl_reduce_max(x, r, dim=1)
                T.ppl_copy(r, O)
            elif operation in ("exp", "exp-extremes", "sigmoid", "swiglu", "softmax"):
                w0 = T.alloc_shared((rows, width), dtype)
                w1 = T.alloc_shared((rows, width), dtype)
                coeff = T.alloc_shared((64, 32), dtype)
                if operation == "softmax":
                    r = T.alloc_shared((rows, 1), dtype)
                    T.ppl_reduce_max(x, r, dim=1)
                    T.ppl_subtract(y, x, r)
                    T.ppl_exp(y, w0, w1, coeff)
                    T.ppl_reduce_sum(y, r, dim=1)
                    T.ppl_div(y, y, r)
                    T.ppl_copy(y, O)
                elif operation in ("sigmoid", "swiglu"):
                    T.ppl_mul_C(y, x, T.float32(-1.0))
                    T.ppl_exp(y, w0, w1, coeff)
                    T.ppl_add_C(y, y, T.float32(1.0))
                    T.ppl_fill(w0, T.float32(1.0))
                    T.ppl_div(y, w0, y)
                    if operation == "swiglu":
                        T.ppl_mul(y, y, x)
                        T.ppl_mul(y, y, x)
                    T.ppl_copy(y, O)
                else:
                    T.ppl_exp(x, w0, w1, coeff)
                    T.ppl_copy(x, O)
            elif operation == "rmsnorm":
                r = T.alloc_shared((rows, 1), dtype)
                T.ppl_mul(y, x, x)
                T.ppl_reduce_sum(y, r, dim=1)
                T.ppl_mul_C(r, r, T.float32(1.0 / width))
                T.ppl_add_C(r, r, T.float32(1e-5))
                T.ppl_rsqrt(r, r)
                T.ppl_mul(y, x, r)
                T.ppl_copy(y, O)

    return kernel


@pytest.mark.parametrize("case", CASES)
def test_extended_ops_lower_through_public_engine(case):
    op, dtype = case.split(".")
    artifact = tilelang.lower(make_kernel(op, dtype), target=TARGET)
    source = artifact.kernel_source
    assert "rvt_" in source
    assert "tpu_bdc_" not in source
    assert "rvt_sync_i" in source


@pytest.mark.parametrize("op", ("exp", "sigmoid"))
@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
def test_exp_family_rejects_unported_dtypes(op, dtype):
    with pytest.raises(ValueError, match="requires FP32"):
        tilelang.lower(make_kernel(op, dtype), target=TARGET)


@pytest.mark.parametrize("op", ("scalar", "rsqrt", "sum", "max", "exp", "sigmoid"))
def test_promoted_ops_retain_tpukernel_lowering(op):
    artifact = tilelang.lower(
        make_kernel(op, "float32", rows=4),
        target="tpu -mcpu=sg2260e -tpu-programming-model=tpukernel")
    assert "tpu_bdc_" in artifact.kernel_source
    assert "rvt_" not in artifact.kernel_source


def run_case(case, runtime, output_dir):
    op, dtype = case.split(".")
    if op.startswith("demo-"):
        return run_demo(case, runtime, output_dir)
    rows, width = 65, (65 if "wide" in op else 33)
    torch.manual_seed(17)
    x = (torch.arange(rows * width).reshape(rows, width) % 17 - 8).float() / 4
    if op == "rsqrt":
        x = x.abs() + 0.25
    elif op.startswith("max"):
        x = -x.abs() - 1
        x[0] = -torch.inf
    elif op == "exp-extremes":
        special = torch.tensor([
            float('nan'),
            float('inf'), -float('inf'), -104., -90., -87., -80., -20., 0., 1., 20., 80., 88., 89.
        ])
        x = special.repeat(
            (rows * width + len(special) - 1) // len(special))[:rows * width].reshape(rows, width)
    x = x.to(getattr(torch, dtype))
    xf = x.float()
    if op == "fill":
        expected = torch.full_like(xf, -2.5)
    elif op == "scalar":
        expected = ((xf * 0.5).to(x.dtype).float() - 1.25).to(x.dtype).float() * -2
    elif op == "rsqrt":
        expected = torch.rsqrt(xf)
    elif op.startswith("sum"):
        expected = xf.sum(1, keepdim=True)
    elif op.startswith("max"):
        expected = xf.max(1, keepdim=True).values
    elif op.startswith("exp"):
        expected = torch.exp(xf)
    elif op == "sigmoid":
        expected = torch.sigmoid(xf)
    elif op == "softmax":
        expected = torch.softmax(xf, dim=1)
    elif op == "swiglu":
        expected = xf * torch.sigmoid(xf) * xf
    elif op == "rmsnorm":
        expected = xf * torch.rsqrt((xf * xf).mean(1, keepdim=True) + 1e-5)
    else:
        raise ValueError(op)
    tilelang.disable_cache()
    compiled = tilelang.compile(
        make_kernel(op, dtype, rows, width), out_idx=[1], target=TARGET, runtime_mode=runtime)
    result = compiled(x)
    if isinstance(result, (tuple, list)):
        result = result[0]
    result = result.float()
    # Quarter-integer scalar/reduction inputs make these oracles exact even
    # under reduced-precision accumulation; transcendental tolerances differ.
    rtol, atol = ((2e-5, 2e-6) if dtype == "float32" else (3e-3, 2e-3) if dtype == "float16" else
                  (2e-2, 1e-2))
    if op in ("fill", "scalar") or op.startswith(("sum", "max")):
        rtol = atol = 0
    torch.testing.assert_close(
        result, expected.to(x.dtype).float(), rtol=rtol, atol=atol, equal_nan=True)
    # Explicitly check non-finite classes; atol alone cannot validate these.
    assert torch.equal(torch.isnan(result), torch.isnan(expected))
    assert torch.equal(torch.isposinf(result), torch.isposinf(expected))
    assert torch.equal(torch.isneginf(result), torch.isneginf(expected))
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"input": x, "output": result, "expected": expected}, output_dir / f"{case}.pt")
        (output_dir / f"{case}.c").write_text(compiled.get_kernel_source())
    finite = torch.isfinite(result) & torch.isfinite(expected)
    error = (result[finite] - expected[finite]).abs().max().item() if finite.any() else 0.
    print(
        "RV_ESSENTIAL_RESULT=" + json.dumps({
            "case": case,
            "runtime": runtime,
            "shape": [rows, width],
            "status": "passed",
            "max_abs_error": error,
            "rtol": rtol,
            "atol": atol,
            "subnormal_relative_accuracy": "not_qualified"
        }),
        flush=True)


def run_demo(case, runtime, output_dir):
    op, dtype = case.split(".")
    shape = (8, 32 if op == "demo-swiglu" else 128 if op == "demo-splitk" else 64)
    torch.manual_seed(31)
    x = torch.randn(shape).clamp(-3, 3).to(getattr(torch, dtype))
    args = [x]
    if op == "demo-swiglu":
        up = torch.randn(shape).to(x.dtype)
        args.append(up)
        expected = x.float() * torch.sigmoid(x.float()) * up.float()
    else:
        expected = x.float() * torch.rsqrt(x.float().square().mean(1, keepdim=True) + 1e-12)
    tilelang.disable_cache()
    compiled = tilelang.compile(
        make_kernel(op, dtype), out_idx=[len(args)], target=TARGET, runtime_mode=runtime)
    result = compiled(*args)
    if isinstance(result, (tuple, list)):
        result = result[0]
    rtol, atol = ((2e-5, 2e-6) if dtype == "float32" else (3e-3, 2e-3) if dtype == "float16" else
                  (2e-2, 1e-2))
    expected = expected.to(x.dtype)
    torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "inputs": args,
            "output": result,
            "expected": expected
        }, output_dir / f"{case}.pt")
        (output_dir / f"{case}.c").write_text(compiled.get_kernel_source())
    print(
        "RV_ESSENTIAL_RESULT=" + json.dumps({
            "case": case,
            "runtime": runtime,
            "shape": shape,
            "status": "passed",
            "rtol": rtol,
            "atol": atol,
            "max_abs_error": (result.float() - expected.float()).abs().max().item()
        }),
        flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--runtime", choices=("cmodel", "pcie"), required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_case(args.case, args.runtime, args.output_dir)
