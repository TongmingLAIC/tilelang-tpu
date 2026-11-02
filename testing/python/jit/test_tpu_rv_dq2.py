# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""RV DQ2 fallback: pytest is compile-only; --case launches CModel only.

Each numerical case must run in a fresh process under
python -m tilelang.jit.tpu_cmodel --output-dir /absolute/new/result -- python ... . No PCIe entry point is
provided: UINT8 transfers and this new RV sequence are not board-validated.
"""
import argparse
import json

import pytest
import torch
import tilelang
import tilelang.language as T

TARGET = "tpu -mcpu=bm1690e -tpu-programming-model=rv"
PACKINGS = ("uint8", "uint16", "uint32")
DTYPES = ("float16", "bfloat16")
# Byte exhaustion, lane boundary, non-word-aligned groups, supported vendor
# group sizes, and long rows. Every packing/output combination is exercised.
SHAPES = ((1, 512, 256), (63, 32, 32), (64, 64, 64), (65, 256, 128), (129, 24, 3), (2, 8, 1),
          (1, 4096, 256), (3, 32, 4))
CASES = [(p, d, *s) for p in PACKINGS for d in DTYPES for s in SHAPES]


def make_kernel(packing="uint32", dtype="float16", rows=65, width=256, group=128):
    words = width // ({"uint8": 2, "uint16": 4, "uint32": 8}[packing])

    @T.prim_func
    def dequant(P: T.Tensor((rows, words), packing), Q: T.Tensor((rows, width // group), "uint32"),
                O: T.Tensor((rows, width), dtype)):
        with T.Kernel(1, is_cpu=True):
            packed = T.alloc_shared((rows, words), packing)
            quant = T.alloc_shared((rows, width // group), "uint32")
            result = T.alloc_shared((rows, width), dtype)
            T.ppl_copy(P, packed)
            T.ppl_copy(Q, quant)
            # Repeated operations exercise scratch lifetime/reuse and ensure
            # that source words and quantization parameters are not clobbered.
            T.ppl_dq2(result, packed, quant, group)
            T.ppl_dq2(result, packed, quant, group)
            T.ppl_copy(result, O)

    return dequant


@pytest.mark.parametrize("packing", PACKINGS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rv_dq2_source(packing, dtype):
    source = tilelang.lower(
        make_kernel(packing, dtype), target=TARGET, runtime_mode="cmodel").kernel_source
    assert "rvt_dq2(" not in source
    assert "tpu_bdc_f16_group_dequant(" not in source
    for instruction in ("rvt_and(", "rvt_lshr(", "rvt_cvt_i2f(", "rvt_fsub(", "rvt_fmul(",
                        "rvt_cfg_round_mode(2)"):
        assert instruction in source
    assert "SIGN(DT_INT8)" in source
    assert "dq_col * 2" in source


@pytest.mark.parametrize("packing", ("uint16", "uint32"))
def test_tpukernel_rejects_wide_packing(packing):
    with pytest.raises(Exception, match="TPU-Kernel requires UINT8 packing"):
        tilelang.lower(
            make_kernel(packing),
            target="tpu -mcpu=bm1690e -tpu-programming-model=tpukernel",
            runtime_mode="cmodel")


@pytest.mark.parametrize("width,group", ((30, 3), (32, 3)))
def test_invalid_shape_rejected(width, group):
    decl = tilelang.tvm.tir.decl_buffer
    out = decl((1, width), "float16", scope="shared")
    src = decl((1, width // 8), "uint32", scope="shared")
    quant = decl((1, width // group), "uint32", scope="shared")
    with pytest.raises(ValueError, match="packed width|gsize to divide"):
        T.ppl_dq2(out, src, quant, group)


@pytest.mark.parametrize("scratch_dtype,scratch_width,access",
                         (("float16", 16, "rw"), ("float32", 8, "rw"), ("float32", 16, "r")))
def test_native_rejects_invalid_scratch(scratch_dtype, scratch_width, access):
    from tilelang.language.copy import buffer_to_tile_region

    @T.prim_func
    def invalid():
        with T.Kernel(1, is_cpu=True):
            out = T.alloc_shared((2, 32), "float16")
            src = T.alloc_shared((2, 4), "uint32")
            quant = T.alloc_shared((2, 1), "uint32")
            scratch = T.alloc_shared((2, scratch_width), scratch_dtype)
            T.evaluate(
                T.call_extern("handle", "tl.tpu.dq2", buffer_to_tile_region(out, "w"),
                              buffer_to_tile_region(src, "r"), buffer_to_tile_region(quant, "r"),
                              32, buffer_to_tile_region(scratch, access)))

    with pytest.raises(Exception, match="scratch"):
        tilelang.lower(invalid, target=TARGET, runtime_mode="cmodel")


def test_gptq_default_packing_matches_builder_and_byte_oracle():
    from tpu_demo.gptq_w4a16.gptq_w4a16 import build_gptq_w4a16, host_inputs, torch_reference

    parameters = dict(m=32, n=64, k=256, group_size=128, dtype="float16", seed=0)
    left, packed, quant = host_inputs(**parameters)
    program = build_gptq_w4a16()
    assert packed.dtype == torch.uint32
    assert str(program.buffer_map[program.params[1]].dtype) == "uint32"
    assert tuple(program.buffer_map[program.params[1]].shape) == packed.shape
    left_bytes, byte_packed, quant_bytes = host_inputs(**parameters, packed_dtype="uint8")
    reference_args = dict(n=64, k=256, group_size=128)
    torch.testing.assert_close(
        torch_reference(left, packed, quant, **reference_args),
        torch_reference(left_bytes, byte_packed, quant_bytes, **reference_args),
        atol=0,
        rtol=0)


def numeric_case(case):
    packing, dtype, rows, width, group = CASES[case]
    words = width // (getattr(torch, packing).itemsize * 2)
    # Exhaust all byte patterns, including 0x1f,0x8f,0xff (right-shift rounding
    # traps). Vary rows to expose lane and stride errors; no device-side oracle.
    raw = (torch.arange(rows * width // 2, dtype=torch.int64) % 256).reshape(rows, width // 2)
    codes = torch.stack((raw & 15, raw >> 4), -1).reshape(rows, width).float()
    p = torch.zeros((rows, words), dtype=torch.int64)
    for i in range(getattr(torch, packing).itemsize):
        p |= raw[:, i::getattr(torch, packing).itemsize] << (8 * i)
    p = p.to(getattr(torch, packing))
    count = rows * (width // group)
    index = torch.arange(count).reshape(rows, -1)
    # Fractional/negative offsets and scales distinguish FP16 bit
    # reinterpretation from integer conversion and unsigned zero points.
    offset = ((index % 23 - 11).float() / 7).half()
    scale = ((index % 17 - 8).float() / 13).half()
    if (rows, width, group) == (3, 32, 4):
        offset[0, :4] = torch.tensor([float("nan"), float("inf"), -float("inf"), -65504])
        scale[0, :4] = torch.tensor([1, 0, -1, 65504])
        scale[1, :4] = torch.tensor([float("nan"), float("inf"), -float("inf"), 0])
    quant = (offset.view(torch.int16).to(torch.int64) & 65535) | (
        (scale.view(torch.int16).to(torch.int64) & 65535) << 16)
    quant = quant.to(torch.uint32)
    expected = ((codes - offset.float().repeat_interleave(group, 1)) *
                scale.float().repeat_interleave(group, 1)).to(getattr(torch, dtype))
    kernel = tilelang.compile(
        make_kernel(packing, dtype, rows, width, group),
        out_idx=[2],
        target=TARGET,
        runtime_mode="cmodel")
    actual = kernel(p, quant)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    return {
        "status": "passed",
        "case": case,
        "packing": packing,
        "dtype": dtype,
        "rows": rows,
        "width": width,
        "group": group,
        "element_count": expected.numel(),
        "max_abs_error": 0
    }


def gptq_case(case):
    from tpu_demo.gptq_w4a16.gptq_w4a16 import build_gptq_w4a16, host_inputs, torch_reference
    from tpu_demo.common import tolerance

    packing, dtype, tiled = [(p, d, t) for p in PACKINGS for d in DTYPES for t in (False, True)
                            ][case]
    m, n, k, group = (64, 128, 256, 128) if tiled else (32, 64, 256, 128)
    left, packed, quant = host_inputs(
        m=m, n=n, k=k, group_size=group, dtype=dtype, seed=19, packed_dtype=packing)
    program = build_gptq_w4a16(
        m=m, n=n, k=k, group_size=group, dtype=dtype, packed_dtype=packing, programming_model="rv")
    kernel = tilelang.compile(program, out_idx=[3], target=TARGET, runtime_mode="cmodel")
    actual = kernel(left, packed, quant)
    expected = torch_reference(left, packed, quant, n=n, k=k, group_size=group).to(left.dtype)
    atol, rtol = tolerance(dtype, "matmul")
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return {
        "status": "passed",
        "gptq_case": case,
        "packing": packing,
        "dtype": dtype,
        "m": m,
        "n": n,
        "k": k,
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": (actual.float() - expected.float()).abs().max().item()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--case", type=int, choices=range(len(CASES)))
    mode.add_argument("--gptq-case", type=int, choices=range(12))
    args = parser.parse_args()
    result = numeric_case(args.case) if args.case is not None else gptq_case(args.gptq_case)
    print(json.dumps(result), flush=True)
