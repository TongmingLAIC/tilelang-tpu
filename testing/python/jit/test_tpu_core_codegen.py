# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Source-only checks for the portable TPU core-op instruction selectors.

These tests deliberately stop after ``tilelang.lower``.  They neither build a
runtime wrapper nor load or dispatch a kernel, so they are safe to run on a
host that also has a PCIe TPU attached.
"""

import os
from pathlib import Path

import pytest

import tilelang
from tilelang import tvm
import tilelang.language as T
from tilelang.engine.tpu_config import TPURuntimeConfig, TPUTargetSpec
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.wrapper import TLWrapper


@T.prim_func
def _portable_core_ops(A: T.Tensor((16, 16), "float16"), B: T.Tensor(
    (16, 16), "float16"), C: T.Tensor((16, 16), "float32"), X: T.Tensor((1, 32), "float32"),
                       Y: T.Tensor((1, 32), "float32"), Z: T.Tensor((1, 32), "float32")):
    """One frontend program shared by TPU-Kernel and RV Tensor codegen."""
    T.func_attr({"global_symbol": "portable_core_ops", "tir.noalias": T.bool(True)})
    with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
        a_shared = T.alloc_shared((16, 16), "float16")
        b_shared = T.alloc_shared((16, 16), "float16")
        acc_shared = T.alloc_shared((16, 16), "float32")
        x_shared = T.alloc_shared((1, 32), "float32")
        y_shared = T.alloc_shared((1, 32), "float32")
        z_shared = T.alloc_shared((1, 32), "float32")

        T.ppl_fill(acc_shared, T.float32(0))
        T.ppl_copy(A, a_shared)
        T.ppl_copy(B, b_shared)
        T.ppl_gemm(a_shared, b_shared, acc_shared, accumulate=True)
        T.ppl_copy(acc_shared, C)

        T.ppl_copy(X, x_shared)
        T.ppl_copy(Y, y_shared)
        T.ppl_add(z_shared, x_shared, y_shared)
        T.ppl_subtract(z_shared, x_shared, y_shared)
        T.ppl_mul(z_shared, x_shared, y_shared)
        T.ppl_div(z_shared, x_shared, y_shared)
        T.ppl_max(z_shared, x_shared, y_shared)
        T.ppl_copy(z_shared, Z)


@T.prim_func
def _portable_and_raw_rv(X: T.Tensor((1, 32), "float32"), Y: T.Tensor((1, 32), "float32")):
    T.func_attr({"global_symbol": "portable_and_raw_rv", "tir.noalias": T.bool(True)})
    with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
        x_shared = T.alloc_shared((1, 32), "float32")
        y_shared = T.alloc_shared((1, 32), "float32")
        z_shared = T.alloc_shared((1, 32), "float32")
        T.ppl_copy(X, x_shared)
        T.ppl_copy(Y, y_shared)
        T.ppl_add(z_shared, x_shared, y_shared)
        # Raw RV calls own their register descriptors and lifecycle.  Mixing
        # one into a compiler-managed portable kernel must fail closed.
        T.rvt_fadd(T.uint64(10), T.uint64(8), T.uint64(9))


@T.prim_func
def _reserved_entry_probe(A: T.Tensor((1,), "float32")):
    T.func_attr({"global_symbol": "reserved_entry_probe", "tir.noalias": T.bool(True)})
    T.evaluate(0)


@T.prim_func
def _global_to_global_copy(A: T.Tensor((1, 32), "float32"), B: T.Tensor((1, 32), "float32")):
    T.func_attr({"global_symbol": "global_to_global_copy", "tir.noalias": T.bool(True)})
    with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
        T.ppl_copy(A, B)


def _portable_copy_program(dtype, transfer):
    shape = (4, 32)

    if transfer == "global":

        @T.prim_func
        def copy_kernel(source: T.Tensor(shape, dtype), destination: T.Tensor(shape, dtype)):
            with T.Kernel(1, is_cpu=True):
                T.ppl_copy(source, destination)

        return copy_kernel

    @T.prim_func
    def copy_kernel(source: T.Tensor(shape, dtype), destination: T.Tensor(shape, dtype)):
        with T.Kernel(1, is_cpu=True):
            source_local = T.alloc_shared(shape, dtype)
            destination_local = T.alloc_shared(shape, dtype)
            T.ppl_copy(source, source_local)
            T.ppl_copy(source_local, destination_local)
            T.ppl_copy(destination_local, destination)

    return copy_kernel


@T.prim_func
def _local_fp16_to_bf16_copy():
    T.func_attr({"global_symbol": "local_fp16_to_bf16_copy"})
    with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
        src = T.alloc_shared((1, 32), "float16")
        dst = T.alloc_shared((1, 32), "bfloat16")
        T.ppl_copy(src, dst)


@T.prim_func
def _local_integer_convert():
    T.func_attr({"global_symbol": "local_integer_convert"})
    with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
        src = T.alloc_shared((1, 32), "int8")
        dst = T.alloc_shared((1, 32), "int16")
        T.ppl_copy(src, dst)


@T.prim_func
def _overwrite_fp16_gemm():
    T.func_attr({"global_symbol": "overwrite_fp16_gemm"})
    with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
        lhs = T.alloc_shared((16, 16), "float16")
        rhs = T.alloc_shared((16, 16), "float16")
        out = T.alloc_shared((16, 16), "float16")
        T.ppl_gemm(lhs, rhs, out, accumulate=False)


@T.prim_func
def _tpukernel_topk(Input: T.Tensor((257,), "float32"), Output: T.Tensor((11,), "float32"),
                    Indices: T.Tensor((11,), "int32")):
    T.func_attr({"global_symbol": "tpukernel_topk", "tir.noalias": T.bool(True)})
    with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
        T.ppl_topk(Output, Indices, Input, 11, True, 257)


def _tpu_target(chip, programming_model):
    return (f"tpu -mcpu={chip} "
            f"-tpu-programming-model={programming_model}")


def _lower_core_ops(chip, programming_model):
    return tilelang.lower(
        _portable_core_ops,
        target=_tpu_target(chip, programming_model),
        runtime_mode="cmodel",
    ).kernel_source


@pytest.fixture(scope="module")
def sg2260e_sources():
    return {
        "tpukernel": _lower_core_ops("sg2260e", "tpukernel"),
        "rv": _lower_core_ops("sg2260e", "rv"),
    }


def test_public_ppl_helpers_form_one_backend_neutral_tir_contract():
    extern_names = set()

    def collect(node):
        if (isinstance(node, tvm.tir.Call) and getattr(node.op, "name", None) == "tir.call_extern"):
            extern_names.add(str(node.args[0].value))

    tvm.tir.stmt_functor.post_order_visit(_portable_core_ops.body, collect)
    assert extern_names == {
        "tl.tpu.copy",
        "tl.tpu.fill",
        "tl.tpu.gemm",
        "tl.tpu.add",
        "tl.tpu.sub",
        "tl.tpu.mul",
        "tl.tpu.div",
        "tl.tpu.max",
    }


def test_sg2260e_tpukernel_selects_only_tpukernel_instructions(sg2260e_sources):
    source = sg2260e_sources["tpukernel"]

    assert "TileLang TPU target: sg2260e, programming model: tpukernel" in source
    assert "TPU-Kernel externs require -tpu-programming-model=tpukernel" in source
    assert "tpu_initialize()" in source
    assert "tpu_poll()" in source
    for instruction in ("tpu_gdma_cpy_S2L(", "tpu_gdma_cpy_L2S(", "tpu_bdc_set_C(",
                        "tpu_bdc_fp_mm(", "tpu_bdc_fp_add(", "tpu_bdc_fp_sub(", "tpu_bdc_fp_mul(",
                        "tpu_bdc_fp_div(", "tpu_bdc_max("):
        assert instruction in source
    assert '#include "rvt_api.h"' not in source
    assert "rvt_kernel_start(" not in source
    assert "rvt_sync_i(" not in source
    assert "rvt_fmm" not in source


def test_sg2260e_rv_selects_only_rv_tensor_instructions(sg2260e_sources):
    source = sg2260e_sources["rv"]

    assert "TileLang TPU target: sg2260e, programming model: rv" in source
    assert "RVT externs require -tpu-programming-model=rv" in source
    assert '#include "atomic_def.h"' in source
    assert '#include "rvt_api.h"' in source
    assert "TILELANG_TPU_OPAQUE_RAW_RVT_ABI" not in source

    # The compiler, rather than the frontend program, owns this lifecycle.
    assert source.count("rvt_kernel_start()") == 1
    assert source.count("rvt_cfg_lanemask(gdma_get_lane_mask())") == 1
    assert source.count("rvt_sync_i(0xdeadbeef, 0)") == 1
    assert source.index("rvt_kernel_start()") > source.index("rvt_cfg_lanemask(")

    for descriptor_fragment in ("rvt_gr(32, PRECISION(DT_FP16), FP8TYPE(DT_FP16)",
                                "rvt_tr(8, PRECISION(DT_FP16), FP8TYPE(DT_FP16)",
                                "PRECISION(DT_FP32), FP8TYPE(DT_FP32)", "FREE_LAYOUT",
                                "HW_ALIGN_LAYOUT"):
        assert descriptor_fragment in source
    for instruction in ("rvt_dma_ld(", "rvt_dma_st(", "rvt_cp(", "rvt_fmm2a_nn(", "rvt_fadd(",
                        "rvt_fsub(", "rvt_fmul(", "rvt_cfg_rsqrt_iter(3)", "rvt_fdiv(",
                        "rvt_fmax("):
        assert instruction in source
    assert "rvt_cr(1, PRECISION(DT_FP32), FP8TYPE(DT_FP32)" in source
    assert "rvt_cfg_quant(0)" in source

    assert "tpu_initialize()" not in source
    assert "tpu_poll()" not in source
    assert "tpu_gdma_cpy_" not in source
    assert "tpu_bdc_fp_" not in source


def test_same_frontend_reaches_distinct_sg2260e_instruction_selectors(sg2260e_sources):
    tpukernel = sg2260e_sources["tpukernel"]
    rv = sg2260e_sources["rv"]

    assert tpukernel != rv
    assert "tpu_bdc_fp_mm(" in tpukernel and "rvt_fmm2a_nn(" not in tpukernel
    assert "rvt_fmm2a_nn(" in rv and "tpu_bdc_fp_mm(" not in rv
    assert "tpu_bdc_fp_div(" in tpukernel and "rvt_fdiv(" not in tpukernel
    assert "rvt_fdiv(" in rv and "tpu_bdc_fp_div(" not in rv


def test_bm1690_tpukernel_source_regression_and_rv_rejection():
    source = _lower_core_ops("bm1690", "tpukernel")
    assert "TileLang TPU target: bm1690, programming model: tpukernel" in source
    assert "tpu_bdc_fp_mm(" in source
    assert "tpu_bdc_fp_add(" in source
    assert '#include "rvt_api.h"' not in source

    with pytest.raises(ValueError, match="bm1690.*does not support programming model.*rv"):
        _lower_core_ops("bm1690", "rv")


def test_topk_capability_is_chip_specific_and_fails_before_runtime():
    bm1690_source = tilelang.lower(
        _tpukernel_topk,
        target=_tpu_target("bm1690", "tpukernel"),
        runtime_mode="cmodel",
    ).kernel_source
    assert "tpu_hau_sort_natural_index(" in bm1690_source

    with pytest.raises(ValueError, match="unavailable on SG2260E"):
        tilelang.lower(
            _tpukernel_topk,
            target=_tpu_target("sg2260e", "tpukernel"),
            runtime_mode="cmodel",
        )


def test_portable_ops_and_raw_rv_abi_cannot_share_a_kernel():
    with pytest.raises(tvm.error.TVMError, match="cannot share a kernel with raw.*rvt"):
        tilelang.lower(
            _portable_and_raw_rv,
            target=_tpu_target("sg2260e", "rv"),
            runtime_mode="cmodel",
        )


def test_reserved_runtime_entry_name_is_rejected():
    reserved = _reserved_entry_probe.with_attr("global_symbol", "main_kernel")
    with pytest.raises(tvm.error.TVMError, match="conflicts with a generated runtime entry"):
        tilelang.lower(
            reserved,
            target=_tpu_target("sg2260e", "tpukernel"),
            runtime_mode="cmodel",
        )


def test_global_to_global_copy_uses_system_memory_instruction():
    tpukernel = tilelang.lower(
        _global_to_global_copy,
        target=_tpu_target("sg2260e", "tpukernel"),
        runtime_mode="cmodel",
    ).kernel_source
    rv = tilelang.lower(
        _global_to_global_copy,
        target=_tpu_target("sg2260e", "rv"),
        runtime_mode="cmodel",
    ).kernel_source

    assert "tpu_gdma_cpy_S2S(" in tpukernel
    assert "tpu_bdc_cpy(" not in tpukernel
    assert "rvt_gr(32" in rv and "rvt_gr(33" in rv
    assert "rvt_dma_cp(33, 32)" in rv


@pytest.mark.parametrize("dtype", ("float16", "float32"))
@pytest.mark.parametrize("transfer", ("local", "global"))
@pytest.mark.parametrize(
    "chip,programming_model",
    (("sg2260e", "tpukernel"), ("sg2260e", "rv"), ("bm1690", "tpukernel")),
)
def test_portable_copy_cases_select_the_expected_instruction_path(dtype, transfer, chip,
                                                                  programming_model):
    source = tilelang.lower(
        _portable_copy_program(dtype, transfer),
        target=_tpu_target(chip, programming_model),
        runtime_mode="cmodel",
    ).kernel_source
    dtype_token = {"float16": "DT_FP16", "float32": "DT_FP32"}[dtype]

    if programming_model == "tpukernel":
        if transfer == "global":
            assert source.count("tpu_gdma_cpy_S2S(") == 1
            copy_lines = [line for line in source.splitlines() if "tpu_gdma_cpy_S2S(" in line]
            assert copy_lines[0].rstrip().endswith(f", {dtype_token});")
            for instruction in ("tpu_gdma_cpy_S2L(", "tpu_bdc_cpy(", "tpu_gdma_cpy_L2S(",
                                "tpu_bdc_cast("):
                assert instruction not in source
        else:
            assert source.count("tpu_gdma_cpy_S2L(") == 1
            assert source.count("tpu_bdc_cpy(") == 1
            assert source.count("tpu_gdma_cpy_L2S(") == 1
            copy_lines = [
                line for line in source.splitlines() if any(
                    instruction in line
                    for instruction in ("tpu_gdma_cpy_S2L(", "tpu_bdc_cpy(", "tpu_gdma_cpy_L2S("))
            ]
            assert len(copy_lines) == 3
            assert all(line.rstrip().endswith(f", {dtype_token});") for line in copy_lines)
            assert "tpu_gdma_cpy_S2S(" not in source
            assert "tpu_bdc_cast(" not in source
        assert "rvt_dma_" not in source
    elif transfer == "global":
        descriptors = [line for line in source.splitlines() if "rvt_gr(" in line]
        assert len(descriptors) == 2
        assert all(f"PRECISION({dtype_token})" in line and f"FP8TYPE({dtype_token})" in line and
                   "FREE_LAYOUT" in line for line in descriptors)
        assert source.count("rvt_dma_cp(33, 32)") == 1
        assert "rvt_dma_ld(" not in source
        assert "rvt_dma_st(" not in source
        assert "rvt_tr(" not in source
        assert "rvt_cvt_" not in source
        assert "tpu_gdma_cpy_" not in source
    else:
        descriptors = [
            line for line in source.splitlines() if "rvt_gr(" in line or "rvt_tr(" in line
        ]
        assert descriptors
        assert all(f"PRECISION({dtype_token})" in line and f"FP8TYPE({dtype_token})" in line and
                   "FREE_LAYOUT" in line for line in descriptors)
        assert source.count("rvt_dma_ld(9, 32)") == 1
        assert source.count("rvt_dma_cp(9, 8)") == 1
        assert source.count("rvt_dma_st(32, 8)") == 1
        assert "rvt_cvt_" not in source
        assert "tpu_gdma_cpy_" not in source


@pytest.mark.parametrize(
    ("chip", "programming_model", "instruction"),
    (("bm1690", "tpukernel", "tpu_bdc_cast("), ("sg2260e", "tpukernel", "tpu_bdc_cast("),
     ("sg2260e", "rv", "rvt_cvt_f2f(9, 8)")),
)
def test_fp16_to_bf16_copy_conversion_selects_validated_backend(chip, programming_model,
                                                                instruction):
    source = tilelang.lower(
        _local_fp16_to_bf16_copy,
        target=_tpu_target(chip, programming_model),
        runtime_mode="cmodel",
    ).kernel_source
    assert instruction in source


@pytest.mark.parametrize("programming_model", ("tpukernel", "rv"))
def test_integer_copy_conversion_capabilities_fail_closed(programming_model):
    diagnostic = ("requires floating-point operands"
                  if programming_model == "tpukernel" else "rvt_cvt_f2f only accepts")

    with pytest.raises(tvm.error.TVMError, match=diagnostic):
        tilelang.lower(
            _local_integer_convert,
            target=_tpu_target("sg2260e", programming_model),
            runtime_mode="cmodel",
        )


def test_portable_elementwise_rejects_non_matrix_local_layout_at_frontend():
    with pytest.raises(tvm.error.DiagnosticError):

        @T.prim_func
        def _rank4_elementwise():
            T.func_attr({"global_symbol": "rank4_elementwise"})
            with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
                lhs = T.alloc_shared((2, 4, 1, 8), "float32")
                rhs = T.alloc_shared((2, 4, 1, 8), "float32")
                out = T.alloc_shared((2, 4, 1, 8), "float32")
                T.ppl_add(out, lhs, rhs)


def test_overwrite_gemm_accepts_input_dtype_and_selects_non_accumulating_form():
    tpukernel = tilelang.lower(
        _overwrite_fp16_gemm,
        target=_tpu_target("sg2260e", "tpukernel"),
        runtime_mode="cmodel",
    ).kernel_source
    rv = tilelang.lower(
        _overwrite_fp16_gemm,
        target=_tpu_target("sg2260e", "rv"),
        runtime_mode="cmodel",
    ).kernel_source

    assert "DT_FP16, DT_FP16, false)" in tpukernel
    assert "overwrite_fp16_gemm();" in tpukernel
    assert "rvt_fmm2_nn(10, 8, 9, 0, 0, 0)" in rv
    assert "rvt_fmm2a_nn(" not in rv


@pytest.mark.parametrize("programming_model", ("tpukernel", "rv"))
@pytest.mark.parametrize("profiling", (False, True))
def test_sg2260e_pcie_core_ops_compile_and_link_without_loading(programming_model, profiling,
                                                                monkeypatch):
    """Validate the board artifact boundary without dlopen or dispatch."""
    if not os.environ.get("PPL_PROJECT_ROOT"):
        pytest.skip("PPL_PROJECT_ROOT is not configured")
    monkeypatch.delenv("TILELANG_TPU_PROFILE_SESSION", raising=False)
    monkeypatch.delenv("TILELANG_TPU_PROFILE_CHIP", raising=False)
    monkeypatch.delenv("TILELANG_TPU_PROFILE_PROGRAMMING_MODEL", raising=False)
    monkeypatch.delenv("TILELANG_TPU_PROFILE_RUNTIME_MODE", raising=False)
    if profiling:
        monkeypatch.setenv("TILELANG_TPU_PROFILE_SESSION", "1")
        monkeypatch.setenv("TILELANG_TPU_PROFILE_CHIP", "sg2260e")
        monkeypatch.setenv("TILELANG_TPU_PROFILE_PROGRAMMING_MODEL", programming_model)
        monkeypatch.setenv("TILELANG_TPU_PROFILE_RUNTIME_MODE", "pcie")

    target_spec = TPUTargetSpec("sg2260e", programming_model)
    runtime_config = TPURuntimeConfig("pcie")
    target = tvm.target.Target(_tpu_target("sg2260e", programming_model))
    artifact = tilelang.lower(
        _portable_core_ops,
        target=target,
        runtime_mode=runtime_config.runtime_mode,
    )
    generator = LibraryGenerator(target, tpu_target=target_spec, tpu_runtime=runtime_config)
    try:
        wrapper = TLWrapper(target, tpu_workspace_dir=generator.tpu_workspace_dir)
        module = tvm.IRModule({"portable_core_ops": _portable_core_ops})
        wrapper.assign_optimized_module(module)
        wrapper.assign_host_module(artifact.host_mod)
        wrapper.assign_device_module(artifact.device_mod)
        generator.update_lib_code(wrapper.wrap(artifact.kernel_source))
        generator.compile_lib(timeout=60)

        workspace = Path(generator.tpu_workspace_dir)
        assert (workspace / "libkernel.so").is_file()
        assert (workspace / "main.so").is_file()
        has_tpudnn_dependency = (b"libtpudnn.so" in (workspace / "main.so").read_bytes())
        assert has_tpudnn_dependency is profiling
        kernel_source = (workspace / "kernel.c").read_text(encoding="utf-8")
        if programming_model == "rv":
            assert "rvt_fmm2a_nn(" in kernel_source
            assert "tpu_initialize()" not in kernel_source
        else:
            assert "tpu_bdc_fp_mm(" in kernel_source
            assert "rvt_fmm2a_nn(" not in kernel_source
    finally:
        generator.remove_lib()
