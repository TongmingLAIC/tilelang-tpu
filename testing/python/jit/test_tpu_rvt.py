# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import os
from pathlib import Path

import pytest

import tilelang
from tilelang import tvm
import tilelang.language as T
from tilelang.engine.tpu_config import TPURuntimeConfig, TPUTargetSpec
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.ppl_layout import resolve_ppl_layout
from tilelang.jit.adapter.wrapper import TLWrapper


def _tpu_target(chip="bm1690e", programming_model="rv"):
    return (f"tpu -mcpu={chip} "
            f"-tpu-programming-model={programming_model}")


@T.prim_func
def _rvt_smoke_primfunc(A: T.Tensor((1,), "float32")):
    T.func_attr({"global_symbol": "rvt_smoke", "tir.noalias": T.bool(True)})
    T.rvt_kernel_start()
    # This is a lifecycle/control-path smoke test, not a tensor instruction
    # test. Tensor arithmetic additionally requires CR/TR/GR setup.
    T.rvt_sync_all()


@T.prim_func
def _rvt_codegen_primfunc(A: T.Tensor((1,), "float32")):
    """Compile/link coverage for a real RVT tensor instruction.

    The raw register IDs are deliberately not executed: a valid fadd needs
    preceding CR/TR/GR configuration and descriptor lifetime management.
    """
    T.func_attr({"global_symbol": "rvt_codegen", "tir.noalias": T.bool(True)})
    T.rvt_kernel_start()
    T.rvt_fadd(T.uint64(10), T.uint64(8), T.uint64(9))
    T.rvt_sync_all()


@T.prim_func
def _plain_multifunc_primfunc(A: T.Tensor((1,), "float32")):
    T.func_attr({"global_symbol": "plain_multifunc", "tir.noalias": T.bool(True)})
    T.evaluate(0)


@T.prim_func
def _tpukernel_fill_primfunc(A: T.Tensor((1, 64), "float32")):
    T.func_attr({"global_symbol": "tpukernel_fill", "tir.noalias": T.bool(True)})
    with T.Kernel(1, 1, is_cpu=True) as (bx, by):
        local = T.alloc_shared((1, 64), "float32")
        T.ppl_fill(local, T.float32(0))
        T.ppl_copy(local, A)


def test_rvt_call_requires_rvt_symbol_name():
    with pytest.raises(ValueError, match="beginning with 'rvt_'"):
        T.rvt_call("ppl.copy")


def test_rvt_frontend_emits_vendor_extern_calls():
    call = T.rvt_fadd(
        tvm.tir.const(1, "uint64"), tvm.tir.const(2, "uint64"), tvm.tir.const(3, "uint64"))
    assert call.op.same_as(tvm.ir.Op.get("tir.call_extern"))
    assert call.args[0].value == "rvt_fadd"

    dma = T.rvt_call("rvt_dma_hscatter", tvm.tir.const(1, "uint64"), tvm.tir.const(2, "uint64"),
                     tvm.tir.const(3, "uint64"), tvm.tir.const(4, "uint64"),
                     tvm.tir.const(5, "uint64"))
    assert dma.args[0].value == "rvt_dma_hscatter"


def test_rvt_lowering_keeps_explicit_vendor_calls():
    artifact = tilelang.lower(
        _rvt_codegen_primfunc,
        target=_tpu_target("bm1690e", "rv"),
        runtime_mode="cmodel",
    )
    assert artifact.tpu_target == TPUTargetSpec("bm1690e", "rv")
    assert artifact.tpu_runtime == TPURuntimeConfig("cmodel")
    source = artifact.kernel_source
    assert '#ifndef TILELANG_TPU_RV' in source
    assert '#error "RVT externs require -tpu-programming-model=rv"' in source
    assert '#include "rvt_api.h"' in source
    assert "rvt_kernel_start()" in source
    assert "rvt_fadd((uint64_t)10, (uint64_t)8, (uint64_t)9)" in source
    assert "rvt_sync_all()" in source

    with pytest.raises(ValueError, match=r"programming.model.*tpukernel.*rvt_fadd"):
        tilelang.lower(
            _rvt_codegen_primfunc,
            target=_tpu_target("bm1690e", "tpukernel"),
            runtime_mode="cmodel",
        )


def test_raw_rvt_cannot_consume_tilelang_buffer_descriptors():

    @T.prim_func
    def raw_descriptor_argument(A: T.Tensor((32,), "float32")):
        T.func_attr({"global_symbol": "raw_descriptor_argument"})
        T.evaluate(T.rvt_dma_ld(T.uint64(8), A.data))

    with pytest.raises(
            ValueError, match=r"raw-rvt-descriptor-argument.*rvt_dma_ld.*descriptor Var"):
        tilelang.lower(
            raw_descriptor_argument,
            target=_tpu_target("bm1690e", "rv"),
            runtime_mode="cmodel",
        )

    # Direct native callers must not be able to bypass the same ownership
    # boundary by skipping the Python residual-IR verifier.
    native_function = raw_descriptor_argument.with_attr("global_symbol", "raw_descriptor_argument")
    codegen = tvm._ffi.get_global_func("target.build.tilelang_tpu")
    with pytest.raises(
            tvm.error.TVMError, match=r"rvt_dma_ld.*cannot consume TileLang tensor descriptor Var"):
        codegen(
            tvm.IRModule({"raw_descriptor_argument": native_function}),
            tvm.target.Target(_tpu_target("bm1690e", "rv")),
        )

    parameter = raw_descriptor_argument.params[0]
    parameter_call = tvm.tir.call_extern("handle", "rvt_dma_ld", tvm.tir.const(8, "uint64"),
                                         parameter)
    parameter_function = tvm.tir.PrimFunc(
        raw_descriptor_argument.params,
        tvm.tir.Evaluate(parameter_call),
        buffer_map=raw_descriptor_argument.buffer_map,
    ).with_attr("global_symbol", "raw_parameter_handle")
    with pytest.raises(
            tvm.error.TVMError, match=r"rvt_dma_ld.*cannot consume TileLang tensor descriptor Var"):
        codegen(
            tvm.IRModule({"raw_parameter_handle": parameter_function}),
            tvm.target.Target(_tpu_target("bm1690e", "rv")),
        )


def test_raw_rvt_symbol_must_be_a_c_identifier_at_both_boundaries():
    invalid_call = tvm.tir.call_extern("handle", "rvt_bad-name")
    function = tvm.tir.PrimFunc(
        [],
        tvm.tir.Evaluate(invalid_call),
    ).with_attr("global_symbol", "invalid_raw_rvt_symbol")
    module = tvm.IRModule({"invalid_raw_rvt_symbol": function})

    with pytest.raises(ValueError, match=r"raw-rvt-symbol.*rvt_bad-name"):
        tilelang.lower(
            module,
            target=_tpu_target("bm1690e", "rv"),
            runtime_mode="cmodel",
        )

    codegen = tvm._ffi.get_global_func("target.build.tilelang_tpu")
    with pytest.raises(tvm.error.TVMError, match=r"C identifier beginning with rvt_.*rvt_bad-name"):
        codegen(
            module,
            tvm.target.Target(_tpu_target("bm1690e", "rv")),
        )


def test_tpu_codegen_rejects_a_multifunction_runtime_module():
    """One generated main_kernel entry cannot safely wrap multiple PrimFuncs."""
    module = tvm.IRModule({
        "rvt_first": _rvt_smoke_primfunc,
        "plain_last": _plain_multifunc_primfunc,
    })
    with pytest.raises(tvm.error.TVMError, match="exactly one PrimFunc"):
        tilelang.lower(
            module,
            target=_tpu_target("bm1690e", "rv"),
            runtime_mode="cmodel",
        )


def test_tpukernel_externs_have_a_separate_programming_model_fence():
    source = tilelang.lower(
        _tpukernel_fill_primfunc,
        target=_tpu_target("bm1690e", "tpukernel"),
        runtime_mode="cmodel",
    ).kernel_source

    assert "/* TileLang TPU target: bm1690e, programming model: tpukernel */" in source
    assert '#ifndef TILELANG_TPU_TPUKERNEL' in source
    assert '#error "TPU-Kernel externs require -tpu-programming-model=tpukernel"' in source


@pytest.mark.parametrize("extern_name", [
    "tpu_bdc_test_only",
    "tpu_sdma_test_only",
    "tpu_sync_all_bdc",
])
@pytest.mark.parametrize("programming_model", ["tpukernel", "rv"])
def test_raw_tpukernel_extern_is_not_a_supported_tir_abi(extern_name, programming_model):
    raw_tpukernel = tvm.tir.PrimFunc(
        [],
        tvm.tir.Evaluate(tvm.tir.call_extern("handle", extern_name)),
    )
    module = tvm.IRModule({"raw_tpukernel": raw_tpukernel})

    with pytest.raises(ValueError, match=rf"Raw tpu_\* call_extern.*{extern_name}"):
        tilelang.lower(
            module,
            target=_tpu_target("bm1690e", programming_model),
            runtime_mode="cmodel",
        )


@pytest.mark.parametrize("extern_name,programming_model", [
    ("tpu_sdma_test_only", "tpukernel"),
    ("tpu_sdma_test_only", "rv"),
    ("rvt_fadd", "tpukernel"),
])
def test_native_codegen_fences_contractless_direct_ffi_externs(extern_name, programming_model):
    """The native target guard must survive callers that bypass lower()."""
    raw_extern = tvm.tir.PrimFunc(
        [],
        tvm.tir.Evaluate(tvm.tir.call_extern("handle", extern_name)),
    ).with_attr("global_symbol", "native_model_fence")
    target = tvm.target.Target({
        "kind": "tpu",
        "mcpu": "bm1690e",
        "tpu-programming-model": programming_model,
    })
    codegen = tvm._ffi.get_global_func("target.build.tilelang_tpu")

    with pytest.raises(tvm.error.TVMError, match=extern_name):
        codegen(tvm.IRModule({"native_model_fence": raw_extern}), target)


def test_native_codegen_rejects_an_unsupported_chip_model_pair():
    rvt_extern = tvm.tir.PrimFunc(
        [],
        tvm.tir.Evaluate(tvm.tir.call_extern("handle", "rvt_fadd")),
    ).with_attr("global_symbol", "bm_rv_is_invalid")
    target = tvm.target.Target({
        "kind": "tpu",
        "mcpu": "bm1690",
        "tpu-programming-model": "rv",
    })
    codegen = tvm._ffi.get_global_func("target.build.tilelang_tpu")

    with pytest.raises(tvm.error.TVMError, match="does not support"):
        codegen(tvm.IRModule({"bm_rv_is_invalid": rvt_extern}), target)


def test_native_codegen_ignores_workload_model_metadata_for_chip_selection():
    raw_extern = tvm.tir.PrimFunc(
        [],
        tvm.tir.Evaluate(tvm.tir.call_extern("handle", "rvt_sync_all")),
    ).with_attr("global_symbol", "workload_model_metadata")
    target = tvm.target.Target({
        "kind": "tpu",
        "mcpu": "bm1690e",
        "model": "matmul_smoke",
        "tpu-programming-model": "rv",
    })
    codegen = tvm._ffi.get_global_func("target.build.tilelang_tpu")

    source = codegen(tvm.IRModule({"workload_model_metadata": raw_extern}), target)
    assert "TileLang TPU target: bm1690e, programming model: rv" in source


def test_local_bm1690e_ppl_rvt_header_if_sdk_is_configured():
    """A real SDK check is opt-in, so unit tests remain hermetic by default."""
    ppl_root = os.environ.get("PPL_PROJECT_ROOT")
    if not ppl_root:
        pytest.skip("PPL_PROJECT_ROOT is not configured")
    layout = resolve_ppl_layout(ppl_root, "bm1690e")
    assert layout.require_rvt_api().name == "rvt_api.h"

    generator = LibraryGenerator(
        tvm.target.Target(_tpu_target("bm1690e", "rv")),
        tpu_target=TPUTargetSpec("bm1690e", "rv"),
        tpu_runtime=TPURuntimeConfig("cmodel"),
    )
    try:
        definitions, _ = generator._ppl_compile_flags(layout, ".", "rv", "cmodel")
        assert "-DTILELANG_TPU_RV" in definitions
        tpukernel_definitions, _ = generator._ppl_compile_flags(layout, ".", "tpukernel", "cmodel")
        assert "-DTILELANG_TPU_TPUKERNEL" in tpukernel_definitions
    finally:
        generator.remove_lib()


def test_rvt_cmodel_compile_is_private_if_sdk_is_configured():
    """Compile/link the real RVT source without loading or dispatching it."""
    if not os.environ.get("PPL_PROJECT_ROOT"):
        pytest.skip("PPL_PROJECT_ROOT is not configured")

    target_spec = TPUTargetSpec("bm1690e", "rv")
    runtime_config = TPURuntimeConfig("cmodel")
    target = tvm.target.Target(_tpu_target("bm1690e", "rv"))
    artifact = tilelang.lower(
        _rvt_codegen_primfunc,
        target=target,
        runtime_mode=runtime_config.runtime_mode,
    )
    generator = LibraryGenerator(target, tpu_target=target_spec, tpu_runtime=runtime_config)
    try:
        wrapper = TLWrapper(target, tpu_workspace_dir=generator.tpu_workspace_dir)
        wrapper.assign_optimized_module(tvm.IRModule({"rvt_codegen": _rvt_codegen_primfunc}))
        wrapper.assign_host_module(artifact.host_mod)
        wrapper.assign_device_module(artifact.device_mod)
        generator.update_lib_code(wrapper.wrap(artifact.kernel_source))
        generator.compile_lib(timeout=60)

        workspace = Path(generator.tpu_workspace_dir)
        kernel_path = workspace / "libkernel.so"
        main_path = workspace / "main.so"
        assert kernel_path.is_file() and main_path.is_file()
        assert b"rvt_fadd" in (workspace / "kernel.c").read_bytes()
        assert str(kernel_path).encode() in main_path.read_bytes()
        assert b'setenv("TPU_RT_CORE_NUM", TILELANG_TPU_CMODEL_CORE_NUM, 1)' in (
            workspace / "main.cpp").read_bytes()
        assert b"tilelang_tpu_bind_device" in (workspace / "main.cpp").read_bytes()
        assert b"tilelang_tpu_expected_device_id != device_id" in (workspace /
                                                                   "main.cpp").read_bytes()
        assert b"tpudnnEnableProfile" not in main_path.read_bytes()
    finally:
        generator.remove_lib()


def test_tpukernel_pcie_compile_is_private_if_sdk_is_configured():
    """Build a BM1690E PPL kernel for PCIe without dlopen or dispatching it."""
    if not os.environ.get("PPL_PROJECT_ROOT"):
        pytest.skip("PPL_PROJECT_ROOT is not configured")

    target_spec = TPUTargetSpec("bm1690e", "tpukernel")
    runtime_config = TPURuntimeConfig("pcie")
    target = tvm.target.Target(_tpu_target("bm1690e", "tpukernel"))
    artifact = tilelang.lower(
        _tpukernel_fill_primfunc,
        target=target,
        runtime_mode=runtime_config.runtime_mode,
    )
    generator = LibraryGenerator(target, tpu_target=target_spec, tpu_runtime=runtime_config)
    try:
        wrapper = TLWrapper(generator.target, tpu_workspace_dir=generator.tpu_workspace_dir)
        wrapper.assign_optimized_module(tvm.IRModule({"tpukernel_fill": _tpukernel_fill_primfunc}))
        wrapper.assign_host_module(artifact.host_mod)
        wrapper.assign_device_module(artifact.device_mod)
        generator.update_lib_code(wrapper.wrap(artifact.kernel_source))
        generator.compile_lib(timeout=60)

        workspace = Path(generator.tpu_workspace_dir)
        assert (workspace / "libkernel.so").is_file()
        assert (workspace / "main.so").is_file()
        source = (workspace / "kernel.c").read_text(encoding="utf-8")
        assert "TileLang TPU target: bm1690e" in source
        assert "TPU-Kernel externs require -tpu-programming-model=tpukernel" in source
        header = (workspace / "kernel.h").read_text(encoding="utf-8")
        assert "__bm1690__" not in header
        assert "TPU chip macro is required" in header
        main_source = (workspace / "main.cpp").read_text(encoding="utf-8")
        assert "tpudnnHandleFromStream" in main_source
        assert "tpudnnEnableProfile" in main_source
        assert "tpudnnDisableProfile" in main_source
        assert b"libtpudnn.so" not in (workspace / "main.so").read_bytes()
        assert b"libcdm_daemon_emulator.so" not in (workspace / "main.so").read_bytes()
    finally:
        generator.remove_lib()


def test_rvt_pcie_compile_is_private_without_loading_if_sdk_is_configured():
    """Cross-compile/link RVT without dlopen or a board dispatch."""
    if not os.environ.get("PPL_PROJECT_ROOT"):
        pytest.skip("PPL_PROJECT_ROOT is not configured")

    target_spec = TPUTargetSpec("bm1690e", "rv")
    runtime_config = TPURuntimeConfig("pcie")
    target = tvm.target.Target(_tpu_target("bm1690e", "rv"))
    artifact = tilelang.lower(
        _rvt_codegen_primfunc,
        target=target,
        runtime_mode=runtime_config.runtime_mode,
    )
    generator = LibraryGenerator(target, tpu_target=target_spec, tpu_runtime=runtime_config)
    try:
        wrapper = TLWrapper(target, tpu_workspace_dir=generator.tpu_workspace_dir)
        wrapper.assign_optimized_module(tvm.IRModule({"rvt_codegen": _rvt_codegen_primfunc}))
        wrapper.assign_host_module(artifact.host_mod)
        wrapper.assign_device_module(artifact.device_mod)
        generator.update_lib_code(wrapper.wrap(artifact.kernel_source))
        generator.compile_lib(timeout=60)

        workspace = Path(generator.tpu_workspace_dir)
        kernel_path = workspace / "libkernel.so"
        main_path = workspace / "main.so"
        assert kernel_path.is_file() and main_path.is_file()
        assert b"rvt_fadd" in (workspace / "kernel.c").read_bytes()
        assert str(kernel_path).encode() in main_path.read_bytes()
        assert b"tpudnnEnableProfile" not in main_path.read_bytes()
        assert b"tpudnnDisableProfile" not in main_path.read_bytes()
        assert b"libtpudnn.so" not in main_path.read_bytes()
        assert b"libcdm_daemon_emulator.so" not in main_path.read_bytes()
    finally:
        generator.remove_lib()
