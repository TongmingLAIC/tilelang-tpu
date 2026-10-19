import importlib
from types import SimpleNamespace

import pytest

import tilelang
from tilelang import tvm

kernel_module = importlib.import_module("tilelang.jit.kernel")
jit_api = importlib.import_module("tilelang.jit")
target_utils = importlib.import_module("tilelang.utils.target")
lower_module = importlib.import_module("tilelang.engine.lower")
phase_module = importlib.import_module("tilelang.engine.phase")

from tilelang.engine.tpu_config import (
    TPU_CHIP_SPECS,
    TPURuntimeConfig,
    TPUTargetSpec,
    get_tpu_chip_spec,
    resolve_tpu_runtime,
    resolve_tpu_target,
)
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.kernel import JITKernel
from tilelang.cache.kernel_cache import KernelCache


def _tpu_target(chip="bm1690e", programming_model="tpukernel"):
    return (f"tpu -mcpu={chip} "
            f"-tpu-programming-model={programming_model}")


def test_auto_target_never_implicitly_selects_a_tpu(monkeypatch):
    monkeypatch.setattr(target_utils, "check_cuda_availability", lambda: False)
    monkeypatch.setattr(target_utils, "check_hip_availability", lambda: False)
    # Toolchain/runtime environment is not a second compilation selector.
    monkeypatch.setenv("TILELANG_TPU_CHIP", "bm1690e")
    monkeypatch.setenv("TILELANG_TPU_PROGRAMMING_MODEL", "rv")
    monkeypatch.setenv("PPL_PROJECT_ROOT", "/not/consulted/by-auto-target")
    with pytest.raises(ValueError, match="complete TPU target"):
        target_utils.determine_target("auto")


def test_tpu_pipeline_does_not_run_unvalidated_pipeline_or_vector_passes(monkeypatch):
    target = tvm.target.Target(_tpu_target())
    prim_func = tvm.tir.PrimFunc(
        [],
        tvm.tir.Evaluate(0),
    ).with_attr("global_symbol", "conservative_tpu_pipeline")
    mod = tvm.IRModule({"conservative_tpu_pipeline": prim_func})

    def forbidden_pass():
        raise AssertionError("unsupported TPU optimization was invoked")

    for pass_name in ("LegalizeVectorizedLoop", "PipelinePlanning", "InjectSoftwarePipeline",
                      "VectorizeLoop"):
        monkeypatch.setattr(tilelang.transform, pass_name, forbidden_pass)

    mod = phase_module.LowerAndLegalize(mod, target)
    phase_module.OptimizeForTarget(mod, target)


@pytest.mark.parametrize("phase", [
    phase_module.LowerAndLegalize,
    phase_module.OptimizeForTarget,
])
def test_direct_tpu_phase_entrypoints_require_a_complete_target(phase):
    prim_func = tvm.tir.PrimFunc(
        [],
        tvm.tir.Evaluate(0),
    ).with_attr("global_symbol", "incomplete_phase_target")
    mod = tvm.IRModule({"incomplete_phase_target": prim_func})
    incomplete = tvm.target.Target("tpu -mcpu=bm1690e")

    with pytest.raises(ValueError, match="explicit programming model"):
        phase(mod, incomplete)


@pytest.mark.parametrize("phase", [
    phase_module.LowerAndLegalize,
    phase_module.OptimizeForTarget,
])
@pytest.mark.parametrize("function_target,requested_target", [
    (
        _tpu_target("bm1690", "tpukernel"),
        _tpu_target("bm1690e", "tpukernel"),
    ),
    (
        _tpu_target("bm1690e", "rv"),
        _tpu_target("bm1690e", "tpukernel"),
    ),
])
def test_direct_tpu_phase_entrypoints_reject_a_different_bound_identity(
        phase, function_target, requested_target):
    prim_func = tvm.tir.PrimFunc(
        [],
        tvm.tir.Evaluate(0),
    ).with_attr("global_symbol",
                "phase_target_mismatch").with_attr("target", tvm.target.Target(function_target))
    mod = tvm.IRModule({"phase_target_mismatch": prim_func})

    with pytest.raises(ValueError, match=rf"{phase.__name__} target identity mismatch"):
        phase(mod, tvm.target.Target(requested_target))


def test_tpu_contract_is_revalidated_before_address_assignment(monkeypatch):
    target = _tpu_target("bm1690e", "tpukernel")
    original = tvm.tir.PrimFunc(
        [],
        tvm.tir.Evaluate(0),
    ).with_attr("global_symbol", "contract_order")
    incompatible = tvm.tir.PrimFunc(
        [],
        tvm.tir.Evaluate(tvm.tir.call_extern("handle", "rvt_fadd")),
    ).with_attr("global_symbol", "contract_order")
    rewritten = tvm.IRModule({"contract_order": incompatible})
    address_assignment_called = False

    monkeypatch.setattr(lower_module, "LowerAndLegalize", lambda mod, _target: mod)
    monkeypatch.setattr(lower_module, "OptimizeForTarget", lambda _mod, _target: rewritten)

    def record_address_assignment(mod, _target):
        nonlocal address_assignment_called
        address_assignment_called = True
        return mod

    monkeypatch.setattr(lower_module, "AssignTPUAddresses", record_address_assignment)

    with pytest.raises(ValueError, match="different programming model"):
        tilelang.lower(original, target=target)
    assert not address_assignment_called


def test_compile_forwards_only_canonical_target_and_runtime(monkeypatch):
    captured = {}

    def fake_cached(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(jit_api, "cached", fake_cached)
    target = _tpu_target("bm1690e", "rv")
    jit_api.compile(
        func=object(),
        target=target,
        runtime_mode="cmodel",
    )

    assert captured["target"] == target
    assert captured["runtime_mode"] == "cmodel"


def test_jit_forwards_explicit_tpu_runtime_mode(monkeypatch):
    captured = {}
    adapter = object()

    def fake_compile(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(adapter=adapter)

    monkeypatch.setattr(jit_api, "compile", fake_compile)
    result = jit_api.jit(
        func=object(),
        target=_tpu_target("bm1690e", "rv"),
        runtime_mode="pcie",
    )

    assert result is adapter
    assert captured["runtime_mode"] == "pcie"


def test_tpu_target_kind_registration_matches_backend_contract():
    options = tvm.target.TargetKind.options_from_name("tpu")
    assert options["mcpu"] == "runtime.String"
    assert options["tpu-programming-model"] == "runtime.String"

    target = tvm.target.Target(_tpu_target("bm1690e", "rv"))
    assert "tpu" in target.keys


def test_tpu_chip_capabilities_are_explicit_and_fail_closed():
    sg = get_tpu_chip_spec("BM1690E")
    assert sg.ppl_arch == "tpub_7_1_e"
    assert sg.ppl_compile_definitions == ("__tpub_7_1_e__", "__bm1690e__")
    assert sg.physical_core_count == 4
    assert sg.programming_models == ("tpukernel", "rv")

    bm = get_tpu_chip_spec("bm1690")
    assert bm.ppl_arch == "tpub_7_1"
    assert bm.ppl_compile_definitions == ("__tpub_7_1__", "__sg2260__")
    assert bm.physical_core_count == 8
    assert bm.programming_models == ("tpukernel",)

    with pytest.raises(ValueError, match="does not support programming model"):
        TPUTargetSpec("bm1690", "rv")
    with pytest.raises(ValueError, match="Unsupported TPU chip"):
        TPUTargetSpec("bm1690erv", "rv")


@pytest.mark.parametrize("chip,programming_model,supported", [
    ("bm1690", "tpukernel", True),
    ("bm1690", "rv", False),
    ("bm1690e", "tpukernel", True),
    ("bm1690e", "rv", True),
])
def test_python_and_native_tpu_capability_boundaries_share_one_matrix(chip, programming_model,
                                                                      supported):
    target = tvm.target.Target(_tpu_target(chip, programming_model))
    prim_func = tvm.tir.PrimFunc(
        [],
        tvm.tir.Evaluate(0),
    ).with_attr("global_symbol", "capability_matrix")
    native_codegen = tvm._ffi.get_global_func("target.build.tilelang_tpu")

    if supported:
        assert resolve_tpu_target(target=target) == TPUTargetSpec(chip, programming_model)
        source = native_codegen(tvm.IRModule({"capability_matrix": prim_func}), target)
        assert f"target: {chip}, programming model: {programming_model}" in source
    else:
        with pytest.raises(ValueError, match="does not support programming model"):
            resolve_tpu_target(target=target)
        with pytest.raises(tvm.error.TVMError, match="does not support"):
            native_codegen(tvm.IRModule({"capability_matrix": prim_func}), target)


def test_address_assignment_rejects_a_different_bound_tpu_identity():
    function_target = tvm.target.Target(_tpu_target("bm1690", "tpukernel"))
    requested_target = tvm.target.Target(_tpu_target("bm1690e", "tpukernel"))
    prim_func = tvm.tir.PrimFunc(
        [],
        tvm.tir.Evaluate(0),
    ).with_attr("global_symbol", "address_target_mismatch").with_attr("target", function_target)

    with pytest.raises(ValueError, match="target identity mismatch"):
        phase_module.AssignTPUAddresses(
            tvm.IRModule({"address_target_mismatch": prim_func}), requested_target)


def test_target_and_runtime_are_resolved_as_independent_identities():
    target = tvm.target.Target(_tpu_target("bm1690e", "rv"))
    assert resolve_tpu_target(target=target) == TPUTargetSpec("bm1690e", "rv")
    assert resolve_tpu_runtime(runtime_mode=None) == TPURuntimeConfig("cmodel")
    assert resolve_tpu_runtime(runtime_mode="pcie") == TPURuntimeConfig("pcie")


def test_tpu_target_uses_a_backend_specific_dispatch_key():
    target = tvm.target.Target(_tpu_target("bm1690e", "rv"))

    assert [str(key) for key in target.keys] == ["tpu"]
    assert "cpu" not in [str(key) for key in target.keys]


def test_tpu_target_requires_both_canonical_compile_axes():
    with pytest.raises(ValueError, match="explicit physical chip"):
        resolve_tpu_target(target=tvm.target.Target("tpu -tpu-programming-model=tpukernel"))
    with pytest.raises(ValueError, match="explicit programming model"):
        resolve_tpu_target(target=tvm.target.Target("tpu -mcpu=bm1690e"))
    with pytest.raises(ValueError, match="Unsupported TPU programming model"):
        resolve_tpu_target(
            target=tvm.target.Target("tpu -mcpu=bm1690e -tpu-programming-model=legacy"))


def test_target_model_is_workload_metadata_not_a_chip_alias():
    target = tvm.target.Target({
        "kind": "tpu",
        "mcpu": "bm1690e",
        "model": "matmul_smoke",
        "tpu-programming-model": "tpukernel",
    })
    assert resolve_tpu_target(target=target) == TPUTargetSpec("bm1690e", "tpukernel")
    assert target.model == "matmul_smoke"


def test_native_and_python_target_boundaries_share_normalization():
    target = tvm.target.Target({
        "kind": "tpu",
        "mcpu": " BM1690E ",
        "tpu-programming-model": " tpukernel ",
    })
    assert resolve_tpu_target(target=target) == TPUTargetSpec("bm1690e", "tpukernel")

    prim_func = tvm.tir.PrimFunc(
        [],
        tvm.tir.Evaluate(0),
    ).with_attr("global_symbol", "normalized_target")
    source = tvm._ffi.get_global_func("target.build.tilelang_tpu")(tvm.IRModule(
        {"normalized_target": prim_func}), target)
    assert "target: bm1690e, programming model: tpukernel" in source


@pytest.mark.parametrize("function_target,build_target,message", [
    (
        _tpu_target("bm1690", "tpukernel"),
        _tpu_target("bm1690e", "tpukernel"),
        "PrimFunc chip bm1690 disagrees with build target chip bm1690e",
    ),
    (
        _tpu_target("bm1690e", "rv"),
        _tpu_target("bm1690e", "tpukernel"),
        "PrimFunc programming model rv disagrees with build target",
    ),
])
def test_native_tpu_codegen_rejects_a_mismatched_primfunc_target(function_target, build_target,
                                                                 message):
    prim_func = tvm.tir.PrimFunc(
        [],
        tvm.tir.Evaluate(0),
    ).with_attr("global_symbol",
                "native_target_mismatch").with_attr("target", tvm.target.Target(function_target))
    codegen = tvm._ffi.get_global_func("target.build.tilelang_tpu")

    with pytest.raises(tvm.error.TVMError, match=message):
        codegen(
            tvm.IRModule({"native_target_mismatch": prim_func}),
            tvm.target.Target(build_target),
        )


@pytest.mark.parametrize("extern_name", ["AtomicAdd", "cuda_helper"])
def test_native_tpu_codegen_rejects_unknown_externs(extern_name):
    prim_func = tvm.tir.PrimFunc(
        [],
        tvm.tir.Evaluate(tvm.tir.call_extern("handle", extern_name)),
    ).with_attr("global_symbol", "native_unknown_extern")
    codegen = tvm._ffi.get_global_func("target.build.tilelang_tpu")

    with pytest.raises(tvm.error.TVMError, match="Unknown external call"):
        codegen(
            tvm.IRModule({"native_unknown_extern": prim_func}),
            tvm.target.Target(_tpu_target()),
        )


def test_native_tpu_codegen_does_not_emit_cuda_math_constants():
    body = tvm.tir.SeqStmt([
        tvm.tir.Evaluate(tvm.tir.FloatImm("float32", float("inf"))),
        tvm.tir.Evaluate(tvm.tir.FloatImm("float32", float("nan"))),
        tvm.tir.Evaluate(tvm.tir.FloatImm("float64", float("-inf"))),
        tvm.tir.Evaluate(tvm.tir.FloatImm("bfloat16", float("inf"))),
    ])
    prim_func = tvm.tir.PrimFunc(
        [],
        body,
    ).with_attr("global_symbol", "native_math_constants")
    codegen = tvm._ffi.get_global_func("target.build.tilelang_tpu")

    source = codegen(
        tvm.IRModule({"native_math_constants": prim_func}),
        tvm.target.Target(_tpu_target()),
    )

    assert "CUDART_" not in source
    assert "__builtin_inff()" in source
    assert "__builtin_nanf(\"\")" in source
    assert "-__builtin_inf()" in source
    assert "bfloat16_t(__builtin_inff())" in source


def test_native_tpu_codegen_let_type_does_not_depend_on_name():
    shared_scalar = tvm.tir.Var("shared_scalar", "int32")
    body = tvm.tir.LetStmt(
        shared_scalar,
        tvm.tir.IntImm("int32", 7),
        tvm.tir.Evaluate(shared_scalar),
    )
    prim_func = tvm.tir.PrimFunc(
        [],
        body,
    ).with_attr("global_symbol", "native_scalar_let")
    codegen = tvm._ffi.get_global_func("target.build.tilelang_tpu")

    source = codegen(
        tvm.IRModule({"native_scalar_let": prim_func}),
        tvm.target.Target(_tpu_target()),
    )

    assert "int32_t shared_scalar = 7;" in source
    assert "__tilelang_tpu_tensor_info shared_scalar" not in source


def test_native_tpu_codegen_rejects_direct_scalar_tensor_access():
    source = tvm.tir.decl_buffer((8,), "float32", name="source")
    load = tvm.tir.BufferLoad(source, [tvm.tir.IntImm("int32", 0)])
    prim_func = tvm.tir.PrimFunc(
        [source.data],
        tvm.tir.Evaluate(load),
        buffer_map={
            source.data: source
        },
    ).with_attr("global_symbol", "native_scalar_tensor_access")
    codegen = tvm._ffi.get_global_func("target.build.tilelang_tpu")

    with pytest.raises(
            tvm.error.TVMError, match="Direct scalar BufferLoad/BufferStore.*unsupported"):
        codegen(
            tvm.IRModule({"native_scalar_tensor_access": prim_func}),
            tvm.target.Target(_tpu_target()),
        )


def test_native_tpu_codegen_rejects_nonserial_loops_and_attributes():
    loop_var = tvm.tir.Var("i", "int32")
    parallel = tvm.tir.For(loop_var, 0, 4, tvm.tir.ForKind.PARALLEL, tvm.tir.Evaluate(0))
    parallel_func = tvm.tir.PrimFunc(
        [],
        parallel,
    ).with_attr("global_symbol", "native_parallel_loop")
    codegen = tvm._ffi.get_global_func("target.build.tilelang_tpu")
    target = tvm.target.Target(_tpu_target())

    with pytest.raises(tvm.error.TVMError, match="supports only serial and unrolled loops"):
        codegen(tvm.IRModule({"native_parallel_loop": parallel_func}), target)

    attribute_func = tvm.tir.PrimFunc(
        [],
        tvm.tir.AttrStmt(
            tvm.tir.StringImm("payload"), "pragma_import_c",
            tvm.tir.StringImm("side_effecting_source"), tvm.tir.Evaluate(0)),
    ).with_attr("global_symbol", "native_residual_attribute")
    with pytest.raises(tvm.error.TVMError, match="Residual AttrStmt pragma_import_c"):
        codegen(
            tvm.IRModule({"native_residual_attribute": attribute_func}),
            target,
        )


@pytest.mark.parametrize("chip", ["bm1690", "bm1690e"])
def test_native_tpukernel_codegen_rejects_pure_rv_extern_bypass(chip):
    function = tvm.tir.PrimFunc(
        [],
        tvm.tir.Evaluate(tvm.tir.call_pure_extern("int32", "rvt_fadd")),
    ).with_attr("global_symbol", "pure_rv_bypass")
    codegen = tvm._ffi.get_global_func("target.build.tilelang_tpu")

    with pytest.raises(tvm.error.TVMError, match="call_pure_extern has no TPU semantic ABI"):
        codegen(
            tvm.IRModule({"pure_rv_bypass": function}),
            tvm.target.Target(_tpu_target(chip, "tpukernel")),
        )


def test_native_tpu_codegen_rejects_descriptor_let_aliases():
    pointer_type = tvm.ir.PointerType(tvm.ir.PrimType("float32"), "shared")
    tile = tvm.tir.Var("tile", pointer_type)
    alias = tvm.tir.Var("alias", "handle")
    body = tvm.tir.Allocate(
        tile,
        "float32",
        [32],
        tvm.tir.IntImm("bool", 1),
        tvm.tir.LetStmt(alias, tile, tvm.tir.Evaluate(0)),
    )
    prim_func = tvm.tir.PrimFunc(
        [],
        body,
    ).with_attr("global_symbol",
                "native_descriptor_let").with_attr("tilelang.tpu.lmem.address.tile",
                                                   tvm.tir.IntImm("int64", 0))
    codegen = tvm._ffi.get_global_func("target.build.tilelang_tpu")

    with pytest.raises(tvm.error.TVMError, match="descriptor cannot be bound"):
        codegen(
            tvm.IRModule({"native_descriptor_let": prim_func}),
            tvm.target.Target(_tpu_target()),
        )


def test_runtime_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported TPU runtime mode"):
        resolve_tpu_runtime(runtime_mode="soc")
    with pytest.raises(ValueError, match="Unsupported TPU runtime mode"):
        resolve_tpu_runtime(runtime_mode="")


def test_tpu_chip_capability_registry_is_read_only():
    with pytest.raises(TypeError):
        TPU_CHIP_SPECS["future-chip"] = get_tpu_chip_spec("bm1690")


def test_tpu_persistent_cache_save_and_load_are_disabled_without_a_manifest():
    cache = KernelCache()
    target = _tpu_target("bm1690e", "tpukernel")
    tpu_kernel = SimpleNamespace(target=tvm.target.Target(target))

    with pytest.raises(RuntimeError, match="manifest"):
        cache._save_kernel_to_disk("unsafe-tpu-artifact", tpu_kernel)
    with pytest.raises(RuntimeError, match="manifest"):
        cache._load_kernel_from_disk("unsafe-tpu-artifact", target=target)


def test_jit_kernel_resolves_canonical_tpu_identities_without_compiling():
    kernel = JITKernel(
        target=_tpu_target("bm1690e", "rv"),
        execution_backend="ctypes",
        from_database=True,
        runtime_mode="cmodel",
    )

    assert kernel.tpu_target == TPUTargetSpec("bm1690e", "rv")
    assert kernel.tpu_runtime == TPURuntimeConfig("cmodel")


def test_tpu_jit_rejects_dlpack_before_lowering(monkeypatch):

    def forbidden_lower(*_args, **_kwargs):
        raise AssertionError("TPU lowering must not run for an unsupported adapter")

    monkeypatch.setattr(tilelang, "lower", forbidden_lower)
    with pytest.raises(ValueError, match=r"TPU targets do not support execution_backend='dlpack'"):
        JITKernel(
            func=object(),
            target=_tpu_target("bm1690e", "rv"),
            execution_backend="dlpack",
            runtime_mode="cmodel",
        )


def test_non_tpu_jit_rejects_tpu_runtime_selection():
    with pytest.raises(ValueError, match="only valid for a TPU target"):
        JITKernel(
            target="llvm",
            execution_backend="ctypes",
            from_database=True,
            runtime_mode="cmodel",
        )


def test_non_tpu_lower_rejects_tpu_runtime_selection():
    prim_func = tvm.tir.PrimFunc(
        [],
        tvm.tir.Evaluate(0),
    ).with_attr("global_symbol", "runtime_axis_requires_tpu")

    with pytest.raises(ValueError, match="only valid for a TPU target"):
        tilelang.lower(prim_func, target="c", runtime_mode="cmodel")


def test_lower_keeps_non_tpu_targets_on_the_regular_codegen_path():
    A = tvm.tir.decl_buffer((4,), "float32", name="A")
    B = tvm.tir.decl_buffer((4,), "float32", name="B")
    i = tvm.tir.Var("i", "int32")
    body = tvm.tir.For(i, 0, 4, tvm.tir.ForKind.SERIAL,
                       tvm.tir.BufferStore(B, tvm.tir.BufferLoad(A, [i]), [i]))
    prim_func = tvm.tir.PrimFunc(
        [A.data, B.data],
        body,
        buffer_map={
            A.data: A,
            B.data: B
        },
    ).with_attr("global_symbol", "cpu_copy")

    artifact = tilelang.lower(prim_func, target="c")

    assert artifact.tpu_target is None
    assert artifact.tpu_runtime is None
    assert "cpu_copy" in artifact.kernel_source


@pytest.mark.parametrize("extern_name", ["ppl.fill", "tpu_sync_all_bdc", "rvt_fadd"])
def test_tpu_externs_cannot_silently_lower_for_a_non_tpu_target(extern_name):
    prim_func = tvm.tir.PrimFunc(
        [],
        tvm.tir.Evaluate(tvm.tir.call_extern("handle", extern_name)),
    ).with_attr("global_symbol", "must_select_tpu")

    with pytest.raises(ValueError, match="complete TPU target"):
        tilelang.lower(prim_func, target="c")


def test_tpu_externs_cannot_silently_lower_after_auto_selects_c(monkeypatch):
    prim_func = tvm.tir.PrimFunc(
        [],
        tvm.tir.Evaluate(tvm.tir.call_extern("handle", "ppl.fill")),
    ).with_attr("global_symbol", "auto_must_select_tpu")
    monkeypatch.setattr(lower_module, "determine_target", lambda target: tvm.target.Target("c"))

    with pytest.raises(ValueError, match="complete TPU target"):
        tilelang.lower(prim_func, target="auto")


def test_library_generator_receives_target_and_runtime_identities():
    target = tvm.target.Target(_tpu_target("bm1690e", "rv"))
    target_spec = TPUTargetSpec("bm1690e", "rv")
    runtime_config = TPURuntimeConfig("pcie")
    generator = LibraryGenerator(
        target,
        tpu_target=target_spec,
        tpu_runtime=runtime_config,
    )
    try:
        assert generator.tpu_target is target_spec
        assert generator.tpu_runtime is runtime_config
    finally:
        generator.remove_lib()


@pytest.mark.parametrize("backend,adapter_name", [
    ("ctypes", "CtypesKernelAdapter"),
    ("cython", "CythonKernelAdapter"),
])
def test_database_adapter_forwards_target_and_runtime_identities(monkeypatch, backend,
                                                                 adapter_name):
    captured = {}

    class FakeAdapter:

        @classmethod
        def from_database(cls, **kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(kernel_module, adapter_name, FakeAdapter)
    target = _tpu_target("bm1690e", "rv")
    kernel = JITKernel(
        target=target,
        execution_backend=backend,
        from_database=True,
        runtime_mode="cmodel",
    )

    adapter = kernel._create_adapter_from_database(
        params=[],
        result_idx=[],
        target=target,
        func_or_mod=object(),
        kernel_global_source="",
        kernel_lib_path="/tmp/tilelang-unused.so",
    )

    assert adapter is not None
    assert captured["tpu_target"] == TPUTargetSpec("bm1690e", "rv")
    assert captured["tpu_runtime"] == TPURuntimeConfig("cmodel")
