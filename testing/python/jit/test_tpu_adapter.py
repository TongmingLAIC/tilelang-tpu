# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import ctypes
import gc
import os
from pathlib import Path
import subprocess
import sys
import weakref

import pytest
import torch
import importlib

tpu_adapter = importlib.import_module("tilelang.jit.adapter.tpu")

from tilelang import tvm
import tilelang.language as T
from tilelang.engine.param import KernelParam
from tilelang.engine.tpu_config import TPURuntimeConfig, TPUTargetSpec
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.ctypes.adapter import CtypesKernelAdapter
from tilelang.jit.adapter.cython import adapter as cython_adapter
from tilelang.jit.adapter.cython.adapter import CythonKernelAdapter
from tilelang.jit.adapter.tpu import (
    make_tpu_forward,
    reject_unverified_tpu_database_artifact,
)
from tilelang.jit.adapter.wrapper import TLTPUSourceWrapper
from tilelang.temp_storage import user_temp_root


class _FakeRun:

    def __init__(self, status=0):
        self.status = status
        self.input_values = None

    def __call__(self, argv):
        self.input_values = list((ctypes.c_float * 3).from_address(argv[0]))
        if self.status == 0:
            ctypes.memmove(argv[1], argv[0], 3 * ctypes.sizeof(ctypes.c_float))
        return self.status


class _FakeLibrary:

    def __init__(self, status=0):
        self.tilelang_tpu_run = _FakeRun(status)


class _FakePPLLayout:

    def __init__(self, runtime_identity):
        self.runtime_identity = runtime_identity

    def runtime_identity_for(self, runtime_mode):
        return self.runtime_identity


_TEST_SDK_IDENTITY = ("/test/ppl", "/test/ppl/runtime", "/test/ppl/backend")


def test_cython_adapter_cache_is_below_external_tilelang_cache(monkeypatch, tmp_path):
    cache_root = tmp_path / "tilelang-cache"
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(cache_root))

    cache_dir = cython_adapter._cython_cache_root()

    assert cache_dir == cache_root.resolve() / "cython-adapter/v1"
    assert cache_root.is_dir()
    assert not cache_dir.is_relative_to(Path(cython_adapter.__file__).resolve().parent)


def test_importing_tpu_adapter_does_not_build_cython_wrapper(tmp_path):
    cache_root = tmp_path / "import-cache"
    environment = dict(os.environ)
    environment.update({
        "PATH": os.pathsep.join(("/usr/bin", "/bin")),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TILELANG_CACHE_DIR": str(cache_root),
    })

    subprocess.run(
        [
            sys.executable, "-c",
            "import tilelang; from tilelang.jit.adapter.cython import CythonKernelAdapter"
        ],
        check=True,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert not (cache_root / "cython-adapter").exists()


def test_cython_wrapper_build_uses_python_module_and_atomic_cache(monkeypatch, tmp_path):
    cache_root = tmp_path / "build-cache"
    commands = []

    class FakeWrapper:
        pass

    def fake_run(command, *, check):
        assert check is True
        commands.append([str(item) for item in command])
        output = Path(command[command.index("-o") + 1])
        output.write_bytes(b"generated")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setenv("TILELANG_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(cython_adapter, "get_cplus_compiler", lambda: "/usr/bin/g++")
    monkeypatch.setattr(cython_adapter.subprocess, "run", fake_run)
    monkeypatch.setattr(cython_adapter, "_load_cython_extension", lambda _library: FakeWrapper)

    assert cython_adapter._load_cython_kernel_wrapper() is FakeWrapper
    assert cython_adapter._load_cython_kernel_wrapper() is FakeWrapper
    assert commands[0][:3] == [sys.executable, "-m", "cython"]
    assert commands[1][0] == str(Path("/usr/bin/g++").resolve())
    assert len(commands) == 2
    assert len(list(cache_root.glob("cython-adapter/v1/*/manifest.json"))) == 1
    assert len(list(cache_root.glob("cython-adapter/v1/*/cython_wrapper.so"))) == 1
    assert not list(cache_root.rglob("*.tmp"))


def _tpu_target(chip="sg2260e", programming_model="tpukernel"):
    return tvm.target.Target(f"tpu -mcpu={chip} "
                             f"-tpu-programming-model={programming_model}")


def _library_generator(chip="sg2260e", programming_model="tpukernel", runtime_mode="cmodel"):
    return LibraryGenerator(
        _tpu_target(chip, programming_model),
        tpu_target=TPUTargetSpec(chip, programming_model),
        tpu_runtime=TPURuntimeConfig(runtime_mode),
    )


def _mark_verified_tpu_artifact(generator,
                                lib_path="/tmp/not-loaded.so",
                                sdk_identity=_TEST_SDK_IDENTITY):
    """Model a private artifact emitted by ``LibraryGenerator.compile_lib``.

    Unit tests must not resolve a real PPL SDK just to test the Python-side
    dlopen safety gates.  This intentionally sets both pieces of provenance
    that a successful TPU compilation captures.
    """
    generator._ppl_layout = _FakePPLLayout(sdk_identity)
    generator.libpath = lib_path
    generator._tpu_compiled_libpath = os.path.realpath(lib_path)
    generator._tpu_compiled_runtime_identity = sdk_identity


def _two_tensor_params():
    return [
        KernelParam(torch.float32, [3]),
        KernelParam(torch.float32, [3]),
    ]


@T.prim_func
def _two_buffer_wrapper_primfunc(
        A: T.Tensor((1,), "float32"),
        B: T.Tensor((1,), "float32"),
):
    T.func_attr({"global_symbol": "opaque_rvt", "tir.noalias": T.bool(True)})
    B[0] = A[0]


def test_tpu_forward_uses_data_ptr_and_allocates_output():
    lib = _FakeLibrary()
    forward = make_tpu_forward(lib, _two_tensor_params(), [1], {})
    base = torch.arange(10, dtype=torch.float32)
    source = base[2:5]  # contiguous, but its storage offset is non-zero

    result = forward(source)

    assert lib.tilelang_tpu_run.input_values == [2.0, 3.0, 4.0]
    assert torch.equal(result, source)


def test_tpu_forward_does_not_copy_outputs_after_failure():
    lib = _FakeLibrary(status=-7)
    forward = make_tpu_forward(lib, _two_tensor_params(), [1], {})
    output = torch.full((3,), -1.0)

    with pytest.raises(RuntimeError, match="status -7"):
        forward(torch.arange(3, dtype=torch.float32), output)

    assert torch.equal(output, torch.full((3,), -1.0))


def test_tpu_forward_copies_explicit_buffers_for_opaque_externs():
    """An opaque extern may mutate an argument without yielding result_idx."""
    lib = _FakeLibrary()
    forward = make_tpu_forward(lib, _two_tensor_params(), [], {})
    source = torch.tensor([4.0, 5.0, 6.0])
    destination = torch.full((3,), -1.0)

    assert forward(source, destination) == 0
    assert torch.equal(destination, source)


def test_tpu_forward_rejects_aliased_storage():
    forward = make_tpu_forward(_FakeLibrary(), _two_tensor_params(), [], {})
    storage = torch.arange(6, dtype=torch.float32)

    with pytest.raises(ValueError, match="aliased Tensor storage"):
        forward(storage[:3], storage[3:])


def test_raw_rvt_wrapper_copies_all_buffers_even_with_result_idx(tmp_path):
    """Raw RVT pointer effects are opaque to TIR's result-index analysis."""
    TLTPUSourceWrapper(
        scheduled_ir_module=tvm.IRModule({"opaque_rvt": _two_buffer_wrapper_primfunc}),
        source=("#define TILELANG_TPU_OPAQUE_RAW_RVT_ABI 1\n"
                "void opaque_rvt(void) { rvt_kernel_start(); }"),
        target=_tpu_target("sg2260e", "rv"),
        output_indices=[1],
        output_dir=str(tmp_path),
    )
    main_source = (tmp_path / "main.cpp").read_text(encoding="utf-8")
    assert main_source.count("tpuRtMemcpyD2S(") == 2


def test_tpu_host_wrapper_uses_positional_cpp_identifiers(tmp_path):
    """Duplicate or punctuation-bearing TIR hints cannot corrupt the host ABI."""

    first = tvm.tir.decl_buffer((1,), "float32", name="same-name")
    second = tvm.tir.decl_buffer((1,), "float32", name="same-name")
    prim_func = tvm.tir.PrimFunc(
        [first.data, second.data],
        tvm.tir.Evaluate(0),
        buffer_map={
            first.data: first,
            second.data: second
        },
    ).with_attr("global_symbol", "main_kernel")

    TLTPUSourceWrapper(
        scheduled_ir_module=tvm.IRModule({"main_kernel": prim_func}),
        source="void main_kernel(void) {}",
        target=_tpu_target(),
        output_dir=str(tmp_path),
    )

    main_source = (tmp_path / "main.cpp").read_text(encoding="utf-8")
    kernel_source = (tmp_path / "kernel.cpp").read_text(encoding="utf-8")
    assert "same-name" not in main_source
    assert "char* arg_0 = static_cast<char*>(args[0]);" in main_source
    assert "char* arg_1 = static_cast<char*>(args[1]);" in main_source
    assert "void *dev_arg_0 = nullptr;" in main_source
    assert "void *dev_arg_1 = nullptr;" in main_source
    assert (main_source.index("auto start =") < main_source.index("int rst = main_kernel(") <
            main_source.index("auto end ="))
    assert (kernel_source.index("tpuRtKernelLaunch(")
            < kernel_source.index("tpuRtStreamSynchronize(stream)"))


def test_tpu_host_wrapper_requires_explicit_output_directory():
    with pytest.raises(ValueError, match="explicit output directory"):
        TLTPUSourceWrapper(
            scheduled_ir_module=tvm.IRModule({"opaque_rvt": _two_buffer_wrapper_primfunc}),
            source="void opaque_rvt(void) {}",
            target=_tpu_target(),
        )


def test_tpu_workspace_is_private_per_generator():
    first = _library_generator(programming_model="rv")
    second = _library_generator(programming_model="rv")
    try:
        assert first.tpu_workspace_dir != second.tpu_workspace_dir
        assert Path(first.tpu_workspace_dir).is_dir()
        assert Path(second.tpu_workspace_dir).is_dir()
        assert Path(first.tpu_workspace_dir).parent == user_temp_root()
        assert Path(second.tpu_workspace_dir).parent == user_temp_root()
    finally:
        first.remove_lib()
        second.remove_lib()


def test_tpu_workspace_is_cleaned_when_generator_becomes_unreachable():
    generator = _library_generator()
    workspace = Path(generator.tpu_workspace_dir)
    generator_ref = weakref.ref(generator)

    del generator
    gc.collect()

    assert generator_ref() is None
    assert not workspace.exists()


def test_loaded_tpu_library_retains_its_private_workspace(monkeypatch):
    monkeypatch.setattr(tpu_adapter, "_TPU_RUNTIME_IDENTITY", None)
    generator = _library_generator()
    workspace = Path(generator.tpu_workspace_dir)
    lib_path = str(workspace / "main.so")
    _mark_verified_tpu_artifact(generator, lib_path=lib_path)

    class FakeLibrary:
        pass

    library = FakeLibrary()
    library.tilelang_tpu_bind_device = lambda _device_id: 0
    monkeypatch.setattr(ctypes, "CDLL", lambda _path, loaded_library=library: loaded_library)

    loaded = generator.load_lib(lib_path)
    generator_ref = weakref.ref(generator)
    del generator
    gc.collect()

    assert generator_ref() is not None
    assert workspace.is_dir()

    # Release the test double retained by monkeypatch as well as both local
    # handles.  The library-to-generator ownership edge should then disappear.
    monkeypatch.undo()
    del loaded
    del library
    gc.collect()

    assert generator_ref() is None
    assert not workspace.exists()


def test_library_generator_close_is_idempotent():
    generator = _library_generator()
    workspace = Path(generator.tpu_workspace_dir)

    generator.close()
    generator.close()

    assert generator.tpu_workspace_dir is None
    assert not workspace.exists()


@pytest.mark.parametrize("variable,wrong_value", [
    ("TILELANG_TPU_PROFILE_CHIP", "bm1690"),
    ("TILELANG_TPU_PROFILE_PROGRAMMING_MODEL", "rv"),
    ("TILELANG_TPU_PROFILE_RUNTIME_MODE", "pcie"),
])
def test_profile_build_session_validates_all_selection_axes(monkeypatch, variable, wrong_value):
    target = TPUTargetSpec("sg2260e", "tpukernel")
    runtime = TPURuntimeConfig("cmodel")
    monkeypatch.setenv("TILELANG_TPU_PROFILE_SESSION", "1")
    monkeypatch.setenv("TILELANG_TPU_PROFILE_CHIP", target.chip)
    monkeypatch.setenv("TILELANG_TPU_PROFILE_PROGRAMMING_MODEL", target.programming_model)
    monkeypatch.setenv("TILELANG_TPU_PROFILE_RUNTIME_MODE", runtime.runtime_mode)

    assert LibraryGenerator._tpu_profile_session(target, runtime)

    monkeypatch.setenv(variable, wrong_value)
    with pytest.raises(ValueError, match="identity disagrees"):
        LibraryGenerator._tpu_profile_session(target, runtime)


def test_tpu_runtime_identity_is_process_global(monkeypatch):
    monkeypatch.setattr(tpu_adapter, "_TPU_RUNTIME_IDENTITY", None)
    sg = TPUTargetSpec("sg2260e", "tpukernel")
    bm = TPUTargetSpec("bm1690", "tpukernel")
    cmodel = TPURuntimeConfig("cmodel")

    tpu_adapter.reserve_tpu_runtime_identity(
        sg, cmodel, device_id=0, sdk_identity=_TEST_SDK_IDENTITY)
    tpu_adapter.reserve_tpu_runtime_identity(
        sg, cmodel, device_id=0, sdk_identity=_TEST_SDK_IDENTITY)
    with pytest.raises(RuntimeError, match="fresh process"):
        tpu_adapter.reserve_tpu_runtime_identity(
            bm, cmodel, device_id=0, sdk_identity=_TEST_SDK_IDENTITY)
    with pytest.raises(RuntimeError, match="fresh process"):
        tpu_adapter.reserve_tpu_runtime_identity(
            sg, TPURuntimeConfig("pcie"), device_id=0, sdk_identity=_TEST_SDK_IDENTITY)
    with pytest.raises(RuntimeError, match="fresh process"):
        tpu_adapter.reserve_tpu_runtime_identity(
            sg,
            cmodel,
            device_id=0,
            sdk_identity=("/test/another-ppl", "/test/another/runtime", "/test/another/backend"))


def test_tpu_runtime_identity_rejects_boolean_device_id(monkeypatch):
    monkeypatch.setattr(tpu_adapter, "_TPU_RUNTIME_IDENTITY", None)
    with pytest.raises(ValueError, match="non-negative int"):
        tpu_adapter.reserve_tpu_runtime_identity(
            TPUTargetSpec("sg2260e", "tpukernel"),
            TPURuntimeConfig("cmodel"),
            device_id=True,
            sdk_identity=_TEST_SDK_IDENTITY,
        )


def test_pcie_library_load_is_fail_closed(monkeypatch):
    generator = _library_generator(programming_model="rv", runtime_mode="pcie")
    try:
        monkeypatch.delenv("TILELANG_TPU_ALLOW_PCIE_LOAD", raising=False)
        with pytest.raises(RuntimeError, match="TILELANG_TPU_ALLOW_PCIE_LOAD=1"):
            generator.load_lib("/tmp/not-loaded.so")
    finally:
        generator.remove_lib()


@pytest.mark.parametrize("device_id", [None, "", "-1", "not-a-device", str(2**31)])
def test_pcie_library_load_requires_a_valid_device_id_before_dlopen(monkeypatch, device_id):
    generator = _library_generator(runtime_mode="pcie")
    try:
        monkeypatch.setenv("TILELANG_TPU_ALLOW_PCIE_LOAD", "1")
        if device_id is None:
            monkeypatch.delenv("TILELANG_TPU_DEVICE_ID", raising=False)
        else:
            monkeypatch.setenv("TILELANG_TPU_DEVICE_ID", device_id)
        with pytest.raises(RuntimeError, match="TILELANG_TPU_DEVICE_ID"):
            generator.load_lib("/tmp/not-loaded.so")
    finally:
        generator.remove_lib()


def test_pcie_library_load_validates_device_id_before_calling_cdll(monkeypatch):
    monkeypatch.setattr(tpu_adapter, "_TPU_RUNTIME_IDENTITY", None)
    generator = _library_generator(runtime_mode="pcie")
    try:
        monkeypatch.setenv("TILELANG_TPU_ALLOW_PCIE_LOAD", "1")
        monkeypatch.setenv("TILELANG_TPU_DEVICE_ID", "0")
        _mark_verified_tpu_artifact(generator)
        captured = {}

        class FakeLibrary:
            pass

        fake_library = FakeLibrary()

        def bind_device(device_id):
            captured["device_id"] = device_id
            return 0

        fake_library.tilelang_tpu_bind_device = bind_device

        def fake_cdll(path):
            captured["path"] = path
            return fake_library

        monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
        assert generator.load_lib("/tmp/not-loaded.so") is fake_library
        assert captured["path"] == "/tmp/not-loaded.so"
        assert captured["device_id"] == 0
    finally:
        generator.remove_lib()


def test_cmodel_library_load_binds_device_zero_before_returning(monkeypatch):
    monkeypatch.setattr(tpu_adapter, "_TPU_RUNTIME_IDENTITY", None)
    generator = _library_generator(programming_model="rv")
    try:
        _mark_verified_tpu_artifact(generator)
        captured = {}

        class FakeLibrary:
            pass

        fake_library = FakeLibrary()

        def bind_device(device_id):
            captured["device_id"] = device_id
            return 0

        fake_library.tilelang_tpu_bind_device = bind_device
        monkeypatch.setattr(ctypes, "CDLL", lambda path: fake_library)
        assert generator.load_lib("/tmp/not-loaded.so") is fake_library
        assert captured["device_id"] == 0
    finally:
        generator.remove_lib()


def test_tpu_load_rejects_a_host_library_that_rejects_device_binding(monkeypatch):
    monkeypatch.setattr(tpu_adapter, "_TPU_RUNTIME_IDENTITY", None)
    generator = _library_generator()
    try:
        _mark_verified_tpu_artifact(generator)

        class FakeLibrary:
            pass

        fake_library = FakeLibrary()
        fake_library.tilelang_tpu_bind_device = lambda _device_id: -3
        monkeypatch.setattr(ctypes, "CDLL", lambda path: fake_library)
        with pytest.raises(RuntimeError, match="rejected the reserved runtime device"):
            generator.load_lib("/tmp/not-loaded.so")
    finally:
        generator.remove_lib()


def test_pcie_load_rejects_a_cmodel_runtime_identity_before_dlopen(monkeypatch):
    monkeypatch.setattr(tpu_adapter, "_TPU_RUNTIME_IDENTITY", None)
    tpu_adapter.reserve_tpu_runtime_identity(
        TPUTargetSpec("sg2260e", "tpukernel"),
        TPURuntimeConfig("cmodel"),
        device_id=0,
        sdk_identity=_TEST_SDK_IDENTITY)
    generator = _library_generator(runtime_mode="pcie")
    try:
        monkeypatch.setenv("TILELANG_TPU_ALLOW_PCIE_LOAD", "1")
        monkeypatch.setenv("TILELANG_TPU_DEVICE_ID", "0")
        _mark_verified_tpu_artifact(generator)
        loaded = []
        monkeypatch.setattr(ctypes, "CDLL", lambda path: loaded.append(path))
        with pytest.raises(RuntimeError, match="fresh process"):
            generator.load_lib("/tmp/not-loaded.so")
        assert loaded == []
    finally:
        generator.remove_lib()


def test_cmodel_load_rejects_a_pcie_runtime_identity_before_dlopen(monkeypatch):
    monkeypatch.setattr(tpu_adapter, "_TPU_RUNTIME_IDENTITY", None)
    tpu_adapter.reserve_tpu_runtime_identity(
        TPUTargetSpec("sg2260e", "tpukernel"),
        TPURuntimeConfig("pcie"),
        device_id=0,
        sdk_identity=_TEST_SDK_IDENTITY)
    generator = _library_generator()
    try:
        _mark_verified_tpu_artifact(generator)
        loaded = []
        monkeypatch.setattr(ctypes, "CDLL", lambda path: loaded.append(path))
        with pytest.raises(RuntimeError, match="fresh process"):
            generator.load_lib("/tmp/not-loaded.so")
        assert loaded == []
    finally:
        generator.remove_lib()


def test_tpu_load_rejects_sdk_identity_transition_before_dlopen(monkeypatch):
    monkeypatch.setattr(tpu_adapter, "_TPU_RUNTIME_IDENTITY", None)
    target_spec = TPUTargetSpec("sg2260e", "tpukernel")
    runtime_config = TPURuntimeConfig("cmodel")
    tpu_adapter.reserve_tpu_runtime_identity(
        target_spec, runtime_config, device_id=0, sdk_identity=_TEST_SDK_IDENTITY)
    generator = _library_generator()
    try:
        _mark_verified_tpu_artifact(
            generator,
            sdk_identity=("/test/other-ppl", "/test/other/runtime", "/test/other/backend"))
        loaded = []
        monkeypatch.setattr(ctypes, "CDLL", lambda path: loaded.append(path))
        with pytest.raises(RuntimeError, match="fresh process"):
            generator.load_lib("/tmp/not-loaded.so")
        assert loaded == []
    finally:
        generator.remove_lib()


def test_tpu_load_rejects_unverified_prebuilt_artifact_before_dlopen(monkeypatch):
    monkeypatch.setattr(tpu_adapter, "_TPU_RUNTIME_IDENTITY", None)
    generator = _library_generator()
    try:
        loaded = []
        monkeypatch.setattr(ctypes, "CDLL", lambda path: loaded.append(path))
        with pytest.raises(RuntimeError, match="verified manifest"):
            generator.load_lib("/tmp/not-loaded.so")
        assert loaded == []
    finally:
        generator.remove_lib()


def test_tpu_load_rejects_another_path_from_the_same_generator_before_dlopen(monkeypatch):
    monkeypatch.setattr(tpu_adapter, "_TPU_RUNTIME_IDENTITY", None)
    generator = _library_generator()
    try:
        _mark_verified_tpu_artifact(generator, lib_path="/tmp/compiled-main.so")
        loaded = []
        monkeypatch.setattr(ctypes, "CDLL", lambda path: loaded.append(path))
        with pytest.raises(RuntimeError, match="not produced by this LibraryGenerator"):
            generator.load_lib("/tmp/another-prebuilt-main.so")
        assert loaded == []
    finally:
        generator.remove_lib()


@pytest.mark.parametrize("adapter_cls", [CtypesKernelAdapter, CythonKernelAdapter])
def test_tpu_database_adapters_reject_prebuilt_artifacts_before_dlopen(monkeypatch, adapter_cls):
    raw_prim_func = tvm.tir.PrimFunc(
        [],
        tvm.tir.Evaluate(tvm.tir.const(0, "int32")),
    ).with_attr("global_symbol", "cached_tpu_artifact")
    loaded = []
    monkeypatch.setattr(ctypes, "CDLL", lambda path: loaded.append(path))

    with pytest.raises(RuntimeError, match="manifest"):
        adapter_cls.from_database(
            params=[],
            result_idx=[],
            target=_tpu_target("sg2260e", "tpukernel"),
            func_or_mod=raw_prim_func,
            kernel_global_source="",
            kernel_lib_path="/tmp/prebuilt-tpu-main.so",
            tpu_target=TPUTargetSpec("sg2260e", "tpukernel"),
            tpu_runtime=TPURuntimeConfig("cmodel"),
        )
    assert loaded == []


def test_tpu_database_artifact_loading_is_disabled_without_manifest():
    with pytest.raises(RuntimeError, match="manifest"):
        reject_unverified_tpu_database_artifact(_tpu_target("sg2260e", "tpukernel"))
