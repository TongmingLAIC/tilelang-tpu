# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Filesystem-only tests for content-addressed TPU toolchain identity."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from tilelang.jit.adapter import tpu_toolchain_identity as identity_module
from tilelang.jit.adapter.tpu_toolchain_identity import (
    TPUToolchainIdentityError,
    capture_tpu_toolchain_identity,
    file_content_identity,
    tree_content_identity,
)


def _write(path: Path, content: bytes, *, executable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_file_identity_is_stable_and_content_sensitive(tmp_path):
    artifact = _write(tmp_path / "libexample.so", b"version-one")

    first = file_content_identity(artifact)
    second = file_content_identity(artifact)
    assert first == second
    assert len(first["sha256"]) == 64

    artifact.write_bytes(b"version-two")
    changed = file_content_identity(artifact)
    assert changed["sha256"] != first["sha256"]


def test_file_identity_records_and_validates_a_symlink_target(tmp_path):
    target = _write(tmp_path / "actual.so", b"binary")
    alias = tmp_path / "current.so"
    alias.symlink_to(target.name)

    observed = file_content_identity(alias)

    assert observed["path"] == str(alias)
    assert observed["resolved_path"] == str(target)


@pytest.mark.parametrize("kind", ("missing", "directory", "fifo"))
def test_file_identity_rejects_missing_or_non_regular_paths(tmp_path, kind):
    path = tmp_path / kind
    if kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)

    with pytest.raises(TPUToolchainIdentityError):
        file_content_identity(path)


def test_file_identity_rejects_a_change_during_hashing(tmp_path, monkeypatch):
    artifact = _write(tmp_path / "changing.so", b"before")
    original = identity_module._hash_open_file

    def mutate_after_read(file_object):
        digest = original(file_object)
        artifact.write_bytes(b"after-with-different-size")
        return digest

    monkeypatch.setattr(identity_module, "_hash_open_file", mutate_after_read)

    with pytest.raises(TPUToolchainIdentityError, match="changed while hashing"):
        file_content_identity(artifact)


def test_tree_identity_covers_paths_symlinks_and_contents(tmp_path):
    tree = tmp_path / "include"
    target = _write(tree / "detail/value.h", b"#define VALUE 1\n")
    (tree / "value.h").symlink_to(target.relative_to(tree))

    first = tree_content_identity(tree)
    second = tree_content_identity(tree)
    assert first == second
    assert first["file_count"] == 2
    assert json.loads(json.dumps(first)) == first

    target.write_bytes(b"#define VALUE 2\n")
    assert tree_content_identity(tree)["sha256"] != first["sha256"]


def test_tree_identity_rejects_special_entries(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    os.mkfifo(tree / "pipe")

    with pytest.raises(TPUToolchainIdentityError, match="non-regular"):
        tree_content_identity(tree)


def _fake_toolchain(tmp_path: Path):
    root = tmp_path / "ppl"
    _write(root / "deps/chip/chip_map.json", b'{"bm1690":"bm","bm1690e":"sg"}')
    common = root / "deps/common"
    runtime = root / "deps/runtime/tpuv7-runtime/lib"
    helper = _write(common / "dev/utils/src/ppl_helper.c", b"helper")
    common_directories = {
        "kernel_common_include": common / "dev/kernel",
        "device_utils_include": common / "dev/utils/include",
        "host_include": common / "host/include",
        "runtime_include": root / "deps/runtime/tpuv7-runtime/include",
    }
    for index, directory in enumerate(common_directories.values()):
        _write(directory / "api.h", f"common-{index}".encode())
    _write(runtime / "libtpuv7_rt.so", b"cmodel-rt")
    _write(runtime / "libcdm_daemon_emulator.so", b"daemon")

    cross_root = root / "third_party/toolchains_dir/cross"
    cross_gcc = _write(cross_root / "bin/riscv64-unknown-linux-gnu-gcc", b"gcc", executable=True)
    _write(cross_root / "libexec/gcc/cc1", b"cc1")
    _write(cross_root / "sysroot/lib/libc.so", b"libc")

    pcie_runtime = tmp_path / "installed/lib"
    _write(pcie_runtime / "libtpuv7_rt.so", b"pcie-rt")
    tpu_smi = _write(tmp_path / "installed/bin/tpu-smi", b"smi", executable=True)

    layouts = {}
    for chip, arch, cores in (("bm1690", "bm", 8), ("bm1690e", "sg", 4)):
        chip_root = root / "deps/chip" / arch
        _write(chip_root / "config_common.cmake", f"add_definitions(-DARCH_{arch.upper()})\n".encode())
        kernel_include = chip_root / "TPU1686/kernel/include"
        backend = chip_root / "lib"
        tpudnn_include = chip_root / "TPU1686/tpuDNN/include"
        _write(kernel_include / "kernel.h", chip.encode())
        emulator = _write(backend / "libtpuv7_emulator.so", b"emulator-" + chip.encode())
        firmware = _write(backend / "libfirmware_core.a", b"firmware-" + chip.encode())
        tpudnn = _write(backend / "libtpudnn.so", b"tpudnn-" + chip.encode())
        _write(tpudnn_include / "tpudnn.h", chip.encode())

        def require_runtime(_mode, environment=None):
            del environment

        def require_profiling(_mode, environment=None):
            del environment

        layouts[chip] = SimpleNamespace(
            arch=arch,
            physical_core_count=cores,
            compile_definitions=(f"ARCH_{arch.upper()}",),
            kernel_common_include=common_directories["kernel_common_include"],
            device_utils_include=common_directories["device_utils_include"],
            host_include=common_directories["host_include"],
            runtime_include=common_directories["runtime_include"],
            ppl_helper_source=helper,
            runtime_lib=runtime,
            kernel_include=kernel_include,
            backend_lib=backend,
            emulator_library=emulator,
            firmware_archive=firmware,
            tpudnn_include=tpudnn_include,
            tpudnn_library=tpudnn,
            require_runtime=require_runtime,
            require_profiling=require_profiling,
            pcie_cross_gcc=lambda path=cross_gcc: path,
            pcie_runtime_lib=lambda environment=None, path=pcie_runtime: path,
        )

    compiler = _write(tmp_path / "host/bin/cc", b"host-cc", executable=True)
    compiler_cxx = _write(tmp_path / "host/bin/c++", b"host-cxx", executable=True)
    tilelang = _write(tmp_path / "build/libtilelang.so", b"tilelang")
    tvm = _write(tmp_path / "build/libtvm.so", b"tvm")
    return SimpleNamespace(
        root=root,
        layouts=layouts,
        cross_root=cross_root,
        pcie_runtime=pcie_runtime,
        tpu_smi=tpu_smi,
        compiler=compiler,
        compiler_cxx=compiler_cxx,
        tilelang=tilelang,
        tvm=tvm,
    )


def _capture(fake, runtime_mode):
    return capture_tpu_toolchain_identity(
        fake.root,
        runtime_mode,
        environment={"TILELANG_TPU_PCIE_RUNTIME_PATH": str(fake.pcie_runtime)},
        tilelang_library=fake.tilelang,
        tvm_library=fake.tvm,
        host_c_compiler=fake.compiler,
        host_cxx_compiler=fake.compiler_cxx,
        tpu_smi=fake.tpu_smi,
    )


def test_capture_cmodel_identity_covers_loaded_libraries_and_both_chips(tmp_path, monkeypatch):
    fake = _fake_toolchain(tmp_path)
    monkeypatch.setattr(identity_module, "resolve_ppl_layout",
                        lambda _root, chip: fake.layouts[chip])

    first = _capture(fake, "cmodel")
    second = _capture(fake, "cmodel")

    assert first == second
    assert set(first["chips"]) == {"bm1690", "bm1690e"}
    assert first["compiler_runtime"]["tilelang_library"]["sha256"]
    assert first["compiler_runtime"]["tvm_library"]["sha256"]
    assert first["ppl_common"]["ppl_helper"]["sha256"]
    assert first["cmodel"]["runtime_tree"]["file_count"] == 2
    assert "pcie" not in first
    assert json.loads(json.dumps(first)) == first


def test_capture_pcie_identity_covers_cross_toolchain_firmware_tpudnn_runtime_and_smi(
        tmp_path, monkeypatch):
    fake = _fake_toolchain(tmp_path)
    monkeypatch.setattr(identity_module, "resolve_ppl_layout",
                        lambda _root, chip: fake.layouts[chip])

    first = _capture(fake, "pcie")
    assert first["pcie"]["cross_toolchain_tree"]["file_count"] == 3
    assert first["pcie"]["installed_runtime_library"]["sha256"]
    assert first["pcie"]["tpu_smi"]["sha256"]
    for chip in ("bm1690", "bm1690e"):
        assert first["pcie"]["chips"][chip]["firmware_archive"]["sha256"]
        assert first["pcie"]["chips"][chip]["tpudnn_library"]["sha256"]

    (fake.cross_root / "sysroot/lib/libc.so").write_bytes(b"changed-libc")
    second = _capture(fake, "pcie")
    assert (second["pcie"]["cross_toolchain_tree"]["sha256"]
            != first["pcie"]["cross_toolchain_tree"]["sha256"])


def test_capture_identity_covers_sdk_compile_configuration(tmp_path, monkeypatch):
    fake = _fake_toolchain(tmp_path)
    monkeypatch.setattr(identity_module, "resolve_ppl_layout",
                        lambda _root, chip: fake.layouts[chip])
    first = _capture(fake, "cmodel")
    configuration = fake.root / "deps/chip/sg/config_common.cmake"
    configuration.write_text("add_definitions(-DUPDATED_VENDOR_MACRO)\n")

    second = _capture(fake, "cmodel")

    assert (first["chips"]["bm1690e"]["compile_configuration"]["sha256"]
            != second["chips"]["bm1690e"]["compile_configuration"]["sha256"])


def test_capture_rejects_a_component_changed_after_its_own_hash(tmp_path, monkeypatch):
    fake = _fake_toolchain(tmp_path)
    monkeypatch.setattr(identity_module, "resolve_ppl_layout",
                        lambda _root, chip: fake.layouts[chip])
    original = identity_module._capture_stability_snapshot
    calls = 0

    def mutate_before_final_snapshot(files, trees):
        nonlocal calls
        calls += 1
        if calls == 2:
            fake.tilelang.write_bytes(b"tilelang-changed-after-file-hash")
        return original(files, trees)

    monkeypatch.setattr(identity_module, "_capture_stability_snapshot",
                        mutate_before_final_snapshot)

    with pytest.raises(TPUToolchainIdentityError, match="complete identity capture"):
        _capture(fake, "cmodel")
