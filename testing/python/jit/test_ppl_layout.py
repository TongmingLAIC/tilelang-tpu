from pathlib import Path
import json

import pytest

from tilelang.jit.adapter.ppl_layout import resolve_ppl_layout


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _make_ppl_17_layout(tmp_path, chip="bm1690", arch="tpub_7_1"):
    (tmp_path / "deps/chip").mkdir(parents=True)
    (tmp_path / "deps/chip/chip_map.json").write_text(json.dumps({chip: arch}), encoding="utf-8")
    for directory in (
            f"deps/chip/{arch}/TPU1686/kernel/include",
            f"deps/chip/{arch}/TPU1686/tpuDNN/include",
            f"deps/chip/{arch}/lib",
            "deps/common/dev/kernel",
            "deps/common/dev/utils/include",
            "deps/common/host/include",
            "deps/runtime/tpuv7-runtime/include",
            "deps/runtime/tpuv7-runtime/lib",
    ):
        (tmp_path / directory).mkdir(parents=True)
    _touch(tmp_path / "deps/common/dev/utils/src/ppl_helper.c")
    _touch(tmp_path / "deps/runtime/tpuv7-runtime/lib/libtpuv7_rt.so")
    _touch(tmp_path / "deps/runtime/tpuv7-runtime/lib/libcdm_daemon_emulator.so")
    _touch(tmp_path / f"deps/chip/{arch}/lib/libtpuv7_emulator.so")
    _touch(tmp_path / f"deps/chip/{arch}/lib/libfirmware_core.a")
    _touch(tmp_path / f"deps/chip/{arch}/lib/libtpudnn.so")
    (tmp_path / f"deps/chip/{arch}/config_common.cmake").write_text(
        f"add_definitions(-D__{chip}__)\n", encoding="utf-8")


def test_resolve_ppl_17_layout(tmp_path):
    arch = "tpub_7_1"
    _make_ppl_17_layout(tmp_path, arch=arch)

    layout = resolve_ppl_layout(str(tmp_path), "bm1690")

    assert layout.arch == arch
    assert layout.physical_core_count == 8
    assert layout.compile_definitions == ("__tpub_7_1__", "__sg2260__", "__bm1690__")
    assert layout.firmware_archive == (tmp_path / f"deps/chip/{arch}/lib/libfirmware_core.a")
    assert layout.tpudnn_include == (tmp_path / f"deps/chip/{arch}/TPU1686/tpuDNN/include")
    assert layout.tpudnn_library == (tmp_path / f"deps/chip/{arch}/lib/libtpudnn.so")
    assert layout.root == tmp_path.resolve()
    assert layout.runtime_identity_for("cmodel") == (
        str(tmp_path.resolve()),
        str((tmp_path / "deps/runtime/tpuv7-runtime/lib").resolve()),
        str((tmp_path / f"deps/chip/{arch}/lib").resolve()),
    )


def test_resolve_bm1690e_core_count(tmp_path):
    arch = "tpub_7_1_e"
    _make_ppl_17_layout(tmp_path, chip="bm1690e", arch=arch)

    layout = resolve_ppl_layout(str(tmp_path), "bm1690e")

    assert layout.arch == arch
    assert layout.physical_core_count == 4
    assert layout.compile_definitions == ("__tpub_7_1_e__", "__bm1690e__")


def test_vendor_macros_come_from_the_sdk_without_changing_target_identity(tmp_path):
    arch = "tpub_7_1_e"
    _make_ppl_17_layout(tmp_path, chip="bm1690e", arch=arch)
    configuration = tmp_path / f"deps/chip/{arch}/config_common.cmake"
    configuration.write_text("add_definitions(-D__vendor_device_e__)\n", encoding="utf-8")

    layout = resolve_ppl_layout(str(tmp_path), "BM1690E")

    assert layout.chip == "bm1690e"
    assert layout.compile_definitions == (
        "__tpub_7_1_e__", "__bm1690e__", "__vendor_device_e__")
    configuration.write_text("add_definitions(-D__replacement_vendor_macro__)\n", encoding="utf-8")
    assert layout.compile_definitions[-1] == "__replacement_vendor_macro__"


def test_unsupported_sdk_compile_flags_are_rejected(tmp_path):
    _make_ppl_17_layout(tmp_path)
    configuration = tmp_path / "deps/chip/tpub_7_1/config_common.cmake"
    configuration.write_text("add_definitions(-include unexpected.h)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported PPL compile definition"):
        _ = resolve_ppl_layout(str(tmp_path), "bm1690").compile_definitions


def test_rvt_api_is_required_for_rv_mode(tmp_path):
    arch = "tpub_7_1_e"
    _make_ppl_17_layout(tmp_path, chip="bm1690e", arch=arch)
    layout = resolve_ppl_layout(str(tmp_path), "bm1690e")

    with pytest.raises(FileNotFoundError, match="rvt_api.h"):
        layout.require_rvt_api()

    _touch(tmp_path / f"deps/chip/{arch}/TPU1686/kernel/include/rvt_api.h")
    assert layout.require_rvt_api() == layout.rvt_api_header


def test_rejects_missing_required_chip_map(tmp_path):
    with pytest.raises(FileNotFoundError, match="expected chip map"):
        resolve_ppl_layout(str(tmp_path), "bm1690")


def test_base_resolution_does_not_require_runtime_or_profiling_artifacts(tmp_path):
    _make_ppl_17_layout(tmp_path)
    for relative in ("deps/runtime/tpuv7-runtime/lib/libtpuv7_rt.so",
                     "deps/runtime/tpuv7-runtime/lib/libcdm_daemon_emulator.so",
                     "deps/chip/tpub_7_1/lib/libtpuv7_emulator.so",
                     "deps/chip/tpub_7_1/lib/libfirmware_core.a",
                     "deps/chip/tpub_7_1/lib/libtpudnn.so"):
        (tmp_path / relative).unlink()
    for relative in ("deps/common/host/include", "deps/runtime/tpuv7-runtime/include",
                     "deps/runtime/tpuv7-runtime/lib", "deps/chip/tpub_7_1/TPU1686/tpuDNN/include",
                     "deps/chip/tpub_7_1/lib"):
        (tmp_path / relative).rmdir()

    layout = resolve_ppl_layout(str(tmp_path), "bm1690")

    assert layout.arch == "tpub_7_1"


def test_cmodel_requires_emulator_but_not_firmware_or_tpudnn(tmp_path):
    _make_ppl_17_layout(tmp_path)
    (tmp_path / "deps/chip/tpub_7_1/lib/libfirmware_core.a").unlink()
    (tmp_path / "deps/chip/tpub_7_1/lib/libtpudnn.so").unlink()
    (tmp_path / "deps/chip/tpub_7_1/TPU1686/tpuDNN/include").rmdir()
    layout = resolve_ppl_layout(str(tmp_path), "bm1690")

    assert layout.require_runtime("cmodel") is layout
    assert layout.require_profiling("cmodel") is layout

    layout.emulator_library.unlink()
    with pytest.raises(FileNotFoundError, match="TPUv7 emulator"):
        layout.require_runtime("cmodel")


def test_pcie_requires_firmware_but_not_emulator_or_tpudnn(tmp_path, monkeypatch):
    _make_ppl_17_layout(tmp_path)
    gcc = tmp_path / ("third_party/toolchains_dir/release/bin/"
                      "riscv64-unknown-linux-gnu-gcc")
    _touch(gcc)
    board_runtime = tmp_path / "installed-board-runtime/lib"
    _touch(board_runtime / "libtpuv7_rt.so")
    monkeypatch.setenv("TILELANG_TPU_PCIE_RUNTIME_PATH", str(board_runtime))
    (tmp_path / "deps/chip/tpub_7_1/lib/libtpuv7_emulator.so").unlink()
    (tmp_path / "deps/chip/tpub_7_1/lib/libtpudnn.so").unlink()
    (tmp_path / "deps/chip/tpub_7_1/TPU1686/tpuDNN/include").rmdir()
    layout = resolve_ppl_layout(str(tmp_path), "bm1690")

    assert layout.require_runtime("pcie") is layout

    layout.firmware_archive.unlink()
    with pytest.raises(FileNotFoundError, match="firmware archive"):
        layout.require_runtime("pcie")


def test_only_pcie_profiling_requires_tpudnn(tmp_path, monkeypatch):
    _make_ppl_17_layout(tmp_path)
    gcc = tmp_path / ("third_party/toolchains_dir/release/bin/"
                      "riscv64-unknown-linux-gnu-gcc")
    _touch(gcc)
    board_runtime = tmp_path / "installed-board-runtime/lib"
    _touch(board_runtime / "libtpuv7_rt.so")
    monkeypatch.setenv("TILELANG_TPU_PCIE_RUNTIME_PATH", str(board_runtime))
    layout = resolve_ppl_layout(str(tmp_path), "bm1690")
    layout.tpudnn_library.unlink()

    assert layout.require_runtime("pcie") is layout
    with pytest.raises(FileNotFoundError, match="TPUDNN profiling library"):
        layout.require_profiling("pcie")

    _touch(layout.tpudnn_library)
    assert layout.require_profiling("pcie") is layout


def test_rejects_unknown_chip_in_ppl_17_layout(tmp_path):
    _make_ppl_17_layout(tmp_path)

    with pytest.raises(ValueError, match="bm1690e"):
        resolve_ppl_layout(str(tmp_path), "bm1690e")


def test_rejects_chip_map_that_disagrees_with_the_capability_registry(tmp_path):
    _make_ppl_17_layout(tmp_path, chip="bm1690e", arch="tpub_7_1")

    with pytest.raises(ValueError, match="disagrees with TileLang's validated capability"):
        resolve_ppl_layout(str(tmp_path), "bm1690e")


def test_pcie_cross_compiler_is_discovered_without_a_pinned_sdk_version(tmp_path):
    _make_ppl_17_layout(tmp_path, chip="bm1690e", arch="tpub_7_1_e")
    gcc = tmp_path / ("third_party/toolchains_dir/any-ppl-release/bin/"
                      "riscv64-unknown-linux-gnu-gcc")
    _touch(gcc)

    layout = resolve_ppl_layout(str(tmp_path), "bm1690e")
    assert layout.pcie_cross_gcc() == gcc


def test_pcie_cross_compiler_rejects_ambiguous_sdk_toolchains(tmp_path):
    _make_ppl_17_layout(tmp_path)
    for release in ("first", "second"):
        _touch(tmp_path / (f"third_party/toolchains_dir/{release}/bin/"
                           "riscv64-unknown-linux-gnu-gcc"))

    layout = resolve_ppl_layout(str(tmp_path), "bm1690")
    with pytest.raises(ValueError, match="selection is ambiguous"):
        layout.pcie_cross_gcc()


def test_pcie_runtime_is_separate_from_the_sdk_cmodel_runtime(tmp_path, monkeypatch):
    _make_ppl_17_layout(tmp_path)
    layout = resolve_ppl_layout(str(tmp_path), "bm1690")
    board_runtime = tmp_path / "installed-board-runtime/lib"
    _touch(board_runtime / "libtpuv7_rt.so")
    monkeypatch.setenv("TILELANG_TPU_PCIE_RUNTIME_PATH", str(board_runtime))

    assert layout.pcie_runtime_lib() == board_runtime.resolve()
    assert layout.runtime_identity_for("pcie") == (
        str(tmp_path.resolve()),
        str(board_runtime.resolve()),
        str((tmp_path / "deps/chip/tpub_7_1/lib").resolve()),
    )

    monkeypatch.setenv("TILELANG_TPU_PCIE_RUNTIME_PATH", str(layout.runtime_lib))
    with pytest.raises(ValueError, match="SDK CModel runtime"):
        layout.pcie_runtime_lib()
