# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Resolve paths in the supported PPL 1.7 SDK release layout."""

from dataclasses import dataclass
import json
import os
import re
from pathlib import Path
from typing import Mapping, Optional, Tuple

from tilelang.engine.tpu_config import get_tpu_chip_spec


@dataclass(frozen=True)
class PPLLayout:
    root: Path
    chip: str
    kernel_include: Path
    kernel_common_include: Path
    device_utils_include: Path
    host_include: Path
    runtime_include: Path
    tpudnn_include: Path
    runtime_lib: Path
    backend_lib: Path
    tpudnn_library: Path
    ppl_helper_source: Path
    emulator_library: Path
    firmware_archive: Path
    toolchains_root: Path

    @property
    def arch(self) -> str:
        """PPL architecture selected by the canonical chip capability table."""

        return get_tpu_chip_spec(self.chip).ppl_arch

    @property
    def physical_core_count(self) -> int:
        """Physical cores exposed by this chip's CModel runtime."""

        return get_tpu_chip_spec(self.chip).physical_core_count

    @property
    def compile_definitions(self) -> Tuple[str, ...]:
        """Combine canonical target macros with the installed SDK's definitions.

        Vendor header macros can retain SDK-specific chip names. Read those
        names from the SDK's build configuration rather than exposing them as
        TileLang target identities.
        """

        definitions = list(get_tpu_chip_spec(self.chip).ppl_compile_definitions)
        configuration = self.root / "deps/chip" / self.arch / "config_common.cmake"
        source = configuration.read_text(encoding="utf-8")
        for declaration in re.findall(r"^\s*add_definitions\(([^)]*)\)", source, re.MULTILINE):
            for flag in declaration.split():
                if not re.fullmatch(r"-D[A-Za-z_]\w*", flag):
                    raise ValueError(f"Unsupported PPL compile definition {flag!r} in {configuration}")
                if flag[2:] not in definitions:
                    definitions.append(flag[2:])
        return tuple(definitions)

    def pcie_runtime_lib(self, environment: Optional[Mapping[str, str]] = None) -> Path:
        """Resolve the installed board runtime, never the SDK CModel runtime.

        PPL 1.7 compiles against headers in ``deps/`` but prepends the installed
        ``/opt/tpuv7/tpuv7-current/lib`` for PCIe execution.  The SDK's own
        ``deps/runtime/tpuv7-runtime/lib/libtpuv7_rt.so`` depends on
        ``libcdm_daemon_emulator.so`` and therefore cannot drive the board.
        An explicit override supports non-standard driver installations.
        """

        source = os.environ if environment is None else environment
        raw_path = source.get(
            "TILELANG_TPU_PCIE_RUNTIME_PATH",
            "/opt/tpuv7/tpuv7-current/lib",
        )
        runtime_lib = Path(raw_path).expanduser().resolve()
        if runtime_lib == self.runtime_lib.resolve():
            raise ValueError("TILELANG_TPU_PCIE_RUNTIME_PATH resolves to PPL's SDK CModel "
                             "runtime. PCIe must use the installed TPUv7 board runtime.")
        runtime_so = runtime_lib / "libtpuv7_rt.so"
        if not runtime_so.is_file():
            raise FileNotFoundError("TPUv7 PCIe board runtime is missing; expected "
                                    f"{runtime_so}. Install the matching TPUv7 driver runtime or "
                                    "set TILELANG_TPU_PCIE_RUNTIME_PATH to its lib directory.")
        return runtime_lib

    def runtime_identity_for(
        self,
        runtime_mode: str,
        environment: Optional[Mapping[str, str]] = None,
    ) -> Tuple[str, str, str]:
        """Return the SDK/runtime identity for one validated host mode."""

        if runtime_mode == "cmodel":
            runtime_lib = self.runtime_lib
        elif runtime_mode == "pcie":
            runtime_lib = self.pcie_runtime_lib(environment)
        else:
            raise ValueError(f"Unsupported TPU runtime mode: {runtime_mode!r}")
        return tuple(str(path.resolve()) for path in (self.root, runtime_lib, self.backend_lib))

    @property
    def include_dirs(self) -> Tuple[Path, ...]:
        """Base PPL include directories, excluding optional profiling APIs."""

        return (
            self.kernel_include,
            self.kernel_common_include,
            self.device_utils_include,
        )

    def include_dirs_for(
        self,
        runtime_mode: str,
        *,
        profiling: bool = False,
        environment: Optional[Mapping[str, str]] = None,
    ) -> Tuple[Path, ...]:
        """Return validated includes for one concrete build configuration."""

        self.require_runtime(runtime_mode, environment=environment)
        includes = self.include_dirs + (self.host_include, self.runtime_include)
        if profiling:
            self.require_profiling(runtime_mode, environment=environment)
            if runtime_mode == "pcie":
                includes += (self.tpudnn_include,)
        return includes

    def require_base(self) -> "PPLLayout":
        """Validate files shared by every PPL 1.7 device/host compilation."""

        return _require_paths(
            self,
            "base compilation",
            {
                "kernel headers": self.kernel_include,
                "common kernel headers": self.kernel_common_include,
                "device helper headers": self.device_utils_include,
                "ppl_helper.c": self.ppl_helper_source,
            },
        )

    def require_runtime(
        self,
        runtime_mode: str,
        *,
        environment: Optional[Mapping[str, str]] = None,
    ) -> "PPLLayout":
        """Validate only the artifacts needed by one host runtime mode."""

        self.require_base()
        if runtime_mode not in ("cmodel", "pcie"):
            raise ValueError(f"Unsupported TPU runtime mode: {runtime_mode!r}")
        _require_paths(
            self,
            f"{runtime_mode} host compilation",
            {
                "host headers": self.host_include,
                "TPUv7 runtime headers": self.runtime_include,
            },
        )
        if runtime_mode == "cmodel":
            return _require_paths(
                self,
                "CModel runtime",
                {
                    "TPUv7 CModel runtime": self.runtime_lib / "libtpuv7_rt.so",
                    "CModel daemon": self.runtime_lib / "libcdm_daemon_emulator.so",
                    "TPUv7 emulator": self.emulator_library,
                },
            )
        if runtime_mode == "pcie":
            # These resolvers also reject an SDK emulator runtime and an
            # ambiguous/missing cross-toolchain before compilation starts.
            self.pcie_runtime_lib(environment)
            self.pcie_cross_gcc()
            return _require_paths(
                self,
                "PCIe runtime",
                {"firmware archive": self.firmware_archive},
            )
        raise AssertionError("validated TPU runtime mode was not handled")

    def require_profiling(
        self,
        runtime_mode: str,
        *,
        environment: Optional[Mapping[str, str]] = None,
    ) -> "PPLLayout":
        """Validate extra vendor artifacts used by instruction profiling.

        CModel tracing is enabled by ``FILE_DUMP_CMD`` and needs no TPUDNN
        dependency.  PCIe recording is implemented through TPUDNN and thus
        validates its header and library only for an actual PCIe profile.
        """

        self.require_runtime(runtime_mode, environment=environment)
        if runtime_mode == "cmodel":
            return self
        if runtime_mode == "pcie":
            return _require_paths(
                self,
                "PCIe profiling",
                {
                    "TPUDNN profiling headers": self.tpudnn_include,
                    "TPUDNN profiling library": self.tpudnn_library,
                },
            )
        # ``require_runtime`` rejects this first; retain a local guard so this
        # method stays correct if its implementation changes later.
        raise ValueError(f"Unsupported TPU runtime mode: {runtime_mode!r}")

    @property
    def rvt_api_header(self) -> Path:
        """The PPL 1.7 RV Tensor ABI header for this chip architecture."""
        return self.kernel_include / "rvt_api.h"

    def require_rvt_api(self) -> Path:
        """Return the RVT header or explain why the RV target is invalid."""
        spec = get_tpu_chip_spec(self.chip)
        if not spec.supports("rv"):
            raise ValueError(f"TPU chip {spec.name!r} does not support the RV programming model")
        header = self.rvt_api_header
        if not header.is_file():
            raise FileNotFoundError(
                "PPL 1.7 RV target requested, but this chip SDK does not provide "
                f"rvt_api.h: {header}")
        return header

    def pcie_cross_gcc(self) -> Path:
        """Find the one PPL-provided PCIe compiler without pinning an SDK version.

        CModel users do not need a cross compiler, so discovery is intentionally
        delayed until a PCIe build.  More than one candidate is an ambiguity,
        not a reason to silently select an arbitrary SDK revision.
        """
        candidates = tuple(sorted(self.toolchains_root.glob("*/bin/riscv64-unknown-linux-gnu-gcc")))
        if not candidates:
            raise FileNotFoundError("PPL PCIe cross compiler is missing; expected "
                                    f"riscv64-unknown-linux-gnu-gcc under {self.toolchains_root}")
        if len(candidates) != 1:
            rendered = "\n  ".join(str(candidate) for candidate in candidates)
            raise ValueError(
                "PPL PCIe cross compiler selection is ambiguous; expected one candidate:\n  " +
                rendered)
        return candidates[0]


def _require_paths(layout: PPLLayout, requirement_group: str, required: Mapping[str,
                                                                                Path]) -> PPLLayout:
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete PPL 1.7 {requirement_group} requirements:\n  " +
                                "\n  ".join(missing))
    return layout


def resolve_ppl_layout(ppl_root: str, chip: str) -> PPLLayout:
    """Return the PPL 1.7 toolchain paths for one physical ``chip``.

    TileLang-TPU deliberately supports only the PPL 1.7 ``deps/`` release
    layout. Keeping one layout avoids silently compiling a kernel with an
    inconsistent header/library mixture.
    """
    chip_spec = get_tpu_chip_spec(chip)
    chip = chip_spec.name
    root = Path(ppl_root).expanduser().resolve()
    chip_map_path = root / "deps/chip/chip_map.json"
    if not chip_map_path.is_file():
        raise FileNotFoundError("PPL 1.7 SDK layout is required; expected chip map at "
                                f"{chip_map_path}.")

    try:
        chip_map = json.loads(chip_map_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid PPL 1.7 chip map: {chip_map_path}") from exc
    if not isinstance(chip_map, dict) or chip not in chip_map:
        raise ValueError(f"Chip {chip!r} is not present in {chip_map_path}")

    arch = chip_map[chip]
    if not isinstance(arch, str):
        raise ValueError(f"Invalid architecture for chip {chip!r} in {chip_map_path}")
    if arch != chip_spec.ppl_arch:
        raise ValueError("PPL SDK chip map disagrees with TileLang's validated capability "
                         f"for {chip!r}: expected {chip_spec.ppl_arch!r}, got {arch!r}")

    chip_root = root / "deps/chip" / arch
    runtime_root = root / "deps/runtime/tpuv7-runtime"
    common_root = root / "deps/common"
    return PPLLayout(
        root=root,
        chip=chip,
        kernel_include=chip_root / "TPU1686/kernel/include",
        kernel_common_include=common_root / "dev/kernel",
        device_utils_include=common_root / "dev/utils/include",
        host_include=common_root / "host/include",
        runtime_include=runtime_root / "include",
        tpudnn_include=chip_root / "TPU1686/tpuDNN/include",
        runtime_lib=runtime_root / "lib",
        backend_lib=chip_root / "lib",
        tpudnn_library=chip_root / "lib/libtpudnn.so",
        ppl_helper_source=common_root / "dev/utils/src/ppl_helper.c",
        emulator_library=chip_root / "lib/libtpuv7_emulator.so",
        firmware_archive=chip_root / "lib/libfirmware_core.a",
        toolchains_root=root / "third_party/toolchains_dir",
    ).require_base()
