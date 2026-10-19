# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Content-addressed identity for the TPU toolchain inputs in this manifest.

This module performs filesystem inspection only.  It never imports a vendor
runtime, initializes a TPU, or executes a CModel/PCIe program.  Every recorded
file is hashed through an open file descriptor and checked again afterwards;
directory identities additionally verify that their complete topology did not
change while it was being read.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Any, Mapping, Optional, Sequence, Union

from .ppl_layout import resolve_ppl_layout


class TPUToolchainIdentityError(RuntimeError):
    """The requested toolchain is missing, ambiguous, or changed while read."""


_CHUNK_SIZE = 1024 * 1024


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_mode,
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _hash_open_file(file_object: Any) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = file_object.read(_CHUNK_SIZE)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


PathLike = Union[str, os.PathLike]


def file_content_identity(path: PathLike) -> dict[str, Any]:
    """Return a stable SHA-256 identity for one regular file.

    A logical symlink is accepted only when its fully resolved target is a
    regular file.  Both spellings are recorded because loader-facing paths such
    as ``tpuv7-current`` are operationally meaningful, while the resolved path
    prevents two different installations from being conflated.
    """

    logical = Path(path).expanduser().absolute()
    try:
        logical_before = logical.lstat()
        resolved = logical.resolve(strict=True)
        resolved_before = resolved.stat()
    except OSError as error:
        raise TPUToolchainIdentityError(
            f"cannot resolve toolchain file {logical}: {error}") from error
    if not (stat.S_ISREG(logical_before.st_mode) or stat.S_ISLNK(logical_before.st_mode)):
        raise TPUToolchainIdentityError(
            f"toolchain path is neither a regular file nor a file symlink: {logical}")
    if not stat.S_ISREG(resolved_before.st_mode):
        raise TPUToolchainIdentityError(
            f"resolved toolchain path is not a regular file: {resolved}")

    try:
        descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    except OSError as error:
        raise TPUToolchainIdentityError(
            f"cannot open toolchain file {resolved}: {error}") from error
    try:
        opened_before = os.fstat(descriptor)
        if not stat.S_ISREG(opened_before.st_mode):
            raise TPUToolchainIdentityError(
                f"opened toolchain path is not a regular file: {resolved}")
        with os.fdopen(descriptor, "rb", closefd=False) as input_file:
            sha256 = _hash_open_file(input_file)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    try:
        logical_after = logical.lstat()
        resolved_after_path = logical.resolve(strict=True)
        resolved_after = resolved_after_path.stat()
    except OSError as error:
        raise TPUToolchainIdentityError(
            f"toolchain file disappeared while hashing {logical}: {error}") from error
    stable = (
        _stat_signature(resolved_before) == _stat_signature(opened_before) ==
        _stat_signature(opened_after) == _stat_signature(resolved_after) and
        _stat_signature(logical_before) == _stat_signature(logical_after) and
        resolved_after_path == resolved)
    if not stable:
        raise TPUToolchainIdentityError(f"toolchain file changed while hashing: {logical}")
    return {
        "path": str(logical),
        "resolved_path": str(resolved),
        "size": opened_after.st_size,
        "sha256": sha256,
    }


def _tree_snapshot(root: Path) -> tuple[tuple[str, tuple[int, ...], Optional[str]], ...]:
    entries = []
    try:
        for current, directories, filenames in os.walk(root, followlinks=False):
            directories.sort()
            filenames.sort()
            current_path = Path(current)
            for directory in directories:
                child = current_path / directory
                child_stat = child.lstat()
                if not stat.S_ISDIR(child_stat.st_mode):
                    raise TPUToolchainIdentityError(
                        f"toolchain tree contains a non-directory entry in its directory set: "
                        f"{child}")
            for filename in filenames:
                child = current_path / filename
                child_stat = child.lstat()
                if stat.S_ISREG(child_stat.st_mode):
                    link_target = None
                    signature = _stat_signature(child_stat)
                elif stat.S_ISLNK(child_stat.st_mode):
                    target = child.resolve(strict=True)
                    if not target.is_file():
                        raise TPUToolchainIdentityError(
                            f"toolchain tree symlink does not resolve to a regular file: {child}")
                    link_target = os.readlink(child)
                    # Include the target's metadata as well as the link inode's
                    # metadata, including when a link points outside this tree.
                    signature = _stat_signature(child_stat) + _stat_signature(target.stat())
                else:
                    raise TPUToolchainIdentityError(
                        f"toolchain tree contains a non-regular file: {child}")
                entries.append((
                    child.relative_to(root).as_posix(),
                    signature,
                    link_target,
                ))
    except OSError as error:
        raise TPUToolchainIdentityError(
            f"cannot enumerate toolchain tree {root}: {error}") from error
    return tuple(entries)


def tree_content_identity(path: PathLike) -> dict[str, Any]:
    """Return one compact digest for every file and relative path in a tree."""

    logical = Path(path).expanduser().absolute()
    try:
        resolved = logical.resolve(strict=True)
        root_before = resolved.stat()
    except OSError as error:
        raise TPUToolchainIdentityError(
            f"cannot resolve toolchain tree {logical}: {error}") from error
    if not stat.S_ISDIR(root_before.st_mode):
        raise TPUToolchainIdentityError(f"toolchain tree is not a directory: {logical}")

    before = _tree_snapshot(resolved)
    aggregate = hashlib.sha256()
    aggregate.update(b"tilelang-tpu-toolchain-tree-v1\0")
    total_size = 0
    for relative, _signature, link_target in before:
        identity = file_content_identity(resolved / relative)
        encoded_relative = relative.encode("utf-8", errors="surrogateescape")
        encoded_link = (link_target or "").encode("utf-8", errors="surrogateescape")
        aggregate.update(len(encoded_relative).to_bytes(8, "big"))
        aggregate.update(encoded_relative)
        aggregate.update(len(encoded_link).to_bytes(8, "big"))
        aggregate.update(encoded_link)
        aggregate.update(identity["size"].to_bytes(8, "big"))
        aggregate.update(bytes.fromhex(identity["sha256"]))
        total_size += identity["size"]

    after = _tree_snapshot(resolved)
    try:
        root_after = resolved.stat()
        resolved_after = logical.resolve(strict=True)
    except OSError as error:
        raise TPUToolchainIdentityError(
            f"toolchain tree disappeared while hashing {logical}: {error}") from error
    if (before != after or _stat_signature(root_before) != _stat_signature(root_after) or
            resolved_after != resolved):
        raise TPUToolchainIdentityError(f"toolchain tree changed while hashing: {logical}")
    return {
        "path": str(logical),
        "resolved_path": str(resolved),
        "file_count": len(before),
        "total_size": total_size,
        "sha256": aggregate.hexdigest(),
    }


def _file_state(path: PathLike) -> tuple[Any, ...]:
    """Return metadata used only to detect a change around a full capture."""

    logical = Path(path).expanduser().absolute()
    try:
        logical_stat = logical.lstat()
        resolved = logical.resolve(strict=True)
        resolved_stat = resolved.stat()
    except OSError as error:
        raise TPUToolchainIdentityError(
            f"cannot snapshot toolchain file {logical}: {error}") from error
    if not stat.S_ISREG(resolved_stat.st_mode):
        raise TPUToolchainIdentityError(
            f"resolved toolchain path is not a regular file: {resolved}")
    return (
        str(logical),
        str(resolved),
        _stat_signature(logical_stat),
        _stat_signature(resolved_stat),
    )


def _tree_state(path: PathLike) -> tuple[Any, ...]:
    logical = Path(path).expanduser().absolute()
    try:
        resolved = logical.resolve(strict=True)
        root_stat = resolved.stat()
    except OSError as error:
        raise TPUToolchainIdentityError(
            f"cannot snapshot toolchain tree {logical}: {error}") from error
    if not stat.S_ISDIR(root_stat.st_mode):
        raise TPUToolchainIdentityError(f"toolchain tree is not a directory: {logical}")
    return (
        str(logical),
        str(resolved),
        _stat_signature(root_stat),
        _tree_snapshot(resolved),
    )


def _capture_stability_snapshot(
    files: Sequence[PathLike],
    trees: Sequence[PathLike],
) -> tuple[tuple[tuple[Any, ...], ...], tuple[tuple[Any, ...], ...]]:
    unique_files = sorted({str(Path(path).expanduser().absolute()) for path in files})
    unique_trees = sorted({str(Path(path).expanduser().absolute()) for path in trees})
    return (
        tuple(_file_state(path) for path in unique_files),
        tuple(_tree_state(path) for path in unique_trees),
    )


def _loaded_library_path(module_name: str, attribute_path: Sequence[str]) -> Path:
    module = sys.modules.get(module_name)
    if module is None:
        raise TPUToolchainIdentityError(
            f"cannot identify an unloaded compiler library module: {module_name}")
    value: Any = module
    for attribute in attribute_path:
        value = getattr(value, attribute, None)
        if value is None:
            rendered = ".".join(attribute_path)
            raise TPUToolchainIdentityError(
                f"cannot identify loaded library from {module_name}.{rendered}")
    try:
        path = Path(os.fspath(value))
    except TypeError as error:
        raise TPUToolchainIdentityError(
            f"loaded library path from {module_name} is not path-like: {value!r}") from error
    if not path.is_absolute():
        raise TPUToolchainIdentityError(
            f"loaded library path from {module_name} is not absolute: {path}")
    return path


def _executable(path: PathLike, label: str) -> Path:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except OSError as error:
        raise TPUToolchainIdentityError(f"cannot resolve {label} {path}: {error}") from error
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise TPUToolchainIdentityError(f"{label} is not an executable regular file: {resolved}")
    return resolved


def _same_path(paths: Sequence[Path], label: str) -> Path:
    try:
        resolved = tuple(path.resolve(strict=True) for path in paths)
    except OSError as error:
        raise TPUToolchainIdentityError(f"cannot resolve shared PPL {label}: {error}") from error
    if not resolved or any(path != resolved[0] for path in resolved[1:]):
        raise TPUToolchainIdentityError(f"PPL layouts disagree on their shared {label}: " +
                                        ", ".join(str(path) for path in resolved))
    return resolved[0]


def _find_tpu_smi(pcie_runtime: Path, explicit: Optional[PathLike]) -> Path:
    if explicit is not None:
        return _executable(explicit, "tpu-smi")
    path_tool = shutil.which("tpu-smi")
    candidates = (
        pcie_runtime.parent / "bin/tpu-smi",
        Path("/opt/tpuv7/tpuv7-current/bin/tpu-smi"),
        Path(path_tool) if path_tool else None,
        Path("/opt/tpuv7/tpuv7-runtime_1.9.3/bin/tpu-smi"),
        Path("/opt/tpuv7/tpuv7-runtime_1.9.3.3/bin/tpu-smi"),
        Path("/opt/tpuv7/tpuv7-runtime/bin/tpu-smi"),
    )
    for candidate in candidates:
        if candidate is not None and candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise TPUToolchainIdentityError("cannot find an executable tpu-smi for PCIe identity")


def capture_tpu_toolchain_identity(
    ppl_project_root: PathLike,
    runtime_mode: str,
    *,
    environment: Optional[Mapping[str, str]] = None,
    chips: Sequence[str] = ("bm1690", "bm1690e"),
    tilelang_library: Optional[PathLike] = None,
    tvm_library: Optional[PathLike] = None,
    host_c_compiler: PathLike = "/usr/bin/cc",
    host_cxx_compiler: PathLike = "/usr/bin/c++",
    tpu_smi: Optional[PathLike] = None,
) -> dict[str, Any]:
    """Capture a JSON-comparable identity of the declared TPU build inputs.

    The CModel baseline is always present, including in a PCIe identity, so a
    promotion gate can compare the common compiler/PPL portions directly.
    PCIe additionally covers the installed runtime, profiling library,
    firmware, board utility, and the complete cross-toolchain root (driver,
    cc1/LTO tools, binutils, linker scripts, libraries, and sysroot).

    This is a bounded promotion manifest, not a hermetic build proof. In
    particular, the host C/C++ entries identify the selected driver files but
    do not recursively hash their system headers, linker, libc, or GCC support
    binaries. Python/oracle packages and an optional profile decoder are also
    outside this compiler/runtime manifest and must be recorded separately by
    their consumers.
    """

    if runtime_mode not in ("cmodel", "pcie"):
        raise ValueError("runtime_mode must be 'cmodel' or 'pcie'")
    selected_chips = tuple(chips)
    if not selected_chips or len(set(selected_chips)) != len(selected_chips):
        raise ValueError("chips must be a non-empty sequence without duplicates")
    resolved_environment = dict(os.environ)
    if environment is not None:
        resolved_environment.update(environment)

    try:
        ppl_root = Path(ppl_project_root).expanduser().resolve(strict=True)
    except OSError as error:
        raise TPUToolchainIdentityError(
            f"cannot resolve PPL project root {ppl_project_root}: {error}") from error
    chip_map_path = ppl_root / "deps/chip/chip_map.json"
    chip_map_before_layout = _file_state(chip_map_path)
    layouts = [resolve_ppl_layout(str(ppl_root), chip) for chip in selected_chips]
    if _file_state(chip_map_path) != chip_map_before_layout:
        raise TPUToolchainIdentityError("PPL chip map changed while resolving chip layouts")
    for layout in layouts:
        layout.require_runtime("cmodel", environment=resolved_environment)
        if runtime_mode == "pcie":
            layout.require_profiling("pcie", environment=resolved_environment)

    tilelang_path = (
        Path(tilelang_library) if tilelang_library is not None else _loaded_library_path(
            "tilelang", ("_LIB_PATH",)))
    tvm_path = (
        Path(tvm_library) if tvm_library is not None else _loaded_library_path(
            "tvm._ffi.base", ("_LIB", "_name")))
    host_c_path = _executable(host_c_compiler, "host C compiler")
    host_cxx_path = _executable(host_cxx_compiler, "host C++ compiler")

    common_paths = {
        "kernel_common_include":
            _same_path([layout.kernel_common_include for layout in layouts],
                       "kernel common include"),
        "device_utils_include":
            _same_path([layout.device_utils_include for layout in layouts],
                       "device utilities include"),
        "host_include":
            _same_path([layout.host_include for layout in layouts], "host include"),
        "runtime_include":
            _same_path([layout.runtime_include for layout in layouts], "runtime include"),
    }
    helper_path = _same_path([layout.ppl_helper_source for layout in layouts], "ppl_helper.c")
    cmodel_runtime = _same_path([layout.runtime_lib for layout in layouts],
                                "CModel runtime directory")

    cross_gcc = None
    cross_toolchain_root = None
    pcie_runtime = None
    tpu_smi_path = None
    if runtime_mode == "pcie":
        cross_gcc = _executable(
            _same_path([layout.pcie_cross_gcc() for layout in layouts], "PCIe cross compiler"),
            "PCIe cross GCC",
        )
        cross_toolchain_root = cross_gcc.parent.parent
        pcie_runtime = _same_path(
            [layout.pcie_runtime_lib(resolved_environment) for layout in layouts],
            "installed PCIe runtime directory",
        )
        tpu_smi_path = _find_tpu_smi(pcie_runtime, tpu_smi)

    watched_files = [
        chip_map_path,
        *(ppl_root / "deps/chip" / layout.arch / "config_common.cmake" for layout in layouts),
        tilelang_path,
        tvm_path,
        host_c_path,
        host_cxx_path,
        helper_path,
        *(layout.emulator_library for layout in layouts),
    ]
    watched_trees = [
        *common_paths.values(),
        cmodel_runtime,
        *(layout.kernel_include for layout in layouts),
        *(layout.backend_lib for layout in layouts),
    ]
    if runtime_mode == "pcie":
        assert cross_gcc is not None and cross_toolchain_root is not None
        assert pcie_runtime is not None and tpu_smi_path is not None
        watched_files.extend((
            cross_gcc,
            pcie_runtime / "libtpuv7_rt.so",
            tpu_smi_path,
            *(layout.firmware_archive for layout in layouts),
            *(layout.tpudnn_library for layout in layouts),
        ))
        watched_trees.extend((
            cross_toolchain_root,
            *(layout.tpudnn_include for layout in layouts),
        ))
    stability_before = _capture_stability_snapshot(watched_files, watched_trees)

    chip_identities = {}
    for chip, layout in zip(selected_chips, layouts):  # noqa: B905
        chip_identities[chip] = {
            "ppl_arch": layout.arch,
            "physical_core_count": layout.physical_core_count,
            "compile_definitions": list(layout.compile_definitions),
            "compile_configuration": file_content_identity(
                ppl_root / "deps/chip" / layout.arch / "config_common.cmake"),
            "kernel_include_tree": tree_content_identity(layout.kernel_include),
            "backend_tree": tree_content_identity(layout.backend_lib),
            "emulator_library": file_content_identity(layout.emulator_library),
        }

    result: dict[str, Any] = {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "runtime_mode": runtime_mode,
        "ppl_project_root": str(ppl_root),
        "compiler_runtime": {
            "tilelang_library": file_content_identity(tilelang_path),
            "tvm_library": file_content_identity(tvm_path),
            "host_c_compiler": file_content_identity(host_c_path),
            "host_cxx_compiler": file_content_identity(host_cxx_path),
        },
        "ppl_common": {
            "chip_map": file_content_identity(chip_map_path),
            "include_trees": {
                name: tree_content_identity(path) for name, path in common_paths.items()
            },
            "ppl_helper": file_content_identity(helper_path),
        },
        "cmodel": {
            "runtime_tree": tree_content_identity(cmodel_runtime),
        },
        "chips": chip_identities,
    }

    if runtime_mode == "pcie":
        assert cross_gcc is not None and cross_toolchain_root is not None
        assert pcie_runtime is not None and tpu_smi_path is not None
        result["pcie"] = {
            "cross_gcc": file_content_identity(cross_gcc),
            "cross_toolchain_tree": tree_content_identity(cross_toolchain_root),
            "installed_runtime_library": file_content_identity(pcie_runtime / "libtpuv7_rt.so"),
            "tpu_smi": file_content_identity(tpu_smi_path),
            "chips": {
                chip: {
                    "firmware_archive": file_content_identity(layout.firmware_archive),
                    "tpudnn_include_tree": tree_content_identity(layout.tpudnn_include),
                    "tpudnn_library": file_content_identity(layout.tpudnn_library),
                } for chip, layout in zip(selected_chips, layouts)  # noqa: B905
            },
        }
    stability_after = _capture_stability_snapshot(watched_files, watched_trees)
    if stability_after != stability_before:
        raise TPUToolchainIdentityError(
            "TPU toolchain changed during the complete identity capture")
    return result


__all__ = [
    "TPUToolchainIdentityError",
    "capture_tpu_toolchain_identity",
    "file_content_identity",
    "tree_content_identity",
]
