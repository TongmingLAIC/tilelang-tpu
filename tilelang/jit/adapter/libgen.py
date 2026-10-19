# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
from typing import Optional
from .utils import is_cuda_target, is_hip_target, is_cpu_target, is_tpu_target
from tilelang import tvm as tvm
from tilelang.contrib.nvcc import get_target_compute_version
from tvm.target import Target
import contextlib
import ctypes
import os
import tempfile
import subprocess
import shutil
import logging
import re
import weakref
from tilelang.env import TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR
from tilelang.temp_storage import tilelang_temp_dir
from tilelang.jit.adapter.ppl_layout import PPLLayout, resolve_ppl_layout
from tilelang.engine.tpu_config import (
    TPURuntimeConfig,
    TPUTargetSpec,
    resolve_tpu_target,
)

logger = logging.getLogger(__name__)


def _new_temporary_source(suffix: str) -> str:
    """Create a named source file without leaving an open file descriptor."""

    descriptor, path = tempfile.mkstemp(suffix=suffix, text=True, dir=tilelang_temp_dir())
    os.close(descriptor)
    return path


def _remove_generated_file(path: Optional[str]) -> None:
    """Remove one generator-owned file if it still exists."""

    if path is None:
        return
    with contextlib.suppress(FileNotFoundError):
        os.remove(path)


def _cleanup_tpu_workspace(path: str, owner_pid: int) -> None:
    """Best-effort cleanup used by explicit close and process-exit finalizers.

    A forked child inherits Python finalizers but not ownership of the parent's
    workspace.  Letting that child remove the directory at exit could break a
    still-running parent whose TPU library resolves its private libkernel.so
    lazily.
    """
    if os.getpid() != owner_pid:
        return
    shutil.rmtree(path, ignore_errors=True)


class LibraryGenerator(object):
    srcpath: Optional[str] = None
    libpath: Optional[str] = None
    lib_code: Optional[str] = None

    def __init__(self,
                 target: Target,
                 tpu_target: Optional[TPUTargetSpec] = None,
                 tpu_runtime: Optional[TPURuntimeConfig] = None):
        self.target = Target(target)
        self.tpu_target = None
        self.tpu_runtime = None
        self._ppl_layout: Optional[PPLLayout] = None
        # A TPU host module embeds the absolute path of its private
        # ``libkernel.so``.  Retain the path produced by this generator rather
        # than treating an arbitrary prebuilt ``main.so`` as interchangeable.
        # The latter requires a bundled, validated manifest and is deliberately
        # not supported yet.
        self._tpu_compiled_libpath: Optional[str] = None
        self._tpu_compiled_runtime_identity = None
        # ``srcpath`` and ``libpath`` are public and may later be replaced by
        # callers. Track the files created by ``compile_lib`` separately so
        # cleanup never mistakes caller-owned paths for generator-owned files.
        self._temporary_source_path: Optional[str] = None
        self._temporary_library_path: Optional[str] = None
        if is_tpu_target(self.target):
            if tpu_target is None or tpu_runtime is None:
                raise ValueError("LibraryGenerator requires target and runtime configuration "
                                 "from the compiled artifact")
            resolved_target = resolve_tpu_target(target=self.target)
            if resolved_target != tpu_target:
                raise ValueError("TPU Target and compiled artifact identity disagree: "
                                 f"target={resolved_target}, artifact={tpu_target}")
            self.tpu_target = tpu_target
            self.tpu_runtime = tpu_runtime
        elif tpu_target is not None or tpu_runtime is not None:
            raise ValueError("TPU configuration can only be used with target='tpu'")
        # TPU compilation emits several cooperating sources and shared objects.
        # Keep every generator in its own directory so a later compile cannot
        # replace an already-loaded CModel/PCIe kernel through global template
        # files or PPL_KERNEL_PATH.
        self.tpu_workspace_dir: Optional[str] = None
        self._tpu_workspace_finalizer: Optional[weakref.finalize] = None
        if is_tpu_target(self.target):
            self._ensure_tpu_workspace()

    def _ensure_tpu_workspace(self) -> str:
        if self.tpu_workspace_dir is None:
            self.tpu_workspace_dir = tempfile.mkdtemp(prefix="tilelang-tpu-",
                                                     dir=tilelang_temp_dir())
            # Do not rely on every adapter/user remembering remove_lib().
            # ``weakref.finalize`` also runs for live objects at normal Python
            # shutdown, unlike ``__del__``-only cleanup.  The callback retains
            # only the path, never ``self``, so it cannot create a reference
            # cycle that keeps the generator alive.
            self._tpu_workspace_finalizer = weakref.finalize(self, _cleanup_tpu_workspace,
                                                             self.tpu_workspace_dir, os.getpid())
        return self.tpu_workspace_dir

    def update_lib_code(self, lib_code: str):
        self.lib_code = lib_code

    def _tpu_runtime_sdk_identity(self, lib_path: str):
        """Return the PPL ABI identity captured while compiling ``lib_path``.

        A standalone TPU ``main.so`` has no manifest that proves which PPL
        headers, runtime libraries, private ``libkernel.so``, chip, model, or
        runtime it was built for.  Do not try to infer that from the current
        environment: only the private artifact just compiled by this generator
        is loadable.  Cache/database rehydration stays fail-closed until it
        carries and validates such a manifest.
        """
        if self._ppl_layout is None or self._tpu_compiled_libpath is None or \
                self._tpu_compiled_runtime_identity is None:
            raise RuntimeError("TPU library loading only accepts an artifact compiled by this "
                               "LibraryGenerator instance; prebuilt TPU artifacts require a "
                               "verified manifest and are currently disabled.")
        requested_path = os.path.realpath(os.fspath(lib_path))
        if requested_path != self._tpu_compiled_libpath:
            raise RuntimeError("TPU library loading rejected a path that was not produced by "
                               "this LibraryGenerator instance; rebuild the kernel instead of "
                               "loading a prebuilt TPU artifact.")
        return self._tpu_compiled_runtime_identity

    def load_lib(self, lib_path: Optional[str] = None):
        if lib_path is None:
            lib_path = self.libpath
        tpu_device_id = None
        tpu_sdk_identity = None
        runtime_mode = self.tpu_runtime.runtime_mode if self.tpu_runtime is not None else None
        if is_tpu_target(self.target) and runtime_mode == "pcie":
            if os.environ.get("TILELANG_TPU_ALLOW_PCIE_LOAD") != "1":
                raise RuntimeError(
                    "PCIe TPU library loading is disabled by default because dlopen may "
                    "initialize the board runtime. Complete a CModel numerical smoke first, "
                    "then set TILELANG_TPU_ALLOW_PCIE_LOAD=1 and TILELANG_TPU_DEVICE_ID "
                    "for an explicitly supervised PCIe run.")
            # Keep the Python-side dlopen gate at least as strict as the host
            # template.  Waiting until main.so's init() means merely setting
            # ALLOW_PCIE can still load a library which links the TPU runtime.
            device_id = os.environ.get("TILELANG_TPU_DEVICE_ID")
            if device_id is None or re.fullmatch(r"[0-9]+", device_id) is None or \
                    int(device_id) > 2**31 - 1:
                raise RuntimeError("PCIe TPU library loading requires a non-negative integer "
                                   "TILELANG_TPU_DEVICE_ID before dlopen.")
            assert self.tpu_target is not None and self.tpu_runtime is not None
            # Keep CModel and PCIe from sharing a process-global vendor
            # runtime.  This happens before ctypes.CDLL, so an invalid mode
            # transition never initializes or touches a board runtime.
            from .tpu import reserve_tpu_runtime_identity
            tpu_device_id = int(device_id)
            tpu_sdk_identity = self._tpu_runtime_sdk_identity(lib_path)
            reserve_tpu_runtime_identity(self.tpu_target, self.tpu_runtime, tpu_device_id,
                                         tpu_sdk_identity)
        elif is_tpu_target(self.target) and runtime_mode == "cmodel":
            assert self.tpu_target is not None and self.tpu_runtime is not None
            # The vendor CModel runtime is process-global. Reserve its core
            # topology before dlopen rather than allowing BM1690 (8 cores) and
            # BM1690E (4 cores) to silently share one initialized runtime.
            from .tpu import reserve_tpu_runtime_identity
            tpu_device_id = 0
            tpu_sdk_identity = self._tpu_runtime_sdk_identity(lib_path)
            reserve_tpu_runtime_identity(
                self.tpu_target,
                self.tpu_runtime,
                device_id=tpu_device_id,
                sdk_identity=tpu_sdk_identity)
        library = ctypes.CDLL(lib_path)
        if is_tpu_target(self.target):
            assert tpu_device_id is not None
            assert tpu_sdk_identity is not None
            try:
                bind_device = library.tilelang_tpu_bind_device
            except AttributeError as exc:
                raise RuntimeError(
                    "TPU host library lacks tilelang_tpu_bind_device; rebuild it "
                    "with the current TileLang TPU runtime safety template.") from exc
            bind_device.argtypes = [ctypes.c_int]
            bind_device.restype = ctypes.c_int
            status = bind_device(tpu_device_id)
            if status != 0:
                raise RuntimeError("TPU host library rejected the reserved runtime device "
                                   f"{tpu_device_id} (status {status}).")
            # The TPU host library embeds an absolute path to the private
            # libkernel.so and may resolve it lazily.  Keep this generator (and
            # therefore its workspace finalizer) alive for at least as long as
            # the returned dlopen handle, even when callers do not retain the
            # adapter or generator separately.  ctypes.CDLL is a normal Python
            # object and supports private owner attributes.
            try:
                library._tilelang_tpu_workspace_owner = self
            except (AttributeError, TypeError) as exc:
                raise RuntimeError("Loaded TPU library handle cannot retain ownership of its "
                                   "private compilation workspace") from exc
        return library

    def compile_lib(self, timeout: float = None, with_tl: bool = True):
        target = self.target
        if is_cuda_target(target):
            compute_version = "".join(get_target_compute_version(target).split("."))
            if compute_version == "90":
                compute_version = "90a"
            srcpath = _new_temporary_source(".cu")
            libpath = srcpath.replace(".cu", ".so")

            command = [
                "nvcc",
                "-std=c++17",
                "-w",  # Disable all warning messages
                "-Xcudafe",
                "--diag_suppress=177",
                "--compiler-options",
                "'-fPIC'",
                "-lineinfo",
                "--shared",
                srcpath,
                "-lcuda",
                "-gencode",
                f"arch=compute_{compute_version},code=sm_{compute_version}",
            ]

        elif is_hip_target(target):
            srcpath = _new_temporary_source(".cpp")
            libpath = srcpath.replace(".cpp", ".so")

            command = [
                "hipcc",
                "-std=c++17",
                "-fPIC",
                "--shared",
                srcpath,
            ]
        elif is_cpu_target(target):
            from tilelang.contrib.cc import get_cplus_compiler
            compiler = get_cplus_compiler()
            srcpath = _new_temporary_source(".cpp")
            libpath = srcpath.replace(".cpp", ".so")

            command = [compiler, "-std=c++17", "-fPIC", "-shared", srcpath]
            with_tl = False
            command += [
                "-I" + TILELANG_TEMPLATE_PATH,
            ]
        elif is_tpu_target(target):
            assert self.tpu_target is not None and self.tpu_runtime is not None
            self._tpu_compiled_libpath = None
            self._tpu_compiled_runtime_identity = None
            ppl_root = os.environ.get("PPL_PROJECT_ROOT")
            if not ppl_root:
                raise EnvironmentError("PPL_PROJECT_ROOT environment variable is not set.")
            ppl_layout = resolve_ppl_layout(ppl_root, self.tpu_target.chip)
            self._ppl_layout = ppl_layout
            if self.tpu_target.programming_model == "rv":
                # RV source includes rvt_api.h for both compiler-selected
                # portable TPU operations and explicit low-level rvt_* calls.
                # A missing header is a chip/SDK capability error, not a
                # reason to fall back to the TPU-Kernel path.
                ppl_layout.require_rvt_api()

            runtime_mode = self.tpu_runtime.runtime_mode
            profile_session = self._tpu_profile_session(self.tpu_target, self.tpu_runtime)
            ppl_layout.require_runtime(runtime_mode)
            if profile_session:
                ppl_layout.require_profiling(runtime_mode)
            runtime_identity = ppl_layout.runtime_identity_for(runtime_mode)

            if runtime_mode == "pcie":
                self.tpu_compile_pcie(
                    timeout=timeout,
                    layout=ppl_layout,
                    pcie_runtime_lib=runtime_identity[1],
                    profiling=profile_session,
                )
            elif runtime_mode == "cmodel":
                self.tpu_compile_cmodel(
                    timeout=timeout, layout=ppl_layout, profiling=profile_session)
            else:
                raise ValueError(f"Unsupported TPU runtime mode: {runtime_mode}")
            self.srcpath = self._ensure_tpu_workspace()
            self.libpath = os.path.join(self.srcpath, "main.so")
            self._tpu_compiled_libpath = os.path.realpath(self.libpath)
            self._tpu_compiled_runtime_identity = runtime_identity
            return

        else:
            raise ValueError(f"Unsupported target: {target}")

        try:
            if with_tl:
                command += [
                    "-I" + TILELANG_TEMPLATE_PATH,
                    "-I" + CUTLASS_INCLUDE_DIR,
                ]
                command += ["-diag-suppress=20013"]
            command += ["-o", libpath]

            with open(srcpath, "w", encoding="utf-8") as source_file:
                source_file.write(self.lib_code)
            ret = subprocess.run(command, timeout=timeout)
        except (KeyboardInterrupt, SystemExit):
            _remove_generated_file(libpath)
            _remove_generated_file(srcpath)
            raise
        except Exception as e:
            _remove_generated_file(libpath)
            _remove_generated_file(srcpath)
            raise RuntimeError(f"Compile kernel failed because of {e}") from e

        if ret.returncode != 0:
            _remove_generated_file(libpath)
            _remove_generated_file(srcpath)
            raise RuntimeError(f"Compilation Failed! {command}")

        previous_source_path = self._temporary_source_path
        previous_library_path = self._temporary_library_path
        self.srcpath = srcpath
        self.libpath = libpath
        self._temporary_source_path = srcpath
        self._temporary_library_path = libpath
        if previous_source_path != srcpath:
            _remove_generated_file(previous_source_path)
        if previous_library_path != libpath:
            _remove_generated_file(previous_library_path)

    def remove_lib(self):
        if self.tpu_workspace_dir is not None:
            finalizer = self._tpu_workspace_finalizer
            self._tpu_workspace_finalizer = None
            self.tpu_workspace_dir = None
            self.libpath = None
            self.srcpath = None
            self._tpu_compiled_libpath = None
            self._tpu_compiled_runtime_identity = None
            if finalizer is not None and finalizer.alive:
                finalizer()
            return
        temporary_library_path = self._temporary_library_path
        self._temporary_library_path = None
        _remove_generated_file(temporary_library_path)
        if self.libpath == temporary_library_path:
            self.libpath = None
        temporary_source_path = self._temporary_source_path
        self._temporary_source_path = None
        _remove_generated_file(temporary_source_path)
        if self.srcpath == temporary_source_path:
            self.srcpath = None

    def close(self):
        """Release generated files; safe to call repeatedly.

        A loaded TPU library automatically retains this generator, so normal
        garbage collection cannot remove files that its dlopen handle may
        still need.  Calling ``close`` is explicit invalidation: callers must
        stop using any library handle produced by this generator first.
        """
        self.remove_lib()

    def get_source_path(self):
        return self.srcpath

    def get_lib_path(self):
        return self.libpath

    def set_lib_path(self, libpath):
        self.libpath = libpath

    def set_src_path(self, srcpath):
        self.srcpath = srcpath

    @staticmethod
    def _run_tpu_command(command, task_name, timeout):
        try:
            subprocess.run(command, timeout=timeout, check=True)
        except (OSError, subprocess.SubprocessError) as e:
            raise RuntimeError(f"{task_name} failed: {e}") from e

    @staticmethod
    def _tpu_profile_session(target: TPUTargetSpec, runtime: TPURuntimeConfig) -> bool:
        """Validate and recognize one supervised profiling build.

        The supervisor records all three independent TPU selection axes in the
        worker environment.  Checking only the runtime mode would allow a
        stale/misrouted worker to compile a different chip or programming
        model while still enabling profiling-specific source and libraries.
        """

        if os.environ.get("TILELANG_TPU_PROFILE_SESSION") != "1":
            return False
        expected_identity = {
            "TILELANG_TPU_PROFILE_CHIP": target.chip,
            "TILELANG_TPU_PROFILE_PROGRAMMING_MODEL": target.programming_model,
            "TILELANG_TPU_PROFILE_RUNTIME_MODE": runtime.runtime_mode,
        }
        for variable, expected in expected_identity.items():
            selected = os.environ.get(variable)
            if selected != expected:
                raise ValueError("TPU profile build identity disagrees with the compiled "
                                 f"artifact: {variable}={selected!r}, expected {expected!r}")
        return True

    @staticmethod
    def _ppl_compile_flags(layout: PPLLayout,
                           src_dir: str,
                           programming_model: str,
                           runtime_mode: str,
                           *,
                           profiling: bool = False):
        layout.require_runtime(runtime_mode)
        if profiling:
            layout.require_profiling(runtime_mode)
        definitions = [
            *(f"-D{definition}" for definition in layout.compile_definitions),
        ]
        if programming_model == "rv":
            layout.require_rvt_api()
            definitions.append("-DTILELANG_TPU_RV")
        elif programming_model == "tpukernel":
            definitions.append("-DTILELANG_TPU_TPUKERNEL")
        else:
            raise ValueError("Unsupported TPU programming model "
                             f"{programming_model!r}; expected 'tpukernel' or 'rv'")
        if profiling and runtime_mode == "pcie":
            definitions.append("-DTILELANG_TPU_PCIE_PROFILING")
        includes = [
            f"-I{path}" for path in layout.include_dirs_for(runtime_mode, profiling=profiling)
        ]
        include_dir = os.path.join(src_dir, "include")
        if os.path.isdir(include_dir):
            includes.append(f"-I{include_dir}")
        return definitions, includes

    def tpu_compile_pcie(self,
                         timeout,
                         layout: PPLLayout,
                         pcie_runtime_lib: str,
                         *,
                         profiling: bool = False):
        cross_gcc = str(layout.pcie_cross_gcc())

        src_dir = self._ensure_tpu_workspace()
        definitions, includes = self._ppl_compile_flags(
            layout, src_dir, self.tpu_target.programming_model, "pcie", profiling=profiling)
        common = definitions + ["-Dlibkernel_EXPORTS"
                               ] + includes + ["-O3", "-DNDEBUG", "-fPIC", "-flto"]
        kernel_o = os.path.join(src_dir, "kernel.o")
        helper_o = os.path.join(src_dir, "ppl_helper.o")
        libkernel = os.path.join(src_dir, "libkernel.so")

        self._run_tpu_command(
            [cross_gcc, *common, "-c",
             os.path.join(src_dir, "kernel.c"), "-o", kernel_o], "Compile TPU kernel", timeout)
        self._run_tpu_command(
            [cross_gcc, *common, "-c",
             str(layout.ppl_helper_source), "-o", helper_o], "Compile PPL helper", timeout)
        self._run_tpu_command([
            cross_gcc, "-shared", "-fPIC", "-flto", "-Wl,--no-undefined",
            "-Wl,-soname,libkernel.so", "-o", libkernel, kernel_o, helper_o, "-Wl,--whole-archive",
            str(layout.firmware_archive), "-Wl,--no-whole-archive", "-Wl,-s", "-ldl", "-lm"
        ], "Link PCIe libkernel.so", timeout)

        host_common = definitions + includes + [
            "-O3",
            "-DNDEBUG",
            "-std=c++17",
            "-fPIC",
            f'-DTILELANG_PPL_KERNEL_PATH="{libkernel}"',
        ]
        kernel_host_o = os.path.join(src_dir, "kernel_host.o")
        main_o = os.path.join(src_dir, "main.o")
        self._run_tpu_command([
            "/usr/bin/c++", *host_common, "-c",
            os.path.join(src_dir, "kernel.cpp"), "-o", kernel_host_o
        ], "Compile TPU host wrapper", timeout)
        self._run_tpu_command(
            ["/usr/bin/c++", *host_common, "-c",
             os.path.join(src_dir, "main.cpp"), "-o", main_o], "Compile TPU host entry", timeout)
        host_libraries = ["-ltpuv7_rt"]
        if profiling:
            host_libraries.append("-ltpudnn")
        host_libraries.append("-lpthread")
        self._run_tpu_command([
            "/usr/bin/c++", "-shared", "-fPIC", "-Wl,--no-undefined", "-o",
            os.path.join(src_dir, "main.so"), kernel_host_o, main_o, f"-L{pcie_runtime_lib}",
            f"-L{layout.backend_lib}",
            f"-Wl,--disable-new-dtags,-rpath,{pcie_runtime_lib}:{layout.backend_lib}",
            *host_libraries
        ], "Link PCIe main.so", timeout)

    def tpu_compile_cmodel(self, timeout, layout: PPLLayout, *, profiling: bool = False):
        src_dir = self._ensure_tpu_workspace()
        definitions, includes = self._ppl_compile_flags(
            layout, src_dir, self.tpu_target.programming_model, "cmodel", profiling=profiling)
        definitions.append("-DUSING_CMODEL")
        common = definitions + includes + ["-O3", "-DNDEBUG", "-fPIC"]

        kernel_c = os.path.join(src_dir, "kernel.c")
        kernel_cpp_o = os.path.join(src_dir, "kernel_cpp.o")
        main_cpp_o = os.path.join(src_dir, "main_cpp.o")
        kernel_c_o = os.path.join(src_dir, "kernel_c.o")
        helper_o = os.path.join(src_dir, "ppl_helper_c.o")
        libkernel = os.path.join(src_dir, "libkernel.so")
        main_so = os.path.join(src_dir, "main.so")
        rpath = f"{layout.runtime_lib}:{layout.backend_lib}"
        # TPUv7 defaults to eight emulator cores. BM1690E exposes four, and
        # launching the extra scalar-emulator workers makes them address
        # non-existent cores before the first kernel can complete.  Embed the
        # value in main.so so the runtime process does not depend on an ambient
        # TPU_RT_CORE_NUM value left by the compiler process.
        host_common = common + [
            f'-DTILELANG_PPL_KERNEL_PATH="{libkernel}"',
            f'-DTILELANG_TPU_CMODEL_CORE_NUM="{layout.physical_core_count}"',
        ]

        logger.info("Compiling TPU cmodel kernel for PPL 1.7 architecture %s", layout.arch)
        self._run_tpu_command([
            "/usr/bin/c++", *host_common, "-std=c++17", "-c",
            os.path.join(src_dir, "kernel.cpp"), "-o", kernel_cpp_o
        ], "Compile TPU host wrapper", timeout)
        self._run_tpu_command([
            "/usr/bin/c++", *host_common, "-std=c++17", "-c",
            os.path.join(src_dir, "main.cpp"), "-o", main_cpp_o
        ], "Compile TPU host entry", timeout)
        self._run_tpu_command(
            ["/usr/bin/cc", *common, "-Dkernel_EXPORTS", "-c", kernel_c, "-o", kernel_c_o],
            "Compile TPU cmodel kernel", timeout)
        self._run_tpu_command([
            "/usr/bin/cc", *common, "-Dkernel_EXPORTS", "-c",
            str(layout.ppl_helper_source), "-o", helper_o
        ], "Compile PPL helper", timeout)
        self._run_tpu_command([
            "/usr/bin/cc", "-shared", "-fPIC", "-Wl,--no-undefined", "-Wl,-soname,libkernel.so",
            "-o", libkernel, kernel_c_o, helper_o, f"-Wl,-rpath,{rpath}",
            str(layout.emulator_library), "-lm"
        ], "Link cmodel libkernel.so", timeout)
        self._run_tpu_command([
            "/usr/bin/c++", "-shared", "-fPIC", "-o", main_so, kernel_cpp_o, main_cpp_o,
            f"-L{layout.runtime_lib}", f"-L{layout.backend_lib}",
            f"-Wl,--disable-new-dtags,-rpath,{rpath}", "-ltpuv7_rt", "-lcdm_daemon_emulator",
            "-lpthread"
        ], "Link cmodel main.so", timeout)
