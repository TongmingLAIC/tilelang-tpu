# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
This module provides an auto-tuning infrastructure for TileLang (tl) programs. 
It includes functionality to JIT-compile TileLang programs into a runnable 
kernel adapter using TVM.
"""

from typing import Callable, List, Literal, Union, Any, Optional, Dict

from tilelang import tvm as tvm
from tvm.tir import PrimFunc
from tvm.target import Target

from tilelang.jit.adapter import BaseKernelAdapter
from tilelang.jit.kernel import JITKernel
from tilelang.jit.adapter.tpu_profiling import (
    TPUInstructionProfiler as TPUInstructionProfiler,
    TPUInstructionTiming as TPUInstructionTiming,
    TPUProfileReport as TPUProfileReport,
    TPUProfilingCommandError as TPUProfilingCommandError,
    TPUProfilingConfig as TPUProfilingConfig,
    TPUProfilingError as TPUProfilingError,
    TPUProfilingTimeoutError as TPUProfilingTimeoutError,
    TPURawInstruction as TPURawInstruction,
    parse_cmodel_raw_instruction_dumps as parse_cmodel_raw_instruction_dumps,
    parse_perfai_instruction_timings as parse_perfai_instruction_timings,
    parse_perfai_timeline_events as parse_perfai_timeline_events,
    parse_pcie_decoded_instruction_timings as parse_pcie_decoded_instruction_timings,
    run_tpu_cmodel_profile as run_tpu_cmodel_profile,
    run_tpu_pcie_profile as run_tpu_pcie_profile,
)
from tilelang.utils.target import determine_target, AVAILABLE_TARGETS, is_tpu_target_spec
from tilelang.cache import cached
from logging import getLogger

logger = getLogger(__name__)


def jit(
    func: Callable = None,
    *,  # Enforce keyword-only arguments from here on
    out_idx: Union[List[int], int] = None,
    execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
    target: Union[str, Target] = "auto",
    verbose: bool = False,
    runtime_mode: Optional[Literal["pcie", "cmodel"]] = None,
    **pass_config_kwargs: Any,
) -> BaseKernelAdapter:
    """
    A decorator (or decorator factory) that JIT-compiles a given TileLang PrimFunc 
    into a runnable kernel adapter using TVM. If called with arguments, it returns 
    a decorator that can be applied to a function. If called without arguments, 
    it directly compiles the given function.

    Parameters
    ----------
    func : Callable, optional
        The TileLang PrimFunc to JIT-compile. If None, this function returns a 
        decorator that expects a TileLang PrimFunc.
    out_idx : Union[List[int], int], optional
        The index (or list of indices) of the function outputs. This can be used
        to specify which outputs from the compiled function will be returned.
    execution_backend : Literal["dlpack", "ctypes", "cython"], optional
        The wrapper type to use for the kernel adapter.
    target : Union[str, Target], optional
        The compilation target for TVM. If set to "auto", an appropriate target
        will be inferred automatically. Otherwise, must be one of the supported
        strings in AVAILABLE_TARGETS, a TPU target such as
        ``"tpu -mcpu=bm1690e -tpu-programming-model=rv"``, or a
        TVM Target instance.
    runtime_mode : Literal["pcie", "cmodel"], optional
        TPU execution environment. Leave unset for non-TPU targets.

    Returns
    -------
    BaseKernelAdapter
        An adapter object that encapsulates the compiled function and can be
        used to execute it.

    Raises
    ------
    ValueError
        If the provided target is an unsupported target string.
    """

    # If the target is specified as a string, ensure it is valid and convert to a TVM Target.
    if isinstance(target, str):
        if target not in AVAILABLE_TARGETS and not is_tpu_target_spec(target):
            raise ValueError(f"Invalid target: {target}")
        target = determine_target(target)

    target = Target(target)

    if execution_backend not in ("dlpack", "ctypes", "cython"):
        raise ValueError(f"Invalid execution backend: {execution_backend!r}")

    def _compile_and_create_adapter(tilelang_func: PrimFunc) -> BaseKernelAdapter:
        """
        Compile the given TileLang PrimFunc with TVM and build a kernel adapter.

        Parameters
        ----------
        tilelang_func : tvm.tir.PrimFunc
            The TileLang (TVM TIR) function to compile.

        Returns
        -------
        BaseKernelAdapter
            The compiled and ready-to-run kernel adapter.
        """
        if verbose:
            logger.info(f"Compiling TileLang function:\n{tilelang_func}")
        return compile(
            tilelang_func,
            target=target,
            verbose=verbose,
            execution_backend=execution_backend,
            out_idx=out_idx,
            runtime_mode=runtime_mode,
            **pass_config_kwargs,
        ).adapter

    # If `func` was given, compile it immediately and return the adapter.
    if func is not None:
        return _compile_and_create_adapter(func)

    # Otherwise, return a decorator that expects a function to compile.
    def real_decorator(tilelang_func: PrimFunc) -> BaseKernelAdapter:
        return _compile_and_create_adapter(tilelang_func)

    return real_decorator


def compile(
    func: PrimFunc = None,
    out_idx: Union[List[int], int, None] = None,
    execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
    target: Union[str, Target] = "auto",
    target_host: Union[str, Target] = None,
    verbose: bool = False,
    pass_configs: Optional[Dict[str, Any]] = None,
    runtime_mode: Optional[Literal["pcie", "cmodel"]] = None,
) -> JITKernel:
    """
    Compile the given TileLang PrimFunc with TVM and build a JITKernel.

    ``runtime_mode`` selects CModel or PCIe execution for TPU targets. It is
    rejected for non-TPU targets; when omitted for a TPU target, CModel is
    used. Compilation target selection remains entirely in ``target``.
    """
    return cached(
        func=func,
        out_idx=out_idx,
        execution_backend=execution_backend,
        target=target,
        target_host=target_host,
        verbose=verbose,
        pass_configs=pass_configs,
        runtime_mode=runtime_mode,
    )
