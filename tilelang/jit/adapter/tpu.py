# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Safe PyTorch-facing invocation for the TileLang TPU host ABI."""

import ctypes
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .utils import is_tpu_target

_TPU_EXECUTION_LOCK = threading.RLock()


@dataclass(frozen=True)
class _TPURuntimeIdentity:
    """One process-global vendor runtime identity.

    The vendor API exposed by generated ``main.so`` has no verified process
    teardown call.  A process must therefore not switch simulator/PCIe mode,
    chip topology, programming model, PCIe device, or PPL SDK/runtime identity
    after its first TPU library is loaded.
    """

    runtime_mode: str
    chip: str
    core_count: int
    programming_model: str
    device_id: int
    sdk_identity: Tuple[str, str, str]


_TPU_RUNTIME_IDENTITY: Optional[_TPURuntimeIdentity] = None


def reserve_tpu_runtime_identity(tpu_target, tpu_runtime, device_id: int,
                                 sdk_identity: Tuple[str, str, str]) -> None:
    """Reserve the vendor runtime identity before loading a TPU library.

    CModel and PCIe share process-global vendor runtime state.  Since the
    generated host module has no verified `tpuRtDeInit` lifecycle, letting a
    process switch runtime mode, chip/core topology, programming model, PCIe
    device, or the PPL runtime libraries can reuse stale state despite
    unloading a module.  Fail closed before ``ctypes.CDLL``; use a fresh
    process for every different identity.
    """
    if isinstance(device_id, bool) or not isinstance(device_id, int) or device_id < 0:
        raise ValueError(f"TPU runtime device id must be a non-negative int, got {device_id!r}")
    if len(sdk_identity) != 3:
        raise ValueError("TPU runtime SDK identity must contain root, runtime, and backend paths")
    identity = _TPURuntimeIdentity(
        runtime_mode=tpu_runtime.runtime_mode,
        chip=tpu_target.chip,
        core_count=tpu_target.chip_spec.physical_core_count,
        programming_model=tpu_target.programming_model,
        device_id=device_id,
        sdk_identity=sdk_identity,
    )
    global _TPU_RUNTIME_IDENTITY
    with _TPU_EXECUTION_LOCK:
        if _TPU_RUNTIME_IDENTITY is None:
            _TPU_RUNTIME_IDENTITY = identity
        elif identity != _TPU_RUNTIME_IDENTITY:
            raise RuntimeError("TileLang TPU runtime is already reserved for "
                               f"runtime={_TPU_RUNTIME_IDENTITY.runtime_mode}, "
                               f"chip={_TPU_RUNTIME_IDENTITY.chip}, "
                               f"cores={_TPU_RUNTIME_IDENTITY.core_count}, "
                               f"programming_model={_TPU_RUNTIME_IDENTITY.programming_model}, "
                               f"device={_TPU_RUNTIME_IDENTITY.device_id}, "
                               f"sdk={_TPU_RUNTIME_IDENTITY.sdk_identity[0]}; cannot load "
                               f"runtime={identity.runtime_mode}, chip={identity.chip}, "
                               f"cores={identity.core_count}, "
                               f"programming_model={identity.programming_model}, "
                               f"device={identity.device_id}, sdk={identity.sdk_identity[0]} "
                               "in the same process. "
                               "Use a fresh process for another TPU runtime identity.")


def reject_unverified_tpu_database_artifact(target) -> None:
    """Fail closed for TPU artifacts that lack a verified sidecar manifest.

    A TPU ``main.so`` embeds an absolute private ``libkernel.so`` path.  The
    generic cache/database format copies only the host library and does not
    bind it cryptographically to chip/device/runtime metadata.  In particular,
    treating a PCIe artifact as CModel could bypass the Python dlopen gate.
    Rebuild the artifact until a bundled manifest format exists.
    """
    if is_tpu_target(target):
        raise RuntimeError("Loading TPU artifacts from cache/database is disabled until a "
                           "chip/device/runtime manifest and bundled private libkernel.so are "
                           "implemented. Recompile the TPU kernel in this process instead.")


def _param_dtype(param):
    """Return a torch dtype for a KernelParam-like object."""
    dtype = param.dtype
    if isinstance(dtype, str):
        return getattr(torch, dtype)
    return dtype


def _resolve_shape(index: int, params: Sequence, args: Sequence,
                   dynamic_symbolic_map: Dict) -> Tuple[int, ...]:
    shape = []
    for dim in params[index].shape:
        if isinstance(dim, int):
            shape.append(dim)
            continue
        if hasattr(dim, "value") and isinstance(dim.value, int):
            shape.append(dim.value)
            continue
        if dim not in dynamic_symbolic_map:
            raise ValueError(
                f"Cannot resolve dynamic TPU tensor dimension {dim!r} for parameter {index}.")
        reference_index, reference_dim = dynamic_symbolic_map[dim]
        reference = args[reference_index]
        if not isinstance(reference, torch.Tensor):
            raise ValueError(
                f"Dynamic TPU tensor dimension {dim!r} refers to unavailable parameter "
                f"{reference_index}.")
        shape.append(int(reference.shape[reference_dim]))
    return tuple(shape)


def _validate_tensor(index: int, tensor: torch.Tensor, params: Sequence, args: Sequence,
                     dynamic_symbolic_map: Dict) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("TileLang TPU JIT currently accepts Tensor arguments only; "
                        f"parameter {index} received {type(tensor).__name__}.")
    expected_dtype = _param_dtype(params[index])
    if tensor.dtype != expected_dtype:
        raise TypeError(
            f"TPU parameter {index} has dtype {tensor.dtype}, expected {expected_dtype}.")
    expected_shape = _resolve_shape(index, params, args, dynamic_symbolic_map)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"TPU parameter {index} has shape {tuple(tensor.shape)}, "
                         f"expected {expected_shape}.")
    if tensor.numel() == 0:
        raise ValueError("TileLang TPU JIT does not support zero-sized Tensor arguments.")


def _reject_storage_aliases(args: Sequence[torch.Tensor]) -> None:
    """Reject aliases until the TPU host ABI can preserve pointer offsets.

    Each PrimFunc parameter currently receives an independently allocated host
    staging buffer and device allocation.  Passing two views of one storage
    would therefore change alias semantics, and D2S could overwrite a prior
    result in parameter order.  Conservatively reject shared storage (even
    non-overlapping views) rather than silently miscompile it.
    """
    owners = {}
    for index, tensor in enumerate(args):
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        previous = owners.get(key)
        if previous is not None:
            raise ValueError("TileLang TPU JIT does not support aliased Tensor storage across "
                             f"parameters ({previous} and {index}); pass independent buffers.")
        owners[key] = index


def make_tpu_forward(lib: ctypes.CDLL, params: Sequence, result_idx: List[int],
                     dynamic_symbolic_map: Dict) -> Callable:
    """Bind ``tilelang_tpu_run`` with normal TileLang output semantics.

    The host ABI expects one pointer for every PrimFunc parameter.  TileLang's
    public JIT API, however, accepts input tensors and allocates ``result_idx``
    tensors itself.  This bridge reconciles the two without copying whole
    storages: a contiguous view may have a non-zero storage offset, so each
    staging buffer is copied from ``data_ptr()`` for exactly ``numel * itemsize``
    bytes.
    """
    run = lib.tilelang_tpu_run
    run.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    run.restype = ctypes.c_int
    result_idx = list(result_idx)
    output_indices = set(result_idx)

    def forward(*ins):
        expected_inputs = len(params) - len(result_idx)
        # Prefer the explicit full-parameter convention when both forms are
        # numerically possible (notably result_idx == []).  Otherwise an
        # opaque extern cannot publish mutations through a caller-provided
        # buffer, because the input-only branch would have no copy-back set.
        if len(ins) == len(params):
            # Preserve the previous TPU adapter convention for callers that
            # explicitly provide output tensors.  Opaque device externs do
            # not necessarily appear in TileLang's result-index analysis, so
            # copy every explicit tensor back after a successful run.  This
            # also preserves intentional in-place kernels.
            args = list(ins)
            copy_back_indices = list(range(len(params)))
        elif len(ins) == expected_inputs:
            args = [None] * len(params)
            input_index = 0
            for index in range(len(params)):
                if index not in output_indices:
                    args[index] = ins[input_index]
                    input_index += 1
            output_device = ins[0].device if ins else torch.device("cpu")
            for index in result_idx:
                shape = _resolve_shape(index, params, args, dynamic_symbolic_map)
                args[index] = torch.empty(
                    shape, dtype=_param_dtype(params[index]), device=output_device)
            copy_back_indices = result_idx
        else:
            raise ValueError(
                f"TPU kernel expects {expected_inputs} input tensors "
                f"({len(params)} total parameters, outputs={result_idx}), got {len(ins)}.")

        for index, tensor in enumerate(args):
            _validate_tensor(index, tensor, params, args, dynamic_symbolic_map)
        _reject_storage_aliases(args)

        host_tensors = [tensor.detach().to(device="cpu").contiguous() for tensor in args]
        arg_buffers = []
        arg_sizes = []
        for tensor in host_tensors:
            nbytes = tensor.numel() * tensor.element_size()
            buffer = ctypes.create_string_buffer(nbytes)
            ctypes.memmove(buffer, tensor.data_ptr(), nbytes)
            arg_buffers.append(buffer)
            arg_sizes.append(nbytes)

        argv = (ctypes.c_void_p * len(args))()
        for index, buffer in enumerate(arg_buffers):
            argv[index] = ctypes.cast(buffer, ctypes.c_void_p).value

        # main.so owns process-global TPU runtime objects. ctypes may release
        # the GIL during this call, so serialize all adapters in this process.
        with _TPU_EXECUTION_LOCK:
            owner = getattr(lib, "_tilelang_tpu_workspace_owner", None)
            if owner is not None and owner.tpu_runtime.runtime_mode == "cmodel":
                from tilelang.jit.tpu_cmodel import require_private_cmodel
                require_private_cmodel()
            status = run(argv)
        if status != 0:
            raise RuntimeError(f"TileLang TPU execution failed with status {status}.")

        for index in copy_back_indices:
            output = args[index]
            host_output = torch.empty(tuple(output.shape), dtype=output.dtype, device="cpu")
            ctypes.memmove(host_output.data_ptr(), arg_buffers[index], arg_sizes[index])
            with torch.no_grad():
                output.copy_(host_output.to(device=output.device))

        if len(result_idx) == 1:
            return args[result_idx[0]]
        if result_idx:
            return [args[index] for index in result_idx]
        return status

    return forward
