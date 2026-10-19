# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from typing import Literal, Union
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.contrib import rocm
from tilelang.contrib import nvcc

AVAILABLE_TARGETS = {
    "auto",
    "cuda",
    "hip",
    "webgpu",
    "c",  # represent c source backend
    "llvm",
}


def is_tpu_target_spec(target: Union[str, Target]) -> bool:
    """Whether ``target`` names the TPU target kind, including chip variants.

    TPU selection uses explicit ``-mcpu`` and ``-tpu-programming-model``
    attributes. Keep this narrow exception to the fixed allow-list: arbitrary
    target strings remain rejected, while TPU gets the same target-specialization
    spelling used by the GPU backends.
    """
    if isinstance(target, Target):
        return target.kind.name == "tpu"
    if not isinstance(target, str):
        return False
    try:
        return Target(target).kind.name == "tpu"
    except (TypeError, ValueError):
        return False


def check_cuda_availability() -> bool:
    """
    Check if CUDA is available on the system by locating the CUDA path.
    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    try:
        nvcc.find_cuda_path()
        return True
    except Exception:
        return False


def check_hip_availability() -> bool:
    """
    Check if HIP (ROCm) is available on the system by locating the ROCm path.
    Returns:
        bool: True if HIP is available, False otherwise.
    """
    try:
        rocm.find_rocm_path()
        return True
    except Exception:
        return False


def determine_target(target: Union[str, Target, Literal["auto"]] = "auto",
                     return_object: bool = False) -> Union[str, Target]:
    """
    Determine the appropriate target for compilation.

    Args:
        target (Union[str, Target, Literal["auto"]]): User-specified target.
            - If "auto", CUDA or HIP is selected when available. TPU is never
              inferred because its chip and programming model must be explicit.
            - If a string or Target, it is directly validated.

    Returns:
        Union[str, Target]: The selected target or a valid Target object.

    Raises:
        ValueError: If the target is invalid or no automatic target is available.
    """

    return_var: Union[str, Target] = target

    if target == "auto":
        # Check for CUDA and HIP availability
        is_cuda_available = check_cuda_availability()
        is_hip_available = check_hip_availability()

        # Determine the target based on availability
        if is_cuda_available:
            return_var = "cuda"
        elif is_hip_available:
            return_var = "hip"
        else:
            # TPU compilation requires a complete explicit Target.  Environment
            # variables are toolchain inputs, not a second target selector.
            # The C backend is still available when requested explicitly, but
            # TileLang GPU programs cannot be assumed to be portable C programs.
            raise ValueError("No CUDA or HIP target is available. Select 'c' explicitly for "
                             "portable C code, or provide a complete TPU target such as "
                             "'tpu -mcpu=bm1690e -tpu-programming-model=rv'.")
    else:
        # Validate the target if it's not "auto"
        if not (isinstance(target, Target) or target in AVAILABLE_TARGETS or
                is_tpu_target_spec(target)):
            raise ValueError(f"Target {target} is not supported")
        return_var = target

    if return_object:
        return Target(return_var)
    return return_var
