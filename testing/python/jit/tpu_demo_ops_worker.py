# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""One-shot worker used only by the supervised TPU demo profile matrix."""

from __future__ import annotations

import argparse
import json
import os
import zlib

from typing import Optional, Tuple

from tpu_demo.cases import CHIPS, PROGRAMMING_MODELS, build_cases, case_by_id

RESULT_PREFIX = "TPU_DEMO_RESULT="


def _selection() -> Tuple[str, str, str, bool, Optional[int]]:
    if os.environ.get("TILELANG_TPU_PROFILE_SESSION") != "1":
        raise RuntimeError("the demo worker must run through TPUInstructionProfiler")
    chip = os.environ.get("TILELANG_TPU_PROFILE_CHIP", "")
    programming_model = os.environ.get("TILELANG_TPU_PROFILE_PROGRAMMING_MODEL", "")
    runtime_mode = os.environ.get("TILELANG_TPU_PROFILE_RUNTIME_MODE", "")
    if chip not in CHIPS:
        raise RuntimeError("profile worker has no valid chip selection")
    if programming_model not in PROGRAMMING_MODELS:
        raise RuntimeError("profile worker has no valid programming-model selection")
    if runtime_mode not in ("cmodel", "pcie"):
        raise RuntimeError("profile worker has no valid runtime selection")
    if os.environ.get("TILELANG_TPU_BENCHMARK_RUNS") != "0":
        raise RuntimeError("demo profiling requires exactly one launch")
    if runtime_mode == "cmodel":
        for name in ("TILELANG_TPU_ALLOW_PCIE_LOAD", "TILELANG_TPU_ALLOW_PCIE_PROFILE",
                     "TILELANG_TPU_DEVICE_ID"):
            if name in os.environ:
                raise RuntimeError(f"CModel worker inherited forbidden PCIe setting {name}")
        return chip, programming_model, runtime_mode, False, None
    for name in ("TILELANG_TPU_ALLOW_PCIE_LOAD", "TILELANG_TPU_ALLOW_PCIE_PROFILE",
                 "BMLIB_ENABLE_ALL_PROFILE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"PCIe profile worker is missing {name}=1")
    raw_device_id = os.environ.get("TILELANG_TPU_DEVICE_ID", "")
    if chip != "bm1690e":
        raise RuntimeError("this machine's PCIe demo worker accepts only BM1690E")
    if (not raw_device_id.isascii() or not raw_device_id.isdecimal() or
            int(raw_device_id) > 2**31 - 1):
        raise RuntimeError("PCIe profile worker requires a non-negative 32-bit device id")
    if "TPU_RT_CORE_NUM" in os.environ:
        raise RuntimeError("PCIe profile worker inherited the CModel core-topology setting")
    return chip, programming_model, runtime_mode, True, int(raw_device_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=tuple(case.case_id for case in build_cases()), required=True)
    args = parser.parse_args()
    chip, programming_model, runtime_mode, allow_pcie, device_id = _selection()
    case = case_by_id(args.case)
    if programming_model == "rv" and not case.supports_rv:
        raise RuntimeError(f"{case.operation} has no RV Tensor semantic lowering")

    try:
        # Delay TileLang/Torch imports until the full profiler-created environment
        # and case/backend capability contract have been validated.
        from tpu_demo.run import run_case
        seed = zlib.crc32(args.case.encode("utf-8")) & 0x7FFFFFFF
        result = run_case(
            args.case,
            chip=chip,
            programming_model=programming_model,
            runtime_mode=runtime_mode,
            allow_pcie=allow_pcie,
            device_id=device_id,
            seed=seed,
        )
    except Exception as error:
        failed = {
            "status": "failed",
            "case_id": args.case,
            "chip": chip,
            "programming_model": programming_model,
            "runtime_mode": runtime_mode,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        metrics = getattr(error, "metrics", None)
        if isinstance(metrics, dict):
            failed["metrics"] = metrics
        print(RESULT_PREFIX + json.dumps(failed, sort_keys=True, allow_nan=False), flush=True)
        raise
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
