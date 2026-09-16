# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Run one import-safe TPU demo and check it against a PyTorch oracle."""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from tpu_demo.cases import CHIPS, PROGRAMMING_MODELS, build_cases, case_by_id


def run_case(case_id: str,
             *,
             chip: str,
             programming_model: str,
             runtime_mode: str,
             allow_pcie: bool = False,
             device_id: Optional[int] = None,
             seed: int = 0) -> dict:
    case = case_by_id(case_id)
    common = {
        "dtype": case.dtype,
        "chip": chip,
        "programming_model": programming_model,
        "runtime_mode": runtime_mode,
        "allow_pcie": allow_pcie,
        "device_id": device_id,
        "seed": seed,
    }
    if case.operation.startswith("elementwise-"):
        from tpu_demo.elementwise import run
        return run(operation=case.operation[len("elementwise-"):], **common)
    if case.operation == "matmul":
        from tpu_demo.matmul import run
        return run(**common)
    if case.operation in ("rmsnorm", "rmsnorm-splitk"):
        from tpu_demo.rmsnorm import run
        return run(split_k=case.operation.endswith("-splitk"), **common)
    if case.operation == "rope":
        from tpu_demo.rope import run
        return run(**common)
    if case.operation == "swiglu":
        from tpu_demo.swiglu import run
        return run(**common)
    if case.operation == "flashattn":
        from tpu_demo.flashattn import run
        return run(variant=case.variant, is_causal=case.is_causal, **common)
    raise AssertionError(f"unhandled registered operation {case.operation!r}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=tuple(case.case_id for case in build_cases()), required=True)
    parser.add_argument("--chip", choices=CHIPS, default="sg2260e")
    parser.add_argument("--programming-model", choices=PROGRAMMING_MODELS, default="tpukernel")
    parser.add_argument(
        "--runtime-mode",
        choices=("cmodel",),
        default="cmodel",
        help="board execution is available only through tpu_demo_ops_matrix.py")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = run_case(
            args.case,
            chip=args.chip,
            programming_model=args.programming_model,
            runtime_mode=args.runtime_mode,
            allow_pcie=False,
            device_id=None,
            seed=args.seed,
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
