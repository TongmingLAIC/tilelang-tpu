# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""CModel-only numerical gate for the TPU-Kernel command-group experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chip", choices=("bm1690", "sg2260e"), required=True)
    parser.add_argument("--tiles", type=int, required=True)
    parser.add_argument("--tile-width", type=int, required=True)
    parser.add_argument("--pipelined", action="store_true")
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--address-policy", choices=("greedy", "compact", "portfolio"),
                        default="greedy")
    args = parser.parse_args()
    if os.environ.get("TILELANG_CMODEL_PRIVATE_TMP") != "1":
        raise RuntimeError("pipeline CModel worker requires private IPC isolation")
    os.environ["TPU_RT_CORE_NUM"] = "4" if args.chip == "sg2260e" else "8"
    os.environ["TILELANG_TPU_BENCHMARK_RUNS"] = "0"

    import torch
    import tilelang
    from tilelang import tvm
    from tpu_demo.pipeline_add.pipeline_add import build_pipeline_add

    torch.set_num_threads(1)
    target = tvm.target.Target(
        f"tpu -mcpu={args.chip} -tpu-programming-model=tpukernel")
    kernel = tilelang.compile(
        build_pipeline_add(tiles=args.tiles, tile_width=args.tile_width,
                           pipelined=args.pipelined),
        out_idx=-1, target=target, runtime_mode="cmodel",
        pass_configs={"tl.AddressAssignPolicy": args.address_policy})
    source = kernel.artifact.kernel_source
    if args.pipelined and ("tpu_parallel_start();" not in source or
                           "tpu_parallel_end();" not in source):
        raise AssertionError("parallel commands disappeared during code generation")
    if not args.pipelined and "tpu_parallel_start();" in source:
        raise AssertionError("serial baseline unexpectedly contains parallel commands")
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    (Path.cwd() / "kernel_source.c").write_text(source, encoding="utf-8")

    width = args.tiles * args.tile_width
    generator = torch.Generator().manual_seed(args.seed)
    source_tensor = (torch.randn((1, width), generator=generator) * 0.25).half()
    destination = torch.full_like(source_tensor, float("nan"))
    kernel(source_tensor, destination)
    expected = (source_tensor.float() + 0.25).half()
    actual = destination.float()
    error = (actual - expected.float()).abs()
    exact = bool(torch.equal(destination, expected))
    if not exact:
        raise AssertionError(
            f"pipeline add numerical mismatch: max_abs_error={error.max().item()}")
    functions = tuple(kernel.artifact.host_mod.functions.values())
    if len(functions) != 1:
        raise AssertionError("expected one lowered TPU function")
    attrs = functions[0].attrs
    prefix = "tilelang.tpu.lmem.address."
    addresses = {str(key)[len(prefix):]: int(value)
                 for key, value in attrs._dict().items()
                 if str(key).startswith(prefix)}
    payload = {
        "status": "passed",
        "chip": args.chip,
        "tiles": args.tiles,
        "tile_width": args.tile_width,
        "pipelined": args.pipelined,
        "seed": args.seed,
        "address_policy": args.address_policy,
        "kernel_source_sha256": source_hash,
        "address_map": addresses,
        "max_abs_error": float(error.max().item()),
        "exact": exact,
        "output_sha256": hashlib.sha256(
            destination.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }
    (Path.cwd() / "numeric.json").write_text(json.dumps(payload, indent=2,
                                                          sort_keys=True) + "\n")
    print("TPU_PIPELINE_RESULT=" + json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
