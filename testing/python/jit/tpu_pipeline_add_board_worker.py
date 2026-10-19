# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""One supervised BM1690E numerical and timing session for pipeline add."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmodel-json", required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--measurements", type=int, required=True)
    args = parser.parse_args()
    if (os.environ.get("TILELANG_TPU_BOARD_VALIDATION_SESSION") != "1" or
            os.environ.get("TILELANG_TPU_ALLOW_PCIE_LOAD") != "1" or
            os.environ.get("TILELANG_TPU_DEVICE_ID") != "0" or
            os.environ.get("TILELANG_TPU_PROFILE_SESSION") is not None):
        raise RuntimeError("board worker requires the non-profile supervised device session")
    if args.warmups < 0 or args.measurements <= 0:
        raise ValueError("invalid measurement repetitions")

    import torch
    import tilelang
    from tilelang import tvm
    from tpu_demo.pipeline_add.pipeline_add import build_pipeline_add

    torch.set_num_threads(1)
    record = json.loads(open(args.cmodel_json, encoding="utf-8").read())
    if record.get("status") != "passed" or record.get("chip") != "bm1690e" or not record.get("exact"):
        raise RuntimeError("board candidate has no successful BM1690E CModel gate")
    target = tvm.target.Target("tpu -mcpu=bm1690e -tpu-programming-model=tpukernel")
    kernel = tilelang.compile(
        build_pipeline_add(tiles=record["tiles"], tile_width=record["tile_width"],
                           pipelined=record["pipelined"]),
        out_idx=-1, target=target, runtime_mode="pcie",
        pass_configs={"tl.AddressAssignPolicy": record["address_policy"]})
    source_hash = hashlib.sha256(kernel.artifact.kernel_source.encode()).hexdigest()
    if source_hash != record["kernel_source_sha256"]:
        raise RuntimeError("board kernel source differs from CModel-validated source")
    functions = tuple(kernel.artifact.host_mod.functions.values())
    if len(functions) != 1:
        raise AssertionError("expected one lowered TPU function")
    attrs = functions[0].attrs
    prefix = "tilelang.tpu.lmem.address."
    addresses = {str(key)[len(prefix):]: int(value)
                 for key, value in attrs._dict().items()
                 if str(key).startswith(prefix)}
    if addresses != record["address_map"]:
        raise RuntimeError("board address map differs from CModel-validated layout")

    width = record["tiles"] * record["tile_width"]
    generator = torch.Generator().manual_seed(record["seed"])
    source = (torch.randn((1, width), generator=generator) * 0.25).half()
    destination = torch.full_like(source, float("nan"))
    expected = (source.float() + 0.25).half()
    host_samples_ms = []
    for launch in range(1 + args.warmups + args.measurements):
        started = time.perf_counter_ns()
        kernel(source, destination)
        elapsed = (time.perf_counter_ns() - started) / 1e6
        if launch >= 1 + args.warmups:
            host_samples_ms.append(elapsed)
    if not torch.equal(destination, expected):
        error = (destination.float() - expected.float()).abs().max().item()
        raise AssertionError(f"board numerical mismatch: max_abs_error={error}")
    output_hash = hashlib.sha256(destination.view(torch.uint8).numpy().tobytes()).hexdigest()
    if output_hash != record["output_sha256"]:
        raise AssertionError("board output differs from CModel-validated output")
    payload = {"status": "passed", "chip": "bm1690e", "tiles": record["tiles"],
               "tile_width": record["tile_width"], "pipelined": record["pipelined"],
               "address_policy": record["address_policy"],
               "kernel_source_sha256": source_hash, "address_map": addresses,
               "output_sha256": output_hash, "host_samples_ms": host_samples_ms,
               "warmups": args.warmups, "measurements": args.measurements,
               "numerical_exact": True}
    # The generated host uses C stdio while this worker emits its structured
    # result through Python stdout. Drain C's buffer first so the two outputs
    # cannot splice into one another when stdout is a supervisor pipe.
    ctypes.CDLL(None).fflush(None)
    print("TPU_PIPELINE_BOARD_RESULT=" + json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
