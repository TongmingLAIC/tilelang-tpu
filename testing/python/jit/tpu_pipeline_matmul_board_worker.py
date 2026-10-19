# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""One supervised BM1690E session for the serial/pipelined GEMM experiment."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

import torch


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmodel-json", type=Path, required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--measurements", type=int, required=True)
    parser.add_argument("--persistent-benchmark", action="store_true")
    args = parser.parse_args()
    profiled = os.environ.get("TILELANG_TPU_PROFILE_SESSION") == "1"
    if (os.environ.get("TILELANG_TPU_ALLOW_PCIE_LOAD") != "1" or
            os.environ.get("TILELANG_TPU_DEVICE_ID") != "0" or
            (profiled and os.environ.get("TILELANG_TPU_ALLOW_PCIE_PROFILE") != "1") or
            (not profiled and
             os.environ.get("TILELANG_TPU_BOARD_VALIDATION_SESSION") != "1")):
        raise RuntimeError("board worker requires a supervised device session")
    if args.warmups < 0 or args.measurements <= 0:
        raise ValueError("invalid measurement repetitions")
    if profiled and (args.warmups != 0 or args.measurements != 1):
        raise ValueError("a device profile must execute exactly one kernel launch")
    if args.persistent_benchmark and (profiled or args.warmups != 0):
        raise ValueError("same-allocation timing requires no profiler and zero outer warmups")

    import tilelang
    from tilelang import tvm
    from tpu_demo.common import comparison
    from tpu_demo.matmul.matmul import build_matmul
    from tpu_demo.pipeline_matmul.pipeline_matmul import build_pipelined_matmul

    torch.set_num_threads(1)
    record = json.loads(args.cmodel_json.read_text())
    if (record.get("status") != "passed" or record.get("chip") != "bm1690e" or
            record.get("operation") != "matmul" or
            not record.get("metrics", {}).get("passed")):
        raise RuntimeError("board GEMM candidate has no successful BM1690E CModel gate")
    if args.persistent_benchmark and record.get("benchmark_runs", 0) < 1:
        raise RuntimeError("same-allocation board timing needs repeated-launch CModel validation")
    m, n, k, block_k = (record[name] for name in ("m", "n", "k", "block_k"))
    factory = (build_pipelined_matmul(m=m, n=n, k=k, block_k=block_k)
               if record["pipelined"] else
               build_matmul(m=m, n=n, k=k, block_m=m, block_n=n,
                            block_k=block_k, dtype="float16",
                            programming_model="tpukernel"))
    target = tvm.target.Target("tpu -mcpu=bm1690e -tpu-programming-model=tpukernel")
    pass_configs = {"tl.AddressAssignPolicy": record["address_policy"]}
    if record.get("replay_addresses") is not None:
        if record["address_policy"] != "replay":
            raise RuntimeError("numerical gate supplied replay addresses for a non-replay policy")
        pass_configs["tl.AddressAssignReplayAddresses"] = record["replay_addresses"]
    kernel = tilelang.compile(factory, out_idx=-1, target=target, runtime_mode="pcie",
                              pass_configs=pass_configs)
    source_hash = hashlib.sha256(kernel.artifact.kernel_source.encode()).hexdigest()
    if source_hash != record["kernel_source_sha256"]:
        raise RuntimeError("board GEMM source differs from CModel-validated source")
    functions = tuple(kernel.artifact.host_mod.functions.values())
    if len(functions) != 1:
        raise AssertionError("expected one lowered TPU function")
    attrs = functions[0].attrs
    prefix = "tilelang.tpu.lmem.address."
    addresses = {str(key)[len(prefix):]: int(value)
                 for key, value in attrs._dict().items()
                 if str(key).startswith(prefix)}
    if addresses != record["address_map"]:
        raise RuntimeError("board GEMM address map changed since CModel validation")

    generator = torch.Generator().manual_seed(record["seed"])
    a = (torch.randn((m, k), generator=generator) * 0.25).half()
    b = (torch.randn((k, n), generator=generator) * 0.25).half()
    reference = (a.float() @ b.float()).half()
    if [digest(a), digest(b)] != record["input_sha256"] or \
            digest(reference) != record["reference_sha256"]:
        raise RuntimeError("board numerical input/reference differs from CModel gate")
    c = torch.full((m, n), float("nan"), dtype=torch.float16)
    host_samples_ms = []
    launch_count = 1 if profiled or args.persistent_benchmark else 1 + args.warmups + args.measurements
    for launch in range(launch_count):
        started = time.perf_counter_ns()
        kernel(a, b, c)
        elapsed = (time.perf_counter_ns() - started) / 1e6
        if not profiled and launch >= 1 + args.warmups:
            host_samples_ms.append(elapsed)
    metrics = comparison(c, reference, atol=record["metrics"]["atol"],
                         rtol=record["metrics"]["rtol"])
    output_hash = digest(c)
    if output_hash != record["output_sha256"]:
        raise AssertionError("board GEMM output differs from CModel-validated output")
    payload = {"status": "passed", "operation": "matmul", "chip": "bm1690e",
               "m": m, "n": n, "k": k, "block_k": block_k,
               "pipelined": record["pipelined"],
               "address_policy": record["address_policy"],
               "kernel_source_sha256": source_hash, "address_map": addresses,
               "output_sha256": output_hash, "metrics": metrics,
               "host_samples_ms": host_samples_ms,
               "persistent_benchmark": args.persistent_benchmark,
               "warmups": args.warmups, "measurements": args.measurements}
    ctypes.CDLL(None).fflush(None)
    print("TPU_PIPELINE_BOARD_RESULT=" + json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
