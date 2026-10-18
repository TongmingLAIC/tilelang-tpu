# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""CModel-only numerical gate for serial versus overlapped TPU-Kernel GEMM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chip", choices=("bm1690", "sg2260e"), required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--block-k", type=int, required=True)
    parser.add_argument("--pipelined", action="store_true")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--address-policy", choices=("greedy", "compact", "replay"),
                        default="greedy")
    parser.add_argument("--replay-addresses")
    parser.add_argument("--benchmark-runs", type=int, default=0,
                        help="extra launches with one device allocation for correctness gating")
    args = parser.parse_args()
    if (args.address_policy == "replay") != (args.replay_addresses is not None):
        parser.error("replay policy requires --replay-addresses, and other policies forbid it")
    if os.environ.get("TILELANG_CMODEL_PRIVATE_TMP") != "1":
        raise RuntimeError("CModel worker requires a private IPC namespace")
    os.environ["TPU_RT_CORE_NUM"] = "4" if args.chip == "sg2260e" else "8"
    if args.benchmark_runs < 0:
        parser.error("--benchmark-runs cannot be negative")
    os.environ["TILELANG_TPU_BENCHMARK_RUNS"] = str(args.benchmark_runs)

    import tilelang
    from tilelang import tvm
    from tpu_demo.common import comparison, tolerance
    from tpu_demo.matmul.matmul import build_matmul
    from tpu_demo.pipeline_matmul.pipeline_matmul import build_pipelined_matmul

    torch.set_num_threads(1)
    factory = (build_pipelined_matmul(m=args.m, n=args.n, k=args.k,
                                     block_k=args.block_k) if args.pipelined else
               build_matmul(m=args.m, n=args.n, k=args.k, block_m=args.m,
                            block_n=args.n, block_k=args.block_k,
                            dtype="float16", programming_model="tpukernel"))
    target = tvm.target.Target(f"tpu -mcpu={args.chip} -tpu-programming-model=tpukernel")
    pass_configs = {"tl.AddressAssignPolicy": args.address_policy}
    if args.replay_addresses is not None:
        pass_configs["tl.AddressAssignReplayAddresses"] = args.replay_addresses
    kernel = tilelang.compile(factory, out_idx=-1, target=target, runtime_mode="cmodel",
                              pass_configs=pass_configs)
    source = kernel.artifact.kernel_source
    if args.pipelined and "tpu_parallel_start();" not in source:
        raise AssertionError("overlapped GEMM source omitted command groups")
    if not args.pipelined and "tpu_parallel_start();" in source:
        raise AssertionError("serial GEMM source contains a command group")
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    (Path.cwd() / "kernel_source.c").write_text(source, encoding="utf-8")
    generator = torch.Generator().manual_seed(args.seed)
    a = (torch.randn((args.m, args.k), generator=generator) * 0.25).half()
    b = (torch.randn((args.k, args.n), generator=generator) * 0.25).half()
    c = torch.full((args.m, args.n), float("nan"), dtype=torch.float16)
    kernel(a, b, c)
    reference = (a.float() @ b.float()).half()
    atol, rtol = tolerance("float16", "matmul")
    metrics = comparison(c, reference, atol=atol, rtol=rtol)
    functions = tuple(kernel.artifact.host_mod.functions.values())
    if len(functions) != 1:
        raise AssertionError("expected one lowered TPU function")
    attrs = functions[0].attrs
    prefix = "tilelang.tpu.lmem.address."
    addresses = {str(key)[len(prefix):]: int(value)
                 for key, value in attrs._dict().items()
                 if str(key).startswith(prefix)}
    payload = {"status": "passed", "operation": "matmul", "chip": args.chip,
               "m": args.m, "n": args.n, "k": args.k, "block_k": args.block_k,
               "pipelined": args.pipelined, "seed": args.seed,
               "benchmark_runs": args.benchmark_runs,
               "address_policy": args.address_policy,
               "replay_addresses": args.replay_addresses,
               "kernel_source_sha256": source_hash, "address_map": addresses,
               "input_sha256": [digest(a), digest(b)],
               "reference_sha256": digest(reference), "output_sha256": digest(c),
               "metrics": metrics}
    (Path.cwd() / "numeric.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print("TPU_PIPELINE_MATMUL_RESULT=" + json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
