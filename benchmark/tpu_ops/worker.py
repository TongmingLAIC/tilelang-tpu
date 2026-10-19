"""One process, one SDK / frontend / runtime identity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback
import math
from .command_safety import BoardSafetyExclusion


def fixtures(case, implementation, seed):
    import torch
    from tpu_demo.common import torch_dtype
    dtype = torch_dtype(case["dtype"])
    gen = torch.Generator().manual_seed(seed)

    def rand(shape, scale=1):
        return (torch.randn(tuple(shape), generator=gen) * scale).to(dtype)

    op, s = case["op"], case["shape"]
    if op == "pipeline-add":
        a = rand(s)
        return [a, torch.zeros_like(a)], (a.float() + 0.25).to(dtype)
    if op.startswith("elementwise-"):
        a, b = rand(s), rand(s)
        if op.endswith("div"):
            b = (b.abs() + 0.5).to(dtype)
        fn = {
            "add": torch.add,
            "sub": torch.sub,
            "mul": torch.mul,
            "div": torch.div
        }[op.split("-")[-1]]
        expected = fn(a.float(), b.float()).to(dtype)
        return [a, b, torch.zeros_like(a)], expected
    if op in ("matmul", "pipeline-matmul"):
        m, n, k = s
        a, b = rand((m, k), 0.25), rand((k, n), 0.25)
        expected = (a.float() @ b.float()).to(dtype)
        return [a, b, torch.zeros((m, n), dtype=dtype)], expected
    if op.startswith("rmsnorm"):
        x = rand(s)
        weight = (0.5 + torch.rand((1, s[1]), generator=gen)).to(dtype).repeat(s[0], 1)
        expected = (x.float() * torch.rsqrt(x.float().square().mean(1, keepdim=True) + 1e-5) *
                    weight.float()).to(dtype)
        return [x, weight, torch.zeros_like(x)], expected
    if op == "swiglu":
        gate = rand(s).clamp(-3, 3)
        up = rand(s)
        expected = (torch.nn.functional.silu(gate.float()) * up.float()).to(dtype)
        return [gate, up, torch.zeros_like(up)], expected
    if op == "rope":
        from tpu_demo.rope.rope import _cosine_sine, _reference
        x = rand(s)
        cos, sin = _cosine_sine(*s)
        # Both paths receive mathematically identical angles, rounded to the
        # public dtype first. The vendor ABI requires repeated full-width tables.
        cos, sin = cos.to(dtype).float(), sin.to(dtype).float()
        expected = _reference(x, cos, sin)
        if implementation in ("ppl", "tilelang-matched"):
            cos, sin = cos.repeat_interleave(2, 1).to(dtype), sin.repeat_interleave(2, 1).to(dtype)
        return [x, cos, sin, torch.zeros_like(x)], expected
    if op == "flashattn":
        b, seq, heads, d = s
        # Keep long-context outputs well above the public absolute tolerance;
        # tiny zero-mean V would make even a zero kernel look numerically close.
        q, k, v = rand(s, 1.0), rand(s, 1.0), rand(s, 4.0)
        if seed == 17:
            # Decreasing tile maxima stress the online-softmax state update.
            q.zero_()
            q[..., 0] = 1
            k.zero_()
            k[:, :, :, 0] = (torch.linspace(3, -3, seq) * d**0.5).to(dtype)[None, :, None]
        mask = torch.zeros((seq, seq), dtype=torch.float32)
        if case.get("causal", False):
            mask.masked_fill_(torch.triu(torch.ones_like(mask, dtype=torch.bool), 1), -10000)
        # Chunk the CPU oracle to bound peak host memory for S=8192.
        output = torch.empty_like(q)
        for ib in range(b):
            for ih in range(heads):
                for im in range(0, seq, 128):
                    score = q[ib, im:im + 128, ih].float() @ k[ib, :, ih].float().T * d**-0.5
                    p = torch.softmax(score + mask[im:im + 128], dim=-1)
                    output[ib, im:im + 128, ih] = (p @ v[ib, :, ih].float()).to(dtype)
        if implementation in ("ppl", "tilelang-matched"):
            mask = mask.to(dtype)
        return [q, k, v, mask, torch.zeros_like(q)], output
    if op == "llama-mlp":
        m, h, i = s
        x = rand((m, h), 0.25)
        up, gate, down = rand((i, h), 0.02), rand((i, h), 0.02), rand((h, i), 0.02)
        expected = ((torch.nn.functional.silu(x.float() @ gate.float().T) *
                     (x.float() @ up.float().T)) @ down.float().T).to(dtype)
        if implementation in ("ppl", "tilelang-matched", "tilelang-optimized"):
            up, gate, down = up.T.contiguous(), gate.T.contiguous(), down.T.contiguous()
        return [x, up, gate, down, torch.zeros((m, h), dtype=dtype)], expected
    if op == "paged-attention":
        from tpu_demo.paged_attention.paged_attention import index_tables, torch_reference
        blocks, qh, kvh, d = s
        table = tuple(reversed(range(blocks)))
        q = rand((qh, d), 1.0)
        kc, vc = rand((blocks * kvh * 16, d), 1.0), rand((blocks * kvh * 16, d), 4.0)
        idx = index_tables(qh, kvh, table).to(torch.uint32).reshape(-1, 1)
        expected = torch_reference(
            q, kc, vc, q_heads=qh, kv_heads=kvh, block_table=table, head_dim=d).to(dtype)
        staged = qh * blocks * 16
        return [
            q, kc, vc, idx,
            idx.clone(),
            torch.zeros((staged, d), dtype=dtype),
            torch.zeros((staged, d), dtype=dtype),
            torch.zeros_like(q)
        ], expected
    raise NotImplementedError(f"fixtures missing for {op}")


def write_buffers(tensors, directory):
    manifest = []
    hashes = []
    for i, tensor in enumerate(tensors):
        data = bytes(tensor.contiguous().view(__import__('torch').uint8).numpy())
        path = directory / f"arg{i}.bin"
        path.write_bytes(data)
        manifest.append(f"{path} {len(data)}")
        hashes.append(hashlib.sha256(data).hexdigest())
    (directory / "buffers.txt").write_text("\n".join(manifest) + "\n")
    return hashes


def contract_tolerance(case):
    from tpu_demo.common import tolerance
    op = case["op"]
    family = (
        "elementwise-div" if op == "elementwise-div" else "elementwise"
        if op.startswith("elementwise-") or op == "pipeline-add" else "matmul" if op in (
            "matmul",
            "pipeline-matmul") else "rmsnorm" if op.startswith("rmsnorm") else "swiglu" if op in (
                "swiglu", "llama-mlp") else "flashattn" if op in ("flashattn",
                                                                  "paged-attention") else "rope")
    return tolerance(case["dtype"], family)


def compare_output(path, expected, case):
    import torch
    actual = torch.frombuffer(
        bytearray(path.read_bytes()), dtype=expected.dtype).reshape(expected.shape)
    a, e = actual.float(), expected.float()
    atol, rtol = contract_tolerance(case)
    close = torch.isclose(a, e, atol=atol, rtol=rtol)
    maximum = float((a - e).abs().max())
    mean = float((a - e).abs().mean())
    return {
        "passed": bool(torch.isfinite(a).all() and close.all()),
        "atol": atol,
        "rtol": rtol,
        "finite": bool(torch.isfinite(a).all()),
        "max_abs": maximum if math.isfinite(maximum) else None,
        "mean_abs": mean if math.isfinite(mean) else None,
        "mismatched": int((~close).sum()),
        "elements": a.numel()
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trial", type=Path, required=True)
    p.add_argument("--runtime", choices=("cmodel", "pcie"), required=True)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--final", action="store_true")
    p.add_argument(
        "--single-launch",
        action="store_true",
        help="Numerical canary only; no warmup, repeated timing, or PMU")
    args = p.parse_args()
    if args.single_launch and (args.final or args.profile):
        p.error('--single-launch cannot be combined with --final or --profile')
    directory = args.trial.resolve()
    spec = json.loads((directory / "trial.json").read_text())
    result = {"status": "error", "runtime": args.runtime, "trial": str(directory)}
    begin = time.monotonic()
    try:
        if args.runtime == "cmodel":
            from tilelang.jit.tpu_cmodel import require_private_cmodel
            require_private_cmodel()
            os.environ["TPU_RT_CORE_NUM"] = "4"
            # The SDK provides a functional fast arithmetic path for MM2 /
            # convolution / pooling. It still executes the complete device
            # program and all outputs must satisfy the same CPU oracle. Its
            # reduction rounding can differ from the default arithmetic path.
            # This affects validation cost only, never PCIe timing.
            os.environ.setdefault("CMODEL_FAST_EXEC", "1")
            # Each simulator also has runtime polling threads. OpenMP teams
            # for tiny vector commands can spend minutes contending with
            # them. An exact-module RoPE probe takes 0.51 s with one thread
            # and passes the full oracle; the four-thread run took minutes.
            compute_heavy = spec["case"]["op"] in ("matmul",
                                                   "llama-mlp") and spec["case"]["shape"][0] >= 2048
            if spec["case"]["op"] == "flashattn":
                kv = spec["config"]["block_n"]
                if spec["implementation"] in ('ppl', 'tilelang-matched'):
                    kv = min(kv, spec["case"]["shape"][1] // 2)
                compute_heavy = 2 * spec["config"]["block_m"] * kv * spec["case"]["shape"][
                    3] >= 1048576
            os.environ["OMP_NUM_THREADS"] = os.environ.get("TILELANG_BENCH_CMODEL_OMP_THREADS",
                                                           "4" if compute_heavy else "1")
            os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
            os.environ.setdefault("GOMP_SPINCOUNT", "0")
            result["simulated_cores"] = 4
            result["cmodel_execution"] = {
                "CMODEL_FAST_EXEC": os.environ["CMODEL_FAST_EXEC"],
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "OMP_WAIT_POLICY": os.environ.get("OMP_WAIT_POLICY"),
                "GOMP_SPINCOUNT": os.environ.get("GOMP_SPINCOUNT"),
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
            }
        else:
            # Promotion is independently verified again in the worker.
            from .command_safety import trial_exclusion
            exclusion = trial_exclusion(directory)
            prior_board = directory / 'pcie/result.json'
            if exclusion and (not prior_board.exists() or
                              json.loads(prior_board.read_text())['status'] != 'passed'):
                raise NotImplementedError(exclusion['reason'])
            gate = json.loads((directory / "cmodel" / "result.json").read_text())
            if gate["status"] != "passed" or gate["trial_sha256"] != hashlib.sha256(
                (directory / "trial.json").read_bytes()).hexdigest():
                raise RuntimeError("PCIe requires a passing cmodel result for this exact trial")
            atol, rtol = contract_tolerance(spec["case"])
            if len(gate.get("checks", [])) != len(spec.get("seeds", [0, 17])):
                raise RuntimeError("cmodel evidence is missing required complete-shape fixtures")
            for check in gate["checks"]:
                for observed in (check, check["final"]):
                    if not observed.get("passed") or not observed.get("finite"):
                        raise RuntimeError(
                            "cmodel evidence contains a failed/nonfinite numerical check")
                    if not ((observed["atol"] <= atol and observed["rtol"] <= rtol) or
                            observed["max_abs"] <= atol):
                        raise RuntimeError(
                            "cmodel evidence does not establish the current numerical contract")
            from tilelang.jit.adapter.tpu_profiling import (_assert_pcie_device_not_quarantined,
                                                            _pcie_device_session_path)
            _assert_pcie_device_not_quarantined(0)
            from .command_safety import require_board_source
            validated_source = directory / "cmodel/kernel.c"
            audit_path = directory / 'cmodel/command-epoch-audit.json'
            require_board_source(
                spec, validated_source.read_text(),
                hashlib.sha256(validated_source.read_bytes()).hexdigest(),
                json.loads(audit_path.read_text()) if audit_path.exists() else None)
            session_path = _pcie_device_session_path(0)
            session = json.loads(session_path.read_text()) if session_path.exists() else {}
            if session.get("owner_pid") != os.getppid():
                raise RuntimeError(
                    "PCIe worker requires its supervising parent to own the device session; use the run/finalize/profile controller"
                )
            os.environ["TILELANG_TPU_ALLOW_PCIE_LOAD"] = "1"
            os.environ["TILELANG_TPU_DEVICE_ID"] = "0"
            os.environ.pop("TPU_RT_CORE_NUM", None)
        from .build import build_runner, command
        target = directory / ("pcie-canary" if args.single_launch else "pcie-profile"
                              if args.profile else "pcie-final" if args.final else args.runtime)
        target.mkdir(exist_ok=True)

        def progress(phase, **details):
            (target / "progress.json").write_text(
                json.dumps(dict(phase=phase, wall_seconds=time.monotonic() - begin, **details)))

        progress("compiling")
        result["build"] = build_runner(spec["case"], spec["config"], target, spec["sdk"],
                                       spec["model"], args.runtime, spec["implementation"])
        if (args.runtime == "pcie" and gate["build"].get("canonical_device_sha256") and
                result["build"]["canonical_device_sha256"] != gate["build"]["canonical_device_sha256"]):
            raise RuntimeError(
                "cmodel and PCIe device programs differ after removing comments/whitespace")
        result["checks"] = []
        result["input_hashes"] = []
        import torch
        result["torch_version"] = torch.__version__
        result["expected_hashes"] = []
        runs = spec.get("repeats", 50) if args.runtime == "pcie" and not args.profile else 0
        if args.single_launch:
            runs = 0
        if args.final:
            runs = 100 // len(spec.get("seeds", [0, 17]))
        if args.profile:
            os.environ["BMLIB_ENABLE_ALL_PROFILE"] = "1"
        for seed in (spec.get("seeds", [0, 17])[:1]
                     if args.profile else spec.get("seeds", [0, 17])):
            import torch
            torch.set_num_threads(int(os.environ.get("TILELANG_BENCH_TORCH_THREADS", "4")))
            progress("building_oracle", seed=seed)
            tensors, expected = fixtures(spec["case"], spec["implementation"], seed)
            expected_data = bytes(expected.contiguous().view(torch.uint8).numpy())
            result["expected_hashes"].append(hashlib.sha256(expected_data).hexdigest())
            fixture_dir = target / f"seed-{seed}"
            fixture_dir.mkdir(exist_ok=True)
            result["input_hashes"].append(write_buffers(tensors, fixture_dir))
            if args.runtime == "pcie":
                index = len(result["input_hashes"]) - 1
                if result["input_hashes"][-1] != gate["input_hashes"][index]:
                    raise RuntimeError(
                        "PCIe input tensors differ from the cmodel-validated fixtures")
                if gate.get("expected_hashes"
                           ) and result["expected_hashes"][-1] != gate["expected_hashes"][index]:
                    raise RuntimeError("PCIe oracle differs from the cmodel-validated oracle")
            progress("running", seed=seed, repetitions=runs)
            output = command([
                target / "runner", target / "module.so", fixture_dir / "buffers.txt",
                10 if runs else 0, runs,
                int(args.profile)
            ],
                             target,
                             f"seed-{seed}-run",
                             timeout=spec.get("timeout", 180))
            check = compare_output(fixture_dir / f"arg{len(tensors)-1}.bin.out", expected,
                                   spec["case"])
            check["final"] = compare_output(fixture_dir / f"arg{len(tensors)-1}.bin.final",
                                            expected, spec["case"])
            result["checks"].append(check)
            (target / "checks-partial.json").write_text(
                json.dumps(
                    dict(
                        checks=result["checks"],
                        input_hashes=result["input_hashes"],
                        expected_hashes=result["expected_hashes"])))
            if not check["passed"] or not check["final"]["passed"]:
                result["status"] = "numerical_mismatch"
                break
            samples = json.loads(re.search(r"TPU_BENCH_SAMPLES_US=(\[.*\])", output).group(1))
            result.setdefault("samples_us", []).extend(samples)
            # Large deterministic fixtures are regenerated from the saved seed,
            # torch version and hashes. Preserve all failing buffers and small
            # fixtures, without storing hundreds of copies of model weights.
            for file in fixture_dir.iterdir():
                if file.name.startswith("arg") and file.stat().st_size > 4 * 1024 * 1024:
                    file.unlink()
        else:
            result["status"] = "passed"
        if result.get("samples_us"):
            import numpy as np
            result["timing"] = {
                "boundary":
                    "host kernel launch + stream synchronize, resident module/buffers; excludes H2D/D2H",
                "unit":
                    "us",
                "samples":
                    len(result["samples_us"]),
                **{
                    f"p{q}": float(np.percentile(result["samples_us"], q)) for q in (10, 50, 90, 99)
                }
            }
    except BoardSafetyExclusion as error:
        result.update(status='blocked_device_safety', reason=str(error))
    except NotImplementedError as error:
        result.update(status="unsupported", reason=str(error))
    except subprocess.TimeoutExpired as error:
        result.update(status="timeout", reason=str(error))
    except Exception as error:
        result.update(status="error", reason=str(error), traceback=traceback.format_exc())
        if any(
                message in str(error) for message in ("exceeds the TPUv7 descriptor limit",
                                                      "exceeds the TPUv7/PPL dim4 limit")):
            result.update(status="unsupported", category="frontend_descriptor_limit")
        elif spec["case"]["op"] == "gptq-w4a16" and "dq2" in str(error).lower():
            result.update(status="unsupported", category="missing_lowering")
    result["wall_seconds"] = time.monotonic() - begin
    result["trial_sha256"] = hashlib.sha256((directory / "trial.json").read_bytes()).hexdigest()
    target = directory / ("pcie-canary" if args.single_launch else "pcie-profile"
                          if args.profile else "pcie-final" if args.final else args.runtime)
    target.mkdir(exist_ok=True)
    # A concurrent coordinator must see either the queued verdict or the
    # complete numerical evidence, never a partially written JSON file.
    result_file = target / "result.json"
    temporary = target / "result.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, result_file)
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result["status"] in ("passed", "unsupported", "blocked_device_safety") else 1


if __name__ == "__main__":
    raise SystemExit(main())
