"""Serial cmodel-first search with resumable per-candidate evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import contextlib

ROOT = Path(__file__).resolve().parents[2]


def _terminate_group(proc):
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=3)


def _record_board_failure(path, reason, **details):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    if path.exists():
        previous = path.with_name(f"result-before-device-fault-{time.time_ns()}.json")
        path.rename(previous)
    temporary = path.with_suffix('.fault.tmp')
    temporary.write_text(
        json.dumps(
            dict(
                status='device_fault',
                reason=reason,
                previous_result=str(previous) if previous else None,
                **details),
            indent=2) + '\n')
    os.replace(temporary, path)


def supervised(command, log, env, timeout, board_result=None):
    with Path(log).open("w") as output:
        proc = subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=ROOT,
            start_new_session=True)
        try:
            if board_result is None:
                return proc.wait(timeout=timeout)
            # The calling thread owns the device lock. Poll here rather than
            # in a background thread so a Fault can durably quarantine that
            # session before its process group is stopped.
            board_result = Path(board_result)
            deadline = time.monotonic() + timeout
            while proc.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    return proc.wait(timeout=min(5, remaining))
                except subprocess.TimeoutExpired:
                    reason = None
                    try:
                        monitor = subprocess.run(
                            ["/opt/tpuv7/tpuv7-current/bin/tpu-smi", "--noloop", "--json_format"],
                            env=dict(env, LD_LIBRARY_PATH="/opt/tpuv7/tpuv7-current/lib"),
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=10)
                        board_result.parent.mkdir(parents=True, exist_ok=True)
                        (board_result.parent / "health-inflight.json").write_text(monitor.stdout)
                        chip = json.loads(monitor.stdout)["card0"]["chip0"]
                        if monitor.returncode or chip["status"] != "Active":
                            reason = f"in-flight device health failed: {chip}"
                    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError) as error:
                        reason = f"cannot establish in-flight device health: {error}"
                    if reason:
                        from tilelang.jit.adapter.tpu_profiling import TPUInstructionProfiler
                        TPUInstructionProfiler.fail_closed_pcie_device(
                            0, reason=reason, process_group=proc.pid)
                        _terminate_group(proc)
                        _record_board_failure(
                            board_result,
                            reason,
                            process_group=proc.pid,
                            health_evidence=str(board_result.parent / "health-inflight.json"))
                        return 125
            return proc.returncode
        except subprocess.TimeoutExpired:
            if board_result is not None:
                from tilelang.jit.adapter.tpu_profiling import TPUInstructionProfiler
                reason = "board supervisor deadline exceeded"
                TPUInstructionProfiler.fail_closed_pcie_device(
                    0, reason=reason, process_group=proc.pid)
                _record_board_failure(board_result, reason, process_group=proc.pid)
            _terminate_group(proc)
            return 124


def health(directory, label, result_path=None):
    from tilelang.jit.adapter.tpu_profiling import TPUInstructionProfiler
    command = ["/opt/tpuv7/tpuv7-current/bin/tpu-smi", "--noloop", "--json_format"]
    env = dict(os.environ, LD_LIBRARY_PATH="/opt/tpuv7/tpuv7-current/lib")
    try:
        result = subprocess.run(
            command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        Path(directory, f"health-{label}.json").write_text(result.stdout)
        data = json.loads(result.stdout)
        chip = data["card0"]["chip0"]
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError) as error:
        TPUInstructionProfiler.fail_closed_pcie_device(
            0, reason=f"Cannot establish board health at {label}: {error}; evidence in {directory}")
        if result_path is not None:
            _record_board_failure(result_path, f"Cannot establish postflight health: {error}")
        raise RuntimeError("cannot establish device health; quarantined") from error
    # Utilization is averaged over the preceding sampling window and can
    # remain nonzero just after a successful worker has released all memory.
    if result.returncode or chip.get("status") != "Active" or chip.get("mem_usage") != "0MB":
        TPUInstructionProfiler.fail_closed_pcie_device(
            0, reason=f"Board health check {label} failed: {chip}; evidence in {directory}")
        if result_path is not None:
            _record_board_failure(result_path, f"Postflight board health failed: {chip}")
        raise RuntimeError(f"device not idle/active: {chip}")
    return chip


def execute_trial(trial, allow_pcie, cmodel_cpus=None):
    from .command_safety import trial_exclusion, BoardSafetyExclusion
    spec = json.loads((trial / "trial.json").read_text())
    exclusion = trial_exclusion(trial)
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT),
        PPL_PROJECT_ROOT=spec["sdk"],
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4")
    for name in list(env):
        if name.startswith("TILELANG_TPU_PROFILE") or name in ("BMLIB_ENABLE_ALL_PROFILE",
                                                               "FILE_DUMP_CMD"):
            env.pop(name, None)
    result_path = trial / "cmodel/result.json"
    if not result_path.exists() and exclusion.get('skip_cmodel'):
        return 'skipped_device_safety'
    if not result_path.exists():
        cmd = [
            sys.executable, "-m", "tilelang.jit.tpu_cmodel", "--output-dir",
            str(trial / "cmodel"), "--timeout",
            str(spec["timeout"] + 60), "--", sys.executable, "-m", "benchmark.tpu_ops.worker",
            "--trial",
            str(trial), "--runtime", "cmodel"
        ]
        if cmodel_cpus:
            cmd = ["taskset", "-c", cmodel_cpus, *cmd]
        rc = supervised(cmd, trial / "cmodel-launch.log", env, spec["timeout"] + 90)
        if not result_path.exists():
            (trial / "cmodel").mkdir(exist_ok=True)
            result_path.write_text(
                json.dumps(dict(status="timeout" if rc == 124 else "launch_failed", returncode=rc)))
    result = json.loads(result_path.read_text())
    if result["status"] != "passed" or not allow_pcie:
        return result["status"]
    if (trial / "pcie/result.json").exists():
        return json.loads((trial / "pcie/result.json").read_text())["status"]
    if exclusion:
        return 'blocked_device_safety'
    # A functional simulator can miss device behavior (for example the SDK
    # RMSNorm lane broadcast). Check one complete numerical launch before
    # permitting any warmup or repeated measurement of a new device program.
    canary_path = trial / 'pcie-canary/result.json'
    if canary_path.exists():
        canary_status = json.loads(canary_path.read_text())['status']
    else:
        from .promote import canary
        try:
            canary_status = canary(trial)
        except NotImplementedError as error:
            canary_path.parent.mkdir(exist_ok=True)
            canary_status = 'blocked_device_safety' if isinstance(error, BoardSafetyExclusion) else 'unsupported'
            canary_path.write_text(json.dumps(dict(status=canary_status, reason=str(error))) + '\n')
    if canary_status != 'passed':
        return canary_status
    from tilelang.jit.adapter.tpu_profiling import TPUInstructionProfiler
    with TPUInstructionProfiler.exclusive_pcie_device(0):
        health(trial, "before")
        cmd = [
            sys.executable, "-m", "benchmark.tpu_ops.worker", "--trial",
            str(trial), "--runtime", "pcie"
        ]
        if env.get("TILELANG_BENCH_PCIE_CPUS"):
            cmd = ["taskset", "-c", env["TILELANG_BENCH_PCIE_CPUS"], *cmd]
        path = trial / "pcie/result.json"
        rc = supervised(
            cmd, trial / "pcie-launch.log", env, spec["timeout"] + 90, board_result=path)
        worker_status = json.loads(path.read_text()).get("status") if path.exists() else None
        if rc in (124, 125) or worker_status in ("timeout", "device_fault"):
            TPUInstructionProfiler.fail_closed_pcie_device(
                0,
                reason=f"operator comparison unsafe completion ({worker_status}, rc={rc}): {trial}")
            raise RuntimeError("board worker failed or timed out; device quarantined")
        health(trial, "after", result_path=path)
    path = trial / "pcie/result.json"
    return json.loads(path.read_text())["status"] if path.exists() else "launch_failed"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--case", action="append", default=[])
    p.add_argument("--op", action="append", default=[])
    p.add_argument("--dtype", action="append", default=[])
    p.add_argument("--model", action="append", choices=("rv", "tpukernel"), default=[])
    p.add_argument("--implementation", action="append", default=[])
    p.add_argument("--budget", type=int, default=3)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="timing repetitions per seed during search; finalists use 100 total")
    p.add_argument("--allow-pcie", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--recommendations", type=Path)
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument(
        '--include-existing',
        action='store_true',
        help='Also finish historical candidates in the selected problem scope')
    p.add_argument("--cmodel-workers", type=int, choices=(1, 3), default=1)
    args = p.parse_args()
    from .workloads import cases, configs, clean_config, validation_seeds, watchdog_budget
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = [
        c for c in cases() if (not args.case or c["id"] in args.case) and
        (not args.op or c["op"] in args.op) and (not args.dtype or c["dtype"] in args.dtype)
    ]
    recommendations = json.loads(args.recommendations.read_text()) if args.recommendations else {}
    spaces_path = args.recommendations.with_name("spaces.json") if args.recommendations else None
    recommended_spaces = json.loads(
        spaces_path.read_text()) if spaces_path and spaces_path.exists() else {}
    plans = []
    for c in selected:
        for model in args.model or ("rv", "tpukernel"):
            for implementation in args.implementation or ("ppl", "tilelang", "tilelang-matched",
                                                          "tilelang-optimized"):
                if implementation == "tilelang-matched" and (c["op"].startswith("elementwise-") or
                                                             c["op"] in ("matmul", "pipeline-add",
                                                                         "pipeline-matmul")):
                    # These public algorithms already match the PPL sequence;
                    # a second label would compile the identical program.
                    continue
                candidate_configs = configs(c, implementation, model)
                key = f"{c['id']}/{model}/{implementation}"
                order = recommendations.get(key, list(range(len(candidate_configs))))
                if key in recommendations and not order:
                    # Empty recommendations are accepted only after the
                    # recommender verifies a concrete frontend capability gate.
                    continue
                if key in recommended_spaces:
                    old_space = recommended_spaces[key]
                    wanted = [old_space[i] for i in order]
                    # A tightened legality filter can remove candidates after
                    # recommendation. Preserve model ordering for surviving
                    # configurations; never reinterpret an old ordinal.
                    order = [
                        i for config in wanted for i, now in enumerate(candidate_configs)
                        if clean_config(config) == clean_config(now)
                    ]
                    order += [i for i in range(len(candidate_configs)) if i not in order]
                # Validate model output against the enumerated legal space.
                if len(set(order)) != len(order) or any(
                        type(i) is not int or not 0 <= i < len(candidate_configs) for i in order):
                    raise ValueError(f"invalid recommendation for {key}")
                for idx in order[:args.budget]:
                    cfg = clean_config(candidate_configs[idx])
                    config_id = hashlib.sha256(json.dumps(cfg,
                                                          sort_keys=True).encode()).hexdigest()[:10]
                    trial = output / c["id"] / model / implementation / config_id
                    spec = dict(
                        case=c,
                        config=cfg,
                        model=model,
                        implementation=implementation,
                        sdk=str(args.sdk.resolve()),
                        seeds=validation_seeds(c),
                        repeats=args.repeats,
                        timeout=watchdog_budget(c, args.timeout),
                        estimated_local_bytes_per_lane=candidate_configs[idx]
                        ["estimated_local_bytes_per_lane"])
                    plans.append(dict(path=str(trial), spec=spec))
    if args.include_existing:
        planned = {item['path'] for item in plans}
        selected_ids = {case['id'] for case in selected}
        for existing in output.glob('*/*/*/*/trial.json'):
            spec = json.loads(existing.read_text())
            path = str(existing.parent)
            if (path not in planned and spec['case']['id'] in selected_ids and
                (not args.model or spec['model'] in args.model) and
                (not args.implementation or spec['implementation'] in args.implementation)):
                plans.append(dict(path=path, spec=spec))
    (output / "plan.json").write_text(json.dumps(plans, indent=2) + "\n")
    print(f"Selected {len(selected)} problems; {len(plans)} candidates", flush=True)
    if args.plan_only:
        return
    trials = []
    for plan in plans:
        trial = Path(plan["path"])
        trial.mkdir(parents=True, exist_ok=True)
        path = trial / "trial.json"
        payload = json.dumps(plan["spec"], sort_keys=True, indent=2) + "\n"
        if path.exists() and path.read_text() != payload:
            previous = json.loads(path.read_text())
            metadata = ('estimated_local_bytes_per_lane', 'tuning_origin')
            prior_essential = {key: value for key, value in previous.items() if key not in metadata}
            proposed_essential = {key: value for key, value in plan['spec'].items() if key not in metadata}
            if prior_essential != proposed_essential:
                raise RuntimeError(f"trial specification changed: {trial}; choose a fresh output")
            # Retain the exact validated bytes, including the original estimate
            # and candidate origin. Metadata refinement never rewrites evidence.
            payload = path.read_text()
            plan['spec'] = previous
        path.write_text(payload)
        if args.retry_failed:
            for mode in ("cmodel", "pcie"):
                result = trial / mode / "result.json"
                if result.exists() and json.loads(
                        result.read_text())["status"] in ("error", "launch_failed"):
                    (trial / mode).rename(trial / f"{mode}-previous-{time.time_ns()}")
        trials.append(trial)
    (output / 'plan.json').write_text(json.dumps(plans, indent=2) + '\n')
    if args.cmodel_workers == 1:
        for i, trial in enumerate(trials):
            status = execute_trial(trial, args.allow_pcie)
            print(f"[{i+1}/{len(plans)}] {trial.relative_to(output)}: {status}", flush=True)
    else:
        # This host has 16 physical cores with SMT siblings at +16. CModel
        # processes get disjoint physical cores; the serialized board runner
        # and all finalist timing use the remaining four physical cores.
        if not set(range(32)).issubset(os.sched_getaffinity(0)):
            raise RuntimeError(
                "three-worker CPU partition requires CPUs 0..31; use --cmodel-workers 1")
        groups = ["0-3,16-19", "4-7,20-23", "8-11,24-27"]
        os.environ["TILELANG_BENCH_PCIE_CPUS"] = "12-15,28-31"
        (output / "execution.json").write_text(
            json.dumps(
                dict(
                    cmodel_cpu_partitions=groups,
                    pcie_cpus=os.environ["TILELANG_BENCH_PCIE_CPUS"],
                    cmodel_workers=3,
                    board_workers=1),
                indent=2))
        iterator = iter(trials)
        completed = 0
        with ThreadPoolExecutor(max_workers=3) as pool:
            pending = {}
            for slot in range(3):
                trial = next(iterator, None)
                if trial is not None:
                    pending[pool.submit(execute_trial, trial, False, groups[slot])] = (trial, slot)
            while pending:
                ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in ready:
                    trial, slot = pending.pop(future)
                    status = future.result()
                    following = next(iterator, None)
                    if following is not None:
                        pending[pool.submit(execute_trial, following, False,
                                            groups[slot])] = (following, slot)
                    if status == "passed" and args.allow_pcie:
                        status = execute_trial(trial, True)
                    completed += 1
                    print(
                        f"[{completed}/{len(plans)}] {trial.relative_to(output)}: {status}",
                        flush=True)
                    if completed % 25 == 0:
                        from .report import report
                        report(output)
    from .report import report
    report(output)


if __name__ == "__main__":
    main()
