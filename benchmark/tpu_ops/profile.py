"""Record a separate device trace for each measured winner."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from .run import health, supervised, ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--decoder-python", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument(
        "--representative",
        action="store_true",
        help="profile the smallest measured problem per op/backend/implementation/dtype")
    args = p.parse_args()
    output = args.output.resolve()
    if (output / "execution.json").exists():
        os.environ["TILELANG_BENCH_PCIE_CPUS"] = json.loads(
            (output / "execution.json").read_text())["pcie_cpus"]
    from .report import report
    from tilelang.jit.adapter.tpu_profiling import TPUInstructionProfiler
    report(output)
    winners = json.loads((output / "best.json").read_text())
    if args.representative:
        representatives = {}
        for row in winners:
            key = (row["op"], row["model"], row["implementation"], row["dtype"])
            if key not in representatives or math.prod(row["shape"]) < math.prod(
                    representatives[key]["shape"]):
                representatives[key] = row
        winners = list(representatives.values())
    (output / "profile-selection.json").write_text(
        json.dumps(
            dict(representative=args.representative, trials=[row["path"] for row in winners]),
            indent=2))
    if args.limit:
        winners = winners[:args.limit]
    for row in winners:
        trial = Path(row["path"])
        spec = json.loads((trial / "trial.json").read_text())
        target = trial / "pcie-profile"
        env = dict(
            os.environ,
            PYTHONPATH=str(ROOT),
            PPL_PROJECT_ROOT=spec["sdk"],
            OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4")
        if not (target / "result.json").exists():
            with TPUInstructionProfiler.exclusive_pcie_device(0):
                health(trial, "profile-before")
                cmd = [
                    sys.executable, "-m", "benchmark.tpu_ops.worker", "--trial",
                    str(trial), "--runtime", "pcie", "--profile"
                ]
                if env.get("TILELANG_BENCH_PCIE_CPUS"):
                    cmd = ["taskset", "-c", env["TILELANG_BENCH_PCIE_CPUS"], *cmd]
                rc = supervised(
                    cmd,
                    trial / "profile-launch.log",
                    env,
                    spec["timeout"] + 90,
                    board_result=target / "result.json")
                result_path = target / "result.json"
                status = json.loads(
                    result_path.read_text()).get("status") if result_path.exists() else None
                if rc in (124, 125) or status in ("timeout", "device_fault"):
                    TPUInstructionProfiler.fail_closed_pcie_device(
                        0, reason=f"profile timeout {trial}")
                    raise RuntimeError("profile timed out; device quarantined")
                health(trial, "profile-after", result_path=target / "result.json")
        if not (target / "result.json").exists():
            continue
        result = json.loads((target / "result.json").read_text())
        if result["status"] != "passed":
            continue
        decoder = ROOT / "tilelang/jit/_tpu_pcie_profile_decoder.py"
        completed = subprocess.run([
            str(args.decoder_python),
            str(decoder), "--profile-dir",
            str(target), "--arch", "tpub_7_1_e"
        ],
                                   text=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   timeout=120)
        (target / "decode.log").write_text(completed.stdout)
        paths = list(target.glob("decoded_*/tilelang_pcie_profile.json"))
        if completed.returncode or not paths:
            print(f"decoder failed for {trial}", flush=True)
            continue
        events = [e for path in paths for e in json.loads(path.read_text())["events"]]
        engine_counts = Counter((event["engine"], event.get("core_id")) for event in events)
        if engine_counts and max(engine_counts.values()) >= 1000000:
            (target / "device_timing.json").write_text(
                json.dumps(
                    dict(
                        status="record_budget_reached",
                        reason="An engine reached the native recorder's one-million-record budget; do not report a potentially truncated span."
                    ),
                    indent=2))
            print(
                f"PMU record budget reached for {trial}; host timing remains available", flush=True)
            continue
        spans = [e for e in events if e["end"] >= e["begin"]]
        if spans:
            summary = dict(
                device_command_span_us=(max(e["end"] for e in spans) - min(e["begin"]
                                                                           for e in spans)) / 1000,
                event_count=len(events),
                boundary="single separately profiled launch, first-to-last recorded device command; excludes scalar work before/after commands",
                decoded_reports=list(map(str, paths)))
            (target / "device_timing.json").write_text(json.dumps(summary, indent=2))
            print(
                f"{row['case']} {row['model']} {row['implementation']}: {summary['device_command_span_us']:.3f} us",
                flush=True)


if __name__ == "__main__":
    main()
