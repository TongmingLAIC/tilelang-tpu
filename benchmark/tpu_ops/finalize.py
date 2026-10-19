"""Remeasure the two fastest search candidates with 100 samples each."""
import argparse
import csv
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
from .run import supervised, health, ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--top", type=int, default=2)
    p.add_argument(
        "--plan-only",
        action="store_true",
        help="Write finalists and search-stage fixed-tile comparisons without accessing the board")
    args = p.parse_args()
    output = args.output.resolve()
    if (output / "execution.json").exists():
        os.environ["TILELANG_BENCH_PCIE_CPUS"] = json.loads(
            (output / "execution.json").read_text())["pcie_cpus"]
    from .report import report
    from tilelang.jit.adapter.tpu_profiling import TPUInstructionProfiler
    report(output)
    groups = defaultdict(list)
    for row in json.loads((output / "all_trials.json").read_text()):
        if row["cmodel_status"] == "passed" and row["pcie_status"] == "passed":
            groups[row["case"], row["model"], row["implementation"]].append(row)
    chosen = {
        row["path"]: row for rows in groups.values()
        for row in sorted(rows, key=lambda r: r["search_p50_us"])[:args.top]
    }
    controls = []
    for case, model in sorted({(key[0], key[1]) for key in groups}):
        orig = groups.get((case, model, "tilelang"), [])
        matched = groups.get((case, model, "tilelang-matched"), [])
        ppl = groups.get((case, model, "ppl"), [])
        identity = bool(
            orig and (orig[0]["op"].startswith("elementwise-") or
                      orig[0]["op"] in ("matmul", "pipeline-add", "pipeline-matmul")))
        if identity:
            matched = orig
        common = []
        for o in orig:
            for m in matched:
                if o["effective_config"] == m["effective_config"]:
                    references = [r for r in ppl if r["effective_config"] == o["effective_config"]]
                    common.append(
                        (references[0]["search_p50_us"] if references else o["search_p50_us"], o, m,
                         references))
        if common:
            # Prefer a three-way identical configuration when available.
            triplets = [c for c in common if c[3]]
            _, o, m, references = min(triplets or common, key=lambda c: c[0])
            for row in [o, m, *references]:
                chosen[row["path"]] = row
            controls.append(
                dict(
                    case=case,
                    model=model,
                    config=o["effective_config"],
                    original=o["path"],
                    matched=m["path"],
                    ppl=references[0]["path"] if references else None,
                    identical_algorithm=identity))
    (output / "controls.json").write_text(json.dumps(controls, indent=2))
    (output / "finalist-plan.json").write_text(json.dumps(list(chosen.values()), indent=2) + '\n')
    by_path = {row["path"]: row for rows in groups.values() for row in rows}
    search_controls = []
    for item in controls:
        original, matched = by_path[item["original"]], by_path[item["matched"]]
        ppl = by_path[item["ppl"]] if item.get("ppl") else None
        search_controls.append(
            dict(
                case=item["case"],
                model=item["model"],
                config=item["config"],
                timing_stage="search",
                identity=item["identical_algorithm"],
                original_us=original["search_p50_us"],
                matched_us=matched["search_p50_us"],
                ppl_us=ppl["search_p50_us"] if ppl else None,
                original_samples=original["samples"],
                matched_samples=matched["samples"],
                original_over_matched=original["search_p50_us"] / matched["search_p50_us"]))
    (output /
     "ablation_fixed_tiles_search.json").write_text(json.dumps(search_controls, indent=2) + '\n')
    if search_controls:
        with (output / "ablation_fixed_tiles_search.csv").open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=search_controls[0].keys())
            writer.writeheader()
            writer.writerows(search_controls)
    if args.plan_only:
        print(
            f"Prepared {len(chosen)} finalists and {len(controls)} search-stage controls; no board access",
            flush=True)
        return
    for row in chosen.values():
        trial = Path(row["path"])
        spec = json.loads((trial / "trial.json").read_text())
        result = trial / "pcie-final/result.json"
        if result.exists():
            continue
        env = dict(
            os.environ,
            PYTHONPATH=str(ROOT),
            PPL_PROJECT_ROOT=spec["sdk"],
            OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4")
        with TPUInstructionProfiler.exclusive_pcie_device(0):
            health(trial, "final-before")
            cmd = [
                sys.executable, "-m", "benchmark.tpu_ops.worker", "--trial",
                str(trial), "--runtime", "pcie", "--final"
            ]
            if env.get("TILELANG_BENCH_PCIE_CPUS"):
                cmd = ["taskset", "-c", env["TILELANG_BENCH_PCIE_CPUS"], *cmd]
            rc = supervised(
                cmd, trial / "final-launch.log", env, spec["timeout"] + 90, board_result=result)
            status = json.loads(result.read_text()).get("status") if result.exists() else None
            if rc in (124, 125) or status in ("timeout", "device_fault"):
                TPUInstructionProfiler.fail_closed_pcie_device(
                    0, reason=f"final measurement timeout: {trial}")
                raise RuntimeError("finalist timed out; device quarantined")
            health(trial, "final-after", result_path=result)
        print(f"Final: {trial.relative_to(output)}", flush=True)
    report(output)


if __name__ == "__main__":
    main()
