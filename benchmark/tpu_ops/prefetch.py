"""Prevalidate expensive future trials while the serial board search advances.

Queued verdicts are deliberately non-passing. The board coordinator can skip
them and resume after this process finishes; it can never promote them early.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import tempfile
from .run import supervised, ROOT


def validate(trial, omp_threads=None, cmodel_cpus="0-11,16-27"):
    from tilelang.jit.tpu_cmodel import prepare_cmodel_command
    from tilelang.temp_storage import user_temp_root
    spec = json.loads((trial / "trial.json").read_text())
    target = trial / "cmodel"
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT),
        PPL_PROJECT_ROOT=spec["sdk"],
        OMP_NUM_THREADS=str(omp_threads or 1),
        OPENBLAS_NUM_THREADS="1",
        TILELANG_BENCH_TORCH_THREADS="4" if spec["case"]["op"] == "flashattn" else "1")
    if omp_threads is not None:
        env["TILELANG_BENCH_CMODEL_OMP_THREADS"] = str(omp_threads)
    with tempfile.TemporaryDirectory(
            prefix="comparison-prefetch-", dir=user_temp_root()) as private:
        command, isolated = prepare_cmodel_command([
            sys.executable, "-m", "benchmark.tpu_ops.worker", "--trial",
            str(trial), "--runtime", "cmodel"
        ],
                                                   cwd=target,
                                                   environment=env,
                                                   private_tmp_dir=Path(private))
        command = ["taskset", "-c", cmodel_cpus, *command]
        rc = supervised(command, trial / "cmodel-prefetch.log", isolated, spec["timeout"] + 60)
        result = target / "result.json"
        status = json.loads(result.read_text())["status"]
        if status == "queued":
            result.write_text(
                json.dumps(dict(status="timeout" if rc == 124 else "launch_failed", returncode=rc)))
        (target / "prefetch_isolation.json").write_text(
            json.dumps(
                dict(command=command, returncode=rc, timeout=spec["timeout"] + 60), indent=2))
    return json.loads((target / "result.json").read_text())["status"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument(
        "--omp-threads",
        type=int,
        choices=(1, 2, 4),
        help="Override the worker's per-family cmodel thread policy")
    args = p.parse_args()
    output = args.output.resolve()
    selected = []
    for path in sorted(output.glob("*/*/*/*/trial.json")):
        spec = json.loads(path.read_text())
        case = spec["case"]
        if case["op"] not in ("matmul", "llama-mlp") or case["shape"][0] != 2048:
            continue
        trial = path.parent
        target = trial / "cmodel"
        if target.exists():
            continue
        target.mkdir()
        (target / "result.json").write_text(
            json.dumps(dict(status="queued", producer="cmodel-only prefetch")))
        selected.append(trial)
    (output / "prefetch.json").write_text(
        json.dumps(
            dict(
                cmodel_only=True,
                workers=args.workers,
                omp_threads=args.omp_threads,
                cpus="0-11,16-27",
                trials=list(map(str, selected))),
            indent=2))
    print(f"Queued {len(selected)} full-shape correctness trials", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(validate, trial, args.omp_threads): trial for trial in selected}
        for future in as_completed(futures):
            print(f"Prefetch {futures[future].relative_to(output)}: {future.result()}", flush=True)


if __name__ == "__main__":
    main()
