# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Run the TPU-Kernel/RV FP8 capability matrix in supervised fresh processes.

CModel remains the safe default. A PCIe run promotes matching, clean BM1690
and SG2260E CModel evidence: it compiles a read-only Git snapshot, pins the
captured native compiler libraries, holds one exclusive device session,
checks board health around every launch, and stops on the first failure.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Optional

from tilelang.jit import TPUInstructionProfiler, TPUProfilingConfig

if __package__:
    from .tpu_matrix_common import (
        git_source_identity,
        matrix_target_scope,
        unique_prefixed_json_payload,
        validate_promotion_stages,
    )
    from .tpu_demo_ops_matrix import (
        _COMMON_TOOLCHAIN_FIELDS,
        assert_source_identity_unchanged,
        assert_toolchain_identity_unchanged,
        board_health,
        materialize_execution_snapshot,
        pin_native_worker_libraries,
        pin_worker_cache_environment,
        pin_worker_environment,
        remove_execution_scratch,
        toolchain_identity,
        worker_environment,
        write_json,
    )
    from .tpu_profile_matrix_common import (
        profile_report_summary as _report_summary,
        validate_profile_report as _validate_profile_report,
    )
else:
    from tpu_matrix_common import (
        git_source_identity,
        matrix_target_scope,
        unique_prefixed_json_payload,
        validate_promotion_stages,
    )
    from tpu_demo_ops_matrix import (
        _COMMON_TOOLCHAIN_FIELDS,
        assert_source_identity_unchanged,
        assert_toolchain_identity_unchanged,
        board_health,
        materialize_execution_snapshot,
        pin_native_worker_libraries,
        pin_worker_cache_environment,
        pin_worker_environment,
        remove_execution_scratch,
        toolchain_identity,
        worker_environment,
        write_json,
    )
    from tpu_profile_matrix_common import (
        profile_report_summary as _report_summary,
        validate_profile_report as _validate_profile_report,
    )

_CHIPS = ("sg2260e", "bm1690")
_MATRIX_KIND = "tpu_fp8_ops"
_SCHEMA_VERSION = 1
_WORKER_RESULT_PREFIX = "TPU_FP8_NUMERIC_RESULT="
_DTYPES = ("e4m3", "e5m2")
_PROGRAMMING_MODELS = ("tpukernel", "rv")
_CASES = (
    "copy",
    "copy-global-to-global",
    "fill-zero",
    "fill-nonzero",
    "cast-to-fp8",
    "cast-from-fp8",
    "cast-fp16-to-fp8",
    "cast-fp8-to-fp16",
    "cast-bf16-to-fp8",
    "cast-fp8-to-bf16",
    "add",
    "sub",
    "mul",
    "max",
    "add-broadcast",
    "sub-broadcast",
    "mul-broadcast",
    "max-broadcast",
    "add-scalar",
    "mul-scalar",
    "embedding",
    "reduce-sum",
    "reduce-max",
    "gemm-nn-overwrite",
    "gemm-nn-accumulate",
    "gemm-nt-overwrite",
    "gemm-nt-accumulate",
)
_TPUKERNEL_UNSUPPORTED_CASES = frozenset({"reduce-sum"})


def _selected_cases(args: argparse.Namespace) -> tuple[str, ...]:
    """Return the explicit or backend-specific default FP8 case set."""
    cases = tuple(args.cases) if args.cases else tuple(
        case for case in _CASES
        if args.programming_model != "tpukernel" or case not in _TPUKERNEL_UNSUPPORTED_CASES)
    unsupported = _TPUKERNEL_UNSUPPORTED_CASES.intersection(cases)
    if args.programming_model == "tpukernel" and unsupported:
        raise RuntimeError("TPU-Kernel has no validated FP8 implementation for: " +
                           ", ".join(sorted(unsupported)))
    return cases


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-mode", choices=("cmodel", "pcie"), default="cmodel")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--chip", choices=_CHIPS, action="append", dest="chips")
    parser.add_argument(
        "--programming-model",
        choices=_PROGRAMMING_MODELS,
        default="tpukernel",
    )
    parser.add_argument("--dtype", choices=_DTYPES, action="append", dest="dtypes")
    parser.add_argument("--case", choices=_CASES, action="append", dest="cases")
    parser.add_argument("--device-id", type=int)
    parser.add_argument("--allow-pcie", action="store_true")
    parser.add_argument("--allow-pcie-profile", action="store_true")
    parser.add_argument("--bm-cmodel-summary", type=Path)
    parser.add_argument("--sg-cmodel-summary", type=Path)
    parser.add_argument("--all-pcie-cases", action="store_true")
    parser.add_argument("--require-decoded-timing", action="store_true")
    parser.add_argument("--pcie-decoder-python", type=Path)
    parser.add_argument("--pcie-decoder-pythonpath", type=Path, action="append", default=[])
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if (isinstance(args.timeout, bool) or not isinstance(args.timeout, (int, float)) or
            not math.isfinite(float(args.timeout)) or args.timeout <= 0):
        raise ValueError("--timeout must be a positive finite number")
    decoder_options = bool(args.pcie_decoder_python or args.pcie_decoder_pythonpath)
    if args.runtime_mode == "pcie":
        if not args.allow_pcie or not args.allow_pcie_profile:
            raise RuntimeError("PCIe requires both --allow-pcie and --allow-pcie-profile")
        if (isinstance(args.device_id, bool) or not isinstance(args.device_id, int) or
                args.device_id != 0):
            raise RuntimeError("this single-card validation host accepts only --device-id 0")
        if args.chips != ["sg2260e"]:
            raise RuntimeError("PCIe requires exactly one explicit --chip sg2260e")
        if not args.cases and not args.all_pcie_cases:
            raise RuntimeError("PCIe requires an explicit --case or --all-pcie-cases")
        if args.bm_cmodel_summary is None or args.sg_cmodel_summary is None:
            raise RuntimeError(
                "PCIe requires --bm-cmodel-summary and --sg-cmodel-summary promotion evidence")
    elif (args.device_id is not None or args.allow_pcie or args.allow_pcie_profile or
          args.require_decoded_timing or decoder_options or args.bm_cmodel_summary or
          args.sg_cmodel_summary or args.all_pcie_cases):
        raise RuntimeError(
            "PCIe acknowledgements, promotion, and decoder options are invalid for CModel")
    elif args.chips is None or len(args.chips) != 1:
        raise RuntimeError("CModel promotion requires exactly one explicit --chip")
    if args.programming_model == "rv" and args.chips != ["sg2260e"]:
        raise RuntimeError("RV Tensor FP8 validation requires exactly --chip sg2260e")
    _selected_cases(args)


def _worker_environment(repo_root: Path, scratch: Path, runtime_mode: str,
                        device_id: Optional[int]) -> dict[str, str]:
    environment = worker_environment(repo_root, runtime_mode, device_id)
    environment = pin_worker_cache_environment(environment, scratch)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _profiling_config(args: argparse.Namespace, output_dir: Path, chip: str,
                      label: str) -> TPUProfilingConfig:
    return TPUProfilingConfig(
        chip=chip,
        programming_model=args.programming_model,
        runtime_mode=args.runtime_mode,
        output_dir=output_dir,
        label=label,
        timeout_s=args.timeout,
        # Preserve PPL's optional CModel PerfAI decode path as well as PCIe
        # recorder decoding. Raw-only CModel evidence remains explicit when
        # no compatible PerfAI installation is configured.
        postprocess=True,
        pcie_decoder_python=args.pcie_decoder_python,
        pcie_decoder_pythonpath=tuple(args.pcie_decoder_pythonpath),
    )


def _write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    """Write a matrix summary through the shared atomic JSON primitive."""
    write_json(path, summary)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _worker_payload(stdout_path: Path) -> dict[str, Any]:
    return unique_prefixed_json_payload(stdout_path, _WORKER_RESULT_PREFIX, "FP8 worker")


def _validate_worker_payload(
    payload: Mapping[str, Any],
    *,
    chip: str,
    programming_model: str,
    runtime_mode: str,
    dtype: str,
    case: str,
) -> None:
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise RuntimeError("FP8 worker result has the wrong schema_version")
    expected = {
        "status": "passed",
        "chip": chip,
        "programming_model": programming_model,
        "runtime_mode": runtime_mode,
        "dtype": dtype,
        "case": case,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise RuntimeError(f"FP8 worker result {field} does not identify the scheduled case")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("passed") is not True:
        raise RuntimeError("FP8 worker result lacks metrics.passed=true")


def _validate_pcie_promotion(
    repo_root: Path,
    args: argparse.Namespace,
    dtypes: tuple[str, ...],
    cases: tuple[str, ...],
    current_toolchain: Mapping[str, Any],
    pcie_started_at: str,
) -> dict[str, Any]:
    """Require matching BM1690 and SG2260E CModel FP8 evidence.

    BM1690 contributes TPU-Kernel evidence for cases shared by both programming
    models.  RV-only cases still require passing SG2260E/RV evidence, but cannot
    require a nonexistent BM1690/RV result or an unsupported TPU-Kernel result.
    """
    assert args.bm_cmodel_summary is not None and args.sg_cmodel_summary is not None
    if (current_toolchain.get("runtime_mode") != "pcie" or
            not isinstance(current_toolchain.get("pcie"), dict)):
        raise RuntimeError("PCIe promotion requires a complete PCIe toolchain identity")
    current = git_source_identity(repo_root)
    if current.get("implementation_worktree_dirty") is not False:
        raise RuntimeError(
            "PCIe promotion requires the current implementation worktree to be clean")
    commit = current.get("git_commit")
    source_digest = current.get("source_state_sha256")
    if not isinstance(commit, str) or not commit:
        raise RuntimeError("PCIe promotion requires a verifiable Git commit")
    if not isinstance(source_digest, str) or not source_digest:
        raise RuntimeError("PCIe promotion requires a verifiable source-state digest")

    bm, sg = validate_promotion_stages(
        args.bm_cmodel_summary,
        args.sg_cmodel_summary,
        matrix_kind=_MATRIX_KIND,
        schema_version=_SCHEMA_VERSION,
        bm_allowed_scope=(("bm1690", "tpukernel"),),
        sg_allowed_scope=(("sg2260e", "tpukernel"), ("sg2260e", "rv")),
        pcie_started_at=pcie_started_at,
    )
    for label, payload, expected_model in (
        ("BM1690", bm, "tpukernel"),
        ("SG2260E", sg, args.programming_model),
    ):
        if payload.get("git_commit") != commit:
            raise RuntimeError(f"{label} FP8 summary commit does not match current source")
        if payload.get("source_state_sha256") != source_digest:
            raise RuntimeError(f"{label} FP8 summary source digest does not match current source")
        if payload.get("programming_model") != expected_model:
            raise RuntimeError(f"{label} FP8 summary is not {expected_model} evidence")
        observed_toolchain = payload.get("toolchain_identity")
        if not isinstance(observed_toolchain, dict):
            raise RuntimeError(f"{label} FP8 summary has no toolchain identity")
        if observed_toolchain.get("runtime_mode") != "cmodel":
            raise RuntimeError(f"{label} FP8 summary has the wrong toolchain runtime mode")
        for field in _COMMON_TOOLCHAIN_FIELDS:
            if observed_toolchain.get(field) != current_toolchain.get(field):
                raise RuntimeError(
                    f"{label} FP8 summary content identity for {field} does not match")
    if bm.get("git_commit") != sg.get("git_commit"):
        raise RuntimeError("BM1690 and SG2260E FP8 summaries use different commits")

    bm_results = bm.get("cases")
    sg_results = sg.get("cases")
    if not isinstance(bm_results, dict) or not isinstance(sg_results, dict):
        raise RuntimeError("FP8 promotion summary has no case result map")
    missing = []
    bm_cases = tuple(case for case in cases if case not in _TPUKERNEL_UNSUPPORTED_CASES)
    for dtype in dtypes:
        for case in cases:
            required_results = [(f"sg2260e/{args.programming_model}/{dtype}/{case}",
                                 sg_results, args.programming_model)]
            if case in bm_cases:
                required_results.insert(
                    0, (f"bm1690/tpukernel/{dtype}/{case}", bm_results, "tpukernel"))
            for key, results, expected_model in required_results:
                result = results.get(key)
                if not isinstance(result, dict) or result.get("status") != "passed":
                    missing.append(key)
                    continue
                raw_count = result.get("raw_instruction_count")
                if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count <= 0:
                    raise RuntimeError(f"FP8 promotion result has no raw instructions: {key}")
                numeric = result.get("numeric")
                if not isinstance(numeric, dict):
                    raise RuntimeError(f"FP8 promotion result has no numeric payload: {key}")
                expected_chip, key_model, expected_dtype, expected_case = key.split("/", 3)
                if key_model != expected_model:
                    raise RuntimeError(f"FP8 promotion result has the wrong model: {key}")
                _validate_worker_payload(
                    numeric,
                    chip=expected_chip,
                    programming_model=expected_model,
                    runtime_mode="cmodel",
                    dtype=expected_dtype,
                    case=expected_case,
                )
    if missing:
        raise RuntimeError("PCIe FP8 promotion evidence is missing passing cases: " +
                           ", ".join(sorted(set(missing))))
    return {
        "git_commit": commit,
        "source_state_sha256": source_digest,
        "bm1690_summary": bm["_resolved_path"],
        "bm1690_summary_sha256": bm["_sha256"],
        "sg2260e_summary": sg["_resolved_path"],
        "sg2260e_summary_sha256": sg["_sha256"],
        "validated_case_count": len(dtypes) * len(cases),
        "bm1690_validated_case_count": len(dtypes) * len(bm_cases),
        "sg2260e_validated_case_count": len(dtypes) * len(cases),
    }


def _run_matrix(args: argparse.Namespace, repo_root: Path, output_dir: Path,
                environment: Mapping[str, str]) -> int:
    chips = tuple(args.chips or _CHIPS)
    dtypes = tuple(args.dtypes or _DTYPES)
    cases = _selected_cases(args)
    summary: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "matrix_kind": _MATRIX_KIND,
        "status": "running",
        "runtime_mode": args.runtime_mode,
        "programming_model": args.programming_model,
        "acceptance":
            ("numeric-raw-and-decoded-timing" if args.require_decoded_timing else "numeric-and-raw"
            ),
        "decoded_timing_required": args.require_decoded_timing,
        "complete": False,
        "started_at": _utc_now(),
        "finished_at": None,
        "scheduled_case_count": len(chips) * len(dtypes) * len(cases),
        "completed_case_count": 0,
        "passed_case_count": 0,
        "failed_case_count": 0,
        "cancelled_case_count": 0,
        "target_scope": matrix_target_scope((chip, args.programming_model) for chip in chips),
        "scheduled": [{
            "chip": chip,
            "programming_model": args.programming_model,
            "dtype": dtype,
            "case": case,
        } for chip in chips for dtype in dtypes for case in cases],
        "cases": {},
    }
    summary.update(git_source_identity(repo_root))
    summary_path = output_dir / "summary.json"
    _write_summary(summary_path, summary)
    execution_root = repo_root
    worker_environment_values = dict(environment)

    try:
        summary["toolchain_identity"] = toolchain_identity(worker_environment_values,
                                                           args.runtime_mode)
        worker_environment_values = pin_native_worker_libraries(worker_environment_values,
                                                                summary["toolchain_identity"])
        if args.runtime_mode == "pcie":
            assert args.device_id == 0
            summary["promotion_evidence"] = _validate_pcie_promotion(
                repo_root,
                args,
                dtypes,
                cases,
                summary["toolchain_identity"],
                summary["started_at"],
            )
            snapshot_root = Path(worker_environment_values["TMPDIR"]) / "execution-source"
            summary["execution_snapshot"] = materialize_execution_snapshot(
                repo_root, snapshot_root, summary["promotion_evidence"]["git_commit"])
            execution_root = snapshot_root
            worker_environment_values = pin_worker_environment(
                worker_environment_values,
                repo_root=repo_root,
                snapshot_root=snapshot_root,
                toolchain_identity=summary["toolchain_identity"],
            )
        if args.require_decoded_timing:
            profiler = TPUInstructionProfiler(
                _profiling_config(args, output_dir, chips[0], "pcie-decoder-preflight"))
            summary["decoder_preflight"] = {
                "status":
                    "passed",
                "identity":
                    dict(profiler.preflight_pcie_decoder(environment=worker_environment_values)),
            }
        if args.runtime_mode == "pcie":
            tpu_smi = Path(summary["toolchain_identity"]["pcie"]["tpu_smi"]["path"])
            summary["board_preflight"] = board_health(0, tpu_smi)
        _write_summary(summary_path, summary)
    except KeyboardInterrupt:
        summary.update({
            "status": "cancelled",
            "failed_phase": "preflight",
            "finished_at": _utc_now()
        })
        _write_summary(summary_path, summary)
        raise
    except Exception as exc:
        summary.update({
            "status": "failed",
            "failed_phase": "preflight",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "finished_at": _utc_now()
        })
        _write_summary(summary_path, summary)
        print(f"STOP preflight: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    worker = execution_root / "testing/python/jit/tpu_fp8_ops_worker.py"
    for chip in chips:
        for dtype in dtypes:
            for case in cases:
                key = f"{chip}/{args.programming_model}/{dtype}/{case}"
                print(f"RUN {key}", flush=True)
                command = [sys.executable, str(worker), "--dtype", dtype, "--case", case]
                launch_attempted = False
                postflight_attempted = False
                result: Optional[dict[str, Any]] = None
                try:
                    if args.runtime_mode == "pcie":
                        assert_source_identity_unchanged(repo_root, summary)
                        assert_toolchain_identity_unchanged(worker_environment_values,
                                                            summary["toolchain_identity"])
                    profiler = TPUInstructionProfiler(
                        _profiling_config(args, output_dir, chip,
                                          f"{chip}-{args.programming_model}-{dtype}-{case}"))
                    launch_attempted = True
                    report = (
                        profiler.run_pcie(command, environment=worker_environment_values)
                        if args.runtime_mode == "pcie" else profiler.run_cmodel(
                            command, environment=worker_environment_values))
                    _validate_profile_report(
                        report, require_decoded_timing=args.require_decoded_timing)
                    numeric = _worker_payload(Path(report.stdout_path))
                    _validate_worker_payload(
                        numeric,
                        chip=chip,
                        programming_model=args.programming_model,
                        runtime_mode=args.runtime_mode,
                        dtype=dtype,
                        case=case,
                    )
                    result = _report_summary(
                        report, require_decoded_timing=args.require_decoded_timing)
                    result["numeric"] = numeric
                    if args.runtime_mode == "pcie":
                        tpu_smi = Path(summary["toolchain_identity"]["pcie"]["tpu_smi"]["path"])
                        postflight_attempted = True
                        result["board_postflight"] = board_health(
                            0, tpu_smi, quarantine_on_failure=True)
                    summary["cases"][key] = result
                    print(
                        f"PASS {key} raw={len(report.raw_instructions)} "
                        f"timed={len(report.instruction_timings)}",
                        flush=True)
                except KeyboardInterrupt as error:
                    summary["status"] = "cancelled"
                    summary["stopped_after"] = key
                    if postflight_attempted and result is not None:
                        result.update({
                            "status": "cancelled",
                            "execution_status": "passed",
                            "failed_phase": "board-postflight-settle",
                            "error_type": "KeyboardInterrupt",
                            "error": "board postflight was interrupted",
                        })
                        postflight_evidence = getattr(error, "evidence", None)
                        if isinstance(postflight_evidence, Mapping):
                            result["board_postflight_failure"] = dict(postflight_evidence)
                        summary["cases"][key] = result
                        summary["completed_case_count"] += 1
                        summary["cancelled_case_count"] += 1
                    if (args.runtime_mode == "pcie" and launch_attempted and
                            not postflight_attempted):
                        try:
                            tpu_smi = Path(summary["toolchain_identity"]["pcie"]["tpu_smi"]["path"])
                            summary["board_after_cancel"] = board_health(
                                0, tpu_smi, quarantine_on_failure=True)
                        except Exception as health_error:
                            summary["board_after_cancel_error"] = (
                                f"{type(health_error).__name__}: {health_error}")
                    summary["finished_at"] = _utc_now()
                    _write_summary(summary_path, summary)
                    raise
                except Exception as exc:
                    if postflight_attempted and result is not None:
                        result.update({
                            "status": "failed",
                            "execution_status": "passed",
                            "failed_phase": "board-postflight-settle",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        })
                        postflight_evidence = getattr(exc, "evidence", None)
                        if postflight_evidence is not None:
                            result["board_postflight_failure"] = postflight_evidence
                    else:
                        result = {
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc)
                        }
                    if (args.runtime_mode == "pcie" and launch_attempted and
                            not postflight_attempted):
                        try:
                            tpu_smi = Path(summary["toolchain_identity"]["pcie"]["tpu_smi"]["path"])
                            result["board_after_failure"] = board_health(
                                0, tpu_smi, quarantine_on_failure=True)
                        except Exception as health_error:
                            result["board_after_failure_error"] = (
                                f"{type(health_error).__name__}: {health_error}")
                    summary["cases"][key] = result
                    summary["completed_case_count"] += 1
                    summary["failed_case_count"] += 1
                    summary["status"] = "failed"
                    summary["stopped_after"] = key
                    summary["finished_at"] = _utc_now()
                    _write_summary(summary_path, summary)
                    print(f"STOP {key}: {type(exc).__name__}: {exc}", file=sys.stderr)
                    return 1
                summary["completed_case_count"] += 1
                summary["passed_case_count"] += 1
                _write_summary(summary_path, summary)

    try:
        ending_source = git_source_identity(repo_root)
        ending_toolchain = toolchain_identity(worker_environment_values, args.runtime_mode)
    except Exception as exc:
        summary.update({
            "status": "failed",
            "complete": False,
            "failed_phase": "final-identity-check",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "finished_at": _utc_now()
        })
        _write_summary(summary_path, summary)
        return 1
    source_fields = ("git_commit", "implementation_worktree_dirty", "source_state_sha256")
    source_changed = (not isinstance(summary.get("source_state_sha256"), str) or any(
        ending_source.get(field) != summary.get(field) for field in source_fields))
    toolchain_changed = ending_toolchain != summary.get("toolchain_identity")
    if source_changed or toolchain_changed:
        summary.update({
            "status": "failed",
            "complete": False,
            "failed_phase": "final-identity-check",
            "source_changed_during_run": ending_source if source_changed else None,
            "toolchain_changed_during_run": toolchain_changed,
            "finished_at": _utc_now()
        })
        _write_summary(summary_path, summary)
        return 1
    summary["status"] = "passed"
    summary["complete"] = True
    summary["finished_at"] = _utc_now()
    _write_summary(summary_path, summary)
    print(f"MATRIX_OK {summary_path}", flush=True)
    return 0


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    chips = tuple(args.chips or _CHIPS)
    dtypes = tuple(args.dtypes or _DTYPES)
    cases = _selected_cases(args)
    if (len(set(chips)) != len(chips) or len(set(dtypes)) != len(dtypes) or
            len(set(cases)) != len(cases)):
        raise RuntimeError("duplicate chip, dtype, or case selections are not allowed")

    repo_root = Path(__file__).resolve().parents[3]
    output_dir = args.output_dir.expanduser().resolve()
    if (output_dir / "summary.json").exists():
        raise RuntimeError(f"refusing to overwrite an existing matrix summary in {output_dir}")
    if output_dir.is_dir() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to mix a matrix with non-empty directory {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(
            f"refusing to reuse an existing matrix output directory {output_dir}") from exc

    scratch = Path(tempfile.mkdtemp(prefix=".scratch-", dir=output_dir))
    try:
        environment = _worker_environment(repo_root, scratch, args.runtime_mode, args.device_id)
        if args.runtime_mode == "pcie":
            lock_acquired = False
            try:
                with TPUInstructionProfiler.exclusive_pcie_device(0):
                    lock_acquired = True
                    return _run_matrix(args, repo_root, output_dir, environment)
            except Exception as exc:
                summary_path = output_dir / "summary.json"
                failure: dict[str, Any] = {}
                if summary_path.is_file():
                    try:
                        payload = json.loads(summary_path.read_text(encoding="utf-8"))
                        if isinstance(payload, dict):
                            failure.update(payload)
                    except (OSError, ValueError):
                        pass
                if not failure:
                    failure.update({
                        "schema_version": _SCHEMA_VERSION,
                        "matrix_kind": _MATRIX_KIND,
                        "runtime_mode": "pcie",
                        "programming_model": args.programming_model,
                        "started_at": _utc_now()
                    })
                    failure.update(git_source_identity(repo_root))
                failure.update({
                    "status": "failed",
                    "complete": False,
                    "failed_phase": "pcie-session" if lock_acquired else "device-lock",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "finished_at": _utc_now()
                })
                _write_summary(summary_path, failure)
                print(f"PCIe session stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 1
        return _run_matrix(args, repo_root, output_dir, environment)
    finally:
        remove_execution_scratch(scratch)


if __name__ == "__main__":
    raise SystemExit(main())
