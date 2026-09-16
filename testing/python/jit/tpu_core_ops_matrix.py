# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Run the TPU core-op numerical/profile matrix in isolated fresh processes.

Every case is compiled, loaded, launched once, checked against PyTorch, and
profiled by ``TPUInstructionProfiler``.  The first failure stops the matrix;
the profiler has already terminated that worker's process group before this
runner records the partial result.  PCIe additionally requires two explicit
CLI acknowledgements and a numeric device id.  PCIe decoded timing is
best-effort unless ``--require-decoded-timing`` explicitly makes it part of
the acceptance contract; numerical correctness and raw trace collection do
not depend on an optional vendor decoder being installed.
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
        write_json,
    )
    from tpu_profile_matrix_common import (
        profile_report_summary as _report_summary,
        validate_profile_report as _validate_profile_report,
    )

_COPY_CASES = ("copy-fp32-local-roundtrip", "copy-fp32-global-to-global",
               "copy-fp16-local-roundtrip", "copy-fp16-global-to-global")
_CONVERSION_CASES = ("copy-fp16-to-bf16", "copy-bf16-to-fp16")
_MAX_CASES = ("elementwise-max-fp16-dense", "elementwise-max-bf16-dense",
              "elementwise-max-fp32-dense", "elementwise-max-fp16-broadcast",
              "elementwise-max-bf16-broadcast", "elementwise-max-fp32-broadcast",
              "elementwise-max-fp32-negative-infinity")
_BROADCAST_CASES = tuple(
    f"elementwise-{operation}-{dtype}-broadcast" for operation in ("add", "sub", "mul", "div")
    for dtype in ("fp16", "bf16", "fp32"))
_FP32_TRANSPOSE_A_CASES = ("matmul-fp32-transpose-a-overwrite",
                           "matmul-fp32-transpose-a-accumulate")
_CASES = ("elementwise-add", "elementwise-sub", "elementwise-mul", "elementwise-div",
          *_BROADCAST_CASES, *_MAX_CASES, "matmul", *_FP32_TRANSPOSE_A_CASES, *_COPY_CASES,
          *_CONVERSION_CASES)
_MATRIX_KIND = "tpu_core_ops"
_SCHEMA_VERSION = 1
_WORKER_RESULT_PREFIX = "TPU_CORE_NUMERIC_RESULT="
_CMODEL_CONFIGS = (("sg2260e", "tpukernel"), ("sg2260e", "rv"), ("bm1690", "tpukernel"))
_PCIE_CONFIGS = (("sg2260e", "tpukernel"), ("sg2260e", "rv"))


def _worker_environment(repo_root: Path, runtime_mode: str,
                        device_id: Optional[int]) -> dict[str, str]:
    if __package__:
        from .tpu_demo_ops_matrix import worker_environment
    else:
        from tpu_demo_ops_matrix import worker_environment
    return worker_environment(repo_root, runtime_mode, device_id)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-mode", choices=("cmodel", "pcie"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--chip", choices=("sg2260e", "bm1690"))
    parser.add_argument("--programming-model", choices=("tpukernel", "rv"))
    parser.add_argument("--case", choices=_CASES, action="append", dest="cases")
    parser.add_argument("--device-id", type=int)
    parser.add_argument("--allow-pcie", action="store_true")
    parser.add_argument("--allow-pcie-profile", action="store_true")
    parser.add_argument("--bm-cmodel-summary", type=Path)
    parser.add_argument("--sg-cmodel-summary", type=Path)
    parser.add_argument(
        "--all-pcie-cases",
        action="store_true",
        help="explicitly acknowledge scheduling every selected low-level PCIe case",
    )
    parser.add_argument(
        "--pcie-decoder-python",
        type=Path,
        help="Python executable used only by the offline PCIe decoder child",
    )
    parser.add_argument(
        "--pcie-decoder-pythonpath",
        type=Path,
        action="append",
        default=[],
        help=("decoder-only import directory; repeat for multiple directories; "
              "never added to the compile/worker PYTHONPATH"),
    )
    parser.add_argument(
        "--require-decoded-timing",
        action="store_true",
        help=("make parser_status=ready and at least one valid decoded device "
              "timing row mandatory; by default PCIe accepts numerical success "
              "plus raw trace and treats decoding as best-effort"),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("--timeout must be finite and positive")
    if args.runtime_mode == "pcie":
        if not args.allow_pcie or not args.allow_pcie_profile:
            raise RuntimeError("PCIe requires both --allow-pcie and --allow-pcie-profile")
        if (isinstance(args.device_id, bool) or not isinstance(args.device_id, int) or
                not 0 <= args.device_id <= 2**31 - 1):
            raise RuntimeError("PCIe requires a non-negative 32-bit --device-id")
        if args.device_id != 0:
            raise RuntimeError("this single-card validation host accepts only --device-id 0")
        if args.chip != "sg2260e":
            raise RuntimeError("this machine requires explicit --chip sg2260e for PCIe cases")
        if not args.cases and not args.all_pcie_cases:
            raise RuntimeError("PCIe requires an explicit --case or --all-pcie-cases")
        if args.bm_cmodel_summary is None or args.sg_cmodel_summary is None:
            raise RuntimeError(
                "PCIe requires --bm-cmodel-summary and --sg-cmodel-summary promotion evidence")
    elif args.device_id is not None or args.allow_pcie or args.allow_pcie_profile:
        raise RuntimeError("PCIe acknowledgements must not be supplied to CModel")
    elif args.all_pcie_cases:
        raise RuntimeError("--all-pcie-cases is valid only with PCIe")
    elif args.chip is None:
        raise RuntimeError("CModel promotion requires one explicit --chip per invocation")
    if args.runtime_mode != "pcie" and (args.pcie_decoder_python is not None or
                                        args.pcie_decoder_pythonpath or args.require_decoded_timing
                                        or args.bm_cmodel_summary or args.sg_cmodel_summary):
        raise RuntimeError("PCIe decoder/promotion options must not be supplied to CModel")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _worker_payload(stdout_path: Path) -> dict[str, Any]:
    return unique_prefixed_json_payload(stdout_path, _WORKER_RESULT_PREFIX, "core-op worker")


def _validate_worker_payload(
    payload: Mapping[str, Any],
    *,
    chip: str,
    programming_model: str,
    runtime_mode: str,
    case: str,
) -> None:
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise RuntimeError("core-op worker result has the wrong schema_version")
    expected = {
        "status": "passed",
        "chip": chip,
        "programming_model": programming_model,
        "runtime_mode": runtime_mode,
        "case": case,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise RuntimeError(
                f"core-op worker result {field} does not identify the scheduled case")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("passed") is not True:
        raise RuntimeError("core-op worker result lacks metrics.passed=true")


def _validate_pcie_promotion(
    repo_root: Path,
    args: argparse.Namespace,
    configurations: tuple[tuple[str, str], ...],
    cases: tuple[str, ...],
    current_toolchain: Mapping[str, Any],
    pcie_started_at: str,
) -> dict[str, Any]:
    """Require matching BM1690 and SG2260E CModel evidence for every launch."""

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
    for label, payload in (("BM1690", bm), ("SG2260E", sg)):
        if payload.get("git_commit") != commit:
            raise RuntimeError(f"{label} core-op summary commit does not match current source")
        if payload.get("source_state_sha256") != source_digest:
            raise RuntimeError(
                f"{label} core-op summary source digest does not match current source")
        observed_toolchain = payload.get("toolchain_identity")
        if not isinstance(observed_toolchain, dict):
            raise RuntimeError(f"{label} core-op summary has no toolchain identity")
        if observed_toolchain.get("runtime_mode") != "cmodel":
            raise RuntimeError(f"{label} core-op summary has the wrong toolchain runtime mode")
        for field in _COMMON_TOOLCHAIN_FIELDS:
            if observed_toolchain.get(field) != current_toolchain.get(field):
                raise RuntimeError(
                    f"{label} core-op summary content identity for {field} does not match")

    def passing_cases(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        results = payload.get("cases")
        if not isinstance(results, dict):
            raise RuntimeError("core-op promotion summary has no case result map")
        return results

    bm_results = passing_cases(bm)
    sg_results = passing_cases(sg)
    missing = []
    for _chip, programming_model in configurations:
        for case in cases:
            required = (
                (f"bm1690/tpukernel/{case}", bm_results),
                (f"sg2260e/{programming_model}/{case}", sg_results),
            )
            for key, results in required:
                result = results.get(key)
                if not isinstance(result, dict) or result.get("status") != "passed":
                    missing.append(key)
                    continue
                raw_count = result.get("raw_instruction_count")
                if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count <= 0:
                    raise RuntimeError(f"core-op promotion result has no raw instructions: {key}")
                numeric = result.get("numeric")
                if not isinstance(numeric, dict):
                    raise RuntimeError(f"core-op promotion result has no numeric payload: {key}")
                expected_chip, expected_model, expected_case = key.split("/", 2)
                _validate_worker_payload(
                    numeric,
                    chip=expected_chip,
                    programming_model=expected_model,
                    runtime_mode="cmodel",
                    case=expected_case,
                )
    if missing:
        raise RuntimeError("PCIe core-op promotion evidence is missing passing cases: " +
                           ", ".join(sorted(set(missing))))
    return {
        "git_commit": commit,
        "source_state_sha256": source_digest,
        "bm1690_summary": bm["_resolved_path"],
        "bm1690_summary_sha256": bm["_sha256"],
        "sg2260e_summary": sg["_resolved_path"],
        "sg2260e_summary_sha256": sg["_sha256"],
        "validated_case_count": len(configurations) * len(cases),
    }


def _run_matrix(args: argparse.Namespace, repo_root: Path, output_dir: Path,
                configurations: tuple[tuple[str, str], ...], cases: tuple[str, ...],
                environment: dict[str, str]) -> int:
    summary_path = output_dir / "summary.json"
    execution_root = repo_root
    worker_environment_values = dict(environment)
    summary: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "matrix_kind": _MATRIX_KIND,
        "status": "running",
        "runtime_mode": args.runtime_mode,
        "acceptance":
            ("numeric-raw-and-decoded-timing" if args.require_decoded_timing else "numeric-and-raw"
            ),
        "decoded_timing_required": args.require_decoded_timing,
        "complete": False,
        "started_at": _utc_now(),
        "finished_at": None,
        "scheduled_case_count": len(configurations) * len(cases),
        "completed_case_count": 0,
        "passed_case_count": 0,
        "failed_case_count": 0,
        "cancelled_case_count": 0,
        "scheduled": [{
            "chip": chip,
            "programming_model": model,
            "case": case
        } for chip, model in configurations for case in cases],
        "target_scope": matrix_target_scope(configurations),
        "cases": {},
    }
    summary.update(git_source_identity(repo_root))
    write_json(summary_path, summary)
    decoder_config = {
        "pcie_decoder_python": getattr(args, "pcie_decoder_python", None),
        "pcie_decoder_pythonpath": tuple(getattr(args, "pcie_decoder_pythonpath", ()) or ()),
    }

    try:
        summary["toolchain_identity"] = toolchain_identity(worker_environment_values,
                                                           args.runtime_mode)
        worker_environment_values = pin_native_worker_libraries(worker_environment_values,
                                                                summary["toolchain_identity"])
        if args.runtime_mode == "pcie":
            assert args.device_id is not None
            summary["promotion_evidence"] = _validate_pcie_promotion(
                repo_root,
                args,
                configurations,
                cases,
                summary["toolchain_identity"],
                summary["started_at"],
            )
            snapshot_root = Path(worker_environment_values["TMPDIR"]) / "execution-source"
            summary["execution_snapshot"] = materialize_execution_snapshot(
                repo_root,
                snapshot_root,
                summary["promotion_evidence"]["git_commit"],
            )
            execution_root = snapshot_root
            worker_environment_values = pin_worker_environment(
                worker_environment_values,
                repo_root=repo_root,
                snapshot_root=snapshot_root,
                toolchain_identity=summary["toolchain_identity"],
            )
        if args.require_decoded_timing:
            chip, programming_model = configurations[0]
            print("PREFLIGHT PCIe decoder", flush=True)
            preflight_config = TPUProfilingConfig(
                chip=chip,
                programming_model=programming_model,
                runtime_mode="pcie",
                output_dir=output_dir,
                label="pcie-decoder-preflight",
                timeout_s=args.timeout,
                postprocess=True,
                **decoder_config,
            )
            summary["decoder_preflight"] = {
                "status":
                    "ready",
                "identity":
                    dict(
                        TPUInstructionProfiler(preflight_config).preflight_pcie_decoder(
                            environment=worker_environment_values)),
            }
        if args.runtime_mode == "pcie":
            assert args.device_id is not None
            tpu_smi = Path(summary["toolchain_identity"]["pcie"]["tpu_smi"]["path"])
            summary["board_preflight"] = board_health(args.device_id, tpu_smi)
        write_json(summary_path, summary)
    except KeyboardInterrupt:
        summary.update({
            "status": "cancelled",
            "failed_phase": "preflight",
            "finished_at": _utc_now(),
        })
        write_json(summary_path, summary)
        raise
    except Exception as exc:
        summary.update({
            "status": "failed",
            "failed_phase": "preflight",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "finished_at": _utc_now(),
        })
        write_json(summary_path, summary)
        print(f"STOP preflight: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    worker = execution_root / "testing/python/jit/tpu_profile_worker.py"
    for chip, programming_model in configurations:
        for case in cases:
            key = f"{chip}/{programming_model}/{case}"
            print(f"RUN {key}", flush=True)
            command = [sys.executable, str(worker), "--case", case]
            launch_attempted = False
            postflight_attempted = False
            result: Optional[dict[str, Any]] = None
            try:
                if args.runtime_mode == "pcie":
                    assert_source_identity_unchanged(repo_root, summary)
                    assert_toolchain_identity_unchanged(worker_environment_values,
                                                        summary["toolchain_identity"])
                config = TPUProfilingConfig(
                    chip=chip,
                    programming_model=programming_model,
                    runtime_mode=args.runtime_mode,
                    output_dir=output_dir,
                    label=f"{chip}-{programming_model}-{case}",
                    timeout_s=args.timeout,
                    # Decode CModel traces whenever a caller supplies PerfAI;
                    # raw-only evidence remains valid when it is unavailable.
                    postprocess=True,
                    **decoder_config,
                )
                profiler = TPUInstructionProfiler(config)
                launch_attempted = True
                report = (
                    profiler.run_pcie(command, environment=worker_environment_values)
                    if args.runtime_mode == "pcie" else profiler.run_cmodel(
                        command, environment=worker_environment_values))
                _validate_profile_report(report, require_decoded_timing=args.require_decoded_timing)
                numeric = _worker_payload(Path(report.stdout_path))
                _validate_worker_payload(
                    numeric,
                    chip=chip,
                    programming_model=programming_model,
                    runtime_mode=args.runtime_mode,
                    case=case,
                )
                result = _report_summary(report, require_decoded_timing=args.require_decoded_timing)
                result["numeric"] = numeric
                if args.runtime_mode == "pcie":
                    assert args.device_id is not None
                    tpu_smi = Path(summary["toolchain_identity"]["pcie"]["tpu_smi"]["path"])
                    postflight_attempted = True
                    result["board_postflight"] = board_health(
                        args.device_id, tpu_smi, quarantine_on_failure=True)
                summary["cases"][key] = result
                print(
                    f"PASS {key} raw={len(report.raw_instructions)} "
                    f"timed={len(report.instruction_timings)}",
                    flush=True,
                )
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
                if (args.runtime_mode == "pcie" and launch_attempted and not postflight_attempted):
                    assert args.device_id is not None
                    try:
                        tpu_smi = Path(summary["toolchain_identity"]["pcie"]["tpu_smi"]["path"])
                        summary["board_after_cancel"] = board_health(
                            args.device_id, tpu_smi, quarantine_on_failure=True)
                    except Exception as health_error:
                        summary["board_after_cancel_error"] = (
                            f"{type(health_error).__name__}: {health_error}")
                summary["finished_at"] = _utc_now()
                write_json(summary_path, summary)
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
                        "error": str(exc),
                    }
                if (args.runtime_mode == "pcie" and launch_attempted and not postflight_attempted):
                    assert args.device_id is not None
                    try:
                        tpu_smi = Path(summary["toolchain_identity"]["pcie"]["tpu_smi"]["path"])
                        result["board_after_failure"] = board_health(
                            args.device_id, tpu_smi, quarantine_on_failure=True)
                    except Exception as health_error:
                        result["board_after_failure_error"] = (
                            f"{type(health_error).__name__}: {health_error}")
                summary["cases"][key] = result
                summary["completed_case_count"] += 1
                summary["failed_case_count"] += 1
                summary["status"] = "failed"
                summary["stopped_after"] = key
                summary["finished_at"] = _utc_now()
                write_json(summary_path, summary)
                print(f"STOP {key}: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 1
            summary["completed_case_count"] += 1
            summary["passed_case_count"] += 1
            write_json(summary_path, summary)

    try:
        ending_identity = git_source_identity(repo_root)
        source_fields = ("git_commit", "implementation_worktree_dirty", "source_state_sha256")
        source_changed = (not isinstance(summary.get("source_state_sha256"), str) or any(
            ending_identity.get(field) != summary.get(field) for field in source_fields))
        ending_toolchain = toolchain_identity(worker_environment_values, args.runtime_mode)
    except KeyboardInterrupt:
        summary.update({
            "status": "cancelled",
            "complete": False,
            "failed_phase": "final-identity-check",
            "finished_at": _utc_now(),
        })
        write_json(summary_path, summary)
        raise
    except Exception as exc:
        summary.update({
            "status": "failed",
            "complete": False,
            "failed_phase": "final-identity-check",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "finished_at": _utc_now(),
        })
        write_json(summary_path, summary)
        print(f"STOP final identity check: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if source_changed or ending_toolchain != summary["toolchain_identity"]:
        summary["status"] = "failed"
        summary["complete"] = False
        summary["failed_phase"] = "final-identity-check"
        summary["source_changed_during_run"] = ending_identity if source_changed else None
        summary["toolchain_changed_during_run"] = ending_toolchain != summary["toolchain_identity"]
        summary["finished_at"] = _utc_now()
        write_json(summary_path, summary)
        print("STOP source/toolchain identity changed during matrix execution", file=sys.stderr)
        return 1
    summary["status"] = "passed"
    summary["complete"] = True
    summary["finished_at"] = _utc_now()
    write_json(summary_path, summary)
    print(f"MATRIX_OK {summary_path}", flush=True)
    return 0


def main() -> int:
    args = _parse_args()
    _validate_args(args)

    repo_root = Path(__file__).resolve().parents[3]
    configurations = _PCIE_CONFIGS if args.runtime_mode == "pcie" else _CMODEL_CONFIGS
    configurations = tuple(
        item for item in configurations if (args.chip is None or item[0] == args.chip) and
        (args.programming_model is None or item[1] == args.programming_model))
    if not configurations:
        raise RuntimeError("the requested chip/programming-model pair is not in this matrix")
    cases = tuple(args.cases) if args.cases else _CASES
    if len(set(cases)) != len(cases):
        raise RuntimeError("duplicate --case selections are not allowed")
    output_dir = args.output_dir.expanduser().resolve()
    if (output_dir / "summary.json").exists():
        raise RuntimeError(f"refusing to overwrite an existing matrix summary in {output_dir}")
    if output_dir.is_dir() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to mix a matrix with non-empty directory {output_dir}")
    # Claim the output path atomically.  Accepting an existing empty directory
    # would leave a check/create race in which two runners could overwrite the
    # same atomic summary while taking the device lock in sequence.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise RuntimeError(
            f"refusing to reuse an existing matrix output directory {output_dir}") from error
    scratch_dir = Path(tempfile.mkdtemp(prefix=".scratch-", dir=output_dir))
    try:
        environment = _worker_environment(repo_root, args.runtime_mode, args.device_id)
        environment = pin_worker_cache_environment(environment, scratch_dir)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        if args.runtime_mode == "pcie":
            assert args.device_id is not None
            lock_acquired = False
            try:
                with TPUInstructionProfiler.exclusive_pcie_device(args.device_id):
                    lock_acquired = True
                    return _run_matrix(args, repo_root, output_dir, configurations, cases,
                                       environment)
            except Exception as error:
                summary_path = output_dir / "summary.json"
                failure: dict[str, Any] = {}
                if summary_path.is_file():
                    try:
                        candidate = json.loads(summary_path.read_text(encoding="utf-8"))
                        if isinstance(candidate, dict):
                            failure.update(candidate)
                    except (OSError, ValueError):
                        pass
                if not failure:
                    failure.update({
                        "schema_version": _SCHEMA_VERSION,
                        "matrix_kind": _MATRIX_KIND,
                        "runtime_mode": "pcie",
                        "started_at": _utc_now(),
                    })
                    failure.update(git_source_identity(repo_root))
                failure.update({
                    "status": "failed",
                    "complete": False,
                    "failed_phase": "pcie-session" if lock_acquired else "device-lock",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "finished_at": _utc_now(),
                })
                write_json(summary_path, failure)
                print(f"PCIe session stopped: {type(error).__name__}: {error}", file=sys.stderr)
                return 1
        return _run_matrix(args, repo_root, output_dir, configurations, cases, environment)
    finally:
        # Delete only the unique directory created by this invocation.  Trace,
        # decoder, and report artifacts are siblings and remain intact.
        remove_execution_scratch(scratch_dir)


if __name__ == "__main__":
    raise SystemExit(main())
