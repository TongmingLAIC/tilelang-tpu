# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Run the TPU-Kernel numerical conformance matrix in isolated processes.

The default matrix is CModel-only and covers BM1690 plus SG2260E. PCIe is
restricted to SG2260E device 0 and requires two acknowledgements, an explicit
case scope, and matching clean BM1690/SG2260E CModel summaries.
Every case owns a fresh parent-death-supervised process group.  On a timeout
or first failure, the runner sends SIGTERM to that group, waits a bounded
grace interval, follows with SIGKILL if necessary, records the partial JSON
report, and stops.  If an outer watchdog kills the runner itself, the shared
TPU supervisor kills the worker group before it can become orphaned.

Examples::

    python testing/python/jit/tpukernel_ops_matrix.py \
        --output-dir research/artifacts/tpukernel-numeric

    python testing/python/jit/tpukernel_ops_matrix.py \
        --output-dir research/artifacts/tpukernel-reductions \
        --chip sg2260e --op reduce-sum --op reduce-max

    # Deliberately dangerous: use exact clean CModel summaries from this runner.
    python testing/python/jit/tpukernel_ops_matrix.py \
        --output-dir research/artifacts/tpukernel-pcie \
        --runtime-mode pcie --allow-pcie --allow-pcie-load \
        --device-id 0 --chip sg2260e --op add \
        --bm-cmodel-summary research/artifacts/tpukernel-bm/summary.json \
        --sg-cmodel-summary research/artifacts/tpukernel-sg/summary.json
"""

from __future__ import annotations

import argparse
from contextlib import suppress
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Optional, Sequence

from tilelang.temp_storage import tilelang_temp_dir

from tpukernel_ops_worker import (
    CaseSpec,
    _CHIPS,
    _CORE_COUNTS,
    _FLOAT_DTYPES,
    _RESULT_PREFIX,
    _RUNTIME_MODES,
    build_case_specs,
)

if __package__:
    from .tpu_matrix_common import (
        git_source_identity,
        matrix_target_scope,
        unique_prefixed_json_payload_text,
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
        worker_environment as _base_worker_environment,
        write_json as _write_json,
    )
else:
    from tpu_matrix_common import (
        git_source_identity,
        matrix_target_scope,
        unique_prefixed_json_payload_text,
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
        worker_environment as _base_worker_environment,
        write_json as _write_json,
    )

_INTEGER_DTYPES = ("int8", "uint8", "int16", "uint16", "int32", "uint32")
_ALL_DTYPES = _FLOAT_DTYPES + _INTEGER_DTYPES
_OPERATIONS = tuple(sorted({case.operation for case in build_case_specs()}))
_SCHEMA_VERSION = 1
_MATRIX_KIND = "tpukernel_ops"
_PROCESS_PIPE_DRAIN_S = 5.0
_TPU_PROCESS_SUPERVISOR = (
    Path(__file__).resolve().parents[3] / "tilelang" / "jit" / "_tpu_profile_supervisor.py")
_PCIE_CHILD_UNSET_KEYS = (
    "TILELANG_TPU_ALLOW_PCIE_PROFILE",
    "TILELANG_TPU_PROFILE_SESSION",
    "TILELANG_TPU_PROFILE_CHIP",
    "TILELANG_TPU_PROFILE_PROGRAMMING_MODEL",
    "TILELANG_TPU_PROFILE_RUNTIME_MODE",
    "TILELANG_TPU_PROFILE_OUTPUT_DIR",
    "BMLIB_ENABLE_ALL_PROFILE",
    "FILE_DUMP_CMD",
    "PROFILE_BOOK_KEEPING",
    "PROFILE_RECORD_SIZE",
    "TPU_RT_CORE_NUM",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory in which summary.json and per-case JSON are written",
    )
    parser.add_argument(
        "--runtime-mode",
        choices=_RUNTIME_MODES,
        default="cmodel",
        help="defaults to the board-safe CModel runtime",
    )
    parser.add_argument(
        "--chip",
        choices=_CHIPS,
        action="append",
        dest="chips",
        help="select exactly one explicit chip for each matrix invocation",
    )
    parser.add_argument(
        "--op",
        choices=_OPERATIONS,
        action="append",
        dest="operations",
        help="repeat to select operation families",
    )
    parser.add_argument(
        "--dtype",
        choices=_ALL_DTYPES,
        action="append",
        dest="dtypes",
        help="repeat to select case output/payload dtypes",
    )
    parser.add_argument(
        "--case",
        choices=tuple(case.case_id for case in build_case_specs()),
        action="append",
        dest="case_ids",
        help="repeat to select exact case ids (intersects --op/--dtype)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="compile+load+one-launch timeout for each fresh worker",
    )
    parser.add_argument(
        "--kill-grace",
        type=float,
        default=3.0,
        help="seconds between process-group SIGTERM and SIGKILL",
    )
    parser.add_argument(
        "--allow-pcie",
        action="store_true",
        help="mandatory explicit acknowledgement for PCIe execution",
    )
    parser.add_argument(
        "--allow-pcie-load",
        action="store_true",
        help="second mandatory acknowledgement that this matrix loads the physical board",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        help="mandatory non-negative board id for PCIe execution",
    )
    parser.add_argument(
        "--bm-cmodel-summary",
        type=Path,
        help="clean BM1690 CModel summary required to promote exact PCIe cases",
    )
    parser.add_argument(
        "--sg-cmodel-summary",
        type=Path,
        help="clean SG2260E CModel summary required to promote exact PCIe cases",
    )
    parser.add_argument(
        "--all-pcie-cases",
        action="store_true",
        help="explicitly acknowledge scheduling the full supported PCIe matrix",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="print the filtered declarative case list as JSON and exit safely",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("--timeout must be finite and positive")
    if not math.isfinite(args.kill_grace) or args.kill_grace < 0:
        raise ValueError("--kill-grace must be finite and non-negative")
    for option, values in (("--chip", args.chips), ("--op", args.operations),
                           ("--dtype", args.dtypes), ("--case", args.case_ids)):
        if values and len(values) != len(set(values)):
            raise RuntimeError(f"duplicate {option} selections are not allowed")
    if args.list_cases:
        if (args.allow_pcie or args.allow_pcie_load or args.device_id is not None or
                args.bm_cmodel_summary is not None or args.sg_cmodel_summary is not None or
                args.all_pcie_cases):
            raise RuntimeError("hardware execution controls are invalid with --list-cases")
        return
    if args.runtime_mode == "pcie":
        if not args.allow_pcie or not args.allow_pcie_load:
            raise RuntimeError(
                "PCIe numerical execution requires --allow-pcie and --allow-pcie-load")
        if (isinstance(args.device_id, bool) or not isinstance(args.device_id, int) or
                args.device_id != 0):
            raise RuntimeError("this single-card validation host accepts only --device-id 0")
        if args.chips is None or tuple(dict.fromkeys(args.chips)) != ("sg2260e",):
            raise RuntimeError("this machine can run PCIe cases only for explicit --chip sg2260e")
        filters_selected = bool(args.operations or args.case_ids)
        if args.all_pcie_cases and (filters_selected or args.dtypes):
            raise RuntimeError(
                "--all-pcie-cases cannot be combined with --op/--case/--dtype filters")
        if not args.all_pcie_cases and not filters_selected:
            raise RuntimeError("PCIe requires an explicit --op/--case subset or --all-pcie-cases")
        if args.bm_cmodel_summary is None or args.sg_cmodel_summary is None:
            raise RuntimeError(
                "PCIe requires --bm-cmodel-summary and --sg-cmodel-summary promotion evidence")
    else:
        if (args.allow_pcie or args.allow_pcie_load or args.device_id is not None or
                args.bm_cmodel_summary is not None or args.sg_cmodel_summary is not None or
                args.all_pcie_cases):
            raise RuntimeError("PCIe controls are invalid for CModel execution")
        if args.chips is None or len(args.chips) != 1:
            raise RuntimeError("CModel promotion requires exactly one explicit --chip")
    if not args.list_cases and args.output_dir is None:
        raise RuntimeError("--output-dir is required unless --list-cases is used")


def _unique_in_order(values: Optional[Sequence[str]], default: Sequence[str]) -> tuple[str, ...]:
    source = values if values else default
    return tuple(dict.fromkeys(source))


def _selected_cases(args: argparse.Namespace) -> tuple[CaseSpec, ...]:
    operations = set(args.operations or _OPERATIONS)
    dtypes = set(args.dtypes or _ALL_DTYPES)
    case_ids = set(args.case_ids) if args.case_ids else None
    selected = tuple(case for case in build_case_specs()
                     if case.operation in operations and case.dtype in dtypes and
                     (case_ids is None or case.case_id in case_ids))
    if not selected:
        raise RuntimeError("the requested case filters select no TPU-Kernel probes")
    return selected


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _worker_environment(base_environment: Mapping[str, str], scratch_dir: Path, case: CaseSpec,
                        chip: str, args: argparse.Namespace) -> dict[str, str]:
    environment = pin_worker_cache_environment(base_environment, scratch_dir)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["TILELANG_TPU_BENCHMARK_RUNS"] = "0"
    environment.update({
        "TILELANG_TPU_NUMERIC_SESSION": "1",
        "TILELANG_TPU_NUMERIC_CASE": case.case_id,
        "TILELANG_TPU_NUMERIC_CHIP": chip,
        "TILELANG_TPU_NUMERIC_RUNTIME_MODE": args.runtime_mode,
    })

    # Never allow inherited shell state to turn a CModel command into a board
    # load or silently turn a numerical run into a profiling run.  Conversely,
    # install PCIe gates only after CLI validation.
    for name in ("TILELANG_TPU_NUMERIC_ALLOW_PCIE", "TILELANG_TPU_ALLOW_PCIE_LOAD",
                 "TILELANG_TPU_ALLOW_PCIE_PROFILE", "TILELANG_TPU_DEVICE_ID",
                 "TILELANG_TPU_PROFILE_SESSION", "TILELANG_TPU_PROFILE_CHIP",
                 "TILELANG_TPU_PROFILE_PROGRAMMING_MODEL", "TILELANG_TPU_PROFILE_RUNTIME_MODE",
                 "TILELANG_TPU_PROFILE_OUTPUT_DIR", "BMLIB_ENABLE_ALL_PROFILE", "FILE_DUMP_CMD",
                 "PROFILE_BOOK_KEEPING", "PROFILE_RECORD_SIZE", "TPU_RT_CORE_NUM"):
        environment.pop(name, None)
    if args.runtime_mode == "pcie":
        assert args.device_id is not None
        environment.update({
            "TILELANG_TPU_NUMERIC_ALLOW_PCIE": "1",
            "TILELANG_TPU_ALLOW_PCIE_LOAD": "1",
            "TILELANG_TPU_DEVICE_ID": str(args.device_id),
        })
    else:
        environment["TPU_RT_CORE_NUM"] = str(_CORE_COUNTS[chip])
    return environment


def _json_normalized(value: Any) -> Any:
    """Normalize tuple-bearing declarative specs like an actual JSON summary."""

    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _validate_numeric_identity(payload: Mapping[str, Any], *, case: CaseSpec, chip: str,
                               runtime_mode: str) -> None:
    """Reject a successful marker unless it names this exact scheduled probe."""

    expected = {
        "status": "passed",
        "case": _json_normalized(case.to_json()),
        "chip": chip,
        "programming_model": "tpukernel",
        "runtime_mode": runtime_mode,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise RuntimeError(f"worker result has wrong {field}: expected {value!r}, "
                               f"observed {payload.get(field)!r}")
    if not isinstance(payload.get("metrics"), dict):
        raise RuntimeError("worker result has no numerical metrics object")


def _validate_pcie_promotion(repo_root: Path, args: argparse.Namespace, cases: tuple[CaseSpec, ...],
                             current_toolchain: Mapping[str, Any],
                             pcie_started_at: str) -> dict[str, Any]:
    """Require clean, content-identical CModel success for every PCIe case."""

    assert args.bm_cmodel_summary is not None and args.sg_cmodel_summary is not None
    if (current_toolchain.get("runtime_mode") != "pcie" or
            not isinstance(current_toolchain.get("pcie"), dict)):
        raise RuntimeError("PCIe promotion requires a complete PCIe toolchain identity")
    current = git_source_identity(repo_root)
    if current.get("implementation_worktree_dirty") is not False:
        raise RuntimeError(
            "PCIe promotion requires the current implementation worktree to be clean")
    highlights = ("git_commit", "source_state_sha256")
    if any(not isinstance(current.get(field), str) or not current[field] for field in highlights):
        raise RuntimeError("PCIe promotion requires a verifiable source identity")

    bm, sg = validate_promotion_stages(
        args.bm_cmodel_summary,
        args.sg_cmodel_summary,
        matrix_kind=_MATRIX_KIND,
        schema_version=_SCHEMA_VERSION,
        bm_allowed_scope=(("bm1690", "tpukernel"),),
        sg_allowed_scope=(("sg2260e", "tpukernel"),),
        pcie_started_at=pcie_started_at,
    )
    for label, payload in (("BM1690", bm), ("SG2260E", sg)):
        if payload.get("git_commit") != current["git_commit"]:
            raise RuntimeError(f"{label} TPU-Kernel summary commit does not match current source")
        if payload.get("source_state_sha256") != current["source_state_sha256"]:
            raise RuntimeError(
                f"{label} TPU-Kernel summary source digest does not match current source")
        if payload.get("programming_model") != "tpukernel":
            raise RuntimeError(f"{label} promotion summary is not TPU-Kernel evidence")
        observed_toolchain = payload.get("toolchain_identity")
        if not isinstance(observed_toolchain, dict):
            raise RuntimeError(f"{label} TPU-Kernel summary has no toolchain identity")
        if observed_toolchain.get("runtime_mode") != "cmodel":
            raise RuntimeError(f"{label} TPU-Kernel summary has the wrong runtime mode")
        for field in _COMMON_TOOLCHAIN_FIELDS:
            if observed_toolchain.get(field) != current_toolchain.get(field):
                raise RuntimeError(
                    f"{label} TPU-Kernel toolchain identity for {field} does not match")

    def exact_results(payload: Mapping[str, Any], chip: str) -> dict[str, Mapping[str, Any]]:
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            raise RuntimeError("TPU-Kernel promotion summary has no result list")
        indexed: dict[str, Mapping[str, Any]] = {}
        for result in raw_results:
            if not isinstance(result, dict):
                raise RuntimeError("TPU-Kernel promotion summary contains a non-object result")
            raw_case = result.get("case")
            if not isinstance(raw_case, dict) or not isinstance(raw_case.get("case_id"), str):
                raise RuntimeError("TPU-Kernel promotion result has no exact case id")
            if result.get("chip") != chip:
                continue
            case_id = raw_case["case_id"]
            if case_id in indexed:
                raise RuntimeError(f"duplicate TPU-Kernel promotion result: {chip}/{case_id}")
            indexed[case_id] = result
        return indexed

    bm_results = exact_results(bm, "bm1690")
    sg_results = exact_results(sg, "sg2260e")
    missing = []
    for case in cases:
        expected_case = _json_normalized(case.to_json())
        for chip, results in (("bm1690", bm_results), ("sg2260e", sg_results)):
            result = results.get(case.case_id)
            if result is None or result.get("status") != "passed":
                missing.append(f"cmodel/{chip}/{case.case_id}")
                continue
            if result.get("runtime_mode") != "cmodel" or result.get("case") != expected_case:
                raise RuntimeError(
                    f"TPU-Kernel promotion result identity mismatch: {chip}/{case.case_id}")
            if not isinstance(result.get("metrics"), dict):
                raise RuntimeError(
                    f"TPU-Kernel promotion result has no metrics: {chip}/{case.case_id}")
    if missing:
        raise RuntimeError("PCIe TPU-Kernel promotion evidence is missing passing exact cases: " +
                           ", ".join(sorted(missing)))
    return {
        "git_commit": current["git_commit"],
        "source_state_sha256": current["source_state_sha256"],
        "bm1690_summary": bm["_resolved_path"],
        "bm1690_summary_sha256": bm["_sha256"],
        "sg2260e_summary": sg["_resolved_path"],
        "sg2260e_summary_sha256": sg["_sha256"],
        "validated_case_count": len(cases),
    }


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Treat an inaccessible group as live; the subsequent signal error is
        # retained in the machine-readable termination report.
        return True


def _terminate_process_group(process: subprocess.Popen[str],
                             grace_seconds: float) -> dict[str, Any]:
    """Terminate the worker's dedicated process group, escalating if live."""

    process_group = process.pid
    report: dict[str, Any] = {
        "process_group": process_group,
        "sigterm_sent": False,
        "sigkill_sent": False,
        "errors": [],
    }
    try:
        os.killpg(process_group, signal.SIGTERM)
        report["sigterm_sent"] = True
    except ProcessLookupError:
        pass
    except OSError as error:
        report["errors"].append(f"SIGTERM: {type(error).__name__}: {error}")

    deadline = time.monotonic() + grace_seconds
    while _process_group_exists(process_group) and time.monotonic() < deadline:
        process.poll()  # Reap the direct worker if SIGTERM made it exit.
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    process.poll()
    if _process_group_exists(process_group):
        try:
            os.killpg(process_group, signal.SIGKILL)
            report["sigkill_sent"] = True
        except ProcessLookupError:
            pass
        except OSError as error:
            report["errors"].append(f"SIGKILL: {type(error).__name__}: {error}")

    try:
        process.wait(timeout=max(1.0, grace_seconds))
    except subprocess.TimeoutExpired:
        report["errors"].append("direct worker did not reap after SIGKILL")
    report["returncode_after_termination"] = process.poll()
    report["group_exists_after_termination"] = _process_group_exists(process_group)
    return report


def _timeout_output_as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _terminate_and_collect(process: subprocess.Popen[str],
                           grace_seconds: float) -> tuple[str, str, dict[str, Any]]:
    """Terminate a timed-out worker group and drain pipes with a hard bound.

    An escaped or uninterruptible descendant can retain an inherited stdout or
    stderr descriptor after the supervised group has died.  An unbounded final
    ``communicate()`` would then defeat the per-case TPU watchdog.  Preserve
    whatever output is available, close this runner's pipe handles, and return
    the machine-readable termination report even in that pathological case.
    """

    termination = _terminate_process_group(process, grace_seconds)
    termination["pipe_drain_timeout_seconds"] = _PROCESS_PIPE_DRAIN_S
    termination["pipe_drain_timed_out"] = False
    try:
        stdout, stderr = process.communicate(timeout=_PROCESS_PIPE_DRAIN_S)
        return stdout, stderr, termination
    except subprocess.TimeoutExpired as error:
        stdout = _timeout_output_as_text(error.output)
        stderr = _timeout_output_as_text(error.stderr)
        diagnostic = ("TileLang TPU numerical watchdog: process group did not close its "
                      f"output pipes within {_PROCESS_PIPE_DRAIN_S:g}s after termination")
        termination["pipe_drain_timed_out"] = True
        termination["errors"].append(diagnostic)
        stderr = f"{stderr}\n{diagnostic}\n" if stderr else diagnostic + "\n"
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                with suppress(OSError):
                    pipe.close()
        return stdout, stderr, termination


def _spawn_guarded_worker(command: Sequence[str], *, cwd: Path,
                          environment: Mapping[str, str]) -> subprocess.Popen[str]:
    """Start one numerical worker under the shared TPU process-tree guard.

    A new session alone protects the runner's own process group, but it also
    lets a worker outlive the runner if an outer watchdog kills the latter.
    Reuse the profiling path's Linux parent-death supervisor so that such a
    death kills the private worker group before it can become an orphan.
    """

    if not sys.platform.startswith("linux"):
        raise RuntimeError("safe TPU numerical execution requires Linux PR_SET_PDEATHSIG")
    if not _TPU_PROCESS_SUPERVISOR.is_file():
        raise RuntimeError("TPU process-tree supervisor is missing: "
                           f"{_TPU_PROCESS_SUPERVISOR}")
    guarded_command = [
        sys.executable,
        str(_TPU_PROCESS_SUPERVISOR),
        "--parent-pid",
        str(os.getpid()),
        "--",
        *command,
    ]
    return subprocess.Popen(
        guarded_command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def _worker_payload(stdout: str) -> dict[str, Any]:
    return unique_prefixed_json_payload_text(stdout, _RESULT_PREFIX, "TPU-Kernel worker")


def _case_directory(output_dir: Path, chip: str, runtime_mode: str, case: CaseSpec) -> Path:
    # Case ids are generated from a closed alnum/hyphen/dot vocabulary.
    if any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-." for char in case.case_id):
        raise AssertionError(f"unsafe generated case id {case.case_id!r}")
    return output_dir / "cases" / runtime_mode / chip / case.case_id


def _run_one(args: argparse.Namespace, base_environment: Mapping[str, str], output_dir: Path,
             worker: Path, chip: str, case: CaseSpec) -> dict[str, Any]:
    case_dir = _case_directory(output_dir, chip, args.runtime_mode, case)
    case_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path(tempfile.mkdtemp(prefix=".scratch-", dir=tilelang_temp_dir()))
    command = [
        sys.executable,
        "-u",
        str(worker),
        "--case-id",
        case.case_id,
        "--chip",
        chip,
        "--runtime-mode",
        args.runtime_mode,
    ]
    launched_command = list(command)
    try:
        environment = _worker_environment(base_environment, scratch_dir, case, chip, args)
    except BaseException:
        if scratch_dir.exists():
            remove_execution_scratch(scratch_dir)
        raise
    started_at = _utc_now()
    begin = time.monotonic()
    process: Optional[subprocess.Popen[str]] = None
    stdout = ""
    stderr = ""
    termination: Optional[dict[str, Any]] = None
    timed_out = False
    launch_attempted = False
    launch_error: Optional[str] = None
    returncode: Optional[int] = None
    try:
        # Keep autotuner/vendor/compiler byproducts in disposable storage;
        # only the explicit JSON report escapes into case_dir.  The shared
        # supervisor additionally guarantees cleanup if this runner itself is
        # killed by an outer watchdog.
        if args.runtime_mode == "pcie":
            from tilelang.jit import TPUInstructionProfiler

            assert args.device_id is not None
            launch_attempted = True
            # The shared supervisor deliberately overlays an environment on
            # its own process environment. Put the numerical worker below
            # ``env -u`` so inherited profiling/CModel controls cannot leak
            # back in after this runner has sanitized them.
            pcie_command = ["/usr/bin/env"]
            for name in _PCIE_CHILD_UNSET_KEYS:
                pcie_command.extend(("-u", name))
            pcie_command.extend(command)
            launched_command = pcie_command
            supervised = TPUInstructionProfiler.run_supervised_pcie_probe(
                args.device_id,
                pcie_command,
                cwd=scratch_dir,
                environment=environment,
                timeout_s=args.timeout,
            )
            stdout = supervised.stdout
            stderr = supervised.stderr
            timed_out = supervised.timed_out
            returncode = supervised.returncode
            if (supervised.timed_out or supervised.left_live_descendant or
                    not supervised.cleanup_complete):
                termination = {
                    "process_group": supervised.process_group,
                    "left_live_descendant": supervised.left_live_descendant,
                    "cleanup_complete": supervised.cleanup_complete,
                }
        else:
            from tilelang.jit.tpu_cmodel import prepare_cmodel_command

            launched_command, environment = prepare_cmodel_command(
                command, cwd=scratch_dir, environment=environment,
                private_tmp_dir=scratch_dir)
            launch_attempted = True
            process = _spawn_guarded_worker(
                launched_command, cwd=scratch_dir, environment=environment)
            try:
                stdout, stderr = process.communicate(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                # Do not append TimeoutExpired's partial copies: the bounded drain
                # returns the complete captured streams when all descriptors close,
                # or its own latest partial copies when a stuck/escaped descendant
                # retains an inherited pipe.
                stdout, stderr, termination = _terminate_and_collect(process, args.kill_grace)
            returncode = process.poll()
            if returncode == 0 and termination is None and \
                    _process_group_exists(process.pid):
                # A successful direct worker is not allowed to background ordinary
                # children.  Once the supervisor exits, its parent-death contract
                # no longer protects such processes, so clean the group and fail
                # this case before another TPU workload can start.
                termination = _terminate_process_group(process, args.kill_grace)
                launch_error = ("RuntimeError: successful worker left a live descendant in "
                                "its supervised process group")
                returncode = process.poll()
            if returncode not in (0, None) and termination is None:
                # The direct worker may have exited while compiler/runtime children
                # survived.  Clean its still-addressable process group on first
                # failure before any subsequent case could start.
                termination = _terminate_process_group(process, args.kill_grace)
                returncode = process.poll()
    except BaseException as error:
        launch_error = f"{type(error).__name__}: {error}"
        if process is not None:
            termination = _terminate_process_group(process, args.kill_grace)
            returncode = process.poll()
    finally:
        if scratch_dir.exists():
            try:
                remove_execution_scratch(scratch_dir)
            except BaseException as error:
                cleanup_error = f"scratch cleanup failed: {type(error).__name__}: {error}"
                launch_error = (f"{launch_error}; {cleanup_error}"
                                if launch_error else cleanup_error)

    elapsed = time.monotonic() - begin
    parsed_payload: Optional[dict[str, Any]] = None
    parse_error: Optional[str] = None
    try:
        parsed_payload = _worker_payload(stdout)
        if parsed_payload is not None and parsed_payload.get("status") == "passed":
            _validate_numeric_identity(
                parsed_payload, case=case, chip=chip, runtime_mode=args.runtime_mode)
    except RuntimeError as error:
        parse_error = str(error)

    passed = (
        launch_error is None and not timed_out and returncode == 0 and parse_error is None and
        parsed_payload is not None and parsed_payload.get("status") == "passed")
    if timed_out:
        failure = f"worker exceeded {args.timeout:g}s timeout"
    elif launch_error is not None:
        failure = launch_error
    elif parse_error is not None:
        failure = parse_error
    elif parsed_payload is not None and parsed_payload.get("status") != "passed":
        failure = str(parsed_payload.get("error", "worker reported failure"))
    elif returncode != 0:
        failure = f"worker exited with status {returncode}"
    elif parsed_payload is None:
        failure = "worker produced no machine-readable result marker"
    else:
        failure = None

    result: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "case": case.to_json(),
        "chip": chip,
        "programming_model": "tpukernel",
        "runtime_mode": args.runtime_mode,
        "started_at": started_at,
        "elapsed_seconds": elapsed,
        "timeout_seconds": args.timeout,
        "command": launched_command,
        "returncode": returncode,
        "timed_out": timed_out,
        "launch_attempted": launch_attempted,
        "failure": failure,
        "termination": termination,
        "worker_result": parsed_payload,
        "stdout": stdout,
        "stderr": stderr,
    }
    try:
        _write_json(case_dir / "result.json", result)
    except BaseException as error:
        diagnostic = f"result file write failed: {type(error).__name__}: {error}"
        result["status"] = "failed"
        result["failure"] = (f"{result['failure']}; {diagnostic}"
                             if result.get("failure") else diagnostic)
        result["result_file_error"] = diagnostic
    return result


def _summary_case(result: Mapping[str, Any], result_path: Path, output_dir: Path) -> dict[str, Any]:
    worker_result = result.get("worker_result")
    compact: dict[str, Any] = {
        "status": result["status"],
        "case": result["case"],
        "chip": result["chip"],
        "runtime_mode": result["runtime_mode"],
        "elapsed_seconds": result["elapsed_seconds"],
        "result_file": str(result_path.relative_to(output_dir)),
    }
    if result.get("failure") is not None:
        compact["failure"] = result["failure"]
    if result.get("result_file_error") is not None:
        compact["result_file_error"] = result["result_file_error"]
    for field in ("execution_status", "execution_failure", "failed_phase", "board_postflight",
                  "board_postflight_error", "board_postflight_failure"):
        if field in result:
            compact[field] = result[field]
    if isinstance(worker_result, dict):
        if "metrics" in worker_result:
            compact["metrics"] = worker_result["metrics"]
        if "timing" in worker_result:
            compact["worker_timing"] = worker_result["timing"]
    return compact


def _run_matrix(args: argparse.Namespace, repo_root: Path, output_dir: Path, chips: tuple[str, ...],
                cases: tuple[CaseSpec, ...], base_environment: Mapping[str, str]) -> int:
    summary_path = output_dir / "summary.json"
    execution_root = repo_root
    worker_environment_values = dict(base_environment)
    scheduled_case_specs = tuple(
        case for chip in chips for case in cases if chip in case.supported_chips)
    scheduled = [{
        "chip": chip,
        "case": case.to_json()
    } for chip in chips for case in cases if chip in case.supported_chips]
    if not scheduled:
        raise RuntimeError("none of the selected TPU-Kernel probes is supported by the "
                           "selected chip set")
    summary: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "matrix_kind": _MATRIX_KIND,
        "status": "running",
        "complete": False,
        "programming_model": "tpukernel",
        "runtime_mode": args.runtime_mode,
        "chips": list(chips),
        "timeout_seconds_per_case": args.timeout,
        "kill_grace_seconds": args.kill_grace,
        "started_at": _utc_now(),
        "finished_at": None,
        "scheduled_case_count": len(scheduled),
        "completed_case_count": 0,
        "passed_case_count": 0,
        "failed_case_count": 0,
        "cancelled_case_count": 0,
        "target_scope": matrix_target_scope((entry["chip"], "tpukernel") for entry in scheduled),
        "scheduled": scheduled,
        "results": [],
    }
    summary.update(git_source_identity(repo_root))
    _write_json(summary_path, summary)

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
                scheduled_case_specs,
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
            tpu_smi = Path(summary["toolchain_identity"]["pcie"]["tpu_smi"]["path"])
            summary["board_preflight"] = board_health(args.device_id, tpu_smi)
        _write_json(summary_path, summary)
    except KeyboardInterrupt:
        summary.update({
            "status": "cancelled",
            "failed_phase": "preflight",
            "finished_at": _utc_now(),
        })
        _write_json(summary_path, summary)
        raise
    except Exception as error:
        summary.update({
            "status": "failed",
            "failed_phase": "preflight",
            "error_type": type(error).__name__,
            "error": str(error),
            "finished_at": _utc_now(),
        })
        _write_json(summary_path, summary)
        print(f"STOP preflight: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1

    worker = execution_root / "testing/python/jit/tpukernel_ops_worker.py"
    for chip in chips:
        for case in cases:
            if chip not in case.supported_chips:
                continue
            key = f"{args.runtime_mode}/{chip}/{case.case_id}"
            print(f"RUN {key}", flush=True)
            try:
                if args.runtime_mode == "pcie":
                    assert_source_identity_unchanged(repo_root, summary)
                    assert_toolchain_identity_unchanged(worker_environment_values,
                                                        summary["toolchain_identity"])
                result = _run_one(args, worker_environment_values, output_dir, worker, chip, case)
            except BaseException as error:
                # This path covers runner bookkeeping failures. CModel cleanup
                # is bounded in _run_one; an unreapable PCIe group has already
                # made the device session persistently fail-closed.
                result = {
                    "schema_version": _SCHEMA_VERSION,
                    "status": "failed",
                    "case": case.to_json(),
                    "chip": chip,
                    "programming_model": "tpukernel",
                    "runtime_mode": args.runtime_mode,
                    "started_at": _utc_now(),
                    "elapsed_seconds": 0.0,
                    "timeout_seconds": args.timeout,
                    "timed_out": False,
                    "failure": f"runner error: {type(error).__name__}: {error}",
                }
                case_dir = _case_directory(output_dir, chip, args.runtime_mode, case)
                _write_json(case_dir / "result.json", result)

            result_path = (
                _case_directory(output_dir, chip, args.runtime_mode, case) / "result.json")
            if args.runtime_mode == "pcie" and result.get("launch_attempted"):
                assert args.device_id is not None
                postflight_exception: Optional[BaseException] = None
                try:
                    tpu_smi = Path(summary["toolchain_identity"]["pcie"]["tpu_smi"]["path"])
                    result["board_postflight"] = board_health(
                        args.device_id, tpu_smi, quarantine_on_failure=True)
                except BaseException as health_error:
                    postflight_exception = health_error
                    execution_status = result["status"]
                    execution_failure = result.get("failure")
                    result["execution_status"] = execution_status
                    if execution_failure is not None:
                        result["execution_failure"] = execution_failure
                    result["failed_phase"] = "board-postflight-settle"
                    result["board_postflight_error"] = (
                        f"{type(health_error).__name__}: {health_error}")
                    health_evidence = getattr(health_error, "evidence", None)
                    if isinstance(health_evidence, Mapping):
                        result["board_postflight_failure"] = dict(health_evidence)
                    result["status"] = ("cancelled" if isinstance(health_error, KeyboardInterrupt)
                                        else "failed")
                    postflight_failure = ("board postflight failed: " +
                                          result["board_postflight_error"])
                    result["failure"] = (f"{execution_failure}; {postflight_failure}"
                                         if execution_failure else postflight_failure)
                try:
                    _write_json(result_path, result)
                except Exception as error:
                    diagnostic = (f"result file write failed: {type(error).__name__}: {error}")
                    result["status"] = "failed"
                    result["failure"] = (f"{result['failure']}; {diagnostic}"
                                         if result.get("failure") else diagnostic)
                    result["result_file_error"] = diagnostic
                if result["status"] == "cancelled":
                    summary["results"].append(_summary_case(result, result_path, output_dir))
                    summary["completed_case_count"] += 1
                    summary["cancelled_case_count"] += 1
                    summary["status"] = "cancelled"
                    summary["stopped_after"] = key
                    summary["finished_at"] = _utc_now()
                    _write_json(summary_path, summary)
                    assert postflight_exception is not None
                    raise postflight_exception
            summary["results"].append(_summary_case(result, result_path, output_dir))
            summary["completed_case_count"] += 1
            if result["status"] == "passed":
                summary["passed_case_count"] += 1
                print(f"PASS {key}", flush=True)
                _write_json(summary_path, summary)
                continue

            summary["failed_case_count"] += 1
            summary["status"] = "failed"
            summary["stopped_after"] = key
            summary["finished_at"] = _utc_now()
            _write_json(summary_path, summary)
            print(f"STOP {key}: {result.get('failure')}", file=sys.stderr, flush=True)
            return 1

    try:
        ending_source = git_source_identity(repo_root)
        ending_toolchain = toolchain_identity(worker_environment_values, args.runtime_mode)
    except KeyboardInterrupt:
        summary.update({
            "status": "cancelled",
            "complete": False,
            "failed_phase": "final-identity-check",
            "finished_at": _utc_now(),
        })
        _write_json(summary_path, summary)
        raise
    except Exception as error:
        summary.update({
            "status": "failed",
            "complete": False,
            "failed_phase": "final-identity-check",
            "error_type": type(error).__name__,
            "error": str(error),
            "finished_at": _utc_now(),
        })
        _write_json(summary_path, summary)
        print(
            f"STOP final identity check: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True)
        return 1
    source_fields = ("git_commit", "implementation_worktree_dirty", "source_state_sha256")
    source_changed = any(ending_source.get(field) != summary.get(field) for field in source_fields)
    toolchain_changed = ending_toolchain != summary.get("toolchain_identity")
    if source_changed or toolchain_changed:
        summary.update({
            "status": "failed",
            "complete": False,
            "failed_phase": "final-identity-check",
            "source_changed_during_run": ending_source if source_changed else None,
            "toolchain_changed_during_run": toolchain_changed,
            "finished_at": _utc_now(),
        })
        _write_json(summary_path, summary)
        print(
            "STOP source/toolchain identity changed during matrix execution",
            file=sys.stderr,
            flush=True)
        return 1

    summary["status"] = "passed"
    summary["complete"] = True
    summary["finished_at"] = _utc_now()
    _write_json(summary_path, summary)
    print(f"MATRIX_OK {summary_path}", flush=True)
    return 0


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    chips = _unique_in_order(args.chips, _CHIPS)
    cases = _selected_cases(args)
    if args.list_cases:
        print(
            json.dumps(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "programming_model": "tpukernel",
                    "runtime_mode": args.runtime_mode,
                    "chips": list(chips),
                    "case_count_by_chip": {
                        chip: sum(chip in case.supported_chips for case in cases) for chip in chips
                    },
                    "cases": [case.to_json() for case in cases],
                },
                indent=2,
                sort_keys=True,
                allow_nan=False))
        return 0
    assert args.output_dir is not None
    repo_root = Path(__file__).resolve().parents[3]
    output_dir = args.output_dir.expanduser().resolve()
    if (output_dir / "summary.json").exists():
        raise RuntimeError(f"refusing to overwrite an existing matrix summary in {output_dir}")
    if output_dir.is_dir() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to mix a matrix with non-empty directory {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise RuntimeError(
            f"refusing to reuse an existing matrix output directory {output_dir}") from error

    scratch_dir = Path(tempfile.mkdtemp(prefix=".scratch-", dir=tilelang_temp_dir()))
    try:
        environment = _base_worker_environment(repo_root, args.runtime_mode, args.device_id)
        environment["TMPDIR"] = str(scratch_dir)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        if args.runtime_mode == "pcie":
            from tilelang.jit import TPUInstructionProfiler

            assert args.device_id is not None
            lock_acquired = False
            try:
                with TPUInstructionProfiler.exclusive_pcie_device(args.device_id):
                    lock_acquired = True
                    return _run_matrix(args, repo_root, output_dir, chips, cases, environment)
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
                        "programming_model": "tpukernel",
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
                _write_json(summary_path, failure)
                print(
                    f"PCIe session stopped: {type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True)
                return 1
        return _run_matrix(args, repo_root, output_dir, chips, cases, environment)
    finally:
        if scratch_dir.exists():
            remove_execution_scratch(scratch_dir)


if __name__ == "__main__":
    raise SystemExit(main())
