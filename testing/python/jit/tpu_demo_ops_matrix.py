# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Profile and validate every public TPU demo in isolated fresh processes.

Use separate invocations in this promotion order: BM1690 CModel, SG2260E
CModel, then SG2260E PCIe. Each case gets one compile and one launch under the
shared parent-death/process-group watchdog. The first failure stops the matrix
before another TPU launch.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any, Callable, Mapping, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tpu_demo.cases import (DTYPES, OPERATIONS, TARGET_CONFIGS, DemoCase, build_cases,
                            kernel_variant)

if __package__:
    from .tpu_matrix_common import (
        git_source_identity,
        matrix_target_scope,
        unique_prefixed_json_payload,
        validate_promotion_stages,
    )
else:
    from tpu_matrix_common import (
        git_source_identity,
        matrix_target_scope,
        unique_prefixed_json_payload,
        validate_promotion_stages,
    )

RESULT_PREFIX = "TPU_DEMO_RESULT="
MATRIX_KIND = "tpu_demo_ops"
SCHEMA_VERSION = 1
CMODEL_CONFIGS = TARGET_CONFIGS
PCIE_CONFIGS = tuple(pair for pair in TARGET_CONFIGS if pair[0] == "sg2260e")
_WORKER_TOOL_PATH = os.pathsep.join(("/usr/bin", "/bin"))
_BOARD_IDLE_SETTLE_TIMEOUT_S = 10.0
_BOARD_IDLE_POLL_INTERVAL_S = 0.25
_BOARD_IDLE_REQUIRED_ZERO_SAMPLES = 2


class BoardHealthError(RuntimeError):
    """A bounded board observation failed, with serializable partial evidence."""

    def __init__(self, message: str, evidence: Mapping[str, Any]):
        super().__init__(message)
        self.evidence = dict(evidence)


class BoardHealthCancelled(KeyboardInterrupt):
    """A board observation was interrupted after preserving partial evidence."""

    def __init__(self, message: str, evidence: Mapping[str, Any]):
        super().__init__(message)
        self.evidence = dict(evidence)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-mode", choices=("cmodel", "pcie"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--chip", choices=("bm1690", "sg2260e"))
    parser.add_argument("--programming-model", choices=("tpukernel", "rv"))
    parser.add_argument("--op", choices=OPERATIONS, action="append", dest="operations")
    parser.add_argument("--dtype", choices=DTYPES, action="append", dest="dtypes")
    parser.add_argument(
        "--case",
        choices=tuple(case.case_id for case in build_cases()),
        action="append",
        dest="case_ids")
    parser.add_argument("--device-id", type=int)
    parser.add_argument("--allow-pcie", action="store_true")
    parser.add_argument("--allow-pcie-profile", action="store_true")
    parser.add_argument("--pcie-decoder-python", type=Path)
    parser.add_argument("--pcie-decoder-pythonpath", type=Path, action="append", default=[])
    parser.add_argument("--require-decoded-timing", action="store_true")
    parser.add_argument("--bm-cmodel-summary", type=Path)
    parser.add_argument("--sg-cmodel-summary", type=Path)
    parser.add_argument(
        "--all-pcie-cases",
        action="store_true",
        help="explicitly acknowledge scheduling every selected PCIe case")
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="print the declarative case registry without loading a TPU runtime")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.list_cases:
        if (args.runtime_mode is not None or args.output_dir is not None or
                args.device_id is not None or args.allow_pcie or args.allow_pcie_profile or
                args.all_pcie_cases or args.pcie_decoder_python or args.pcie_decoder_pythonpath or
                args.require_decoded_timing or args.bm_cmodel_summary or args.sg_cmodel_summary):
            raise RuntimeError("--list-cases cannot be combined with execution arguments")
        if args.chip == "bm1690" and args.programming_model == "rv":
            raise RuntimeError("BM1690 does not support the RV programming model")
        return
    if args.runtime_mode is None or args.output_dir is None:
        raise RuntimeError("execution requires --runtime-mode and --output-dir")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("--timeout must be finite and positive")
    if args.runtime_mode == "pcie":
        if not args.allow_pcie or not args.allow_pcie_profile:
            raise RuntimeError("PCIe requires --allow-pcie and --allow-pcie-profile")
        if (isinstance(args.device_id, bool) or not isinstance(args.device_id, int) or
                not 0 <= args.device_id <= 2**31 - 1):
            raise RuntimeError("PCIe requires a non-negative 32-bit --device-id")
        if args.device_id != 0:
            raise RuntimeError("this single-card validation host accepts only --device-id 0")
        if args.chip != "sg2260e":
            raise RuntimeError("this machine requires explicit --chip sg2260e for PCIe cases")
        if not args.case_ids and not args.operations and not args.all_pcie_cases:
            raise RuntimeError("PCIe requires an explicit --case/--op filter or --all-pcie-cases")
        if args.bm_cmodel_summary is None or args.sg_cmodel_summary is None:
            raise RuntimeError(
                "PCIe requires --bm-cmodel-summary and --sg-cmodel-summary promotion evidence")
    elif args.allow_pcie or args.allow_pcie_profile or args.device_id is not None:
        raise RuntimeError("PCIe acknowledgement/device arguments are invalid in CModel mode")
    elif args.all_pcie_cases:
        raise RuntimeError("--all-pcie-cases is valid only in PCIe mode")
    elif args.chip is None:
        raise RuntimeError("CModel promotion requires one explicit --chip per invocation")
    if args.runtime_mode != "pcie" and (args.pcie_decoder_python or args.pcie_decoder_pythonpath or
                                        args.require_decoded_timing or args.bm_cmodel_summary or
                                        args.sg_cmodel_summary):
        raise RuntimeError("PCIe decoder arguments are invalid in CModel mode")


def selected_cases(args: argparse.Namespace) -> tuple[DemoCase, ...]:
    operations = set(args.operations or tuple(case.operation for case in build_cases()))
    dtypes = set(args.dtypes or tuple(case.dtype for case in build_cases()))
    case_ids = set(args.case_ids) if args.case_ids else None
    cases = tuple(case for case in build_cases()
                  if case.operation in operations and case.dtype in dtypes and
                  (case_ids is None or case.case_id in case_ids) and
                  (args.programming_model != "rv" or case.supports_rv))
    if not cases:
        raise RuntimeError("the case filters select no TPU demos")
    return cases


def worker_environment(repo_root: Path, runtime_mode: str,
                       device_id: Optional[int]) -> dict[str, str]:
    ppl_root = os.environ.get("PPL_PROJECT_ROOT")
    if not ppl_root:
        raise RuntimeError("PPL_PROJECT_ROOT must identify the configured PPL 1.7 SDK")
    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    environment = {
        "PPL_PROJECT_ROOT":
            str(Path(ppl_root).expanduser().resolve()),
        # Workers receive an allowlist environment rather than the caller's
        # shell state. Keep the host tool search path deterministic while
        # retaining the binutils required by /usr/bin/cc and /usr/bin/c++
        # (notably collect2's lookup of ld). PCIe's cross-GCC is invoked by
        # absolute path and locates its companion tools from its SDK prefix.
        "PATH":
            _WORKER_TOOL_PATH,
        "PYTHONPATH":
            os.pathsep.join(item for item in (str(repo_root), inherited_pythonpath) if item),
        "PYTHONDONTWRITEBYTECODE":
            "1",
    }
    # These are decoder locations, not target/runtime selectors. Preserve
    # them explicitly so CModel profiling can perform the same optional
    # PerfAI post-processing as PPL's ``--profiling`` flow while all hardware
    # gates continue to be constructed solely by this runner.
    for name in ("PPL_PERFAI_ROOT", "PPL_THIRD_PARTY_PATH"):
        value = os.environ.get(name)
        if value:
            environment[name] = str(Path(value).expanduser().resolve())
    sdk_runtime = (Path(environment["PPL_PROJECT_ROOT"]) /
                   "deps/runtime/tpuv7-runtime/lib").resolve()
    inherited_libraries = []
    for item in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if not item:
            continue
        resolved = Path(item).expanduser().resolve()
        if runtime_mode != "pcie" or resolved != sdk_runtime:
            inherited_libraries.append(str(resolved))
    if runtime_mode == "pcie":
        board_runtime = Path(
            os.environ.get(
                "TILELANG_TPU_PCIE_RUNTIME_PATH",
                "/opt/tpuv7/tpuv7-current/lib",
            )).expanduser().resolve()
        environment["TILELANG_TPU_PCIE_RUNTIME_PATH"] = str(board_runtime)
        library_roots = (str(board_runtime), *inherited_libraries)
    else:
        library_roots = (str(sdk_runtime), *inherited_libraries)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(library_roots))
    if runtime_mode == "pcie":
        assert device_id is not None
        environment.update({
            "TILELANG_TPU_ALLOW_PCIE_LOAD": "1",
            "TILELANG_TPU_ALLOW_PCIE_PROFILE": "1",
            "TILELANG_TPU_DEVICE_ID": str(device_id),
        })
    return environment


def pin_worker_cache_environment(environment: Mapping[str, str],
                                 scratch_dir: Path) -> dict[str, str]:
    """Pin all TileLang-generated caches below one disposable worker scratch."""

    resolved_scratch = scratch_dir.expanduser().resolve()
    resolved_scratch.mkdir(parents=True, exist_ok=True)
    if not resolved_scratch.is_dir():
        raise RuntimeError(f"worker scratch path is not a directory: {resolved_scratch}")
    pinned = dict(environment)
    pinned["TMPDIR"] = str(resolved_scratch)
    # This explicit setting makes workers independent of HOME and prevents a
    # read-only execution snapshot from falling back to a source-local cache.
    pinned["TILELANG_CACHE_DIR"] = str(resolved_scratch / "tilelang-cache")
    return pinned


def toolchain_identity(environment: Mapping[str, str], runtime_mode: str) -> dict[str, Any]:
    """Hash every compiler, SDK, and runtime component used by this matrix."""
    from tilelang.jit.adapter.tpu_toolchain_identity import (
        capture_tpu_toolchain_identity,)

    resolved_environment = dict(os.environ)
    resolved_environment.update(environment)
    return capture_tpu_toolchain_identity(
        resolved_environment["PPL_PROJECT_ROOT"],
        runtime_mode,
        environment=resolved_environment,
    )


_COMMON_TOOLCHAIN_FIELDS = (
    "schema_version",
    "hash_algorithm",
    "ppl_project_root",
    "compiler_runtime",
    "ppl_common",
    "cmodel",
    "chips",
)


def assert_toolchain_identity_unchanged(environment: Mapping[str, str],
                                        expected: Mapping[str, Any]) -> None:
    current = toolchain_identity(environment, "pcie")
    if current != expected:
        raise RuntimeError(
            "content-addressed TPU toolchain identity changed; refusing the next PCIe launch")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_git_archive(
    repository: Path,
    commit: str,
    destination: Path,
    paths: tuple[str, ...],
) -> None:
    command = ["git", "archive", "--format=tar", commit, *paths]
    try:
        completed = subprocess.run(
            command,
            cwd=repository,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"cannot materialize promoted Git source: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git archive failed for {repository} at {commit}: {detail}")

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:") as archive:
        for member in archive.getmembers():
            target = (root / member.name).resolve()
            if os.path.commonpath((str(root), str(target))) != str(root):
                raise RuntimeError(f"Git archive contains an escaping path: {member.name}")
            if member.issym():
                linked = (target.parent / member.linkname).resolve()
                if os.path.commonpath((str(root), str(linked))) != str(root):
                    raise RuntimeError(f"Git archive contains an escaping symlink: {member.name}")
            elif member.islnk():
                linked = (root / member.linkname).resolve()
                if os.path.commonpath((str(root), str(linked))) != str(root):
                    raise RuntimeError(f"Git archive contains an escaping hardlink: {member.name}")
        archive.extractall(root)


def materialize_execution_snapshot(repo_root: Path, destination: Path,
                                   commit: str) -> dict[str, str]:
    """Create a private source snapshot for promoted PCIe workers.

    Per-launch digest checks stop the matrix when the shared checkout changes;
    workers compile from this exact Git archive as the stronger protection
    against a check/compile race. The TVM Python submodule is archived at the
    gitlink recorded by the same parent commit.
    """

    if destination.exists():
        raise RuntimeError(f"execution snapshot destination already exists: {destination}")
    _extract_git_archive(
        repo_root,
        commit,
        destination,
        ("VERSION", "tilelang", "tpu_demo", "testing/python/jit", "src"),
    )
    tree = subprocess.run(
        ["git", "ls-tree", commit, "--", "3rdparty/tvm"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
        check=False,
    )
    fields = tree.stdout.strip().split()
    if tree.returncode != 0 or len(fields) < 4 or fields[0] != "160000" or fields[1] != "commit":
        raise RuntimeError("promoted commit has no exact 3rdparty/tvm gitlink")
    tvm_commit = fields[2]
    tvm_repository = repo_root / "3rdparty/tvm"
    observed_tvm = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tvm_repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
        check=False,
    )
    if observed_tvm.returncode != 0 or observed_tvm.stdout.strip() != tvm_commit:
        raise RuntimeError(f"TVM submodule does not match promoted gitlink {tvm_commit}")
    _extract_git_archive(tvm_repository, tvm_commit, destination / "3rdparty/tvm", ("python",))
    _make_tree_read_only(destination)
    return {
        "kind": "git-archive",
        "git_commit": commit,
        "tvm_git_commit": tvm_commit,
        "read_only": True,
    }


def _make_tree_read_only(root: Path) -> None:
    """Remove write bits without following archive symlinks."""

    for current, directories, filenames in os.walk(root, topdown=False, followlinks=False):
        current_path = Path(current)
        for name in filenames:
            path = current_path / name
            metadata = path.lstat()
            if not stat.S_ISLNK(metadata.st_mode):
                path.chmod(stat.S_IMODE(metadata.st_mode) & ~0o222)
        for name in directories:
            path = current_path / name
            metadata = path.lstat()
            if not stat.S_ISLNK(metadata.st_mode):
                path.chmod(stat.S_IMODE(metadata.st_mode) & ~0o222)
    metadata = root.lstat()
    root.chmod(stat.S_IMODE(metadata.st_mode) & ~0o222)


def remove_execution_scratch(path: Path) -> None:
    """Remove a matrix scratch tree, restoring owner access when needed."""

    for current, directories, _filenames in os.walk(path, topdown=True, followlinks=False):
        current_path = Path(current)
        metadata = current_path.lstat()
        current_path.chmod(stat.S_IMODE(metadata.st_mode) | stat.S_IRWXU)
        for name in directories:
            child = current_path / name
            child_metadata = child.lstat()
            if not stat.S_ISLNK(child_metadata.st_mode):
                child.chmod(stat.S_IMODE(child_metadata.st_mode) | stat.S_IRWXU)

    def make_writable_and_retry(function, raw_path, _error_info):
        target = Path(raw_path)
        metadata = target.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            target.chmod(stat.S_IMODE(metadata.st_mode) | stat.S_IRWXU)
        elif not stat.S_ISLNK(metadata.st_mode):
            target.chmod(stat.S_IMODE(metadata.st_mode) | stat.S_IWUSR)
        function(raw_path)

    shutil.rmtree(path, onerror=make_writable_and_retry)


def _resolved_identity_file(toolchain: Mapping[str, Any], key: str) -> Path:
    """Resolve one native compiler-runtime file from a captured identity."""

    try:
        raw_path = toolchain["compiler_runtime"][key]["resolved_path"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"toolchain identity is missing compiler_runtime.{key}.resolved_path") from error
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError(
            f"toolchain identity has an invalid compiler_runtime.{key}.resolved_path")
    path = Path(raw_path)
    if not path.is_absolute():
        raise RuntimeError(
            f"toolchain identity compiler_runtime.{key}.resolved_path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"captured compiler runtime no longer exists: {path}") from error
    if not resolved.is_file() or resolved != path:
        raise RuntimeError(f"captured compiler runtime is not a canonical regular file: {path}")
    return resolved


def pin_native_worker_libraries(
    environment: Mapping[str, str],
    toolchain_identity: Mapping[str, Any],
) -> dict[str, str]:
    """Route a fresh worker to the exact TileLang/TVM DSOs just identified."""

    pinned = dict(environment)
    tilelang_library = _resolved_identity_file(toolchain_identity, "tilelang_library")
    tvm_library = _resolved_identity_file(toolchain_identity, "tvm_library")
    tilelang_library_root = str(tilelang_library.parent)
    tvm_library_root = str(tvm_library.parent)
    pinned["TILELANG_LIBRARY_PATH"] = tilelang_library_root
    pinned["TVM_LIBRARY_PATH"] = tvm_library_root
    inherited_libraries = tuple(
        item for item in pinned.get("LD_LIBRARY_PATH", "").split(os.pathsep) if item)
    pinned["LD_LIBRARY_PATH"] = os.pathsep.join(
        dict.fromkeys((
            tilelang_library_root,
            tvm_library_root,
            *inherited_libraries,
        )))
    return pinned


def pin_worker_environment(environment: Mapping[str, str], *, repo_root: Path, snapshot_root: Path,
                           toolchain_identity: Mapping[str, Any]) -> dict[str, str]:
    """Pin promoted workers to the source snapshot and captured native DSOs."""

    pinned = pin_native_worker_libraries(environment, toolchain_identity)
    inherited = []
    for raw_path in environment.get("PYTHONPATH", "").split(os.pathsep):
        if not raw_path:
            continue
        path = Path(raw_path).expanduser().resolve()
        try:
            inside_checkout = os.path.commonpath((str(repo_root), str(path))) == str(repo_root)
        except ValueError:
            inside_checkout = False
        if not inside_checkout:
            inherited.append(str(path))
    tvm_python = snapshot_root / "3rdparty/tvm/python"
    pinned["PYTHONPATH"] = os.pathsep.join((str(snapshot_root), str(tvm_python), *inherited))
    pinned["TVM_IMPORT_PYTHON_PATH"] = str(tvm_python)
    pinned["TL_TEMPLATE_PATH"] = str(snapshot_root / "src")
    return pinned


def assert_source_identity_unchanged(repo_root: Path, expected: Mapping[str,
                                                                        Any]) -> dict[str, Any]:
    current = git_source_identity(repo_root)
    required = ("git_commit", "source_state_sha256")
    if current.get("implementation_worktree_dirty") is not False or any(
            current.get(field) != expected.get(field) for field in required):
        raise RuntimeError("promoted source identity changed; refusing the next PCIe launch")
    return current


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    try:
        with pending.open("w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def validate_board_snapshot(payload: Any,
                            device_id: int,
                            *,
                            require_idle: bool = True) -> dict[str, Any]:
    """Validate the exact single-card topology used to map logical device 0."""

    if device_id != 0:
        raise RuntimeError(
            "this single-card host can prove logical-to-physical identity only for device 0")
    if not isinstance(payload, dict):
        raise RuntimeError("tpu-smi JSON root is not an object")
    card_num = payload.get("card_num")
    chip_num = payload.get("chip_num")
    if (isinstance(card_num, bool) or not isinstance(card_num, int) or card_num != 1 or
            isinstance(chip_num, bool) or not isinstance(chip_num, int) or chip_num != 1):
        raise RuntimeError("PCIe preflight requires exactly one visible card containing one chip")
    card_key = f"card{device_id}"
    card = payload.get(card_key)
    if not isinstance(card, dict):
        raise RuntimeError(f"tpu-smi JSON has no selected {card_key}")
    card_idx = card.get("card_idx")
    if isinstance(card_idx, bool) or not isinstance(card_idx, int) or card_idx != device_id:
        raise RuntimeError(f"tpu-smi {card_key} has inconsistent card_idx={card_idx!r}")
    chip_count = card.get("chip_num_of_card")
    if isinstance(chip_count, bool) or not isinstance(chip_count, int) or chip_count != 1:
        raise RuntimeError(f"tpu-smi reported invalid chip count for {card_key}")
    chips = []
    for chip_index in range(chip_count):
        chip = card.get(f"chip{chip_index}")
        if not isinstance(chip, dict) or chip.get("status") != "Active":
            observed = None if not isinstance(chip, dict) else chip.get("status")
            raise RuntimeError(
                f"tpu-smi requires exact Active status for {card_key}/chip{chip_index}; "
                f"got {observed!r}")
        observed_index = chip.get("chip_index_of_card")
        if (isinstance(observed_index, bool) or not isinstance(observed_index, int) or
                observed_index != chip_index):
            raise RuntimeError(f"tpu-smi {card_key}/chip{chip_index} has inconsistent chip index")
        tpu_util = chip.get("tpu_util")
        if (not isinstance(tpu_util, str) or
                re.fullmatch(r"(?:0|[1-9][0-9]?|100)%", tpu_util) is None):
            raise RuntimeError(f"{card_key}/chip{chip_index} is not idle: "
                               f"tpu_util={tpu_util!r}")
        if require_idle and tpu_util != "0%":
            raise RuntimeError(f"{card_key}/chip{chip_index} is not idle: "
                               f"tpu_util={tpu_util!r}")
        chips.append(chip)
    return {
        "device_id": device_id,
        "status": "Active",
        "card_key": card_key,
        "chip_count": chip_count,
        "chips": chips,
    }


def _safe_json_evidence(payload: Any) -> Any:
    """Detach a JSON-safe probe payload without invoking arbitrary repr hooks."""

    try:
        return json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    except Exception:
        return {
            "evidence_unavailable": "payload-is-not-json-serializable",
            "payload_type": type(payload).__name__,
        }


def _wait_for_idle_board_snapshot(
        probe: Callable[[float], Any],
        device_id: int,
        *,
        timeout_s: float = _BOARD_IDLE_SETTLE_TIMEOUT_S,
        poll_interval_s: float = _BOARD_IDLE_POLL_INTERVAL_S,
        required_zero_samples: int = _BOARD_IDLE_REQUIRED_ZERO_SAMPLES) -> dict[str, Any]:
    """Wait for interval-separated idle samples within one monotonic deadline."""

    if not math.isfinite(timeout_s) or timeout_s < 0:
        raise ValueError("board idle-settle timeout must be finite and non-negative")
    if not math.isfinite(poll_interval_s) or poll_interval_s <= 0:
        raise ValueError("board idle-settle poll interval must be finite and positive")
    if (isinstance(required_zero_samples, bool) or not isinstance(required_zero_samples, int) or
            required_zero_samples < 2):
        raise ValueError("board idle-settle requires at least two zero-utilization samples")
    started = time.monotonic()
    deadline = started + timeout_s
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "status": "settling",
        "device_id": device_id,
        "timeout_s": timeout_s,
        "poll_interval_s": poll_interval_s,
        "required_consecutive_zero_samples": required_zero_samples,
        "started_at": utc_now(),
        "finished_at": None,
        "settle_seconds": None,
        "sample_count": 0,
        "consecutive_zero_samples": 0,
        "initial_tpu_util": None,
        "last_tpu_util": None,
        "max_observed_util_percent": None,
        "samples": [],
    }
    zero_streak = 0

    def record_sample(*,
                      probe_started: float,
                      probe_finished: float,
                      payload_available: bool,
                      raw_payload: Any = None,
                      snapshot: Optional[Mapping[str, Any]] = None,
                      probe_error: Optional[BaseException] = None,
                      validation_error: Optional[BaseException] = None,
                      utilization: Optional[list[str]] = None) -> dict[str, Any]:
        payload_evidence = (_safe_json_evidence(raw_payload) if payload_available else None)
        observed_at = time.monotonic()
        sample = {
            "sample_index": len(evidence["samples"]),
            "observed_at": utc_now(),
            "elapsed_s": round(observed_at - started, 6),
            "probe_duration_s": round(probe_finished - probe_started, 6),
            "payload_available": payload_available,
            "payload": payload_evidence,
            "probe_error_type": (type(probe_error).__name__ if probe_error is not None else None),
            "probe_error_message": (str(probe_error) if probe_error is not None else None),
            "validation_error_type":
                (type(validation_error).__name__ if validation_error is not None else None),
            "validation_error_message":
                (str(validation_error) if validation_error is not None else None),
            "status": None if snapshot is None else snapshot.get("status"),
            "tpu_util": utilization,
            "consecutive_zero_samples": zero_streak,
            "within_deadline": observed_at <= deadline,
        }
        evidence["samples"].append(sample)
        evidence["sample_count"] = len(evidence["samples"])
        return sample

    def fail(reason: str, message: str, *, cause: Optional[BaseException] = None):
        evidence.update({
            "status": "failed",
            "failure_reason": reason,
            "finished_at": utc_now(),
            "settle_seconds": round(time.monotonic() - started, 6),
        })
        error = BoardHealthError(message, evidence)
        if cause is None:
            raise error
        raise error from cause

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            fail(
                "idle-timeout",
                f"device {device_id} did not provide {required_zero_samples} consecutive "
                f"idle samples within {timeout_s:g}s; "
                f"last_tpu_util={evidence['last_tpu_util']!r}",
            )
        probe_started = time.monotonic()
        try:
            payload = probe(remaining)
        except KeyboardInterrupt as error:
            probe_finished = time.monotonic()
            record_sample(
                probe_started=probe_started,
                probe_finished=probe_finished,
                payload_available=False,
                probe_error=error,
            )
            evidence.update({
                "status": "cancelled",
                "failure_reason": "observation-cancelled",
                "finished_at": utc_now(),
                "settle_seconds": round(time.monotonic() - started, 6),
            })
            raise BoardHealthCancelled(f"device {device_id} board observation was interrupted",
                                       evidence) from error
        except Exception as error:
            probe_finished = time.monotonic()
            record_sample(
                probe_started=probe_started,
                probe_finished=probe_finished,
                payload_available=False,
                probe_error=error,
            )
            fail(
                "probe-or-snapshot-invalid",
                f"device {device_id} board observation failed: "
                f"{type(error).__name__}: {error}",
                cause=error,
            )
        probe_finished = time.monotonic()
        try:
            snapshot = validate_board_snapshot(payload, device_id, require_idle=False)
        except Exception as error:
            record_sample(
                probe_started=probe_started,
                probe_finished=probe_finished,
                payload_available=True,
                raw_payload=payload,
                validation_error=error,
            )
            fail(
                "probe-or-snapshot-invalid",
                f"device {device_id} board observation failed: "
                f"{type(error).__name__}: {error}",
                cause=error,
            )
        utilization = [chip["tpu_util"] for chip in snapshot["chips"]]
        utilization_percent = [int(value[:-1]) for value in utilization]
        zero_streak = zero_streak + 1 if all(value == 0 for value in utilization_percent) else 0
        sample = record_sample(
            probe_started=probe_started,
            probe_finished=probe_finished,
            payload_available=True,
            raw_payload=payload,
            snapshot=snapshot,
            utilization=utilization,
        )
        evidence["consecutive_zero_samples"] = zero_streak
        evidence["last_tpu_util"] = utilization
        if evidence["initial_tpu_util"] is None:
            evidence["initial_tpu_util"] = utilization
        maximum = max(utilization_percent)
        previous_maximum = evidence["max_observed_util_percent"]
        evidence["max_observed_util_percent"] = (
            maximum if previous_maximum is None else max(previous_maximum, maximum))
        if not sample["within_deadline"]:
            fail(
                "idle-timeout",
                f"device {device_id} board observation returned after the "
                f"{timeout_s:g}s idle-settle deadline; last_tpu_util={utilization!r}",
            )
        if zero_streak >= required_zero_samples:
            evidence.update({
                "status": "passed",
                "finished_at": utc_now(),
                "settle_seconds": sample["elapsed_s"],
            })
            return {
                **snapshot,
                "idle_probe_count": evidence["sample_count"],
                "idle_settle_seconds": evidence["settle_seconds"],
                "initial_tpu_util": evidence["initial_tpu_util"],
                "idle_settle": evidence,
            }
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            fail(
                "idle-timeout",
                f"device {device_id} did not provide {required_zero_samples} consecutive "
                f"idle samples within {timeout_s:g}s; initial_tpu_util="
                f"{evidence['initial_tpu_util']!r}, last_tpu_util={utilization!r}",
            )
        time.sleep(min(poll_interval_s, remaining))


def _pcie_driver_identity() -> list[dict[str, str]]:
    """Return the unique physical TPUv7 endpoint owned by the SG host driver."""

    driver_root = Path("/sys/bus/pci/drivers/sg-host-drv")
    devices = []
    if driver_root.is_dir():
        for entry in sorted(driver_root.iterdir()):
            if not entry.is_symlink() or ":" not in entry.name:
                continue
            vendor_path = entry / "vendor"
            device_path = entry / "device"
            if vendor_path.is_file() and device_path.is_file():
                devices.append({
                    "bdf": entry.name,
                    "vendor": vendor_path.read_text(encoding="ascii").strip().lower(),
                    "device": device_path.read_text(encoding="ascii").strip().lower(),
                    "driver": "sg-host-drv",
                })
    matching_devices = [
        device for device in devices
        if device["vendor"] == "0x1f1c" and device["device"] == "0x1690"
    ]
    if len(matching_devices) != 1:
        raise RuntimeError("PCIe preflight requires exactly one SG host-driver TPUv7 device; "
                           f"found {len(matching_devices)}")
    return matching_devices


def board_health(device_id: int,
                 tpu_smi: Optional[Path] = None,
                 *,
                 quarantine_on_failure: bool = False) -> dict[str, Any]:
    """Prove one card is stably idle while holding its complete session lock.

    A preflight observation is read-only and normally leaves no persistent
    marker when the board is unavailable.  A postflight caller sets
    ``quarantine_on_failure`` because failure to prove quiescence after a TPU
    dispatch must keep the current session fail-closed.
    """

    from tilelang.jit import TPUInstructionProfiler

    if not isinstance(quarantine_on_failure, bool):
        raise ValueError("quarantine_on_failure must be a boolean")

    runtime_root = Path(
        os.environ.get("TILELANG_TPU_PCIE_RUNTIME_PATH",
                       "/opt/tpuv7/tpuv7-current/lib")).expanduser().resolve()
    runtime_tool = runtime_root.parent / "bin" / "tpu-smi"
    current = Path("/opt/tpuv7/tpuv7-current/bin/tpu-smi")
    path_tool = shutil.which("tpu-smi")
    bundled = Path("/opt/tpuv7/tpuv7-runtime_1.9.3/bin/tpu-smi")
    legacy = Path("/opt/tpuv7/tpuv7-runtime/bin/tpu-smi")
    candidates = ((tpu_smi,) if tpu_smi is not None else
                  (runtime_tool, current, Path(path_tool) if path_tool else None, bundled, legacy))
    executable = next(
        (str(path.expanduser().absolute())
         for path in candidates
         if path is not None and path.is_file() and os.access(path, os.X_OK)),
        None,
    )
    if executable is None:
        raise RuntimeError("PCIe preflight cannot find tpu-smi")
    sdk_runtime = None
    ppl_root = os.environ.get("PPL_PROJECT_ROOT")
    if ppl_root:
        sdk_runtime = (Path(ppl_root).expanduser().resolve() /
                       "deps/runtime/tpuv7-runtime/lib").resolve()
    inherited_libraries = []
    for item in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if not item:
            continue
        resolved = Path(item).expanduser().resolve()
        if sdk_runtime is None or resolved != sdk_runtime:
            inherited_libraries.append(str(resolved))
    probe_environment = {
        "LD_LIBRARY_PATH": os.pathsep.join(
            dict.fromkeys((str(runtime_root), *inherited_libraries)))
    }

    def probe(remaining_s: float):
        completed = TPUInstructionProfiler.run_supervised_pcie_probe(
            device_id,
            [executable, f"--dev={device_id}", "--noloop", "--json_format"],
            cwd=_REPO_ROOT,
            environment=probe_environment,
            timeout_s=remaining_s,
        )
        if completed.timed_out:
            raise RuntimeError("tpu-smi board observation exceeded its remaining deadline")
        if completed.left_live_descendant:
            raise RuntimeError(
                "tpu-smi preflight left a descendant process; the supervised group was stopped")
        if completed.returncode != 0:
            raise RuntimeError(f"tpu-smi preflight failed with status {completed.returncode}: "
                               f"{completed.stderr.strip()}")
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"tpu-smi returned invalid JSON: {error}") from error

    # tpu-smi can report the just-finished synchronized launch for one sample.
    # Hold the same device lock for every sample and the physical-device check.
    # A persistently busy or non-Active board cannot race with another TileLang
    # launch and is never accepted on the strength of one transient 0% sample.
    with TPUInstructionProfiler.exclusive_pcie_device(device_id):
        try:
            snapshot = _wait_for_idle_board_snapshot(
                probe,
                device_id,
                timeout_s=_BOARD_IDLE_SETTLE_TIMEOUT_S,
                poll_interval_s=_BOARD_IDLE_POLL_INTERVAL_S,
                required_zero_samples=_BOARD_IDLE_REQUIRED_ZERO_SAMPLES,
            )
            matching_devices = _pcie_driver_identity()
            return {
                **snapshot,
                "tpu_smi": executable,
                "pcie_driver_candidates": matching_devices,
            }
        except BaseException as error:
            if quarantine_on_failure:
                reason = ("postflight-did-not-settle" if
                          (isinstance(error, BoardHealthError) and
                           error.evidence.get("failure_reason") == "idle-timeout") else
                          "postflight-observation-incomplete")
                # This public operation first marks the active session unsafe,
                # so even failure to create the secondary marker retains the
                # crash-persistent session marker on context exit.
                quarantine = TPUInstructionProfiler.fail_closed_pcie_device(
                    device_id, reason=reason)
                if isinstance(error, (BoardHealthError, BoardHealthCancelled)):
                    quarantine_evidence: dict[str, Any] = {
                        "status": "quarantined",
                        "requested_reason": reason,
                        "marker": str(quarantine),
                    }
                    try:
                        persisted = json.loads(quarantine.read_text(encoding="utf-8"))
                    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                        persisted = None
                    if isinstance(persisted, dict):
                        quarantine_evidence.update({
                            field: persisted[field]
                            for field in ("reason", "process_group", "process_group_known")
                            if field in persisted
                        })
                    error.evidence["quarantine"] = quarantine_evidence
            raise


def worker_payload(stdout_path: Path) -> dict[str, Any]:
    payload = unique_prefixed_json_payload(stdout_path, RESULT_PREFIX, "demo worker")
    if payload.get("status") not in ("passed", "failed"):
        raise RuntimeError("demo worker emitted an invalid result payload")
    return payload


def numeric_payload(stdout_path: Path) -> dict[str, Any]:
    payload = worker_payload(stdout_path)
    if payload.get("status") != "passed":
        raise RuntimeError("demo worker emitted a non-passing result payload")
    return payload


def validate_numeric_identity(payload: Mapping[str, Any], *, chip: str, programming_model: str,
                              runtime_mode: str, case: DemoCase) -> None:
    expected = {
        "operation": case.operation,
        "dtype": case.dtype,
        "chip": chip,
        "programming_model": programming_model,
        "runtime_mode": runtime_mode,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise RuntimeError(f"demo worker result identity mismatch for {field}: "
                               f"expected {value!r}, got {payload.get(field)!r}")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("passed") is not True:
        raise RuntimeError("demo worker result lacks metrics.passed=true")
    if metrics.get("finite") is not True:
        raise RuntimeError("demo worker result lacks metrics.finite=true")
    parameters = payload.get("parameters")
    expected_kernel = kernel_variant(case.operation, case.dtype, programming_model)
    if (not isinstance(parameters, dict) or parameters.get("kernel_variant") != expected_kernel):
        observed = None if not isinstance(parameters, dict) else parameters.get("kernel_variant")
        raise RuntimeError("demo worker result kernel variant mismatch: "
                           f"expected {expected_kernel!r}, got {observed!r}")
    if case.variant != "default" and (not isinstance(parameters, dict) or
                                      parameters.get("variant") != case.variant):
        raise RuntimeError("demo worker result does not identify the scheduled variant")
    if case.operation == "flashattn" and (not isinstance(parameters, dict) or
                                          parameters.get("is_causal") is not case.is_causal):
        raise RuntimeError("demo worker result does not identify the scheduled causal mode")


def selected_configurations(args: argparse.Namespace) -> tuple[tuple[str, str], ...]:
    source = PCIE_CONFIGS if args.runtime_mode == "pcie" else CMODEL_CONFIGS
    configurations = tuple(
        pair for pair in source if (args.chip is None or pair[0] == args.chip) and
        (args.programming_model is None or pair[1] == args.programming_model))
    if not configurations:
        raise RuntimeError("the requested chip/programming-model pair is not valid")
    return configurations


def scheduled_cases(
    configurations: tuple[tuple[str, str], ...],
    cases: tuple[DemoCase, ...],
) -> tuple[tuple[str, str, DemoCase], ...]:
    scheduled = tuple((chip, programming_model, case)
                      for chip, programming_model in configurations
                      for case in cases
                      if programming_model != "rv" or case.supports_rv)
    if not scheduled:
        raise RuntimeError("no selected demo is supported by the requested backend")
    return scheduled


def validate_pcie_promotion(
    repo_root: Path,
    args: argparse.Namespace,
    scheduled: tuple[tuple[str, str, DemoCase], ...],
    current_toolchain: Mapping[str, Any],
    pcie_started_at: str,
) -> dict[str, Any]:
    assert args.bm_cmodel_summary is not None and args.sg_cmodel_summary is not None
    if (current_toolchain.get("runtime_mode") != "pcie" or
            not isinstance(current_toolchain.get("pcie"), dict)):
        raise RuntimeError("PCIe promotion requires a complete PCIe toolchain identity")
    current = git_source_identity(repo_root)
    if current.get("implementation_worktree_dirty") is not False:
        raise RuntimeError(
            "PCIe promotion requires the current implementation worktree to be clean")
    current_commit = current.get("git_commit")
    if not isinstance(current_commit, str) or not current_commit:
        raise RuntimeError("PCIe promotion requires a verifiable Git commit")
    current_state = current.get("source_state_sha256")
    if not isinstance(current_state, str) or not current_state:
        raise RuntimeError("PCIe promotion requires a verifiable source-state digest")

    bm, sg = validate_promotion_stages(
        args.bm_cmodel_summary,
        args.sg_cmodel_summary,
        matrix_kind=MATRIX_KIND,
        schema_version=SCHEMA_VERSION,
        bm_allowed_scope=(("bm1690", "tpukernel"),),
        sg_allowed_scope=(("sg2260e", "tpukernel"), ("sg2260e", "rv")),
        pcie_started_at=pcie_started_at,
    )

    for label, payload in (("BM1690", bm), ("SG2260E", sg)):
        if payload.get("git_commit") != current_commit:
            raise RuntimeError(
                f"{label} promotion summary commit does not match current {current_commit}")
        if payload.get("source_state_sha256") != current_state:
            raise RuntimeError(
                f"{label} promotion summary source state does not match current worktree")
        observed_toolchain = payload.get("toolchain_identity")
        if not isinstance(observed_toolchain, dict):
            raise RuntimeError(f"{label} promotion summary has no toolchain identity")
        if observed_toolchain.get("runtime_mode") != "cmodel":
            raise RuntimeError(f"{label} promotion summary has the wrong toolchain runtime mode")
        for field in _COMMON_TOOLCHAIN_FIELDS:
            if observed_toolchain.get(field) != current_toolchain.get(field):
                raise RuntimeError(f"{label} promotion summary content identity for {field} "
                                   "does not match current toolchain")
    if bm.get("git_commit") != sg.get("git_commit"):
        raise RuntimeError("BM1690 and SG2260E promotion summaries use different commits")

    def passed_results(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        results = payload.get("results")
        if not isinstance(results, list):
            raise RuntimeError("promotion summary has no result list")
        passed: dict[str, Mapping[str, Any]] = {}
        seen = set()
        for result in results:
            if not isinstance(result, dict) or not isinstance(result.get("key"), str):
                raise RuntimeError("promotion summary contains a result without an exact key")
            key = result["key"]
            if key in seen:
                raise RuntimeError(f"promotion summary contains a duplicate result key: {key}")
            seen.add(key)
            if (result.get("status") == "passed" and isinstance(result.get("numeric"), dict) and
                    result["numeric"].get("status") == "passed"):
                passed[key] = result
        return passed

    bm_passed = passed_results(bm)
    sg_passed = passed_results(sg)
    missing = []
    for _chip, programming_model, case in scheduled:
        bm_key = f"cmodel/bm1690/tpukernel/{case.case_id}"
        sg_key = f"cmodel/sg2260e/{programming_model}/{case.case_id}"
        bm_result = bm_passed.get(bm_key)
        sg_result = sg_passed.get(sg_key)
        if bm_result is None:
            missing.append(bm_key)
        else:
            validate_numeric_identity(
                bm_result["numeric"],
                chip="bm1690",
                programming_model="tpukernel",
                runtime_mode="cmodel",
                case=case)
            if not isinstance(bm_result.get("raw_instruction_count"),
                              int) or (bm_result["raw_instruction_count"] <= 0):
                raise RuntimeError(f"promotion result has no raw instructions: {bm_key}")
        if sg_result is None:
            missing.append(sg_key)
        else:
            validate_numeric_identity(
                sg_result["numeric"],
                chip="sg2260e",
                programming_model=programming_model,
                runtime_mode="cmodel",
                case=case)
            if not isinstance(sg_result.get("raw_instruction_count"),
                              int) or (sg_result["raw_instruction_count"] <= 0):
                raise RuntimeError(f"promotion result has no raw instructions: {sg_key}")
    if missing:
        raise RuntimeError("PCIe promotion evidence is missing passing cases: " +
                           ", ".join(sorted(set(missing))))
    return {
        "git_commit": current_commit,
        "source_state_sha256": current_state,
        "bm1690_summary": bm["_resolved_path"],
        "bm1690_summary_sha256": bm["_sha256"],
        "sg2260e_summary": sg["_resolved_path"],
        "sg2260e_summary_sha256": sg["_sha256"],
        "validated_case_count": len(scheduled),
    }


def run_matrix(
    args: argparse.Namespace,
    repo_root: Path,
    output_dir: Path,
    scheduled: tuple[tuple[str, str, DemoCase], ...],
    environment: Mapping[str, str],
) -> int:
    summary_path = output_dir / "summary.json"
    execution_root = repo_root
    worker_environment_values = dict(environment)
    summary: dict[str, Any] = {
        "schema_version":
            SCHEMA_VERSION,
        "matrix_kind":
            MATRIX_KIND,
        "status":
            "running",
        "complete":
            False,
        "runtime_mode":
            args.runtime_mode,
        "acceptance":
            ("numeric-raw-and-decoded-timing" if args.require_decoded_timing else "numeric-and-raw"
            ),
        "decoded_timing_required":
            args.require_decoded_timing,
        "started_at":
            utc_now(),
        "finished_at":
            None,
        "scheduled_case_count":
            len(scheduled),
        "completed_case_count":
            0,
        "passed_case_count":
            0,
        "failed_case_count":
            0,
        "cancelled_case_count":
            0,
        "scheduled": [{
            "chip": chip,
            "programming_model": programming_model,
            "case": case.to_json(),
        } for chip, programming_model, case in scheduled],
        "target_scope":
            matrix_target_scope(
                (chip, programming_model) for chip, programming_model, _case in scheduled),
        "results": [],
    }
    summary.update(git_source_identity(repo_root))
    write_json(summary_path, summary)

    try:
        from tilelang.jit import TPUInstructionProfiler, TPUProfilingConfig
        if __package__:
            from .tpu_profile_matrix_common import (
                profile_report_summary,
                validate_profile_report,
            )
        else:
            from tpu_profile_matrix_common import (
                profile_report_summary,
                validate_profile_report,
            )

        summary["toolchain_identity"] = toolchain_identity(worker_environment_values,
                                                           args.runtime_mode)
        worker_environment_values = pin_native_worker_libraries(worker_environment_values,
                                                                summary["toolchain_identity"])

        if args.runtime_mode == "pcie":
            assert args.device_id is not None
            summary["promotion_evidence"] = validate_pcie_promotion(
                repo_root,
                args,
                scheduled,
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
            first_chip, first_model, _first_case = scheduled[0]
            preflight_config = TPUProfilingConfig(
                chip=first_chip,
                programming_model=first_model,
                runtime_mode="pcie",
                output_dir=output_dir,
                label="decoder-preflight",
                timeout_s=args.timeout,
                postprocess=True,
                pcie_decoder_python=args.pcie_decoder_python,
                pcie_decoder_pythonpath=tuple(args.pcie_decoder_pythonpath),
            )
            if args.require_decoded_timing:
                summary["decoder_preflight"] = dict(
                    TPUInstructionProfiler(preflight_config).preflight_pcie_decoder(
                        environment=worker_environment_values))
            tpu_smi = Path(summary["toolchain_identity"]["pcie"]["tpu_smi"]["path"])
            summary["board_preflight"] = board_health(args.device_id, tpu_smi)
        write_json(summary_path, summary)
    except Exception as error:
        summary.update({
            "status": "failed",
            "complete": False,
            "failed_phase": "preflight",
            "error_type": type(error).__name__,
            "error": str(error),
            "finished_at": utc_now(),
        })
        write_json(summary_path, summary)
        print(f"PREFLIGHT_STOP: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1

    worker = execution_root / "testing/python/jit/tpu_demo_ops_worker.py"
    decoder_config = {
        "pcie_decoder_python": args.pcie_decoder_python,
        "pcie_decoder_pythonpath": tuple(args.pcie_decoder_pythonpath),
    }
    for chip, programming_model, case in scheduled:
        key = f"{args.runtime_mode}/{chip}/{programming_model}/{case.case_id}"
        print(f"RUN {key}", flush=True)
        command = [sys.executable, str(worker), "--case", case.case_id]
        existing_artifacts = set(output_dir.iterdir())
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
                label=f"{chip}-{programming_model}-{case.case_id}".replace(".", "-"),
                timeout_s=args.timeout,
                # CModel can decode through an explicitly configured PerfAI;
                # PCIe uses the isolated recorder decoder.  Absence of the
                # optional CModel decoder remains visible as parser_status,
                # while raw instructions and numerical correctness are kept.
                postprocess=True,
                **decoder_config,
            )
            profiler = TPUInstructionProfiler(config)
            launch_attempted = True
            report = (
                profiler.run_pcie(command, environment=worker_environment_values)
                if args.runtime_mode == "pcie" else profiler.run_cmodel(
                    command, environment=worker_environment_values))
            validate_profile_report(report, require_decoded_timing=args.require_decoded_timing)
            numeric = numeric_payload(report.stdout_path)
            validate_numeric_identity(
                numeric,
                chip=chip,
                programming_model=programming_model,
                runtime_mode=args.runtime_mode,
                case=case)
            result = profile_report_summary(
                report, require_decoded_timing=args.require_decoded_timing)
            result.update({
                "key": key,
                "chip": chip,
                "programming_model": programming_model,
                "case": case.to_json(),
                "numeric": numeric,
            })
            if args.runtime_mode == "pcie":
                assert args.device_id is not None
                tpu_smi = Path(summary["toolchain_identity"]["pcie"]["tpu_smi"]["path"])
                postflight_attempted = True
                result["board_postflight"] = board_health(
                    args.device_id, tpu_smi, quarantine_on_failure=True)
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
                summary["results"].append(result)
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
            summary["finished_at"] = utc_now()
            write_json(summary_path, summary)
            raise
        except Exception as error:
            if postflight_attempted and result is not None:
                result.update({
                    "status": "failed",
                    "execution_status": "passed",
                    "failed_phase": "board-postflight-settle",
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
                postflight_evidence = getattr(error, "evidence", None)
                if postflight_evidence is not None:
                    result["board_postflight_failure"] = postflight_evidence
            else:
                new_artifacts = sorted(
                    (path for path in output_dir.iterdir() if path not in existing_artifacts),
                    key=lambda path: path.stat().st_mtime_ns,
                )
                result = {
                    "status": "failed",
                    "key": key,
                    "chip": chip,
                    "programming_model": programming_model,
                    "case": case.to_json(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                if new_artifacts:
                    case_artifact = new_artifacts[-1]
                    result["artifact_dir"] = str(case_artifact)
                    stdout_path = case_artifact / "worker.stdout.log"
                    stderr_path = case_artifact / "worker.stderr.log"
                    if stdout_path.is_file():
                        result["stdout_path"] = str(stdout_path)
                        with suppress(OSError, ValueError, RuntimeError, json.JSONDecodeError):
                            result["numeric"] = worker_payload(stdout_path)
                    if stderr_path.is_file():
                        result["stderr_path"] = str(stderr_path)
            if (args.runtime_mode == "pcie" and launch_attempted and not postflight_attempted):
                assert args.device_id is not None
                try:
                    tpu_smi = Path(summary["toolchain_identity"]["pcie"]["tpu_smi"]["path"])
                    result["board_after_failure"] = board_health(
                        args.device_id, tpu_smi, quarantine_on_failure=True)
                except Exception as health_error:
                    result["board_after_failure_error"] = (
                        f"{type(health_error).__name__}: {health_error}")
            summary["results"].append(result)
            summary["completed_case_count"] += 1
            summary["failed_case_count"] += 1
            summary["status"] = "failed"
            summary["stopped_after"] = key
            summary["finished_at"] = utc_now()
            write_json(summary_path, summary)
            print(f"STOP {key}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
            return 1

        summary["results"].append(result)
        summary["completed_case_count"] += 1
        summary["passed_case_count"] += 1
        write_json(summary_path, summary)
        print(
            f"PASS {key} raw={result['raw_instruction_count']} "
            f"timed={result['timed_instruction_count']}",
            flush=True,
        )

    try:
        ending_identity = git_source_identity(repo_root)
        ending_toolchain = toolchain_identity(worker_environment_values, args.runtime_mode)
    except Exception as error:
        summary.update({
            "status": "failed",
            "complete": False,
            "failed_phase": "final-identity-check",
            "error_type": type(error).__name__,
            "error": str(error),
            "finished_at": utc_now(),
        })
        write_json(summary_path, summary)
        print(
            f"STOP final identity check: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True)
        return 1
    source_fields = ("git_commit", "implementation_worktree_dirty", "source_state_sha256")
    source_changed = (not isinstance(summary.get("source_state_sha256"), str) or any(
        ending_identity.get(field) != summary.get(field) for field in source_fields))
    toolchain_changed = ending_toolchain != summary.get("toolchain_identity")
    if source_changed or toolchain_changed:
        summary["status"] = "failed"
        summary["complete"] = False
        summary["source_changed_during_run"] = (ending_identity if source_changed else None)
        summary["toolchain_changed_during_run"] = toolchain_changed
        summary["finished_at"] = utc_now()
        write_json(summary_path, summary)
        print(
            "STOP source/toolchain identity changed during matrix execution",
            file=sys.stderr,
            flush=True)
        return 1
    summary["status"] = "passed"
    summary["complete"] = True
    summary["finished_at"] = utc_now()
    write_json(summary_path, summary)
    print(f"MATRIX_OK {summary_path}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    validate_args(args)
    if args.list_cases:
        cases = selected_cases(args)
        print(json.dumps([case.to_json() for case in cases], indent=2, sort_keys=True))
        return 0
    assert args.output_dir is not None
    repo_root = _REPO_ROOT
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
    cases = selected_cases(args)
    configurations = selected_configurations(args)
    scheduled = scheduled_cases(configurations, cases)
    environment = worker_environment(repo_root, args.runtime_mode, args.device_id)
    scratch_dir = Path(tempfile.mkdtemp(prefix=".scratch-", dir=output_dir))
    environment = pin_worker_cache_environment(environment, scratch_dir)
    try:
        if args.runtime_mode == "pcie":
            from tilelang.jit import TPUInstructionProfiler

            assert args.device_id is not None
            try:
                with TPUInstructionProfiler.exclusive_pcie_device(args.device_id):
                    return run_matrix(args, repo_root, output_dir, scheduled, environment)
            except Exception as error:
                summary_path = output_dir / "summary.json"
                if not summary_path.exists():
                    failure = {
                        "schema_version": SCHEMA_VERSION,
                        "matrix_kind": MATRIX_KIND,
                        "status": "failed",
                        "complete": False,
                        "runtime_mode": "pcie",
                        "failed_phase": "device-lock",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "started_at": utc_now(),
                        "finished_at": utc_now(),
                    }
                    failure.update(git_source_identity(repo_root))
                    write_json(summary_path, failure)
                print(
                    f"PCIE_SESSION_STOP: {type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True)
                return 1
        return run_matrix(args, repo_root, output_dir, scheduled, environment)
    finally:
        remove_execution_scratch(scratch_dir)


if __name__ == "__main__":
    raise SystemExit(main())
