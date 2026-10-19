# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Shared, dependency-free metadata helpers for TPU matrix runners."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping


def unique_prefixed_json_payload_text(
    output: str,
    prefix: str,
    label: str,
) -> dict[str, Any]:
    """Parse exactly one machine payload from supervised worker output."""

    markers = [line[len(prefix):] for line in output.splitlines() if line.startswith(prefix)]
    if not markers:
        raise RuntimeError(f"{label} emitted no machine-readable result")
    if len(markers) != 1:
        raise RuntimeError(f"{label} emitted multiple machine-readable results")
    try:
        payload = json.loads(markers[0])
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{label} emitted invalid result JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} result must be a JSON object")
    return payload


def unique_prefixed_json_payload(path: Path, prefix: str, label: str) -> dict[str, Any]:
    """Read exactly one machine payload from a supervised worker log."""

    try:
        output = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise RuntimeError(f"cannot read {label} output {path}: {error}") from error
    return unique_prefixed_json_payload_text(output, prefix, label)


def matrix_target_scope(targets: Iterable[tuple[str, str]],) -> list[dict[str, str]]:
    """Return a stable, duplicate-free chip/programming-model manifest."""

    unique = tuple(dict.fromkeys(targets))
    return [{
        "chip": chip,
        "programming_model": programming_model
    } for chip, programming_model in unique]


def _summary_time(payload: Mapping[str, Any], field: str, label: str) -> datetime:
    value = payload.get(field)
    if not isinstance(value, str):
        raise RuntimeError(f"{label} promotion summary has no {field} timestamp")
    # ``datetime.fromisoformat`` learned the trailing-Z spelling after Python
    # 3.8. Normalize it so the matrix contract remains usable on Python 3.8.
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise RuntimeError(f"{label} promotion summary has an invalid {field} timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeError(f"{label} promotion summary {field} timestamp is not timezone-aware")
    return parsed


def _load_promotion_summary(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label} promotion summary {resolved}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} promotion summary is not a JSON object: {resolved}")
    if payload.get("status") != "passed" or payload.get("complete") is not True:
        raise RuntimeError(f"{label} promotion summary is not complete and passing: {resolved}")
    if payload.get("runtime_mode") != "cmodel":
        raise RuntimeError(f"{label} promotion summary is not CModel evidence: {resolved}")
    if payload.get("implementation_worktree_dirty") is not False:
        raise RuntimeError(f"{label} promotion evidence must come from a clean worktree")
    payload["_resolved_path"] = str(resolved)
    payload["_sha256"] = hashlib.sha256(raw).hexdigest()
    return payload


def validate_promotion_stages(
    bm_summary_path: Path,
    sg_summary_path: Path,
    *,
    matrix_kind: str,
    schema_version: int,
    bm_allowed_scope: Iterable[tuple[str, str]],
    sg_allowed_scope: Iterable[tuple[str, str]],
    pcie_started_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate ordered, single-chip CModel promotion evidence."""

    bm = _load_promotion_summary(bm_summary_path, "BM1690")
    sg = _load_promotion_summary(sg_summary_path, "BM1690E")
    if bm["_resolved_path"] == sg["_resolved_path"]:
        raise RuntimeError("BM1690 and BM1690E promotion summaries must be different files")

    def validate_identity_and_scope(
        payload: Mapping[str, Any],
        label: str,
        allowed_scope: Iterable[tuple[str, str]],
    ) -> None:
        if payload.get("matrix_kind") != matrix_kind:
            raise RuntimeError(f"{label} promotion summary has the wrong matrix_kind")
        observed_schema = payload.get("schema_version")
        if type(observed_schema) is not int or observed_schema != schema_version:
            raise RuntimeError(f"{label} promotion summary has the wrong schema_version")
        raw_scope = payload.get("target_scope")
        if not isinstance(raw_scope, list) or not raw_scope:
            raise RuntimeError(f"{label} promotion summary has no target_scope")
        observed = []
        for entry in raw_scope:
            if not isinstance(entry, dict):
                raise RuntimeError(
                    f"{label} promotion summary contains a non-object target_scope entry")
            chip = entry.get("chip")
            programming_model = entry.get("programming_model")
            if not isinstance(chip, str) or not isinstance(programming_model, str):
                raise RuntimeError(f"{label} promotion summary has an invalid target_scope entry")
            observed.append((chip, programming_model))
        if len(observed) != len(set(observed)):
            raise RuntimeError(f"{label} promotion summary has duplicate target_scope entries")
        allowed = set(allowed_scope)
        if not set(observed).issubset(allowed):
            raise RuntimeError(
                f"{label} promotion summary mixes targets outside its single-chip scope")

        scheduled = payload.get("scheduled")
        if not isinstance(scheduled, list) or not scheduled:
            raise RuntimeError(f"{label} promotion summary has no scheduled manifest")
        scheduled_scope = []
        scheduled_identities = []
        default_model = payload.get("programming_model")

        def work_identity(
            entry: Mapping[str, Any],
            chip: str,
            programming_model: str,
        ) -> tuple[str, str, Any, Any]:
            case = entry.get("case")
            case_id = case.get("case_id") if isinstance(case, dict) else case
            dtype = entry.get("dtype")
            if case_id is None and isinstance(entry.get("key"), str):
                key_parts = entry["key"].split("/")
                if len(key_parts) == 4:
                    _runtime_mode, key_chip, key_model, case_id = key_parts
                    if (key_chip, key_model) != (chip, programming_model):
                        raise RuntimeError(
                            f"{label} promotion summary result key disagrees with its target")
            if not isinstance(case_id, str) or not case_id:
                raise RuntimeError(
                    f"{label} promotion summary contains a result without an exact case")
            if dtype is not None and not isinstance(dtype, str):
                raise RuntimeError(f"{label} promotion summary contains an invalid dtype identity")
            return chip, programming_model, dtype, case_id

        for entry in scheduled:
            if not isinstance(entry, dict):
                raise RuntimeError(
                    f"{label} promotion summary contains a non-object scheduled entry")
            chip = entry.get("chip")
            programming_model = entry.get("programming_model", default_model)
            if not isinstance(chip, str) or not isinstance(programming_model, str):
                raise RuntimeError(f"{label} promotion summary has an invalid scheduled target")
            scheduled_scope.append((chip, programming_model))
            scheduled_identities.append(work_identity(entry, chip, programming_model))
        if set(scheduled_scope) != set(observed):
            raise RuntimeError(
                f"{label} promotion summary target_scope does not match scheduled work")

        raw_results = payload.get("results")
        result_identities = []
        result_statuses = []
        if raw_results is not None:
            if not isinstance(raw_results, list):
                raise RuntimeError(f"{label} promotion summary has an invalid result list")
            result_entries = raw_results
            for entry in result_entries:
                if not isinstance(entry, dict):
                    raise RuntimeError(f"{label} promotion summary contains a non-object result")
                chip = entry.get("chip")
                programming_model = entry.get("programming_model", default_model)
                if not isinstance(chip, str) or not isinstance(programming_model, str):
                    raise RuntimeError(f"{label} promotion summary has an invalid result target")
                if (chip, programming_model) not in set(observed):
                    raise RuntimeError(
                        f"{label} promotion summary contains a result outside target_scope")
                result_identities.append(work_identity(entry, chip, programming_model))
                result_statuses.append(entry.get("status"))
        else:
            raw_cases = payload.get("cases")
            if not isinstance(raw_cases, dict):
                raise RuntimeError(f"{label} promotion summary has no result collection")
            result_entries = list(raw_cases.values())
            for key, entry in raw_cases.items():
                if not isinstance(key, str) or not isinstance(entry, dict):
                    raise RuntimeError(f"{label} promotion summary contains an invalid case result")
                parts = key.split("/")
                if len(parts) not in (3, 4):
                    raise RuntimeError(
                        f"{label} promotion summary contains an invalid case result key")
                chip, programming_model = parts[:2]
                if (chip, programming_model) not in set(observed):
                    raise RuntimeError(
                        f"{label} promotion summary contains a result outside target_scope")
                dtype = parts[2] if len(parts) == 4 else None
                case_id = parts[-1]
                result_identities.append((chip, programming_model, dtype, case_id))
                result_statuses.append(entry.get("status"))

        if Counter(scheduled_identities) != Counter(result_identities):
            raise RuntimeError(
                f"{label} promotion summary scheduled work does not match its results")
        if any(status not in ("passed", "failed") for status in result_statuses):
            raise RuntimeError(f"{label} promotion summary contains an invalid result status")
        scheduled_count = len(scheduled_identities)
        passed_count = sum(status == "passed" for status in result_statuses)
        failed_count = sum(status == "failed" for status in result_statuses)
        expected_counts = {
            "scheduled_case_count": scheduled_count,
            "completed_case_count": len(result_identities),
            "passed_case_count": passed_count,
            "failed_case_count": failed_count,
        }
        for field, expected in expected_counts.items():
            value = payload.get(field)
            if type(value) is not int or value != expected:
                raise RuntimeError(f"{label} promotion summary has inconsistent {field}")
        if failed_count or passed_count != scheduled_count:
            raise RuntimeError(f"{label} promotion summary is not internally complete and passing")

    validate_identity_and_scope(bm, "BM1690", bm_allowed_scope)
    validate_identity_and_scope(sg, "BM1690E", sg_allowed_scope)

    bm_started = _summary_time(bm, "started_at", "BM1690")
    bm_finished = _summary_time(bm, "finished_at", "BM1690")
    sg_started = _summary_time(sg, "started_at", "BM1690E")
    sg_finished = _summary_time(sg, "finished_at", "BM1690E")
    pcie_started = _summary_time({"started_at": pcie_started_at}, "started_at", "PCIe")
    if bm_started > bm_finished:
        raise RuntimeError("CModel promotion order is invalid: BM1690 starts after it finishes")
    if sg_started > sg_finished:
        raise RuntimeError("CModel promotion order is invalid: BM1690E starts after it finishes")
    if bm_finished > sg_started:
        raise RuntimeError(
            "CModel promotion order is invalid: BM1690 must finish before BM1690E starts")
    if sg_finished > pcie_started:
        raise RuntimeError("promotion order is invalid: BM1690E must finish before PCIe starts")
    return bm, sg


def git_source_identity(repo_root: Path) -> dict[str, Any]:
    """Record the tracked implementation state used to launch a matrix.

    Research prose and ignored trace artifacts do not affect generated code,
    so the dirty check deliberately excludes ``research/**``.  Untracked
    implementation files remain part of the identity: otherwise a newly added
    worker or frontend module could be mislabeled as a clean HEAD. Missing Git is
    not silently treated as a clean checkout: the summary retains null
    identity fields plus a diagnostic so consumers can reject unverifiable
    evidence.
    """

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if not revision:
            raise RuntimeError("git returned an empty HEAD revision")
        tracked_status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                ".",
                ":(exclude)research/**",
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        tracked_diff = subprocess.run(
            ["git", "diff", "HEAD", "--binary", "--", ".", ":(exclude)research/**"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=False,
            timeout=5,
        ).stdout
        untracked_output = subprocess.run(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                ".",
                ":(exclude)research/**",
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=False,
            timeout=5,
        ).stdout
        source_hash = hashlib.sha256()
        source_hash.update(revision.encode("utf-8"))
        source_hash.update(b"\0tracked-diff\0")
        source_hash.update(tracked_diff)
        for raw_path in sorted(path for path in untracked_output.split(b"\0") if path):
            relative_path = os.fsdecode(raw_path)
            source_path = repo_root / relative_path
            source_hash.update(b"\0untracked-path\0")
            source_hash.update(raw_path)
            source_hash.update(b"\0untracked-content\0")
            source_hash.update(source_path.read_bytes())
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        return {
            "git_commit": None,
            "implementation_worktree_dirty": None,
            "source_state_sha256": None,
            "source_identity_scope": "tracked and untracked files excluding research/**",
            "git_identity_error": f"{type(error).__name__}: {error}",
        }
    return {
        "git_commit": revision,
        "implementation_worktree_dirty": bool(tracked_status.strip()),
        "source_state_sha256": source_hash.hexdigest(),
        "source_identity_scope": "tracked and untracked files excluding research/**",
    }
