# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Pure safety-policy tests for the full TPU-Kernel numerical matrix."""

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import tpukernel_ops_matrix as matrix


def _args(runtime_mode="cmodel", **overrides):
    values = {
        "runtime_mode": runtime_mode,
        "output_dir": Path("matrix-output"),
        "timeout": 10.0,
        "kill_grace": 1.0,
        "chips": ("bm1690e",),
        "operations": ("add",) if runtime_mode == "pcie" else None,
        "dtypes": None,
        "case_ids": None,
        "allow_pcie": runtime_mode == "pcie",
        "allow_pcie_load": runtime_mode == "pcie",
        "device_id": 0 if runtime_mode == "pcie" else None,
        "bm_cmodel_summary": Path("bm.json") if runtime_mode == "pcie" else None,
        "sg_cmodel_summary": Path("sg.json") if runtime_mode == "pcie" else None,
        "all_pcie_cases": False,
        "list_cases": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _source_identity():
    return {
        "git_commit": "0123456789abcdef",
        "implementation_worktree_dirty": False,
        "source_state_sha256": "source-state",
    }


def _toolchain_identity(runtime_mode):
    identity = {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "runtime_mode": runtime_mode,
        "ppl_project_root": "/sdk/ppl-1.7",
        "compiler_runtime": {
            "tilelang_library": {
                "sha256": "tilelang"
            }
        },
        "ppl_common": {
            "chip_map": {
                "sha256": "chip-map"
            }
        },
        "cmodel": {
            "runtime_tree": {
                "sha256": "cmodel-runtime"
            }
        },
        "chips": {
            "bm1690": {
                "kernel_include_tree": {
                    "sha256": "bm1690"
                }
            },
            "bm1690e": {
                "kernel_include_tree": {
                    "sha256": "bm1690e"
                }
            },
        },
    }
    if runtime_mode == "pcie":
        identity["pcie"] = {
            "installed_runtime_library": {
                "sha256": "board-runtime"
            },
            "tpu_smi": {
                "path": "/sbin/tpu-smi",
                "sha256": "smi"
            },
        }
    return identity


def _case(case_id="add.float16.dense"):
    return next(case for case in matrix.build_case_specs() if case.case_id == case_id)


@pytest.mark.parametrize(
    ("runtime_mode", "chip", "device_id"),
    (("cmodel", "bm1690", None), ("pcie", "bm1690e", 0)),
)
def test_direct_worker_environment_has_deterministic_host_tool_path(monkeypatch, tmp_path,
                                                                    runtime_mode, chip, device_id):
    monkeypatch.setenv("PPL_PROJECT_ROOT", str(tmp_path / "ppl"))
    monkeypatch.setenv("PATH", "/caller/toolchain:/volatile/bin")
    base = matrix._base_worker_environment(tmp_path, runtime_mode, device_id)

    environment = matrix._worker_environment(
        base,
        tmp_path / "scratch",
        _case(),
        chip,
        _args(runtime_mode),
    )

    expected_tool_path = os.pathsep.join(("/usr/bin", "/bin"))
    assert environment["PATH"] == expected_tool_path
    assert "/caller/toolchain" not in environment["PATH"]
    assert shutil.which("ld", path=environment["PATH"]) == "/usr/bin/ld"
    assert environment["TMPDIR"] == str((tmp_path / "scratch").resolve())
    assert environment["TILELANG_CACHE_DIR"] == str((tmp_path / "scratch/tilelang-cache").resolve())


def test_validate_args_requires_scoped_promoted_single_device_pcie():
    matrix._validate_args(_args("pcie"))

    bad = (
        ({
            "allow_pcie_load": False
        }, "--allow-pcie-load"),
        ({
            "device_id": 1
        }, "only --device-id 0"),
        ({
            "chips": ("bm1690",)
        }, "only for explicit --chip bm1690e"),
        ({
            "operations": None
        }, "explicit --op/--case subset"),
        ({
            "bm_cmodel_summary": None
        }, "--bm-cmodel-summary"),
        ({
            "all_pcie_cases": True
        }, "cannot be combined"),
    )
    for overrides, diagnostic in bad:
        with pytest.raises(RuntimeError, match=diagnostic):
            matrix._validate_args(_args("pcie", **overrides))


@pytest.mark.parametrize("device_id", (False, 0.0, "0", None, 1))
def test_validate_args_requires_exact_integer_device_zero(device_id):
    with pytest.raises(RuntimeError, match="only --device-id 0"):
        matrix._validate_args(_args("pcie", device_id=device_id))


def test_validate_args_accepts_only_explicit_unfiltered_full_pcie_matrix():
    matrix._validate_args(_args("pcie", operations=None, all_pcie_cases=True))


def test_validate_args_rejects_pcie_controls_in_cmodel():
    with pytest.raises(RuntimeError, match="invalid for CModel"):
        matrix._validate_args(_args(allow_pcie_load=True))


@pytest.mark.parametrize("chips", (None, ("bm1690", "bm1690e")))
def test_validate_args_requires_exactly_one_explicit_cmodel_chip(chips):
    with pytest.raises(RuntimeError, match="exactly one explicit --chip"):
        matrix._validate_args(_args(chips=chips))


def test_validate_numeric_identity_requires_exact_case_and_backend():
    case = _case()
    payload = {
        "status": "passed",
        "case": matrix._json_normalized(case.to_json()),
        "chip": "bm1690e",
        "programming_model": "tpukernel",
        "runtime_mode": "pcie",
        "metrics": {
            "passed": True
        },
    }
    matrix._validate_numeric_identity(payload, case=case, chip="bm1690e", runtime_mode="pcie")
    payload["case"] = matrix._json_normalized(_case("sub.float16.dense").to_json())
    with pytest.raises(RuntimeError, match="wrong case"):
        matrix._validate_numeric_identity(payload, case=case, chip="bm1690e", runtime_mode="pcie")


def test_worker_payload_rejects_multiple_structured_markers():
    payload = {
        "status": "passed",
        "case": _case().to_json(),
        "chip": "bm1690e",
        "programming_model": "tpukernel",
        "runtime_mode": "cmodel",
        "metrics": {},
    }
    marker = matrix._RESULT_PREFIX + json.dumps(payload)

    with pytest.raises(RuntimeError, match="multiple machine-readable"):
        matrix._worker_payload(marker + "\n" + marker + "\n")


def _promotion_summary(case, chip):
    source = _source_identity()
    if chip == "bm1690":
        started_at, finished_at = ("2026-09-08T00:00:00+00:00", "2026-09-08T00:01:00+00:00")
    else:
        started_at, finished_at = ("2026-09-08T00:02:00+00:00", "2026-09-08T00:03:00+00:00")
    return {
        **source,
        "schema_version":
            matrix._SCHEMA_VERSION,
        "matrix_kind":
            matrix._MATRIX_KIND,
        "status":
            "passed",
        "complete":
            True,
        "runtime_mode":
            "cmodel",
        "programming_model":
            "tpukernel",
        "started_at":
            started_at,
        "finished_at":
            finished_at,
        "toolchain_identity":
            _toolchain_identity("cmodel"),
        "scheduled_case_count":
            1,
        "completed_case_count":
            1,
        "passed_case_count":
            1,
        "failed_case_count":
            0,
        "target_scope": [{
            "chip": chip,
            "programming_model": "tpukernel",
        }],
        "scheduled": [{
            "chip": chip,
            "programming_model": "tpukernel",
            "case": matrix._json_normalized(case.to_json()),
        }],
        "results": [{
            "status": "passed",
            "case": matrix._json_normalized(case.to_json()),
            "chip": chip,
            "runtime_mode": "cmodel",
            "metrics": {
                "passed": True
            },
        }],
    }


def test_promotion_requires_matching_exact_bm_and_sg_results(monkeypatch, tmp_path):
    case = _case()
    bm_path = tmp_path / "bm.json"
    sg_path = tmp_path / "sg.json"
    bm_path.write_text(json.dumps(_promotion_summary(case, "bm1690")))
    sg_path.write_text(json.dumps(_promotion_summary(case, "bm1690e")))
    monkeypatch.setattr(matrix, "git_source_identity", lambda _root: _source_identity())
    args = _args("pcie", bm_cmodel_summary=bm_path, sg_cmodel_summary=sg_path)

    evidence = matrix._validate_pcie_promotion(
        tmp_path,
        args,
        (case,),
        _toolchain_identity("pcie"),
        "2026-09-08T00:04:00+00:00",
    )
    assert evidence["validated_case_count"] == 1

    broken = _promotion_summary(case, "bm1690e")
    broken["results"][0]["case"] = matrix._json_normalized(_case("sub.float16.dense").to_json())
    sg_path.write_text(json.dumps(broken))
    with pytest.raises(RuntimeError, match="scheduled work does not match"):
        matrix._validate_pcie_promotion(
            tmp_path,
            args,
            (case,),
            _toolchain_identity("pcie"),
            "2026-09-08T00:04:00+00:00",
        )


def test_pcie_matrix_pins_snapshot_guards_each_case_and_checks_board(monkeypatch, tmp_path):
    events = []
    source = _source_identity()
    toolchain = _toolchain_identity("pcie")
    case = _case()
    args = _args("pcie")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    scratch = output_dir / ".scratch"
    scratch.mkdir()

    monkeypatch.setattr(matrix, "git_source_identity", lambda _root: dict(source))
    monkeypatch.setattr(matrix, "toolchain_identity", lambda _env, _mode: dict(toolchain))
    monkeypatch.setattr(matrix, "pin_native_worker_libraries",
                        lambda environment, _identity: dict(environment))
    monkeypatch.setattr(
        matrix, "_validate_pcie_promotion", lambda *_args: {
            "git_commit": source["git_commit"],
            "source_state_sha256": source["source_state_sha256"],
        })

    def snapshot(_root, destination, _commit):
        events.append("snapshot")
        return {"kind": "git-archive", "read_only": True}

    def pin(environment, *, repo_root, snapshot_root, toolchain_identity):
        del repo_root, toolchain_identity
        events.append("pin")
        pinned = dict(environment)
        pinned["PYTHONPATH"] = str(snapshot_root)
        return pinned

    monkeypatch.setattr(matrix, "materialize_execution_snapshot", snapshot)
    monkeypatch.setattr(matrix, "pin_worker_environment", pin)
    monkeypatch.setattr(matrix, "assert_source_identity_unchanged",
                        lambda _root, _summary: events.append("source-guard"))
    monkeypatch.setattr(matrix, "assert_toolchain_identity_unchanged",
                        lambda _environment, _identity: events.append("toolchain-guard"))
    monkeypatch.setattr(
        matrix, "board_health", lambda device, smi, **_kwargs: events.append("board") or {
            "device_id": device,
            "tpu_smi": str(smi),
            "status": "Active"
        })

    def run_one(_args, environment, out, worker, chip, selected_case):
        events.append("dispatch")
        assert "execution-source" in str(worker)
        assert "execution-source" in environment["PYTHONPATH"]
        result = {
            "schema_version": 1,
            "status": "passed",
            "case": selected_case.to_json(),
            "chip": chip,
            "programming_model": "tpukernel",
            "runtime_mode": "pcie",
            "elapsed_seconds": 0.1,
            "launch_attempted": True,
            "worker_result": {
                "metrics": {
                    "passed": True
                }
            },
        }
        matrix._write_json(
            matrix._case_directory(out, chip, "pcie", selected_case) / "result.json",
            result,
        )
        return result

    monkeypatch.setattr(matrix, "_run_one", run_one)
    status = matrix._run_matrix(args, tmp_path, output_dir, ("bm1690e",), (case,), {
        "PPL_PROJECT_ROOT": "/sdk",
        "TMPDIR": str(scratch)
    })

    assert status == 0
    assert events == [
        "snapshot",
        "pin",
        "board",
        "source-guard",
        "toolchain-guard",
        "dispatch",
        "board",
    ]
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["complete"] is True
    assert summary["execution_snapshot"]["read_only"] is True
    assert summary["results"][0]["status"] == "passed"


@pytest.mark.parametrize(
    ("interrupted", "execution_failed"),
    ((False, False), (True, False), (False, True)),
)
def test_failed_postflight_is_not_retried(monkeypatch, tmp_path, interrupted, execution_failed):
    source = _source_identity()
    toolchain = _toolchain_identity("pcie")
    case = _case()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    scratch = output_dir / ".scratch"
    scratch.mkdir()
    board_calls = []

    monkeypatch.setattr(matrix, "git_source_identity", lambda _root: dict(source))
    monkeypatch.setattr(matrix, "toolchain_identity", lambda _env, _mode: dict(toolchain))
    monkeypatch.setattr(matrix, "pin_native_worker_libraries",
                        lambda environment, _identity: dict(environment))
    monkeypatch.setattr(matrix, "_validate_pcie_promotion",
                        lambda *_args: {"git_commit": source["git_commit"]})
    monkeypatch.setattr(matrix, "materialize_execution_snapshot", lambda *_args: {
        "kind": "git-archive",
        "read_only": True
    })
    monkeypatch.setattr(matrix, "pin_worker_environment",
                        lambda environment, **_kwargs: dict(environment))
    monkeypatch.setattr(matrix, "assert_source_identity_unchanged", lambda *_args: None)
    monkeypatch.setattr(matrix, "assert_toolchain_identity_unchanged", lambda *_args: None)

    def health(*_args, **kwargs):
        board_calls.append(dict(kwargs))
        if len(board_calls) == 2:
            if interrupted:
                raise KeyboardInterrupt
            raise RuntimeError("postflight unhealthy")
        return {"status": "Active"}

    monkeypatch.setattr(matrix, "board_health", health)

    def run_one(_args, _environment, out, _worker, chip, selected_case):
        result = {
            "status": "failed" if execution_failed else "passed",
            "case": selected_case.to_json(),
            "chip": chip,
            "runtime_mode": "pcie",
            "elapsed_seconds": 0.1,
            "launch_attempted": True,
            "worker_result": {
                "metrics": {
                    "passed": True
                }
            },
        }
        if execution_failed:
            result["failure"] = "numeric mismatch"
        matrix._write_json(
            matrix._case_directory(out, chip, "pcie", selected_case) / "result.json", result)
        return result

    monkeypatch.setattr(matrix, "_run_one", run_one)

    def invoke_matrix():
        return matrix._run_matrix(
            _args("pcie"), tmp_path, output_dir, ("bm1690e",), (case,), {"TMPDIR": str(scratch)})

    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            invoke_matrix()
    else:
        assert invoke_matrix() == 1
    assert board_calls == [{}, {"quarantine_on_failure": True}]
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["completed_case_count"] == 1
    assert summary["cancelled_case_count"] == (1 if interrupted else 0)
    assert summary["failed_case_count"] == (0 if interrupted else 1)
    result = summary["results"][0]
    assert result["failure"].startswith("numeric mismatch; board postflight failed"
                                        if execution_failed else "board postflight failed")
    assert result["execution_status"] == ("failed" if execution_failed else "passed")
    if execution_failed:
        assert result["execution_failure"] == "numeric mismatch"
    assert result["failed_phase"] == "board-postflight-settle"
    assert result["status"] == ("cancelled" if interrupted else "failed")


def test_pcie_worker_uses_fail_closed_supervisor_and_unsets_inherited_profile_controls(
        monkeypatch, tmp_path):
    from tilelang.jit import TPUInstructionProfiler

    case = _case()
    payload = {
        "status": "passed",
        "case": matrix._json_normalized(case.to_json()),
        "chip": "bm1690e",
        "programming_model": "tpukernel",
        "runtime_mode": "pcie",
        "metrics": {
            "passed": True
        },
    }
    observed = {}

    def run_probe(device_id, command, *, cwd, environment, timeout_s):
        observed.update({
            "device_id": device_id,
            "command": command,
            "cwd": cwd,
            "environment": environment,
            "timeout": timeout_s,
        })
        return SimpleNamespace(
            stdout=matrix._RESULT_PREFIX + json.dumps(payload) + "\n",
            stderr="",
            returncode=0,
            timed_out=False,
            left_live_descendant=False,
            cleanup_complete=True,
            process_group=123,
        )

    monkeypatch.setattr(TPUInstructionProfiler, "run_supervised_pcie_probe", run_probe)
    result = matrix._run_one(
        _args("pcie"),
        {
            "PPL_PROJECT_ROOT": "/sdk",
            "PYTHONPATH": "/snapshot"
        },
        tmp_path,
        Path("/snapshot/testing/python/jit/tpukernel_ops_worker.py"),
        "bm1690e",
        case,
    )

    assert result["status"] == "passed"
    assert observed["device_id"] == 0
    assert observed["command"][0] == "/usr/bin/env"
    for name in matrix._PCIE_CHILD_UNSET_KEYS:
        index = observed["command"].index(name)
        assert observed["command"][index - 1] == "-u"
    assert observed["command"][-9:] == [
        matrix.sys.executable,
        "-u",
        "/snapshot/testing/python/jit/tpukernel_ops_worker.py",
        "--case-id",
        case.case_id,
        "--chip",
        "bm1690e",
        "--runtime-mode",
        "pcie",
    ]
    assert "TILELANG_TPU_ALLOW_PCIE_PROFILE" not in observed["environment"]
    assert observed["environment"]["TILELANG_TPU_ALLOW_PCIE_LOAD"] == "1"
    assert not any(path.name.startswith(".scratch-") for path in tmp_path.rglob("*"))


def test_pcie_worker_result_write_failure_preserves_launch_evidence(monkeypatch, tmp_path):
    from tilelang.jit import TPUInstructionProfiler

    case = _case()
    payload = {
        "status": "passed",
        "case": matrix._json_normalized(case.to_json()),
        "chip": "bm1690e",
        "programming_model": "tpukernel",
        "runtime_mode": "pcie",
        "metrics": {
            "passed": True
        },
    }
    monkeypatch.setattr(
        TPUInstructionProfiler,
        "run_supervised_pcie_probe",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=matrix._RESULT_PREFIX + json.dumps(payload) + "\n",
            stderr="",
            returncode=0,
            timed_out=False,
            left_live_descendant=False,
            cleanup_complete=True,
            process_group=123,
        ),
    )

    def fail_write(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(matrix, "_write_json", fail_write)
    result = matrix._run_one(
        _args("pcie"),
        {
            "PPL_PROJECT_ROOT": "/sdk",
            "PYTHONPATH": "/snapshot"
        },
        tmp_path,
        Path("/snapshot/testing/python/jit/tpukernel_ops_worker.py"),
        "bm1690e",
        case,
    )

    assert result["status"] == "failed"
    assert result["launch_attempted"] is True
    assert result["result_file_error"].endswith("OSError: disk full")


def test_main_refuses_even_an_existing_empty_output_directory(monkeypatch, tmp_path):
    output = tmp_path / "claimed"
    output.mkdir()
    monkeypatch.setattr(matrix, "_parse_args", lambda: _args(output_dir=output))

    with pytest.raises(RuntimeError, match="refusing to reuse"):
        matrix.main()
