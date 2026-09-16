# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Pure safety and acceptance tests for the FP8 profiling matrix."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import tpu_fp8_ops_matrix as matrix
import tpu_fp8_ops_worker as worker


def _args(runtime_mode="cmodel", **overrides):
    values = {
        "runtime_mode": runtime_mode,
        "programming_model": "tpukernel",
        "output_dir": Path("matrix-output"),
        "timeout": 10.0,
        "chips": ["sg2260e"],
        "dtypes": ["e4m3"],
        "cases": ["copy"],
        "device_id": 0 if runtime_mode == "pcie" else None,
        "allow_pcie": runtime_mode == "pcie",
        "allow_pcie_profile": runtime_mode == "pcie",
        "bm_cmodel_summary": Path("bm.json") if runtime_mode == "pcie" else None,
        "sg_cmodel_summary": Path("sg.json") if runtime_mode == "pcie" else None,
        "all_pcie_cases": False,
        "require_decoded_timing": False,
        "pcie_decoder_python": None,
        "pcie_decoder_pythonpath": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _source_identity():
    return {
        "git_commit": "0123456789abcdef",
        "implementation_worktree_dirty": False,
        "source_identity_scope": "tracked and untracked files excluding research/**",
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
                "resolved_path": "/native/libtilelang.so",
                "sha256": "tl"
            },
            "tvm_library": {
                "resolved_path": "/native/libtvm.so",
                "sha256": "tvm"
            },
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
                    "sha256": "bm"
                }
            },
            "sg2260e": {
                "kernel_include_tree": {
                    "sha256": "sg"
                }
            },
        },
    }
    if runtime_mode == "pcie":
        identity["pcie"] = {
            "installed_runtime_library": {
                "sha256": "runtime"
            },
            "tpu_smi": {
                "path": "/sbin/tpu-smi",
                "sha256": "smi"
            },
        }
    return identity


def _timing():
    return SimpleNamespace(engine="bd", unit="ns", begin=2, end=7, duration=5)


def _report(*, decoded=False):
    timings = (_timing(),) if decoded else ()
    return SimpleNamespace(
        output_dir=Path("profile"),
        stdout_path=Path("stdout"),
        stderr_path=Path("stderr"),
        parser_status="ready" if decoded else "not-requested",
        parser_message=None,
        raw_trace_files=(Path("global.profile"),),
        raw_instructions=(SimpleNamespace(engine="bd", opcode="15"),),
        instruction_timings=timings,
        has_raw_trace=True,
        has_instruction_timings=bool(timings),
        decoder_identity={"package": "bigTpuProfile"} if decoded else {},
    )


def _profile_environment(monkeypatch,
                         runtime_mode,
                         *,
                         chip="sg2260e",
                         programming_model="tpukernel"):
    for name in worker._PCIE_GATE_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TILELANG_TPU_PROFILE_SESSION", "1")
    monkeypatch.setenv("TILELANG_TPU_PROFILE_CHIP", chip)
    monkeypatch.setenv("TILELANG_TPU_PROFILE_PROGRAMMING_MODEL", programming_model)
    monkeypatch.setenv("TILELANG_TPU_PROFILE_RUNTIME_MODE", runtime_mode)
    monkeypatch.setenv("TILELANG_TPU_BENCHMARK_RUNS", "0")


def _patch_identity(monkeypatch, runtime_mode):
    source = _source_identity()
    toolchain = _toolchain_identity(runtime_mode)
    monkeypatch.setattr(matrix, "git_source_identity", lambda _root: dict(source))
    monkeypatch.setattr(
        matrix,
        "toolchain_identity",
        lambda _environment, requested:
        (dict(toolchain) if requested == runtime_mode else _toolchain_identity(requested)),
    )
    monkeypatch.setattr(
        matrix,
        "pin_native_worker_libraries",
        lambda environment, _identity: dict(environment),
    )
    monkeypatch.setattr(matrix, "_worker_payload", lambda _path: {"fixture": "worker-payload"})
    monkeypatch.setattr(matrix, "_validate_worker_payload", lambda _payload, **_expected: None)
    return source, toolchain


def _fp8_worker_payload(*,
                        chip="sg2260e",
                        programming_model="tpukernel",
                        runtime_mode="cmodel",
                        dtype="e4m3",
                        case="copy"):
    return {
        "schema_version": 1,
        "status": "passed",
        "chip": chip,
        "programming_model": programming_model,
        "runtime_mode": runtime_mode,
        "dtype": dtype,
        "case": case,
        "metrics": {
            "passed": True
        },
    }


def test_fp8_worker_payload_requires_one_exact_structured_result(tmp_path):
    payload = _fp8_worker_payload()
    stdout = tmp_path / "worker.stdout.log"
    stdout.write_text(matrix._WORKER_RESULT_PREFIX + json.dumps(payload) + "\n", encoding="utf-8")
    parsed = matrix._worker_payload(stdout)
    matrix._validate_worker_payload(
        parsed,
        chip="sg2260e",
        programming_model="tpukernel",
        runtime_mode="cmodel",
        dtype="e4m3",
        case="copy")

    stdout.write_text(
        "\n".join((matrix._WORKER_RESULT_PREFIX + json.dumps(payload),) * 2),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="multiple machine-readable"):
        matrix._worker_payload(stdout)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", True),
        ("status", "failed"),
        ("chip", "bm1690"),
        ("programming_model", "rv"),
        ("runtime_mode", "pcie"),
        ("dtype", "e5m2"),
        ("case", "fill-zero"),
        ("metrics", {
            "passed": False
        }),
    ),
)
def test_fp8_worker_payload_rejects_misattributed_result(field, value):
    payload = _fp8_worker_payload()
    payload[field] = value
    with pytest.raises(RuntimeError, match="schema_version|scheduled case|metrics"):
        matrix._validate_worker_payload(
            payload,
            chip="sg2260e",
            programming_model="tpukernel",
            runtime_mode="cmodel",
            dtype="e4m3",
            case="copy")


def test_fp8_worker_failure_emits_one_structured_result(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["tpu_fp8_ops_worker.py", "--dtype", "e4m3", "--case", "copy"])
    monkeypatch.setattr(worker, "_profile_selection", lambda: ("sg2260e", "tpukernel", "cmodel"))
    monkeypatch.setattr(
        worker, "_run_copy", lambda *_args, **_kwargs:
        (_ for _ in ()).throw(RuntimeError("numeric failure")))

    with pytest.raises(RuntimeError, match="numeric failure"):
        worker.main()
    markers = [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith(worker._RESULT_PREFIX)
    ]
    assert len(markers) == 1
    payload = json.loads(markers[0][len(worker._RESULT_PREFIX):])
    assert payload["status"] == "failed"
    assert payload["dtype"] == "e4m3"
    assert payload["case"] == "copy"


def _patch_pcie_preflight(monkeypatch, events):
    source, toolchain = _patch_identity(monkeypatch, "pcie")

    def promotion(_root, _args, dtypes, cases, identity, _pcie_started_at):
        assert dtypes == ("e4m3",)
        assert cases == ("copy",)
        assert identity == toolchain
        events.append("promotion")
        return {"git_commit": source["git_commit"]}

    def native_pin(environment, identity):
        assert identity == toolchain
        events.append("native-pin")
        pinned = dict(environment)
        pinned["TILELANG_LIBRARY_PATH"] = "/native"
        pinned["TVM_LIBRARY_PATH"] = "/native"
        return pinned

    def snapshot(_root, destination, commit):
        assert commit == source["git_commit"]
        events.append("snapshot")
        return {"kind": "git-archive", "git_commit": commit, "read_only": True}

    def pin(environment, *, repo_root, snapshot_root, toolchain_identity):
        del repo_root
        assert toolchain_identity == toolchain
        events.append("pin")
        pinned = dict(environment)
        pinned["PYTHONPATH"] = str(snapshot_root)
        return pinned

    def source_guard(_root, expected):
        assert expected["source_state_sha256"] == source["source_state_sha256"]
        events.append("source-guard")

    def toolchain_guard(_environment, expected):
        assert expected == toolchain
        events.append("toolchain-guard")

    def health(device_id, tpu_smi, *, quarantine_on_failure=False):
        assert device_id == 0
        assert tpu_smi == Path("/sbin/tpu-smi")
        if events.count("board-health") > 0:
            assert quarantine_on_failure is True
        events.append("board-health")
        return {"device_id": 0, "status": "Active"}

    monkeypatch.setattr(matrix, "_validate_pcie_promotion", promotion)
    monkeypatch.setattr(matrix, "pin_native_worker_libraries", native_pin)
    monkeypatch.setattr(matrix, "materialize_execution_snapshot", snapshot)
    monkeypatch.setattr(matrix, "pin_worker_environment", pin)
    monkeypatch.setattr(matrix, "assert_source_identity_unchanged", source_guard)
    monkeypatch.setattr(matrix, "assert_toolchain_identity_unchanged", toolchain_guard)
    monkeypatch.setattr(matrix, "board_health", health)


def test_cmodel_worker_rejects_inherited_pcie_gate(monkeypatch):
    _profile_environment(monkeypatch, "cmodel")
    monkeypatch.setenv("TILELANG_TPU_ALLOW_PCIE_LOAD", "1")
    with pytest.raises(RuntimeError, match="refuses PCIe"):
        worker._profile_selection()


@pytest.mark.parametrize("programming_model", ("tpukernel", "rv"))
def test_pcie_worker_accepts_only_sg2260e_device_zero_and_all_gates(monkeypatch, programming_model):
    _profile_environment(monkeypatch, "pcie", programming_model=programming_model)
    monkeypatch.setenv("TILELANG_TPU_ALLOW_PCIE_LOAD", "1")
    monkeypatch.setenv("TILELANG_TPU_ALLOW_PCIE_PROFILE", "1")
    monkeypatch.setenv("TILELANG_TPU_DEVICE_ID", "0")
    with pytest.raises(RuntimeError, match="recorder gates"):
        worker._profile_selection()

    monkeypatch.setenv("BMLIB_ENABLE_ALL_PROFILE", "1")
    assert worker._profile_selection() == ("sg2260e", programming_model, "pcie")
    monkeypatch.setenv("TILELANG_TPU_DEVICE_ID", "1")
    with pytest.raises(RuntimeError, match="only numeric device id 0"):
        worker._profile_selection()
    monkeypatch.setenv("TILELANG_TPU_DEVICE_ID", "0")
    monkeypatch.setenv("TILELANG_TPU_PROFILE_CHIP", "bm1690")
    with pytest.raises(RuntimeError, match="chip=sg2260e"):
        worker._profile_selection()


def test_validate_args_accepts_only_explicit_promoted_pcie_scope():
    matrix._validate_args(_args("pcie"))
    matrix._validate_args(_args("pcie", cases=None, all_pcie_cases=True))
    matrix._validate_args(_args("pcie", programming_model="rv"))
    matrix._validate_args(_args("pcie", programming_model="rv", cases=None, all_pcie_cases=True))


def test_default_fp8_cases_are_programming_model_specific():
    tpukernel = matrix._selected_cases(_args(cases=None))
    rv = matrix._selected_cases(_args(cases=None, programming_model="rv"))
    assert "reduce-max" in tpukernel
    assert "reduce-sum" not in tpukernel
    assert "reduce-sum" in rv
    with pytest.raises(RuntimeError, match="no validated FP8 implementation"):
        matrix._validate_args(_args(cases=["reduce-sum"]))


@pytest.mark.parametrize(
    ("overrides", "diagnostic"),
    (
        ({
            "allow_pcie": False
        }, "both --allow-pcie"),
        ({
            "allow_pcie_profile": False
        }, "both --allow-pcie"),
        ({
            "device_id": 1
        }, "only --device-id 0"),
        ({
            "device_id": False
        }, "only --device-id 0"),
        ({
            "device_id": 0.0
        }, "only --device-id 0"),
        ({
            "chips": ["bm1690"]
        }, "--chip sg2260e"),
        ({
            "chips": ["sg2260e", "sg2260e"]
        }, "--chip sg2260e"),
        ({
            "cases": None
        }, "explicit --case"),
        ({
            "bm_cmodel_summary": None
        }, "--bm-cmodel-summary"),
        ({
            "sg_cmodel_summary": None
        }, "--bm-cmodel-summary"),
    ),
)
def test_validate_args_rejects_unsafe_or_unpromoted_pcie(overrides, diagnostic):
    with pytest.raises(RuntimeError, match=diagnostic):
        matrix._validate_args(_args("pcie", **overrides))


def test_cmodel_cli_rejects_every_pcie_control():
    controls = (
        {
            "allow_pcie": True
        },
        {
            "allow_pcie_profile": True
        },
        {
            "device_id": 0
        },
        {
            "require_decoded_timing": True
        },
        {
            "pcie_decoder_pythonpath": [Path("decoder")]
        },
        {
            "bm_cmodel_summary": Path("bm.json")
        },
        {
            "sg_cmodel_summary": Path("sg.json")
        },
        {
            "all_pcie_cases": True
        },
    )
    for control in controls:
        with pytest.raises(RuntimeError, match="invalid for CModel"):
            matrix._validate_args(_args(**control))


@pytest.mark.parametrize("chips", (None, ["bm1690", "sg2260e"]))
def test_cmodel_cli_requires_exactly_one_explicit_chip(chips):
    with pytest.raises(RuntimeError, match="exactly one explicit --chip"):
        matrix._validate_args(_args(chips=chips))


def test_worker_environment_delegates_sanitization_and_owns_scratch(monkeypatch, tmp_path):
    observed = []

    def base(repo_root, runtime_mode, device_id):
        observed.append((repo_root, runtime_mode, device_id))
        return {"PPL_PROJECT_ROOT": "/sdk", "PYTHONDONTWRITEBYTECODE": "poison"}

    monkeypatch.setattr(matrix, "worker_environment", base)
    result = matrix._worker_environment(tmp_path, tmp_path / "scratch", "pcie", 0)
    assert observed == [(tmp_path, "pcie", 0)]
    assert result["TMPDIR"] == str(tmp_path / "scratch")
    assert result["PYTHONDONTWRITEBYTECODE"] == "1"


def _promotion_summary(chip, programming_model="tpukernel"):
    if chip == "bm1690":
        started_at, finished_at = ("2026-09-08T00:00:00+00:00", "2026-09-08T00:01:00+00:00")
    else:
        started_at, finished_at = ("2026-09-08T00:02:00+00:00", "2026-09-08T00:03:00+00:00")
    return {
        "schema_version": matrix._SCHEMA_VERSION,
        "matrix_kind": matrix._MATRIX_KIND,
        "status": "passed",
        "complete": True,
        "runtime_mode": "cmodel",
        "programming_model": programming_model,
        "implementation_worktree_dirty": False,
        "git_commit": _source_identity()["git_commit"],
        "source_state_sha256": _source_identity()["source_state_sha256"],
        "started_at": started_at,
        "finished_at": finished_at,
        "toolchain_identity": _toolchain_identity("cmodel"),
        "scheduled_case_count": 1,
        "completed_case_count": 1,
        "passed_case_count": 1,
        "failed_case_count": 0,
        "target_scope": [{
            "chip": chip,
            "programming_model": programming_model,
        }],
        "scheduled": [{
            "chip": chip,
            "programming_model": programming_model,
            "dtype": "e4m3",
            "case": "copy",
        }],
        "cases": {
            f"{chip}/{programming_model}/e4m3/copy": {
                "status": "passed",
                "raw_instruction_count": 1,
                "numeric": _fp8_worker_payload(chip=chip, programming_model=programming_model),
            },
        },
    }


def _promotion_fixture(monkeypatch, tmp_path, programming_model="tpukernel"):
    monkeypatch.setattr(matrix, "git_source_identity", lambda _root: _source_identity())
    bm = _promotion_summary("bm1690")
    sg = _promotion_summary("sg2260e", programming_model)
    bm_path, sg_path = tmp_path / "bm.json", tmp_path / "sg.json"
    bm_path.write_text(json.dumps(bm), encoding="utf-8")
    sg_path.write_text(json.dumps(sg), encoding="utf-8")
    args = _args(
        "pcie",
        programming_model=programming_model,
        bm_cmodel_summary=bm_path,
        sg_cmodel_summary=sg_path,
    )
    return SimpleNamespace(bm=bm, sg=sg, bm_path=bm_path, sg_path=sg_path, args=args)


@pytest.mark.parametrize("programming_model", ("tpukernel", "rv"))
def test_pcie_promotion_accepts_same_clean_source_and_content_toolchain(
        monkeypatch, tmp_path, programming_model):
    fixture = _promotion_fixture(monkeypatch, tmp_path, programming_model)
    evidence = matrix._validate_pcie_promotion(
        tmp_path,
        fixture.args,
        ("e4m3",),
        ("copy",),
        _toolchain_identity("pcie"),
        "2026-09-08T00:04:00+00:00",
    )
    assert evidence["validated_case_count"] == 1
    assert len(evidence["bm1690_summary_sha256"]) == 64
    assert len(evidence["sg2260e_summary_sha256"]) == 64


def test_rv_promotion_requires_sg_evidence_but_not_unsupported_bm_reduce_sum(
        monkeypatch, tmp_path):
    fixture = _promotion_fixture(monkeypatch, tmp_path, "rv")
    reduce_sum_key = "sg2260e/rv/e4m3/reduce-sum"
    fixture.sg["scheduled"].append({
        "chip": "sg2260e",
        "programming_model": "rv",
        "dtype": "e4m3",
        "case": "reduce-sum",
    })
    fixture.sg["cases"][reduce_sum_key] = {
        "status": "passed",
        "raw_instruction_count": 1,
        "numeric": _fp8_worker_payload(
            programming_model="rv",
            case="reduce-sum",
        ),
    }
    for field in ("scheduled_case_count", "completed_case_count", "passed_case_count"):
        fixture.sg[field] = 2
    fixture.sg_path.write_text(json.dumps(fixture.sg), encoding="utf-8")

    evidence = matrix._validate_pcie_promotion(
        tmp_path,
        fixture.args,
        ("e4m3",),
        ("copy", "reduce-sum"),
        _toolchain_identity("pcie"),
        "2026-09-08T00:04:00+00:00",
    )

    assert evidence["validated_case_count"] == 2
    assert evidence["bm1690_validated_case_count"] == 1
    assert evidence["sg2260e_validated_case_count"] == 2


@pytest.mark.parametrize("failure", ("missing", "source", "toolchain", "raw", "numeric"))
def test_pcie_promotion_rejects_missing_or_mixed_evidence(monkeypatch, tmp_path, failure):
    fixture = _promotion_fixture(monkeypatch, tmp_path)
    if failure == "missing":
        fixture.bm["cases"] = {}
        diagnostic = "scheduled work does not match"
        path, payload = fixture.bm_path, fixture.bm
    elif failure == "source":
        fixture.sg["source_state_sha256"] = "other"
        diagnostic = "source digest"
        path, payload = fixture.sg_path, fixture.sg
    elif failure == "toolchain":
        fixture.sg["toolchain_identity"]["ppl_common"] = {"sha256": "other"}
        diagnostic = "content identity"
        path, payload = fixture.sg_path, fixture.sg
    elif failure == "raw":
        next(iter(fixture.sg["cases"].values()))["raw_instruction_count"] = 0
        diagnostic = "no raw instructions"
        path, payload = fixture.sg_path, fixture.sg
    else:
        next(iter(fixture.sg["cases"].values()))["numeric"]["dtype"] = "e5m2"
        diagnostic = "scheduled case"
        path, payload = fixture.sg_path, fixture.sg
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match=diagnostic):
        matrix._validate_pcie_promotion(
            tmp_path,
            fixture.args,
            ("e4m3",),
            ("copy",),
            _toolchain_identity("pcie"),
            "2026-09-08T00:04:00+00:00",
        )


def test_pcie_preflight_snapshot_guards_and_board_checks_wrap_dispatch(monkeypatch, tmp_path):
    events = []

    class FakeProfiler:

        def __init__(self, config):
            self.config = config

        def preflight_pcie_decoder(self, *, environment):
            events.append("decoder-preflight")
            return {"package": "bigTpuProfile"}

        def run_pcie(self, command, *, environment):
            assert "execution-source" in command[1]
            assert "execution-source" in environment["PYTHONPATH"]
            events.append("dispatch")
            return _report(decoded=True)

    monkeypatch.setattr(matrix, "TPUInstructionProfiler", FakeProfiler)
    _patch_pcie_preflight(monkeypatch, events)
    output = tmp_path / "output"
    output.mkdir()
    status = matrix._run_matrix(
        _args("pcie", require_decoded_timing=True), tmp_path, output, {
            "PPL_PROJECT_ROOT": "/sdk",
            "TMPDIR": str(tmp_path / "scratch")
        })
    assert status == 0
    assert events == [
        "native-pin",
        "promotion",
        "snapshot",
        "pin",
        "decoder-preflight",
        "board-health",
        "source-guard",
        "toolchain-guard",
        "dispatch",
        "board-health",
    ]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "passed"
    assert summary["complete"] is True
    assert summary["execution_snapshot"]["read_only"] is True


def test_failed_decoder_preflight_stops_before_board_or_dispatch(monkeypatch, tmp_path):
    events, dispatches = [], []

    class FakeProfiler:

        def __init__(self, config):
            self.config = config

        def preflight_pcie_decoder(self, *, environment):
            del environment
            raise RuntimeError("decoder unavailable")

        def run_pcie(self, command, *, environment):
            dispatches.append((command, environment))
            raise AssertionError("PCIe dispatch must not be reached")

    monkeypatch.setattr(matrix, "TPUInstructionProfiler", FakeProfiler)
    _patch_pcie_preflight(monkeypatch, events)
    output = tmp_path / "output"
    output.mkdir()
    assert matrix._run_matrix(
        _args("pcie", require_decoded_timing=True), tmp_path, output, {
            "PPL_PROJECT_ROOT": "/sdk",
            "TMPDIR": str(tmp_path / "scratch")
        }) == 1
    assert dispatches == []
    assert "board-health" not in events
    summary = json.loads((output / "summary.json").read_text())
    assert summary["failed_phase"] == "preflight"
    assert summary["error"] == "decoder unavailable"


@pytest.mark.parametrize("interrupted", (False, True))
def test_failed_board_postflight_is_not_retried_and_stops_matrix(monkeypatch, tmp_path,
                                                                 interrupted):
    events = []

    class BoardHealthError(RuntimeError):

        def __init__(self, message):
            super().__init__(message)
            self.evidence = {"samples": [{"tpu_util": "9%"}], "settled": False}

    class FakeProfiler:

        def __init__(self, config):
            self.config = config

        def run_pcie(self, command, *, environment):
            del command, environment
            events.append("dispatch")
            return _report()

    monkeypatch.setattr(matrix, "TPUInstructionProfiler", FakeProfiler)
    _patch_pcie_preflight(monkeypatch, events)
    health_calls = []

    def health(device_id, tpu_smi, *, quarantine_on_failure=False):
        del device_id, tpu_smi
        health_calls.append(quarantine_on_failure)
        if len(health_calls) == 2:
            if interrupted:
                raise KeyboardInterrupt
            raise BoardHealthError("board not idle")
        return {"status": "Active"}

    monkeypatch.setattr(matrix, "board_health", health)
    output = tmp_path / "output"
    output.mkdir()

    def invoke_matrix():
        return matrix._run_matrix(
            _args("pcie"), tmp_path, output, {
                "PPL_PROJECT_ROOT": "/sdk",
                "TMPDIR": str(tmp_path / "scratch")
            })

    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            invoke_matrix()
    else:
        assert invoke_matrix() == 1
    assert health_calls == [False, True]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["completed_case_count"] == 1
    assert summary["cancelled_case_count"] == (1 if interrupted else 0)
    assert summary["failed_case_count"] == (0 if interrupted else 1)
    failed = summary["cases"]["sg2260e/tpukernel/e4m3/copy"]
    assert failed["status"] == ("cancelled" if interrupted else "failed")
    assert failed["execution_status"] == "passed"
    assert failed["failed_phase"] == "board-postflight-settle"
    assert failed["error"] == ("board postflight was interrupted"
                               if interrupted else "board not idle")
    assert "board_after_failure" not in failed
    assert failed["artifact_dir"] == "profile"
    assert failed["raw_instruction_count"] == 1
    assert failed["numeric"] == {"fixture": "worker-payload"}
    if interrupted:
        assert "board_postflight_failure" not in failed
    else:
        assert failed["board_postflight_failure"] == {
            "samples": [{
                "tpu_util": "9%"
            }],
            "settled": False
        }


def test_identity_guard_failure_stops_before_first_pcie_dispatch(monkeypatch, tmp_path):
    events, dispatches = [], []

    class FakeProfiler:

        def __init__(self, config):
            self.config = config

        def run_pcie(self, command, *, environment):
            dispatches.append(command)
            return _report()

    monkeypatch.setattr(matrix, "TPUInstructionProfiler", FakeProfiler)
    _patch_pcie_preflight(monkeypatch, events)

    def reject(_root, _expected):
        raise RuntimeError("source changed")

    monkeypatch.setattr(matrix, "assert_source_identity_unchanged", reject)
    output = tmp_path / "output"
    output.mkdir()
    assert matrix._run_matrix(
        _args("pcie"), tmp_path, output, {
            "PPL_PROJECT_ROOT": "/sdk",
            "TMPDIR": str(tmp_path / "scratch")
        }) == 1
    assert dispatches == []
    summary = json.loads((output / "summary.json").read_text())
    assert summary["stopped_after"] == "sg2260e/tpukernel/e4m3/copy"
    assert "board_after_failure" not in summary["cases"][summary["stopped_after"]]


def test_runner_stops_after_first_cmodel_failure(monkeypatch, tmp_path):
    dispatches = []

    class FakeProfiler:

        def __init__(self, config):
            self.config = config

        def run_cmodel(self, command, *, environment):
            dispatches.append(command)
            raise RuntimeError("first failed")

    monkeypatch.setattr(matrix, "TPUInstructionProfiler", FakeProfiler)
    _patch_identity(monkeypatch, "cmodel")
    output = tmp_path / "output"
    output.mkdir()
    assert matrix._run_matrix(
        _args(cases=["copy", "fill-zero"]), tmp_path, output, {"PPL_PROJECT_ROOT": "/sdk"}) == 1
    assert len(dispatches) == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["completed_case_count"] == 1
    assert summary["failed_case_count"] == 1
    assert summary["stopped_after"].endswith("/copy")


def test_cmodel_worker_receives_content_identified_native_libraries(monkeypatch, tmp_path):
    seen = []

    class FakeProfiler:

        def __init__(self, config):
            self.config = config

        def run_cmodel(self, command, *, environment):
            del command
            seen.append(dict(environment))
            return _report()

    monkeypatch.setattr(matrix, "TPUInstructionProfiler", FakeProfiler)
    _patch_identity(monkeypatch, "cmodel")

    def pin(environment, identity):
        assert identity == _toolchain_identity("cmodel")
        pinned = dict(environment)
        pinned["TILELANG_LIBRARY_PATH"] = "/captured/tilelang"
        pinned["TVM_LIBRARY_PATH"] = "/captured/tvm"
        return pinned

    monkeypatch.setattr(matrix, "pin_native_worker_libraries", pin)
    output = tmp_path / "output"
    output.mkdir()
    assert matrix._run_matrix(_args(), tmp_path, output, {"PPL_PROJECT_ROOT": "/sdk"}) == 0
    assert seen[0]["TILELANG_LIBRARY_PATH"] == "/captured/tilelang"
    assert seen[0]["TVM_LIBRARY_PATH"] == "/captured/tvm"


def test_final_toolchain_capture_failure_cannot_leave_passing_summary(monkeypatch, tmp_path):

    class FakeProfiler:

        def __init__(self, config):
            self.config = config

        def run_cmodel(self, command, *, environment):
            return _report()

    monkeypatch.setattr(matrix, "TPUInstructionProfiler", FakeProfiler)
    monkeypatch.setattr(matrix, "git_source_identity", lambda _root: _source_identity())
    captures = iter((_toolchain_identity("cmodel"), RuntimeError("identity unavailable")))

    def capture(_environment, _mode):
        value = next(captures)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(matrix, "toolchain_identity", capture)
    monkeypatch.setattr(
        matrix,
        "pin_native_worker_libraries",
        lambda environment, _identity: dict(environment),
    )
    monkeypatch.setattr(matrix, "_worker_payload", lambda _path: {"fixture": "worker-payload"})
    monkeypatch.setattr(matrix, "_validate_worker_payload", lambda _payload, **_expected: None)
    output = tmp_path / "output"
    output.mkdir()
    assert matrix._run_matrix(_args(), tmp_path, output, {"PPL_PROJECT_ROOT": "/sdk"}) == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["complete"] is False
    assert summary["failed_phase"] == "final-identity-check"


def test_main_holds_one_exclusive_device_session_and_cleans_only_scratch(monkeypatch, tmp_path):
    events = []
    output = tmp_path / "new-output"
    args = _args("pcie", output_dir=output)

    class Session:

        def __enter__(self):
            events.append("lock-enter")

        def __exit__(self, *_args):
            events.append("lock-exit")

    class FakeProfiler:

        @staticmethod
        def exclusive_pcie_device(device_id):
            assert device_id == 0
            return Session()

    def run(_args, _root, output_dir, environment):
        events.append("run")
        assert Path(environment["TMPDIR"]).is_dir()
        assert environment["TILELANG_CACHE_DIR"] == str(
            Path(environment["TMPDIR"]) / "tilelang-cache")
        (output_dir / "preserved.log").write_text("trace", encoding="utf-8")
        return 0

    original_cleanup = matrix.remove_execution_scratch

    def cleanup(path):
        events.append("cleanup")
        original_cleanup(path)

    monkeypatch.setattr(matrix, "_parse_args", lambda: args)
    monkeypatch.setattr(
        matrix,
        "worker_environment",
        lambda _root, _mode, _device: {},
    )
    monkeypatch.setattr(matrix, "_run_matrix", run)
    monkeypatch.setattr(matrix, "TPUInstructionProfiler", FakeProfiler)
    monkeypatch.setattr(matrix, "remove_execution_scratch", cleanup)
    assert matrix.main() == 0
    assert events == ["lock-enter", "run", "lock-exit", "cleanup"]
    assert (output / "preserved.log").read_text() == "trace"
    assert not any(output.glob(".scratch-*"))


def test_main_exclusively_claims_output_directory(monkeypatch, tmp_path):
    output = tmp_path / "already-exists"
    output.mkdir()
    monkeypatch.setattr(matrix, "_parse_args", lambda: _args(output_dir=output))
    with pytest.raises(RuntimeError, match="refusing to reuse"):
        matrix.main()


def test_fp8_matrix_preserves_contract_case_keys_and_adds_max():
    assert "max" in matrix._CASES
    assert "max-broadcast" in matrix._CASES
    assert f"bm1690/tpukernel/e4m3/{matrix._CASES[0]}".count("/") == 3
