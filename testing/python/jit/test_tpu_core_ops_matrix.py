# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Acceptance-policy tests for the isolated TPU core-op matrix runner."""

import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import tpu_core_ops_matrix as matrix_module
import tpu_fp8_ops_worker
import tpu_matrix_common
import tpu_profile_worker
import tpukernel_ops_worker
from tpu_core_ops_matrix import (
    _report_summary,
    _validate_profile_report,
)


def _report(*, parser_status="unavailable", timings=(), has_raw_trace=True):
    return SimpleNamespace(
        output_dir="profile",
        stdout_path=Path("worker.stdout.log"),
        parser_status=parser_status,
        parser_message="decoder is not installed",
        raw_trace_files=("global.profile",) if has_raw_trace else (),
        raw_instructions=(SimpleNamespace(engine="bd", opcode="15"),),
        instruction_timings=tuple(timings),
        has_raw_trace=has_raw_trace,
        has_instruction_timings=bool(timings),
    )


def _timing(*, begin=2, end=7, duration=5, unit="ns"):
    return SimpleNamespace(engine="bd", unit=unit, begin=begin, end=end, duration=duration)


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
            "sg2260e": {
                "kernel_include_tree": {
                    "sha256": "sg2260e"
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
                "sha256": "tpu-smi"
            },
        }
    return identity


def _matrix_args(runtime_mode="cmodel", **overrides):
    values = {
        "runtime_mode": runtime_mode,
        "output_dir": Path("matrix-output"),
        "timeout": 10.0,
        "chip": "sg2260e",
        "programming_model": "rv" if runtime_mode == "pcie" else None,
        "cases": ("elementwise-add",),
        "device_id": 0 if runtime_mode == "pcie" else None,
        "allow_pcie": runtime_mode == "pcie",
        "allow_pcie_profile": runtime_mode == "pcie",
        "bm_cmodel_summary": Path("bm-summary.json") if runtime_mode == "pcie" else None,
        "sg_cmodel_summary": Path("sg-summary.json") if runtime_mode == "pcie" else None,
        "all_pcie_cases": False,
        "pcie_decoder_python": None,
        "pcie_decoder_pythonpath": (),
        "require_decoded_timing": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _patch_identity(monkeypatch, runtime_mode):
    source = _source_identity()
    toolchain = _toolchain_identity(runtime_mode)
    monkeypatch.setattr(matrix_module, "git_source_identity", lambda _repo_root: dict(source))
    monkeypatch.setattr(
        matrix_module,
        "toolchain_identity",
        lambda _environment, requested_mode:
        (dict(toolchain)
         if requested_mode == runtime_mode else _toolchain_identity(requested_mode)),
    )
    monkeypatch.setattr(
        matrix_module,
        "pin_native_worker_libraries",
        lambda environment, _identity: dict(environment),
    )
    monkeypatch.setattr(
        matrix_module,
        "_worker_payload",
        lambda _path: {"fixture": "worker-payload"},
    )
    monkeypatch.setattr(
        matrix_module,
        "_validate_worker_payload",
        lambda _payload, **_expected: None,
    )
    return source, toolchain


def _core_worker_payload(*,
                         chip="sg2260e",
                         programming_model="rv",
                         runtime_mode="cmodel",
                         case="elementwise-add"):
    return {
        "schema_version": 1,
        "status": "passed",
        "chip": chip,
        "programming_model": programming_model,
        "runtime_mode": runtime_mode,
        "case": case,
        "metrics": {
            "passed": True
        },
    }


def test_core_worker_payload_requires_one_exact_structured_result(tmp_path):
    payload = _core_worker_payload()
    stdout = tmp_path / "worker.stdout.log"
    stdout.write_text(
        "diagnostic\n" + matrix_module._WORKER_RESULT_PREFIX + json.dumps(payload) + "\n",
        encoding="utf-8",
    )
    parsed = matrix_module._worker_payload(stdout)
    matrix_module._validate_worker_payload(
        parsed,
        chip="sg2260e",
        programming_model="rv",
        runtime_mode="cmodel",
        case="elementwise-add")

    stdout.write_text(
        "\n".join((matrix_module._WORKER_RESULT_PREFIX + json.dumps(payload),) * 2),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="multiple machine-readable"):
        matrix_module._worker_payload(stdout)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", True),
        ("status", "failed"),
        ("chip", "bm1690"),
        ("programming_model", "tpukernel"),
        ("runtime_mode", "pcie"),
        ("case", "elementwise-sub"),
        ("metrics", {
            "passed": False
        }),
    ),
)
def test_core_worker_payload_rejects_misattributed_result(field, value):
    payload = _core_worker_payload()
    payload[field] = value
    with pytest.raises(RuntimeError, match="schema_version|scheduled case|metrics"):
        matrix_module._validate_worker_payload(
            payload,
            chip="sg2260e",
            programming_model="rv",
            runtime_mode="cmodel",
            case="elementwise-add")


def test_core_worker_failure_emits_one_structured_result(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["tpu_profile_worker.py", "--case", "elementwise-add"])
    monkeypatch.setattr(tpu_profile_worker, "_profile_selection", lambda:
                        ("sg2260e", "rv", "cmodel"))
    monkeypatch.setattr(
        tpu_profile_worker, "_elementwise", lambda *_args, **_kwargs:
        (_ for _ in ()).throw(RuntimeError("numeric failure")))

    with pytest.raises(RuntimeError, match="numeric failure"):
        tpu_profile_worker.main()
    markers = [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith(tpu_profile_worker._RESULT_PREFIX)
    ]
    assert len(markers) == 1
    payload = json.loads(markers[0][len(tpu_profile_worker._RESULT_PREFIX):])
    assert payload["status"] == "failed"
    assert payload["case"] == "elementwise-add"


def _patch_pcie_preflight(monkeypatch, events):
    source, toolchain = _patch_identity(monkeypatch, "pcie")

    def promote(_root, _args, _configurations, _cases, _identity, _pcie_started_at):
        events.append("promotion")
        return {
            "git_commit": source["git_commit"],
            "source_state_sha256": source["source_state_sha256"],
        }

    def snapshot(_root, destination, commit):
        events.append("snapshot")
        assert commit == source["git_commit"]
        return {"kind": "git-archive", "git_commit": commit, "read_only": True}

    def pin(environment, *, repo_root, snapshot_root, toolchain_identity):
        del repo_root
        assert toolchain_identity == toolchain
        events.append("pin")
        pinned = dict(environment)
        pinned["PYTHONPATH"] = str(snapshot_root)
        return pinned

    def source_guard(_root, expected):
        events.append("source-guard")
        assert expected["source_state_sha256"] == source["source_state_sha256"]

    def toolchain_guard(_environment, expected):
        events.append("toolchain-guard")
        assert expected == toolchain

    def health(device_id, tpu_smi, *, quarantine_on_failure=False):
        events.append("board-health")
        assert device_id == 0
        assert tpu_smi == Path("/sbin/tpu-smi")
        if events.count("board-health") > 1:
            assert quarantine_on_failure is True
        return {"device_id": device_id, "status": "Active"}

    monkeypatch.setattr(matrix_module, "_validate_pcie_promotion", promote)
    monkeypatch.setattr(matrix_module, "materialize_execution_snapshot", snapshot)
    monkeypatch.setattr(matrix_module, "pin_worker_environment", pin)
    monkeypatch.setattr(matrix_module, "assert_source_identity_unchanged", source_guard)
    monkeypatch.setattr(matrix_module, "assert_toolchain_identity_unchanged", toolchain_guard)
    monkeypatch.setattr(matrix_module, "board_health", health)
    return source, toolchain


_INVALID_TIMINGS = (
    _timing(duration=None),
    _timing(duration=True),
    _timing(duration=float("nan")),
    _timing(duration=float("inf")),
    _timing(duration=-1),
    _timing(begin=True),
    _timing(begin=float("nan")),
    _timing(end=float("inf")),
    _timing(begin=8, end=7, duration=1),
    _timing(unit=""),
    _timing(unit="cycles"),
)


@pytest.mark.parametrize(
    "source_name",
    (
        "testing/python/jit/tpukernel_ops_worker.py",
        "testing/python/jit/tpu_fp8_ops_worker.py",
        "testing/python/jit/tpu_profile_worker.py",
        "tpu_demo/elementwise/elementwise.py",
        "tpu_demo/flashattn/flashattn.py",
        "tpu_demo/matmul/matmul.py",
        "tpu_demo/rmsnorm/rmsnorm.py",
        "tpu_demo/rope/rope.py",
        "tpu_demo/swiglu/swiglu.py",
    ),
)
def test_tir_script_workers_keep_evaluated_annotations(source_name):
    """TVM Script consumes ``T.Tensor`` annotations as live objects."""

    repo_root = Path(__file__).resolve().parents[3]
    tree = ast.parse((repo_root / source_name).read_text(encoding="utf-8"))
    annotation_futures = [
        alias.name for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "__future__" for alias in node.names
    ]

    assert "annotations" not in annotation_futures


def test_fp8_elementwise_case_names_preserve_unsuffixed_operations():
    cases = ("add", "sub", "mul", "max", "add-broadcast", "sub-broadcast", "mul-broadcast",
             "max-broadcast")

    assert [tpu_fp8_ops_worker._elementwise_operation(case) for case in cases
           ] == ["add", "sub", "mul", "max", "add", "sub", "mul", "max"]


def test_git_source_identity_records_revision_dirty_state_and_digest(monkeypatch):
    responses = iter((
        SimpleNamespace(stdout="55c1c6d\n"),
        SimpleNamespace(stdout=" M tracked.py\n"),
        SimpleNamespace(stdout=b"tracked diff"),
        SimpleNamespace(stdout=b""),
    ))
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return next(responses)

    monkeypatch.setattr(tpu_matrix_common.subprocess, "run", fake_run)

    identity = tpu_matrix_common.git_source_identity(matrix_module.Path("/repo"))

    assert identity["git_commit"] == "55c1c6d"
    assert identity["implementation_worktree_dirty"] is True
    assert identity["source_identity_scope"] == (
        "tracked and untracked files excluding research/**")
    assert len(identity["source_state_sha256"]) == 64
    assert calls[1][0][-2:] == [".", ":(exclude)research/**"]
    assert calls[2][0][1:4] == ["diff", "HEAD", "--binary"]
    assert calls[3][0][1:4] == ["ls-files", "--others", "--exclude-standard"]
    assert all(call[1]["timeout"] == 5 for call in calls)


def test_portable_copy_cases_are_worker_cases_and_default_matrix_cases():
    expected = (
        "copy-fp32-local-roundtrip",
        "copy-fp32-global-to-global",
        "copy-fp16-local-roundtrip",
        "copy-fp16-global-to-global",
    )

    assert tuple(tpu_profile_worker._COPY_CASES) == expected
    assert expected == matrix_module._COPY_CASES
    assert all(case in matrix_module._CASES for case in expected)


def test_fp16_bf16_conversion_cases_are_portable_matrix_cases():
    expected = {
        "copy-fp16-to-bf16": ("float16", "bfloat16"),
        "copy-bf16-to-fp16": ("bfloat16", "float16"),
    }
    assert expected == tpu_profile_worker._CONVERSION_CASES
    assert tuple(expected) == matrix_module._CONVERSION_CASES
    assert matrix_module._CASES[-len(expected):] == tuple(expected)


def test_fp32_transpose_a_cases_cover_overwrite_and_accumulation():
    expected = {
        "matmul-fp32-transpose-a-overwrite": False,
        "matmul-fp32-transpose-a-accumulate": True,
    }
    assert expected == tpu_profile_worker._FP32_TRANSPOSE_A_CASES
    assert tuple(expected) == matrix_module._FP32_TRANSPOSE_A_CASES
    assert all(case in matrix_module._CASES for case in expected)


def test_portable_max_cases_cover_dtypes_broadcast_and_negative_infinity():
    expected_profile_cases = {
        "elementwise-max-fp16-dense": ("float16", "dense"),
        "elementwise-max-bf16-dense": ("bfloat16", "dense"),
        "elementwise-max-fp32-dense": ("float32", "dense"),
        "elementwise-max-fp16-broadcast": ("float16", "broadcast"),
        "elementwise-max-bf16-broadcast": ("bfloat16", "broadcast"),
        "elementwise-max-fp32-broadcast": ("float32", "broadcast"),
        "elementwise-max-fp32-negative-infinity": ("float32", "negative-infinity"),
    }
    assert expected_profile_cases == tpu_profile_worker._MAX_CASES
    assert tuple(expected_profile_cases) == matrix_module._MAX_CASES
    assert all(case in matrix_module._CASES for case in expected_profile_cases)

    tpukernel_cases = {
        case.case_id: dict(case.parameters)
        for case in tpukernel_ops_worker.build_case_specs()
        if case.operation == "max"
    }
    assert tpukernel_cases == {
        "max.float16.dense": {},
        "max.bfloat16.dense": {},
        "max.float32.dense": {},
        "max.float32.broadcast": {
            "broadcast_rhs": True
        },
        "max.float32.negative-infinity": {
            "negative_infinity_sentinel": True
        },
    }


def test_portable_base_elementwise_matrix_includes_all_base_float_w_broadcasts():
    expected = {
        f"elementwise-{operation}-{dtype}-broadcast": (operation, dtype)
        for operation in ("add", "sub", "mul", "div") for dtype in ("fp16", "bf16", "fp32")
    }
    assert expected == tpu_profile_worker._BROADCAST_CASES
    assert tuple(expected) == matrix_module._BROADCAST_CASES
    assert all(case in matrix_module._CASES for case in expected)


def test_runner_dispatches_every_portable_copy_case(monkeypatch, tmp_path):
    calls = []
    configs = []

    class FakeProfiler:

        def __init__(self, config):
            configs.append(config)

        def run_cmodel(self, command, *, environment):
            calls.append((command, environment))
            return _report()

    monkeypatch.setattr(matrix_module, "TPUInstructionProfiler", FakeProfiler)
    _patch_identity(monkeypatch, "cmodel")
    output_dir = tmp_path / "matrix"
    output_dir.mkdir()
    args = _matrix_args()

    status = matrix_module._run_matrix(
        args,
        tmp_path,
        output_dir,
        (("sg2260e", "rv"),),
        matrix_module._COPY_CASES,
        {"PPL_PROJECT_ROOT": "/sdk"},
    )

    assert status == 0
    assert [call[0][-2:] for call in calls
           ] == [["--case", case] for case in matrix_module._COPY_CASES]
    assert all(call[0][0] == sys.executable for call in calls)
    assert [config.label for config in configs
           ] == [f"sg2260e-rv-{case}" for case in matrix_module._COPY_CASES]
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["complete"] is True
    assert set(summary["cases"]) == set(f"sg2260e/rv/{case}" for case in matrix_module._COPY_CASES)


def test_required_decoding_preflights_before_any_pcie_dispatch(monkeypatch, tmp_path):
    events = []

    class FakeProfiler:

        def __init__(self, config):
            self.config = config

        def preflight_pcie_decoder(self, *, environment):
            events.append("decoder-preflight")
            return {
                "package": "bigTpuProfile",
                "package_version": "0.3.5",
                "parser_api": "bigTpuProfile.bmprofile_perfAI_2260.BMProfileParserPerfAI.parse",
            }

        def run_pcie(self, command, *, environment):
            events.append("dispatch")
            assert "execution-source" in command[1]
            assert "execution-source" in environment["PYTHONPATH"]
            return _report(parser_status="ready", timings=(_timing(),))

    monkeypatch.setattr(matrix_module, "TPUInstructionProfiler", FakeProfiler)
    _patch_pcie_preflight(monkeypatch, events)
    output_dir = tmp_path / "matrix"
    output_dir.mkdir()
    args = _matrix_args("pcie", require_decoded_timing=True)

    status = matrix_module._run_matrix(
        args,
        tmp_path,
        output_dir,
        (("sg2260e", "rv"),),
        ("elementwise-add",),
        {
            "PPL_PROJECT_ROOT": "/sdk",
            "TMPDIR": str(tmp_path / "scratch")
        },
    )

    assert status == 0
    assert events == [
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
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["decoder_preflight"] == {
        "status": "ready",
        "identity": {
            "package": "bigTpuProfile",
            "package_version": "0.3.5",
            "parser_api": "bigTpuProfile.bmprofile_perfAI_2260.BMProfileParserPerfAI.parse",
        },
    }


def test_failed_decoder_preflight_stops_matrix_without_dispatch(monkeypatch, tmp_path):
    dispatches = []
    events = []

    class FakeProfiler:

        def __init__(self, config):
            pass

        def preflight_pcie_decoder(self, *, environment):
            raise RuntimeError("decoder unavailable")

        def run_pcie(self, command, *, environment):
            dispatches.append(command)
            raise AssertionError("hardware dispatch must not be reached")

    monkeypatch.setattr(matrix_module, "TPUInstructionProfiler", FakeProfiler)
    _patch_pcie_preflight(monkeypatch, events)
    output_dir = tmp_path / "matrix"
    output_dir.mkdir()
    args = _matrix_args("pcie", require_decoded_timing=True)

    status = matrix_module._run_matrix(
        args,
        tmp_path,
        output_dir,
        (("sg2260e", "rv"),),
        ("elementwise-add",),
        {
            "PPL_PROJECT_ROOT": "/sdk",
            "TMPDIR": str(tmp_path / "scratch")
        },
    )

    assert status == 1
    assert dispatches == []
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["failed_phase"] == "preflight"
    assert summary["error"] == "decoder unavailable"
    assert summary["cases"] == {}


def test_validate_args_accepts_only_explicit_promoted_pcie_scope():
    matrix_module._validate_args(_matrix_args("pcie"))
    matrix_module._validate_args(_matrix_args("pcie", cases=None, all_pcie_cases=True))


@pytest.mark.parametrize(
    ("overrides", "diagnostic"),
    (
        ({
            "allow_pcie": False
        }, "--allow-pcie and --allow-pcie-profile"),
        ({
            "allow_pcie_profile": False
        }, "--allow-pcie and --allow-pcie-profile"),
        ({
            "device_id": 1
        }, "only --device-id 0"),
        ({
            "device_id": False
        }, "non-negative 32-bit"),
        ({
            "chip": "bm1690"
        }, "explicit --chip sg2260e"),
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
        matrix_module._validate_args(_matrix_args("pcie", **overrides))


def test_validate_args_rejects_pcie_controls_in_cmodel():
    with pytest.raises(RuntimeError, match="must not be supplied to CModel"):
        matrix_module._validate_args(_matrix_args("cmodel", allow_pcie=True, device_id=0))


def test_validate_args_requires_explicit_single_chip_for_cmodel_and_pcie():
    for runtime_mode in ("cmodel", "pcie"):
        with pytest.raises(RuntimeError, match="explicit --chip|one explicit --chip"):
            matrix_module._validate_args(_matrix_args(runtime_mode, chip=None))


def _core_promotion_summary(*, chip, programming_model, case="elementwise-add"):
    if chip == "bm1690":
        started_at, finished_at = ("2026-09-08T00:00:00+00:00", "2026-09-08T00:01:00+00:00")
    else:
        started_at, finished_at = ("2026-09-08T00:02:00+00:00", "2026-09-08T00:03:00+00:00")
    return {
        "schema_version": matrix_module._SCHEMA_VERSION,
        "matrix_kind": matrix_module._MATRIX_KIND,
        "status": "passed",
        "complete": True,
        "runtime_mode": "cmodel",
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
            "case": case,
        }],
        "cases": {
            f"{chip}/{programming_model}/{case}": {
                "status":
                    "passed",
                "raw_instruction_count":
                    1,
                "numeric":
                    _core_worker_payload(chip=chip, programming_model=programming_model, case=case),
            },
        },
    }


def _promotion_fixture(monkeypatch, tmp_path):
    source = _source_identity()
    monkeypatch.setattr(matrix_module, "git_source_identity", lambda _repo_root: dict(source))
    bm_path = tmp_path / "bm.json"
    sg_path = tmp_path / "sg.json"
    bm = _core_promotion_summary(chip="bm1690", programming_model="tpukernel")
    sg = _core_promotion_summary(chip="sg2260e", programming_model="rv")
    bm_path.write_text(json.dumps(bm), encoding="utf-8")
    sg_path.write_text(json.dumps(sg), encoding="utf-8")
    args = _matrix_args("pcie", bm_cmodel_summary=bm_path, sg_cmodel_summary=sg_path)
    return SimpleNamespace(
        source=source,
        bm=bm,
        sg=sg,
        bm_path=bm_path,
        sg_path=sg_path,
        args=args,
    )


def _validate_promotion(fixture, tmp_path):
    return matrix_module._validate_pcie_promotion(
        tmp_path,
        fixture.args,
        (("sg2260e", "rv"),),
        ("elementwise-add",),
        _toolchain_identity("pcie"),
        "2026-09-08T00:04:00+00:00",
    )


def test_pcie_promotion_accepts_bm_and_sg_content_matched_evidence(monkeypatch, tmp_path):
    fixture = _promotion_fixture(monkeypatch, tmp_path)

    evidence = _validate_promotion(fixture, tmp_path)

    assert evidence["git_commit"] == fixture.source["git_commit"]
    assert evidence["validated_case_count"] == 1
    assert len(evidence["bm1690_summary_sha256"]) == 64
    assert len(evidence["sg2260e_summary_sha256"]) == 64


@pytest.mark.parametrize(
    "failure",
    ("missing-bm-case", "toolchain-mismatch", "source-mismatch", "numeric-mismatch"),
)
def test_pcie_promotion_rejects_incomplete_or_identity_mixed_evidence(monkeypatch, tmp_path,
                                                                      failure):
    fixture = _promotion_fixture(monkeypatch, tmp_path)
    if failure == "missing-bm-case":
        fixture.bm["cases"] = {}
        diagnostic = "scheduled work does not match"
        path = fixture.bm_path
        payload = fixture.bm
    elif failure == "toolchain-mismatch":
        fixture.sg["toolchain_identity"]["ppl_common"] = {"sha256": "different"}
        diagnostic = "content identity"
        path = fixture.sg_path
        payload = fixture.sg
    elif failure == "source-mismatch":
        fixture.sg["source_state_sha256"] = "different-source"
        diagnostic = "source digest"
        path = fixture.sg_path
        payload = fixture.sg
    else:
        next(iter(fixture.sg["cases"].values()))["numeric"]["case"] = "elementwise-sub"
        diagnostic = "scheduled case"
        path = fixture.sg_path
        payload = fixture.sg
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match=diagnostic):
        _validate_promotion(fixture, tmp_path)


def test_runner_stops_after_first_case_failure(monkeypatch, tmp_path):
    dispatches = []

    class FakeProfiler:

        def __init__(self, config):
            self.config = config

        def run_cmodel(self, command, *, environment):
            dispatches.append((command, environment))
            raise RuntimeError("first case failed")

    monkeypatch.setattr(matrix_module, "TPUInstructionProfiler", FakeProfiler)
    _patch_identity(monkeypatch, "cmodel")
    output_dir = tmp_path / "matrix"
    output_dir.mkdir()

    status = matrix_module._run_matrix(
        _matrix_args(),
        tmp_path,
        output_dir,
        (("sg2260e", "rv"),),
        ("elementwise-add", "elementwise-sub"),
        {"PPL_PROJECT_ROOT": "/sdk"},
    )

    assert status == 1
    assert len(dispatches) == 1
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["complete"] is False
    assert summary["completed_case_count"] == 1
    assert summary["failed_case_count"] == 1
    assert summary["stopped_after"] == "sg2260e/rv/elementwise-add"


def test_pcie_per_case_identity_guard_runs_before_dispatch(monkeypatch, tmp_path):
    events = []
    dispatches = []

    class FakeProfiler:

        def __init__(self, config):
            self.config = config

        def run_pcie(self, command, *, environment):
            dispatches.append((command, environment))
            return _report()

    monkeypatch.setattr(matrix_module, "TPUInstructionProfiler", FakeProfiler)
    _patch_pcie_preflight(monkeypatch, events)

    def changed_source(_root, _expected):
        events.append("source-guard-rejected")
        raise RuntimeError("source changed")

    monkeypatch.setattr(matrix_module, "assert_source_identity_unchanged", changed_source)
    output_dir = tmp_path / "matrix"
    output_dir.mkdir()

    status = matrix_module._run_matrix(
        _matrix_args("pcie"),
        tmp_path,
        output_dir,
        (("sg2260e", "rv"),),
        ("elementwise-add", "elementwise-sub"),
        {
            "PPL_PROJECT_ROOT": "/sdk",
            "TMPDIR": str(tmp_path / "scratch")
        },
    )

    assert status == 1
    assert dispatches == []
    assert events[-1] == "source-guard-rejected"
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["stopped_after"] == "sg2260e/rv/elementwise-add"
    result = summary["cases"][summary["stopped_after"]]
    assert "board_after_failure" not in result


@pytest.mark.parametrize("interrupted", (False, True))
def test_failed_postflight_is_not_retried_on_a_suspect_board(monkeypatch, tmp_path, interrupted):
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

    monkeypatch.setattr(matrix_module, "TPUInstructionProfiler", FakeProfiler)
    _patch_pcie_preflight(monkeypatch, events)
    health_calls = []

    def health(device_id, tpu_smi, *, quarantine_on_failure=False):
        health_calls.append((device_id, tpu_smi, quarantine_on_failure))
        if len(health_calls) == 2:
            if interrupted:
                raise KeyboardInterrupt
            raise BoardHealthError("postflight board health failed")
        return {"device_id": device_id, "status": "Active"}

    monkeypatch.setattr(matrix_module, "board_health", health)
    output_dir = tmp_path / "matrix"
    output_dir.mkdir()

    def invoke_matrix():
        return matrix_module._run_matrix(
            _matrix_args("pcie"), tmp_path, output_dir, (("sg2260e", "rv"),),
            ("elementwise-add", "elementwise-sub"), {
                "PPL_PROJECT_ROOT": "/sdk",
                "TMPDIR": str(tmp_path / "scratch")
            })

    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            invoke_matrix()
    else:
        assert invoke_matrix() == 1
    assert len(health_calls) == 2
    assert events.count("dispatch") == 1
    assert health_calls == [
        (0, Path("/sbin/tpu-smi"), False),
        (0, Path("/sbin/tpu-smi"), True),
    ]
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["completed_case_count"] == 1
    assert summary["cancelled_case_count"] == (1 if interrupted else 0)
    assert summary["failed_case_count"] == (0 if interrupted else 1)
    result = summary["cases"]["sg2260e/rv/elementwise-add"]
    assert "board_after_failure" not in result
    assert result["status"] == ("cancelled" if interrupted else "failed")
    assert result["execution_status"] == "passed"
    assert result["failed_phase"] == "board-postflight-settle"
    assert result["error"] == ("board postflight was interrupted"
                               if interrupted else "postflight board health failed")
    assert result["artifact_dir"] == "profile"
    assert result["raw_instruction_count"] == 1
    assert result["numeric"] == {"fixture": "worker-payload"}
    if interrupted:
        assert "board_postflight_failure" not in result
    else:
        assert result["board_postflight_failure"] == {
            "samples": [{
                "tpu_util": "9%"
            }],
            "settled": False
        }


def test_final_identity_capture_failure_cannot_leave_passing_summary(monkeypatch, tmp_path):

    class FakeProfiler:

        def __init__(self, config):
            self.config = config

        def run_cmodel(self, command, *, environment):
            return _report()

    monkeypatch.setattr(matrix_module, "TPUInstructionProfiler", FakeProfiler)
    monkeypatch.setattr(matrix_module, "git_source_identity", lambda _repo_root: _source_identity())
    captures = iter((_toolchain_identity("cmodel"), RuntimeError("identity unavailable")))

    def capture(_environment, _runtime_mode):
        value = next(captures)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(matrix_module, "toolchain_identity", capture)
    monkeypatch.setattr(
        matrix_module,
        "pin_native_worker_libraries",
        lambda environment, _identity: dict(environment),
    )
    monkeypatch.setattr(matrix_module, "_worker_payload",
                        lambda _path: {"fixture": "worker-payload"})
    monkeypatch.setattr(matrix_module, "_validate_worker_payload",
                        lambda _payload, **_expected: None)
    output_dir = tmp_path / "matrix"
    output_dir.mkdir()

    status = matrix_module._run_matrix(_matrix_args(), tmp_path, output_dir, (("sg2260e", "rv"),),
                                       ("elementwise-add",), {"PPL_PROJECT_ROOT": "/sdk"})

    assert status == 1
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["complete"] is False
    assert summary["failed_phase"] == "final-identity-check"


def test_main_holds_one_exclusive_device_session_and_cleans_scratch(monkeypatch, tmp_path):
    events = []
    output_dir = tmp_path / "matrix"
    args = _matrix_args("pcie", output_dir=output_dir)

    class DeviceSession:

        def __enter__(self):
            events.append("lock-enter")

        def __exit__(self, error_type, error, traceback):
            del error_type, error, traceback
            events.append("lock-exit")

    class FakeProfiler:

        @staticmethod
        def exclusive_pcie_device(device_id):
            assert device_id == 0
            return DeviceSession()

    def run(_args, _root, _output, _configurations, _cases, environment):
        events.append("run")
        assert Path(environment["TMPDIR"]).is_dir()
        assert environment["TILELANG_CACHE_DIR"] == str(
            Path(environment["TMPDIR"]) / "tilelang-cache")
        return 0

    original_cleanup = matrix_module.remove_execution_scratch

    def cleanup(path):
        events.append("cleanup")
        original_cleanup(path)

    monkeypatch.setattr(matrix_module, "_parse_args", lambda: args)
    monkeypatch.setattr(matrix_module, "_worker_environment",
                        lambda *_args: {"PPL_PROJECT_ROOT": "/sdk"})
    monkeypatch.setattr(matrix_module, "_run_matrix", run)
    monkeypatch.setattr(matrix_module, "TPUInstructionProfiler", FakeProfiler)
    monkeypatch.setattr(matrix_module, "remove_execution_scratch", cleanup)

    assert matrix_module.main() == 0
    assert events == ["lock-enter", "run", "lock-exit", "cleanup"]
    assert not any(output_dir.glob(".scratch-*"))


def test_main_cleans_scratch_when_environment_construction_fails(monkeypatch, tmp_path):
    output_dir = tmp_path / "matrix"
    args = _matrix_args("cmodel", output_dir=output_dir)
    monkeypatch.setattr(matrix_module, "_parse_args", lambda: args)

    def fail_environment(*_args):
        raise RuntimeError("environment unavailable")

    monkeypatch.setattr(matrix_module, "_worker_environment", fail_environment)

    with pytest.raises(RuntimeError, match="environment unavailable"):
        matrix_module.main()
    assert not any(output_dir.glob(".scratch-*"))


def test_main_rejects_even_empty_existing_output_to_prevent_runner_race(monkeypatch, tmp_path):
    output_dir = tmp_path / "matrix"
    output_dir.mkdir()
    monkeypatch.setattr(matrix_module, "_parse_args",
                        lambda: _matrix_args("cmodel", output_dir=output_dir))

    with pytest.raises(RuntimeError, match="existing matrix output directory"):
        matrix_module.main()


def test_numeric_and_raw_acceptance_does_not_require_vendor_decoder():
    report = _report()

    _validate_profile_report(report, require_decoded_timing=False)
    summary = _report_summary(report, require_decoded_timing=False)

    assert summary["status"] == "passed"
    assert summary["parser_status"] == "unavailable"
    assert summary["decoded_timing_required"] is False
    assert summary["decoded_timing_accepted"] is False


def test_raw_trace_remains_mandatory_for_every_acceptance_policy():
    report = _report(has_raw_trace=False)

    with pytest.raises(RuntimeError, match="no profiling trace"):
        _validate_profile_report(report, require_decoded_timing=False)


def test_explicit_decoded_timing_acceptance_rejects_missing_decoder():
    report = _report()

    with pytest.raises(RuntimeError, match="explicitly required"):
        _validate_profile_report(report, require_decoded_timing=True)


def test_explicit_decoded_timing_acceptance_accepts_valid_rows():
    report = _report(parser_status="ready", timings=(_timing(),))

    _validate_profile_report(report, require_decoded_timing=True)
    summary = _report_summary(report, require_decoded_timing=True)

    assert summary["decoded_timing_required"] is True
    assert summary["decoded_timing_accepted"] is True
    assert summary["timed_instruction_count"] == 1


@pytest.mark.parametrize(
    "timing",
    _INVALID_TIMINGS,
)
def test_explicit_decoded_timing_acceptance_rejects_invalid_rows(timing):
    report = _report(parser_status="ready", timings=(timing,))

    with pytest.raises(RuntimeError, match="invalid instruction interval"):
        _validate_profile_report(report, require_decoded_timing=True)


@pytest.mark.parametrize("timing", _INVALID_TIMINGS)
def test_optional_decoded_timing_never_accepts_invalid_rows(timing):
    report = _report(parser_status="ready", timings=(timing,))

    _validate_profile_report(report, require_decoded_timing=False)
    summary = _report_summary(report, require_decoded_timing=False)

    assert summary["status"] == "passed"
    assert summary["decoded_timing_accepted"] is False
    assert summary["timing_by_engine_and_unit"] == {}
