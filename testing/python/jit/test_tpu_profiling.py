# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Unit tests for the isolated PPL-style TPU instruction profile worker."""

from pathlib import Path
from contextlib import suppress
import fcntl
import hashlib
import importlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Optional

import pytest

from tilelang.engine.tpu_config import TPURuntimeConfig, TPUTargetSpec
from tilelang.jit.adapter import tpu_profiling as tpu_profiling_module
from tilelang.jit.adapter.tpu_profiling import (
    TPUInstructionProfiler,
    TPUProfilingCommandError,
    TPUProfilingConfig,
    TPUProfilingError,
    TPUProfilingTimeoutError,
    parse_pcie_decoded_instruction_timings,
    parse_perfai_instruction_timings,
    parse_perfai_timeline_events,
    run_tpu_cmodel_profile,
    run_tpu_pcie_profile,
)


def _fake_trace_worker(label: str, programming_model: str = "tpukernel", sleep_s: float = 0.0):
    """A child command that models the CModel's relative FILE_DUMP_CMD output."""

    source = ("import os\n"
              "import time\n"
              "from pathlib import Path\n"
              f"time.sleep({sleep_s!r})\n"
              "label = os.environ['FILE_DUMP_CMD']\n"
              "assert '/' not in label\n"
              "assert os.environ['TPU_RT_CORE_NUM'] == '4'\n"
              "assert os.environ['TILELANG_TPU_PROFILE_CHIP'] == 'bm1690e'\n"
              "assert os.environ['TILELANG_TPU_PROFILE_PROGRAMMING_MODEL'] == "
              f"'{programming_model}'\n"
              "assert os.environ['TILELANG_TPU_PROFILE_RUNTIME_MODE'] == 'cmodel'\n"
              "assert os.environ['TILELANG_TPU_BENCHMARK_RUNS'] == '0'\n"
              "assert 'TILELANG_TPU_ALLOW_PCIE_LOAD' not in os.environ\n"
              "assert 'TILELANG_TPU_ALLOW_PCIE_PROFILE' not in os.environ\n"
              "assert 'TILELANG_TPU_DEVICE_ID' not in os.environ\n"
              "Path('.pkl_memoize_py3').mkdir()\n"
              "Path(label + '-0-0.BD.0').write_bytes(b'raw-bd')\n"
              "Path(label + '-0-0.GDMA.0').write_bytes(b'raw-gdma')\n"
              "Path(label + '-0-0.BD.0.txt').write_text(\n"
              "    'bd cmd_id=7 bd_func=15\\n', encoding='utf-8')\n"
              "Path(label + '-0-0.GDMA.0.txt').write_text(\n"
              "    'gdma cmd_id=9 gdma_func=6\\n', encoding='utf-8')\n")
    return [sys.executable, "-c", source]


def _fake_pcie_trace_worker(programming_model: str = "tpukernel",
                            profile_name: Optional[str] = "global.profile",
                            profile_payload: bytes = b"raw-pcie",
                            forbidden_pythonpath: Optional[Path] = None):
    """Model the environment and recorder artifact of one PCIe launch."""

    profile_write = ""
    if profile_name is not None:
        profile_write = (f"(profile / {profile_name!r}).write_bytes({profile_payload!r})\n")
    pythonpath_check = ""
    if forbidden_pythonpath is not None:
        pythonpath_check = (f"assert {str(forbidden_pythonpath)!r} not in "
                            "os.environ.get('PYTHONPATH', '').split(os.pathsep)\n")
    source = ("import os\n"
              "from pathlib import Path\n"
              "assert 'FILE_DUMP_CMD' not in os.environ\n"
              "assert os.environ['TILELANG_TPU_PROFILE_SESSION'] == '1'\n"
              "assert os.environ['TILELANG_TPU_PROFILE_CHIP'] == 'bm1690e'\n"
              "assert os.environ['TILELANG_TPU_PROFILE_PROGRAMMING_MODEL'] == "
              f"'{programming_model}'\n"
              "assert os.environ['TILELANG_TPU_PROFILE_RUNTIME_MODE'] == 'pcie'\n"
              "assert os.environ['TILELANG_TPU_BENCHMARK_RUNS'] == '0'\n"
              "assert os.environ['TILELANG_TPU_ALLOW_PCIE_LOAD'] == '1'\n"
              "assert os.environ['TILELANG_TPU_ALLOW_PCIE_PROFILE'] == '1'\n"
              "assert os.environ['TILELANG_TPU_DEVICE_ID'] == '0'\n"
              "assert os.environ['BMLIB_ENABLE_ALL_PROFILE'] == '1'\n"
              "assert os.environ['PROFILE_RECORD_SIZE'] == '4096'\n"
              "assert os.environ['PROFILE_BOOK_KEEPING'] == '1'\n"
              f"{pythonpath_check}"
              "Path('.pkl_memoize_py3').mkdir()\n"
              "profile = Path('cdm_profile_data_dev0-0')\n"
              "profile.mkdir()\n"
              f"{profile_write}")
    return [sys.executable, "-c", source]


def _pcie_profile_environment(extra=None):
    environment = {
        "TILELANG_TPU_ALLOW_PCIE_LOAD": "1",
        "TILELANG_TPU_ALLOW_PCIE_PROFILE": "1",
        "TILELANG_TPU_DEVICE_ID": "0",
    }
    if extra:
        environment.update(extra)
    return environment


def _create_fake_pcie_decoder_packages(root: Path) -> Path:
    package_root = root / "vendor-python"
    big_profile = package_root / "bigTpuProfile"
    big_profile.mkdir(parents=True)
    (big_profile / "__init__.py").write_text(
        "import os\n"
        "assert 'TILELANG_TPU_ALLOW_PCIE_LOAD' not in os.environ\n"
        "assert 'TILELANG_TPU_ALLOW_PCIE_PROFILE' not in os.environ\n"
        "assert 'TILELANG_TPU_DEVICE_ID' not in os.environ\n"
        "assert 'BMLIB_ENABLE_ALL_PROFILE' not in os.environ\n"
        "__version__ = '0.2.0'\n",
        encoding="utf-8",
    )
    (big_profile / "bmprofile_perfAI_2260.py").write_text(
        "from types import SimpleNamespace\n"
        "class BMProfileParserPerfAI:\n"
        "    def parse(self, path):\n"
        "        record = ({\"Function Name\": \"rvt_fadd\", "
        "\"Start Time(ns)\": 5, \"End Time(ns)\": 12, \"Cmd Id\": 3}, "
        "None, {\"Core Id\": 0})\n"
        "        return SimpleNamespace(bd_events=[[record]], gdma_events=[], "
        "sdma_events=[], cdma_events=[])\n",
        encoding="utf-8",
    )
    return package_root


def _create_fake_perfai(root: Path, expected_chip: str = "bm1690e", sleep_s: float = 0.0) -> Path:
    perfai_root = root / "PerfAI"
    perfai_root.mkdir()
    runner = perfai_root / "AutoRunner.sh"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "while [ $# -gt 0 ]; do\n"
        "  case \"$1\" in\n"
        "    -d) run_dir=$2; shift 2 ;;\n"
        "    -e) chip=$2; shift 2 ;;\n"
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
        f"test \"${{chip}}\" = {expected_chip}\n"
        "test \"${TILELANG_PROFILE_TEST_TOKEN:-}\" = inherited\n"
        f"sleep {sleep_s!r}\n"
        "mkdir -p \"${run_dir}/result_profiling/output/PerfWeb\"\n"
        "cat > \"${run_dir}/result_profiling/output/PerfWeb/profile_data.js\" <<'EOF'\n"
        "let categories = [\"cpu\", \"bdc\", \"gdma\"];\n"
        "let time_header = [\"engine\", \"begin_us\", \"end_us\", \"type\", \"quality\", \"instruction\"];\n"
        "let time_data = [[0, 1, 2, 0, 1, \"host_call\"], [1, 10, 13, 4, 1, \"tiu_mul\"], [2, 11, 15, 0, 1, \"dma_load\"]];\n"
        "EOF\n",
        encoding="utf-8",
    )
    runner.chmod(runner.stat().st_mode | stat.S_IXUSR)
    return perfai_root


def test_cmodel_profile_worker_uses_a_private_cwd_and_keeps_raw_trace(tmp_path):
    output_dir = tmp_path / "profile"
    config = TPUProfilingConfig(
        chip="bm1690e", output_dir=output_dir, label="unit-trace", postprocess=False)

    report = run_tpu_cmodel_profile(
        _fake_trace_worker(config.label),
        config,
        environment={
            "TILELANG_TPU_ALLOW_PCIE_LOAD": "1",
            "TILELANG_TPU_ALLOW_PCIE_PROFILE": "1",
            "TILELANG_TPU_DEVICE_ID": "0",
        },
    )

    assert report.output_dir.parent == output_dir.resolve()
    assert report.parser_status == "not-requested"
    assert [path.name for path in report.raw_trace_files] == [
        "unit-trace-0-0.BD.0",
        "unit-trace-0-0.BD.0.txt",
        "unit-trace-0-0.GDMA.0",
        "unit-trace-0-0.GDMA.0.txt",
    ]
    assert report.stdout_path.is_file()
    assert report.stderr_path.is_file()
    assert not (report.output_dir / ".pkl_memoize_py3").exists()
    assert [(item.engine, item.core_id, item.command_id, item.opcode)
            for item in report.raw_instructions] == [
                ("bd", 0, 7, "15"),
                ("gdma", 0, 9, "6"),
            ]


def test_default_profile_output_is_discoverable_and_not_system_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = TPUProfilingConfig(chip="bm1690e", label="local-output", postprocess=False)

    report = TPUInstructionProfiler(config).run_cmodel(_fake_trace_worker(config.label))

    assert report.output_dir.parent == (tmp_path / "tilelang-tpu-profiles").resolve()


def test_cmodel_profile_runs_explicit_perfai_and_returns_instruction_durations(tmp_path):
    perfai_root = _create_fake_perfai(tmp_path)
    config = TPUProfilingConfig(
        chip="bm1690e",
        output_dir=tmp_path / "profile",
        label="unit-trace",
        perfai_root=perfai_root,
    )

    report = TPUInstructionProfiler(config).run_cmodel(
        _fake_trace_worker(config.label),
        environment={"TILELANG_PROFILE_TEST_TOKEN": "inherited"},
    )

    assert report.parser_status == "ready"
    assert report.perfai_report_path is not None
    assert len(report.timeline_events) == 3
    assert [(item.engine, item.duration, item.unit, item.fields["instruction"])
            for item in report.instruction_timings] == [
                ("bdc", 3.0, "us", "tiu_mul"),
                ("gdma", 4.0, "us", "dma_load"),
            ]


def test_profile_parser_accepts_multiline_perfai_timeline(tmp_path):
    profile_data = tmp_path / "profile_data.js"
    profile_data.write_text(
        "const categories = [\n  \"bdc\",\n];\n"
        "let time_header = [\"engine\", \"start_cycle\", \"end_cycle\"];\n"
        "var time_data = [\n  [0, 4, 9],\n];\n",
        encoding="utf-8",
    )

    timings = parse_perfai_instruction_timings(profile_data)

    assert len(timings) == 1
    assert timings[0].engine == "bdc"
    assert timings[0].duration == 5.0
    assert timings[0].unit == "cycles"


def test_profile_parser_keeps_host_events_out_of_instruction_timings(tmp_path):
    profile_data = tmp_path / "profile_data.js"
    profile_data.write_text(
        "let categories = [\"cpu\", \"bdc\"];\n"
        "let time_header = [\"engine\", \"begin_us\", \"end_us\", \"func_type\", \"info\"];\n"
        "let time_data = [[0, 0, 8, \"host_call\", \"host\"], [1, 2, 7, \"bd_id=3\", \"tiu_mul<br>cycle=5\"]];\n",
        encoding="utf-8",
    )

    timeline = parse_perfai_timeline_events(profile_data)
    timings = parse_perfai_instruction_timings(profile_data)

    assert len(timeline) == 2
    assert [(item.engine, item.command_id, item.opcode, item.duration) for item in timings
           ] == [("bdc", 3, "tiu_mul", 5.0)]


def test_rv_profile_without_sdk_uses_the_canonical_decoder_name(tmp_path):
    perfai_root = _create_fake_perfai(tmp_path, expected_chip="bm1690erv")
    config = TPUProfilingConfig(
        chip="bm1690e",
        programming_model="rv",
        output_dir=tmp_path / "profile",
        label="unit-trace",
        perfai_root=perfai_root,
    )

    report = TPUInstructionProfiler(config).run_cmodel(
        _fake_trace_worker(config.label, programming_model="rv"),
        environment={"TILELANG_PROFILE_TEST_TOKEN": "inherited"},
    )

    assert report.parser_status == "ready"


@pytest.mark.parametrize("model,expected", [
    ("tpukernel", "vendor_decoder_e"),
    ("rv", "vendor_decoder_rv"),
])
def test_perfai_decoder_alias_is_resolved_from_sdk_architecture(tmp_path, model, expected):
    from tilelang.jit.adapter.tpu_profiling import _perfai_chip_name

    mapping = tmp_path / "deps/chip/chip_map.json"
    mapping.parent.mkdir(parents=True)
    mapping.write_text(json.dumps({
        "vendor_decoder_e": "tpub_7_1_e",
        "vendor_decoder_rv": "tpub_7_1_e_rv",
    }))
    config = TPUProfilingConfig(chip="BM1690E", programming_model=model)

    assert _perfai_chip_name(config, {"PPL_PROJECT_ROOT": str(tmp_path)}) == expected
    assert config.chip == "bm1690e"


def test_profile_config_rejects_invalid_chip_model_and_nonfinite_timeout():
    with pytest.raises(ValueError, match="does not support programming model.*rv"):
        TPUProfilingConfig(chip="bm1690", programming_model="rv")
    with pytest.raises(ValueError, match="Unsupported TPU programming model"):
        TPUProfilingConfig(chip="bm1690e", programming_model="legacy")
    with pytest.raises(ValueError, match="positive number"):
        TPUProfilingConfig(chip="bm1690e", timeout_s=float("nan"))
    with pytest.raises(ValueError, match="positive number"):
        TPUProfilingConfig(chip="bm1690e", timeout_s=float("inf"))
    config = TPUProfilingConfig(chip="bm1690e", programming_model="tpukernel")
    assert config.programming_model == "tpukernel"
    assert config.target_spec == TPUTargetSpec("bm1690e", "tpukernel")
    assert config.target_spec.chip_spec.physical_core_count == 4
    assert config.runtime_config == TPURuntimeConfig("cmodel")


@pytest.mark.parametrize("runtime_mode", ("cmodel", "pcie"))
def test_profiler_preflights_only_the_selected_runtime_contract(monkeypatch, runtime_mode):
    calls = []

    class FakeLayout:

        def require_profiling(self, selected_runtime, *, environment):
            calls.append((selected_runtime, environment["PPL_PROJECT_ROOT"]))

    def fake_resolve(root, chip):
        calls.append((root, chip))
        return FakeLayout()

    monkeypatch.setattr(tpu_profiling_module, "resolve_ppl_layout", fake_resolve)
    profiler = TPUInstructionProfiler(TPUProfilingConfig(chip="bm1690e", runtime_mode=runtime_mode))

    profiler._validate_ppl_dependencies({"PPL_PROJECT_ROOT": "/ppl-1.7"})

    assert calls == [
        ("/ppl-1.7", "bm1690e"),
        (runtime_mode, "/ppl-1.7"),
    ]


def test_profiler_allows_generic_parser_worker_without_a_ppl_sdk(monkeypatch):
    monkeypatch.setattr(
        tpu_profiling_module,
        "resolve_ppl_layout",
        lambda *_args: pytest.fail("generic worker must not resolve PPL"),
    )
    profiler = TPUInstructionProfiler(TPUProfilingConfig(chip="bm1690e"))

    profiler._validate_ppl_dependencies({})


def test_cmodel_profile_timeout_terminates_the_worker_process_group(tmp_path):
    config = TPUProfilingConfig(
        chip="bm1690e", output_dir=tmp_path, timeout_s=0.1, postprocess=False)
    command = [
        sys.executable,
        "-c",
        "import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "time.sleep(30)",
    ]

    with pytest.raises(TPUProfilingTimeoutError, match="process group was terminated"):
        TPUInstructionProfiler(config).run_cmodel(command)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="process-group descendant cleanup is a Linux TPU safety contract",
)
@pytest.mark.parametrize("runtime_mode", ("cmodel", "pcie"))
def test_profile_rejects_success_with_lingering_descendant(tmp_path, monkeypatch, runtime_mode):
    """A successful supervisor leader may not leave an ordinary child alive."""

    child_pid_path = tmp_path / f"{runtime_mode}-lingering-child-pid.txt"
    child_ready_path = tmp_path / f"{runtime_mode}-lingering-child-ready.txt"
    child_source = ("import os, signal, sys, time\n"
                    "from pathlib import Path\n"
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                    "Path(sys.argv[1]).write_text(str(os.getpid()))\n"
                    "time.sleep(30)\n")
    worker_source = ("import os, subprocess, sys, time\n"
                     "from pathlib import Path\n"
                     f"child_source = {child_source!r}\n"
                     "child = subprocess.Popen(\n"
                     "    [sys.executable, '-c', child_source, "
                     "os.environ['TILELANG_TEST_CHILD_READY']],\n"
                     "    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
                     "ready = Path(os.environ['TILELANG_TEST_CHILD_READY'])\n"
                     "while not ready.is_file():\n"
                     "    time.sleep(0.01)\n"
                     "Path(os.environ['TILELANG_TEST_CHILD_PID']).write_text(str(child.pid))\n"
                     "if os.environ['TILELANG_TPU_PROFILE_RUNTIME_MODE'] == 'cmodel':\n"
                     "    label = os.environ['FILE_DUMP_CMD']\n"
                     "    Path(label + '-0-0.BD.0').write_bytes(b'raw-bd')\n"
                     "else:\n"
                     "    profile = Path('cdm_profile_data_dev0-0')\n"
                     "    profile.mkdir()\n"
                     "    (profile / 'global.profile').write_bytes(b'raw-pcie')\n")
    environment = os.environ.copy()
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "TILELANG_TEST_CHILD_PID": str(child_pid_path),
        "TILELANG_TEST_CHILD_READY": str(child_ready_path),
    })
    if runtime_mode == "pcie":
        environment.update(_pcie_profile_environment())
    monkeypatch.setattr(tpu_profiling_module, "_PROCESS_TERMINATION_GRACE_S", 0.05)
    monkeypatch.setattr(tpu_profiling_module, "_PROCESS_KILL_GRACE_S", 0.5)

    config = TPUProfilingConfig(
        chip="bm1690e",
        runtime_mode=runtime_mode,
        output_dir=tmp_path / f"{runtime_mode}-profile",
        timeout_s=3.0,
        postprocess=False,
    )
    profiler = TPUInstructionProfiler(config)
    run = profiler.run_cmodel if runtime_mode == "cmodel" else profiler.run_pcie

    child_pid = None
    try:
        started = time.monotonic()
        with pytest.raises(TPUProfilingCommandError, match="leaving a live descendant"):
            run([sys.executable, "-c", worker_source], environment=environment)
        elapsed = time.monotonic() - started
        _wait_for_file(child_pid_path)
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))

        assert elapsed < 2.0
        deadline = time.monotonic() + 2.0
        while _pid_is_running(child_pid):
            if time.monotonic() >= deadline:
                raise AssertionError("successful TPU profile worker left its child running")
            time.sleep(0.02)
    finally:
        if child_pid is not None and _pid_is_running(child_pid):
            with suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)


def test_kill_and_drain_remains_bounded_when_process_cannot_be_reaped(monkeypatch):

    class Pipe:
        closed = False

        def close(self):
            self.closed = True

    class StuckProcess:
        pid = 123456789

        def __init__(self):
            self.stdout = Pipe()
            self.stderr = Pipe()

        def poll(self):
            return None

        def wait(self, timeout):
            raise subprocess.TimeoutExpired("stuck", timeout)

        def communicate(self, timeout):
            raise subprocess.TimeoutExpired(
                "stuck", timeout, output="partial stdout", stderr="partial stderr")

    signals = []
    monkeypatch.setattr(tpu_profiling_module.os, "killpg", lambda pid, signum: signals.append(
        (pid, signum)))
    monkeypatch.setattr(tpu_profiling_module, "_PROCESS_TERMINATION_GRACE_S", 0.001)
    monkeypatch.setattr(tpu_profiling_module, "_PROCESS_KILL_GRACE_S", 0.001)
    monkeypatch.setattr(tpu_profiling_module, "_PROCESS_PIPE_DRAIN_S", 0.001)

    process = StuckProcess()
    terminated = tpu_profiling_module._terminate_and_collect(process)

    assert terminated.stdout == "partial stdout"
    assert "partial stderr" in terminated.stderr
    assert "did not close its output pipes" in terminated.stderr
    assert "could not be reaped" in terminated.stderr
    assert terminated.cleanup_complete is False
    assert terminated.left_live_descendant is True
    assert terminated.process_group == process.pid
    assert [item for item in signals if item[1] != 0] == [
        (process.pid, signal.SIGTERM),
        (process.pid, signal.SIGKILL),
    ]
    assert (process.pid, 0) in signals
    assert process.stdout.closed and process.stderr.closed


def test_profile_deadline_covers_worker_and_perfai_runtime(tmp_path):
    """PerfAI gets only the budget left after the CModel profile worker."""

    perfai_root = _create_fake_perfai(tmp_path, sleep_s=30)
    config = TPUProfilingConfig(
        chip="bm1690e",
        output_dir=tmp_path / "profile",
        label="deadline",
        perfai_root=perfai_root,
        timeout_s=1.0,
    )

    started = time.monotonic()
    report = TPUInstructionProfiler(config).run_cmodel(
        _fake_trace_worker(config.label, sleep_s=0.4),
        environment={"TILELANG_PROFILE_TEST_TOKEN": "inherited"},
    )
    elapsed = time.monotonic() - started

    assert report.parser_status == "timed-out"
    # A pre-fix implementation allowed the worker's 1s timeout and a new 1s
    # AutoRunner timeout to add. Give process startup margin, but require one
    # shared 1s budget rather than roughly 1.4s here.
    assert elapsed < 1.25


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="PerfAI lock timeout uses a Linux abstract socket",
)
def test_profile_deadline_covers_perfai_lock_wait(tmp_path):
    """A contended mutable PerfAI workspace cannot add a second timeout."""

    perfai_root = _create_fake_perfai(tmp_path)
    token = hashlib.sha256(str(perfai_root.resolve()).encode("utf-8")).hexdigest()[:20]
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            ("import socket, sys, time; "
             "handle = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM); "
             "handle.bind('\\0tilelang-perfai-' + sys.argv[1]); "
             "print('locked', flush=True); time.sleep(30)"),
            token,
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"
        config = TPUProfilingConfig(
            chip="bm1690e",
            output_dir=tmp_path / "profile",
            label="lock-deadline",
            perfai_root=perfai_root,
            timeout_s=1.0,
        )
        started = time.monotonic()
        report = TPUInstructionProfiler(config).run_cmodel(
            _fake_trace_worker(config.label, sleep_s=0.4),
            environment={"TILELANG_PROFILE_TEST_TOKEN": "inherited"},
        )
        elapsed = time.monotonic() - started
    finally:
        if holder.poll() is None:
            holder.terminate()
        holder.wait(timeout=5)

    assert report.parser_status == "busy"
    assert elapsed < 1.25


def _wait_for_file(path: Path, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise AssertionError(f"Timed out waiting for {path}")
        time.sleep(0.02)


def _pid_is_running(pid: int) -> bool:
    """Treat a reparented zombie as stopped for the no-orphan assertion."""

    stat_path = Path(f"/proc/{pid}/stat")
    try:
        fields = stat_path.read_text(encoding="utf-8").split()
    except (FileNotFoundError, ProcessLookupError):
        # procfs may remove the entry between open and read, reporting either
        # FileNotFoundError or ProcessLookupError depending on the exact race.
        return False
    return len(fields) > 2 and fields[2] != "Z"


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="parent-death guard is a Linux TPU profiling safety contract",
)
def test_parent_death_guard_kills_worker_and_ordinary_descendant(tmp_path):
    """Killing the profiler parent must not orphan its detached profile group."""

    worker_pids_path = tmp_path / "worker-pids.txt"
    supervisor_pid_path = tmp_path / "supervisor-pid.txt"
    supervisor_path = (
        Path(__file__).resolve().parents[3] / "tilelang" / "jit" / "_tpu_profile_supervisor.py")
    target_source = (
        "import os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "grandchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "Path(sys.argv[1]).write_text(f'{os.getpid()} {grandchild.pid} {os.getpgrp()}')\n"
        "time.sleep(30)\n")
    controller_source = (
        "import os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        f"target_source = {target_source!r}\n"
        "supervisor = subprocess.Popen([\n"
        "    sys.executable, sys.argv[1], '--parent-pid', str(os.getpid()), '--',\n"
        "    sys.executable, '-c', target_source, sys.argv[2],\n"
        "], start_new_session=True)\n"
        "Path(sys.argv[3]).write_text(str(supervisor.pid))\n"
        "time.sleep(30)\n")
    controller = subprocess.Popen([
        sys.executable,
        "-c",
        controller_source,
        str(supervisor_path),
        str(worker_pids_path),
        str(supervisor_pid_path),
    ])
    supervisor_pid = None
    try:
        _wait_for_file(supervisor_pid_path)
        _wait_for_file(worker_pids_path)
        supervisor_pid = int(supervisor_pid_path.read_text(encoding="utf-8"))
        worker_pid, descendant_pid, worker_pgid = map(
            int,
            worker_pids_path.read_text(encoding="utf-8").split())
        assert worker_pgid == supervisor_pid
        assert os.getpgid(descendant_pid) == supervisor_pid

        os.kill(controller.pid, signal.SIGKILL)
        controller.wait(timeout=5)
        deadline = time.monotonic() + 5
        while _pid_is_running(worker_pid) or _pid_is_running(descendant_pid):
            if time.monotonic() >= deadline:
                raise AssertionError("parent-death guard left a profile process running")
            time.sleep(0.02)
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait(timeout=5)
        if supervisor_pid is not None and _pid_is_running(supervisor_pid):
            with suppress(ProcessLookupError):
                os.killpg(supervisor_pid, signal.SIGKILL)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="parent-death guard is a Linux TPU numerical-runner safety contract",
)
def test_tpukernel_matrix_parent_death_guard_prevents_orphans(tmp_path):
    """Killing the numerical matrix runner must kill its worker tree."""

    worker_pids_path = tmp_path / "numeric-worker-pids.txt"
    supervisor_pid_path = tmp_path / "numeric-supervisor-pid.txt"
    matrix_dir = Path(__file__).resolve().parent
    target_source = (
        "import os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "grandchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "Path(sys.argv[1]).write_text(f'{os.getpid()} {grandchild.pid} {os.getpgrp()}')\n"
        "time.sleep(30)\n")
    controller_source = ("import os, sys, time\n"
                         "from pathlib import Path\n"
                         "sys.path.insert(0, sys.argv[1])\n"
                         "from tpukernel_ops_matrix import _spawn_guarded_worker\n"
                         f"target_source = {target_source!r}\n"
                         "supervisor = _spawn_guarded_worker(\n"
                         "    [sys.executable, '-c', target_source, sys.argv[2]],\n"
                         "    cwd=Path(sys.argv[4]), environment=os.environ.copy())\n"
                         "Path(sys.argv[3]).write_text(str(supervisor.pid))\n"
                         "time.sleep(30)\n")
    controller_env = os.environ.copy()
    controller_env["PYTHONDONTWRITEBYTECODE"] = "1"
    controller = subprocess.Popen([
        sys.executable,
        "-c",
        controller_source,
        str(matrix_dir),
        str(worker_pids_path),
        str(supervisor_pid_path),
        str(tmp_path),
    ],
                                  env=controller_env)
    supervisor_pid = None
    try:
        _wait_for_file(supervisor_pid_path)
        _wait_for_file(worker_pids_path)
        supervisor_pid = int(supervisor_pid_path.read_text(encoding="utf-8"))
        worker_pid, descendant_pid, worker_pgid = map(
            int,
            worker_pids_path.read_text(encoding="utf-8").split())
        assert worker_pgid == supervisor_pid
        assert os.getpgid(descendant_pid) == supervisor_pid

        os.kill(controller.pid, signal.SIGKILL)
        controller.wait(timeout=5)
        deadline = time.monotonic() + 5
        while any(_pid_is_running(pid) for pid in (supervisor_pid, worker_pid, descendant_pid)):
            if time.monotonic() >= deadline:
                raise AssertionError("parent-death guard left a TPU numerical process running")
            time.sleep(0.02)
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait(timeout=5)
        if supervisor_pid is not None and _pid_is_running(supervisor_pid):
            with suppress(ProcessLookupError):
                os.killpg(supervisor_pid, signal.SIGKILL)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="process-group timeout supervision is a Linux TPU safety contract",
)
def test_tpukernel_matrix_timeout_has_bounded_pipe_drain(tmp_path, monkeypatch):
    """An escaped pipe holder cannot turn a numerical timeout into a hang."""

    # Exercise the outer process-group supervisor without a PID namespace:
    # the namespace would itself reap the escaped child, masking this case.
    from tilelang.jit import tpu_cmodel
    monkeypatch.setattr(tpu_cmodel, "prepare_cmodel_command",
                        lambda command, *, cwd, environment, private_tmp_dir: (
                            list(command), dict(environment)))

    matrix_dir = Path(__file__).resolve().parent
    monkeypatch.syspath_prepend(str(matrix_dir))
    matrix_module = importlib.import_module("tpukernel_ops_matrix")
    monkeypatch.setattr(matrix_module, "_PROCESS_PIPE_DRAIN_S", 0.1)

    pid_path = tmp_path / "timeout-worker-pids.txt"
    worker_path = tmp_path / "timeout-worker.py"
    worker_path.write_text(
        "import os, signal, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "ignore_term = 'import signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'\n"
        "controlled = subprocess.Popen([sys.executable, '-c', ignore_term])\n"
        "escaped = subprocess.Popen(\n"
        "    [sys.executable, '-c', ignore_term], start_new_session=True)\n"
        "Path(os.environ['TILELANG_TEST_TIMEOUT_PIDS']).write_text(\n"
        "    f'{os.getpid()} {controlled.pid} {escaped.pid}')\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["TILELANG_TEST_TIMEOUT_PIDS"] = str(pid_path)
    monkeypatch.setattr(
        matrix_module,
        "_worker_environment",
        lambda *_args, **_kwargs: environment,
    )
    args = SimpleNamespace(runtime_mode="cmodel", timeout=1.0, kill_grace=0.1)
    case = matrix_module.CaseSpec("timeout-pipe", "copy", "float32")

    escaped_pid = None
    try:
        started = time.monotonic()
        result = matrix_module._run_one(
            args,
            Path(__file__).resolve().parents[3],
            tmp_path / "matrix",
            worker_path,
            "bm1690e",
            case,
        )
        elapsed = time.monotonic() - started
        _wait_for_file(pid_path)
        worker_pid, controlled_pid, escaped_pid = map(int,
                                                      pid_path.read_text(encoding="utf-8").split())

        assert elapsed < 2.5
        assert result["timed_out"] is True
        assert result["status"] == "failed"
        assert result["termination"]["pipe_drain_timed_out"] is True
        assert "did not close its output pipes" in result["stderr"]
        deadline = time.monotonic() + 2
        while _pid_is_running(worker_pid) or _pid_is_running(controlled_pid):
            if time.monotonic() >= deadline:
                raise AssertionError("numerical timeout left an in-group worker process running")
            time.sleep(0.02)
        # The deliberate start_new_session descendant is outside the private
        # group and keeps the inherited pipes open, which is what exercises the
        # bounded drain.  It is not claimed as an ordinarily supervised child.
        assert _pid_is_running(escaped_pid)
    finally:
        if escaped_pid is not None and _pid_is_running(escaped_pid):
            with suppress(ProcessLookupError):
                os.killpg(escaped_pid, signal.SIGKILL)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="process-group descendant cleanup is a Linux TPU safety contract",
)
def test_tpukernel_matrix_rejects_success_with_lingering_descendant(tmp_path, monkeypatch):
    """A nominally successful worker may not background an in-group child."""

    # Keep host PIDs visible to this process-supervisor unit test. Actual IPC
    # namespace cleanup is tested separately in test_tpu_cmodel.py.
    from tilelang.jit import tpu_cmodel
    monkeypatch.setattr(tpu_cmodel, "prepare_cmodel_command",
                        lambda command, *, cwd, environment, private_tmp_dir: (
                            list(command), dict(environment)))

    matrix_dir = Path(__file__).resolve().parent
    monkeypatch.syspath_prepend(str(matrix_dir))
    matrix_module = importlib.import_module("tpukernel_ops_matrix")

    child_pid_path = tmp_path / "lingering-child-pid.txt"
    child_ready_path = tmp_path / "lingering-child-ready.txt"
    worker_path = tmp_path / "lingering-worker.py"
    child_source = ("import os, signal, sys, time\n"
                    "from pathlib import Path\n"
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                    "Path(sys.argv[1]).write_text(str(os.getpid()))\n"
                    "time.sleep(30)\n")
    worker_path.write_text(
        "import json, os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        f"child_source = {child_source!r}\n"
        "child = subprocess.Popen(\n"
        "    [sys.executable, '-c', child_source, "
        "os.environ['TILELANG_TEST_CHILD_READY']],\n"
        "    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "ready = Path(os.environ['TILELANG_TEST_CHILD_READY'])\n"
        "while not ready.is_file():\n"
        "    time.sleep(0.01)\n"
        "Path(os.environ['TILELANG_TEST_CHILD_PID']).write_text(str(child.pid))\n"
        "print('TPUKERNEL_NUMERIC_RESULT=' + "
        "json.dumps({'status': 'passed'}), flush=True)\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["TILELANG_TEST_CHILD_PID"] = str(child_pid_path)
    environment["TILELANG_TEST_CHILD_READY"] = str(child_ready_path)
    monkeypatch.setattr(
        matrix_module,
        "_worker_environment",
        lambda *_args, **_kwargs: environment,
    )
    args = SimpleNamespace(runtime_mode="cmodel", timeout=3.0, kill_grace=0.1)
    case = matrix_module.CaseSpec("lingering-child", "copy", "float32")

    child_pid = None
    try:
        result = matrix_module._run_one(
            args,
            Path(__file__).resolve().parents[3],
            tmp_path / "matrix-lingering",
            worker_path,
            "bm1690e",
            case,
        )
        _wait_for_file(child_pid_path)
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))

        assert result["status"] == "failed"
        assert result["returncode"] == 0
        assert "left a live descendant" in result["failure"]
        assert result["termination"] is not None
        deadline = time.monotonic() + 2
        while _pid_is_running(child_pid):
            if time.monotonic() >= deadline:
                raise AssertionError("successful numerical worker left its child running")
            time.sleep(0.02)
    finally:
        if child_pid is not None and _pid_is_running(child_pid):
            with suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)


def test_pcie_profile_environment_requires_two_acknowledgements():
    profiler = TPUInstructionProfiler(TPUProfilingConfig(chip="bm1690e", runtime_mode="pcie"))

    with pytest.raises(TPUProfilingError, match="ALLOW_PCIE_LOAD"):
        profiler.pcie_profile_environment_overrides({})
    with pytest.raises(TPUProfilingError, match="ALLOW_PCIE_PROFILE"):
        profiler.pcie_profile_environment_overrides({"TILELANG_TPU_ALLOW_PCIE_LOAD": "1"})
    with pytest.raises(TPUProfilingError, match="DEVICE_ID"):
        profiler.pcie_profile_environment_overrides({
            "TILELANG_TPU_ALLOW_PCIE_LOAD": "1",
            "TILELANG_TPU_ALLOW_PCIE_PROFILE": "1",
            "TILELANG_TPU_DEVICE_ID": str(2**31),
        })

    environment = profiler.pcie_profile_environment_overrides({
        "TILELANG_TPU_ALLOW_PCIE_LOAD": "1",
        "TILELANG_TPU_ALLOW_PCIE_PROFILE": "1",
        "TILELANG_TPU_DEVICE_ID": "0",
    })
    assert environment == {
        "BMLIB_ENABLE_ALL_PROFILE": "1",
        "PROFILE_RECORD_SIZE": "4096",
        "PROFILE_BOOK_KEEPING": "1",
    }


def test_pcie_decoder_config_normalizes_only_explicit_pcie_paths(tmp_path):
    decoder_path = tmp_path / "decoder-python"
    decoder_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    decoder_path.chmod(decoder_path.stat().st_mode | stat.S_IXUSR)
    package_path = tmp_path / "packages"
    package_path.mkdir()

    config = TPUProfilingConfig(
        chip="bm1690e",
        runtime_mode="pcie",
        pcie_decoder_python=decoder_path,
        pcie_decoder_pythonpath=[package_path],
    )

    assert config.pcie_decoder_python == decoder_path.resolve()
    assert config.pcie_decoder_pythonpath == (package_path.resolve(),)
    with pytest.raises(ValueError, match="only valid"):
        TPUProfilingConfig(
            chip="bm1690e",
            runtime_mode="cmodel",
            pcie_decoder_pythonpath=(package_path,),
        )
    with pytest.raises(TypeError, match="sequence"):
        TPUProfilingConfig(
            chip="bm1690e",
            runtime_mode="pcie",
            pcie_decoder_pythonpath=package_path,
        )
    with pytest.raises(ValueError, match="not a directory or importable"):
        TPUProfilingConfig(
            chip="bm1690e",
            runtime_mode="pcie",
            pcie_decoder_pythonpath=(tmp_path / "missing",),
        )


def test_pcie_decoder_pythonpath_accepts_importable_archive(tmp_path):
    import zipfile

    archive = tmp_path / "decoder.body"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("decoder_package/__init__.py", "")

    config = TPUProfilingConfig(
        chip="bm1690e",
        runtime_mode="pcie",
        pcie_decoder_pythonpath=(archive,),
    )

    assert config.pcie_decoder_pythonpath == (archive.resolve(),)


def test_pcie_decoder_python_preserves_virtualenv_symlink(tmp_path):
    interpreter = tmp_path / "python-real"
    interpreter.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    interpreter.chmod(interpreter.stat().st_mode | stat.S_IXUSR)
    virtualenv_launcher = tmp_path / "venv-python"
    virtualenv_launcher.symlink_to(interpreter)

    config = TPUProfilingConfig(
        chip="bm1690e",
        runtime_mode="pcie",
        pcie_decoder_python=virtualenv_launcher,
    )

    assert config.pcie_decoder_python == virtualenv_launcher.absolute()
    assert config.pcie_decoder_python != interpreter.resolve()


def test_pcie_decoder_preflight_is_offline_and_returns_api_identity(tmp_path):
    package_root = _create_fake_pcie_decoder_packages(tmp_path)
    config = TPUProfilingConfig(
        chip="bm1690e",
        runtime_mode="pcie",
        pcie_decoder_python=sys.executable,
        pcie_decoder_pythonpath=(package_root,),
    )

    identity = TPUInstructionProfiler(config).preflight_pcie_decoder(
        environment=_pcie_profile_environment())

    assert identity == {
        "package": "bigTpuProfile",
        "package_version": "0.2.0",
        "parser_api": "bigTpuProfile.bmprofile_perfAI_2260.BMProfileParserPerfAI.parse",
    }


def test_pcie_decoder_preflight_reports_only_the_required_package(tmp_path, monkeypatch):
    missing_decoder = tmp_path / "missing_decoder.py"
    missing_decoder.write_text("raise SystemExit(3)\n", encoding="utf-8")
    monkeypatch.setattr(tpu_profiling_module, "_PCIE_PROFILE_DECODER_PATH", missing_decoder)
    profiler = TPUInstructionProfiler(TPUProfilingConfig(chip="bm1690e", runtime_mode="pcie"))

    with pytest.raises(TPUProfilingError, match="preinstalled bigTpuProfile") as exc_info:
        profiler.preflight_pcie_decoder(environment=_pcie_profile_environment())

    assert "PerfAI" not in str(exc_info.value)


def test_pcie_profile_worker_isolated_and_keeps_raw_trace(tmp_path):
    config = TPUProfilingConfig(
        chip="bm1690e",
        programming_model="rv",
        runtime_mode="pcie",
        output_dir=tmp_path / "profile",
        postprocess=False,
    )

    report = run_tpu_pcie_profile(
        _fake_pcie_trace_worker("rv"),
        config,
        environment=_pcie_profile_environment(),
    )

    assert report.parser_status == "not-requested"
    assert [path.name for path in report.raw_trace_files] == ["cdm_profile_data_dev0-0"]
    assert report.raw_instructions == ()
    assert not (report.output_dir / ".pkl_memoize_py3").exists()


@pytest.mark.parametrize(
    "profile_name,profile_payload",
    (
        (None, b""),
        ("global.profile", b""),
        ("unexpected.bin", b"raw-pcie"),
    ),
)
def test_pcie_profile_rejects_empty_or_unrecognized_raw_artifacts(tmp_path, profile_name,
                                                                  profile_payload):
    config = TPUProfilingConfig(
        chip="bm1690e",
        runtime_mode="pcie",
        output_dir=tmp_path / "profile",
        postprocess=False,
    )

    report = TPUInstructionProfiler(config).run_pcie(
        _fake_pcie_trace_worker(profile_name=profile_name, profile_payload=profile_payload),
        environment=_pcie_profile_environment(),
    )

    assert report.raw_trace_files == ()
    assert report.has_raw_trace is False


def test_pcie_profile_requires_all_gates_before_spawning(tmp_path):
    profiler = TPUInstructionProfiler(
        TPUProfilingConfig(chip="bm1690e", runtime_mode="pcie", output_dir=tmp_path))

    with pytest.raises(TPUProfilingError, match="ALLOW_PCIE_PROFILE"):
        profiler.run_pcie(
            [sys.executable, "-c", "raise SystemExit(99)"],
            environment={
                "TILELANG_TPU_ALLOW_PCIE_LOAD": "1",
                "TILELANG_TPU_DEVICE_ID": "0",
            },
        )


def test_pcie_device_lock_is_session_wide_reentrant_and_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(tpu_profiling_module, "_PCIE_DEVICE_LOCK_ROOT", tmp_path)
    lock_path = tmp_path / "tilelang-tpu-device-17.lock"
    session_path = tmp_path / "tilelang-tpu-device-17.session.json"

    with TPUInstructionProfiler.exclusive_pcie_device(17) as outer:
        assert outer == lock_path
        session = json.loads(session_path.read_text(encoding="utf-8"))
        assert session["status"] == "active"
        assert session["device_id"] == 17
        with TPUInstructionProfiler.exclusive_pcie_device(17) as nested:
            assert nested == lock_path
            assert session_path.is_file()
        with lock_path.open("a+", encoding="utf-8") as competitor, \
                pytest.raises(BlockingIOError):
            fcntl.flock(competitor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    assert not session_path.exists()
    with lock_path.open("a+", encoding="utf-8") as available:
        fcntl.flock(available.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(available.fileno(), fcntl.LOCK_UN)


def test_pcie_device_lock_rejects_crash_persistent_prior_session(tmp_path, monkeypatch):
    monkeypatch.setattr(tpu_profiling_module, "_PCIE_DEVICE_LOCK_ROOT", tmp_path)
    marker = tmp_path / "tilelang-tpu-device-8.session.json"
    marker.write_text(
        json.dumps({
            "status": "active",
            "device_id": 8,
            "owner_pid": 1234
        }),
        encoding="utf-8",
    )

    with pytest.raises(TPUProfilingError, match="incomplete prior session"), \
            TPUInstructionProfiler.exclusive_pcie_device(8):
        pass

    assert marker.is_file()


def test_pcie_device_quarantine_is_persistent_and_never_auto_cleared(tmp_path, monkeypatch):
    monkeypatch.setattr(tpu_profiling_module, "_PCIE_DEVICE_LOCK_ROOT", tmp_path)
    marker = tmp_path / "tilelang-tpu-device-3.quarantine.json"

    with TPUInstructionProfiler.exclusive_pcie_device(3):
        observed = TPUInstructionProfiler.fail_closed_pcie_device(
            3, process_group=8765, reason="unit-test unreaped process group")
        assert observed == marker
        with pytest.raises(TPUProfilingError, match="became unsafe"), \
                TPUInstructionProfiler.exclusive_pcie_device(3):
            pass

    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["status"] == "quarantined"
    assert payload["device_id"] == 3
    assert payload["process_group"] == 8765
    assert payload["process_group_known"] is True
    with pytest.raises(TPUProfilingError, match="fail-closed"), \
            TPUInstructionProfiler.exclusive_pcie_device(3):
        pass


def test_pcie_device_fail_closed_without_observed_process_group(tmp_path, monkeypatch):
    monkeypatch.setattr(tpu_profiling_module, "_PCIE_DEVICE_LOCK_ROOT", tmp_path)
    quarantine = tmp_path / "tilelang-tpu-device-12.quarantine.json"
    session = tmp_path / "tilelang-tpu-device-12.session.json"

    with TPUInstructionProfiler.exclusive_pcie_device(12):
        observed = TPUInstructionProfiler.fail_closed_pcie_device(
            12, reason="postflight board idle state was not established")
        assert observed == quarantine

    payload = json.loads(quarantine.read_text(encoding="utf-8"))
    assert payload["reason"] == "postflight board idle state was not established"
    assert payload["process_group"] is None
    assert payload["process_group_known"] is False
    assert session.is_file()


def test_pcie_device_fail_closed_is_idempotent_inside_unsafe_session(tmp_path, monkeypatch):
    monkeypatch.setattr(tpu_profiling_module, "_PCIE_DEVICE_LOCK_ROOT", tmp_path)
    quarantine = tmp_path / "tilelang-tpu-device-18.quarantine.json"

    with TPUInstructionProfiler.exclusive_pcie_device(18):
        first = TPUInstructionProfiler.fail_closed_pcie_device(
            18, reason="probe process group was not reaped", process_group=4321)
        second = TPUInstructionProfiler.fail_closed_pcie_device(
            18, reason="postflight observation was incomplete")

    assert first == quarantine
    assert second == quarantine
    payload = json.loads(quarantine.read_text(encoding="utf-8"))
    assert payload["reason"] == "probe process group was not reaped"
    assert payload["process_group"] == 4321
    assert (tmp_path / "tilelang-tpu-device-18.session.json").is_file()


@pytest.mark.parametrize("reason", [None, "", " ", " leading", "trailing ", "line\nbreak", 17])
def test_pcie_device_fail_closed_rejects_unstable_reason_without_poisoning_session(
        tmp_path, monkeypatch, reason):
    monkeypatch.setattr(tpu_profiling_module, "_PCIE_DEVICE_LOCK_ROOT", tmp_path)
    session = tmp_path / "tilelang-tpu-device-13.session.json"
    quarantine = tmp_path / "tilelang-tpu-device-13.quarantine.json"

    with TPUInstructionProfiler.exclusive_pcie_device(13):
        with pytest.raises(ValueError, match="reason must be a non-empty printable string"):
            TPUInstructionProfiler.fail_closed_pcie_device(13, reason=reason)
        assert session.is_file()

    assert not session.exists()
    assert not quarantine.exists()


@pytest.mark.parametrize("process_group", [False, 0, -1, 1.5, "123"])
def test_pcie_device_fail_closed_rejects_invalid_process_group_without_poisoning_session(
        tmp_path, monkeypatch, process_group):
    monkeypatch.setattr(tpu_profiling_module, "_PCIE_DEVICE_LOCK_ROOT", tmp_path)

    with TPUInstructionProfiler.exclusive_pcie_device(14), \
            pytest.raises(ValueError, match="process_group must be a positive integer or None"):
        TPUInstructionProfiler.fail_closed_pcie_device(
            14, reason="postflight incomplete", process_group=process_group)

    assert not (tmp_path / "tilelang-tpu-device-14.session.json").exists()
    assert not (tmp_path / "tilelang-tpu-device-14.quarantine.json").exists()


def test_pcie_device_fail_closed_requires_owned_session(tmp_path, monkeypatch):
    monkeypatch.setattr(tpu_profiling_module, "_PCIE_DEVICE_LOCK_ROOT", tmp_path)

    with pytest.raises(TPUProfilingError, match="without owning its session lock"):
        TPUInstructionProfiler.fail_closed_pcie_device(
            15, reason="postflight board idle state was not established")


def test_pcie_quarantine_creation_failure_retains_active_session(tmp_path, monkeypatch):
    monkeypatch.setattr(tpu_profiling_module, "_PCIE_DEVICE_LOCK_ROOT", tmp_path)
    session = tmp_path / "tilelang-tpu-device-6.session.json"
    quarantine = tmp_path / "tilelang-tpu-device-6.quarantine.json"

    with TPUInstructionProfiler.exclusive_pcie_device(6):
        real_open = os.open

        def fail_quarantine_open(path, flags, mode=0o777, *, dir_fd=None):
            if Path(path) == quarantine:
                raise PermissionError("simulated read-only lock directory")
            if dir_fd is None:
                return real_open(path, flags, mode)
            return real_open(path, flags, mode, dir_fd=dir_fd)

        monkeypatch.setattr(tpu_profiling_module.os, "open", fail_quarantine_open)
        with pytest.raises(TPUProfilingError, match="session marker .* retained fail-closed"):
            TPUInstructionProfiler.fail_closed_pcie_device(
                6, process_group=999, reason="simulated incomplete cleanup")
        with pytest.raises(TPUProfilingError, match="became unsafe"), \
                TPUInstructionProfiler.exclusive_pcie_device(6):
            pass

    assert session.is_file()
    assert not quarantine.exists()
    with pytest.raises(TPUProfilingError, match="incomplete prior session"), \
            TPUInstructionProfiler.exclusive_pcie_device(6):
        pass


def test_pcie_fail_closed_keyboard_interrupt_retains_active_session(tmp_path, monkeypatch):
    monkeypatch.setattr(tpu_profiling_module, "_PCIE_DEVICE_LOCK_ROOT", tmp_path)
    session = tmp_path / "tilelang-tpu-device-16.session.json"
    quarantine = tmp_path / "tilelang-tpu-device-16.quarantine.json"
    real_open = os.open

    def interrupt_quarantine_open(path, flags, mode=0o777, *, dir_fd=None):
        if Path(path) == quarantine:
            raise KeyboardInterrupt
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(tpu_profiling_module.os, "open", interrupt_quarantine_open)
    with pytest.raises(KeyboardInterrupt), \
            TPUInstructionProfiler.exclusive_pcie_device(16):
        TPUInstructionProfiler.fail_closed_pcie_device(
            16, reason="postflight observation was interrupted")

    assert session.is_file()
    assert not quarantine.exists()
    with pytest.raises(TPUProfilingError, match="incomplete prior session"), \
            TPUInstructionProfiler.exclusive_pcie_device(16):
        pass


def test_pcie_probe_quarantines_incomplete_process_group_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(tpu_profiling_module, "_PCIE_DEVICE_LOCK_ROOT", tmp_path)
    incomplete = tpu_profiling_module.TPUSupervisedCommandResult(
        stdout="",
        stderr="stuck",
        returncode=None,
        timed_out=True,
        left_live_descendant=True,
        cleanup_complete=False,
        process_group=4567,
    )
    monkeypatch.setattr(tpu_profiling_module, "run_tpu_supervised_command",
                        lambda *args, **kwargs: incomplete)

    with TPUInstructionProfiler.exclusive_pcie_device(4), \
            pytest.raises(TPUProfilingError, match="quarantined"):
        TPUInstructionProfiler.run_supervised_pcie_probe(
            4, [sys.executable, "-c", "pass"], cwd=tmp_path)

    marker = tmp_path / "tilelang-tpu-device-4.quarantine.json"
    assert json.loads(marker.read_text(encoding="utf-8"))["process_group"] == 4567


def test_pcie_probe_requires_session_lock(tmp_path):
    with pytest.raises(TPUProfilingError, match="requires ownership"):
        TPUInstructionProfiler.run_supervised_pcie_probe(
            5, [sys.executable, "-c", "pass"], cwd=tmp_path)


@pytest.mark.parametrize("device_id", (-1, 2**31, True, "0"))
def test_pcie_device_lock_rejects_invalid_device_ids(device_id):
    with pytest.raises(ValueError, match="non-negative 32-bit"):
        TPUInstructionProfiler.exclusive_pcie_device(device_id)


def test_pcie_profile_decodes_with_preinstalled_vendor_packages(tmp_path):
    package_root = _create_fake_pcie_decoder_packages(tmp_path)
    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath = os.pathsep.join(item for item in (str(package_root), inherited_pythonpath) if item)
    config = TPUProfilingConfig(
        chip="bm1690e",
        programming_model="rv",
        runtime_mode="pcie",
        output_dir=tmp_path / "profile",
    )

    report = TPUInstructionProfiler(config).run_pcie(
        _fake_pcie_trace_worker("rv"),
        environment=_pcie_profile_environment({"PYTHONPATH": pythonpath}),
    )

    assert report.parser_status == "ready"
    assert len(report.decoded_report_paths) == 1
    assert report.perfai_report_paths == ()
    assert [
        (item.engine, item.duration, item.unit, item.opcode) for item in report.instruction_timings
    ] == [("bdc", 7.0, "ns", "rvt_fadd")]


def test_explicit_pcie_decoder_pythonpath_never_reaches_profile_worker(tmp_path):
    package_root = _create_fake_pcie_decoder_packages(tmp_path)
    config = TPUProfilingConfig(
        chip="bm1690e",
        programming_model="rv",
        runtime_mode="pcie",
        output_dir=tmp_path / "profile",
        pcie_decoder_python=sys.executable,
        pcie_decoder_pythonpath=(package_root,),
    )

    report = TPUInstructionProfiler(config).run_pcie(
        _fake_pcie_trace_worker("rv", forbidden_pythonpath=package_root.resolve()),
        environment=_pcie_profile_environment(),
    )

    assert report.parser_status == "ready"
    assert report.decoder_identity == {
        "package": "bigTpuProfile",
        "package_version": "0.2.0",
        "parser_api": "bigTpuProfile.bmprofile_perfAI_2260.BMProfileParserPerfAI.parse",
    }


def test_pcie_decoder_requires_canonical_tilelang_json(tmp_path, monkeypatch):
    decoder = tmp_path / "legacy_only_decoder.py"
    decoder.write_text(
        "from pathlib import Path\n"
        "Path('profile_data.js').write_text('let time_data = [];\\n')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(tpu_profiling_module, "_PCIE_PROFILE_DECODER_PATH", decoder)
    config = TPUProfilingConfig(
        chip="bm1690e",
        programming_model="rv",
        runtime_mode="pcie",
        output_dir=tmp_path / "profile",
    )

    report = TPUInstructionProfiler(config).run_pcie(
        _fake_pcie_trace_worker("rv"),
        environment=_pcie_profile_environment(),
    )

    assert report.parser_status == "missing-report"
    assert report.decoded_report_paths == ()
    assert report.perfai_report_paths == ()
    assert report.instruction_timings == ()


def test_pcie_canonical_decoder_report_rejects_an_unknown_schema(tmp_path):
    report_path = tmp_path / "tilelang_pcie_profile.json"
    report_path.write_text('{"schema_version": 2, "events": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported TileLang PCIe profile schema"):
        parse_pcie_decoded_instruction_timings(report_path)


@pytest.mark.parametrize(
    "event_update",
    (
        {
            "begin": True
        },
        {
            "begin": float("nan")
        },
        {
            "end": float("inf")
        },
        {
            "begin": 8,
            "end": 7
        },
        {
            "unit": ""
        },
        {
            "unit": "cycles"
        },
    ),
)
def test_pcie_canonical_decoder_report_rejects_invalid_timing(tmp_path, event_update):
    event = {
        "engine": "bdc",
        "begin": 2,
        "end": 7,
        "unit": "ns",
        "fields": {},
    }
    event.update(event_update)
    report_path = tmp_path / "tilelang_pcie_profile.json"
    report_path.write_text(
        json.dumps({
            "schema_version": 1,
            "events": [event]
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="PCIe profile event 0 is invalid"):
        parse_pcie_decoded_instruction_timings(report_path)


def test_cmodel_runner_rejects_a_pcie_configuration_before_spawning(tmp_path):
    profiler = TPUInstructionProfiler(
        TPUProfilingConfig(chip="bm1690e", runtime_mode="pcie", output_dir=tmp_path))

    with pytest.raises(TPUProfilingError, match="not dispatched"):
        profiler.run_cmodel([sys.executable, "-c", "raise SystemExit(0)"])


def _profile_worker_command(case: str):
    worker = Path(__file__).with_name("tpu_profile_worker.py")
    return [sys.executable, str(worker), "--case", case]


def _profile_worker_environment():
    """Keep the real worker self-contained after profiler changes its cwd."""

    repo_root = Path(__file__).resolve().parents[3]
    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath = os.pathsep.join(item for item in (str(repo_root), inherited_pythonpath) if item)
    return {
        "PPL_PROJECT_ROOT": os.environ["PPL_PROJECT_ROOT"],
        "PYTHONPATH": pythonpath,
    }


def _run_opt_in_cmodel_profile_or_abort(config: TPUProfilingConfig, case: str):
    """Stop this opt-in test session after a vendor-worker failure.

    The profiler has already stopped the worker tree.  Ending pytest prevents
    the next opt-in emulator case from running after a timeout, failed JIT
    worker, or other vendor-runtime error; the outer ``timeout`` remains the
    final guard if pytest itself becomes unresponsive.
    """

    try:
        return TPUInstructionProfiler(config).run_cmodel(
            _profile_worker_command(case),
            environment=_profile_worker_environment(),
        )
    except TPUProfilingError as exc:
        pytest.exit(
            f"Aborting opt-in TPU CModel profile session after {case} failed: {exc}",
            returncode=2,
        )


def _pcie_profile_worker_environment():
    environment = _profile_worker_environment()
    for name in ("TILELANG_TPU_ALLOW_PCIE_LOAD", "TILELANG_TPU_ALLOW_PCIE_PROFILE",
                 "TILELANG_TPU_DEVICE_ID"):
        value = os.environ.get(name)
        if value is None:
            pytest.skip(f"{name} is required for an opt-in PCIe profile")
        environment[name] = value
    return environment


def _run_opt_in_pcie_profile_or_abort(config: TPUProfilingConfig, case: str):
    """Stop the hardware test session after the first vendor-runtime failure."""

    try:
        return TPUInstructionProfiler(config).run_pcie(
            _profile_worker_command(case),
            environment=_pcie_profile_worker_environment(),
        )
    except TPUProfilingError as exc:
        pytest.exit(
            f"Aborting opt-in TPU PCIe profile session after {case} failed: {exc}",
            returncode=2,
        )


def _assert_profile_worker_result(report, *, case: str, runtime_mode: str):
    prefix = "TPU_CORE_NUMERIC_RESULT="
    payloads = [json.loads(line[len(prefix):])
                for line in report.stdout_path.read_text(encoding="utf-8").splitlines()
                if line.startswith(prefix)]
    assert len(payloads) == 1
    assert payloads[0]["schema_version"] == 1
    assert payloads[0]["case"] == case
    assert payloads[0]["runtime_mode"] == runtime_mode
    assert payloads[0]["status"] == "passed"
    assert payloads[0]["metrics"]["passed"] is True


@pytest.mark.skipif(
    os.environ.get("TILELANG_TPU_RUN_CMODEL_PROFILE") != "1",
    reason="set TILELANG_TPU_RUN_CMODEL_PROFILE=1 for the isolated CModel profile worker",
)
def test_bm1690e_tpukernel_cmodel_profile_worker_collects_real_raw_trace(tmp_path):
    """Opt-in end-to-end test: fresh compile, numerical check, and raw command dump."""

    if not os.environ.get("PPL_PROJECT_ROOT"):
        pytest.skip("PPL_PROJECT_ROOT is not configured")
    config = TPUProfilingConfig(
        chip="bm1690e",
        programming_model="tpukernel",
        output_dir=tmp_path,
        label="bm1690e-tpukernel-matmul",
        timeout_s=60,
    )

    report = _run_opt_in_cmodel_profile_or_abort(config, "tpukernel-matmul")

    _assert_profile_worker_result(report, case="tpukernel-matmul", runtime_mode="cmodel")
    assert report.has_raw_trace
    assert report.raw_instructions
    # The local SDK may not bundle PerfAI; raw-only is an intentional result.
    assert report.parser_status in ("unavailable", "ready", "no-device-command-events")


@pytest.mark.skipif(
    os.environ.get("TILELANG_TPU_RUN_CMODEL_PROFILE") != "1",
    reason="set TILELANG_TPU_RUN_CMODEL_PROFILE=1 for the isolated CModel profile worker",
)
def test_bm1690e_rv_cmodel_profile_worker_runs_control_path(tmp_path):
    """Opt-in RV control-path test, intentionally not an RV tensor numeric claim."""

    if not os.environ.get("PPL_PROJECT_ROOT"):
        pytest.skip("PPL_PROJECT_ROOT is not configured")
    config = TPUProfilingConfig(
        chip="bm1690e",
        programming_model="rv",
        output_dir=tmp_path,
        label="bm1690e-rv-control",
        timeout_s=60,
        postprocess=False,
    )

    report = _run_opt_in_cmodel_profile_or_abort(config, "rv-control")

    _assert_profile_worker_result(report, case="rv-control", runtime_mode="cmodel")
    assert report.has_raw_trace
    assert report.raw_instructions


@pytest.mark.skipif(
    os.environ.get("TILELANG_TPU_RUN_PCIE_PROFILE") != "1",
    reason="set TILELANG_TPU_RUN_PCIE_PROFILE=1 for the isolated PCIe profile worker",
)
def test_bm1690e_tpukernel_pcie_profile_worker_collects_real_raw_trace(tmp_path):
    """Opt-in board test: numerical matmul plus TPUDNN recorder output."""

    if not os.environ.get("PPL_PROJECT_ROOT"):
        pytest.skip("PPL_PROJECT_ROOT is not configured")
    config = TPUProfilingConfig(
        chip="bm1690e",
        programming_model="tpukernel",
        runtime_mode="pcie",
        output_dir=tmp_path,
        label="bm1690e-pcie-tpukernel-matmul",
        timeout_s=60,
    )

    report = _run_opt_in_pcie_profile_or_abort(config, "tpukernel-matmul")

    _assert_profile_worker_result(report, case="tpukernel-matmul", runtime_mode="pcie")
    assert report.has_raw_trace
    assert report.parser_status in ("unavailable", "ready", "no-device-command-events")
    if report.parser_status == "ready":
        assert report.decoded_report_path is not None
        assert report.instruction_timings


@pytest.mark.skipif(
    os.environ.get("TILELANG_TPU_RUN_PCIE_PROFILE") != "1",
    reason="set TILELANG_TPU_RUN_PCIE_PROFILE=1 for the isolated PCIe profile worker",
)
def test_bm1690e_rv_pcie_profile_worker_runs_control_path(tmp_path):
    """Opt-in RV board control-path test, not RV tensor arithmetic proof."""

    if not os.environ.get("PPL_PROJECT_ROOT"):
        pytest.skip("PPL_PROJECT_ROOT is not configured")
    config = TPUProfilingConfig(
        chip="bm1690e",
        programming_model="rv",
        runtime_mode="pcie",
        output_dir=tmp_path,
        label="bm1690e-pcie-rv-control",
        timeout_s=60,
        postprocess=False,
    )

    report = _run_opt_in_pcie_profile_or_abort(config, "rv-control")

    _assert_profile_worker_result(report, case="rv-control", runtime_mode="pcie")
    assert report.has_raw_trace
