# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""CModel IPC isolation contracts; no TPU is launched by these tests."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import pytest

from tilelang.jit.tpu_cmodel import (
    check_cmodel_fifo, prepare_cmodel_command, require_private_cmodel)
from tilelang.temp_storage import private_cmodel_mount_ok, user_temp_root


@pytest.fixture
def private_tmp():
    with tempfile.TemporaryDirectory(prefix="test-cmodel-", dir=user_temp_root()) as path:
        yield Path(path)


def test_default_python_temp_is_under_user_home():
    root = user_temp_root()
    assert root == Path.home().resolve() / "tmp" / "tilelang"
    assert tempfile.gettempdir() == str(root)
    with tempfile.NamedTemporaryFile() as temporary:
        assert Path(temporary.name).parent == root


def test_fifo_preflight_never_changes_existing_path(tmp_path, monkeypatch):
    fifo = tmp_path / "communication"
    check_cmodel_fifo(fifo)
    os.mkfifo(fifo, 0o600)
    check_cmodel_fifo(fifo)
    info = fifo.stat()
    monkeypatch.setattr(os, "access", lambda *args, **kwargs: False)
    with pytest.raises(RuntimeError, match="python -m tilelang.jit.tpu_cmodel"):
        check_cmodel_fifo(fifo)
    assert fifo.stat() == info


def test_fifo_preflight_rejects_regular_file_and_symlink(tmp_path):
    regular = tmp_path / "regular"
    regular.write_text("preserve")
    link = tmp_path / "link"
    link.symlink_to(regular)
    for path in (regular, link):
        with pytest.raises(RuntimeError, match="not safely readable/writable"):
            check_cmodel_fifo(path)
    assert regular.read_text() == "preserve"


def test_cmodel_rejects_shared_tmp_even_when_fifo_is_accessible(monkeypatch):
    from tilelang.jit import tpu_cmodel

    monkeypatch.setattr(tpu_cmodel, "check_cmodel_fifo", lambda: None)
    monkeypatch.setattr(tpu_cmodel, "_private_tmp_mount", lambda: False)
    monkeypatch.setenv("TILELANG_CMODEL_PRIVATE_TMP", "1")
    with pytest.raises(RuntimeError, match="needs a private /tmp"):
        require_private_cmodel()
    monkeypatch.setattr(tpu_cmodel, "_private_tmp_mount", lambda: True)
    monkeypatch.delenv("TILELANG_CMODEL_PRIVATE_TMP", raising=False)
    with pytest.raises(RuntimeError, match="needs a private /tmp"):
        require_private_cmodel()


def test_isolation_command_and_environment(private_tmp, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: "/usr/bin/bwrap")
    environment = {"HOME": "/keep", "TMPDIR": "/shared",
                   "TILELANG_TPU_ALLOW_PCIE_LOAD": "1", "TILELANG_TPU_DEVICE_ID": "0"}
    working_dir = private_tmp / "work"
    working_dir.mkdir()
    command, child = prepare_cmodel_command(
        ["python", "worker.py"], cwd=working_dir, environment=environment,
        private_tmp_dir=private_tmp)
    assert command[-3:] == ["--", "python", "worker.py"]
    for flag in ("--unshare-user", "--unshare-pid", "--unshare-net", "--die-with-parent"):
        assert flag in command
    assert command[command.index("--bind") + 1:command.index("--bind") + 3] == [
        str(private_tmp), "/tmp"]
    assert command[command.index("--dev") + 1] == "/dev"
    assert child["TMPDIR"] == "/tmp" and child["HOME"] == "/keep"
    assert child["TILELANG_CMODEL_PRIVATE_TMP_HOST"] == str(private_tmp)
    assert "TILELANG_TPU_ALLOW_PCIE_LOAD" not in child
    assert "TILELANG_TPU_DEVICE_ID" not in child
    assert environment["TMPDIR"] == "/shared"


def test_missing_bwrap_has_no_unisolated_fallback(private_tmp, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: None)
    working_dir = private_tmp / "work"
    working_dir.mkdir()
    with pytest.raises(RuntimeError, match="refusing a shared /tmp fallback"):
        prepare_cmodel_command(["python"], cwd=working_dir, environment={},
                               private_tmp_dir=private_tmp)


@pytest.mark.parametrize("cwd", (Path("/"), Path("/tmp")))
def test_rejects_broad_writable_mount(cwd, private_tmp):
    with pytest.raises(ValueError, match="dedicated"):
        prepare_cmodel_command(["python"], cwd=cwd, environment={},
                               private_tmp_dir=private_tmp)


def test_forged_private_marker_does_not_allow_shared_tmp(private_tmp, monkeypatch):
    monkeypatch.setenv("TILELANG_CMODEL_PRIVATE_TMP", "1")
    monkeypatch.setenv("TILELANG_CMODEL_PRIVATE_TMP_HOST", str(private_tmp))
    assert not private_cmodel_mount_ok()


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap not installed")
def test_private_fifo_is_not_host_fifo(private_tmp):
    # The source and interpreter are outside /tmp; the sole writable mount is
    # the test output directory. No TPU SDK or board is involved.
    host = Path("/tmp/sg_wakeup_stream")
    before = host.lstat() if host.exists() else None
    source = ("import json, os; from pathlib import Path; "
              "from tilelang.jit.tpu_cmodel import require_private_cmodel; "
              "require_private_cmodel(); "
              "p=Path('/tmp/sg_wakeup_stream'); assert not p.exists(); "
              "os.mkfifo(p, 0o600); "
              "Path('observed.json').write_text(json.dumps({'uid':p.stat().st_uid, "
              "'device_nodes':list(map(str, Path('/dev').glob('tpu*')))}))")
    working_dir = private_tmp / "work"
    working_dir.mkdir()
    command, environment = prepare_cmodel_command(
        [sys.executable, "-c", source], cwd=working_dir, environment=os.environ,
        private_tmp_dir=private_tmp)
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=10)
    if "No permissions to create" in result.stderr or "Operation not permitted" in result.stderr:
        pytest.skip("unprivileged namespaces disabled")
    assert result.returncode == 0, result.stderr
    assert json.loads((working_dir / "observed.json").read_text())["device_nodes"] == []
    assert (private_tmp / "sg_wakeup_stream").is_fifo()
    assert (host.lstat() if host.exists() else None) == before


def test_cli_records_launch_failure(private_tmp):
    output = private_tmp / "failed"
    module = Path(__file__).resolve().parents[3] / "tilelang/jit/tpu_cmodel.py"
    environment = dict(os.environ, PATH="")
    result = subprocess.run(
        [sys.executable, str(module), "--output-dir", str(output), "--", "unused"],
        env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    summary = json.loads((output / "isolation_summary.json").read_text())
    assert summary["status"] == "launch_failed"
    assert "refusing" in (output / "stderr.log").read_text()


def test_cli_rejects_host_tmp_output_before_creating_it():
    output = Path("/tmp") / f"tilelang-cmodel-rejected-{os.getpid()}-{time.time_ns()}"
    module = Path(__file__).resolve().parents[3] / "tilelang/jit/tpu_cmodel.py"
    result = subprocess.run(
        [sys.executable, str(module), "--output-dir", str(output), "--", "unused"],
        capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "outside the host /tmp" in result.stderr
    assert not output.exists()


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap not installed")
def test_cli_timeout_is_recorded(private_tmp):
    output = private_tmp / "timed-out"
    module = Path(__file__).resolve().parents[3] / "tilelang/jit/tpu_cmodel.py"
    result = subprocess.run(
        [sys.executable, str(module), "--output-dir", str(output), "--timeout", "0.25",
         "--", sys.executable, "-c", "import time; time.sleep(20)"],
        capture_output=True, text=True, timeout=10)
    if "Operation not permitted" in result.stderr or "No permissions to create" in result.stderr:
        pytest.skip("unprivileged namespaces disabled")
    assert result.returncode == 124
    assert json.loads((output / "isolation_summary.json").read_text())["status"] == "timeout"


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap not installed")
def test_namespace_reaps_child_that_escapes_process_group(private_tmp):
    child = "import time; from pathlib import Path; time.sleep(2); Path('leaked').touch()"
    source = ("import subprocess, sys, time; from pathlib import Path; "
              f"subprocess.Popen([sys.executable, '-c', {child!r}], start_new_session=True); "
              "Path('ready').touch(); time.sleep(30)")
    working_dir = private_tmp / "work"
    working_dir.mkdir()
    command, environment = prepare_cmodel_command(
        [sys.executable, "-c", source], cwd=working_dir, environment=os.environ,
        private_tmp_dir=private_tmp)
    process = subprocess.Popen(command, env=environment, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 5
        while not (working_dir / "ready").exists() and process.poll() is None:
            if time.monotonic() > deadline:
                pytest.fail("namespace worker did not start")
            time.sleep(0.01)
        if process.poll() is not None:
            diagnostic = process.communicate()[1].decode()
            if "Operation not permitted" in diagnostic or "No permissions" in diagnostic:
                pytest.skip("unprivileged namespaces disabled")
            pytest.fail(diagnostic)
        process.terminate()
        process.communicate(timeout=5)
        time.sleep(2.2)
        assert not (working_dir / "leaked").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
