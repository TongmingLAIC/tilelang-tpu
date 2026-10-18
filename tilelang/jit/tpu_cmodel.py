# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Launch CModel with user-owned private IPC paths, without exposing PCIe devices.

PPL 1.7 uses the literal /tmp/sg_wakeup_stream, independent of TMPDIR.
Run arbitrary CModel scripts as::

    python -m tilelang.jit.tpu_cmodel --output-dir /absolute/new/result -- \
        python /absolute/path/to/worker.py

The child sees /tmp, but it is a private bind mount backed by a fresh directory
under ~/tmp/tilelang, never the host's shared /tmp. It can write only to its
output directory and that private scratch directory. Use absolute input/source
paths; compile and load kernels inside the child.
Linux user namespaces and bubblewrap are required. There is no shared-IPC
fallback and this module never modifies an existing host FIFO.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Mapping, Sequence

from tilelang.temp_storage import private_cmodel_mount_ok, user_temp_root


def check_cmodel_fifo(path: Path = Path("/tmp/sg_wakeup_stream")) -> None:
    """Fail before vendor initialization on a known unsafe communication path.

    This is a diagnostic, not a concurrency guarantee. Only a private mount
    namespace prevents races with another runtime using the same FIFO.
    """
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISFIFO(info.st_mode) or not os.access(
            path, os.R_OK | os.W_OK, effective_ids=True):
        raise RuntimeError(
            f"CModel communication FIFO is not safely readable/writable: {path} "
            f"(uid={info.st_uid}, gid={info.st_gid}, mode={stat.S_IMODE(info.st_mode):04o}). "
            "PPL may hang while loading a module. Run the script with "
            "python -m tilelang.jit.tpu_cmodel --output-dir /absolute/new/result -- "
            "python /absolute/path/to/script.py. Do not chmod or delete another user's FIFO.")


def _private_tmp_mount() -> bool:
    """Check that /tmp is bound to this worker's private user directory."""
    return private_cmodel_mount_ok()


def require_private_cmodel() -> None:
    """Reject CModel execution outside the launcher's private /tmp mount.

    A currently accessible shared FIFO is still unsafe: another process can
    reuse the same literal name between module loads. Check the mount and its
    user-owned backing directory as well as the launcher marker.
    """
    check_cmodel_fifo()
    if os.environ.get("TILELANG_CMODEL_PRIVATE_TMP") != "1" or not _private_tmp_mount():
        raise RuntimeError(
            "CModel execution needs a private /tmp even when the shared FIFO is currently "
            "accessible. Run the complete Python script with "
            "python -m tilelang.jit.tpu_cmodel --output-dir /absolute/new/result -- "
            "python /absolute/path/to/script.py.")


def prepare_cmodel_command(command: Sequence[str], *, cwd: Path,
                           environment: Mapping[str, str],
                           private_tmp_dir: Path) -> tuple[list[str], dict[str, str]]:
    """Build one isolated worker command; the caller owns its watchdog."""
    if not command:
        raise ValueError("a CModel command is required")
    if not sys.platform.startswith("linux"):
        raise RuntimeError("private CModel IPC requires Linux namespaces")
    cwd = cwd.resolve(strict=True)
    if not cwd.is_dir() or cwd == Path("/") or cwd.is_relative_to(Path("/tmp")):
        raise ValueError("CModel needs a dedicated writable working directory")
    if private_tmp_dir.is_symlink():
        raise ValueError("CModel private temporary directory must not be a symlink")
    private_tmp_dir = private_tmp_dir.resolve(strict=True)
    root = user_temp_root()
    if (private_tmp_dir == root or not private_tmp_dir.is_relative_to(root) or
            not private_tmp_dir.is_dir()):
        raise ValueError(f"CModel private temporary directory must be below {root}")
    info = private_tmp_dir.stat()
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("CModel private temporary directory must be owned by this user "
                         "with mode 0700")
    bwrap = shutil.which("bwrap", path=environment.get("PATH"))
    if bwrap is None:
        raise RuntimeError("bubblewrap (bwrap) is required for private CModel IPC; "
                           "refusing a shared /tmp fallback")
    worker_env = dict(environment)
    for name in tuple(worker_env):
        if name.startswith("TILELANG_TPU_ALLOW_PCIE") or name == "TILELANG_TPU_DEVICE_ID":
            worker_env.pop(name)
    worker_env.update({"TMPDIR": "/tmp", "TMP": "/tmp", "TEMP": "/tmp",
                       "TILELANG_CACHE_DIR": "/tmp/tilelang-cache",
                       "PYTHONDONTWRITEBYTECODE": "1", "TILELANG_CMODEL_PRIVATE_TMP": "1",
                       "TILELANG_CMODEL_PRIVATE_TMP_HOST": str(private_tmp_dir)})
    isolated = [bwrap, "--unshare-user", "--unshare-pid", "--unshare-net",
                "--die-with-parent", "--ro-bind", "/", "/", "--dev", "/dev",
                "--proc", "/proc", "--bind", str(private_tmp_dir), "/tmp",
                "--bind", str(cwd), str(cwd),
                "--chdir", str(cwd), "--", *map(os.fspath, command)]
    return isolated, worker_env


def _stop(process: subprocess.Popen) -> tuple[str, str]:
    """Bound both process termination and pipe draining."""
    for signum in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass
        try:
            return process.communicate(timeout=3)
        except subprocess.TimeoutExpired as error:
            latest = error
    for pipe in (process.stdout, process.stderr):
        if pipe is not None:
            pipe.close()
    def as_text(value):
        return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
    return as_text(latest.output), as_text(latest.stderr) + "\nCModel pipe drain timed out.\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("provide a command and a finite positive timeout")
    output = args.output_dir.resolve()
    if output == Path("/") or output.is_relative_to(Path("/tmp")):
        parser.error("the CModel result directory must be outside the host /tmp")
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    begin = time.monotonic()
    process = None
    private_tmp = None
    status, returncode, stdout, stderr = "launch_failed", 1, "", ""
    launched = []
    try:
        private_tmp = tempfile.TemporaryDirectory(prefix="cmodel-", dir=user_temp_root())
        launched, environment = prepare_cmodel_command(
            command, cwd=output, environment=os.environ,
            private_tmp_dir=Path(private_tmp.name))
        process = subprocess.Popen(launched, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=args.timeout)
            returncode = process.returncode
            status = "passed" if returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            stdout, stderr = _stop(process)
            status, returncode = "timeout", 124
    except (OSError, RuntimeError, ValueError) as error:
        stderr = f"{type(error).__name__}: {error}\n"
    except BaseException:
        if process is not None:
            stdout, stderr = _stop(process)
        status, returncode = "interrupted", 130
        raise
    finally:
        if private_tmp is not None:
            try:
                private_tmp.cleanup()
            except OSError as error:
                status, returncode = "cleanup_failed", 1
                stderr += f"CModel private temporary cleanup failed: {error}\n"
        (output / "stdout.log").write_text(stdout, encoding="utf-8")
        (output / "stderr.log").write_text(stderr, encoding="utf-8")
        summary = {"schema_version": 1, "started_at_utc": started, "status": status,
                   "returncode": returncode, "timeout_seconds": args.timeout,
                   "wall_seconds": time.monotonic() - begin, "command": command,
                   "isolation_command": launched, "ppl_root": os.environ.get("PPL_PROJECT_ROOT")}
        (output / "isolation_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(stdout, end="")
    print(stderr, end="", file=sys.stderr)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return returncode if returncode >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
