# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Safe PPL-style instruction profiling for TileLang TPU test programs.

TileLang emits raw PPL C directly and does not pass through ``ppl-compile``.
Consequently PPL's CLI-level ``--profiling``/``--autotune`` switch is not
directly applicable here. What is reusable is its runtime protocol:

* on CModel, execute one freshly compiled test worker in a dedicated directory
  with ``FILE_DUMP_CMD`` set, then optionally run an explicitly supplied
  PerfAI installation over the raw command dumps;
* on PCIe, enable TPUDNN command recording around exactly one launch in a
  supervised worker, behind separate load/profile/device safety gates, then
  optionally decode ``cdm_profile_data_dev*`` with already-installed vendor
  Python packages.

The child-process boundary is intentional.  Both the vendor runtime and the
process working directory are global state, and a profile worker must never
load an arbitrary prebuilt ``main.so`` from another JIT instance.  The command
passed to :class:`TPUInstructionProfiler` is therefore expected to compile and
load its own private TileLang JIT artifact.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
import zipfile

try:
    import fcntl
except ImportError:  # pragma: no cover - importability guard for non-POSIX hosts
    fcntl = None

from tilelang.engine.tpu_config import TPURuntimeConfig, TPUTargetSpec
from tilelang.jit.adapter.ppl_layout import resolve_ppl_layout

PathLike = Union[str, os.PathLike]

_DEVICE_COMMAND_ENGINES = frozenset((
    "bd",
    "bdc",
    "tiu",
    "gdma",
    "sdma",
    "vsdma",
    "cdma",
    "dma",
))
_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_RAW_TRACE_ARTIFACT_RE = re.compile(r"^(?P<label>[A-Za-z0-9][A-Za-z0-9_.-]*)-"
                                    r"(?P<launch>\d+)-(?P<group>\d+)\."
                                    r"(?P<engine>BD|GDMA|SDMA|VSDMA)\.(?P<core>\d+)(?:\.txt)?$")
_RAW_TRACE_RE = re.compile(r"^(?P<label>[A-Za-z0-9][A-Za-z0-9_.-]*)-"
                           r"(?P<launch>\d+)-(?P<group>\d+)\."
                           r"(?P<engine>BD|GDMA|SDMA|VSDMA)\.(?P<core>\d+)\.txt$")
_PCIE_RAW_PROFILE_FILE_RE = re.compile(r"^(?:global|cdmlib\d+_\d+)\.profile$")

_PROFILE_SUPERVISOR_PATH = Path(__file__).parent.parent / "_tpu_profile_supervisor.py"
_PCIE_PROFILE_DECODER_PATH = Path(__file__).parent.parent / "_tpu_pcie_profile_decoder.py"
_PCIE_DECODED_REPORT_NAME = "tilelang_pcie_profile.json"
_PCIE_DECODER_IDENTITY_PREFIX = "TILELANG_TPU_PCIE_DECODER_IDENTITY="
_PCIE_DECODER_PREFLIGHT_TIMEOUT_S = 10.0
_PCIE_HARDWARE_ENVIRONMENT_KEYS = frozenset((
    "BMLIB_ENABLE_ALL_PROFILE",
    "FILE_DUMP_CMD",
    "PROFILE_BOOK_KEEPING",
    "PROFILE_RECORD_SIZE",
    "TILELANG_TPU_ALLOW_PCIE_LOAD",
    "TILELANG_TPU_ALLOW_PCIE_PROFILE",
    "TILELANG_TPU_BENCHMARK_RUNS",
    "TILELANG_TPU_DEVICE_ID",
    "TILELANG_TPU_PROFILE_CHIP",
    "TILELANG_TPU_PROFILE_OUTPUT_DIR",
    "TILELANG_TPU_PROFILE_PROGRAMMING_MODEL",
    "TILELANG_TPU_PROFILE_RUNTIME_MODE",
    "TILELANG_TPU_PROFILE_SESSION",
))

_PCIE_DEVICE_LOCK_STATE = threading.local()
_PCIE_DEVICE_LOCK_ROOT = Path("/run/lock")


def _pcie_device_quarantine_path(device_id: int) -> Path:
    return _PCIE_DEVICE_LOCK_ROOT / f"tilelang-tpu-device-{device_id}.quarantine.json"


def _pcie_device_session_path(device_id: int) -> Path:
    return _PCIE_DEVICE_LOCK_ROOT / f"tilelang-tpu-device-{device_id}.session.json"


def _pcie_device_session_owned(device_id: int) -> bool:
    held = getattr(_PCIE_DEVICE_LOCK_STATE, "held", {})
    state = held.get(device_id)
    return isinstance(state, dict) and state.get("depth", 0) > 0


def _pcie_device_lock_owned(device_id: int) -> bool:
    """Return whether this thread may still issue commands in its session."""

    if not _pcie_device_session_owned(device_id):
        return False
    return _PCIE_DEVICE_LOCK_STATE.held[device_id].get("safe_to_release") is not False


def _assert_pcie_device_not_quarantined(device_id: int) -> None:
    marker = _pcie_device_quarantine_path(device_id)
    if not marker.exists():
        return
    try:
        detail = marker.read_text(encoding="utf-8").strip()
    except OSError as error:
        detail = f"unreadable marker: {error}"
    raise TPUProfilingError(
        f"TPU device {device_id} is fail-closed by {marker}. A prior supervised "
        "PCIe session could not establish a safe final board state. Inspect the "
        "recorded reason and any available PGID, then recover the board manually "
        "before removing the marker"
        f"{': ' + detail if detail else '.'}")


def _create_pcie_device_session(device_id: int) -> Path:
    """Create a crash-persistent lease for one outermost PCIe session.

    ``flock`` is released by the kernel when a matrix parent dies.  That is
    useful for ordinary serialization but unsafe as the only board guard: a
    descendant blocked in the driver can outlive the parent.  This marker is
    removed only by the lock context's orderly exit, so SIGKILL, parent death,
    or interpreter abort leaves the next invocation fail-closed.
    """

    marker = _pcie_device_session_path(device_id)
    payload = {
        "schema_version": 1,
        "status": "active",
        "device_id": device_id,
        "owner_pid": os.getpid(),
        "host": socket.gethostname(),
        "created_unix_ns": time.time_ns(),
    }
    encoded = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        try:
            detail = marker.read_text(encoding="utf-8").strip()
        except OSError as read_error:
            detail = f"unreadable marker: {read_error}"
        raise TPUProfilingError(
            f"TPU device {device_id} has an incomplete prior session recorded by "
            f"{marker}. Inspect the board and prior owner before removing the marker"
            f"{': ' + detail if detail else '.'}") from error
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        # A partial marker is still the safe state: it prevents another launch
        # until an operator has inspected the board.
        raise
    return marker


def _validate_pcie_quarantine_reason(reason: str) -> str:
    """Return a stable, human-readable reason suitable for a durable marker."""

    if not isinstance(reason, str) or not reason or reason != reason.strip() or \
            not reason.isprintable():
        raise ValueError(
            "reason must be a non-empty printable string without surrounding whitespace")
    return reason


def _validate_pcie_process_group(process_group: Optional[int]) -> Optional[int]:
    """Validate an observed process group, or preserve an explicitly unknown one."""

    if process_group is not None and (isinstance(process_group, bool) or
                                      not isinstance(process_group, int) or process_group <= 0):
        raise ValueError("process_group must be a positive integer or None")
    return process_group


def _quarantine_pcie_device(
    device_id: int,
    *,
    process_group: Optional[int],
    reason: str,
) -> Path:
    """Persistently fail-close one device after its safe state becomes uncertain.

    The marker is intentionally never removed automatically. ``tpu-smi`` can
    report an idle card while a process remains blocked in the driver, so only
    an operator who has inspected the reason, any recorded PGID, and recovered
    the board may remove it. Callers must own the session lock to prevent a
    competing launch between detecting the failure and creating the marker.
    """

    reason = _validate_pcie_quarantine_reason(reason)
    process_group = _validate_pcie_process_group(process_group)
    if not _pcie_device_session_owned(device_id):
        raise TPUProfilingError(
            f"Cannot quarantine TPU device {device_id} without owning its session lock")
    # Set this before touching the quarantine path.  Creating that second
    # marker can itself fail (for example ENOSPC or EACCES); the already
    # durable session marker must then survive lock release and keep the board
    # fail-closed.
    held = _PCIE_DEVICE_LOCK_STATE.held
    held[device_id]["safe_to_release"] = False
    marker = _pcie_device_quarantine_path(device_id)
    payload = {
        "schema_version": 1,
        "status": "quarantined",
        "device_id": device_id,
        "process_group": process_group,
        "process_group_known": process_group is not None,
        "reason": reason,
        "owner_pid": os.getpid(),
        "host": socket.gethostname(),
        "created_unix_ns": time.time_ns(),
    }
    encoded = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return marker
    except OSError as error:
        session_path = held[device_id]["session_path"]
        raise TPUProfilingError(
            f"Cannot create PCIe quarantine marker {marker}: {error}. "
            f"The active session marker {session_path} is retained fail-closed") from error
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        # Even a partial marker remains fail-closed. Do not unlink it and
        # accidentally permit another board launch.
        raise
    return marker


@contextmanager
def _exclusive_pcie_device_lock(device_id: int):
    """Serialize a PCIe session, allowing nested ownership in one thread."""

    if fcntl is None or not sys.platform.startswith("linux"):
        raise TPUProfilingError("Safe PCIe device ownership requires Linux flock support")
    held = getattr(_PCIE_DEVICE_LOCK_STATE, "held", None)
    if held is None:
        held = {}
        _PCIE_DEVICE_LOCK_STATE.held = held
    if device_id in held:
        if held[device_id].get("safe_to_release") is False:
            raise TPUProfilingError(
                f"TPU device {device_id} became unsafe during this PCIe session; "
                "no further command may be launched")
        _assert_pcie_device_not_quarantined(device_id)
        held[device_id]["depth"] += 1
        try:
            yield held[device_id]["path"]
        finally:
            held[device_id]["depth"] -= 1
        return

    lock_path = _PCIE_DEVICE_LOCK_ROOT / f"tilelang-tpu-device-{device_id}.lock"
    try:
        # Opening an existing file with O_CREAT can be denied by Linux
        # fs.protected_regular in a sticky /run/lock even when its group grants
        # write access. Open an existing shared lock without O_CREAT, while
        # retaining exclusive creation for the first owner and the creation
        # race.
        try:
            lock_file = lock_path.open("r+", encoding="utf-8")
        except FileNotFoundError:
            try:
                lock_file = lock_path.open("x+", encoding="utf-8")
            except FileExistsError:
                lock_file = lock_path.open("r+", encoding="utf-8")
    except OSError as error:
        raise TPUProfilingError(f"Cannot open PCIe device lock {lock_path}: {error}") from error
    with lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            message = (f"TPU device {device_id} is already owned by another TileLang PCIe session")
            raise TPUProfilingError(message) from error
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()}\n")
        lock_file.flush()
        _assert_pcie_device_not_quarantined(device_id)
        session_path = _create_pcie_device_session(device_id)
        held[device_id] = {
            "depth": 1,
            "path": lock_path,
            "session_path": session_path,
            "safe_to_release": True,
        }
        try:
            yield lock_path
        finally:
            state = held.get(device_id)
            if state is None or state["depth"] != 1:
                raise TPUProfilingError(
                    f"PCIe device {device_id} lock ownership became inconsistent")
            try:
                if state["safe_to_release"]:
                    try:
                        session_path.unlink()
                    except FileNotFoundError as error:
                        # An external removal defeats the crash-detection
                        # invariant. Persist a stronger marker before releasing
                        # the file lock.
                        quarantine = _quarantine_pcie_device(
                            device_id,
                            process_group=os.getpid(),
                            reason=("active PCIe session marker disappeared before orderly exit"),
                        )
                        raise TPUProfilingError(
                            f"PCIe session marker disappeared; device {device_id} is "
                            f"quarantined by {quarantine}") from error
            finally:
                held.pop(device_id, None)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class TPUProfilingError(RuntimeError):
    """Base class for TPU profiling failures that preserve collected artifacts."""


class TPUProfilingTimeoutError(TPUProfilingError):
    """A profile worker exceeded its timeout and its process group was stopped."""


class TPUProfilingCommandError(TPUProfilingError):
    """The isolated profile worker exited unsuccessfully."""


class _PerfAIWorkspaceBusy(RuntimeError):
    """The vendor PerfAI work directory is already owned by another session."""


def _normalize_decoder_python(value: PathLike) -> Path:
    """Resolve and validate an explicitly selected offline decoder Python."""

    try:
        raw_value = os.fspath(value)
    except TypeError as exc:
        raise TypeError("pcie_decoder_python must be a path-like value.") from exc
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError("pcie_decoder_python must be a non-empty filesystem path.")
    # Keep a virtual-environment launcher spelling intact. Resolving the
    # ``bin/python`` symlink to /usr/bin/python discards pyvenv.cfg discovery
    # and silently loses the decoder's installed dependencies.
    path = Path(os.path.abspath(Path(raw_value).expanduser()))
    if not path.is_file():
        raise ValueError(f"pcie_decoder_python is not a file: {path}")
    if not os.access(path, os.X_OK):
        raise ValueError(f"pcie_decoder_python is not executable: {path}")
    return path


def _normalize_decoder_pythonpath(values: Sequence[PathLike]) -> Tuple[Path, ...]:
    """Resolve decoder-only directories or importable zip/wheel archives."""

    if isinstance(values, (str, bytes, os.PathLike)):
        raise TypeError("pcie_decoder_pythonpath must be a sequence of import paths, "
                        "not one path-like value.")
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise TypeError("pcie_decoder_pythonpath must be a sequence of import paths.") from exc
    normalized = []
    for value in raw_values:
        try:
            raw_value = os.fspath(value)
        except TypeError as exc:
            raise TypeError("Every pcie_decoder_pythonpath entry must be path-like.") from exc
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError("pcie_decoder_pythonpath entries must be non-empty paths.")
        path = Path(raw_value).expanduser().resolve()
        if not path.is_dir() and not (path.is_file() and zipfile.is_zipfile(path)):
            raise ValueError("pcie_decoder_pythonpath entry is not a directory or importable "
                             f"zip/wheel archive: {path}")
        if path in normalized:
            raise ValueError(f"pcie_decoder_pythonpath contains a duplicate entry: {path}")
        normalized.append(path)
    return tuple(normalized)


@dataclass(frozen=True)
class TPUProfilingConfig:
    """Configuration for one isolated TPU instruction-profile session.

    PCIe dispatch is fail-closed: :meth:`TPUInstructionProfiler.run_pcie`
    requires independent load/profile acknowledgements plus an explicit device
    ID.  Its worker must compile and load its own private JIT artifact, just as
    the CModel path does.
    """

    chip: str
    programming_model: str = "tpukernel"
    runtime_mode: str = "cmodel"
    output_dir: Optional[PathLike] = None
    label: str = "tilelang"
    timeout_s: float = 60.0
    perfai_root: Optional[PathLike] = None
    postprocess: bool = True
    profile_record_size: int = 4096
    profile_book_keeping: int = 1
    pcie_decoder_python: Optional[PathLike] = None
    pcie_decoder_pythonpath: Tuple[PathLike, ...] = ()

    def __post_init__(self) -> None:
        if self.runtime_mode not in ("cmodel", "pcie"):
            raise ValueError(f"Unsupported TPU profiling runtime mode {self.runtime_mode!r}; "
                             f"expected 'cmodel' or 'pcie'.")
        if isinstance(self.timeout_s, bool) or \
                not isinstance(self.timeout_s, (int, float)) or \
                not math.isfinite(float(self.timeout_s)) or self.timeout_s <= 0:
            raise ValueError("TPU profiling timeout_s must be a positive number.")
        if not _LABEL_RE.fullmatch(self.label):
            raise ValueError("TPU profiling label must start with an ASCII letter/digit and "
                             "contain only letters, digits, '.', '_' or '-'.")
        if not isinstance(self.profile_record_size, int) or self.profile_record_size <= 0:
            raise ValueError("TPU profile_record_size must be a positive integer.")
        if not isinstance(self.profile_book_keeping, int) or self.profile_book_keeping < 0:
            raise ValueError("TPU profile_book_keeping must be a non-negative integer.")
        # The dual-backend capability registry is the one authority for chip,
        # programming model and core topology.
        # Profiling must not maintain a second copy of this table.
        target_spec = TPUTargetSpec(
            chip=self.chip,
            programming_model=self.programming_model,
        )
        runtime_config = TPURuntimeConfig(runtime_mode=self.runtime_mode)
        object.__setattr__(self, "chip", target_spec.chip)
        object.__setattr__(self, "programming_model", target_spec.programming_model)
        object.__setattr__(self, "runtime_mode", runtime_config.runtime_mode)
        if (self.pcie_decoder_python is not None or self.pcie_decoder_pythonpath) and \
                runtime_config.runtime_mode != "pcie":
            raise ValueError("pcie_decoder_python and pcie_decoder_pythonpath are only valid "
                             "for runtime_mode='pcie'.")
        if self.pcie_decoder_python is not None:
            object.__setattr__(self, "pcie_decoder_python",
                               _normalize_decoder_python(self.pcie_decoder_python))
        object.__setattr__(self, "pcie_decoder_pythonpath",
                           _normalize_decoder_pythonpath(self.pcie_decoder_pythonpath))

    @property
    def target_spec(self) -> TPUTargetSpec:
        """Return the canonical compile-time identity for this session."""

        return TPUTargetSpec(
            chip=self.chip,
            programming_model=self.programming_model,
        )

    @property
    def runtime_config(self) -> TPURuntimeConfig:
        """Return the canonical host runtime selection for this session."""

        return TPURuntimeConfig(runtime_mode=self.runtime_mode)


@dataclass(frozen=True)
class TPUInstructionTiming:
    """One device-command timeline row emitted by PerfAI's ``profile_data.js``.

    ``instruction_timings`` intentionally excludes CPU/subnet/layer timeline
    events.  The complete PerfAI timeline remains available in
    :attr:`TPUProfileReport.timeline_events` for diagnostics.
    """

    engine: str
    begin: Any
    end: Any
    duration: Optional[float]
    unit: str
    core_id: Optional[int] = None
    command_id: Optional[int] = None
    opcode: Optional[str] = None
    fields: Mapping[str, Any] = field(default_factory=dict)


def _pcie_instruction_timing_error(timing: Any) -> Optional[str]:
    """Return why a decoded PCIe timing row is invalid, or ``None``."""

    for field_name in ("begin", "end", "duration"):
        value = getattr(timing, field_name, None)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{field_name} must be a numeric nanosecond value"
        try:
            finite = math.isfinite(value)
        except (OverflowError, TypeError, ValueError):
            finite = False
        if not finite:
            return f"{field_name} must be finite"
    if timing.unit != "ns":
        return "unit must be exactly 'ns'"
    if timing.duration < 0:
        return "duration must be non-negative"
    if timing.end < timing.begin:
        return "end must not precede begin"
    return None


def _pcie_instruction_timings_error(timings: Sequence[Any]) -> Optional[str]:
    """Validate all decoded PCIe rows through one shared acceptance predicate."""

    if not timings:
        return "no decoded instruction timings"
    for index, timing in enumerate(timings):
        error = _pcie_instruction_timing_error(timing)
        if error is not None:
            return f"event {index}: {error}"
    return None


def _is_valid_pcie_raw_trace_artifact(raw_path: PathLike) -> bool:
    """Return whether one recorder directory contains non-empty profile data."""

    path = Path(raw_path)
    if not path.is_dir():
        return False
    try:
        profile_files = tuple(candidate for candidate in path.iterdir() if candidate.is_file() and
                              _PCIE_RAW_PROFILE_FILE_RE.fullmatch(candidate.name) is not None)
        return bool(profile_files) and all(
            candidate.stat().st_size > 0 for candidate in profile_files)
    except OSError:
        # Recorder output may disappear or become inaccessible between worker
        # exit and collection.  Such a path is not usable raw evidence.
        return False


@dataclass(frozen=True)
class TPURawInstruction:
    """One decoded textual CModel command-dump line, without a fabricated time."""

    engine: str
    core_id: int
    command_id: Optional[int]
    opcode: Optional[str]
    text: str
    source_path: Path


@dataclass(frozen=True)
class TPUProfileReport:
    """Artifacts and optional decoded timing rows from one profile worker."""

    config: TPUProfilingConfig
    output_dir: Path
    command: Tuple[str, ...]
    stdout_path: Path
    stderr_path: Path
    raw_trace_files: Tuple[Path, ...]
    raw_instructions: Tuple[TPURawInstruction, ...]
    parser_status: str
    parser_message: Optional[str] = None
    perfai_report_path: Optional[Path] = None
    perfai_report_paths: Tuple[Path, ...] = ()
    decoded_report_path: Optional[Path] = None
    decoded_report_paths: Tuple[Path, ...] = ()
    decoder_identity: Mapping[str, str] = field(default_factory=dict)
    timeline_events: Tuple[TPUInstructionTiming, ...] = ()
    instruction_timings: Tuple[TPUInstructionTiming, ...] = ()

    @property
    def has_raw_trace(self) -> bool:
        if self.config.runtime_mode == "pcie":
            return any(_is_valid_pcie_raw_trace_artifact(path) for path in self.raw_trace_files)
        return bool(self.raw_trace_files)

    @property
    def has_instruction_timings(self) -> bool:
        return bool(self.instruction_timings)


def _copy_environment(extra: Optional[Mapping[str, str]]) -> MutableMapping[str, str]:
    environment: MutableMapping[str, str] = dict(os.environ)
    if extra is not None:
        for key, value in extra.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("TPU profiling environment keys and values must be strings.")
            environment[key] = value
    return environment


def _pcie_decoder_environment(config: TPUProfilingConfig,
                              environment: Optional[Mapping[str, str]]) -> MutableMapping[str, str]:
    """Build an offline-only decoder environment without PCIe permissions.

    An explicit ``pcie_decoder_pythonpath`` replaces, rather than extends, the
    caller's ``PYTHONPATH``.  The worker environment is never modified, so a
    separately installed decoder stack cannot shadow compiler dependencies.
    When no isolated decoder path is configured, the decoder inherits the
    caller's ``PYTHONPATH``.
    """

    decoder_env = _copy_environment(environment)
    for key in _PCIE_HARDWARE_ENVIRONMENT_KEYS:
        decoder_env.pop(key, None)
    if config.pcie_decoder_pythonpath:
        decoder_env["PYTHONPATH"] = os.pathsep.join(
            str(path) for path in config.pcie_decoder_pythonpath)
        decoder_env["PYTHONNOUSERSITE"] = "1"
    return decoder_env


def _pcie_decoder_python(config: TPUProfilingConfig) -> str:
    return str(config.pcie_decoder_python or sys.executable)


def _parse_pcie_decoder_identity(stdout: str) -> Mapping[str, str]:
    """Extract and validate the decoder helper's machine-readable identity."""

    identity_line = next((line for line in reversed(stdout.splitlines())
                          if line.startswith(_PCIE_DECODER_IDENTITY_PREFIX)), None)
    if identity_line is None:
        raise ValueError("The offline PCIe decoder did not report its package/API identity.")
    payload = json.loads(identity_line[len(_PCIE_DECODER_IDENTITY_PREFIX):])
    required = ("package", "package_version", "parser_api")
    if not isinstance(payload, dict) or any(
            not isinstance(payload.get(key), str) or not payload[key] for key in required):
        raise ValueError("The offline PCIe decoder reported an invalid package/API identity.")
    return {key: payload[key] for key in required}


def _read_pcie_decoder_identity(report_paths: Sequence[Path]) -> Mapping[str, str]:
    """Return the common decoder identity recorded in canonical reports."""

    identities = []
    for path in report_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_identity = payload.get("decoder_identity") if isinstance(payload, dict) else None
        if not isinstance(raw_identity, dict):
            raise ValueError(f"TileLang PCIe profile has no decoder identity: {path}")
        identity = {
            key: raw_identity.get(key) for key in ("package", "package_version", "parser_api")
        }
        if any(not isinstance(value, str) or not value for value in identity.values()):
            raise ValueError(f"TileLang PCIe profile has an invalid decoder identity: {path}")
        identities.append(identity)
    if not identities:
        return {}
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("TileLang PCIe profile reports were produced by different decoders.")
    return identities[0]


def _profile_output_dir(config: TPUProfilingConfig) -> Path:
    if config.output_dir is None:
        # A profile is an inspection artifact, not an anonymous scratch file.
        # Keep the default under the caller's working directory so successful
        # sessions never leave an undiscoverable directory in /tmp.
        output_root = Path.cwd().resolve() / "tilelang-tpu-profiles"
    else:
        output_root = Path(config.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    # Never emulate PPL's ``rmtree(<target>/profiling)`` behaviour. A caller
    # may preserve many regression profiles under one root, so every run owns
    # a new subdirectory and cannot erase a prior trace.
    return Path(tempfile.mkdtemp(prefix=f"{config.label}-", dir=output_root))


def _profile_deadline(timeout_s: float) -> float:
    """Return the absolute deadline shared by worker, lock, and PerfAI."""

    return time.monotonic() + timeout_s


def _remaining_timeout(deadline: float) -> float:
    """Return remaining wall-clock budget without ever extending a session."""

    return max(0.0, deadline - time.monotonic())


def _spawn_guarded_profile_process(command: Sequence[str], *, cwd: Path,
                                   environment: Mapping[str, str]) -> subprocess.Popen:
    """Spawn ``command`` below a parent-death-aware process-tree supervisor.

    The direct child is a standalone Python supervisor in its own session.
    It arms Linux parent-death ``SIGTERM`` immediately after exec, with a
    double parent-PID check that closes the setup race, then keeps the real
    worker/PerfAI command in its private process group.  Therefore an outer
    timeout that terminates pytest cannot leave an ordinary CModel worker or
    AutoRunner descendant running.
    """

    if not sys.platform.startswith("linux"):
        raise TPUProfilingError("Safe TPU instruction profiling requires Linux PR_SET_PDEATHSIG; "
                                "refusing to create a detached profile worker on this platform.")
    if not _PROFILE_SUPERVISOR_PATH.is_file():
        raise TPUProfilingError(f"TPU profile supervisor is missing: {_PROFILE_SUPERVISOR_PATH}")
    return subprocess.Popen(
        [
            sys.executable,
            str(_PROFILE_SUPERVISOR_PATH), "--parent-pid",
            str(os.getpid()), "--", *command
        ],
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=True,
    )


_PROCESS_TERMINATION_GRACE_S = 5.0
_PROCESS_KILL_GRACE_S = 5.0
_PROCESS_PIPE_DRAIN_S = 5.0
_PROCESS_GROUP_POLL_S = 0.02


def _process_group_exists(process_group: int) -> bool:
    """Return whether a saved guarded process group still has members.

    The supervisor is the group leader, but it can exit before a background
    compiler/decoder child.  Consequently this probe must use the saved PGID
    rather than the leader's ``poll()`` state.
    """

    try:
        os.killpg(process_group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # The private group should be owned by this process.  If that invariant
        # is ever broken, treating the group as live is safer than declaring a
        # TPU worker tree clean.
        return True


def _wait_for_process_group_exit(process: subprocess.Popen, process_group: int,
                                 timeout_s: float) -> bool:
    """Poll both the saved PGID and direct supervisor within one hard bound."""

    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        leader_reaped = process.poll() is not None
        if not _process_group_exists(process_group):
            return leader_reaped or process.poll() is not None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(_PROCESS_GROUP_POLL_S, remaining))


def _kill_process_group(process: subprocess.Popen, *, process_group: Optional[int] = None) -> bool:
    """Stop a guarded worker and all of its children without touching its parent shell.

    The direct process is the lightweight supervisor.  ``SIGTERM`` makes its
    handler kill the private group that contains the command and ordinary
    descendants before exiting.  The SIGKILL fallback only covers a broken
    supervisor and is intentionally not presented as a substitute for that
    guarded path.  Group cleanup is deliberately independent of the direct
    supervisor's state: a successful command may have backgrounded a child and
    allowed the group leader to exit first.
    """

    saved_process_group = process.pid if process_group is None else process_group
    if _process_group_exists(saved_process_group):
        try:
            os.killpg(saved_process_group, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            # Retain the bounded escalation path.  Its final group probe makes
            # an unclean result visible to the caller without masking the
            # original timeout/cancellation exception.
            pass
    if _wait_for_process_group_exit(process, saved_process_group, _PROCESS_TERMINATION_GRACE_S):
        return True
    if _process_group_exists(saved_process_group):
        try:
            os.killpg(saved_process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            pass
    # A process stuck in uninterruptible kernel sleep cannot be reaped by user
    # space.  Never turn the TPU watchdog into another unbounded wait.
    return _wait_for_process_group_exit(process, saved_process_group, _PROCESS_KILL_GRACE_S)


@dataclass(frozen=True)
class _GuardedProcessOutput:
    stdout: str
    stderr: str
    left_live_descendant: bool
    cleanup_complete: bool


@dataclass(frozen=True)
class TPUSupervisedCommandResult:
    """Bounded result from a command running below the TPU process supervisor."""

    stdout: str
    stderr: str
    returncode: Optional[int]
    timed_out: bool
    left_live_descendant: bool
    cleanup_complete: bool
    process_group: int


def _communicate_guarded_process(process: subprocess.Popen, *,
                                 timeout: float) -> _GuardedProcessOutput:
    """Collect one guarded command and reject a successful leader-only exit.

    ``Popen.communicate`` waits only for the direct supervisor.  A descendant
    that redirected its stdio can survive that wait unnoticed, so always probe
    the PGID captured before communicate and clean it within the same bounded
    TERM/KILL policy used by timeout and cancellation paths.
    """

    process_group = process.pid
    stdout, stderr = process.communicate(timeout=timeout)
    left_live_descendant = _process_group_exists(process_group)
    cleanup_complete = True
    if left_live_descendant:
        cleanup_complete = _kill_process_group(process, process_group=process_group)
        if not cleanup_complete:
            diagnostic = ("TileLang TPU watchdog: a completed supervisor left a live "
                          "descendant and its process group could not be fully reaped")
            stderr = f"{stderr}\n{diagnostic}\n" if stderr else diagnostic + "\n"
    return _GuardedProcessOutput(
        stdout=stdout,
        stderr=stderr,
        left_live_descendant=left_live_descendant,
        cleanup_complete=cleanup_complete,
    )


def _timeout_output_as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _terminate_and_collect(process: subprocess.Popen) -> TPUSupervisedCommandResult:
    """Terminate a guarded group and drain its pipes with a final hard bound.

    A killed command descendant can retain an inherited pipe while stuck in a
    driver call.  ``Popen.communicate()`` without a timeout would then defeat
    the outer hardware watchdog.  Preserve whatever output is available,
    close this process's pipe handles, and return even in that pathological
    case.
    """

    process_group = process.pid
    reaped = _kill_process_group(process, process_group=process_group)
    try:
        stdout, stderr = process.communicate(timeout=_PROCESS_PIPE_DRAIN_S)
        if not reaped:
            diagnostic = ("TileLang TPU watchdog: process group "
                          f"{process_group} could not be fully reaped")
            stderr = f"{stderr}\n{diagnostic}\n" if stderr else diagnostic + "\n"
        return TPUSupervisedCommandResult(
            stdout=stdout,
            stderr=stderr,
            returncode=process.poll(),
            timed_out=True,
            left_live_descendant=not reaped,
            cleanup_complete=reaped,
            process_group=process_group,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _timeout_output_as_text(exc.output)
        stderr = _timeout_output_as_text(exc.stderr)
        diagnostic = ("TileLang TPU watchdog: process group did not close its output "
                      f"pipes within {_PROCESS_PIPE_DRAIN_S:g}s after termination")
        if not reaped:
            diagnostic += "; the supervisor process also could not be reaped"
        stderr = f"{stderr}\n{diagnostic}\n" if stderr else diagnostic + "\n"
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                with suppress(OSError):
                    pipe.close()
        return TPUSupervisedCommandResult(
            stdout=stdout,
            stderr=stderr,
            returncode=process.poll(),
            timed_out=True,
            left_live_descendant=not reaped,
            cleanup_complete=reaped,
            process_group=process_group,
        )


def run_tpu_supervised_command(
    command: Sequence[PathLike],
    *,
    cwd: PathLike,
    environment: Optional[Mapping[str, str]] = None,
    timeout_s: float,
    fail_closed_device_id: Optional[int] = None,
) -> TPUSupervisedCommandResult:
    """Run a finite command without an unbounded timeout cleanup path.

    Timeout and process-group cleanup state are returned explicitly so a board
    caller cannot mistake a timed-out command for a cleanly stopped process
    tree.
    """

    if not command:
        raise ValueError("supervised TPU command must not be empty")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("supervised TPU command timeout must be finite and positive")
    if fail_closed_device_id is not None and not _pcie_device_lock_owned(fail_closed_device_id):
        raise TPUProfilingError(
            "fail-closed supervised command requires ownership of its PCIe device lock")
    process = _spawn_guarded_profile_process(
        tuple(os.fspath(item) for item in command),
        cwd=Path(cwd).expanduser().resolve(),
        environment=_copy_environment(environment),
    )
    try:
        output = _communicate_guarded_process(process, timeout=timeout_s)
        return TPUSupervisedCommandResult(
            stdout=output.stdout,
            stderr=output.stderr,
            returncode=process.returncode,
            timed_out=False,
            left_live_descendant=output.left_live_descendant,
            cleanup_complete=output.cleanup_complete,
            process_group=process.pid,
        )
    except subprocess.TimeoutExpired:
        return _terminate_and_collect(process)
    except BaseException:
        terminated = _terminate_and_collect(process)
        if fail_closed_device_id is not None and not terminated.cleanup_complete:
            _quarantine_pcie_device(
                fail_closed_device_id,
                process_group=terminated.process_group,
                reason="supervised PCIe command cancellation could not fully reap its group",
            )
        raise


def _remove_worker_transient_caches(output_dir: Path) -> None:
    """Remove import-time caches from the profiler-owned result directory."""

    cache_path = output_dir / f".pkl_memoize_py{sys.version_info.major}"
    if cache_path.is_symlink() or cache_path.is_file():
        cache_path.unlink()
    elif cache_path.is_dir():
        shutil.rmtree(cache_path)


def _write_worker_logs(output_dir: Path, stdout: str, stderr: str) -> Tuple[Path, Path]:
    stdout_path = output_dir / "worker.stdout.log"
    stderr_path = output_dir / "worker.stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    # TVM's test-only pickle_memoize decorator creates this cache in cwd at
    # import time.  Profile directories are durable evidence, not cache roots;
    # the matrix-owned TILELANG_CACHE_DIR/TMPDIR carry all disposable state.
    _remove_worker_transient_caches(output_dir)
    return stdout_path, stderr_path


def _write_parser_logs(output_dir: Path, stdout: str, stderr: str) -> Tuple[Path, Path]:
    stdout_path = output_dir / "perfai.stdout.log"
    stderr_path = output_dir / "perfai.stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return stdout_path, stderr_path


@contextmanager
def _exclusive_perfai_workspace(perfai_root: Path, deadline: float):
    """Serialize TileLang sessions that share PPL's mutable PerfAI workspace.

    PPL's own driver runs AutoRunner from the PerfAI directory, whose
    ``auto_build`` area is mutable.  Do not copy PPL's destructive cleanup;
    instead, own a Linux abstract-socket name keyed by the resolved root. This
    is a best-effort inter-process lock for TileLang sessions on the same
    machine; users must still keep unrelated direct PPL invocations out of
    that root.
    """

    # Linux abstract UNIX sockets are released automatically when the owning
    # process closes them.  Unlike a flock file in /tmp, this cannot leave a
    # stale intermediate behind after SIGKILL.
    token = hashlib.sha256(str(perfai_root).encode("utf-8")).hexdigest()[:20]
    address = "\0tilelang-perfai-" + token
    lock_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    while True:
        try:
            lock_socket.bind(address)
            break
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                lock_socket.close()
                raise
            if time.monotonic() >= deadline:
                lock_socket.close()
                raise _PerfAIWorkspaceBusy(f"PerfAI workspace is busy: {perfai_root}") from None
            time.sleep(0.05)
    try:
        yield
    finally:
        lock_socket.close()


def _find_perfai_root(config: TPUProfilingConfig, environment: Mapping[str, str]) -> Optional[Path]:
    candidates = []
    if config.perfai_root is not None:
        candidates.append(Path(config.perfai_root))
    if environment.get("PPL_PERFAI_ROOT"):
        candidates.append(Path(environment["PPL_PERFAI_ROOT"]))
    if environment.get("PPL_THIRD_PARTY_PATH"):
        candidates.append(Path(environment["PPL_THIRD_PARTY_PATH"]) / "PerfAI")
    for candidate in candidates:
        runner = candidate.expanduser() / "AutoRunner.sh"
        if runner.is_file():
            return candidate.expanduser().resolve()
    return None


def _perfai_chip_name(config: TPUProfilingConfig) -> str:
    """Return the PPL chip spelling expected by PerfAI's ``-e`` flag.

    PPL invokes ``get_chip_name(chip_arch)`` before calling AutoRunner.  Its
    mapping distinguishes the RV profile decoder as ``sg2260erv`` even though
    TileLang deliberately exposes the physical chip as ``sg2260e`` plus a
    separate programming-model axis.  Keep that vendor-only spelling at this
    adapter boundary; it is not a third TileLang chip target.
    """

    if config.programming_model == "rv":
        return "sg2260erv"
    # This preserves PPL's CModel special case in ppl_compile.py: BM1690's
    # PerfAI target is named sg2260.
    return "sg2260" if config.chip == "bm1690" else config.chip


def _pcie_profile_arch(config: TPUProfilingConfig) -> str:
    """Return the decoder architecture used by PPL's TPUv7 PCIe path."""

    if config.programming_model == "rv":
        return "tpub_7_1_e_rv"
    return config.target_spec.chip_spec.ppl_arch


def _extract_js_array(text: str, variable: str) -> Sequence[Any]:
    """Read one array assignment from the small JS data file PerfAI emits.

    The output is data-only JavaScript, but it is not guaranteed to be strict
    JSON.  Try JSON first and then Python literal syntax after translating the
    three JSON scalar spellings.  The bracket scanner avoids a fragile
    line-oriented parser and handles arrays formatted over many lines.
    """

    match = re.search(rf"(?:let|var|const)\s+{re.escape(variable)}\s*=", text)
    if match is None:
        raise ValueError(f"PerfAI profile data does not define {variable!r}.")
    start = text.find("[", match.end())
    if start < 0:
        raise ValueError(f"PerfAI profile data has no array value for {variable!r}.")

    depth = 0
    quote: Optional[str] = None
    escaped = False
    end = None
    for index in range(start, len(text)):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is None:
        raise ValueError(f"PerfAI profile data has an unterminated {variable!r} array.")

    source = text[start:end]
    try:
        value = json.loads(source)
    except json.JSONDecodeError:
        python_source = re.sub(r"\btrue\b", "True", source)
        python_source = re.sub(r"\bfalse\b", "False", python_source)
        python_source = re.sub(r"\bnull\b", "None", python_source)
        try:
            value = ast.literal_eval(python_source)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Could not parse PerfAI {variable!r} array as data.") from exc
    if not isinstance(value, list):
        raise ValueError(f"PerfAI {variable!r} is not an array.")
    return value


def _timing_unit(headers: Sequence[Any]) -> str:
    names = " ".join(str(item).lower() for item in headers)
    if "cycle" in names:
        return "cycles"
    if "usec" in names or "_us" in names or "(us" in names:
        return "us"
    if "_ms" in names or "(ms" in names:
        return "ms"
    return "unknown"


def _raw_command_id(text: str) -> Optional[int]:
    match = re.search(r"\b(?:cmd_id|bd_id|gdma_id|sdma_id)=(\d+)", text)
    return int(match.group(1)) if match is not None else None


def _raw_opcode(text: str) -> Optional[str]:
    match = re.search(r"\b(?:bd_func|gdma_func|sdma_func)=([^\s]+)", text)
    return match.group(1) if match is not None else None


def _integer_field(fields: Mapping[str, Any], *names: str) -> Optional[int]:
    normalized = {str(key).strip().lower(): value for key, value in fields.items()}
    for name in names:
        value = normalized.get(name)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
            return int(value.strip())
    return None


def _string_field(fields: Mapping[str, Any], *names: str) -> Optional[str]:
    normalized = {str(key).strip().lower(): value for key, value in fields.items()}
    for name in names:
        value = normalized.get(name)
        if isinstance(value, str) and value:
            return value
    return None


def _timing_identity(
        fields: Mapping[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """Best-effort identity extraction without assuming one PerfAI schema.

    PerfAI releases have changed the extra columns in ``time_data``.  Preserve
    the original row in ``fields`` in all cases, and only project familiar
    fields when their names make the interpretation unambiguous.
    """

    core_id = _integer_field(fields, "core", "core_id", "coreid", "core_idx")
    command_id = _integer_field(fields, "cmd_id", "command_id", "bd_id", "gdma_id", "sdma_id")
    opcode = _string_field(fields, "opcode", "op_name", "func_name", "instruction")
    # PPL's historical PerfWeb schema stores ``bd_id=...`` / ``gdma_id=...``
    # in ``func_type`` rather than a separate command-id column. Preserve that
    # version-specific convention as a best-effort projection.
    for text in (_string_field(fields,
                               "func_type"), _string_field(fields,
                                                           "type"), _string_field(fields, "info")):
        if text is None:
            continue
        if command_id is None:
            command_id = _raw_command_id(text)
        if opcode is None:
            opcode = _raw_opcode(text)
    info = _string_field(fields, "info")
    if opcode is None and info is not None and "<br>" in info:
        # PPL's static BD/GDMA rows prefix the human-readable function name
        # before the first HTML line break. Do not mislabel metric-only rows.
        candidate = info.split("<br>", 1)[0].strip()
        if candidate and not re.match(r"(?:cycle|speed|bytes|ops_ratio)=", candidate):
            opcode = candidate
    return core_id, command_id, opcode


def parse_cmodel_raw_instruction_dumps(
        raw_trace_files: Sequence[PathLike]) -> Tuple[TPURawInstruction, ...]:
    """Parse CModel text dumps into command metadata without inventing latency.

    The emulator's ``*.txt`` sidecars describe engine, core, dependency, and
    command-id information. They do not contain measured begin/end timestamps,
    so callers receive explicit raw records while PerfAI is unavailable rather
    than a misleading approximation labelled as execution time.
    """

    records = []
    for raw_path_like in raw_trace_files:
        raw_path = Path(raw_path_like)
        match = _RAW_TRACE_RE.fullmatch(raw_path.name)
        if match is None:
            continue
        engine = match.group("engine").lower()
        core_id = int(match.group("core"))
        try:
            lines = raw_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line in lines:
            text = line.strip()
            if not text:
                continue
            records.append(
                TPURawInstruction(
                    engine=engine,
                    core_id=core_id,
                    command_id=_raw_command_id(text),
                    opcode=_raw_opcode(text),
                    text=text,
                    source_path=raw_path,
                ))
    return tuple(records)


def parse_perfai_timeline_events(profile_data_path: PathLike) -> Tuple[TPUInstructionTiming, ...]:
    """Extract every PerfAI timeline event from a web-report data file.

    PPL's own ``profiling_parser.py`` only prints an aggregate
    ``summary_data`` table.  PerfAI's ``time_data`` also contains CPU,
    subnet, and layer events, so this low-level parser preserves all rows
    rather than calling all of them instructions.
    """

    path = Path(profile_data_path)
    source = path.read_text(encoding="utf-8")
    categories = _extract_js_array(source, "categories")
    headers = _extract_js_array(source, "time_header")
    rows = _extract_js_array(source, "time_data")
    if len(headers) < 3:
        raise ValueError("PerfAI time_header must contain category, begin, and end fields.")

    unit = _timing_unit(headers)
    timings = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            raise ValueError(f"PerfAI time_data row {row_index} is not a valid timeline row.")
        category = row[0]
        if isinstance(category, int) and 0 <= category < len(categories):
            engine = str(categories[category])
        else:
            engine = str(category)
        begin, end = row[1], row[2]
        duration: Optional[float] = None
        if isinstance(begin, (int, float)) and isinstance(end, (int, float)):
            duration = float(end - begin)
        fields = {str(headers[index]): row[index] for index in range(min(len(headers), len(row)))}
        core_id, command_id, opcode = _timing_identity(fields)
        timings.append(
            TPUInstructionTiming(
                engine=engine,
                begin=begin,
                end=end,
                duration=duration,
                unit=unit,
                core_id=core_id,
                command_id=command_id,
                opcode=opcode,
                fields=fields,
            ))
    return tuple(timings)


def parse_perfai_instruction_timings(
        profile_data_path: PathLike) -> Tuple[TPUInstructionTiming, ...]:
    """Return only recognized TPU command-engine rows from a PerfAI timeline.

    This deliberately filters PPL's host/subnet timeline rows.  Unknown
    vendor categories remain available through :func:`parse_perfai_timeline_events`
    instead of being misrepresented as individual TPU instructions.
    """

    return tuple(
        event for event in parse_perfai_timeline_events(profile_data_path)
        if event.engine.strip().lower() in _DEVICE_COMMAND_ENGINES)


def parse_pcie_decoded_instruction_timings(
        report_path: PathLike) -> Tuple[TPUInstructionTiming, ...]:
    """Read TileLang's stable JSON projection of a bigTpuProfile result."""

    path = Path(report_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported TileLang PCIe profile schema: {path}")
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise ValueError(f"TileLang PCIe profile has no events array: {path}")
    events = []
    for index, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, dict):
            raise ValueError(f"PCIe profile event {index} is not an object: {path}")
        engine = raw_event.get("engine")
        begin = raw_event.get("begin")
        end = raw_event.get("end")
        unit = raw_event.get("unit")
        fields = raw_event.get("fields", {})
        if not isinstance(engine, str) or not isinstance(unit, str) or \
                not isinstance(begin, (int, float)) or \
                not isinstance(end, (int, float)) or \
                not isinstance(fields, dict):
            raise ValueError(f"PCIe profile event {index} has invalid fields: {path}")
        events.append(
            TPUInstructionTiming(
                engine=engine,
                begin=begin,
                end=end,
                duration=float(end - begin),
                unit=unit,
                core_id=raw_event.get("core_id"),
                command_id=raw_event.get("command_id"),
                opcode=raw_event.get("opcode"),
                fields=fields,
            ))
        error = _pcie_instruction_timing_error(events[-1])
        if error is not None:
            raise ValueError(f"PCIe profile event {index} is invalid: {error}: {path}")
    return tuple(events)


class TPUInstructionProfiler:
    """Run one fresh TileLang TPU test worker with PPL-compatible profiling."""

    def __init__(self, config: TPUProfilingConfig):
        self.config = config

    @staticmethod
    def exclusive_pcie_device(device_id: int):
        """Own one logical PCIe device across a complete supervised session."""

        if isinstance(device_id,
                      bool) or not isinstance(device_id, int) or not (0 <= device_id <= 2**31 - 1):
            raise ValueError("device_id must be a non-negative 32-bit integer")
        return _exclusive_pcie_device_lock(device_id)

    @staticmethod
    def fail_closed_pcie_device(
        device_id: int,
        *,
        reason: str,
        process_group: Optional[int] = None,
    ) -> Path:
        """Permanently stop further launches when a board is no longer proven safe.

        The caller must already own :meth:`exclusive_pcie_device`. The active
        session is first marked unsafe in memory, ensuring its durable session
        marker survives every exception (including ``KeyboardInterrupt``).
        Creation of the more descriptive quarantine marker is then attempted.
        ``process_group=None`` records that no responsible PGID was observed,
        which is expected for an inconclusive postflight board-health check.

        Neither marker is removed automatically. An operator must inspect and
        recover the board before clearing them.
        """

        if isinstance(device_id,
                      bool) or not isinstance(device_id, int) or not (0 <= device_id <= 2**31 - 1):
            raise ValueError("device_id must be a non-negative 32-bit integer")
        return _quarantine_pcie_device(
            device_id,
            process_group=process_group,
            reason=reason,
        )

    @staticmethod
    def run_supervised_pcie_probe(
        device_id: int,
        command: Sequence[PathLike],
        *,
        cwd: PathLike,
        environment: Optional[Mapping[str, str]] = None,
        timeout_s: float = 10.0,
    ) -> TPUSupervisedCommandResult:
        """Run a read-only board probe under an already-owned device lock.

        Incomplete process-group cleanup permanently quarantines the device.
        The method does not acquire the lock itself so callers can keep one
        lock across a complete promotion session; nested public lock ownership
        remains available for standalone probes.
        """

        if isinstance(device_id,
                      bool) or not isinstance(device_id, int) or not (0 <= device_id <= 2**31 - 1):
            raise ValueError("device_id must be a non-negative 32-bit integer")
        if not _pcie_device_lock_owned(device_id):
            raise TPUProfilingError(
                f"PCIe probe requires ownership of TPU device {device_id}'s session lock")
        _assert_pcie_device_not_quarantined(device_id)
        result = run_tpu_supervised_command(
            command,
            cwd=cwd,
            environment=environment,
            timeout_s=timeout_s,
            fail_closed_device_id=device_id,
        )
        if not result.cleanup_complete:
            marker = _quarantine_pcie_device(
                device_id,
                process_group=result.process_group,
                reason="read-only PCIe probe process group could not be fully reaped",
            )
            raise TPUProfilingError(f"PCIe probe left process group {result.process_group} alive; "
                                    f"TPU device {device_id} is quarantined by {marker}")
        return result

    def _validate_ppl_dependencies(self, environment: Mapping[str, str]) -> None:
        """Fail before spawning when a TileLang/PPL worker declares its SDK.

        The profiler also accepts generic commands used by parser/unit tests,
        so an absent ``PPL_PROJECT_ROOT`` is not itself an error here.  Once a
        worker supplies the root, however, the runtime is known and only that
        runtime's profiling contract is validated.
        """

        ppl_root = environment.get("PPL_PROJECT_ROOT")
        if ppl_root is None:
            return
        if not ppl_root.strip():
            raise TPUProfilingError("PPL_PROJECT_ROOT must not be empty for a TPU profile worker.")
        try:
            layout = resolve_ppl_layout(ppl_root, self.config.chip)
            layout.require_profiling(self.config.runtime_mode, environment=environment)
        except (OSError, ValueError) as exc:
            raise TPUProfilingError("PPL 1.7 dependency preflight failed for "
                                    f"{self.config.runtime_mode} profiling: {exc}") from exc

    def preflight_pcie_decoder(self,
                               *,
                               environment: Optional[Mapping[str,
                                                             str]] = None) -> Mapping[str, str]:
        """Validate the offline decoder without loading or dispatching a TPU.

        The check runs the decoder helper under the same process-tree watchdog
        as profiling, but with every board/profile permission removed.  Matrix
        runners that require decoded timing can call this before their first
        hardware worker and fail without risking a launch when the selected
        Python environment lacks the supported structured API.
        """

        if self.config.runtime_mode != "pcie":
            raise TPUProfilingError("preflight_pcie_decoder only supports runtime_mode='pcie'.")
        if not _PCIE_PROFILE_DECODER_PATH.is_file():
            raise TPUProfilingError(f"TileLang's offline PCIe profile decoder helper is missing: "
                                    f"{_PCIE_PROFILE_DECODER_PATH}")
        decoder_env = _pcie_decoder_environment(self.config, environment)
        command = (
            _pcie_decoder_python(self.config),
            str(_PCIE_PROFILE_DECODER_PATH),
            "--preflight",
        )
        decoder: Optional[subprocess.Popen] = None
        timeout = min(float(self.config.timeout_s), _PCIE_DECODER_PREFLIGHT_TIMEOUT_S)
        try:
            decoder = _spawn_guarded_profile_process(
                command,
                cwd=_PCIE_PROFILE_DECODER_PATH.parent.resolve(),
                environment=decoder_env,
            )
            guarded_output = _communicate_guarded_process(decoder, timeout=timeout)
        except subprocess.TimeoutExpired:
            assert decoder is not None
            terminated = _terminate_and_collect(decoder)
            detail = (terminated.stderr or terminated.stdout).strip()
            raise TPUProfilingError(
                f"Offline PCIe decoder preflight exceeded {timeout:g}s and its process "
                f"group was terminated{': ' + detail if detail else '.'}") from None
        except (OSError, subprocess.SubprocessError, TPUProfilingError) as exc:
            raise TPUProfilingError(f"Could not start offline PCIe decoder preflight: {exc}") \
                from exc
        except BaseException:
            if decoder is not None:
                with suppress(Exception):
                    _terminate_and_collect(decoder)
            raise

        if guarded_output.left_live_descendant:
            cleanup = ("was terminated"
                       if guarded_output.cleanup_complete else "could not be fully reaped")
            raise TPUProfilingError(
                "Offline PCIe decoder preflight left a live descendant; its supervised "
                f"process group {cleanup}.")
        detail = (guarded_output.stderr or guarded_output.stdout).strip()
        if decoder.returncode == 3:
            raise TPUProfilingError(
                "Offline PCIe decoding requires preinstalled bigTpuProfile; no package "
                f"was installed automatically{': ' + detail if detail else '.'}")
        if decoder.returncode == 4:
            raise TPUProfilingError("The installed bigTpuProfile does not provide the callable "
                                    f"BMProfileParserPerfAI.parse API required by TileLang"
                                    f"{': ' + detail if detail else '.'}")
        if decoder.returncode != 0:
            raise TPUProfilingError(
                f"Offline PCIe decoder preflight exited with status {decoder.returncode}"
                f"{': ' + detail if detail else '.'}")
        try:
            return _parse_pcie_decoder_identity(guarded_output.stdout)
        except (json.JSONDecodeError, ValueError) as exc:
            raise TPUProfilingError(str(exc)) from exc

    def pcie_profile_environment_overrides(self,
                                           environment: Optional[Mapping[str, str]] = None
                                          ) -> Mapping[str, str]:
        """Return PPL's recorder overrides after PCIe safety preflight.

        The returned mapping is intentionally not a complete environment and
        this method never loads a library or touches a board.  The generated
        PCIe host enables TPUDNN recording only when :meth:`run_pcie` also marks
        the child as an isolated profiling session.

        No loading, initialization, or board dispatch occurs here.
        """

        if self.config.runtime_mode != "pcie":
            raise ValueError("pcie_profile_environment_overrides requires runtime_mode='pcie'.")
        resolved = _copy_environment(environment)
        if resolved.get("TILELANG_TPU_ALLOW_PCIE_LOAD") != "1":
            raise TPUProfilingError("PCIe profiling requires TILELANG_TPU_ALLOW_PCIE_LOAD=1; "
                                    "this helper does not initialize a board.")
        if resolved.get("TILELANG_TPU_ALLOW_PCIE_PROFILE") != "1":
            raise TPUProfilingError("PCIe profiling requires the separate acknowledgement "
                                    "TILELANG_TPU_ALLOW_PCIE_PROFILE=1.")
        device_id = resolved.get("TILELANG_TPU_DEVICE_ID", "")
        if re.fullmatch(r"[0-9]+", device_id) is None or int(device_id) > 2**31 - 1:
            raise TPUProfilingError("PCIe profiling requires a non-negative integer "
                                    "TILELANG_TPU_DEVICE_ID.")
        return {
            "BMLIB_ENABLE_ALL_PROFILE": "1",
            "PROFILE_RECORD_SIZE": str(self.config.profile_record_size),
            "PROFILE_BOOK_KEEPING": str(self.config.profile_book_keeping),
        }

    def run_pcie(self,
                 command: Sequence[PathLike],
                 *,
                 environment: Optional[Mapping[str, str]] = None) -> TPUProfileReport:
        """Run one PCIe worker while holding the shared per-device lock."""

        if self.config.runtime_mode != "pcie":
            raise TPUProfilingError("run_pcie only supports runtime_mode='pcie'.")
        resolved = _copy_environment(environment)
        self.pcie_profile_environment_overrides(resolved)
        device_id = int(resolved["TILELANG_TPU_DEVICE_ID"])
        with _exclusive_pcie_device_lock(device_id):
            return self._run_pcie_locked(command, environment=environment)

    def _run_pcie_locked(self,
                         command: Sequence[PathLike],
                         *,
                         environment: Optional[Mapping[str, str]] = None) -> TPUProfileReport:
        """Run one explicitly authorized PCIe profile worker.

        The generated TileLang host wraps the same ``tpuRt`` stream/module in a
        TPUDNN handle, enables recording, performs exactly one kernel launch and
        synchronization, and disables recording before copying results back.
        The worker has the same parent-death/process-group watchdog as CModel.

        Offline decoding never installs packages.  When ``bigTpuProfile`` is
        absent, raw ``cdm_profile_data_dev*`` artifacts are retained and the
        report returns ``parser_status='unavailable'``.
        """

        if self.config.runtime_mode != "pcie":
            raise TPUProfilingError("run_pcie only supports runtime_mode='pcie'.")
        if not command:
            raise ValueError("TPU profiling command must not be empty.")
        normalized_command = tuple(os.fspath(item) for item in command)
        deadline = _profile_deadline(float(self.config.timeout_s))
        worker_env = _copy_environment(environment)
        worker_env.pop("TPU_RT_CORE_NUM", None)
        worker_env.update(self.pcie_profile_environment_overrides(worker_env))
        device_id = int(worker_env["TILELANG_TPU_DEVICE_ID"])
        self._validate_ppl_dependencies(worker_env)
        output_dir = _profile_output_dir(self.config)
        worker_env.pop("FILE_DUMP_CMD", None)
        worker_env["TILELANG_TPU_PROFILE_SESSION"] = "1"
        worker_env["TILELANG_TPU_PROFILE_OUTPUT_DIR"] = str(output_dir)
        worker_env["TILELANG_TPU_PROFILE_CHIP"] = self.config.chip
        worker_env["TILELANG_TPU_PROFILE_PROGRAMMING_MODEL"] = \
            self.config.programming_model
        worker_env["TILELANG_TPU_PROFILE_RUNTIME_MODE"] = "pcie"
        # The host template independently suppresses benchmark loops during a
        # profile session; keep the worker contract explicit too.
        worker_env["TILELANG_TPU_BENCHMARK_RUNS"] = "0"

        try:
            process = _spawn_guarded_profile_process(
                normalized_command,
                cwd=output_dir,
                environment=worker_env,
            )
        except (OSError, subprocess.SubprocessError, TPUProfilingError) as exc:
            stdout_path, stderr_path = _write_worker_logs(output_dir, "", str(exc))
            raise TPUProfilingCommandError(f"Could not start PCIe TPU profile worker. Logs: "
                                           f"{stdout_path}, {stderr_path}") from exc
        try:
            remaining = _remaining_timeout(deadline)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(normalized_command, 0)
            guarded_output = _communicate_guarded_process(process, timeout=remaining)
            stdout, stderr = guarded_output.stdout, guarded_output.stderr
        except subprocess.TimeoutExpired:
            terminated = _terminate_and_collect(process)
            stdout, stderr = terminated.stdout, terminated.stderr
            stdout_path, stderr_path = _write_worker_logs(output_dir, stdout, stderr)
            quarantine = ""
            if not terminated.cleanup_complete:
                marker = _quarantine_pcie_device(
                    device_id,
                    process_group=terminated.process_group,
                    reason="PCIe profile worker timed out and could not be fully reaped",
                )
                quarantine = (f" Process group {terminated.process_group} remains live; device "
                              f"{device_id} is quarantined by {marker}.")
            cleanup = ("was terminated"
                       if terminated.cleanup_complete else "could not be fully reaped")
            raise TPUProfilingTimeoutError(
                f"PCIe TPU profile session exceeded its total {self.config.timeout_s}s "
                f"deadline; its worker process group {cleanup}. Logs: "
                f"{stdout_path}, {stderr_path}.{quarantine}") from None
        except BaseException:
            # Preserve cancellation semantics while applying the same bounded
            # process-tree cleanup and pipe drain as an explicit timeout.
            terminated = _terminate_and_collect(process)
            if not terminated.cleanup_complete:
                _quarantine_pcie_device(
                    device_id,
                    process_group=terminated.process_group,
                    reason="PCIe profile worker cancellation could not fully reap its process group",
                )
            raise

        stdout_path, stderr_path = _write_worker_logs(output_dir, stdout, stderr)
        if guarded_output.left_live_descendant:
            cleanup = ("was terminated"
                       if guarded_output.cleanup_complete else "could not be fully reaped")
            quarantine = ""
            if not guarded_output.cleanup_complete:
                marker = _quarantine_pcie_device(
                    device_id,
                    process_group=process.pid,
                    reason="PCIe profile worker left an unreaped descendant",
                )
                quarantine = f" Device {device_id} is quarantined by {marker}."
            raise TPUProfilingCommandError(
                "PCIe TPU profile worker exited while leaving a live descendant; "
                f"its supervised process group {cleanup}. Logs: "
                f"{stdout_path}, {stderr_path}.{quarantine}")
        if process.returncode != 0:
            raise TPUProfilingCommandError(
                f"PCIe TPU profile worker exited with status {process.returncode}. "
                f"Logs: {stdout_path}, {stderr_path}")

        raw_files = tuple(
            sorted(
                path for path in output_dir.glob("cdm_profile_data_dev*")
                if _is_valid_pcie_raw_trace_artifact(path)))
        parser_status = "not-requested"
        parser_message: Optional[str] = None
        report_paths: Tuple[Path, ...] = ()
        decoded_report_paths: Tuple[Path, ...] = ()
        decoder_identity: Mapping[str, str] = {}
        timeline_events: Tuple[TPUInstructionTiming, ...] = ()
        timings: Tuple[TPUInstructionTiming, ...] = ()

        if self.config.postprocess:
            if not raw_files:
                parser_status = "no-raw-trace"
                parser_message = ("The PCIe worker completed but produced no "
                                  "cdm_profile_data_dev* artifact; the decoder was not invoked.")
            elif not _PCIE_PROFILE_DECODER_PATH.is_file():
                parser_status = "unavailable"
                parser_message = ("TileLang's offline PCIe profile decoder helper is missing; "
                                  "raw recorder artifacts were kept.")
            else:
                decoder: Optional[subprocess.Popen] = None
                decoder_guarded_output: Optional[_GuardedProcessOutput] = None
                try:
                    remaining = _remaining_timeout(deadline)
                    if remaining <= 0:
                        parser_status = "deadline-exhausted"
                        parser_message = ("The PCIe worker used the total profile deadline; "
                                          "offline decoding was not started.")
                    else:
                        decoder_env = _pcie_decoder_environment(self.config, worker_env)
                        decoder = _spawn_guarded_profile_process(
                            [
                                _pcie_decoder_python(self.config),
                                str(_PCIE_PROFILE_DECODER_PATH), "--profile-dir",
                                str(output_dir), "--arch",
                                _pcie_profile_arch(self.config)
                            ],
                            cwd=output_dir,
                            environment=decoder_env,
                        )
                        decoder_guarded_output = _communicate_guarded_process(
                            decoder, timeout=_remaining_timeout(deadline))
                        decoder_stdout = decoder_guarded_output.stdout
                        decoder_stderr = decoder_guarded_output.stderr
                except subprocess.TimeoutExpired:
                    assert decoder is not None
                    terminated = _terminate_and_collect(decoder)
                    decoder_stdout, decoder_stderr = terminated.stdout, terminated.stderr
                    parser_stdout, parser_stderr = _write_parser_logs(output_dir, decoder_stdout,
                                                                      decoder_stderr)
                    parser_status = "timed-out"
                    parser_message = ("The offline PCIe decoder exceeded the remaining total "
                                      f"profile deadline. Logs: {parser_stdout}, {parser_stderr}")
                    decoder = None
                except (OSError, subprocess.SubprocessError, TPUProfilingError) as exc:
                    parser_stdout, parser_stderr = _write_parser_logs(output_dir, "", str(exc))
                    parser_status = "failed"
                    parser_message = (f"Could not start the offline PCIe decoder. Logs: "
                                      f"{parser_stdout}, {parser_stderr}")
                    decoder = None
                except BaseException:
                    if decoder is not None:
                        with suppress(Exception):
                            _terminate_and_collect(decoder)
                    raise

                if decoder is not None:
                    parser_stdout, parser_stderr = _write_parser_logs(output_dir, decoder_stdout,
                                                                      decoder_stderr)
                    if (decoder_guarded_output is not None and
                            decoder_guarded_output.left_live_descendant):
                        cleanup = ("was terminated" if decoder_guarded_output.cleanup_complete else
                                   "could not be fully reaped")
                        parser_status = "failed"
                        parser_message = ("The offline PCIe decoder exited while leaving a live "
                                          f"descendant; its supervised process group {cleanup}. "
                                          f"Logs: {parser_stdout}, {parser_stderr}")
                    elif decoder.returncode == 3:
                        parser_status = "unavailable"
                        parser_message = ("bigTpuProfile is not installed; no package was "
                                          "installed automatically and raw PCIe traces were kept. "
                                          f"Logs: {parser_stdout}, {parser_stderr}")
                    elif decoder.returncode == 4:
                        parser_status = "incompatible"
                        parser_message = (
                            "The installed bigTpuProfile does not provide the callable "
                            "BMProfileParserPerfAI.parse API required by TileLang. Logs: "
                            f"{parser_stdout}, {parser_stderr}")
                    elif decoder.returncode != 0:
                        parser_status = "failed"
                        parser_message = (
                            f"The offline PCIe decoder exited with status "
                            f"{decoder.returncode}. Logs: {parser_stdout}, {parser_stderr}")
                    else:
                        decoded_report_paths = tuple(
                            sorted(output_dir.rglob(_PCIE_DECODED_REPORT_NAME)))
                        if decoded_report_paths:
                            try:
                                decoder_identity = _read_pcie_decoder_identity(decoded_report_paths)
                                stdout_identity = _parse_pcie_decoder_identity(decoder_stdout)
                                if decoder_identity != stdout_identity:
                                    raise ValueError(
                                        "The offline PCIe decoder identity differs between its "
                                        "stdout and canonical report.")
                                timings = tuple(
                                    event for path in decoded_report_paths
                                    for event in parse_pcie_decoded_instruction_timings(path))
                            except (json.JSONDecodeError, ValueError) as exc:
                                parser_status = "invalid-report"
                                parser_message = str(exc)
                            else:
                                timeline_events = timings
                                parser_status = ("ready" if timings else "no-device-command-events")
                                if not timings:
                                    parser_message = (
                                        "bigTpuProfile decoded the PCIe recorder "
                                        "output but returned no device-command events.")
                        else:
                            parser_status = "missing-report"
                            parser_message = ("The offline PCIe decoder completed without a "
                                              "canonical TileLang JSON artifact. Logs: "
                                              f"{parser_stdout}, {parser_stderr}")

        return TPUProfileReport(
            config=self.config,
            output_dir=output_dir,
            command=normalized_command,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            raw_trace_files=raw_files,
            raw_instructions=(),
            parser_status=parser_status,
            parser_message=parser_message,
            perfai_report_path=report_paths[0] if report_paths else None,
            perfai_report_paths=report_paths,
            decoded_report_path=(decoded_report_paths[0] if decoded_report_paths else None),
            decoded_report_paths=decoded_report_paths,
            decoder_identity=decoder_identity,
            timeline_events=timeline_events,
            instruction_timings=timings,
        )

    def run_cmodel(self,
                   command: Sequence[PathLike],
                   *,
                   environment: Optional[Mapping[str, str]] = None) -> TPUProfileReport:
        """Run ``command`` in an isolated CModel profiling session.

        ``command`` must compile/load the TileLang kernel inside this child
        process.  The worker starts a new process group; a timeout terminates
        the whole group, matching the safety discipline used for CModel smoke
        tests.  PCIe is rejected here rather than accidentally running through
        the CModel collection mechanism.
        """

        if self.config.runtime_mode != "cmodel":
            raise TPUProfilingError(
                "run_cmodel only supports runtime_mode='cmodel'. PCIe profiling "
                "requires a separately reviewed one-launch worker and is not dispatched.")
        if not command:
            raise ValueError("TPU profiling command must not be empty.")
        normalized_command = tuple(os.fspath(item) for item in command)
        # This is one wall-clock budget for the whole session, not a fresh
        # timeout for each stage.  In particular, a contended PerfAI lock plus
        # AutoRunner cannot extend a user-visible 60s CModel watchdog into a
        # multi-minute operation.
        deadline = _profile_deadline(float(self.config.timeout_s))
        worker_env = _copy_environment(environment)
        self._validate_ppl_dependencies(worker_env)
        output_dir = _profile_output_dir(self.config)
        # This is the PPL CModel contract.  A relative label is required by the
        # emulator, hence the isolated worker cwd instead of a global parent
        # process chdir.
        worker_env["FILE_DUMP_CMD"] = self.config.label
        worker_env["TILELANG_TPU_PROFILE_SESSION"] = "1"
        worker_env["TILELANG_TPU_PROFILE_OUTPUT_DIR"] = str(output_dir)
        worker_env["TILELANG_TPU_PROFILE_CHIP"] = self.config.chip
        worker_env["TILELANG_TPU_PROFILE_PROGRAMMING_MODEL"] = \
            self.config.programming_model
        worker_env["TILELANG_TPU_PROFILE_RUNTIME_MODE"] = "cmodel"
        # Profiling is exactly one launch.  The generated TileLang host
        # template otherwise honors an inherited benchmark loop.
        worker_env["TILELANG_TPU_BENCHMARK_RUNS"] = "0"
        # PPL's CModel driver sets this for SG2260E.  Make the topology
        # explicit for both supported chips so a worker does not inherit a
        # previous process's emulator-core setting.
        worker_env["TPU_RT_CORE_NUM"] = str(self.config.target_spec.chip_spec.physical_core_count)
        # PPL's CModel flow uses FILE_DUMP_CMD, not BMLIB's PCIe recorder.
        worker_env.pop("BMLIB_ENABLE_ALL_PROFILE", None)
        # Never inherit a previously acknowledged board session into a CModel
        # test worker.  The command remains user-supplied, but TileLang's own
        # JIT loader cannot accidentally see a PCIe opt-in from its parent.
        worker_env.pop("TILELANG_TPU_ALLOW_PCIE_LOAD", None)
        worker_env.pop("TILELANG_TPU_ALLOW_PCIE_PROFILE", None)
        worker_env.pop("TILELANG_TPU_DEVICE_ID", None)

        try:
            process = _spawn_guarded_profile_process(
                normalized_command,
                cwd=output_dir,
                environment=worker_env,
            )
        except (OSError, subprocess.SubprocessError, TPUProfilingError) as exc:
            stdout_path, stderr_path = _write_worker_logs(output_dir, "", str(exc))
            raise TPUProfilingCommandError(f"Could not start CModel TPU profile worker. Logs: "
                                           f"{stdout_path}, {stderr_path}") from exc
        try:
            remaining = _remaining_timeout(deadline)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(normalized_command, 0)
            guarded_output = _communicate_guarded_process(process, timeout=remaining)
            stdout, stderr = guarded_output.stdout, guarded_output.stderr
        except subprocess.TimeoutExpired:
            terminated = _terminate_and_collect(process)
            stdout, stderr = terminated.stdout, terminated.stderr
            stdout_path, stderr_path = _write_worker_logs(output_dir, stdout, stderr)
            raise TPUProfilingTimeoutError(
                f"CModel TPU profile session exceeded its total {self.config.timeout_s}s "
                f"deadline and its worker process group was terminated. Logs: "
                f"{stdout_path}, {stderr_path}") from None
        except BaseException:
            # KeyboardInterrupt, test-runner cancellation, and similar paths
            # must not bypass the same process-tree cleanup as a timeout.
            with suppress(Exception):
                _terminate_and_collect(process)
            raise

        stdout_path, stderr_path = _write_worker_logs(output_dir, stdout, stderr)
        if guarded_output.left_live_descendant:
            cleanup = ("was terminated"
                       if guarded_output.cleanup_complete else "could not be fully reaped")
            raise TPUProfilingCommandError(
                "CModel TPU profile worker exited while leaving a live descendant; "
                f"its supervised process group {cleanup}. Logs: "
                f"{stdout_path}, {stderr_path}")
        if process.returncode != 0:
            raise TPUProfilingCommandError(
                f"CModel TPU profile worker exited with status {process.returncode}. "
                f"Logs: {stdout_path}, {stderr_path}")

        raw_files = tuple(
            sorted(
                path for path in output_dir.glob(f"{self.config.label}-*")
                if _RAW_TRACE_ARTIFACT_RE.fullmatch(path.name) is not None))
        raw_instructions = parse_cmodel_raw_instruction_dumps(raw_files)
        parser_status = "not-requested"
        parser_message: Optional[str] = None
        perfai_report_path: Optional[Path] = None
        timeline_events: Tuple[TPUInstructionTiming, ...] = ()
        timings: Tuple[TPUInstructionTiming, ...] = ()

        if self.config.postprocess:
            perfai_root = _find_perfai_root(self.config, worker_env)
            if perfai_root is None:
                parser_status = "unavailable"
                parser_message = (
                    "PerfAI AutoRunner.sh was not found. Raw CModel command dumps were "
                    "kept; set PPL_PERFAI_ROOT or TPUProfilingConfig.perfai_root to "
                    "decode per-instruction timings.")
            elif not raw_files:
                parser_status = "no-raw-trace"
                parser_message = (
                    "The CModel worker completed but produced no FILE_DUMP_CMD artifacts; "
                    "PerfAI was not invoked.")
            else:
                runner = perfai_root / "AutoRunner.sh"
                parser: Optional[subprocess.Popen] = None
                parser_guarded_output: Optional[_GuardedProcessOutput] = None
                parser_stdout: Optional[Path] = None
                parser_stderr: Optional[Path] = None
                remaining = _remaining_timeout(deadline)
                if remaining <= 0:
                    parser_status = "deadline-exhausted"
                    parser_message = (
                        "The CModel worker used the total profile deadline; PerfAI was not "
                        "started and raw command dumps were kept.")
                else:
                    try:
                        # PPL's AutoRunner uses a mutable ``auto_build`` directory.
                        # Both lock acquisition and the parser share the same absolute
                        # deadline as the CModel worker; neither gets a second timeout.
                        with _exclusive_perfai_workspace(perfai_root, deadline):
                            remaining = _remaining_timeout(deadline)
                            if remaining <= 0:
                                parser_status = "deadline-exhausted"
                                parser_message = (
                                    "The total profile deadline expired while waiting for the "
                                    "PerfAI workspace; AutoRunner was not started.")
                            else:
                                parser_env = dict(worker_env)
                                # This is a parser process, not a CModel worker. Keeping
                                # the raw-dump label here could make a future AutoRunner
                                # child accidentally write into the parsing directory.
                                parser_env.pop("FILE_DUMP_CMD", None)
                                parser = _spawn_guarded_profile_process(
                                    [
                                        "bash",
                                        str(runner), "-d",
                                        str(output_dir), "-e",
                                        _perfai_chip_name(self.config)
                                    ],
                                    cwd=perfai_root,
                                    environment=parser_env,
                                )
                                remaining = _remaining_timeout(deadline)
                                if remaining <= 0:
                                    terminated = _terminate_and_collect(parser)
                                    parser_stdout_text = terminated.stdout
                                    parser_stderr_text = terminated.stderr
                                    parser_stdout, parser_stderr = _write_parser_logs(
                                        output_dir, parser_stdout_text, parser_stderr_text)
                                    parser_status = "deadline-exhausted"
                                    parser_message = (
                                        "The total profile deadline expired while AutoRunner was "
                                        "starting; its process group was terminated. Logs: "
                                        f"{parser_stdout}, {parser_stderr}")
                                    parser = None
                                else:
                                    try:
                                        parser_guarded_output = \
                                            _communicate_guarded_process(
                                                parser, timeout=remaining)
                                        parser_stdout_text = \
                                            parser_guarded_output.stdout
                                        parser_stderr_text = \
                                            parser_guarded_output.stderr
                                    except subprocess.TimeoutExpired:
                                        terminated = _terminate_and_collect(parser)
                                        parser_stdout_text = terminated.stdout
                                        parser_stderr_text = terminated.stderr
                                        parser_stdout, parser_stderr = _write_parser_logs(
                                            output_dir, parser_stdout_text, parser_stderr_text)
                                        parser_status = "timed-out"
                                        parser_message = (
                                            "PerfAI AutoRunner.sh exceeded the remaining total profile "
                                            "deadline and its process group was terminated. Logs: "
                                            f"{parser_stdout}, {parser_stderr}")
                                        parser = None
                    except _PerfAIWorkspaceBusy as exc:
                        parser_status = "busy"
                        parser_message = (
                            f"{exc}; the total profile deadline expired before AutoRunner could "
                            "start.")
                    except (OSError, subprocess.SubprocessError, TPUProfilingError) as exc:
                        parser_stdout, parser_stderr = _write_parser_logs(output_dir, "", str(exc))
                        parser_status = "failed"
                        parser_message = (f"Could not start PerfAI AutoRunner.sh. Logs: "
                                          f"{parser_stdout}, {parser_stderr}")
                        parser = None
                    except BaseException:
                        # Preserve cancellation semantics, but never leave an
                        # AutoRunner process tree behind when pytest is stopped.
                        if parser is not None:
                            with suppress(Exception):
                                _terminate_and_collect(parser)
                        raise
                if parser is not None:
                    parser_stdout, parser_stderr = _write_parser_logs(output_dir,
                                                                      parser_stdout_text,
                                                                      parser_stderr_text)
                if (parser is not None and parser_guarded_output is not None and
                        parser_guarded_output.left_live_descendant):
                    cleanup = ("was terminated" if parser_guarded_output.cleanup_complete else
                               "could not be fully reaped")
                    parser_status = "failed"
                    parser_message = ("PerfAI AutoRunner.sh exited while leaving a live "
                                      f"descendant; its supervised process group {cleanup}. "
                                      f"Logs: {parser_stdout}, {parser_stderr}")
                elif parser is not None and parser.returncode != 0:
                    parser_status = "failed"
                    parser_message = (
                        f"PerfAI AutoRunner.sh exited with status {parser.returncode}. "
                        f"Logs: {parser_stdout}, {parser_stderr}")
                elif parser is not None:
                    candidate = (
                        output_dir / "result_profiling" / "output" / "PerfWeb" / "profile_data.js")
                    if not candidate.is_file():
                        parser_status = "missing-report"
                        parser_message = (
                            "PerfAI completed without the expected PerfWeb/profile_data.js "
                            f"artifact. Logs: {parser_stdout}, {parser_stderr}")
                    else:
                        perfai_report_path = candidate
                        try:
                            timeline_events = parse_perfai_timeline_events(candidate)
                            timings = tuple(
                                event for event in timeline_events
                                if event.engine.strip().lower() in _DEVICE_COMMAND_ENGINES)
                        except ValueError as exc:
                            parser_status = "invalid-report"
                            parser_message = str(exc)
                        else:
                            parser_status = "ready" if timings else "no-device-command-events"
                            if not timings:
                                parser_message = (
                                    "PerfAI produced a timeline but no recognized TPU command "
                                    "engine rows; inspect timeline_events and update the parser "
                                    "for this PerfAI schema.")

        return TPUProfileReport(
            config=self.config,
            output_dir=output_dir,
            command=normalized_command,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            raw_trace_files=raw_files,
            raw_instructions=raw_instructions,
            parser_status=parser_status,
            parser_message=parser_message,
            perfai_report_path=perfai_report_path,
            perfai_report_paths=(perfai_report_path,) if perfai_report_path else (),
            decoded_report_path=None,
            decoded_report_paths=(),
            timeline_events=timeline_events,
            instruction_timings=timings,
        )


def run_tpu_cmodel_profile(command: Sequence[PathLike],
                           config: TPUProfilingConfig,
                           *,
                           environment: Optional[Mapping[str, str]] = None) -> TPUProfileReport:
    """Convenience wrapper for a one-off isolated CModel profile worker."""

    return TPUInstructionProfiler(config).run_cmodel(command, environment=environment)


def run_tpu_pcie_profile(command: Sequence[PathLike],
                         config: TPUProfilingConfig,
                         *,
                         environment: Optional[Mapping[str, str]] = None) -> TPUProfileReport:
    """Convenience wrapper for one explicitly authorized PCIe profile worker."""

    return TPUInstructionProfiler(config).run_pcie(command, environment=environment)


__all__ = [
    "TPUInstructionProfiler",
    "TPUInstructionTiming",
    "TPUProfileReport",
    "TPURawInstruction",
    "TPUProfilingCommandError",
    "TPUProfilingConfig",
    "TPUProfilingError",
    "TPUProfilingTimeoutError",
    "parse_perfai_timeline_events",
    "parse_perfai_instruction_timings",
    "parse_pcie_decoded_instruction_timings",
    "parse_cmodel_raw_instruction_dumps",
    "run_tpu_cmodel_profile",
    "run_tpu_pcie_profile",
]
