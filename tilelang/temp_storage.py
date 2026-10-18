# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Keep TileLang scratch files in a private directory under the user's home."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import tempfile


def _check_private_directory(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or path.is_symlink():
        raise RuntimeError(f"TileLang temporary path is not a directory: {path}")
    if os.name == "posix" and (info.st_uid != os.geteuid() or
                               stat.S_IMODE(info.st_mode) & 0o077):
        raise RuntimeError(f"TileLang temporary directory must be owned by the current user "
                           f"and private (mode 0700): {path}")


def user_temp_root() -> Path:
    """Create and validate ``~/tmp/tilelang`` without using the shared /tmp."""
    home = Path.home().resolve(strict=True)
    parent = home / "tmp"
    if parent.is_symlink():
        raise RuntimeError(f"TileLang temporary parent must not be a symlink: {parent}")
    parent.mkdir(mode=0o700, exist_ok=True)
    if not parent.is_dir():
        raise RuntimeError(f"TileLang temporary parent is not a directory: {parent}")
    root = parent / "tilelang"
    if root.is_symlink():
        raise RuntimeError(f"TileLang temporary root must not be a symlink: {root}")
    root.mkdir(mode=0o700, exist_ok=True)
    _check_private_directory(root)
    return root


def private_cmodel_mount_ok() -> bool:
    """Verify that /tmp is a bind mount of this worker's user-owned directory."""
    if os.environ.get("TILELANG_CMODEL_PRIVATE_TMP") != "1":
        return False
    raw_backing = os.environ.get("TILELANG_CMODEL_PRIVATE_TMP_HOST")
    if not raw_backing:
        return False
    try:
        backing = Path(raw_backing).resolve(strict=True)
        root = user_temp_root()
        if backing == root or not backing.is_relative_to(root):
            return False
        _check_private_directory(backing)
        visible = Path("/tmp").stat()
        source = backing.stat()
        if (visible.st_dev, visible.st_ino) != (source.st_dev, source.st_ino):
            return False
        rows = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except (OSError, RuntimeError, ValueError):
        return False
    for row in rows:
        mount, separator, _filesystem = row.partition(" - ")
        fields = mount.split()
        if separator and len(fields) > 4 and fields[3] == str(backing) and fields[4] == "/tmp":
            return True
    return False


def tilelang_temp_dir() -> Path:
    """Return the host user root, or its /tmp alias inside an isolated CModel."""
    if os.environ.get("TILELANG_CMODEL_PRIVATE_TMP") == "1":
        if not private_cmodel_mount_ok():
            raise RuntimeError("CModel temporary directory is not a verified private bind mount")
        return Path("/tmp")
    return user_temp_root()


def configure_process_temp() -> Path:
    """Route Python and inherited subprocess scratch to TileLang storage."""
    directory = tilelang_temp_dir()
    for name in ("TMPDIR", "TMP", "TEMP"):
        os.environ[name] = str(directory)
    # tempfile may have cached the old default before TileLang was imported.
    tempfile.tempdir = str(directory)
    return directory
