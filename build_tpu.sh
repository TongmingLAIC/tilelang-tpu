#!/usr/bin/env bash
# Build the TileLang TPU native libraries from the source checkout.
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
build_dir="$repo_root/build-tpu"
tvm_config="$repo_root/3rdparty/tvm/cmake/config.cmake"
build_config="$build_dir/config.cmake"
tilelang_tmp_root="${HOME:?HOME must be set}/tmp/tilelang"

if [[ -L "$HOME/tmp" || -L "$tilelang_tmp_root" ]]; then
  echo "TileLang temporary directories must not be symlinks" >&2
  exit 1
fi
install -d -m 700 "$HOME/tmp" "$tilelang_tmp_root"
if [[ ! -O "$HOME/tmp" || ! -O "$tilelang_tmp_root" ]]; then
  echo "TileLang temporary directories must be owned by the current user" >&2
  exit 1
fi
export TMPDIR="$tilelang_tmp_root" TMP="$tilelang_tmp_root" TEMP="$tilelang_tmp_root"

if [[ ! -f "$tvm_config" ]]; then
  echo "The TVM submodule is not ready: $tvm_config" >&2
  echo "Run: git submodule update --init --recursive 3rdparty/tvm" >&2
  exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "CMake was not found. Activate the project virtual environment first." >&2
  exit 1
fi

mkdir -p "$build_dir"
if [[ ! -f "$build_config" ]]; then
  cp "$tvm_config" "$build_config"
fi

cores="$(nproc)"
jobs=$((cores * 3 / 4))
if ((jobs < 1)); then
  jobs=1
fi

echo "Configuring TileLang-TPU in $build_dir"
cmake -S "$repo_root" -B "$build_dir"

echo "Building TileLang-TPU with $jobs parallel job(s)"
cmake --build "$build_dir" --parallel "$jobs"

echo "Build completed: $build_dir"
