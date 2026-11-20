# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""CModel must switch resident JIT modules without dispatching stale code."""

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

from tilelang.jit.adapter import tpu
from tilelang.temp_storage import user_temp_root


def test_cmodel_release_failure_blocks_further_switches(monkeypatch):
    monkeypatch.setattr(tpu, "_CMODEL_ACTIVE_LIBRARY", None)
    monkeypatch.setattr(tpu, "_CMODEL_RELEASE_FAILURE", None)
    releases = []

    def fail_to_release():
        releases.append(True)
        return -17

    first = SimpleNamespace(tilelang_tpu_release_cmodel=fail_to_release)
    second = SimpleNamespace()
    tpu._select_cmodel_library(first)
    tpu._select_cmodel_library(first)
    assert not releases

    with pytest.raises(RuntimeError, match="release failed with status -17"):
        tpu._select_cmodel_library(second)
    with pytest.raises(RuntimeError, match="fresh isolated CModel process"):
        tpu._select_cmodel_library(first)
    assert releases == [True]


def _run_alternating_modules(programming_model):
    import torch
    import tilelang
    import tilelang.language as T

    def compile_add(value):
        @T.prim_func
        def add(A: T.Tensor((1, 32), "float32"), B: T.Tensor((1, 32), "float32")):
            with T.Kernel(1, is_cpu=True):
                tile = T.alloc_shared((1, 32), "float32")
                T.ppl_copy(A, tile)
                T.ppl_add_C(tile, tile, value)
                T.ppl_copy(tile, B)

        return tilelang.compile(
            add, out_idx=-1, target=f"tpu -mcpu=bm1690e -tpu-programming-model={programming_model}",
            runtime_mode="cmodel")

    left, right = compile_add(1.0), compile_add(7.0)
    data = torch.full((1, 32), 2.0)
    for kernel, value in ((left, 3.0), (left, 3.0), (right, 9.0), (left, 3.0), (right, 9.0)):
        output = torch.full_like(data, float("nan"))
        kernel(data, output)
        torch.testing.assert_close(output, torch.full_like(output, value), atol=0, rtol=0)


@pytest.mark.skipif(os.environ.get("TILELANG_TPU_RUN_CMODEL_JIT") != "1",
                    reason="requires the isolated BM1690E CModel and PPL SDK")
@pytest.mark.parametrize("programming_model", ["tpukernel", "rv"])
def test_cmodel_jit_dispatches_the_requested_module(programming_model):
    env = dict(os.environ, OMP_NUM_THREADS="1", TILELANG_TPU_BENCHMARK_RUNS="0")
    env.pop("TILELANG_TPU_POST_EACH_CALL", None)
    with TemporaryDirectory(dir=user_temp_root()) as base:
        subprocess.run([
            sys.executable, "-m", "tilelang.jit.tpu_cmodel", "--output-dir",
            str(Path(base) / "isolated"), "--timeout", "90", "--", sys.executable,
            str(Path(__file__).resolve()), programming_model,
        ], check=True, env=env, timeout=105)


if __name__ == "__main__":
    _run_alternating_modules(sys.argv[1])
