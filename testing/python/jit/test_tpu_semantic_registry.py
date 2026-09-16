# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Keep the source-level TPU semantic registries closed and aligned."""

import ast
import re
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_SEMANTIC_EXTERN = re.compile(r'"(tl\.(?:tpu|tpukernel)\.[a-z0-9_]+)"')

_PORTABLE_EXTERNS = {
    "tl.tpu.embedding",
    "tl.tpu.add_scalar",
    "tl.tpu.mul_scalar",
    "tl.tpu.rsqrt",
    "tl.tpu.reduce_sum",
    "tl.tpu.reduce_max",
    "tl.tpu.exp",
    "tl.tpu.add",
    "tl.tpu.copy",
    "tl.tpu.div",
    "tl.tpu.fill",
    "tl.tpu.gemm",
    "tl.tpu.max",
    "tl.tpu.mul",
    "tl.tpu.sub",
}
_TPUKERNEL_EXTERNS = {
    "tl.tpukernel.gather",
    "tl.tpukernel.topk",
}
_PORTABLE_IMPLEMENTED_IN_TPUKERNEL = {
    "tl.tpu.embedding",
    "tl.tpu.add_scalar",
    "tl.tpu.mul_scalar",
    "tl.tpu.rsqrt",
    "tl.tpu.reduce_sum",
    "tl.tpu.reduce_max",
    "tl.tpu.exp",
}


def _semantic_externs(relative_path):
    source = (_REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
    return set(_SEMANTIC_EXTERN.findall(source))


def _literal_assignment(relative_path, name):
    path = _REPOSITORY_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if (isinstance(node, ast.Assign) and
                any(isinstance(target, ast.Name) and target.id == name for target in node.targets)):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing literal assignment {name} in {relative_path}")


def test_semantic_extern_registry_is_isomorphic_across_compiler_layers():
    """An extern is valid only when every source-level compiler boundary owns it."""
    expected = _PORTABLE_EXTERNS | _TPUKERNEL_EXTERNS

    assert _semantic_externs("tilelang/language/customize.py") == expected
    assert _semantic_externs("tilelang/engine/lower.py") == expected
    assert _semantic_externs("src/transform/address_assign.cc") == expected
    assert _semantic_externs("src/target/codegen_tpu.cc") == _PORTABLE_EXTERNS
    assert _semantic_externs("src/target/codegen_tpukernel.cc") == (
        _TPUKERNEL_EXTERNS | _PORTABLE_IMPLEMENTED_IN_TPUKERNEL)


def test_every_semantic_extern_has_exact_region_operand_positions():
    expected = _PORTABLE_EXTERNS | _TPUKERNEL_EXTERNS
    positions = _literal_assignment("tilelang/engine/lower.py", "_TPU_SEMANTIC_REGION_ARGS")

    assert set(positions) == expected
    assert all(
        tuple(indices) == tuple(range(1,
                                      len(indices) + 1)) for indices in positions.values())
