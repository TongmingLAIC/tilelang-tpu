# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Pure-Python safety and shape contracts for the public TPU demos."""

import json
from pathlib import Path

import pytest
import torch
import tilelang

from tilelang.engine.tpu_config import TPU_CHIP_SPECS
from tilelang.language import customize
from tpu_demo.cases import (CHIP_CORE_COUNTS, OPERATIONS, RV_SUPPORTED_OPERATIONS, TARGET_CONFIGS,
                            build_cases, kernel_variant)
from tpu_demo.common import DemoNumericalMismatch, comparison, validate_selection
from tpu_demo.elementwise import build_elementwise
from tpu_demo.flashattn import build_flashattn
from tpu_demo.flashattn.flashattn import _attention_mask, _reference, _validation_inputs
from tpu_demo.matmul import build_matmul
from tpu_demo.rmsnorm import build_rmsnorm, build_rmsnorm_splitk
from tpu_demo.rope import build_rope
from tpu_demo.swiglu import build_swiglu


def test_registry_is_unique_complete_and_capability_scoped():
    cases = build_cases()
    assert len(cases) == 73
    assert len({case.case_id for case in cases}) == len(cases)
    assert sum(case.supports_rv for case in cases) == len(cases)
    assert set(OPERATIONS) == RV_SUPPORTED_OPERATIONS
    flash_cases = [case for case in cases if case.operation == "flashattn"]
    assert len(flash_cases) == 30
    assert {case.variant for case in flash_cases} == {"balanced", "descending-max", "weighted-keys"}
    assert {case.is_causal for case in flash_cases} == {False, True}


@pytest.mark.parametrize("invalid", (1, 0, "false", None))
def test_flashattn_requires_a_boolean_causal_flag(invalid):
    with pytest.raises(TypeError, match="boolean is_causal"):
        _attention_mask(32, invalid)


def _build_case_kernel(case, programming_model):
    if case.operation.startswith("elementwise-"):
        return build_elementwise(case.operation.removeprefix("elementwise-"), dtype=case.dtype)
    if case.operation == "matmul":
        return build_matmul(dtype=case.dtype, programming_model=programming_model)
    if case.operation == "rmsnorm":
        return build_rmsnorm(dtype=case.dtype)
    if case.operation == "rmsnorm-splitk":
        return build_rmsnorm_splitk(dtype=case.dtype)
    if case.operation == "rope":
        return build_rope(dtype=case.dtype)
    if case.operation == "swiglu":
        return build_swiglu(dtype=case.dtype)
    if case.operation == "flashattn":
        return build_flashattn(dtype=case.dtype)
    raise AssertionError(f"unhandled test operation: {case.operation}")


@pytest.mark.parametrize("programming_model", ("tpukernel", "rv"))
@pytest.mark.parametrize("case", build_cases(), ids=lambda case: case.case_id)
def test_each_matrix_case_selects_the_declared_kernel(case, programming_model):
    program = _build_case_kernel(case, programming_model)
    assert str(program.attrs["global_symbol"]) == kernel_variant(case.operation, case.dtype,
                                                                 programming_model)


def test_kernel_variants_cover_only_distinct_frontend_expressions():
    actual = {
        kernel_variant(case.operation, case.dtype, programming_model)
        for case in build_cases()
        for programming_model in ("tpukernel", "rv")
    }
    assert actual == {
        "elementwise_add",
        "elementwise_sub",
        "elementwise_mul",
        "elementwise_div",
        "matmul_low_precision",
        "matmul_fp32",
        "rmsnorm_low_precision",
        "rmsnorm_fp32",
        "rmsnorm_splitk_low_precision",
        "rmsnorm_splitk_fp32",
        "rope",
        "swiglu_low_precision",
        "swiglu_fp32",
        "flashattn_low_precision",
        "flashattn_fp32",
    }


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32"))
def test_flashattn_weighted_keys_rejects_uniform_weight_degeneracy(dtype):
    q, k, v = _validation_inputs(dtype, "weighted-keys", seed=0)
    expected = _reference(q, k, v, _attention_mask(q.shape[1], False), dtype).float()
    compute_v = v.to(torch.bfloat16).float() if dtype == "float32" else v.float()
    uniform = compute_v.mean(dim=1, keepdim=True).expand_as(expected)
    assert torch.max(torch.abs(expected - uniform)).item() > 0.1


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32"))
def test_flashattn_causal_mask_changes_the_reference(dtype):
    q, k, v = _validation_inputs(dtype, "weighted-keys", seed=0)
    noncausal = _reference(q, k, v, _attention_mask(q.shape[1], False), dtype)
    causal = _reference(q, k, v, _attention_mask(q.shape[1], True), dtype)
    assert torch.max(torch.abs(noncausal.float() - causal.float())).item() > 0.05


def test_demo_implementations_do_not_call_composite_rope_or_sigmoid_ops():
    demo_root = Path(__file__).resolve().parents[3] / "tpu_demo"
    rope_source = (demo_root / "rope/rope.py").read_text(encoding="utf-8")
    swiglu_source = (demo_root / "swiglu/swiglu.py").read_text(encoding="utf-8")
    assert "ppl_rope_add" not in rope_source
    assert "ppl_sigmoid" not in swiglu_source
    for operation in ("ppl_copy", "ppl_mul", "ppl_subtract", "ppl_add"):
        assert operation in rope_source
    for operation in ("ppl_exp", "ppl_add_C", "ppl_div", "ppl_mul"):
        assert operation in swiglu_source
    assert not (demo_root.parent / "tilelang/language/ppl_llama.py").exists()


def test_public_tpu_ops_document_backend_dtype_support_and_mapping():
    operation_names = (
        "ppl_gemm",
        "ppl_copy",
        "ppl_fill",
        "ppl_add",
        "ppl_subtract",
        "ppl_mul",
        "ppl_div",
        "ppl_max",
        "ppl_add_C",
        "ppl_mul_C",
        "ppl_exp",
        "ppl_rsqrt",
        "ppl_reduce_sum",
        "ppl_reduce_max",
        "ppl_gather",
        "ppl_embedding",
        "ppl_topk",
    )
    for name in operation_names:
        assert "Dtype support:" in getattr(customize, name).__doc__

    mapping = (Path(__file__).resolve().parents[3] /
               "tpu_demo/OP_MAPPING.md").read_text(encoding="utf-8")
    for name in operation_names:
        assert f"`{name}`" in mapping


@pytest.mark.parametrize("programming_model", ("tpukernel", "rv"))
def test_fp32_matmul_uses_one_portable_frontend_and_native_backend_mapping(programming_model):
    source = tilelang.lower(
        build_matmul(dtype="float32", programming_model=programming_model),
        target=f"tpu -mcpu=sg2260e -tpu-programming-model={programming_model}").kernel_source
    if programming_model == "rv":
        assert "rvt_fmm_nn" in source
        assert "rvt_fmm2" not in source
    else:
        assert "tpu_bdc_fp32_mm" in source


def test_lightweight_demo_target_registry_matches_compiler_capabilities():
    assert {
        name: spec.physical_core_count for name, spec in TPU_CHIP_SPECS.items()
    } == CHIP_CORE_COUNTS
    assert {(name, programming_model)
            for name, spec in TPU_CHIP_SPECS.items()
            for programming_model in spec.programming_models} == set(TARGET_CONFIGS)


def test_backend_selection_rejects_unsupported_or_unsupervised_paths(monkeypatch):
    with pytest.raises(ValueError, match="BM1690.*RV Tensor"):
        validate_selection(
            chip="bm1690",
            programming_model="rv",
            runtime_mode="cmodel",
            supports_rv=True,
            allow_pcie=False,
            device_id=None)
    with pytest.raises(ValueError, match="does not yet expose"):
        validate_selection(
            chip="sg2260e",
            programming_model="rv",
            runtime_mode="cmodel",
            supports_rv=False,
            allow_pcie=False,
            device_id=None)
    monkeypatch.delenv("TILELANG_TPU_PROFILE_SESSION", raising=False)
    with pytest.raises(ValueError, match="supervised demo matrix"):
        validate_selection(
            chip="sg2260e",
            programming_model="tpukernel",
            runtime_mode="pcie",
            supports_rv=False,
            allow_pcie=True,
            device_id=0)


@pytest.mark.parametrize("chip,cores", (("bm1690", "8"), ("sg2260e", "4")))
def test_cmodel_selection_sets_physical_core_topology(monkeypatch, chip, cores):
    monkeypatch.setenv("TILELANG_TPU_ALLOW_PCIE_LOAD", "1")
    validate_selection(
        chip=chip,
        programming_model="tpukernel",
        runtime_mode="cmodel",
        supports_rv=False,
        allow_pcie=False,
        device_id=None)
    assert __import__("os").environ["TPU_RT_CORE_NUM"] == cores
    assert "TILELANG_TPU_ALLOW_PCIE_LOAD" not in __import__("os").environ


def test_comparison_rejects_shape_and_dtype_before_numeric_conversion():
    with pytest.raises(DemoNumericalMismatch, match="shape mismatch"):
        comparison(torch.zeros(2), torch.zeros(3), atol=0, rtol=0)
    with pytest.raises(DemoNumericalMismatch, match="dtype mismatch"):
        comparison(torch.zeros(2, dtype=torch.float16), torch.zeros(2), atol=0, rtol=0)


def test_nonfinite_mismatch_payload_is_json_serializable():
    with pytest.raises(DemoNumericalMismatch) as caught:
        comparison(torch.tensor([float("nan")]), torch.zeros(1), atol=0, rtol=0)
    json.dumps(caught.value.metrics, allow_nan=False)
    assert caught.value.metrics["finite"] is False
    assert caught.value.metrics["max_abs_error"] is None


@pytest.mark.parametrize(
    "builder,kwargs,diagnostic",
    (
        (build_elementwise, {
            "operation": "add",
            "rows": 0
        }, "positive integer"),
        (build_matmul, {
            "m": 17,
            "block_m": 16
        }, "divisible"),
        (build_rmsnorm, {
            "rows": 5,
            "block_rows": 4
        }, "divisible"),
        (build_rmsnorm_splitk, {
            "width": 65,
            "block_k": 32
        }, "divisible"),
        (build_rope, {
            "width": 31
        }, "divisible"),
        (build_swiglu, {
            "width": 33,
            "block_width": 32
        }, "divisible"),
        (build_flashattn, {
            "sequence": 33
        }, "divisible"),
    ),
)
def test_demo_builders_reject_unimplemented_tail_paths(builder, kwargs, diagnostic):
    with pytest.raises(ValueError, match=diagnostic):
        builder(**kwargs)


@pytest.mark.parametrize("epsilon", (0.0, -1.0, float("nan"), float("inf"), "1e-5"))
def test_rmsnorm_rejects_invalid_epsilon(epsilon):
    with pytest.raises(ValueError, match="finite positive epsilon"):
        build_rmsnorm(epsilon=epsilon)
