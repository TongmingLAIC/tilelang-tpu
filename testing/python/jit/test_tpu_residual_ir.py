# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Fail-closed tests for residual TIR at the TPU backend boundary."""

import importlib

import pytest

import tilelang
from tilelang import tvm
from tvm import tir

lower_module = importlib.import_module("tilelang.engine.lower")
adapter_utils = importlib.import_module("tilelang.jit.adapter.utils")


def _target(programming_model="tpukernel"):
    return tvm.target.Target("tpu -mcpu=sg2260e "
                             f"-tpu-programming-model={programming_model}")


def _prim_func(body, name="residual_ir"):
    return tir.PrimFunc([], body).with_attr("global_symbol", name)


def _assert_contract_error(function, category, detail=None, programming_model="tpukernel"):
    module = tvm.IRModule({function.attrs["global_symbol"]: function})
    with pytest.raises(ValueError) as error:
        lower_module.validate_target_module_contract(module, _target(programming_model))
    message = str(error.value)
    assert "target='tpu " in message
    assert f"node category={category}" in message
    if detail is not None:
        assert detail in message


def test_tpu_residual_ir_rejects_ramp_and_vector_buffer_dtype():
    source = tir.decl_buffer((8,), "float32", name="source")
    ramp = tir.Ramp(tir.IntImm("int32", 0), tir.IntImm("int32", 1), 4)
    vector_load = tir.BufferLoad(source, [ramp])
    function = tir.PrimFunc(
        [source.data],
        tir.Evaluate(vector_load),
        buffer_map={
            source.data: source
        },
    ).with_attr("global_symbol", "vector_load")
    _assert_contract_error(function, "Ramp", "vector index")

    vector_data = tir.Var("vector_buffer_data",
                          tvm.ir.PointerType(tvm.ir.PrimType("float32x4"), "local"))
    vector_buffer = tir.decl_buffer((4,), "float32x4", name="vector_buffer", data=vector_data)
    function = _prim_func(tir.DeclBuffer(vector_buffer, tir.Evaluate(0)), "vector_buffer")
    _assert_contract_error(function, "vector-buffer-dtype", "float32x4")


def test_tpu_residual_ir_rejects_direct_scalar_buffer_access():
    source = tir.decl_buffer((8,), "float32", name="source")
    destination = tir.decl_buffer((8,), "float32", name="destination")
    load = tir.BufferLoad(source, [tir.IntImm("int32", 0)])
    function = tir.PrimFunc(
        [source.data],
        tir.Evaluate(load),
        buffer_map={
            source.data: source
        },
    ).with_attr("global_symbol", "scalar_load")
    _assert_contract_error(function, "BufferLoad", "has no TPU descriptor lowering")

    store = tir.BufferStore(destination, tir.FloatImm("float32", 1.0), [tir.IntImm("int32", 0)])
    function = tir.PrimFunc(
        [destination.data],
        store,
        buffer_map={
            destination.data: destination
        },
    ).with_attr("global_symbol", "scalar_store")
    _assert_contract_error(function, "BufferStore", "has no TPU descriptor lowering")


def test_tpu_copy_region_markers_are_not_treated_as_executable_loads():
    source = tir.decl_buffer((8,), "float32", name="source")
    destination = tir.decl_buffer((8,), "float32", name="destination")
    ramp = tir.Ramp(tir.IntImm("int32", 0), tir.IntImm("int32", 1), 4)

    def region(buffer, index, access_mask):
        marker = tir.BufferLoad(buffer, [index])
        return tir.call_intrin("handle", tir.op.Op.get("tl.region"), marker, access_mask, 4)

    copy = tir.call_extern(
        "handle",
        "tl.tpu.copy",
        region(source, ramp, 1),
        region(destination, ramp, 2),
    )
    function = tir.PrimFunc(
        [source.data, destination.data],
        tir.Evaluate(copy),
        buffer_map={
            source.data: source,
            destination.data: destination
        },
    ).with_attr("global_symbol", "copy_region_markers")

    lower_module.validate_target_module_contract(
        tvm.IRModule({"copy_region_markers": function}), _target())

    # The same structural marker has no meaning outside the canonical
    # two-region tl.tpu.copy ABI and must not become a scalar-load escape hatch.
    standalone_region = region(source, tir.IntImm("int32", 0), 1)
    standalone = tir.PrimFunc(
        [source.data],
        tir.Evaluate(standalone_region),
        buffer_map={
            source.data: source
        },
    ).with_attr("global_symbol", "standalone_region")
    _assert_contract_error(standalone, "BufferLoad", "has no TPU descriptor lowering")


def test_tpu_copy_marker_object_cannot_be_reused_as_an_executable_load():
    source = tir.decl_buffer((8,), "float32", name="source")
    destination = tir.decl_buffer((8,), "float32", name="destination")
    shared_marker = tir.BufferLoad(source, [tir.IntImm("int32", 0)])

    def region(marker, access_mask):
        return tir.call_intrin("handle", tir.op.Op.get("tl.region"), marker, access_mask, 8)

    copy = tir.call_extern(
        "handle",
        "tl.tpu.copy",
        region(shared_marker, 1),
        region(tir.BufferLoad(destination, [0]), 2),
    )
    body = tir.SeqStmt([
        tir.Evaluate(copy),
        tir.Evaluate(shared_marker),
    ])
    function = tir.PrimFunc(
        [source.data, destination.data],
        body,
        buffer_map={
            source.data: source,
            destination.data: destination
        },
    ).with_attr("global_symbol", "reused_copy_marker")

    _assert_contract_error(function, "semantic-region-marker-alias", "outside its canonical")


def test_typed_semantic_regions_are_markers_but_bare_access_ptr_is_removed():
    source = tir.decl_buffer((8,), "float32", name="source")
    destination = tir.decl_buffer((8,), "float32", name="destination")

    def region(buffer, access_mask):
        return tir.call_intrin("handle", tir.op.Op.get("tl.region"), tir.BufferLoad(buffer, [0]),
                               access_mask, 8)

    rsqrt = tir.call_extern("handle", "tl.tpu.rsqrt", region(destination, 2), region(source, 1))
    function = tir.PrimFunc(
        [source.data, destination.data],
        tir.Evaluate(rsqrt),
        buffer_map={
            source.data: source,
            destination.data: destination
        },
    ).with_attr("global_symbol", "typed_regions")
    lower_module.validate_target_module_contract(
        tvm.IRModule({"typed_regions": function}), _target())

    access_ptr = source.access_ptr("r")
    removed = tir.PrimFunc(
        [source.data],
        tir.Evaluate(access_ptr),
        buffer_map={
            source.data: source
        },
    ).with_attr("global_symbol", "removed_access_ptr")
    _assert_contract_error(removed, "tensor-operand-ABI", "whole-buffer tl.region")


def test_tpu_residual_ir_rejects_vector_signature_and_allocation_dtypes():
    vector_parameter = tir.Var("vector_parameter", "float32x4")
    function = tir.PrimFunc(
        [vector_parameter],
        tir.Evaluate(0),
    ).with_attr("global_symbol", "vector_parameter")
    _assert_contract_error(function, "vector-parameter-dtype", "float32x4")

    vector_data = tir.Var("vector_data", tvm.ir.PointerType(tvm.ir.PrimType("float32x4"), "local"))
    allocation = tir.Allocate(vector_data, "float32x4", [4], tir.IntImm("bool", 1), tir.Evaluate(0))
    function = _prim_func(allocation, "vector_allocation")
    _assert_contract_error(function, "vector-allocation-dtype", "float32x4")


def test_tpu_residual_ir_rejects_scalar_parameters_without_marshalling_contract():
    scalar = tir.Var("scale", "float32")
    function = tir.PrimFunc(
        [scalar],
        tir.Evaluate(scalar),
    ).with_attr("global_symbol", "scalar_parameter")

    _assert_contract_error(function, "scalar-parameter", "marshals Tensor parameters only")


def test_tpu_residual_ir_rejects_vector_call_result():
    vector_call = tir.call_pure_extern("float32x4", "cuda_vector_helper")
    function = _prim_func(tir.Evaluate(vector_call), "vector_call")
    _assert_contract_error(function, "vector-call", "float32x4")


@pytest.mark.parametrize("op_name", [
    "tir.tvm_storage_sync",
    "tir.ptx_commit_group",
    "tl.SyncThreadsPartialOp",
])
def test_tpu_residual_ir_rejects_gpu_synchronization(op_name):
    op = tvm.ir.Op.get(op_name)
    arguments = [tvm.tir.StringImm("shared")] if op_name == "tir.tvm_storage_sync" else []
    function = _prim_func(tir.Evaluate(tir.Call("int32", op, arguments)), "gpu_sync")
    _assert_contract_error(function, "gpu-synchronization", op_name)


@pytest.mark.parametrize("extern_name", [
    "AtomicAdd",
    "cuda_vector_helper",
    "tl.tpu.typo",
    "tl.tpukernel.typo",
])
def test_tpu_residual_ir_rejects_unknown_externs(extern_name):
    function = _prim_func(tir.Evaluate(tir.call_extern("handle", extern_name)), "unknown_extern")
    _assert_contract_error(function, "call_extern", extern_name)


def test_tpu_residual_ir_rejects_pure_externs():
    function = _prim_func(
        tir.Evaluate(tir.call_pure_extern("float32", "unknown_math")), "pure_extern")
    _assert_contract_error(function, "call_pure_extern", "unknown_math")


def test_tpu_residual_ir_allows_only_owned_semantic_and_model_raw_externs():
    tpukernel = _prim_func(
        tir.Evaluate(tir.call_extern("handle", "tl.tpu.rsqrt")), "owned_tpukernel")
    lower_module.validate_target_module_contract(
        tvm.IRModule({"owned_tpukernel": tpukernel}), _target("tpukernel"))

    raw_rv = _prim_func(tir.Evaluate(tir.call_extern("handle", "rvt_sync_all")), "owned_rv")
    lower_module.validate_target_module_contract(tvm.IRModule({"owned_rv": raw_rv}), _target("rv"))
    _assert_contract_error(raw_rv, "call_extern", "different programming model", "tpukernel")


def test_tpu_shared_allocation_is_not_confused_with_gpu_synchronization():
    data = tir.Var("local_data", tvm.ir.PointerType(tvm.ir.PrimType("float32"), "shared.dyn"))
    allocation = tir.Allocate(
        data,
        "float32", [16],
        tir.IntImm("bool", 1),
        tir.Evaluate(0),
        annotations={"storage_scope": "shared.dyn"})
    function = _prim_func(allocation, "shared_allocation")
    lower_module.validate_target_module_contract(
        tvm.IRModule({"shared_allocation": function}), _target())


@pytest.mark.parametrize("kind", [
    tir.ForKind.PARALLEL,
    tir.ForKind.VECTORIZED,
    tir.ForKind.THREAD_BINDING,
])
def test_tpu_residual_ir_rejects_loops_without_an_execution_mapping(kind):
    loop_var = tir.Var("i", "int32")
    thread_binding = None
    if kind == tir.ForKind.THREAD_BINDING:
        thread_binding = tir.IterVar(
            tvm.ir.Range(0, 4), loop_var, tir.IterVar.ThreadIndex, "threadIdx.x")
    loop = tir.For(loop_var, 0, 4, kind, tir.Evaluate(0), thread_binding=thread_binding)
    function = _prim_func(loop, "unsupported_loop_kind")
    _assert_contract_error(function, "For", "no TPU execution mapping")


def test_tpu_residual_ir_rejects_unconsumed_attributes():
    function = _prim_func(
        tir.AttrStmt(
            tir.StringImm("payload"), "pragma_import_c", tir.StringImm("side_effecting_source"),
            tir.Evaluate(0)), "unconsumed_attribute")
    _assert_contract_error(function, "AttrStmt", "has no residual TPU meaning")


def test_tpu_contract_rejects_a_primfunc_bound_to_another_backend():
    function = _prim_func(
        tir.Evaluate(0),
        "wrong_function_target",
    ).with_attr("target", tvm.target.Target("c"))

    _assert_contract_error(function, "PrimFunc-target", "bound to target kind 'c'")


def test_tpu_contract_rejects_a_mismatched_function_tpu_identity():
    function = _prim_func(
        tir.Evaluate(0),
        "wrong_tpu_identity",
    ).with_attr("target", tvm.target.Target("tpu -mcpu=bm1690 -tpu-programming-model=tpukernel"))

    _assert_contract_error(function, "PrimFunc-target", "identity disagrees")


def test_full_lower_revalidates_pass_output_before_address_assignment(monkeypatch):
    original = _prim_func(tir.Evaluate(0), "full_lower_order")
    injected = _prim_func(tir.Evaluate(tir.call_extern("handle", "AtomicAdd")), "full_lower_order")
    rewritten = tvm.IRModule({"full_lower_order": injected})
    address_assignment_called = False

    monkeypatch.setattr(lower_module, "LowerAndLegalize", lambda mod, _target: mod)
    monkeypatch.setattr(lower_module, "OptimizeForTarget", lambda _mod, _target: rewritten)

    def record_address_assignment(mod, _target):
        nonlocal address_assignment_called
        address_assignment_called = True
        return mod

    monkeypatch.setattr(lower_module, "AssignTPUAddresses", record_address_assignment)

    with pytest.raises(ValueError, match="node category=call_extern.*AtomicAdd"):
        tilelang.lower(original, target=str(_target()))
    assert not address_assignment_called


def test_annotation_only_revalidates_pass_output_before_address_assignment(monkeypatch):
    original = _prim_func(tir.Evaluate(0), "annotation_order")
    injected = _prim_func(tir.Evaluate(tir.call_extern("handle", "AtomicAdd")), "annotation_order")
    rewritten = tvm.IRModule({"annotation_order": injected})
    address_assignment_called = False

    monkeypatch.setattr(adapter_utils, "LowerAndLegalize", lambda mod, _target: mod)
    monkeypatch.setattr(adapter_utils, "OptimizeForTarget", lambda _mod, _target: rewritten)

    def record_address_assignment(mod, _target):
        nonlocal address_assignment_called
        address_assignment_called = True
        return mod

    monkeypatch.setattr(adapter_utils, "AssignTPUAddresses", record_address_assignment)

    with pytest.raises(ValueError, match="node category=call_extern.*AtomicAdd"):
        adapter_utils.get_annotated_mod(original, target=str(_target()), model_type="all")
    assert not address_assignment_called
