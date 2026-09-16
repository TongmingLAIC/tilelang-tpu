# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The compiler for TL programs."""

import os
import os.path as osp
import re
from typing import Union, Optional, Callable, List
import tilelang.transform
from tilelang import tvm as tvm
from tvm import tir
from tvm.ir import CallingConv
from tvm.target import Target
from tilelang.contrib import hipcc, nvcc
from tilelang.engine.param import KernelParam, CompiledArtifact
from tilelang.engine.tpu_config import resolve_tpu_runtime, resolve_tpu_target
from tilelang.utils.target import determine_target
from tilelang.engine.phase import (
    AssignTPUAddresses,
    LowerAndLegalize,
    OptimizeForTarget,
)


def is_cpu_device_backend(target: Target):
    return target.kind.name == "c"


def has_device_kernel_launch(attrs) -> bool:
    """Check if the attributes indicate a device kernel launch."""
    return bool(attrs and "calling_conv" in attrs and
                attrs["calling_conv"] == CallingConv.DEVICE_KERNEL_LAUNCH)


def is_device_call_c_device(func: tir.PrimFunc):
    attrs = func.attrs

    # Check if it's a C target
    if "target" in attrs and attrs["target"].kind.name == "c":
        return True

    return has_device_kernel_launch(attrs)


def is_device_call(func: tir.PrimFunc):
    return has_device_kernel_launch(func.attrs)


def get_device_call(is_device_c: bool = False) -> Callable[[tir.PrimFunc], bool]:
    return is_device_call_c_device if is_device_c else is_device_call


def get_host_call(is_device_c: bool = False) -> Callable[[tir.PrimFunc], bool]:
    return lambda func: not get_device_call(is_device_c)(func)


# ``tl.tpu.*`` is the backend-neutral semantic ABI produced by portable public
# helpers.  Backend-specific operations live in an explicit namespace:
# ``tl.tpukernel.*`` is the semantic TPU-Kernel ABI and ``rvt_`` is the expert
# raw RV ABI.  Raw ``tpu_*`` calls never formed a stable TIR contract: their C
# signatures, address spaces, and lifecycle requirements are SDK details.  We
# classify them only to reject them with a precise error, just like the removed
# historical ``ppl.*`` namespace.
_PORTABLE_TPU_EXTERN_PREFIX = "tl.tpu."
_TPUKERNEL_EXTERN_PREFIX = "tl.tpukernel."
_RVT_EXTERN_PREFIX = "rvt_"
_REMOVED_RAW_TPUKERNEL_EXTERN_PREFIX = "tpu_"
_REMOVED_PPL_EXTERN_PREFIX = "ppl."


def _is_valid_raw_rvt_symbol(name: str) -> bool:
    return re.fullmatch(r"rvt_[A-Za-z0-9_]+", name) is not None


# Keep this list deliberately closed.  Adding a semantic operation requires a
# frontend definition, address/effect analysis, target-specific codegen, and a
# contract test; accepting an arbitrary name from either namespace would let a
# typo fall through to CodeGenC's generic extern emitter.
_PORTABLE_TPU_EXTERNS = frozenset({
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
})
_TPUKERNEL_EXTERNS = frozenset({
    "tl.tpukernel.gather",
    "tl.tpukernel.topk",
})

# Exact tensor-region argument positions in the compiler-owned semantic ABI.
# Scalar/attribute arguments are intentionally absent.  Keeping this table
# closed prevents an arbitrary nested ``tl.region`` from becoming a residual
# BufferLoad escape hatch merely because it appears under a known extern.
_TPU_SEMANTIC_REGION_ARGS = {
    "tl.tpu.embedding": (1, 2, 3),
    "tl.tpu.copy": (1, 2),
    "tl.tpu.fill": (1,),
    "tl.tpu.gemm": (1, 2, 3),
    "tl.tpu.add": (1, 2, 3),
    "tl.tpu.sub": (1, 2, 3),
    "tl.tpu.mul": (1, 2, 3),
    "tl.tpu.div": (1, 2, 3),
    "tl.tpu.max": (1, 2, 3),
    "tl.tpu.add_scalar": (1, 2),
    "tl.tpu.mul_scalar": (1, 2),
    "tl.tpu.exp": (1, 2, 3, 4),
    "tl.tpukernel.gather": (1, 2, 3),
    "tl.tpukernel.topk": (1, 2, 3),
    "tl.tpu.rsqrt": (1, 2),
    "tl.tpu.reduce_sum": (1, 2, 3),
    "tl.tpu.reduce_max": (1, 2, 3),
}

# CUDA/HIP synchronization has no implicit TPU meaning.  Some operations have
# ``barrier`` in their registered name while the async-copy queue primitives do
# not, so retain both a pattern and an explicit set.
_GPU_SYNC_OPS = frozenset({
    "tir.tvm_storage_sync",
    "tir.tvm_global_barrier_kinit",
    "tir.ptx_commit_group",
    "tir.ptx_wait_group",
    "tl.SyncThreadsPartialOp",
    "tl.FenceProxyAsyncOp",
})


def _tpu_contract_error(target: Target, function_name: str, node_category: str,
                        detail: str) -> ValueError:
    """Create one diagnostic format for every residual-IR rejection."""
    return ValueError("TPU residual-IR contract violation: "
                      f"target={str(target)!r}; PrimFunc={function_name!r}; "
                      f"node category={node_category}; {detail}")


def _has_vector_lanes(dtype) -> bool:
    """Return whether a TVM dtype is a fixed/scalable vector dtype."""
    if dtype is None:
        return False
    try:
        return int(tvm.DataType(dtype).lanes) > 1
    except (AttributeError, TypeError, ValueError):
        # A symbolic/scalable lane count must also fail closed.  Scalar TVM
        # DataType instances always expose an integer lane count of one.
        return True


def _vector_dtype_in_type(type_annotation):
    """Return a vector element dtype nested in an IR type, if present."""
    if isinstance(type_annotation, tvm.ir.PrimType):
        return type_annotation.dtype if _has_vector_lanes(type_annotation.dtype) else None
    if isinstance(type_annotation, tvm.ir.PointerType):
        return _vector_dtype_in_type(type_annotation.element_type)
    if isinstance(type_annotation, tvm.ir.TupleType):
        for field in type_annotation.fields:
            vector_dtype = _vector_dtype_in_type(field)
            if vector_dtype is not None:
                return vector_dtype
    return None


def _tpu_extern_programming_model(call: tir.Call) -> Optional[str]:
    """Classify a known TPU ``tir.call_extern`` without guessing other calls."""
    if getattr(call.op, "name", None) != "tir.call_extern" or not call.args:
        return None
    name = getattr(call.args[0], "value", None)
    if not isinstance(name, str):
        return None
    if name in _PORTABLE_TPU_EXTERNS:
        return "portable"
    if name.startswith(_PORTABLE_TPU_EXTERN_PREFIX):
        return "unknown-portable"
    if name.startswith(_REMOVED_PPL_EXTERN_PREFIX):
        return "removed-ppl"
    if name.startswith(_REMOVED_RAW_TPUKERNEL_EXTERN_PREFIX):
        return "removed-raw-tpukernel"
    if name.startswith(_RVT_EXTERN_PREFIX):
        return "rv"
    if name in _TPUKERNEL_EXTERNS:
        return "tpukernel"
    if name.startswith(_TPUKERNEL_EXTERN_PREFIX):
        return "unknown-tpukernel"
    return None


def _collect_tpu_externs(mod: tvm.IRModule):
    """Return classified TPU externs without treating arbitrary calls as TPU.

    This is intentionally limited to the vendor namespaces above.  The result
    is used both to validate a selected TPU programming model and to prevent a
    TPU-specific program from silently lowering as C/CUDA/HIP after
    ``target='auto'`` no longer guesses a TPU.
    """
    externs = []
    for global_var, function in mod.functions.items():
        if not isinstance(function, tir.PrimFunc):
            continue

        def visit(node, function_name=global_var.name_hint):
            if not isinstance(node, tir.Call):
                return
            model = _tpu_extern_programming_model(node)
            if model is not None:
                name = getattr(node.args[0], "value", "<unknown>")
                externs.append((function_name, str(name), model))

        tir.stmt_functor.post_order_visit(function.body, visit)
    return externs


def _reject_tpu_externs_for_non_tpu_target(mod: tvm.IRModule, target: Target) -> None:
    """Fail closed instead of emitting TPU externs into an unrelated backend."""
    externs = _collect_tpu_externs(mod)
    if not externs:
        return
    rendered = ", ".join(f"{func}: {name} ({model})" for func, name, model in externs)
    raise ValueError(f"TPU externs cannot lower for target={target.kind.name!r}: {rendered}. "
                     "Use a complete TPU target such as 'tpu -mcpu=sg2260e "
                     "-tpu-programming-model=rv'. Portable tl.tpu.* calls select their "
                     "backend through the Target; explicit tl.tpukernel.* and raw rvt_* "
                     "calls must match that programming model.")


def _validate_tpu_residual_ir(mod: tvm.IRModule, target: Target, tpu_config) -> None:
    """Reject residual IR without a defined TPU source-emission contract.

    This verifier intentionally runs both before and after target passes.  The
    first check gives frontend authors an immediate diagnostic; the second
    check prevents a pass from introducing CUDA vector/synchronization IR or
    an unowned external call immediately before address assignment/codegen.
    """
    for global_var, function in mod.functions.items():
        if not isinstance(function, tir.PrimFunc):
            continue
        function_name = global_var.name_hint

        # Source PrimFuncs are not target-bound until LowerAndLegalize.  An
        # explicitly different target is not a harmless mixed-module member:
        # the TPU build path emits the full module through one TPU codegen
        # invocation, so skipping it here would let that function cross the
        # wrong backend boundary.
        function_target = function.attrs.get("target") if function.attrs else None
        if function_target is not None and function_target.kind.name != "tpu":
            raise _tpu_contract_error(
                target, function_name, "PrimFunc-target", "function is bound to target kind "
                f"{function_target.kind.name!r}, but the module is being "
                "compiled as TPU")
        if function_target is not None:
            function_tpu_target = resolve_tpu_target(target=function_target)
            if function_tpu_target != tpu_config:
                raise _tpu_contract_error(
                    target, function_name, "PrimFunc-target",
                    "function TPU identity disagrees with the compilation "
                    f"target: function={function_tpu_target}, "
                    f"compilation={tpu_config}")

        externs_by_model = {}

        def collect_extern_model(node, externs_by_model=externs_by_model):
            if not isinstance(node, tir.Call):
                return
            model = _tpu_extern_programming_model(node)
            if model is None:
                return
            extern_name = getattr(node.args[0], "value", "<dynamic-name>")
            externs_by_model.setdefault(model, []).append(str(extern_name))

        tir.stmt_functor.post_order_visit(function.body, collect_extern_model)

        if (tpu_config.chip == "sg2260e" and
                "tl.tpukernel.topk" in externs_by_model.get("tpukernel", ())):
            raise _tpu_contract_error(
                target, function_name, "target-capability",
                "tl.tpukernel.topk is unavailable on SG2260E: the PPL 1.7 "
                "tpub_7_1_e runtime rejects tpu_hau_sort_natural_index")

        for parameter in function.params:
            vector_dtype = _vector_dtype_in_type(parameter.type_annotation)
            if (_has_vector_lanes(getattr(parameter, "dtype", None)) or vector_dtype is not None):
                raise _tpu_contract_error(
                    target, function_name, "vector-parameter-dtype",
                    f"parameter {parameter.name!r} has unsupported dtype "
                    f"{vector_dtype or parameter.dtype}")
            if parameter not in function.buffer_map:
                raise _tpu_contract_error(
                    target, function_name, "scalar-parameter",
                    f"parameter {parameter.name!r} is not buffer-backed; "
                    "the TPU host/device ABI currently marshals Tensor "
                    "parameters only")
        return_vector_dtype = _vector_dtype_in_type(function.ret_type)
        if return_vector_dtype is not None:
            raise _tpu_contract_error(
                target, function_name, "vector-return-dtype",
                f"PrimFunc return type contains unsupported dtype "
                f"{return_vector_dtype}")
        for buffer in function.buffer_map.values():
            if _has_vector_lanes(buffer.dtype):
                raise _tpu_contract_error(
                    target, function_name, "vector-buffer-dtype",
                    f"buffer {buffer.name!r} has unsupported dtype {buffer.dtype}")

        # Raw RVT APIs consume user-managed CR/TR/GR register encodings, not
        # TileLang Buffer data pointers.  Keep the two ownership domains
        # separate even when a caller tries to hide a descriptor Var inside a
        # scalar expression.
        descriptor_data_vars = {buffer.data for buffer in function.buffer_map.values()}
        # Buffer data Vars and their handle parameters are distinct identities
        # in legal TIR.  Record both.  Scalar parameters have already been
        # rejected by the host ABI check above and must not be classified as
        # descriptors.
        descriptor_data_vars.update(
            parameter for parameter in function.params if parameter.dtype == "handle")

        def collect_descriptor_data_vars(node, descriptor_data_vars=descriptor_data_vars):
            if isinstance(node, tir.Allocate):
                descriptor_data_vars.add(node.buffer_var)
            elif isinstance(node, (tir.DeclBuffer, tir.BufferRealize)):
                descriptor_data_vars.add(node.buffer.data)

        tir.stmt_functor.post_order_visit(function.body, collect_descriptor_data_vars)

        # Every compiler-owned TPU semantic extern carries tensor operands as
        # direct ``tl.region`` children.  Their BufferLoad/Ramp nodes are
        # descriptor markers, not executable scalar/vector accesses.  Record
        # only structurally closed region children here; malformed or
        # standalone regions still reach the residual-IR rejection below.
        semantic_region_markers = set()
        canonical_semantic_calls = set()

        def collect_semantic_region_markers(node,
                                            function_name=function_name,
                                            semantic_region_markers=semantic_region_markers,
                                            canonical_semantic_calls=canonical_semantic_calls):
            if (not isinstance(node, tir.Call) or
                    getattr(node.op, "name", None) != "tir.call_extern" or not node.args):
                return
            extern_name = getattr(node.args[0], "value", None)
            positions = _TPU_SEMANTIC_REGION_ARGS.get(extern_name)
            if positions is None or len(node.args) <= positions[-1]:
                return
            regions = [node.args[position] for position in positions]
            if any(not isinstance(region, tir.Call) or
                   getattr(region.op, "name", None) != "tl.region" for region in regions):
                return
            for position, region in zip(positions, regions):  # noqa: B905
                rank = len(region.args) - 2
                if not 1 <= rank <= 4:
                    raise _tpu_contract_error(
                        target, function_name, "semantic-region-ABI",
                        f"{extern_name} tensor argument {position} has rank "
                        f"{rank}; tl.region descriptors require rank 1 "
                        "through 4")
                if (not isinstance(region.args[0], tir.BufferLoad) or
                        len(region.args[0].indices) != rank):
                    raise _tpu_contract_error(
                        target, function_name, "semantic-region-ABI",
                        f"{extern_name} tensor argument {position} must start "
                        "with a BufferLoad whose index rank matches its "
                        "logical extents")

            region_ranks = tuple(len(region.args) - 2 for region in regions)
            region_shapes = tuple(
                tuple(getattr(extent, "value", None)
                      for extent in region.args[2:])
                for region in regions)
            region_data = tuple(region.args[0].buffer.data for region in regions)
            region_dtypes = tuple(str(region.args[0].buffer.dtype) for region in regions)
            if extern_name == "tl.tpu.rsqrt":
                if region_ranks[0] != region_ranks[1]:
                    raise _tpu_contract_error(target, function_name, "semantic-region-ABI",
                                              f"{extern_name} requires matching dst/src ranks")
                if region_shapes[0] != region_shapes[1]:
                    raise _tpu_contract_error(target, function_name, "semantic-region-ABI",
                                              f"{extern_name} requires matching dst/src shapes")
            elif extern_name == "tl.tpu.gemm":
                if any(rank != 2 for rank in region_ranks):
                    operand = ("A", "B", "C")[next(
                        index for index, rank in enumerate(region_ranks) if rank != 2)]
                    raise _tpu_contract_error(target, function_name, "semantic-region-ABI",
                                              f"{extern_name} requires rank-2 {operand}")
                if region_data[2] in region_data[:2]:
                    raise _tpu_contract_error(
                        target, function_name, "semantic-region-ABI",
                        f"{extern_name} output/accumulator C must use storage "
                        "distinct from A and B")
            elif extern_name == "tl.tpu.exp":
                coefficient_index = 3
                if (region_ranks[coefficient_index] != 2 or
                        region_shapes[coefficient_index] != (64, 32)):
                    raise _tpu_contract_error(
                        target, function_name, "semantic-region-ABI",
                        f"{extern_name} coefficient buffer must have shape (64, 32)")
                if (tpu_config.programming_model == "rv" and
                        any(dtype != "float32" for dtype in region_dtypes)):
                    raise _tpu_contract_error(target, function_name, "target-capability",
                                              f"{extern_name} RV lowering requires FP32 tensors")

            canonical_semantic_calls.add(node)
            for region in regions:
                marker = region.args[0]
                semantic_region_markers.add(marker)
                # A Ramp used directly as a region minimum describes the
                # vector-width extent; it is consumed structurally by the TPU
                # region lowering and is not emitted as vector code.  Copy is
                # the only typed semantic that permits subregions/Ramps; its
                # contiguous-region rule is checked early for a clear error.
                for axis, index in enumerate(marker.indices):
                    if isinstance(index, tir.Ramp):
                        if extern_name != "tl.tpu.copy":
                            continue
                        stride = getattr(index.stride, "value", None)
                        lanes = getattr(index.lanes, "value", None)
                        extent = getattr(region.args[axis + 2], "value", None)
                        if stride != 1:
                            raise _tpu_contract_error(
                                target, function_name, "copy-region-ramp",
                                "tl.tpu.copy represents contiguous regions; "
                                f"axis {axis} Ramp must have unit stride, got "
                                f"{index.stride}")
                        if lanes is None or extent is None or lanes != extent:
                            raise _tpu_contract_error(
                                target, function_name, "copy-region-ramp",
                                "tl.tpu.copy Ramp lane count must equal its "
                                f"explicit region extent on axis {axis}; got "
                                f"lanes={index.lanes}, extent="
                                f"{region.args[axis + 2]}")
                        semantic_region_markers.add(index)

        tir.stmt_functor.post_order_visit(function.body, collect_semantic_region_markers)

        # TIR is a DAG, so the same BufferLoad/Ramp ObjectRef can have more
        # than one parent.  A marker is exempt only while reached through its
        # canonical semantic call; reusing that exact node as an executable
        # load or vector expression elsewhere must still fail closed.
        def reject_reused_semantic_marker(node,
                                          function_name=function_name,
                                          canonical_semantic_calls=canonical_semantic_calls,
                                          semantic_region_markers=semantic_region_markers):
            if node in canonical_semantic_calls:
                return False
            if node in semantic_region_markers:
                raise _tpu_contract_error(
                    target, function_name, "semantic-region-marker-alias",
                    "a BufferLoad/Ramp descriptor marker is also referenced "
                    "outside its canonical TPU semantic region")
            return True

        tir.stmt_functor.pre_order_visit(function.body, reject_reused_semantic_marker)

        def visit(node,
                  function_name=function_name,
                  externs_by_model=externs_by_model,
                  descriptor_data_vars=descriptor_data_vars,
                  semantic_region_markers=semantic_region_markers):
            if isinstance(node, tir.Allocate):
                condition = node.condition
                if (not isinstance(condition, tir.IntImm) or int(condition.value) != 1):
                    raise _tpu_contract_error(
                        target, function_name, "Allocate-condition",
                        f"allocation {node.buffer_var.name!r} has condition "
                        f"{condition}; TPU descriptor allocation is "
                        "unconditional and requires compile-time true")

            if isinstance(node, tir.AllocateConst):
                raise _tpu_contract_error(
                    target, function_name, "AllocateConst",
                    "constant arrays have no TPU descriptor/load contract; "
                    "introduce a typed constant-table operation before "
                    "enabling this node")

            if isinstance(node, tir.CustomizedCode):
                raise _tpu_contract_error(
                    target, function_name, "CustomizedCode",
                    "verbatim source injection bypasses TPU programming-model, "
                    "descriptor, and command-lifecycle validation")

            if isinstance(
                    node,
                (tir.BufferRealize, tir.ProducerStore, tir.ProducerRealize, tir.Prefetch)):
                raise _tpu_contract_error(
                    target, function_name,
                    type(node).__name__, "node has no residual TPU source-emission contract; "
                    "consume it in a target pass before codegen")

            if isinstance(node, tir.ProducerLoad):
                raise _tpu_contract_error(target, function_name, "ProducerLoad",
                                          "producer loads have no TPU descriptor lowering")

            if (isinstance(node, tir.For) and
                    node.kind not in (tir.ForKind.SERIAL, tir.ForKind.UNROLLED)):
                raise _tpu_contract_error(
                    target, function_name, "For",
                    f"loop kind {node.kind} has no TPU execution mapping; "
                    "only serial and unrolled loops may reach source codegen")

            if isinstance(node, tir.AttrStmt):
                raise _tpu_contract_error(
                    target, function_name, "AttrStmt",
                    f"attribute {node.attr_key!r} has no residual TPU meaning; "
                    "consume it in a target pass before source codegen")

            if isinstance(node, tir.Ramp):
                if node in semantic_region_markers:
                    return
                raise _tpu_contract_error(target, function_name, "Ramp",
                                          f"vector index {node} has no TPU residual-IR lowering")

            if isinstance(node, (tir.Allocate, tir.AllocateConst)) and \
                    _has_vector_lanes(node.dtype):
                raise _tpu_contract_error(
                    target, function_name, "vector-allocation-dtype",
                    f"allocation {node.buffer_var.name!r} has unsupported "
                    f"dtype {node.dtype}")

            if isinstance(node, (tir.DeclBuffer, tir.BufferRealize)) and \
                    _has_vector_lanes(node.buffer.dtype):
                raise _tpu_contract_error(
                    target, function_name, "vector-buffer-dtype",
                    f"buffer {node.buffer.name!r} has unsupported dtype "
                    f"{node.buffer.dtype}")

            if isinstance(node, tir.BufferLoad):
                if node in semantic_region_markers:
                    return
                raise _tpu_contract_error(
                    target, function_name, "BufferLoad", f"direct scalar/vector load from buffer "
                    f"{node.buffer.name!r} has no TPU descriptor lowering; "
                    "move data with tl.tpu.copy and compute through a typed "
                    "TPU semantic operation")

            if isinstance(node, tir.BufferStore):
                raise _tpu_contract_error(
                    target, function_name, "BufferStore", f"direct scalar/vector store to buffer "
                    f"{node.buffer.name!r} has no TPU descriptor lowering; "
                    "move data with tl.tpu.copy and compute through a typed "
                    "TPU semantic operation")

            if not isinstance(node, tir.Call):
                if isinstance(node, tir.PrimExpr) and _has_vector_lanes(node.dtype):
                    raise _tpu_contract_error(
                        target, function_name, "vector-dtype",
                        f"{type(node).__name__} has unsupported dtype {node.dtype}")
                return

            op_name = getattr(node.op, "name", "<unregistered-call>")
            if _has_vector_lanes(node.dtype):
                raise _tpu_contract_error(
                    target, function_name, "vector-call",
                    f"call {op_name!r} returns unsupported dtype {node.dtype}")

            lowered_op_name = op_name.lower()
            if (op_name in _GPU_SYNC_OPS or "barrier" in lowered_op_name):
                raise _tpu_contract_error(
                    target, function_name, "gpu-synchronization",
                    f"GPU synchronization intrinsic {op_name!r} has no TPU "
                    "residual-IR meaning")

            if op_name == "tir.call_pure_extern":
                extern_name = (
                    getattr(node.args[0], "value", "<dynamic-name>")
                    if node.args else "<missing-name>")
                raise _tpu_contract_error(
                    target, function_name, "call_pure_extern",
                    f"pure external call {extern_name!r} is not part of the "
                    "side-effecting TPU semantic ABI")
            if op_name == "tir.tvm_access_ptr":
                raise _tpu_contract_error(
                    target, function_name, "tensor-operand-ABI",
                    "bare tir.tvm_access_ptr is not a TPU tensor operand; "
                    "compiler-owned semantic calls require a whole-buffer "
                    "tl.region so logical shape and access direction remain "
                    "available to native codegen")
            if op_name != "tir.call_extern":
                return

            extern_name = (getattr(node.args[0], "value", None) if node.args else None)
            if not isinstance(extern_name, str):
                raise _tpu_contract_error(target, function_name, "call_extern",
                                          "external function name must be a compile-time string")
            model = _tpu_extern_programming_model(node)
            if model == "portable":
                return
            if model == "tpukernel":
                if tpu_config.programming_model == "tpukernel":
                    return
                raise _tpu_contract_error(
                    target, function_name, "call_extern",
                    "externs from a different programming model cannot be "
                    f"lowered: {extern_name!r} requires 'tpukernel', selected "
                    f"{tpu_config.programming_model!r}; incompatible externs: "
                    f"{', '.join(externs_by_model.get('tpukernel', []))}")
            if model == "rv":
                if tpu_config.programming_model == "rv":
                    if not _is_valid_raw_rvt_symbol(extern_name):
                        raise _tpu_contract_error(
                            target, function_name, "raw-rvt-symbol",
                            "raw RVT extern name must be a C identifier "
                            f"beginning with 'rvt_'; got {extern_name!r}")
                    referenced_descriptors = set()

                    def find_descriptor_var(candidate, descriptor_data_vars=descriptor_data_vars):
                        if (isinstance(candidate, tir.Var) and candidate in descriptor_data_vars):
                            referenced_descriptors.add(candidate.name)

                    for argument in node.args[1:]:
                        tir.stmt_functor.post_order_visit(argument, find_descriptor_var)
                    if referenced_descriptors:
                        raise _tpu_contract_error(
                            target, function_name, "raw-rvt-descriptor-argument",
                            f"raw RVT extern {extern_name!r} references TileLang "
                            "tensor descriptor Var(s) "
                            f"{', '.join(sorted(referenced_descriptors))}; pass "
                            "user-managed RVT register encodings, or use a "
                            "compiler-owned tl.tpu.* operation")
                    return
                raise _tpu_contract_error(
                    target, function_name, "call_extern",
                    "externs from a different programming model cannot be "
                    f"lowered: {extern_name!r} requires 'rv', selected "
                    f"{tpu_config.programming_model!r}; incompatible externs: "
                    f"{', '.join(externs_by_model.get('rv', []))}")
            if model == "removed-ppl":
                raise _tpu_contract_error(
                    target, function_name, "call_extern",
                    "The internal ppl.* TIR ABI has been removed; use "
                    "backend-neutral tl.tpu.* or explicit tl.tpukernel.* "
                    f"operations. Found {extern_name!r}")
            if model == "removed-raw-tpukernel":
                raise _tpu_contract_error(
                    target, function_name, "call_extern",
                    "Raw tpu_* call_extern is not a supported TIR ABI; use an "
                    "explicit tl.tpukernel.* semantic operation. Found "
                    f"{extern_name!r}")
            if model in {"unknown-portable", "unknown-tpukernel"}:
                raise _tpu_contract_error(
                    target, function_name, "call_extern",
                    f"unknown TPU semantic extern {extern_name!r}; only "
                    "compiler-owned operations in the closed tl.tpu.* and "
                    "tl.tpukernel.* sets are allowed")
            raise _tpu_contract_error(
                target, function_name, "call_extern",
                f"unknown external call {extern_name!r}; TPU kernels only "
                "allow compiler-owned tl.tpu.*, matching tl.tpukernel.*, or "
                "raw rvt_* calls isolated to the RV programming model")

        tir.stmt_functor.post_order_visit(function.body, visit)


def validate_target_module_contract(mod: tvm.IRModule, target: Target):
    """Validate backend extern ownership and return the TPU target identity.

    This is shared by the full lowering path and annotation-only wrapper
    analysis so neither entry point can bypass complete TPU Target validation.
    Non-TPU modules return ``None`` after rejecting any TPU-only externs.
    """
    if target.kind.name == "tpu":
        tpu_target = resolve_tpu_target(target=target)
        _validate_tpu_residual_ir(mod, target, tpu_target)
        return tpu_target
    _reject_tpu_externs_for_non_tpu_target(mod, target)
    return None


@tvm.register_func("tilelang_callback_cuda_compile", override=True)
def tilelang_callback_cuda_compile(code, target):
    project_root = osp.join(osp.dirname(__file__), "../..")
    if "TL_TEMPLATE_PATH" in os.environ:
        tl_template_path = os.environ["TL_TEMPLATE_PATH"]
    else:
        tl_template_path = osp.abspath(osp.join(project_root, "src"))
    # TODO(lei): this indeed should be renamed into
    # TL_CUTLASS_INCLUDE_PATH in the future
    if "TL_CUTLASS_PATH" in os.environ:
        cutlass_path = os.environ["TL_CUTLASS_PATH"]
    else:
        cutlass_path = osp.abspath(osp.join(project_root, "3rdparty/cutlass/include"))
    compute_version = "".join(nvcc.get_target_compute_version(target).split("."))

    # special handle for Hopper
    if compute_version == "90":
        arch = ["-arch=sm_90a"]
        format = "cubin"
    else:
        arch = [f"-arch=sm_{compute_version}"]
        format = "cubin"

    # printing out number of registers
    debug_option = "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage"
    ptx = nvcc.compile_cuda(
        code,
        format,
        arch,
        options=[
            "-std=c++17",
            debug_option,
            "--use_fast_math",
            "-I" + tl_template_path,
            "-I" + cutlass_path,
        ],
        verbose=False,
    )

    return ptx


@tvm.register_func("tilelang_callback_hip_compile", override=True)
def tilelang_callback_hip_compile(code, target):
    project_root = osp.join(osp.dirname(__file__), "../..")
    tl_template_path = osp.abspath(osp.join(project_root, "src"))

    # TODO(lei): actually this indeed should be renamed into
    # TL_COMPOSABLE_KERNEL_INCLUDE_PATH in the future
    if "TL_COMPOSABLE_KERNEL_PATH" in os.environ:
        ck_path = os.environ["TL_COMPOSABLE_KERNEL_PATH"]
    else:
        ck_path = osp.abspath(osp.join(project_root, "3rdparty/composable_kernel/include"))

    hsaco = hipcc.compile_hip(
        code,
        target_format="hsaco",
        options=[
            "-std=c++17",
            "-I" + tl_template_path,
            "-I" + ck_path,
        ],
        verbose=False,
    )

    return hsaco


def extrac_params(func: tir.PrimFunc) -> List[KernelParam]:
    tensor_types = []
    for var in func.params:
        if var in func.buffer_map:
            tensor_types.append(KernelParam.from_buffer(func.buffer_map[var]))
        else:
            tensor_types.append(KernelParam.from_var(var))
    return tensor_types


def canon_target_host(target: Union[str, Target], target_host: Optional[Union[str, Target]]):

    if not target_host:
        target_host = "llvm" if tvm.runtime.enabled("llvm") else "stackvm"

    return target_host


def host_codegen(host_mod: tvm.IRModule, target_host: Target) -> tvm.IRModule:
    host_mod = tir.transform.BindTarget(target_host)(host_mod)
    host_mod = tir.transform.FP8StorageLegalize()(host_mod)
    host_mod = tir.transform.BF16StorageLegalize()(host_mod)
    host_mod = tir.transform.LowerTVMBuiltin()(host_mod)
    host_mod = tir.transform.LowerCustomDatatypes()(host_mod)
    host_mod = tir.transform.LowerIntrin()(host_mod)
    host_mod = tilelang.transform.LowerDeviceStorageAccessInfo()(host_mod)
    host_mod = tir.transform.CombineContextCall()(host_mod)
    if target_host.kind.name == "llvm":
        host_mod = tvm._ffi.get_global_func("target.build.llvm")(host_mod, target_host)
    elif target_host.kind.name == "c":
        host_mod = tvm._ffi.get_global_func("target.build.c")(host_mod, target_host)
    else:
        raise ValueError(f"Target host {target_host.kind.name} is not supported")
    return host_mod


def device_codegen(device_mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    device_mod = tilelang.transform.LowerDeviceStorageAccessInfo()(device_mod)
    device_mod = tir.transform.LowerIntrin()(device_mod)
    device_mod = tir.transform.Simplify()(device_mod)

    if target.kind.name == "cuda":
        device_mod = tvm._ffi.get_global_func("target.build.tilelang_cuda")(device_mod, target)
    elif target.kind.name == "hip":
        device_mod = tvm._ffi.get_global_func("target.build.tilelang_hip")(device_mod, target)
    else:
        raise ValueError(f"Target {target.kind.name} is not supported")

    return device_mod


def device_codegen_without_compile(device_mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    device_mod = tilelang.transform.LowerDeviceStorageAccessInfo()(device_mod)
    device_mod = tir.transform.LowerIntrin()(device_mod)
    device_mod = tir.transform.Simplify()(device_mod)
    if target.kind.name == "cuda":
        device_mod = tvm._ffi.get_global_func("target.build.tilelang_cuda_without_compile")(
            device_mod, target)
    elif target.kind.name == "hip":
        device_mod = tvm._ffi.get_global_func("target.build.tilelang_hip_without_compile")(
            device_mod, target)
    elif target.kind.name == "c":
        device_mod = tvm._ffi.get_global_func("target.build.tilelang_cpp")(device_mod, target)
    elif target.kind.name == "llvm":
        device_mod = tvm._ffi.get_global_func("target.build.llvm")(device_mod, target)
    elif target.kind.name == "webgpu":
        device_mod = tvm._ffi.get_global_func("target.build.tilelang_webgpu")(device_mod, target)
    else:
        raise ValueError(f"Target {target.kind.name} is not supported")

    return device_mod


def lower(
    func_or_mod: Union[tir.PrimFunc, tvm.IRModule],
    target: Union[str, Target] = "auto",
    target_host: Optional[Union[str, Target]] = None,
    runtime_only=False,
    enable_host_codegen=False,
    enable_device_compile=False,
    runtime_mode: Optional[str] = None,
) -> CompiledArtifact:
    '''
        enable_host_codegen: whether to enable host codegen, default is False, as we have our
        own host codegen implementation in jit.
        enable_device_compile: whether to enable device codegen, default is False, as we have our
        own device codegen implementation in jit.
    '''

    mod = func_or_mod
    params = None
    if isinstance(func_or_mod, tir.PrimFunc):
        func = func_or_mod
        params = extrac_params(func) if not runtime_only else None
        mod = tvm.IRModule({func.attrs["global_symbol"]: func})

    if isinstance(target, str):
        target = determine_target(target)

    target_host = canon_target_host(target, target_host)

    target_host = tvm.target.Target.canon_target(target_host)
    target = tvm.target.Target(target, target_host)
    is_tpu = target.kind.name == "tpu"
    tpu_target = None
    tpu_runtime = None
    if is_tpu:
        # TPU target selection is resolved at the backend boundary.  Do not
        # manufacture a BM1690 configuration while lowering CUDA/HIP/etc.;
        # those backends keep their own target and codegen paths.
        tpu_target = validate_target_module_contract(mod, target)
        tpu_runtime = resolve_tpu_runtime(runtime_mode=runtime_mode)
    else:
        if runtime_mode is not None:
            raise ValueError("runtime_mode is only valid for a TPU target")
        validate_target_module_contract(mod, target)

    _is_host_call = get_host_call(is_device_c=is_cpu_device_backend(target))
    _is_device_call = get_device_call(is_device_c=is_cpu_device_backend(target))

    # Phase 1: Lower and legalize the IR
    mod = LowerAndLegalize(mod, target)

    # Phase 2: Optimize the IR for the target
    mod = OptimizeForTarget(mod, target)
    if is_tpu:
        # Passes may introduce or rewrite extern calls. Validate the final TIR
        # contract before the TPU-specific address/effect analysis consumes it.
        validate_target_module_contract(mod, target)
        mod = AssignTPUAddresses(mod, target)
    host_mod = tir.transform.Filter(_is_host_call)(mod)
    device_mod = tir.transform.Filter(_is_device_call)(mod)

    if is_tpu:
        # TPU codegen needs the full module because host/device ownership is
        # represented by the generated TPU ABI rather than TVM's ordinary
        # device module split.
        kernel_source = tvm._ffi.get_global_func("target.build.tilelang_tpu")(mod, target)
        return CompiledArtifact(
            host_mod,
            device_mod,
            params,
            kernel_source,
            tpu_target=tpu_target,
            tpu_runtime=tpu_runtime,
        )

    # Preserve the normal TileLang backend dispatch.  TPU capability/config
    # logic is intentionally absent from this branch, just as CUDA and HIP
    # target choices are isolated from one another.
    codegen_mod = (
        device_codegen(device_mod, target)
        if enable_device_compile else device_codegen_without_compile(device_mod, target))
    if enable_host_codegen:
        host_rt_mod = host_codegen(host_mod, target_host)
        host_rt_mod.import_module(codegen_mod)
        return CompiledArtifact(
            host_rt_mod, device_mod, params, codegen_mod.get_source(), rt_mod=host_rt_mod)

    return CompiledArtifact(host_mod, device_mod, params, codegen_mod.get_source())
