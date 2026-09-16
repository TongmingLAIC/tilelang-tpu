# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
from tvm import tir, IRModule
from tvm.target import Target
import tilelang
from tilelang.engine.tpu_config import resolve_tpu_target


def _validate_tpu_phase_target(mod: IRModule, target: Target, phase_name: str) -> None:
    """Require one complete, consistent TPU identity at a public TPU phase.

    Full lowering validates this contract before entering the pass pipeline,
    but these phase helpers are also imported and called directly by tests and
    downstream tooling.  Do not let that shorter path turn a bare/unknown TPU
    target, or a module already bound to another target, into the shared TPUv7
    pass pipeline.  Unbound source PrimFuncs remain valid so
    ``LowerAndLegalize`` can bind them; the normal pipeline reaches
    ``OptimizeForTarget`` only after that binding.
    """
    selected_target = resolve_tpu_target(target=target)
    for global_var, function in mod.functions.items():
        if not isinstance(function, tir.PrimFunc):
            continue
        function_target = function.attrs.get("target") if function.attrs else None
        if function_target is None:
            continue
        if function_target.kind.name != "tpu":
            raise ValueError(
                f"{phase_name} cannot process PrimFunc {global_var.name_hint!r} bound to "
                f"target kind {function_target.kind.name!r} with a TPU target")
        function_selection = resolve_tpu_target(target=function_target)
        if function_selection != selected_target:
            raise ValueError(f"{phase_name} target identity mismatch for PrimFunc "
                             f"{global_var.name_hint!r}: function={function_selection}, "
                             f"requested={selected_target}")


def LowerAndLegalize(mod: IRModule, target: Target) -> IRModule:
    """Bind and legalize frontend IR for the selected backend.

    TPU deliberately retains semantic ``tl.tpu.*`` externs and scalar loops.
    Layout/tile-op lowering, safe-memory rewriting, and vector legalization
    remain outside this path until they have an explicit TPU contract.
    """
    if target.kind.name == "tpu":
        _validate_tpu_phase_target(mod, target, "LowerAndLegalize")
    mod = tir.transform.BindTarget(target)(mod)

    mod = tilelang.transform.FrontendLegalize()(mod)
    mod = tir.transform.Simplify()(mod)
    # The TPU source generator does not yet implement residual vector Ramp/load
    # expressions.  Preserve scalar loops until that codegen contract exists;
    # turning an explicit vectorized loop into vector IR here would otherwise
    # let code generation silently omit parts of an expression.
    if target.kind.name != "tpu":
        mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    mod = tir.transform.Simplify()(mod)

    return mod


def _finalize_scheduled_ir(
    mod: IRModule,
    *,
    rewrite_storage: bool,
    lower_opaque: bool = True,
    vectorize: bool = True,
) -> IRModule:
    """Run passes whose ordering is shared after target-specific scheduling."""
    if lower_opaque:
        mod = tir.transform.LowerOpaqueBlock()(mod)
    mod = tir.transform.NarrowDataType(32)(mod)
    mod = tir.transform.Simplify()(mod)
    if vectorize:
        mod = tilelang.transform.VectorizeLoop()(mod)
    if rewrite_storage:
        mod = tir.transform.StorageRewrite()(mod)
    mod = tir.transform.UnrollLoop()(mod)
    mod = tir.transform.RenormalizeSplitPattern()(mod)
    return tir.transform.Simplify()(mod)


def _optimize_tpu(mod: IRModule) -> IRModule:
    """Apply only transformations with a validated conservative TPU meaning."""
    mod = tilelang.transform.IfStmtBinding()(mod)
    mod = tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.MergeIfStmt()(mod)
    # StorageRewrite assumes flattened storage and can duplicate the
    # structured DeclBuffer/Allocate pairs consumed by TPU codegen. BM1690 and
    # SG2260E share this LMEM geometry; their programming models diverge later
    # during target-selected code generation, not in these semantic passes.
    # Software-pipeline planning/injection remains disabled until TPU DMA and
    # compute operations have an explicit dependency/token model.  Likewise,
    # vectorization must not run before TPU code generation supports residual
    # vector IR.  Both optimizations have previously produced source that was
    # syntactically plausible but did not preserve the serial program.
    return _finalize_scheduled_ir(mod, rewrite_storage=False, vectorize=False)


def _optimize_hopper(mod: IRModule) -> IRModule:
    mod = tilelang.transform.IfStmtBinding()(mod)
    mod = tilelang.transform.MultiVersionBuffer()(mod)
    mod = tilelang.transform.WarpSpecialized()(mod)
    mod = tilelang.transform.InjectSoftwarePipeline()(mod)
    mod = tir.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.MergeIfStmt()(mod)
    mod = tilelang.transform.RewriteWgmmaSync()(mod)
    mod = tilelang.transform.InjectFenceProxy()(mod)
    return _finalize_scheduled_ir(mod, rewrite_storage=True, lower_opaque=False)


def _optimize_generic(mod: IRModule) -> IRModule:
    mod = tilelang.transform.IfStmtBinding()(mod)
    mod = tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.PipelinePlanning()(mod)
    mod = tilelang.transform.InjectSoftwarePipeline()(mod)
    mod = tilelang.transform.MergeIfStmt()(mod)
    return _finalize_scheduled_ir(mod, rewrite_storage=True)


def OptimizeForTarget(mod: IRModule, target: Target) -> IRModule:
    """Dispatch one explicit pass pipeline per backend family."""
    if target.kind.name == "tpu":
        _validate_tpu_phase_target(mod, target, "OptimizeForTarget")
        return _optimize_tpu(mod)
    if target.kind.name == "cuda" and target.arch == "sm_90":
        return _optimize_hopper(mod)
    return _optimize_generic(mod)


def _uniquify_tpu_local_names(function):
    """Give hygienic macro allocations distinct names for the LMEM attr ABI.

    TIR identity remains authoritative: two DeclBuffers sharing one data Var
    are still rejected by AddressAssign. Only different Vars with colliding
    hints are renamed, including their Allocate definitions and all uses.
    """
    variables = []

    def collect(node):
        if isinstance(node, tir.Allocate):
            variables.append(node.buffer_var)

    tir.stmt_functor.post_order_visit(function.body, collect)
    reserved = {v.name for v in variables}
    seen = set()
    mapping = {}
    for var in variables:
        if var.name in seen:
            suffix = 1
            name = f"{var.name}_macro{suffix}"
            while name in reserved:
                suffix += 1
                name = f"{var.name}_macro{suffix}"
            reserved.add(name)
            mapping[var] = tir.Var(name, var.type_annotation, var.span)
        seen.add(var.name)
    if not mapping:
        return function
    body = tir.stmt_functor.substitute(function.body, mapping)

    def rename_definition(node):
        if isinstance(node, tir.Allocate) and node.buffer_var in mapping:
            return tir.Allocate(mapping[node.buffer_var], node.dtype, node.extents, node.condition,
                                node.body, node.annotations, node.span)
        return None

    body = tir.stmt_functor.ir_transform(body, None, rename_definition, ["tir.Allocate"])
    return function.with_body(body)


def AssignTPUAddresses(mod: IRModule, target: Target) -> IRModule:
    """Assign TPUv7 LMEM addresses after the final TPU contract check.

    Address assignment consumes backend-specific operand effects and memory
    geometry, so accepting a non-TPU target here would be a compiler bug rather
    than a harmless no-op.
    """
    if target.kind.name != "tpu":
        raise ValueError("AssignTPUAddresses requires a TPU target")
    selected_target = resolve_tpu_target(target=target)
    for global_var, function in mod.functions.items():
        if not isinstance(function, tir.PrimFunc):
            continue
        function_target = function.attrs.get("target") if function.attrs else None
        if function_target is None:
            raise ValueError("AssignTPUAddresses requires every PrimFunc to be bound to "
                             f"the selected TPU target; {global_var.name_hint!r} is unbound")
        if function_target.kind.name != "tpu":
            raise ValueError("AssignTPUAddresses cannot process a PrimFunc bound to "
                             f"{function_target.kind.name!r}: {global_var.name_hint!r}")
        function_selection = resolve_tpu_target(target=function_target)
        if function_selection != selected_target:
            raise ValueError("AssignTPUAddresses target identity mismatch for PrimFunc "
                             f"{global_var.name_hint!r}: function={function_selection}, "
                             f"requested={selected_target}")
    mod = IRModule(
        {
            gv: _uniquify_tpu_local_names(f) if isinstance(f, tir.PrimFunc) else f
            for gv, f in mod.functions.items()
        },
        attrs=mod.attrs)
    return tilelang.transform.AddressAssign()(mod)
