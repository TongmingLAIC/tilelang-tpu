# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import contextlib
import io
import itertools
import random

import pytest

import tilelang
from tilelang import tvm
import tilelang.language as T
from tilelang.engine.phase import AssignTPUAddresses, LowerAndLegalize, OptimizeForTarget

BANK_SIZE = 16 * 1024
LMEM_ADDRESS_ATTR_PREFIX = "tilelang.tpu.lmem.address."


def _assigned_attrs(func, *, target_spec="tpu -mcpu=bm1690 -tpu-programming-model=tpukernel"):
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    target = tvm.target.Target(target_spec)
    with contextlib.redirect_stdout(io.StringIO()):
        mod = LowerAndLegalize(mod, target)
        mod = OptimizeForTarget(mod, target)
        mod = AssignTPUAddresses(mod, target)
    return mod["main"].attrs


def _addr(attrs, name):
    key = name if name.startswith(LMEM_ADDRESS_ATTR_PREFIX) else (LMEM_ADDRESS_ATTR_PREFIX + name)
    return int(attrs[key])


def _attr_keys(attrs):
    return [str(key) for key in attrs._dict().keys()]


def _single_attr_with_prefix(attrs, prefix):
    keys = [key for key in _attr_keys(attrs) if key.startswith(LMEM_ADDRESS_ATTR_PREFIX + prefix)]
    assert len(keys) == 1, keys
    return keys[0]


def _raw_local_allocations(*, data_names, buffer_names=None):
    """Build nested Allocate/DeclBuffer nodes without parser name uniquing."""
    if buffer_names is None:
        buffer_names = data_names
    assert len(data_names) == len(buffer_names)
    body = tvm.tir.Evaluate(0)
    for data_name, buffer_name in reversed(list(zip(data_names, buffer_names))):  # noqa: B905
        pointer_type = tvm.ir.PointerType(tvm.ir.PrimType("float32"), "shared")
        data = tvm.tir.Var(data_name, pointer_type)
        buffer = tvm.tir.decl_buffer((32,), "float32", name=buffer_name, data=data, scope="shared")
        body = tvm.tir.Allocate(data, "float32", [32], tvm.tir.IntImm("bool", 1),
                                tvm.tir.DeclBuffer(buffer, body))
    return tvm.tir.PrimFunc([], body).with_attr("global_symbol", "main")


def _pressure_module(sizes, access_groups, symbol="main", access_repetitions=None):
    """Build opaque accesses so AddressAssign conservatively infers conflicts."""
    buffers = {}
    for name, size in sizes.items():
        pointer_type = tvm.ir.PointerType(tvm.ir.PrimType("float32"), "shared")
        data = tvm.tir.Var(name, pointer_type)
        buffers[name] = tvm.tir.decl_buffer(
            (size // 4,), "float32", name=name, data=data, scope="shared")
    repetitions = access_repetitions or [1] * len(access_groups)
    assert len(repetitions) == len(access_groups)
    statements = []
    for index, (group, repeat) in enumerate(zip(access_groups, repetitions)):
        statement = tvm.tir.Evaluate(
            tvm.tir.call_extern("handle", "research.conflict",
                                *(buffers[name].data for name in group)))
        if repeat != 1:
            statement = tvm.tir.For(
                tvm.tir.Var(f"repeat_{index}", "int32"), 0, repeat,
                tvm.tir.ForKind.SERIAL, statement)
        statements.append(statement)
    body = statements[0] if len(statements) == 1 else tvm.tir.SeqStmt(statements)
    for buffer in reversed(list(buffers.values())):
        body = tvm.tir.Allocate(
            buffer.data, buffer.dtype, list(buffer.shape), tvm.tir.IntImm("bool", 1),
            tvm.tir.DeclBuffer(buffer, body))
    function = tvm.tir.PrimFunc([], body).with_attr("global_symbol", symbol)
    target = tvm.target.Target("tpu -mcpu=bm1690e -tpu-programming-model=rv")
    return tvm.tir.transform.BindTarget(target)(tvm.IRModule({symbol: function})), sizes


def _near_capacity_conflict_module():
    """A deterministic complete conflict graph with awkward bank-sized gaps."""
    sizes = {
        "b0": 10240,
        "b1": 22528,
        "b2": 10240,
        "b3": 34816,
        "b4": 34816,
        "b5": 34816,
        "b6": 30720,
        "b7": 18432,
    }
    return _pressure_module(sizes, list(itertools.combinations(sizes, 2)))


def _greedy_failure_recovery_module():
    seed = 4030
    rng = random.Random(seed)
    choices = (6144, 10240, 14336, 18432, 22528, 26624, 30720, 34816, 38912)
    buffer_count = rng.randint(7, 12)
    sizes = {f"b{index}": rng.choice(choices) for index in range(buffer_count)}
    names = tuple(sizes)
    accesses = [(name,) for name in names]
    for _ in range(rng.randint(buffer_count * 2, buffer_count * 4)):
        width = rng.randint(3, min(7, buffer_count))
        accesses.append(tuple(rng.sample(names, width)))
    return _pressure_module(sizes, accesses)


def _assign_with_policy(module, policy=None, **extra_config):
    config = {} if policy is None else {"tl.AddressAssignPolicy": policy}
    config.update(extra_config)
    with tvm.transform.PassContext(config=config):
        return tilelang.transform.AddressAssign()(module)["main"]


def test_parallel_command_group_extends_sequential_tir_uses_to_concurrent_lives():
    """Two commands in one group may overlap even if their calls do not."""

    def make_module(with_group):
        buffers = {}
        for name in ("first", "second"):
            data = tvm.tir.Var(name, tvm.ir.PointerType(tvm.ir.PrimType("float32"), "shared"))
            buffers[name] = tvm.tir.decl_buffer((32,), "float32", name=name,
                                                data=data, scope="shared")
        statements = []
        if with_group:
            statements.append(tvm.tir.Evaluate(tvm.tir.call_extern(
                "handle", "tl.tpukernel.parallel_start")))
        for name in buffers:
            statements.append(tvm.tir.Evaluate(tvm.tir.call_extern(
                "handle", "research.touch", buffers[name].data)))
        if with_group:
            statements.append(tvm.tir.Evaluate(tvm.tir.call_extern(
                "handle", "tl.tpukernel.parallel_end")))
        body = tvm.tir.SeqStmt(statements)
        for buffer in reversed(tuple(buffers.values())):
            body = tvm.tir.Allocate(
                buffer.data, buffer.dtype, list(buffer.shape),
                tvm.tir.IntImm("bool", 1), tvm.tir.DeclBuffer(buffer, body))
        target = tvm.target.Target("tpu -mcpu=bm1690e -tpu-programming-model=tpukernel")
        function = tvm.tir.PrimFunc([], body).with_attr("global_symbol", "main")
        return tvm.tir.transform.BindTarget(target)(tvm.IRModule({"main": function}))

    config = {"tl.AddressAssignEmitProblem": True}
    serial = _assign_with_policy(make_module(False), "compact", **config)
    parallel = _assign_with_policy(make_module(True), "compact", **config)
    assert _addr(serial.attrs, "first") == _addr(serial.attrs, "second")
    assert _addr(parallel.attrs, "first") != _addr(parallel.attrs, "second")
    prefix = "tilelang.tpu.lmem.problem.buffer."
    assert int(parallel.attrs[prefix + "0.start"]) == int(parallel.attrs[prefix + "1.start"])
    assert int(parallel.attrs[prefix + "0.end"]) == int(parallel.attrs[prefix + "1.end"])


def test_ppl_gemm_readwrite_accumulator_is_separated_from_both_inputs():

    @T.prim_func
    def main():
        with T.Kernel(1, is_cpu=True) as _:
            a_shared = T.alloc_shared((64, 1024), "float16")
            b_shared = T.alloc_shared((1024, 64), "float16")
            c_shared = T.alloc_shared((64, 64), "float32")

            T.ppl_fill(c_shared, T.float32(0.0))
            T.ppl_gemm(a_shared, b_shared, c_shared, accumulate=True)

    attrs = _assigned_attrs(main)
    a_addr = _addr(attrs, "a_shared")
    b_addr = _addr(attrs, "b_shared")
    c_addr = _addr(attrs, "c_shared")

    # ppl_gemm implements C += A @ B.  All three operands are therefore read
    # during the GEMM and must not contend for the same local-memory bank.
    assert len({
        a_addr // BANK_SIZE,
        b_addr // BANK_SIZE,
        c_addr // BANK_SIZE,
    }) == 3


def test_elementwise_reads_are_bank_separated_while_outputs_remain_flexible():

    @T.prim_func
    def main():
        with T.Kernel(1, is_cpu=True) as _:
            src0 = T.alloc_shared((64, 1024), "float32")
            src1 = T.alloc_shared((64, 1024), "float32")
            dst = T.alloc_shared((64, 1024), "float32")

            T.ppl_add(dst, src0, src1)

    attrs = _assigned_attrs(main)
    src0_addr = _addr(attrs, "src0")
    src1_addr = _addr(attrs, "src1")
    dst_addr = _addr(attrs, "dst")

    assert src0_addr // BANK_SIZE != src1_addr // BANK_SIZE
    assert dst_addr // BANK_SIZE == src0_addr // BANK_SIZE


def test_dq2_separates_packed_weight_and_scale_reads_without_a_dst_conflict():

    @T.prim_func
    def main():
        with T.Kernel(1, is_cpu=True):
            out = T.alloc_shared((64, 64), "float16")
            packed = T.alloc_shared((64, 32), "uint8")
            scale = T.alloc_shared((64, 2), "uint32")
            T.ppl_dq2(out, packed, scale, 32)

    attrs = _assigned_attrs(main)
    read_banks = {_addr(attrs, "packed") // BANK_SIZE,
                  _addr(attrs, "scale") // BANK_SIZE}
    assert len(read_banks) == 2
    assert _addr(attrs, "out") // BANK_SIZE in read_banks


def test_rv_scatter_separates_local_source_and_index_reads():

    @T.prim_func
    def main(cache: T.Tensor((64, 128), "float16")):
        with T.Kernel(1, is_cpu=True):
            source = T.alloc_shared((64, 128), "float16")
            index = T.alloc_shared((64, 1), "uint32")
            T.ppl_scatter(cache, source, index)

    attrs = _assigned_attrs(main, target_spec="tpu -mcpu=bm1690e -tpu-programming-model=rv")
    assert _addr(attrs, "source") // BANK_SIZE != _addr(attrs, "index") // BANK_SIZE


def test_scatter_add_preserves_distinct_internal_row_tiles():

    @T.prim_func
    def main(dst: T.Tensor((4, 32), "float16"),
             src: T.Tensor((2, 32), "float16"),
             indices: T.Tensor((2, 1), "int32")):
        with T.Kernel(1, is_cpu=True):
            T.ppl_scatter_add(dst, src, indices)

    attrs = _assigned_attrs(main)
    accumulator = _single_attr_with_prefix(attrs, "accumulator")
    source_row = _single_attr_with_prefix(attrs, "source_row")
    assert _addr(attrs, accumulator) // BANK_SIZE != _addr(attrs, source_row) // BANK_SIZE


def test_reduce_tmp_is_separated_from_input_bank():

    @T.prim_func
    def main():
        with T.Kernel(1, is_cpu=True) as _:
            inp = T.alloc_shared((64, 1024), "float32")
            out = T.alloc_shared((64, 1), "float32")

            T.ppl_reduce_sum(inp, out, dim=1)

    attrs = _assigned_attrs(main)
    tmp_name = _single_attr_with_prefix(attrs, "tmp_buffer_sum")
    inp_addr = _addr(attrs, "inp")
    tmp_addr = _addr(attrs, tmp_name)

    assert inp_addr // BANK_SIZE != tmp_addr // BANK_SIZE


def test_exp_composite_operands_are_conservative_bank_clique():

    @T.prim_func
    def main():
        with T.Kernel(1, is_cpu=True) as _:
            out = T.alloc_shared((64, 1024), "float32")
            work0 = T.alloc_shared((64, 1024), "float32")
            work1 = T.alloc_shared((64, 1024), "float32")
            coeff = T.alloc_shared((64, 32), "float32")
            T.ppl_exp(out, work0, work1, coeff)

    attrs = _assigned_attrs(main)
    banks = {
        _addr(attrs, "out") // BANK_SIZE,
        _addr(attrs, "work0") // BANK_SIZE,
        _addr(attrs, "work1") // BANK_SIZE,
        _addr(attrs, "coeff") // BANK_SIZE,
    }

    assert len(banks) == 4


def test_address_assignment_rejects_non_tpu_target():

    @T.prim_func
    def main():
        T.evaluate(0)

    mod = tvm.IRModule({"main": main})
    target = tvm.target.Target("c")
    mod = tvm.tir.transform.BindTarget(target)(mod)

    with pytest.raises(ValueError, match="requires a TPU target"):
        AssignTPUAddresses(mod, target)

    with pytest.raises(tvm.error.TVMError, match="requires a TPU Target"):
        tilelang.transform.AddressAssign()(mod)


@pytest.mark.parametrize("target_spec, message, native_message", [
    ("tpu", "explicit physical chip", "supported target chip"),
    (
        "tpu -mcpu=bm1690e",
        "explicit programming model",
        "requires a normalized target",
    ),
    (
        "tpu -mcpu=bm1690 -tpu-programming-model=rv",
        "does not support programming model",
        "does not support",
    ),
])
def test_address_assignment_requires_a_complete_supported_tpu_target(target_spec, message,
                                                                     native_message):

    @T.prim_func
    def main():
        T.evaluate(0)

    target = tvm.target.Target(target_spec)
    mod = tvm.tir.transform.BindTarget(target)(tvm.IRModule({"main": main}))

    with pytest.raises(ValueError, match=message):
        AssignTPUAddresses(mod, target)

    # The native transform is independently fail-closed for callers that
    # bypass tilelang.engine.phase.
    with pytest.raises(tvm.error.TVMError, match=native_message):
        tilelang.transform.AddressAssign()(mod)


def test_lmem_address_attributes_are_namespaced_and_follow_the_data_var():
    # Buffer.name and Buffer.data.name are independent in legal TIR.  Codegen
    # owns the allocation data Var, so its identity must drive the hand-off.
    function = _raw_local_allocations(
        data_names=("target",), buffer_names=("unrelated_buffer_name",))
    target = tvm.target.Target("tpu -mcpu=bm1690 -tpu-programming-model=tpukernel")
    mod = tvm.tir.transform.BindTarget(target)(tvm.IRModule({"main": function}))
    attrs = tilelang.transform.AddressAssign()(mod)["main"].attrs

    assert attrs["target"].kind.name == "tpu"
    assert _addr(attrs, "target") == 0
    assert "unrelated_buffer_name" not in _attr_keys(attrs)


def test_address_assignment_rejects_duplicate_allocation_data_names():
    function = _raw_local_allocations(data_names=("duplicate", "duplicate"))
    target = tvm.target.Target("tpu -mcpu=bm1690 -tpu-programming-model=tpukernel")
    mod = tvm.tir.transform.BindTarget(target)(tvm.IRModule({"main": function}))

    with pytest.raises(
            tvm.error.TVMError, match="unique local allocation data-variable names.*duplicate"):
        tilelang.transform.AddressAssign()(mod)


def test_portfolio_address_policy_is_fail_safe_and_improves_the_pressure_case():
    module, sizes = _near_capacity_conflict_module()
    default = _assign_with_policy(module)
    greedy = _assign_with_policy(module, "greedy")
    beam = _assign_with_policy(module, "beam")
    portfolio = _assign_with_policy(module, "portfolio")
    agent = _assign_with_policy(module, "agent")

    names = tuple(sizes)
    assert [_addr(default.attrs, name) for name in names] == [
        _addr(greedy.attrs, name) for name in names
    ]
    assert str(default.attrs["tilelang.tpu.lmem.statistics.policy"]) == "greedy"
    assert str(beam.attrs["tilelang.tpu.lmem.statistics.policy"]) == "beam-search"
    assert str(portfolio.attrs[
        "tilelang.tpu.lmem.statistics.policy"]).startswith("portfolio-search-")
    assert str(agent.attrs["tilelang.tpu.lmem.statistics.policy"]).startswith(
        "agent-search-")
    greedy_cost = int(greedy.attrs["tilelang.tpu.lmem.statistics.bank_conflicts"])
    agent_cost = int(agent.attrs["tilelang.tpu.lmem.statistics.bank_conflicts"])
    greedy_peak = int(greedy.attrs["tilelang.tpu.lmem.statistics.peak_bytes"])
    agent_peak = int(agent.attrs["tilelang.tpu.lmem.statistics.peak_bytes"])
    assert agent_cost <= greedy_cost
    assert agent_peak < greedy_peak
    assert [_addr(portfolio.attrs, name) for name in names] == [
        _addr(agent.attrs, name) for name in names
    ]

    intervals = sorted((_addr(agent.attrs, name), _addr(agent.attrs, name) + sizes[name])
                       for name in names)
    assert all(lhs_end <= rhs_start
               for (_, lhs_end), (rhs_start, _) in zip(intervals, intervals[1:]))


def test_address_assignment_rejects_unknown_agent_policy():
    module, _ = _near_capacity_conflict_module()
    with pytest.raises(
            tvm.error.TVMError,
            match="policy must be 'greedy', 'compact', 'beam', 'portfolio'"):
        _assign_with_policy(module, "unvalidated")


def test_address_assignment_exports_and_replays_a_verified_problem():
    module, sizes = _pressure_module(
        {"a": 1024, "b": 2048}, [("a", "b")])
    exported = _assign_with_policy(
        module, "greedy", **{"tl.AddressAssignEmitProblem": True})
    prefix = "tilelang.tpu.lmem.problem."

    assert int(exported.attrs[prefix + "buffer_count"]) == 2
    assert int(exported.attrs[prefix + "edge_count"]) == 1
    names = [str(exported.attrs[prefix + f"buffer.{index}.name"])
             for index in range(2)]
    assert names == ["a", "b"]
    assert [int(exported.attrs[prefix + f"buffer.{index}.size"])
            for index in range(2)] == [sizes[name] for name in names]
    assert int(exported.attrs[prefix + "edge.0.lhs"]) == 0
    assert int(exported.attrs[prefix + "edge.0.rhs"]) == 1
    assert int(exported.attrs[prefix + "edge.0.weight"]) == 1

    replayed = _assign_with_policy(
        module, "replay", **{"tl.AddressAssignReplayAddresses": "16384,0"})
    assert str(replayed.attrs["tilelang.tpu.lmem.statistics.policy"]) == "replay"
    assert _addr(replayed.attrs, "a") == 16384
    assert _addr(replayed.attrs, "b") == 0


    assert int(replayed.attrs[
        "tilelang.tpu.lmem.statistics.explored_states"]) == 0

    with pytest.raises(tvm.error.TVMError, match="overlapping live buffers"):
        _assign_with_policy(
            module, "replay", **{"tl.AddressAssignReplayAddresses": "0,0"})

    with pytest.raises(tvm.error.TVMError, match="must be non-negative"):
        _assign_with_policy(
            module, "replay", **{"tl.AddressAssignReplayAddresses": "-64,0"})
    with pytest.raises(tvm.error.TVMError, match="64-byte aligned"):
        _assign_with_policy(
            module, "replay", **{"tl.AddressAssignReplayAddresses": "1,0"})
    with pytest.raises(tvm.error.TVMError, match="exceeds LMEM capacity"):
        _assign_with_policy(
            module, "replay", **{
                "tl.AddressAssignReplayAddresses":
                    "9223372036854775744,0",
            })


def test_analysis_only_exports_before_allocation_failure():
    module, _ = _pressure_module(
        {"a": 131072, "b": 131136}, [("a", "b")])
    with pytest.raises(tvm.error.TVMError, match="allocation failed"):
        _assign_with_policy(module, "greedy")

    analyzed = _assign_with_policy(
        module,
        "greedy",
        **{
            "tl.AddressAssignEmitProblem": True,
            "tl.AddressAssignAnalysisOnly": True,
        },
    )
    prefix = "tilelang.tpu.lmem.problem."
    assert int(analyzed.attrs[prefix + "buffer_count"]) == 2
    assert [int(analyzed.attrs[prefix + f"buffer.{index}.size"])
            for index in range(2)] == [131072, 131136]
    assert not any(key.startswith(LMEM_ADDRESS_ATTR_PREFIX)
                   for key in _attr_keys(analyzed.attrs))


def test_agent_address_policy_recovers_a_greedy_allocation_failure():
    module, _ = _greedy_failure_recovery_module()
    with pytest.raises(tvm.error.TVMError, match="local memory allocation failed"):
        _assign_with_policy(module, "greedy")
    compact = _assign_with_policy(module, "compact")
    beam = _assign_with_policy(module, "beam")
    portfolio = _assign_with_policy(module, "portfolio")
    agent = _assign_with_policy(module, "agent")
    assert str(compact.attrs["tilelang.tpu.lmem.statistics.policy"]) == (
        "compact-search")
    assert str(beam.attrs["tilelang.tpu.lmem.statistics.policy"]) == "beam-search"
    assert str(portfolio.attrs["tilelang.tpu.lmem.statistics.policy"]) == (
        "portfolio-search-pressure")
    assert str(agent.attrs["tilelang.tpu.lmem.statistics.policy"]) == (
        "agent-search-pressure")
    # ``agent`` remains a compatibility spelling for the auditable portfolio.
    names = tuple(f"b{index}" for index in range(10))
    assert [_addr(portfolio.attrs, name) for name in names] == [
        _addr(agent.attrs, name) for name in names
    ]
    assert int(compact.attrs[
        "tilelang.tpu.lmem.statistics.live_bytes_lower_bound"]) == 208896
    for result in (compact, beam, portfolio, agent):
        peak = int(result.attrs["tilelang.tpu.lmem.statistics.peak_bytes"])
        lower_bound = int(result.attrs[
            "tilelang.tpu.lmem.statistics.live_bytes_lower_bound"])
        slack = int(result.attrs[
            "tilelang.tpu.lmem.statistics.peak_slack_bytes"])
        assert peak >= lower_bound
        assert slack == peak - lower_bound
    assert int(agent.attrs["tilelang.tpu.lmem.statistics.peak_bytes"]) <= 16 * 16 * 1024


def test_full_portfolio_verifies_all_branches_for_a_low_pressure_program():
    module, _ = _pressure_module(
        {"a": 1024, "b": 1024}, [("a", "b")])
    greedy = _assign_with_policy(module, "greedy")
    portfolio = _assign_with_policy(module, "portfolio")
    agent = _assign_with_policy(module, "agent")

    assert str(portfolio.attrs[
        "tilelang.tpu.lmem.statistics.policy"]) == "portfolio-incumbent"
    assert str(agent.attrs[
        "tilelang.tpu.lmem.statistics.policy"]) == "agent-incumbent"
    assert _addr(portfolio.attrs, "a") == _addr(greedy.attrs, "a")
    assert _addr(portfolio.attrs, "b") == _addr(greedy.attrs, "b")
    assert _addr(agent.attrs, "a") == _addr(greedy.attrs, "a")
    assert _addr(agent.attrs, "b") == _addr(greedy.attrs, "b")
    assert int(portfolio.attrs[
        "tilelang.tpu.lmem.statistics.explored_states"]) > int(greedy.attrs[
            "tilelang.tpu.lmem.statistics.explored_states"])


@pytest.mark.parametrize("repeat", [0, 7])
def test_static_loop_frequency_weights_bank_conflict_cost(repeat):
    sizes = {
        "b0": 10240,
        "b1": 22528,
        "b2": 10240,
        "b3": 34816,
        "b4": 34816,
        "b5": 34816,
        "b6": 30720,
        "b7": 18432,
    }
    accesses = list(itertools.combinations(sizes, 2))
    module, _ = _pressure_module(
        sizes, accesses, access_repetitions=[repeat] * len(accesses))
    greedy = _assign_with_policy(module, "greedy")
    unweighted = int(greedy.attrs["tilelang.tpu.lmem.statistics.bank_conflicts"])
    weighted = int(
        greedy.attrs["tilelang.tpu.lmem.statistics.weighted_bank_conflicts"])

    assert unweighted > 0
    assert weighted == unweighted * repeat


def test_copy_aliases_use_canonical_allocations_and_write_effects():
    """Buffer presentation aliases must not disappear from liveness.

    RegionOp may carry a Buffer object distinct from the DeclBuffer while both
    share the same data Var and exact descriptor.  AddressAssign owns storage
    by that Var identity.  The copy source and destination must therefore be
    live together (no overlapping addresses), while a read/write pair need not
    be separated into different banks.
    """
    pointer_type = tvm.ir.PointerType(tvm.ir.PrimType("float32"), "shared")
    a_data = tvm.tir.Var("a", pointer_type)
    b_data = tvm.tir.Var("b", pointer_type)
    a_decl = tvm.tir.decl_buffer((32,), "float32", name="a_decl", data=a_data, scope="shared")
    b_decl = tvm.tir.decl_buffer((32,), "float32", name="b_decl", data=b_data, scope="shared")
    a_alias = tvm.tir.decl_buffer((32,), "float32", name="a_alias", data=a_data, scope="shared")
    b_alias = tvm.tir.decl_buffer((32,), "float32", name="b_alias", data=b_data, scope="shared")

    def region(buffer, access_mask):
        return tvm.tir.call_intrin("handle", tvm.ir.Op.get("tl.region"),
                                   tvm.tir.BufferLoad(buffer, [0]), access_mask, 32)

    copy = tvm.tir.call_extern("handle", "tl.tpu.copy", region(a_alias, 1), region(b_alias, 2))
    body = tvm.tir.Allocate(
        a_data, "float32", [32], tvm.tir.IntImm("bool", 1),
        tvm.tir.DeclBuffer(
            a_decl,
            tvm.tir.Allocate(b_data, "float32", [32], tvm.tir.IntImm("bool", 1),
                             tvm.tir.DeclBuffer(b_decl, tvm.tir.Evaluate(copy)))))
    function = tvm.tir.PrimFunc([], body).with_attr("global_symbol", "main")
    target = tvm.target.Target("tpu -mcpu=bm1690 -tpu-programming-model=tpukernel")
    mod = tvm.tir.transform.BindTarget(target)(tvm.IRModule({"main": function}))

    attrs = tilelang.transform.AddressAssign()(mod)["main"].attrs
    a_addr = _addr(attrs, "a")
    b_addr = _addr(attrs, "b")
    assert abs(a_addr - b_addr) >= 32 * 4
    assert a_addr // BANK_SIZE == b_addr // BANK_SIZE


def _liveness_buffer(name):
    return tvm.tir.decl_buffer((32,), "float32", name=name, scope="shared")


def _liveness_allocate(buffer, body):
    return tvm.tir.Allocate(buffer.data, buffer.dtype, list(buffer.shape),
                            tvm.tir.IntImm("bool", 1), tvm.tir.DeclBuffer(buffer, body))


def _liveness_fill(buffer):
    return tvm.tir.Evaluate(tvm.tir.call_extern("handle", "tl.tpu.fill", buffer.data, 1.0))


def _liveness_copy(src, dst):
    return tvm.tir.Evaluate(tvm.tir.call_extern("handle", "tl.tpu.copy", src.data, dst.data))


def _liveness_attrs(body, target_spec):
    function = tvm.tir.PrimFunc(tvm.tir.analysis.undefined_vars(body),
                                body).with_attr("global_symbol", "main")
    target = tvm.target.Target(target_spec)
    mod = tvm.tir.transform.BindTarget(target)(tvm.IRModule({"main": function}))
    return tilelang.transform.AddressAssign()(mod)["main"].attrs


TPU_ADDRESS_TARGETS = [
    "tpu -mcpu=bm1690 -tpu-programming-model=tpukernel",
    "tpu -mcpu=bm1690e -tpu-programming-model=tpukernel",
    "tpu -mcpu=bm1690e -tpu-programming-model=rv",
]


@pytest.mark.parametrize("target_spec", TPU_ADDRESS_TARGETS)
@pytest.mark.parametrize("loop_kind", ["for", "symbolic", "while", "nested"])
@pytest.mark.parametrize("initialize_before_loop", [False, True])
def test_loop_external_allocations_cannot_alias_later_scratch(target_spec, loop_kind,
                                                              initialize_before_loop):
    weight, scratch, out = [_liveness_buffer(name) for name in ("weight", "scratch", "out")]
    read = _liveness_copy(weight, out)
    if loop_kind == "nested":
        # The read must reach the outer loop's end, beyond the inner loop.
        read = tvm.tir.For(tvm.tir.Var("j", "int32"), 0, 2, tvm.tir.ForKind.SERIAL, read)
    body = tvm.tir.SeqStmt([read, _liveness_fill(scratch), _liveness_copy(scratch, out)])
    if loop_kind == "while":
        body = tvm.tir.While(tvm.tir.Var("keep_running", "bool"), body)
    else:
        extent = tvm.tir.Var("n", "int32") if loop_kind == "symbolic" else 2
        body = tvm.tir.For(tvm.tir.Var("i", "int32"), 0, extent, tvm.tir.ForKind.SERIAL, body)
    if initialize_before_loop:
        body = tvm.tir.SeqStmt([_liveness_fill(weight), body])
    for buffer in (out, scratch, weight):
        body = _liveness_allocate(buffer, body)
    attrs = _liveness_attrs(body, target_spec)

    # Formerly weight=0, scratch=0, out=128: the later write destroyed the
    # next iteration's weight. A first use inside the loop needs protection
    # too; allocation scope, rather than a preceding use, determines this.
    assert [_addr(attrs, name) for name in ("weight", "scratch", "out")] == [0, 128, 256]


@pytest.mark.parametrize("loop_extent", [None, 0, 1])
def test_no_backedge_preserves_sequential_address_reuse(loop_extent):
    weight, scratch, out = [_liveness_buffer(name) for name in ("weight", "scratch", "out")]
    body = tvm.tir.SeqStmt(
        [_liveness_copy(weight, out),
         _liveness_fill(scratch),
         _liveness_copy(scratch, out)])
    if loop_extent is not None:
        body = tvm.tir.For(tvm.tir.Var("i", "int32"), 0, loop_extent, tvm.tir.ForKind.SERIAL, body)
    body = tvm.tir.SeqStmt([_liveness_fill(weight), body])
    for buffer in (out, scratch, weight):
        body = _liveness_allocate(buffer, body)
    attrs = _liveness_attrs(body, TPU_ADDRESS_TARGETS[0])

    # Static zero/one-trip loops cannot carry a value over a backedge. The
    # pass still visits their body, preserving its existing straight-line
    # allocation policy; dead-loop elimination belongs to other passes.
    assert [_addr(attrs, name) for name in ("weight", "scratch", "out")] == [0, 0, 128]


def test_loop_local_allocations_keep_sequential_address_reuse():
    weight, scratch, out = [_liveness_buffer(name) for name in ("weight", "scratch", "out")]
    body = tvm.tir.SeqStmt([
        _liveness_fill(weight),
        _liveness_copy(weight, out),
        _liveness_fill(scratch),
        _liveness_copy(scratch, out)
    ])
    for buffer in (out, scratch, weight):
        body = _liveness_allocate(buffer, body)
    body = tvm.tir.For(tvm.tir.Var("i", "int32"), 0, 2, tvm.tir.ForKind.SERIAL, body)
    attrs = _liveness_attrs(body, TPU_ADDRESS_TARGETS[0])

    # Each iteration owns a fresh weight allocation and initializes it before
    # use; the later scratch write cannot destroy a loop-carried value.
    assert [_addr(attrs, name) for name in ("weight", "scratch", "out")] == [0, 0, 128]


@pytest.mark.parametrize("nested", [False, True])
def test_loop_local_scratch_can_reuse_addresses_without_clobbering_external_weight(nested):
    weight, scratch0, scratch1, out = [
        _liveness_buffer(name) for name in ("weight", "scratch0", "scratch1", "out")
    ]
    body = tvm.tir.SeqStmt([
        _liveness_copy(weight, out),
        _liveness_allocate(
            scratch0, tvm.tir.SeqStmt([_liveness_fill(scratch0),
                                       _liveness_copy(scratch0, out)])),
        _liveness_allocate(
            scratch1, tvm.tir.SeqStmt([_liveness_fill(scratch1),
                                       _liveness_copy(scratch1, out)])),
    ])
    body = tvm.tir.For(tvm.tir.Var("i", "int32"), 0, 2, tvm.tir.ForKind.SERIAL, body)
    body = _liveness_allocate(weight, tvm.tir.SeqStmt([_liveness_fill(weight), body]))
    if nested:
        # Weight is local to the outer loop, external to the inner one;
        # scratch0/1 are local to both and retain sequential reuse.
        body = tvm.tir.For(tvm.tir.Var("j", "int32"), 0, 2, tvm.tir.ForKind.SERIAL, body)
    attrs = _liveness_attrs(_liveness_allocate(out, body), TPU_ADDRESS_TARGETS[0])

    expected = [128, 0, 256, 256] if nested else [0, 128, 256, 256]
    assert [_addr(attrs, name) for name in ("weight", "out", "scratch0", "scratch1")] == expected


def test_while_condition_local_read_survives_body_scratch_writes():
    condition, scratch, out = [_liveness_buffer(name) for name in ("condition", "scratch", "out")]
    body = tvm.tir.While(
        tvm.tir.BufferLoad(condition, [0]) > 0,
        tvm.tir.SeqStmt([_liveness_fill(scratch),
                         _liveness_copy(scratch, out)]))
    for buffer in (out, scratch, condition):
        body = _liveness_allocate(buffer, body)
    attrs = _liveness_attrs(body, TPU_ADDRESS_TARGETS[0])

    assert [_addr(attrs, name) for name in ("condition", "scratch", "out")] == [0, 128, 256]


if __name__ == "__main__":
    test_ppl_gemm_readwrite_accumulator_is_separated_from_both_inputs()
    test_elementwise_reads_are_bank_separated_while_outputs_remain_flexible()
    test_reduce_tmp_is_separated_from_input_bank()
    test_exp_composite_operands_are_conservative_bank_clique()
    test_address_assignment_rejects_non_tpu_target()


def test_dq2_scratch_and_operands_have_disjoint_live_storage():

    @T.prim_func
    def main():
        with T.Kernel(1, is_cpu=True):
            packed = T.alloc_shared((65, 32), "uint32")
            quant = T.alloc_shared((65, 2), "uint32")
            result = T.alloc_shared((65, 256), "float16")
            T.ppl_dq2(result, packed, quant, 128)

    attrs = _assigned_attrs(main)
    scratch = _single_attr_with_prefix(attrs, "scratch")
    # Physical bytes per lane, including the second C block and EU padding.
    intervals = [(_addr(attrs, name), size) for name, size in
                 (("packed", 256), ("quant", 128), ("result", 1024), (scratch, 128))]
    for i, (left, left_size) in enumerate(intervals):
        for right, right_size in intervals[i + 1:]:
            assert left + left_size <= right or right + right_size <= left
