# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Contracts and independently checked numerical cases for RV indexed ops.

Pytest is compile-only. Numerical workers must run in private CModel or under
an external serialized PCIe watchdog; this file does not initialize a device
when imported or collected.

Run one numerical case in a fresh process with --case INDEX --runtime
cmodel|pcie. CASES defines the complete matrix. For CModel, wrap each worker
with ``python -m tilelang.jit.tpu_cmodel --output-dir ABSOLUTE_NEW_DIR --``.
Use the same BM1690E RV target and four runtime cores in both modes. Complete
the CModel matrix before board execution, and stop on any failure or timeout.
"""

import argparse
import json
import math
from pathlib import Path

import pytest
import torch
import tilelang
import tilelang.language as T

TARGET = "tpu -mcpu=bm1690e -tpu-programming-model=rv"


def gather_kernel(dtype="float32", rows=17, count=5, width=7):
    @T.prim_func
    def kernel(
        P: T.Tensor((rows, width), dtype),
        I: T.Tensor((count, 1), "uint32"),
        O: T.Tensor((count, width), dtype),
    ):
        T.func_attr({"global_symbol": f"gather_{dtype}_{rows}_{count}_{width}"})
        with T.Kernel(1, is_cpu=True):
            T.ppl_gather(O, P, I, rows)

    return kernel


def scatter_kernel(dtype="float32", index_dtype="int32", rows=7, count=9, width=5):
    @T.prim_func
    def kernel(
        D: T.Tensor((rows, width), dtype),
        S: T.Tensor((count, width), dtype),
        I: T.Tensor((count, 1), index_dtype),
    ):
        T.func_attr(
            {"global_symbol": f"scatter_{dtype}_{index_dtype}_{rows}_{count}_{width}"}
        )
        with T.Kernel(1, is_cpu=True):
            T.ppl_scatter_add(D, S, I, dim=0)

    return kernel


def topk_kernel(
    dtype="float32",
    length=33,
    k=7,
    descending=True,
    rows=1,
    local=False,
    full_output=False,
    reuse=False,
    index_dtype="int32",
):
    width = max(32, 1 << (length - 1).bit_length())
    output_width = length if full_output else k
    if local:

        @T.prim_func
        def kernel(
            X: T.Tensor((rows, length), dtype),
            V: T.Tensor((rows, output_width), dtype),
            I: T.Tensor((rows, output_width), index_dtype),
        ):
            with T.Kernel(1, is_cpu=True):
                x = T.alloc_shared((rows, length), dtype)
                v = T.alloc_shared((rows, output_width), dtype)
                i = T.alloc_shared((rows, output_width), index_dtype)
                w0 = T.alloc_shared((rows, width), "uint32")
                w1 = T.alloc_shared((rows, width), "uint32")
                w2 = T.alloc_shared((rows, width), "uint32")
                w3 = T.alloc_shared((rows, width), "uint32")
                w4 = T.alloc_shared((rows, width), "uint32")
                w5 = T.alloc_shared((rows, width), "uint32")
                w6 = T.alloc_shared((rows, width), "uint32")
                w7 = T.alloc_shared((rows, width), "uint16")
                T.ppl_copy(X, x)
                T.ppl_copy(V, v)
                T.ppl_copy(I, i)
                T.ppl_topk(
                    v,
                    i,
                    x,
                    k,
                    descending,
                    length,
                    workspace=(w0, w1, w2, w3, w4, w5, w6, w7),
                )
                if reuse:
                    T.ppl_topk(
                        v,
                        i,
                        x,
                        k,
                        descending,
                        length,
                        workspace=(w0, w1, w2, w3, w4, w5, w6, w7),
                    )
                T.ppl_copy(v, V)
                T.ppl_copy(i, I)
    else:

        @T.prim_func
        def kernel(
            X: T.Tensor((length,), dtype),
            V: T.Tensor((k,), dtype),
            I: T.Tensor((k,), "int32"),
        ):
            T.func_attr({"global_symbol": f"topk_{dtype}_{length}_{k}"})
            with T.Kernel(1, is_cpu=True):
                w0 = T.alloc_shared((1, width), "uint32")
                w1 = T.alloc_shared((1, width), "uint32")
                w2 = T.alloc_shared((1, width), "uint32")
                w3 = T.alloc_shared((1, width), "uint32")
                w4 = T.alloc_shared((1, width), "uint32")
                w5 = T.alloc_shared((1, width), "uint32")
                w6 = T.alloc_shared((1, width), "uint32")
                w7 = T.alloc_shared((1, width), "uint16")
                T.ppl_topk(
                    V,
                    I,
                    X,
                    k,
                    descending,
                    length,
                    workspace=(w0, w1, w2, w3, w4, w5, w6, w7),
                )
                if reuse:
                    T.ppl_topk(
                        V,
                        I,
                        X,
                        k,
                        descending,
                        length,
                        workspace=(w0, w1, w2, w3, w4, w5, w6, w7),
                    )

    return kernel


CASES = []
for dtype in ("float16", "bfloat16", "float32", "uint32", "e4m3_float8", "e5m2_float8"):
    for count, width in ((1, 1), (1, 33), (5, 7), (65, 33)):
        CASES.append(dict(op="gather", dtype=dtype, count=count, width=width))
for dtype in ("float16", "bfloat16", "float32"):
    for index_dtype in ("int32", "uint32"):
        for width in (2, 3, 32, 65):
            CASES.append(
                dict(op="scatter", dtype=dtype, index_dtype=index_dtype, width=width)
            )
for dtype in ("float16", "bfloat16", "float32"):
    CASES.append(
        dict(op="scatter", dtype=dtype, index_dtype="uint32", width=2, adversarial=True)
    )
for dtype in ("float32", "int32", "uint32"):
    for length, k in ((1, 1), (17, 1), (17, 17), (33, 7), (65, 65), (257, 19)):
        for descending in (False, True):
            CASES.append(
                dict(op="topk", dtype=dtype, length=length, k=k, descending=descending)
            )
for rows, length, k in (
    (1, 17, 7),
    (63, 33, 7),
    (64, 33, 7),
    (65, 33, 7),
    (129, 17, 7),
):
    for descending in (False, True):
        CASES.append(
            dict(
                op="topk",
                dtype="float32",
                rows=rows,
                length=length,
                k=k,
                descending=descending,
                local=True,
                full_output=True,
                reuse=True,
                index_dtype="uint32",
            )
        )


CASES.append(dict(op="compose", dtype="float32"))


def _payload_dtype(dtype):
    return {"e4m3_float8": torch.float8_e4m3fn, "e5m2_float8": torch.float8_e5m2}.get(
        dtype, getattr(torch, dtype, None)
    )


def _bits(x):
    return x.contiguous().view(torch.uint8)


def numeric_case(index, runtime):
    c = CASES[index]
    if c["op"] == "compose":
        return numeric_composition(index, runtime)
    op = c["op"]
    dtype = c["dtype"]
    dt = _payload_dtype(dtype)
    if op == "gather":
        rows, count, width = 17, c["count"], c["width"]
        p = ((torch.arange(rows * width).reshape(rows, width) % 25) - 12).to(dt)
        idx = (torch.arange(count) * 7 % rows).reshape(count, 1).to(torch.uint32)
        if count == 1:
            idx[0, 0] = rows - 1
        elif count > 1:
            idx[1, 0] = idx[0, 0]
        o = torch.empty((count, width), dtype=dt)
        kernel = tilelang.compile(
            gather_kernel(dtype, rows, count, width),
            out_idx=[2],
            target=TARGET,
            runtime_mode=runtime,
        )
        kernel(p, idx, o)
        expected = _bits(p)[idx.to(torch.int64).flatten()]
        assert torch.equal(_bits(o), expected)
    elif op == "scatter":
        rows, count, width = 7, (3 if c.get("adversarial") else 9), c["width"]
        d = (torch.arange(rows * width).reshape(rows, width) % 7 - 3).to(dt)
        s = (torch.arange(count * width).reshape(count, width) % 9 - 4).to(dt)
        idx = (
            torch.tensor([6, 0, 6, 3, 0, 6, 2, 0, 3][:count])
            .reshape(count, 1)
            .to(getattr(torch, c["index_dtype"]))
        )
        if c.get("adversarial"):
            large = {"float16": 2048, "bfloat16": 256, "float32": 16777216}[dtype]
            d.fill_(1)
            s[0].fill_(large)
            s[1].fill_(1)
            s[2].fill_(-large)
            idx.fill_(0)
        expected = d.clone()
        for j in range(count):
            row = int(idx[j, 0])
            expected[row] = (expected[row].float() + s[j].float()).to(dt)
        kernel = tilelang.compile(
            scatter_kernel(dtype, c["index_dtype"], rows, count, width),
            out_idx=[0],
            target=TARGET,
            runtime_mode=runtime,
        )
        kernel(d, s, idx)
        assert torch.equal(_bits(d), _bits(expected))
    else:
        length, k = c["length"], c["k"]
        rows = c.get("rows", 1)
        local = c.get("local", False)
        output_width = length if c.get("full_output") else k
        g = torch.Generator().manual_seed(53009)
        x = (
            (torch.randint(-7, 8, (rows, length), generator=g) * 0.25).to(dt)
            if dtype == "float32"
            else torch.randint(0, 16, (rows, length), generator=g).to(dt)
        )
        if dtype == "float32":
            special = torch.tensor(
                [
                    0x7FC00001,
                    0xFFC00002,
                    0,
                    0x80000000,
                    0x7F800000,
                    0xFF800000,
                    1,
                    0x80000001,
                ],
                dtype=torch.uint32,
            ).view(torch.float32)
            x[:, : min(length, len(special))] = special[: min(length, len(special))]
        elif dtype == "int32":
            for j, v in enumerate((-2147483648, 2147483647, 0, -1, 1)):
                if j < length:
                    x[:, j] = v
        else:
            for j, v in enumerate((0, 4294967295, 2147483648, 1, 4294967295)):
                if j < length:
                    x[:, j] = v
        expected_indices = []
        for row in x:
            values = row.to(torch.float64).tolist()

            def key(i, values=values):
                v = values[i]
                nan = math.isnan(v)
                return (
                    (0 if nan else 1) if c["descending"] else (1 if nan else 0),
                    0 if nan else (-v if c["descending"] else v),
                )

            expected_indices.append(sorted(range(length), key=key)[:k])
        ids = torch.tensor(expected_indices, dtype=torch.int64)
        expected = torch.gather(
            x.to(torch.int64) if dtype == "uint32" else x, 1, ids
        ).to(dt)
        shape = (rows, output_width) if local else (k,)
        v = torch.full(shape, 123.25 if dtype == "float32" else 123, dtype=dt)
        i = torch.full(
            shape, 123456, dtype=getattr(torch, c.get("index_dtype", "int32"))
        )
        args = {key: value for key, value in c.items() if key != "op"}
        kernel = tilelang.compile(
            topk_kernel(**args), out_idx=[1, 2], target=TARGET, runtime_mode=runtime
        )
        kernel(x if local else x.flatten(), v, i)
        actual = v[:, :k] if local else v.reshape(1, k)
        actual_ids = i[:, :k] if local else i.reshape(1, k)
        assert torch.equal(actual_ids.to(torch.int64), ids), (actual_ids, ids)
        assert torch.equal(_bits(actual), _bits(expected)), (actual, expected)
        if local and output_width > k:
            assert torch.all(v[:, k:] == 123.25) and torch.all(i[:, k:] == 123456), (
                "Top-K overwrote the unspecified output tail"
            )
    return dict(status="passed", case=index, configuration=c, runtime=runtime)


def composition_kernel():
    rows, count, width, k = 7, 5, 5, 2

    @T.prim_func
    def kernel(
        P: T.Tensor((rows, width), "float32"),
        Idx: T.Tensor((count, 1), "uint32"),
        Mid: T.Tensor((count, width), "float32"),
        Dst: T.Tensor((rows, width), "float32"),
        Out: T.Tensor((rows, k), "float32"),
        OutIdx: T.Tensor((rows, k), "int32"),
    ):
        T.func_attr({"global_symbol": f"compose_{rows}_{count}_{width}_{k}"})
        with T.Kernel(1, is_cpu=True):
            T.ppl_gather(Mid, P, Idx, rows)
            T.ppl_scatter_add(Dst, Mid, Idx)
            x = T.alloc_shared((rows, width), "float32")
            v = T.alloc_shared((rows, k), "float32")
            i = T.alloc_shared((rows, k), "int32")
            w0 = T.alloc_shared((rows, 32), "uint32")
            w1 = T.alloc_shared((rows, 32), "uint32")
            w2 = T.alloc_shared((rows, 32), "uint32")
            w3 = T.alloc_shared((rows, 32), "uint32")
            w4 = T.alloc_shared((rows, 32), "uint32")
            w5 = T.alloc_shared((rows, 32), "uint32")
            w6 = T.alloc_shared((rows, 32), "uint32")
            w7 = T.alloc_shared((rows, 32), "uint16")
            T.ppl_copy(Dst, x)
            T.ppl_topk(
                v, i, x, k, True, width, workspace=(w0, w1, w2, w3, w4, w5, w6, w7)
            )
            T.ppl_copy(v, Out)
            T.ppl_copy(i, OutIdx)

    return kernel


def numeric_composition(index, runtime):
    param = (torch.arange(35).reshape(7, 5) % 9 - 4).float()
    ids = torch.tensor([2, 0, 2, 6, 0], dtype=torch.uint32).reshape(5, 1)
    mid = torch.zeros((5, 5))
    dst = torch.ones((7, 5))
    values = torch.empty((7, 2))
    indices = torch.empty((7, 2), dtype=torch.int32)
    expected_mid = param[ids.to(torch.int64).flatten()]
    expected_dst = dst.clone()
    for row, entry in enumerate(ids.to(torch.int64).flatten()):
        expected_dst[entry] += expected_mid[row]
    expected_ids = torch.tensor(
        [
            sorted(range(5), key=lambda j: (-float(row[j]), j))[:2]
            for row in expected_dst
        ]
    )
    expected_values = torch.gather(expected_dst, 1, expected_ids)
    kernel = tilelang.compile(
        composition_kernel(), out_idx=[2, 3, 4, 5], target=TARGET, runtime_mode=runtime
    )
    kernel(param, ids, mid, dst, values, indices)
    assert torch.equal(mid, expected_mid)
    assert torch.equal(dst, expected_dst)
    assert torch.equal(values, expected_values)
    assert torch.equal(indices.to(torch.int64), expected_ids)
    return dict(
        status="passed", case=index, configuration=CASES[index], runtime=runtime
    )


def test_indexed_operations_compose_in_one_kernel():
    tilelang.lower(composition_kernel(), target=TARGET, runtime_mode="cmodel")


@pytest.mark.parametrize(
    "dtype", ("float16", "bfloat16", "float32", "uint32", "e4m3_float8", "e5m2_float8")
)
def test_gather_rv_codegen(dtype):
    source = tilelang.lower(
        gather_kernel(dtype), target=TARGET, runtime_mode="cmodel"
    ).kernel_source
    assert "rvt_dma_hgather(" in source


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32"))
@pytest.mark.parametrize("index_dtype", ("int32", "uint32"))
def test_scatter_rv_codegen(dtype, index_dtype):
    source = tilelang.lower(
        scatter_kernel(dtype, index_dtype), target=TARGET, runtime_mode="cmodel"
    ).kernel_source
    assert "rvt_dma_fhscatter(" in source


def _legacy_indexed_kernel(op):
    from tilelang import tvm

    function = gather_kernel() if op == "gather" else scatter_kernel()
    portable_name = f"tl.tpu.{op}"
    legacy_name = f"tl.tpukernel.{op}"
    rewritten = []

    def use_legacy_name(node):
        if (
            node.op.same_as(tvm.ir.Op.get("tir.call_extern"))
            and isinstance(node.args[0], tvm.tir.StringImm)
            and node.args[0].value == portable_name
        ):
            rewritten.append(node)
            return tvm.tir.Call(
                node.dtype, node.op, [tvm.tir.StringImm(legacy_name), *node.args[1:]]
            )
        return node

    legacy = function.with_body(
        tvm.tir.stmt_functor.ir_transform(
            function.body, None, use_legacy_name, ["tir.Call"]
        )
    )
    assert len(rewritten) == 1
    return function, legacy


@pytest.mark.parametrize("op", ("gather", "scatter_add"))
@pytest.mark.parametrize("chip", ("bm1690", "bm1690e"))
def test_legacy_indexed_externs_preserve_tpukernel_codegen(op, chip):
    target = f"tpu -mcpu={chip} -tpu-programming-model=tpukernel"
    function, legacy = _legacy_indexed_kernel(op)
    source = tilelang.lower(
        function, target=target, runtime_mode="cmodel"
    ).kernel_source
    legacy_source = tilelang.lower(
        legacy, target=target, runtime_mode="cmodel"
    ).kernel_source
    assert legacy_source == source


@pytest.mark.parametrize("op", ("gather", "scatter_add"))
def test_legacy_indexed_externs_still_require_tpukernel(op):
    _, legacy = _legacy_indexed_kernel(op)
    with pytest.raises(ValueError, match="requires 'tpukernel'"):
        tilelang.lower(legacy, target=TARGET, runtime_mode="cmodel")


@pytest.mark.parametrize("local,rows", ((False, 1), (True, 1), (True, 65)))
def test_topk_rv_codegen(local, rows):
    source = tilelang.lower(
        topk_kernel(local=local, rows=rows), target=TARGET, runtime_mode="cmodel"
    ).kernel_source
    assert "rvt_gather(" in source and "tpu_hau_sort_natural_index(" not in source


def _topk_buffers(length=33, k=7):
    from tilelang import tvm

    decl = tvm.tir.decl_buffer
    width = max(32, 1 << (length - 1).bit_length())
    return (
        decl((k,), "float32"),
        decl((k,), "int32"),
        decl((length,), "float32"),
        [
            decl((1, width), "uint16" if i == 7 else "uint32", scope="shared")
            for i in range(8)
        ],
    )


@pytest.mark.parametrize(
    "issue", ("count", "dtype", "shape", "scope", "alias", "operand-alias")
)
def test_topk_rejects_invalid_workspace(issue):
    from tilelang import tvm

    v, i, x, w = _topk_buffers()
    if issue == "count":
        w = w[:-1]
    elif issue == "dtype":
        w[-1] = tvm.tir.decl_buffer((1, 64), "uint32", scope="shared")
    elif issue == "shape":
        w[-1] = tvm.tir.decl_buffer((1, 32), "uint16", scope="shared")
    elif issue == "scope":
        w[0] = tvm.tir.decl_buffer((1, 64), "uint32")
    elif issue == "alias":
        w[1] = w[0]
    else:
        v = tvm.tir.decl_buffer((7,), "float32", data=w[0].data)
    with pytest.raises((ValueError, TypeError)):
        T.ppl_topk(v, i, x, 7, True, 33, workspace=w)


@pytest.mark.parametrize("length", (8192,))
def test_topk_largest_global_workspace_that_fits(length):
    tilelang.lower(
        topk_kernel(length=length, k=1), target=TARGET, runtime_mode="cmodel"
    )


def test_topk_oversized_workspace_is_rejected_before_runtime():
    with pytest.raises(Exception, match="(?i)memory|capacity|allocate|lmem|fit"):
        tilelang.lower(
            topk_kernel(length=16384, k=1), target=TARGET, runtime_mode="cmodel"
        )


def test_scatter_width_one_is_rejected_only_on_rv():
    with pytest.raises(Exception, match="row width >= 2"):
        tilelang.lower(scatter_kernel(width=1), target=TARGET, runtime_mode="cmodel")
    tilelang.lower(
        scatter_kernel(width=1),
        target="tpu -mcpu=bm1690 -tpu-programming-model=tpukernel",
        runtime_mode="cmodel",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=int, required=True)
    p.add_argument("--runtime", choices=("cmodel", "pcie"), required=True)
    a = p.parse_args()
    result = numeric_case(a.case, a.runtime)
    Path("numeric.json").write_text(json.dumps(result, indent=2))
    print("RV_INDEXED_OP_RESULT=" + json.dumps(result), flush=True)
