# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Paged decode attention: gather K/V rows from a paged cache, then attend.

Composition, using only primitives already in the portable contract:

    ppl_embedding   gather the K/V rows named by an index table
    ppl_gemm        S = Q @ K^T  and  O = P @ V
    ppl_reduce_max  per-row softmax max
    ppl_reduce_sum  per-row softmax denominator
    ppl_exp         exp
    ppl_copy / fill / mul_C / subtract / div

Cache layout, matching the vendor's `paged_attention_multicore.pl`:

    flat row = block * kv_heads * block_size + kv_head * block_size + slot

The index tables are built on the host — in a real decoder the block table
comes from the scheduler, so computing it here keeps the kernel free of index
arithmetic.  GQA is supported: `q_heads` must be a multiple of `kv_heads`, and
the heads in one group share a cache head.

Only one decode step is modelled, so every query carries exactly one token.
"""

from typing import Optional, Sequence

import tilelang.language as T
import torch

from tpu_demo.common import (comparison, compile_and_launch, result_payload, tolerance, torch_dtype,
                             validate_dimensions, validate_selection)

OPERATION = "paged-attention"

BLOCK_SIZE = 16
HEAD_DIM = 64

DEFAULT_NUM_BLOCKS = 4
DEFAULT_Q_HEADS = 4
DEFAULT_KV_HEADS = 2
# Deliberately out of order: a sequential table would not exercise paging.
DEFAULT_BLOCK_TABLE = (2, 0, 3, 1)

@T.macro
def attend(h, ctx, scale, q, kstage, vstage, out, q_l, k_l, v_l, s_fp32, s_max, s_shift, s_sum,
           p_fp32, p_h, o_fp32, o_h, exp_work0, exp_work1, exp_coeff):
    """One query head: S = QK^T, row softmax, O = PV.

    `h` is either a Python integer (single core, unrolled) or a runtime value
    (one core per head).
    """
    base = h * ctx

    T.ppl_copy(q[h, 0], q_l)
    T.ppl_copy(kstage[base, 0], k_l)
    T.ppl_copy(vstage[base, 0], v_l)

    T.ppl_fill(s_fp32, T.float32(0.0))
    T.ppl_gemm(q_l, k_l, s_fp32, transpose_B=True, accumulate=False)
    T.ppl_mul_C(s_fp32, s_fp32, T.float32(scale))

    T.ppl_reduce_max(s_fp32, s_max, dim=1)
    T.ppl_subtract(s_shift, s_fp32, s_max)
    T.ppl_exp(s_shift, exp_work0, exp_work1, exp_coeff)
    T.ppl_reduce_sum(s_shift, s_sum, dim=1)
    T.ppl_div(p_fp32, s_shift, s_sum)
    T.ppl_copy(p_fp32, p_h)

    T.ppl_fill(o_fp32, T.float32(0.0))
    T.ppl_gemm(p_h, v_l, o_fp32, transpose_B=False, accumulate=False)
    T.ppl_copy(o_fp32, o_h)
    T.ppl_copy(o_h, out[h, 0])

def build_paged_attention(*,
                          num_blocks: int = DEFAULT_NUM_BLOCKS,
                          q_heads: int = DEFAULT_Q_HEADS,
                          kv_heads: int = DEFAULT_KV_HEADS,
                          head_dim: int = HEAD_DIM,
                          block_size: int = BLOCK_SIZE,
                          dtype: str = "float16",
                          programming_model: str = "rv",
                          multicore: bool = False):
    """Build the decode-attention program.

    With `multicore` the query heads are laid out on the grid, one per core.
    Each core still gathers the whole table into the same staging buffers --
    they all write identical bytes, and every core reads only the region it
    just wrote, so there is no cross-core race.  It is nonetheless slower than
    the single-core form, which gathers once.
    """
    if programming_model not in ("tpukernel", "rv"):
        raise ValueError(f"unsupported TPU programming model: {programming_model!r}")
    torch_dtype(dtype)
    validate_dimensions(
        OPERATION,
        num_blocks=num_blocks,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        block_size=block_size)
    if q_heads % kv_heads:
        raise ValueError("q_heads must be a multiple of kv_heads (GQA)")

    ctx = num_blocks * block_size
    rows = num_blocks * kv_heads * block_size
    staged = q_heads * ctx
    scale = 1.0 / (head_dim**0.5)
    grid = q_heads if multicore else 1

    @T.prim_func
    def paged_attention_kernel(q: T.Tensor((q_heads, head_dim), dtype),
                        kcache: T.Tensor((rows, head_dim), dtype),
                        vcache: T.Tensor((rows, head_dim), dtype),
                        kidx: T.Tensor((staged, 1), "uint32"),
                        vidx: T.Tensor((staged, 1), "uint32"),
                        kstage: T.Tensor((staged, head_dim), dtype),
                        vstage: T.Tensor((staged, head_dim), dtype),
                        out: T.Tensor((q_heads, head_dim), dtype)):
        with T.Kernel(grid, is_cpu=True) as (bx,):
            q_l = T.alloc_shared((1, head_dim), dtype)
            k_l = T.alloc_shared((ctx, head_dim), dtype)
            v_l = T.alloc_shared((ctx, head_dim), dtype)

            s_fp32 = T.alloc_shared((1, ctx), "float32")
            s_max = T.alloc_shared((1, 1), "float32")
            s_shift = T.alloc_shared((1, ctx), "float32")
            s_sum = T.alloc_shared((1, 1), "float32")
            p_fp32 = T.alloc_shared((1, ctx), "float32")
            p_h = T.alloc_shared((1, ctx), dtype)

            o_fp32 = T.alloc_shared((1, head_dim), "float32")
            o_h = T.alloc_shared((1, head_dim), dtype)

            exp_work0 = T.alloc_shared((1, ctx), "float32")
            exp_work1 = T.alloc_shared((1, ctx), "float32")
            # ppl_exp requires a (64, 32) coefficient buffer; unused on RV.
            exp_coeff = T.alloc_shared((64, 32), "float32")

            # Gather every KV row this batch of queries needs, once.
            T.ppl_embedding(kstage, kcache, kidx)
            T.ppl_embedding(vstage, vcache, vidx)

            if multicore:
                attend(bx, ctx, scale, q, kstage, vstage, out, q_l, k_l, v_l, s_fp32, s_max,
                       s_shift, s_sum, p_fp32, p_h, o_fp32, o_h, exp_work0, exp_work1, exp_coeff)
            else:
                for h in range(q_heads):
                    attend(h, ctx, scale, q, kstage, vstage, out, q_l, k_l, v_l, s_fp32, s_max,
                           s_shift, s_sum, p_fp32, p_h, o_fp32, o_h, exp_work0, exp_work1,
                           exp_coeff)

    return paged_attention_kernel, ctx, rows, staged

def paged_rows(kv_head: int, block_table: Sequence[int], kv_heads: int,
               block_size: int = BLOCK_SIZE) -> list[int]:
    """Flat cache row for each slot of each block, in the vendor's index formula."""
    rows = []
    for block in block_table:
        for slot in range(block_size):
            rows.append(block * kv_heads * block_size + kv_head * block_size + slot)
    return rows

def index_tables(q_heads: int, kv_heads: int, block_table: Sequence[int],
                 block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """One index table per query head; a GQA group shares its key/value head."""
    group = q_heads // kv_heads
    table: list[int] = []
    for head in range(q_heads):
        table.extend(paged_rows(head // group, block_table, kv_heads, block_size))
    return torch.tensor(table, dtype=torch.int64)

def torch_reference(q: torch.Tensor, kcache: torch.Tensor, vcache: torch.Tensor, *,
                    q_heads: int, kv_heads: int, block_table: Sequence[int],
                    head_dim: int = HEAD_DIM, block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """Host oracle: gather the same rows and run a plain softmax attention."""
    group = q_heads // kv_heads
    outputs = []
    for head in range(q_heads):
        idx = paged_rows(head // group, block_table, kv_heads, block_size)
        k = kcache[idx].to(torch.float32)
        v = vcache[idx].to(torch.float32)
        scores = q[head:head + 1].to(torch.float32) @ k.T / (head_dim**0.5)
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(probabilities @ v)
    return torch.cat(outputs, dim=0)

def run(*,
        dtype: str,
        chip: str,
        programming_model: str,
        runtime_mode: str,
        allow_pcie: bool = False,
        device_id: Optional[int] = None,
        seed: int = 0) -> dict:
    torch_dtype(dtype)
    validate_selection(
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode,
        supports_rv=True,
        allow_pcie=allow_pcie,
        device_id=device_id,
    )

    num_blocks, q_heads, kv_heads = DEFAULT_NUM_BLOCKS, DEFAULT_Q_HEADS, DEFAULT_KV_HEADS
    block_table = DEFAULT_BLOCK_TABLE
    parameters = {
        "num_blocks": num_blocks,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": HEAD_DIM,
        "block_size": BLOCK_SIZE,
        "block_table": list(block_table),
        "multicore": False,
        "seed": seed,
    }

    host_dtype = torch_dtype(dtype)
    generator = torch.Generator().manual_seed(seed)
    ctx = num_blocks * BLOCK_SIZE
    rows = num_blocks * kv_heads * BLOCK_SIZE
    staged = q_heads * ctx

    q = (torch.randn((q_heads, HEAD_DIM), generator=generator) * 0.5).to(host_dtype)
    kcache = (torch.randn((rows, HEAD_DIM), generator=generator) * 0.5).to(host_dtype)
    vcache = (torch.randn((rows, HEAD_DIM), generator=generator) * 0.5).to(host_dtype)

    tables = index_tables(q_heads, kv_heads, block_table).reshape(staged, 1).to(torch.uint32)
    kstage = torch.zeros((staged, HEAD_DIM), dtype=host_dtype)
    vstage = torch.zeros((staged, HEAD_DIM), dtype=host_dtype)
    destination = torch.zeros((q_heads, HEAD_DIM), dtype=host_dtype)

    program, _, _, _ = build_paged_attention(
        num_blocks=num_blocks,
        q_heads=q_heads,
        kv_heads=kv_heads,
        dtype=dtype,
        programming_model=programming_model)
    timing = compile_and_launch(
        program, (q, kcache, vcache, tables, tables, kstage, vstage, destination),
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode)

    expected = torch_reference(
        q, kcache, vcache, q_heads=q_heads, kv_heads=kv_heads,
        block_table=block_table).to(host_dtype)
    atol, rtol = tolerance(dtype, "flashattn")
    metrics = comparison(destination, expected, atol=atol, rtol=rtol)
    return result_payload(
        operation=OPERATION,
        dtype=dtype,
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode,
        metrics=metrics,
        timing=timing,
        parameters=parameters)
