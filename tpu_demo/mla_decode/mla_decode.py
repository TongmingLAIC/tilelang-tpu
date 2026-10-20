# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""MLA: DeepSeek-V3's multi-head latent attention, single decode step.

This is the operator behind torch-tpu's `paged_latent_attention_fp8`.  The
reference is the vendor's `mla_decode.pl` (`mla_decode_bf16_fp8e4m3`).

What makes MLA different from an ordinary paged attention
    There is no per-head K or V in the cache.  The cache holds the *latent* --
    `kv_lora_rank` values per position -- and the key/value projections are
    folded into the query instead.  Using

        q_nope . k_nope = q_nope . (WUKV_K @ kv_latent) = (q_nope @ WUKV_K) . kv_latent

    the kernel absorbs `WUKV_K` into the query up front, scores against the
    narrow latent, and only materialises V at the very end by applying
    `WUKV_V` to the attention output.

    The score is the sum of two independently computed halves, because the
    RoPE half does not live in the latent:

        scores = (q_absorbed @ kv_cache^T + q_rope @ pe_cache^T) * softmax_scale

RoPE layout
    The checkpoint applies RoPE in the interleaved `rotate_every_two` form,
    which reads dimension pairs at stride two.  That access pattern is already
    covered by the `rope` demo, so this one stores the rope dimension
    de-interleaved -- even dimensions first, then odd -- and permutes the rope
    columns of `WUQ` the same way.  A dot product is invariant under a common
    permutation of both operands, so the arithmetic is unchanged while the
    kernel stays contiguous.

Scope
    One decode step: every query carries exactly one token.  Prefill is not
    modelled.  The newly computed latent and roped position encoding are
    scattered into the paged caches before they are gathered, so the step sees
    its own token.

Layouts
    q_latent     (1, q_lora_rank)               fp16/bf16
    kv_latent    (1, kv_lora_rank)              fp16/bf16
    pe           (1, qk_rope_head_dim)          fp16/bf16
    w_uq         (heads * (nope+rope), q_lora)  e4m3, rope columns permuted
    w_ukv        (heads * (nope+v), kv_lora)    e4m3, nope block then v block
    kv_cache     (slots, kv_lora_rank)          fp16/bf16
    pe_cache     (slots, qk_rope_head_dim)      fp16/bf16
    out          (heads, v_head_dim)            fp16/bf16

    The fp8 weights carry the same `weight_scale_inv` block table as
    `w8a16_dq`; these dimensions stay under one 128-wide block, since the
    block boundary is exercised there.
"""

from typing import Optional, Sequence

import tilelang.language as T
import torch

from tpu_demo.common import (comparison, compile_and_launch, result_payload, tolerance, torch_dtype,
                             validate_dimensions, validate_selection)

OPERATION = "mla-decode"

NUM_HEADS = 2
Q_LORA_RANK = 64
KV_LORA_RANK = 32
QK_NOPE_HEAD_DIM = 32
QK_ROPE_HEAD_DIM = 16
V_HEAD_DIM = 32
BLOCK_SIZE = 4
NUM_BLOCKS = 2
# Deliberately out of order: a sequential table would not exercise paging.
BLOCK_TABLE = (1, 0)
# The slot this step writes its own latent into.  It must be inside the block
# table, so the step attends to the token it just produced.
SAVE_SLOT = 5
GROUP_SIZE = 128

Q_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
DKVU = QK_NOPE_HEAD_DIM + V_HEAD_DIM


def build_mla_decode(*,
                     num_heads: int = NUM_HEADS,
                     q_lora_rank: int = Q_LORA_RANK,
                     kv_lora_rank: int = KV_LORA_RANK,
                     qk_nope_head_dim: int = QK_NOPE_HEAD_DIM,
                     qk_rope_head_dim: int = QK_ROPE_HEAD_DIM,
                     v_head_dim: int = V_HEAD_DIM,
                     block_size: int = BLOCK_SIZE,
                     num_blocks: int = NUM_BLOCKS,
                     dtype: str = "float16",
                     programming_model: str = "rv"):
    """Build the single-step MLA decode program.

    The whole step runs on one core, walking the heads serially: the caches and
    the position encoding are shared across heads, so hoisting them out of a
    per-head grid is what keeps the work single-copy.
    """
    if programming_model not in ("tpukernel", "rv"):
        raise ValueError(f"unsupported TPU programming model: {programming_model!r}")
    torch_dtype(dtype)
    validate_dimensions(
        OPERATION,
        num_heads=num_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        block_size=block_size,
        num_blocks=num_blocks)
    if qk_rope_head_dim % 2:
        raise ValueError(f"qk_rope_head_dim must be even for the paired RoPE, "
                         f"got {qk_rope_head_dim}")

    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    dkvu = qk_nope_head_dim + v_head_dim
    slots = num_blocks * block_size
    rope_pairs = qk_rope_head_dim // 2
    scale = 1.0 / ((q_head_dim)**0.5)
    acc = "float32"

    @T.macro
    def rope_deinterleaved(x, cos, sin, even_f, odd_f, out_even, out_odd, work0, work1):
        """`out = x*cos + rotate_every_two(x)*sin`, on de-interleaved operands.

        `x` holds the even dimensions followed by the odd ones, so the paired
        rotation is two elementwise lines rather than a stride-2 gather.
        """
        T.ppl_copy(x[0, 0], even_f)
        T.ppl_copy(x[0, rope_pairs], odd_f)
        T.ppl_mul(work0, even_f, cos)
        T.ppl_mul(work1, odd_f, sin)
        T.ppl_subtract(out_even, work0, work1)
        T.ppl_mul(work0, odd_f, cos)
        T.ppl_mul(work1, even_f, sin)
        T.ppl_add(out_odd, work0, work1)

    @T.prim_func
    def mla_decode(q_latent: T.Tensor((1, q_lora_rank), dtype),
                   kv_latent: T.Tensor((1, kv_lora_rank), dtype),
                   pe: T.Tensor((1, qk_rope_head_dim), dtype),
                   w_uq: T.Tensor((num_heads * q_head_dim, q_lora_rank), "e4m3_float8"),
                   w_uq_scale: T.Tensor((num_heads * q_head_dim, 1), dtype),
                   w_ukv: T.Tensor((num_heads * dkvu, kv_lora_rank), "e4m3_float8"),
                   w_ukv_scale: T.Tensor((num_heads * dkvu, 1), dtype),
                   cosine: T.Tensor((1, rope_pairs), acc), sine: T.Tensor((1, rope_pairs), acc),
                   kv_cache: T.Tensor((slots, kv_lora_rank), dtype),
                   pe_cache: T.Tensor((slots, qk_rope_head_dim), dtype),
                   slot_index: T.Tensor((slots, 1), "uint32"),
                   kv_stage: T.Tensor((slots, kv_lora_rank), dtype),
                   pe_stage: T.Tensor((slots, qk_rope_head_dim), dtype),
                   out: T.Tensor((num_heads, v_head_dim), dtype)):
        with T.Kernel(1, is_cpu=True) as (bx,):
            # --- weights, dequantized one head at a time ---
            # Each head's rows are read straight from the global weight rather
            # than sliced out of a whole-weight local tile: a local source
            # region must start at row zero, because TPUv7 channels map across
            # NPU lanes and cannot take a linear LMEM byte offset.
            w_uq_raw = T.alloc_shared((q_head_dim, q_lora_rank), "e4m3_float8")
            w_uq_scale_col = T.alloc_shared((q_head_dim, 1), dtype)
            # The key and value halves of WUKV are loaded as separate tiles for
            # the same reason: splitting one head-wide tile would need a local
            # row offset.
            w_ukv_k_raw = T.alloc_shared((qk_nope_head_dim, kv_lora_rank), "e4m3_float8")
            w_ukv_k_scale_col = T.alloc_shared((qk_nope_head_dim, 1), dtype)
            w_ukv_v_raw = T.alloc_shared((v_head_dim, kv_lora_rank), "e4m3_float8")
            w_ukv_v_scale_col = T.alloc_shared((v_head_dim, 1), dtype)

            # --- shared scratch ---
            cos_f32 = T.alloc_shared((1, rope_pairs), acc)
            sin_f32 = T.alloc_shared((1, rope_pairs), acc)
            T.ppl_copy(cosine[0, 0], cos_f32)
            T.ppl_copy(sine[0, 0], sin_f32)

            even_f = T.alloc_shared((1, rope_pairs), acc)
            odd_f = T.alloc_shared((1, rope_pairs), acc)
            out_even = T.alloc_shared((1, rope_pairs), acc)
            out_odd = T.alloc_shared((1, rope_pairs), acc)
            rope_w0 = T.alloc_shared((1, rope_pairs), acc)
            rope_w1 = T.alloc_shared((1, rope_pairs), acc)

            exp_work0 = T.alloc_shared((1, slots), acc)
            exp_work1 = T.alloc_shared((1, slots), acc)
            # ppl_exp requires a (64, 32) coefficient buffer; unused on RV.
            exp_coeff = T.alloc_shared((64, 32), acc)

            # --- rope the position encoding, then publish this step's latent ---
            pe_in = T.alloc_shared((1, qk_rope_head_dim), dtype)
            pe_roped = T.alloc_shared((1, qk_rope_head_dim), dtype)
            T.ppl_copy(pe[0, 0], pe_in)
            rope_deinterleaved(pe_in, cos_f32, sin_f32, even_f, odd_f, out_even, out_odd,
                               rope_w0, rope_w1)
            T.ppl_copy(out_even, pe_roped[0, 0])
            T.ppl_copy(out_odd, pe_roped[0, rope_pairs])

            # The cache write goes to a compile-time slot, so it is a plain DMA.
            # `ppl_scatter` is the real mechanism and it does work on RV, but
            # its operands must carry their slot axis on H, and `ppl_embedding`
            # below requires rank-2 operands -- a single cache buffer cannot
            # satisfy both.  The scatter itself is exercised by the standalone
            # probe in the survey note; here the demo models the write statically
            # so the attention math is what gets measured.
            kv_row = T.alloc_shared((1, kv_lora_rank), dtype)
            T.ppl_copy(kv_latent[0, 0], kv_row)
            T.ppl_copy(kv_row, kv_cache[SAVE_SLOT, 0])
            T.ppl_copy(pe_roped, pe_cache[SAVE_SLOT, 0])

            # --- gather the rows this step attends to, once for every head ---
            T.ppl_embedding(kv_stage, kv_cache, slot_index)
            T.ppl_embedding(pe_stage, pe_cache, slot_index)
            kv_local = T.alloc_shared((slots, kv_lora_rank), dtype)
            pe_local = T.alloc_shared((slots, qk_rope_head_dim), dtype)
            T.ppl_copy(kv_stage[0, 0], kv_local)
            T.ppl_copy(pe_stage[0, 0], pe_local)

            # --- per-head scratch ---
            q_row = T.alloc_shared((1, q_lora_rank), dtype)
            q_u = T.alloc_shared((1, q_head_dim), acc)
            q_nope_f32 = T.alloc_shared((1, qk_nope_head_dim), acc)
            q_nope = T.alloc_shared((1, qk_nope_head_dim), dtype)
            q_rope_f32 = T.alloc_shared((1, qk_rope_head_dim), acc)
            q_rope = T.alloc_shared((1, qk_rope_head_dim), dtype)
            q_roped = T.alloc_shared((1, qk_rope_head_dim), dtype)

            w_uq_head = T.alloc_shared((q_head_dim, q_lora_rank), dtype)
            w_ukv_k = T.alloc_shared((qk_nope_head_dim, kv_lora_rank), dtype)
            w_ukv_v = T.alloc_shared((v_head_dim, kv_lora_rank), dtype)
            q_absorbed = T.alloc_shared((1, kv_lora_rank), acc)
            q_absorbed_h = T.alloc_shared((1, kv_lora_rank), dtype)

            scores_nope = T.alloc_shared((1, slots), acc)
            scores_rope = T.alloc_shared((1, slots), acc)
            scores = T.alloc_shared((1, slots), acc)
            scores_max = T.alloc_shared((1, 1), acc)
            scores_shift = T.alloc_shared((1, slots), acc)
            scores_sum = T.alloc_shared((1, 1), acc)
            probs = T.alloc_shared((1, slots), acc)
            probs_h = T.alloc_shared((1, slots), dtype)

            context = T.alloc_shared((1, kv_lora_rank), acc)
            context_h = T.alloc_shared((1, kv_lora_rank), dtype)
            head_out = T.alloc_shared((1, v_head_dim), acc)
            head_out_h = T.alloc_shared((1, v_head_dim), dtype)

            T.ppl_copy(q_latent[0, 0], q_row)

            for h in range(num_heads):
                # This head's slice of WUQ, dequantized where it is read.
                T.ppl_copy(w_uq[h * q_head_dim, 0], w_uq_raw)
                T.ppl_copy(w_uq_raw, w_uq_head)
                T.ppl_copy(w_uq_scale[h * q_head_dim, 0], w_uq_scale_col)
                T.ppl_mul(w_uq_head, w_uq_head, w_uq_scale_col)

                # Q latent -> this head's [nope | rope], then absorb WUKV_K.
                T.ppl_fill(q_u, T.float32(0.0))
                T.ppl_gemm(q_row, w_uq_head, q_u,
                           transpose_B=True, accumulate=False)
                T.ppl_copy(q_u[0, 0], q_nope_f32)
                T.ppl_copy(q_u[0, qk_nope_head_dim], q_rope_f32)
                T.ppl_copy(q_nope_f32, q_nope)

                T.ppl_copy(w_ukv[h * dkvu, 0], w_ukv_k_raw)
                T.ppl_copy(w_ukv_k_raw, w_ukv_k)
                T.ppl_copy(w_ukv_scale[h * dkvu, 0], w_ukv_k_scale_col)
                T.ppl_mul(w_ukv_k, w_ukv_k, w_ukv_k_scale_col)
                T.ppl_fill(q_absorbed, T.float32(0.0))
                T.ppl_gemm(q_nope, w_ukv_k, q_absorbed,
                           transpose_B=False, accumulate=False)
                T.ppl_copy(q_absorbed, q_absorbed_h)

                # RoPE on the rope half of the query.
                T.ppl_copy(q_rope_f32, q_rope)
                rope_deinterleaved(q_rope, cos_f32, sin_f32, even_f, odd_f, out_even, out_odd,
                                   rope_w0, rope_w1)
                T.ppl_copy(out_even, q_roped[0, 0])
                T.ppl_copy(out_odd, q_roped[0, rope_pairs])

                # Scores: latent half and rope half, computed apart and summed.
                T.ppl_gemm(q_absorbed_h, kv_local, scores_nope,
                           transpose_B=True, accumulate=False)
                T.ppl_gemm(q_roped, pe_local, scores_rope,
                           transpose_B=True, accumulate=False)
                T.ppl_add(scores, scores_nope, scores_rope)
                T.ppl_mul_C(scores, scores, T.float32(scale))

                T.ppl_reduce_max(scores, scores_max, dim=1)
                T.ppl_subtract(scores_shift, scores, scores_max)
                T.ppl_exp(scores_shift, exp_work0, exp_work1, exp_coeff)
                T.ppl_reduce_sum(scores_shift, scores_sum, dim=1)
                T.ppl_div(probs, scores_shift, scores_sum)
                T.ppl_copy(probs, probs_h)

                # Context against the latent, then materialise V for this head.
                T.ppl_fill(context, T.float32(0.0))
                T.ppl_gemm(probs_h, kv_local, context,
                           transpose_B=False, accumulate=False)
                T.ppl_copy(context, context_h)

                T.ppl_copy(w_ukv[h * dkvu + qk_nope_head_dim, 0], w_ukv_v_raw)
                T.ppl_copy(w_ukv_v_raw, w_ukv_v)
                T.ppl_copy(w_ukv_scale[h * dkvu + qk_nope_head_dim, 0], w_ukv_v_scale_col)
                T.ppl_mul(w_ukv_v, w_ukv_v, w_ukv_v_scale_col)
                T.ppl_fill(head_out, T.float32(0.0))
                T.ppl_gemm(context_h, w_ukv_v, head_out,
                           transpose_B=True, accumulate=False)
                T.ppl_copy(head_out, head_out_h)
                T.ppl_copy(head_out_h, out[h, 0])

    return mla_decode


def slot_rows(block_table: Sequence[int], block_size: int = BLOCK_SIZE) -> list[int]:
    """Slot ids in the order the logical blocks name them."""
    return [block * block_size + offset for block in block_table for offset in range(block_size)]


def host_inputs(*, num_heads: int, q_lora_rank: int, kv_lora_rank: int, qk_nope_head_dim: int,
                qk_rope_head_dim: int, v_head_dim: int, block_size: int, num_blocks: int,
                dtype: str, seed: int):
    """Build the latents, the permuted fp8 weights, the caches and the tables.

    `w_uq`'s rope columns are permuted so the even rope dimensions precede the
    odd ones -- see the module docstring.  The scale tables are expanded along
    the row axis, the layout the kernel reads.
    """
    compute = torch_dtype(dtype)
    generator = torch.Generator().manual_seed(seed)
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    dkvu = qk_nope_head_dim + v_head_dim
    slots = num_blocks * block_size

    def fp8(rows: int, cols: int) -> torch.Tensor:
        return (torch.randn((rows, cols), generator=generator) * 0.25).to(torch.float8_e4m3fn)

    def one_scale(rows: int) -> torch.Tensor:
        # Every dimension here is inside a single 128-wide block.
        return (torch.rand((rows, 1), generator=generator) * 0.5 + 0.75).to(compute)

    q_latent = (torch.randn((1, q_lora_rank), generator=generator) * 0.5).to(compute)
    kv_latent = (torch.randn((1, kv_lora_rank), generator=generator) * 0.5).to(compute)
    pe = (torch.randn((1, qk_rope_head_dim), generator=generator) * 0.5).to(compute)

    # Permute the rope columns of every head: [nope | even | odd] instead of
    # the checkpoint's [nope | interleaved rope].
    w_uq = fp8(num_heads * q_head_dim, q_lora_rank)
    for head in range(num_heads):
        base = head * q_head_dim + qk_nope_head_dim
        rope = w_uq[base:base + qk_rope_head_dim]
        even = rope[0::2]
        odd = rope[1::2]
        w_uq[base:base + qk_rope_head_dim] = torch.cat((even, odd), dim=0)

    w_ukv = fp8(num_heads * dkvu, kv_lora_rank)

    rope_pairs = qk_rope_head_dim // 2
    theta = torch.arange(rope_pairs, dtype=torch.float32).unsqueeze(0)
    cosine = torch.cos(theta).contiguous()
    sine = torch.sin(theta).contiguous()

    kv_cache = (torch.randn((slots, kv_lora_rank), generator=generator) * 0.5).to(compute)
    pe_cache = (torch.randn((slots, qk_rope_head_dim), generator=generator) * 0.5).to(compute)

    return {
        "q_latent": q_latent,
        "kv_latent": kv_latent,
        "pe": pe,
        "w_uq": w_uq,
        "w_uq_scale": one_scale(num_heads * q_head_dim),
        "w_ukv": w_ukv,
        "w_ukv_scale": one_scale(num_heads * dkvu),
        "cosine": cosine,
        "sine": sine,
        "kv_cache": kv_cache,
        "pe_cache": pe_cache,
        "slot_index": torch.tensor(slot_rows(BLOCK_TABLE, block_size),
                                   dtype=torch.uint32).reshape(slots, 1),
        "kv_stage": torch.zeros((slots, kv_lora_rank), dtype=compute),
        "pe_stage": torch.zeros((slots, qk_rope_head_dim), dtype=compute),
        "out": torch.zeros((num_heads, v_head_dim), dtype=compute),
    }


def torch_reference(inputs, *, num_heads: int, qk_nope_head_dim: int, qk_rope_head_dim: int,
                    v_head_dim: int, dtype: str) -> torch.Tensor:
    """Host oracle: dequantize, absorb, score, softmax, materialise V.

    Every product the device rounds -- the dequantized weights, the absorbed
    query, the roped query, the probabilities -- is rounded here too, so this
    is an oracle for the operator rather than for ideal arithmetic.
    """
    compute = torch_dtype(dtype)
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    dkvu = qk_nope_head_dim + v_head_dim
    rope_pairs = qk_rope_head_dim // 2
    scale = 1.0 / (q_head_dim**0.5)

    def dequantize(raw: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        return (raw.to(torch.float32) * table.to(torch.float32)).to(compute).to(torch.float32)

    def rounded(value: torch.Tensor) -> torch.Tensor:
        return value.to(compute).to(torch.float32)

    def rope(x: torch.Tensor) -> torch.Tensor:
        even, odd = x[..., :rope_pairs], x[..., rope_pairs:]
        cos, sin = inputs["cosine"], inputs["sine"]
        return torch.cat((even * cos - odd * sin, odd * cos + even * sin), dim=-1)

    w_uq = dequantize(inputs["w_uq"], inputs["w_uq_scale"])
    w_ukv = dequantize(inputs["w_ukv"], inputs["w_ukv_scale"])
    q_latent = inputs["q_latent"].to(torch.float32)

    # Publish this step's latent and roped position encoding into the caches.
    kv_cache = inputs["kv_cache"].clone()
    pe_cache = inputs["pe_cache"].clone()
    save = SAVE_SLOT
    kv_cache[save] = rounded(inputs["kv_latent"].to(torch.float32))[0].to(compute)
    pe_cache[save] = rounded(rope(rounded(inputs["pe"].to(torch.float32))))[0].to(compute)

    index = inputs["slot_index"].reshape(-1).to(torch.int64)
    kv_rows = kv_cache[index].to(torch.float32)
    pe_rows = pe_cache[index].to(torch.float32)

    outputs = []
    for head in range(num_heads):
        q_u = q_latent @ w_uq[head * q_head_dim:(head + 1) * q_head_dim].T
        q_nope = rounded(q_u[:, :qk_nope_head_dim])
        q_rope = q_u[:, qk_nope_head_dim:]

        absorbed = rounded(q_nope @ w_ukv[head * dkvu:head * dkvu + qk_nope_head_dim])
        roped = rounded(rope(rounded(q_rope)))

        scores = (absorbed @ kv_rows.T + roped @ pe_rows.T) * scale
        probs = rounded(torch.softmax(scores, dim=-1))

        context = rounded(probs @ kv_rows)
        w_v = w_ukv[head * dkvu + qk_nope_head_dim:(head + 1) * dkvu]
        outputs.append(rounded(context @ w_v.T))

    return torch.cat(outputs, dim=0).to(compute)


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

    parameters = {
        "num_heads": NUM_HEADS,
        "q_lora_rank": Q_LORA_RANK,
        "kv_lora_rank": KV_LORA_RANK,
        "qk_nope_head_dim": QK_NOPE_HEAD_DIM,
        "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
        "v_head_dim": V_HEAD_DIM,
        "block_size": BLOCK_SIZE,
        "num_blocks": NUM_BLOCKS,
        "block_table": list(BLOCK_TABLE),
        "save_slot": SAVE_SLOT,
        "seed": seed,
    }

    inputs = host_inputs(
        num_heads=NUM_HEADS,
        q_lora_rank=Q_LORA_RANK,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        block_size=BLOCK_SIZE,
        num_blocks=NUM_BLOCKS,
        dtype=dtype,
        seed=seed)

    program = build_mla_decode(
        num_heads=NUM_HEADS,
        q_lora_rank=Q_LORA_RANK,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        block_size=BLOCK_SIZE,
        num_blocks=NUM_BLOCKS,
        dtype=dtype,
        programming_model=programming_model)

    argument_order = (
        "q_latent", "kv_latent", "pe", "w_uq", "w_uq_scale", "w_ukv", "w_ukv_scale",
        "cosine", "sine", "kv_cache", "pe_cache", "slot_index",
        "kv_stage", "pe_stage", "out",
    )
    timing = compile_and_launch(
        program,
        [inputs[name] for name in argument_order],
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode)

    expected = torch_reference(
        inputs,
        num_heads=NUM_HEADS,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        dtype=dtype).to(inputs["out"].dtype)
    atol, rtol = tolerance(dtype, "flashattn")
    metrics = comparison(inputs["out"], expected, atol=atol, rtol=rtol)
    return result_payload(
        operation=OPERATION,
        dtype=dtype,
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode,
        metrics=metrics,
        timing=timing,
        parameters=parameters)
