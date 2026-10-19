"""Demo builders plus explicit PPL-algorithm ablations.

Keep algorithm variants separate from the public examples and from tuning.
"""
import tilelang.language as T


def build_tiled_elementwise(op, rows, width, br, bw, dtype):

    @T.prim_func
    def kernel(a: T.Tensor((rows, width), dtype), b: T.Tensor((rows, width), dtype), out: T.Tensor(
        (rows, width), dtype)):
        with T.Kernel(rows // br, width // bw, is_cpu=True) as (r, w):
            x = T.alloc_shared((br, bw), dtype)
            y = T.alloc_shared((br, bw), dtype)
            z = T.alloc_shared((br, bw), dtype)
            T.ppl_copy(a[r * br, w * bw], x)
            T.ppl_copy(b[r * br, w * bw], y)
            if op == "add":
                T.ppl_add(z, x, y)
            elif op == "sub":
                T.ppl_subtract(z, x, y)
            elif op == "mul":
                T.ppl_mul(z, x, y)
            else:
                T.ppl_div(z, x, y)
            T.ppl_copy(z, out[r * br, w * bw])

    return kernel


def build_double_buffered_matmul(m, n, k, bm, bn, bk, dtype):
    # Extend tpu_demo/pipeline_matmul's tested two-buffer command groups
    # across output tiles. No automatic or RV pipelining is implied.
    steps = k // bk

    @T.prim_func
    def kernel(a: T.Tensor((m, k), dtype), b: T.Tensor((k, n), dtype), out: T.Tensor((m, n),
                                                                                     dtype)):
        with T.Kernel(n // bn, m // bm, is_cpu=True) as (x, y):
            a0 = T.alloc_shared((bm, bk), dtype)
            a1 = T.alloc_shared((bm, bk), dtype)
            b0 = T.alloc_shared((bk, bn), dtype)
            b1 = T.alloc_shared((bk, bn), dtype)
            acc = T.alloc_shared((bm, bn), "float32")
            result = T.alloc_shared((bm, bn), dtype)
            T.ppl_fill(acc, T.float32(0))
            T.ppl_copy(a[y * bm, 0], a0)
            T.ppl_copy(b[0, x * bn], b0)
            for j in T.serial(steps - 1):
                T.tpukernel_parallel_start()
                if j % 2 == 0:
                    T.ppl_gemm(a0, b0, acc, accumulate=True)
                    T.ppl_copy(a[y * bm, (j + 1) * bk], a1)
                    T.ppl_copy(b[(j + 1) * bk, x * bn], b1)
                else:
                    T.ppl_gemm(a1, b1, acc, accumulate=True)
                    T.ppl_copy(a[y * bm, (j + 1) * bk], a0)
                    T.ppl_copy(b[(j + 1) * bk, x * bn], b0)
                T.tpukernel_parallel_end()
            if (steps - 1) % 2 == 0:
                T.ppl_gemm(a0, b0, acc, accumulate=True)
            else:
                T.ppl_gemm(a1, b1, acc, accumulate=True)
            T.ppl_copy(acc, result)
            T.ppl_copy(result, out[y * bm, x * bn])

    return kernel


def build_matched_swiglu(rows, width, br, bw, dtype):
    # The SDK uses storage-dtype exp/add/div, then multiplies by up.
    # RV exposes exp only at FP32, so that combination is rejected explicitly.
    @T.prim_func
    def kernel(gate: T.Tensor((rows, width), dtype), up: T.Tensor((rows, width), dtype),
               out: T.Tensor((rows, width), dtype)):
        with T.Kernel(rows // br, width // bw, is_cpu=True) as (r, w):
            x = T.alloc_shared((br, bw), dtype)
            y = T.alloc_shared((br, bw), dtype)
            negative = T.alloc_shared((br, bw), dtype)
            denom = T.alloc_shared((br, bw), dtype)
            value = T.alloc_shared((br, bw), dtype)
            work0 = T.alloc_shared((br, bw), dtype)
            work1 = T.alloc_shared((br, bw), dtype)
            coeff = T.alloc_shared((64, 32), dtype)
            T.ppl_copy(gate[r * br, w * bw], x)
            T.ppl_copy(up[r * br, w * bw], y)
            T.ppl_mul_C(negative, x, -1.0)
            T.ppl_exp(negative, work0, work1, coeff)
            T.ppl_add_C(denom, negative, 1.0)
            T.ppl_div(value, x, denom)
            T.ppl_mul(value, value, y)
            T.ppl_copy(value, out[r * br, w * bw])

    return kernel


def build_matched_rmsnorm(rows, width, br, bw, dtype):
    step = min(bw, width // 2)

    @T.prim_func
    def kernel(x: T.Tensor((rows, width), dtype), weight: T.Tensor((rows, width), dtype),
               out: T.Tensor((rows, width), dtype)):
        with T.Kernel(rows // br, is_cpu=True) as (r,):
            value = T.alloc_shared((br, step), dtype)
            square = T.alloc_shared((br, step), dtype)
            chunk = T.alloc_shared((br, 1), dtype)
            total = T.alloc_shared((br, 1), dtype)
            inverse = T.alloc_shared((br, 1), dtype)
            fp = T.alloc_shared((br, 1), "float32")
            invfp = T.alloc_shared((br, 1), "float32")
            w = T.alloc_shared((br, step), dtype)
            weight_row = T.alloc_shared((1, step), dtype)
            weight_ones = T.alloc_shared((br, 1), dtype)
            y = T.alloc_shared((br, step), dtype)
            T.ppl_fill(total, 1e-5)
            if br > 1:
                T.ppl_fill(weight_ones, 1.0)
            for k in T.serial(width // step):
                T.ppl_copy(x[r * br, k * step], value)
                T.ppl_mul(square, value, value)
                T.ppl_reduce_sum(square, chunk, dim=1)
                T.ppl_mul_C(chunk, chunk, 1.0 / width)
                T.ppl_add(total, total, chunk)
            T.ppl_copy(total, fp)
            T.ppl_rsqrt(invfp, fp)
            T.ppl_copy(invfp, inverse)
            for k in T.serial(width // step):
                T.ppl_copy(x[r * br, k * step], value)
                T.ppl_mul(y, value, inverse)
                T.ppl_copy(weight[0, k * step], weight_row)
                # Public copy cannot address a nonzero local C lane. Express
                # row broadcast as an exact outer product of ones and gamma;
                # gamma still loads once, with no new normalization math.
                if br == 1:
                    T.ppl_copy(weight_row, w)
                else:
                    T.ppl_gemm(weight_ones, weight_row, w, accumulate=False)
                T.ppl_mul(y, y, w)
                T.ppl_copy(y, out[r * br, k * step])

    return kernel


def build_matched_rope(rows, width, br, bw, dtype):

    @T.prim_func
    def kernel(x: T.Tensor((rows, width), dtype), cos: T.Tensor((rows, width), dtype),
               sin: T.Tensor((rows, width), dtype), out: T.Tensor((rows, width), dtype)):
        with T.Kernel(rows // br, width // bw, is_cpu=True) as (r, w):
            xx = T.alloc_shared((br, bw), dtype)
            cc = T.alloc_shared((br, bw), dtype)
            ss = T.alloc_shared((br, bw), dtype)
            xc = T.alloc_shared((br, bw), dtype)
            xs = T.alloc_shared((br, bw), dtype)
            negative = T.alloc_shared((br, bw), dtype)
            ns = T.alloc_shared((br, bw), dtype)
            result = T.alloc_shared((br, bw), dtype)
            a = T.alloc_shared((br, 1), dtype)
            b = T.alloc_shared((br, 1), dtype)
            c = T.alloc_shared((br, 1), dtype)
            d = T.alloc_shared((br, 1), dtype)
            T.ppl_copy(x[r * br, w * bw], xx)
            T.ppl_copy(cos[r * br, w * bw], cc)
            T.ppl_copy(sin[r * br, w * bw], ss)
            T.ppl_mul(xc, xx, cc)
            T.ppl_mul(xs, xx, ss)
            T.ppl_mul_C(negative, xx, -1.0)
            T.ppl_mul(ns, negative, ss)
            for j in T.serial(bw // 2):
                T.ppl_copy(xc[0, j * 2], a)
                T.ppl_copy(ns[0, j * 2 + 1], b)
                T.ppl_add(a, a, b)
                T.ppl_copy(xc[0, j * 2 + 1], c)
                T.ppl_copy(xs[0, j * 2], d)
                T.ppl_add(c, c, d)
                T.ppl_copy(a, result[0, j * 2])
                T.ppl_copy(c, result[0, j * 2 + 1])
            T.ppl_copy(result, out[r * br, w * bw])

    return kernel


def build_matched_mlp(m, hidden, intermediate, br, bi, dtype):
    # One core; full hidden input resident; intermediate tile is never stored
    # globally. Match the casts in SDK mlp_multicore.pl, including sigmoid.
    @T.prim_func
    def kernel(x: T.Tensor((m, hidden), dtype), up: T.Tensor((hidden, intermediate), dtype),
               gate: T.Tensor((hidden, intermediate), dtype), down: T.Tensor(
                   (intermediate, hidden), dtype), out: T.Tensor((m, hidden), dtype)):
        with T.Kernel(m // br, is_cpu=True) as (r,):
            xx = T.alloc_shared((br, hidden), dtype)
            uw = T.alloc_shared((hidden, bi), dtype)
            gw = T.alloc_shared((hidden, bi), dtype)
            dw = T.alloc_shared((bi, hidden), dtype)
            g = T.alloc_shared((br, bi), dtype)
            u = T.alloc_shared((br, bi), dtype)
            negative = T.alloc_shared((br, bi), dtype)
            sigmoid = T.alloc_shared((br, bi), dtype)
            exp = T.alloc_shared((br, bi), "float32")
            denominator = T.alloc_shared((br, bi), "float32")
            one = T.alloc_shared((br, bi), "float32")
            sigfp = T.alloc_shared((br, bi), "float32")
            work0 = T.alloc_shared((br, bi), "float32")
            work1 = T.alloc_shared((br, bi), "float32")
            coeff = T.alloc_shared((64, 32), "float32")
            acc = T.alloc_shared((br, hidden), "float32")
            y = T.alloc_shared((br, hidden), dtype)
            T.ppl_copy(x[r * br, 0], xx)
            T.ppl_fill(acc, T.float32(0))
            T.ppl_fill(one, T.float32(1))
            for i in T.serial(intermediate // bi):
                T.ppl_copy(up[0, i * bi], uw)
                T.ppl_copy(gate[0, i * bi], gw)
                T.ppl_copy(down[i * bi, 0], dw)
                T.ppl_gemm(xx, gw, g, accumulate=False)
                T.ppl_mul_C(negative, g, -1.0)
                T.ppl_copy(negative, exp)
                T.ppl_exp(exp, work0, work1, coeff)
                T.ppl_add_C(denominator, exp, T.float32(1))
                T.ppl_div(sigfp, one, denominator)
                T.ppl_copy(sigfp, sigmoid)
                T.ppl_mul(g, g, sigmoid)
                T.ppl_gemm(xx, uw, u, accumulate=False)
                T.ppl_mul(g, g, u)
                T.ppl_gemm(g, dw, acc, accumulate=True)
            T.ppl_copy(acc, y)
            T.ppl_copy(y, out[r * br, 0])

    return kernel


def build_matched_attention(batch, seq, heads, d, bm, bn, dtype):
    # SDK flash_attention.pl: scale Q before QK; storage-dtype scores,
    # maxima, denominator, exponentials and PV; reciprocal then multiply.
    step = min(bn, seq // 2)

    @T.prim_func
    def kernel(q: T.Tensor((batch, seq, heads, d), dtype), k: T.Tensor(
        (batch, seq, heads, d), dtype), v: T.Tensor((batch, seq, heads, d), dtype), mask: T.Tensor(
            (seq, seq), dtype), out: T.Tensor((batch, seq, heads, d), dtype)):
        with T.Kernel(seq // bm, heads, batch, is_cpu=True) as (r, h, b):
            qq = T.alloc_shared((bm, d), dtype)
            kk = T.alloc_shared((step, d), dtype)
            vv = T.alloc_shared((step, d), dtype)
            score = T.alloc_shared((bm, step), dtype)
            mm = T.alloc_shared((bm, step), dtype)
            p = T.alloc_shared((bm, step), dtype)
            acc = T.alloc_shared((bm, d), dtype)
            pv = T.alloc_shared((bm, d), dtype)
            maximum = T.alloc_shared((bm, 1), dtype)
            current = T.alloc_shared((bm, 1), dtype)
            newmax = T.alloc_shared((bm, 1), dtype)
            alpha = T.alloc_shared((bm, 1), dtype)
            denominator = T.alloc_shared((bm, 1), dtype)
            partial = T.alloc_shared((bm, 1), dtype)
            inverse = T.alloc_shared((bm, 1), dtype)
            one = T.alloc_shared((bm, 1), dtype)
            work0 = T.alloc_shared((bm, step), dtype)
            work1 = T.alloc_shared((bm, step), dtype)
            rowwork0 = T.alloc_shared((bm, 1), dtype)
            rowwork1 = T.alloc_shared((bm, 1), dtype)
            coeff = T.alloc_shared((64, 32), dtype)
            T.ppl_copy(q[b:b + 1, r * bm:r * bm + bm, h:h + 1, 0:d], qq)
            T.ppl_mul_C(qq, qq, d**-0.5)
            T.ppl_fill(maximum, -15000.0)
            T.ppl_fill(denominator, 0.0)
            T.ppl_fill(acc, 0.0)
            T.ppl_fill(one, 1.0)
            for j in T.serial(seq // step):
                T.ppl_copy(k[b:b + 1, j * step:j * step + step, h:h + 1, 0:d], kk)
                T.ppl_copy(v[b:b + 1, j * step:j * step + step, h:h + 1, 0:d], vv)
                T.ppl_copy(mask[r * bm, j * step], mm)
                T.ppl_gemm(qq, kk, score, transpose_B=True, accumulate=False)
                T.ppl_add(score, score, mm)
                T.ppl_reduce_max(score, current, dim=1)
                T.ppl_max(newmax, maximum, current)
                T.ppl_subtract(alpha, maximum, newmax)
                T.ppl_copy(newmax, maximum)
                T.ppl_exp(alpha, rowwork0, rowwork1, coeff)
                T.ppl_mul(acc, acc, alpha)
                T.ppl_mul(denominator, denominator, alpha)
                T.ppl_subtract(p, score, newmax)
                T.ppl_exp(p, work0, work1, coeff)
                T.ppl_reduce_sum(p, partial, dim=1)
                T.ppl_add(denominator, denominator, partial)
                T.ppl_gemm(p, vv, pv, accumulate=False)
                T.ppl_add(acc, acc, pv)
            T.ppl_div(inverse, one, denominator)
            T.ppl_mul(acc, acc, inverse)
            T.ppl_copy(acc, out[b:b + 1, r * bm:r * bm + bm, h:h + 1, 0:d])

    return kernel


def build_matched_paged(blocks, qh, kvh, d, bn, dtype):
    ctx = blocks * 16
    rows = ctx * kvh
    staged = ctx * qh
    step = min(bn, ctx // 2)

    @T.prim_func
    def kernel(q: T.Tensor((qh, d), dtype), kc: T.Tensor((rows, d), dtype), vc: T.Tensor(
        (rows, d), dtype), ki: T.Tensor((staged, 1), "uint32"), vi: T.Tensor((staged, 1), "uint32"),
               ks: T.Tensor((staged, d), dtype), vs: T.Tensor((staged, d), dtype), out: T.Tensor(
                   (qh, d), dtype)):
        with T.Kernel(1, is_cpu=True):
            qq = T.alloc_shared((1, d), dtype)
            kk = T.alloc_shared((step, d), dtype)
            vv = T.alloc_shared((step, d), dtype)
            score = T.alloc_shared((1, step), dtype)
            p = T.alloc_shared((1, step), dtype)
            acc = T.alloc_shared((1, d), dtype)
            pv = T.alloc_shared((1, d), dtype)
            maximum = T.alloc_shared((1, 1), dtype)
            current = T.alloc_shared((1, 1), dtype)
            newmax = T.alloc_shared((1, 1), dtype)
            alpha = T.alloc_shared((1, 1), dtype)
            denominator = T.alloc_shared((1, 1), dtype)
            partial = T.alloc_shared((1, 1), dtype)
            inverse = T.alloc_shared((1, 1), dtype)
            one = T.alloc_shared((1, 1), dtype)
            work0 = T.alloc_shared((1, step), dtype)
            work1 = T.alloc_shared((1, step), dtype)
            rowwork0 = T.alloc_shared((1, 1), dtype)
            rowwork1 = T.alloc_shared((1, 1), dtype)
            coeff = T.alloc_shared((64, 32), dtype)
            T.ppl_embedding(ks, kc, ki)
            T.ppl_embedding(vs, vc, vi)
            for h in T.serial(qh):
                T.ppl_copy(q[h, 0], qq)
                T.ppl_mul_C(qq, qq, d**-0.5)
                T.ppl_fill(maximum, -15000.0)
                T.ppl_fill(denominator, 0.0)
                T.ppl_fill(acc, 0.0)
                T.ppl_fill(one, 1.0)
                for j in T.serial(ctx // step):
                    T.ppl_copy(ks[h * ctx + j * step, 0], kk)
                    T.ppl_copy(vs[h * ctx + j * step, 0], vv)
                    T.ppl_gemm(qq, kk, score, transpose_B=True, accumulate=False)
                    T.ppl_reduce_max(score, current, dim=1)
                    T.ppl_max(newmax, maximum, current)
                    T.ppl_subtract(alpha, maximum, newmax)
                    T.ppl_copy(newmax, maximum)
                    T.ppl_exp(alpha, rowwork0, rowwork1, coeff)
                    T.ppl_mul(acc, acc, alpha)
                    T.ppl_mul(denominator, denominator, alpha)
                    T.ppl_subtract(p, score, newmax)
                    T.ppl_exp(p, work0, work1, coeff)
                    T.ppl_reduce_sum(p, partial, dim=1)
                    T.ppl_add(denominator, denominator, partial)
                    T.ppl_gemm(p, vv, pv, accumulate=False)
                    T.ppl_add(acc, acc, pv)
                T.ppl_div(inverse, one, denominator)
                T.ppl_mul(acc, acc, inverse)
                T.ppl_copy(acc, out[h, 0])

    return kernel


def build_program(case, config, model, variant):
    if variant == 'tilelang-optimized':
        from tpu_benchmark.registry import build_program as build_optimized
        return build_optimized(case, config, model)
    op, shape, dtype = case["op"], case["shape"], case["dtype"]
    matched = variant == "tilelang-matched"
    if op == "pipeline-add":
        from tpu_demo.pipeline_add.pipeline_add import build_pipeline_add
        if model == "rv" and config.get("stages", 1) == 2:
            raise NotImplementedError("RV manual parallel command groups are not exposed")
        return build_pipeline_add(
            tiles=shape[1] // config["block_width"],
            tile_width=config["block_width"],
            dtype=dtype,
            pipelined=config.get("stages", 1) == 2)
    if op == "pipeline-matmul":
        from tpu_demo.pipeline_matmul.pipeline_matmul import build_pipelined_matmul
        if model == "rv":
            raise NotImplementedError(
                "pipeline_matmul uses TPU-Kernel-only parallel command groups")
        if dtype != "float16":
            return build_double_buffered_matmul(*shape, shape[0], shape[1], config["block_k"],
                                                dtype)
        return build_pipelined_matmul(m=shape[0], n=shape[1], k=shape[2], block_k=config["block_k"])
    if op.startswith("elementwise-"):
        return build_tiled_elementwise(
            op.split("-")[-1], *shape, config["block_rows"], config["block_width"], dtype)
    if op == "matmul":
        if config.get("stages", 1) == 2:
            if model != "tpukernel":
                raise NotImplementedError("RV command groups / software pipeline are not exposed")
            return build_double_buffered_matmul(*shape, config["block_m"], config["block_n"],
                                                config["block_k"], dtype)
        from tpu_demo.matmul import build_matmul
        return build_matmul(
            m=shape[0],
            n=shape[1],
            k=shape[2],
            dtype=dtype,
            programming_model=model,
            **{
                k: v for k, v in config.items() if k.startswith("block_")
            })
    if op in ("rmsnorm", "rmsnorm-splitk"):
        if matched:
            return build_matched_rmsnorm(*shape, config["block_rows"], config["block_width"], dtype)
        from tpu_demo.rmsnorm import build_rmsnorm, build_rmsnorm_splitk
        if op.endswith("splitk"):
            return build_rmsnorm_splitk(
                rows=shape[0],
                width=shape[1],
                dtype=dtype,
                epsilon=1e-5,
                block_rows=config["block_rows"],
                block_k=config["block_width"])
        return build_rmsnorm(
            rows=shape[0],
            width=shape[1],
            dtype=dtype,
            epsilon=1e-5,
            block_rows=config["block_rows"])
    if op == "swiglu":
        if matched:
            if model == "rv" and dtype != "float32":
                raise NotImplementedError(
                    "PPL storage-dtype exp sequence cannot be reproduced: TileLang RV exp accepts FP32 only"
                )
            return build_matched_swiglu(*shape, config["block_rows"], config["block_width"], dtype)
        from tpu_demo.swiglu import build_swiglu
        return build_swiglu(
            rows=shape[0],
            width=shape[1],
            dtype=dtype,
            block_rows=config["block_rows"],
            block_width=config["block_width"])
    if op == "rope":
        if shape[0] > 65535:
            from .large_shape_variants import build_split_global_rope, build_split_global_matched_rope
            if matched:
                return build_split_global_matched_rope(*shape, config["block_rows"],
                                                       config["block_width"], dtype)
            return build_split_global_rope(
                rows=shape[0],
                width=shape[1],
                block_rows=config["block_rows"],
                block_width=config["block_width"],
                dtype=dtype)
        if matched:
            return build_matched_rope(*shape, config["block_rows"], config["block_width"], dtype)
        from tpu_demo.rope import build_rope
        return build_rope(
            rows=shape[0],
            width=shape[1],
            dtype=dtype,
            block_rows=config["block_rows"],
            block_width=config["block_width"])
    if op == "flashattn":
        from tpu_demo.flashattn import build_flashattn
        if matched:
            if model == "rv" and dtype != "float32":
                raise NotImplementedError(
                    "exact PPL storage-dtype online-softmax needs FP16/BF16 exp; TileLang RV exposes FP32 exp only"
                )
            return build_matched_attention(*shape, config["block_m"], config["block_n"], dtype)
        return build_flashattn(
            batch=shape[0],
            sequence=shape[1],
            heads=shape[2],
            head_dim=shape[3],
            dtype=dtype,
            block_m=config["block_m"],
            block_n=config["block_n"])
    if op == "llama-mlp":
        if matched:
            return build_matched_mlp(*shape, config["block_rows"], config["block_i"], dtype)
        from tpu_demo.llama_mlp import build_llama_mlp
        return build_llama_mlp(
            batch_seq=shape[0],
            hidden=shape[1],
            intermediate=shape[2],
            dtype=dtype,
            programming_model=model,
            block_bs=config["block_rows"],
            block_h=config["block_h"],
            block_i=config["block_i"])
    if op == "paged-attention":
        context = shape[0] * 16
        staged_rows, cache_rows = shape[1] * context, shape[2] * context
        if max(staged_rows, cache_rows) > 65535:
            raise NotImplementedError(
                f"Public embedding requires whole 2D compiler-owned descriptors with axes <=65535; "
                f"this paged problem has staged_rows={staged_rows}, cache_rows={cache_rows}. "
                "Per-head descriptor aliases are rejected by current codegen; no equivalent gather tiling is exposed."
            )
        if matched:
            if model == "rv":
                raise NotImplementedError(
                    "PPL storage-dtype online paged softmax needs FP16/BF16 exp; TileLang RV exposes FP32 exp only"
                )
            return build_matched_paged(*shape, config["block_n"], dtype)
        from tpu_demo.paged_attention import build_paged_attention
        return build_paged_attention(
            num_blocks=shape[0],
            q_heads=shape[1],
            kv_heads=shape[2],
            head_dim=shape[3],
            dtype=dtype,
            programming_model=model)[0]
    if op == "gptq-w4a16":
        from tpu_demo.gptq_w4a16 import build_gptq_w4a16
        return build_gptq_w4a16(
            m=shape[0],
            n=shape[1],
            k=shape[2],
            block_m=min(shape[0], 32),
            block_n=64,
            dtype=dtype,
            programming_model=model)
    raise NotImplementedError(f"TileLang builder missing for {op}")
