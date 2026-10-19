"""Model-sized problems; tile dimensions are separate search variables."""
import math
import json
from pathlib import Path

OPERATOR_ORDER = (
    "matmul",
    "elementwise-add",
    "elementwise-sub",
    "elementwise-mul",
    "elementwise-div",
    "rmsnorm",
    "rmsnorm-splitk",
    "swiglu",
    "llama-mlp",
    "rope",
    "flashattn",
    "paged-attention",
    "gptq-w4a16",
    "pipeline-add",
    "pipeline-matmul",
)


def case_sort_key(case):
    op = case["op"]
    return (OPERATOR_ORDER.index(op) if op in OPERATOR_ORDER else len(OPERATOR_ORDER),
            0 if case["dtype"] == "float16" else 1, tuple(case["shape"]),
            bool(case.get("causal", False)), case.get("id", ""))


def default_cases():
    result = []

    def add(op, shape, model="Llama-3.1-8B", **extra):
        for dtype in ("float16", "bfloat16"):
            result.append(
                dict(
                    id=f"{op}-{'x'.join(map(str,shape))}-{dtype}" +
                    ('-causal' if extra.get('causal') else ''),
                    op=op,
                    shape=list(shape),
                    dtype=dtype,
                    source_model=model,
                    **extra))

    for m in (1, 32, 2048):
        for n, k in ((4096, 4096), (1024, 4096), (14336, 4096), (4096, 14336)):
            add("matmul", (m, n, k), phase="prefill" if m == 2048 else "decode")
        for op in ("elementwise-add", "elementwise-sub", "elementwise-mul", "elementwise-div",
                   "rmsnorm", "rmsnorm-splitk"):
            add(op, (m, 4096))
        add("swiglu", (m, 14336))
        add("llama-mlp", (m, 4096, 14336))
        # 32 query heads, 128-wide interleaved RoPE; flatten token/head rows.
        add("rope", (m * 32, 128))
    for seq in (128, 2048, 8192):
        for causal in (False, True):
            add("flashattn", (1, seq, 32, 128), causal=causal, attention_kind="MHA")
    for seq in (512, 4096, 16384):
        add("paged-attention", (seq // 16, 32, 8, 128),
            phase="decode",
            attention_kind="GQA",
            cache_update=False)
    for m in (1, 32, 2048):
        add("gptq-w4a16", (m, 4096, 4096), group_size=128)
    for w in (1024, 4096, 32768):
        add("pipeline-add", (1, w), model="mechanism", phase="pipeline diagnostic")
    for shape in ((32, 128, 4096), (32, 128, 14336), (64, 256, 4096)):
        add("pipeline-matmul", shape, model="projection tile", phase="pipeline diagnostic")
    return result


def cases():
    path = Path(
        __file__).resolve().parents[2] / "research/operator_comparison/configs/workloads.json"
    if not path.exists():
        return default_cases()
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported workload manifest version")
    rows = payload["cases"]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate workload IDs")
    return rows


def configs(case, implementation, model):
    op, s = case["op"], case["shape"]
    result = []
    stages = (1, 2) if implementation == "ppl" or (op in ("matmul", "pipeline-add") and
                                                   model == "tpukernel") else (1,)
    if implementation == 'tilelang-optimized':
        try:
            from tpu_benchmark.registry import candidate_configs
        except ModuleNotFoundError as error:
            if error.name == 'tpu_benchmark':
                return []
            raise
        result = candidate_configs(case, model)
    elif op == "pipeline-matmul":
        result = [dict(block_m=s[0], block_n=s[1], block_k=k, stages=2) for k in (128, 256, 512)]
    elif op == "matmul":
        for bm, bn, bk in ((32, 128, 256), (64, 256, 512), (64, 512, 1024), (32, 128, s[2]),
                           (64, 256, s[2]), (64, 512, s[2])):
            for stage in stages:
                result.append(dict(block_m=min(bm, s[0]), block_n=bn, block_k=bk, stages=stage))
    elif op == "flashattn":
        for bm, bn in ((16, 64), (32, 128), (64, 256)):
            if implementation == "tilelang" and (bn > s[1] or s[1] % bn or s[1] % bm):
                continue
            for stage in stages:
                result.append(dict(block_m=bm, block_n=bn, stages=stage))
    elif op == "llama-mlp":
        for bi, bh in ((128, s[1]), (256, s[1]), (512, s[1])):
            for stage in stages:
                result.append(dict(block_rows=min(s[0], 32), block_i=bi, block_h=bh, stages=stage))
    elif op == "paged-attention":
        # The public demo stages the entire KV context and exposes no KV tile
        # parameter. Do not label three identical programs as different tiles.
        widths = (s[0] * 16,) if implementation == "tilelang" else (128, 256, 512)
        result = [dict(block_m=1, block_n=bn, stages=1) for bn in widths]
    elif op == "gptq-w4a16":
        result = [dict(stages=1)]
    else:
        rows, width = s
        bws = (256, 1024, 2048) if width > 128 else (32, 64, 128)
        if op == "pipeline-add":
            bws = tuple(x for x in (128, 256, 512) if width // x >= 4)
        if op == "rmsnorm" and implementation == "tilelang":
            bws = (width,)
        row_tiles = (64,)
        if op == 'rope':
            row_tiles = (32, 128, 512, 2048, 4096, 8192)
        elif op.startswith('rmsnorm'):
            # The SDK example broadcasts gamma with a single lane broadcast.
            # Board canaries show that only the first 64 rows are initialized
            # when C>64 (both storage dtypes); the RV emulator misses this.
            row_tiles = (64,) if implementation == 'ppl' else (64, 128, 256, 512)
        for br in row_tiles:
            for bw in bws:
                for stage in stages:
                    result.append(dict(block_rows=min(rows, br), block_width=bw, stages=stage))
    # Unique configurations after clamping the row tile.
    seen = set()
    unique = []
    for c in result:
        key = tuple(sorted(c.items()))
        if key not in seen:
            c["estimated_local_bytes_per_lane"] = estimate_local(case, c, implementation)
            if c["estimated_local_bytes_per_lane"] <= int(0.8 * 262144):
                unique.append(c)
            seen.add(key)
    return unique


def estimate_local(case, c, implementation):
    # Conservative live-buffer estimate in the 64-lane TPU layout. Actual
    # compiler allocation is still checked; this is a filter, not a proof.
    def size(rows, width, bytes=2):
        eu = 64 // bytes
        return math.ceil(rows / 64) * math.ceil(width / eu) * eu * bytes

    op, s = case["op"], case["shape"]
    if implementation == 'tilelang-optimized' and op.startswith('rmsnorm'):
        rows, width = c['block_rows'], s[1]
        extra = size(1, width) + size(rows, 1) if c.get('expression') == 'shared_gamma' else 0
        return 2 * size(rows, width, 4) + 2 * size(rows, width) + 3 * size(rows, 1, 4) + 128 + extra
    if implementation == 'tilelang-optimized' and op == 'rope' and c.get('expression') == 'pair_channels':
        return 12 * size(c['block_rows'] * s[1] // 2, 2, 4) + 128
    if op == 'rope':
        r, w = c['block_rows'], c['block_width']
        if implementation == 'ppl':
            return (9 + 3 * (c['stages'] == 2)) * size(r, w) + 128
        if implementation == 'tilelang-matched':
            return 8 * size(r, w) + 4 * size(r, 1) + 128
        return (2 * size(r, w) + 2 * size(r, w // 2, 4) +
                2 * size(r, 1) + 8 * size(r, 1, 4) + 128)
    if op.startswith('rmsnorm'):
        r, w = c['block_rows'], c['block_width']
        if implementation == 'ppl':
            return ((6 + 3 * (c['stages'] == 2)) * size(r, w) + size(1, w) +
                    8 * size(r, 1, 4) + 128)
        if implementation == 'tilelang-matched':
            return 6 * size(r, w) + size(1, w) + 8 * size(r, 1, 4) + 128
        return 3 * size(r, w) + 3 * size(r, w, 4) + 4 * size(r, 1, 4) + 128
    if op in ("matmul", "pipeline-matmul"):
        m, n, k = c["block_m"], c["block_n"], c["block_k"]
        return c["stages"] * (size(m, k) + size(k, n)) + size(m, n, 4) + size(m, n)
    if op == "flashattn":
        m, n, d = c["block_m"], c["block_n"], s[3]
        return 5 * size(m, n, 4) + 5 * size(m, d, 4) + 2 * size(n, d) + 12 * size(m, 1, 4) + 128
    if op == "llama-mlp":
        r, i, h = c["block_rows"], c["block_i"], s[1]
        return size(r, h) * 3 + size(h, i) * 2 + size(i, h) + size(r, i, 4) * 10 + 128
    if "block_width" in c:
        return size(c["block_rows"], c["block_width"], 4) * (14 if op == "swiglu" else 9) + 128
    return 0


def clean_config(config):
    return {k: v for k, v in config.items() if not k.startswith("estimated_")}


def effective_config(case, config, implementation):
    """Actual loop tiles after source-level clamping in the PPL examples."""
    effective = clean_config(config)
    if case['op'] == 'rope' and config.get('expression') == 'pair_channels':
        effective['local_c'] = config['block_rows'] * (case['shape'][1] // 2)
        effective['local_w'] = 2
        effective['layout'] = 'pairs_on_C'
    if case["op"] == "flashattn":
        requested = effective["block_n"]
        if implementation in ('ppl', 'tilelang-matched'):
            effective["block_n"] = min(requested, case["shape"][1] // 2)
        # PPL's max-shape tensors retain the requested capacity even when
        # real-shape iteration is clamped. The matched port allocates step.
        effective[
            "block_n_capacity"] = requested if implementation == "ppl" else effective["block_n"]
    if case["op"] == "paged-attention":
        context = case["shape"][0] * 16
        effective["block_n"] = context if implementation == "tilelang" else min(
            effective["block_n"], context // 2)
        effective["block_n_capacity"] = config["block_n"] if implementation == "ppl" else effective[
            "block_n"]
    return effective


def validation_seeds(case):
    # Full shapes are always checked. Attention retains both a random fixture
    # and an adversarial fixture for online-softmax maximum updates.
    if case["op"] == "matmul" and case["shape"][0] >= 2048:
        return [0]
    if case["op"] == "llama-mlp" and case["shape"][0] >= 32:
        return [0]
    return [0, 17]


def watchdog_budget(case, base):
    if case["op"] == "flashattn" and case["shape"][1] >= 8192:
        return max(base, 3600)
    if case["op"] == "llama-mlp" and case["shape"][0] >= 2048:
        return max(base, 1800)
    return base
