"""Compile-only evidence for demo-wide unsupported primitives."""
import argparse
import json
from pathlib import Path
import traceback


def topk_probe():
    import tilelang.language as T

    @T.prim_func
    def kernel():
        with T.Kernel(1, is_cpu=True):
            source = T.alloc_shared((1, 32), "float32")
            values = T.alloc_shared((1, 2), "float32")
            indices = T.alloc_shared((1, 2), "int32")
            T.ppl_fill(source, T.float32(1))
            T.ppl_topk(values, indices, source, 2, True, 32)

    return kernel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import tilelang
    records = []
    for model in ("rv", "tpukernel"):
        try:
            artifact = tilelang.lower(
                topk_probe(),
                target=f"tpu -mcpu=bm1690e -tpu-programming-model={model}",
                runtime_mode="cmodel")
            (args.output / f"moe-topk-{model}.c").write_text(artifact.kernel_source)
            records.append(
                dict(operator="moe", model=model, status="lowered", primitive="ppl_topk"))
        except Exception as error:
            path = args.output / f"moe-topk-{model}.txt"
            path.write_text(traceback.format_exc())
            records.append(
                dict(
                    operator="moe",
                    model=model,
                    status="unsupported",
                    primitive="ppl_topk",
                    reason=str(error),
                    evidence=str(path)))
    (args.output / "capabilities.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records, ensure_ascii=False))


if __name__ == "__main__":
    main()
