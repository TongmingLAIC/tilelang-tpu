# TPU Operator Examples

This directory contains user-facing TileLang examples for SOPHGO TPUs. Every
operator package exports a `build_*` function for constructing a TileLang
program and a `run` function that executes one numerical comparison against a
PyTorch reference. Low-level instruction tests live under
`testing/python/jit/`.

See [OP_MAPPING.md](./OP_MAPPING.md) for the complete public-op mapping,
instruction selection, dtype matrix, and backend-specific restrictions.

## Operator support

| Operator | Default workload | FP16 | BF16 | FP32 | E4M3 | E5M2 | TPU-Kernel | RV Tensor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Add | Elementwise `(4, 32)` | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Subtract | Elementwise `(4, 32)` | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Multiply | Elementwise `(4, 32)` | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Divide | Elementwise `(4, 32)` | Yes | Yes | Yes | No | No | Yes | Yes |
| Matmul | `32 x 32` by `32 x 32`, tiled `16 x 16 x 16` | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| RMSNorm | Weighted normalization on `(8, 64)` | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Split-K RMSNorm | `(8, 128)`, split into width-32 tiles | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| RoPE | `(8, 32)`, interleaved even/odd elements | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| SwiGLU | `(8, 32)` | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| FlashAttention | BSHD `(1, 32, 1, 16)`, causal and non-causal | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

FP8 division is excluded because direct TPU-Kernel and RV instruction probes
do not produce a valid numerical result. Operators that require division,
exponential, reciprocal square root, or sum reduction convert their FP8
inputs to FP32, perform those independent primitive operations, and convert
the result back. The examples do not use fused sigmoid or fused RoPE ops.

The validated execution combinations are:

| Target | TPU-Kernel CModel | TPU-Kernel PCIe | RV Tensor CModel | RV Tensor PCIe |
| --- | --- | --- | --- | --- |
| BM1690 | Yes | Not verified on this host | Not available | Not available |
| SG2260E | Yes | Yes | Yes | Yes |

## Kernel selection

A frontend kernel is shared when the semantic expression is identical and
only the public dtype or target instruction selection changes. A separate
kernel is used when storage layout, intermediate dtype, or operation sequence
must differ. The result payload records the selected name in
`parameters.kernel_variant`.

| Operator | FP16/BF16/FP8 kernel | FP32 kernel | Important detail |
| --- | --- | --- | --- |
| Add | `elementwise_add` | `elementwise_add` | Direct same-dtype instruction |
| Subtract | `elementwise_sub` | `elementwise_sub` | Direct same-dtype instruction |
| Multiply | `elementwise_mul` | `elementwise_mul` | Direct same-dtype instruction |
| Divide | `elementwise_div` for FP16/BF16 | `elementwise_div` | FP8 is unsupported |
| Matmul | `matmul_low_precision` | `matmul_fp32` | FP32 uses native `local.matrix`; FP8 accumulates in FP32 |
| RMSNorm | `rmsnorm_low_precision` | `rmsnorm_fp32` | Low-precision public tensors use FP32 normalization intermediates |
| Split-K RMSNorm | `rmsnorm_splitk_low_precision` | `rmsnorm_splitk_fp32` | Two serial passes over width tiles |
| RoPE | `rope` | `rope` | One primitive expression; FP32 trigonometric tables and arithmetic |
| SwiGLU | `swiglu_low_precision` | `swiglu_fp32` | Sigmoid is expressed as exp, add, divide, and multiply |
| FlashAttention | `flashattn_low_precision` | `flashattn_fp32` | FP32 public tensors use BF16 matrix operands for NT GEMM |

TPU-Kernel and RV Tensor share these frontend kernels. Their instruction
selection remains independent in backend code generation.

## Builder functions

The operator packages export:

- `tpu_demo.elementwise.build_elementwise`
- `tpu_demo.matmul.build_matmul`
- `tpu_demo.rmsnorm.build_rmsnorm`
- `tpu_demo.rmsnorm.build_rmsnorm_splitk`
- `tpu_demo.rope.build_rope`
- `tpu_demo.swiglu.build_swiglu`
- `tpu_demo.flashattn.build_flashattn`

For example:

```python
from tpu_demo.matmul import build_matmul

program = build_matmul(
    m=32,
    n=32,
    k=32,
    block_m=16,
    block_n=16,
    block_k=16,
    dtype="float16",
    programming_model="rv",
)
```

Builders only create TileLang programs. `tpu_demo.run.run_case` owns input
generation, compilation, execution, and numerical comparison. The
`tpu_demo.cases` registry is dependency-free and can be imported without
loading a TPU runtime.

## List and run examples

Run commands from the repository root after completing the project
installation.

List every case:

```bash
python testing/python/jit/tpu_demo_ops_matrix.py --list-cases
```

List RV Tensor cases:

```bash
python testing/python/jit/tpu_demo_ops_matrix.py \
  --list-cases \
  --programming-model rv
```

Run one SG2260E RV CModel case:

```bash
python -m tpu_demo.run \
  --case matmul.float16 \
  --chip sg2260e \
  --programming-model rv \
  --runtime-mode cmodel
```

`python -m tpu_demo.run` is intentionally CModel-only. Use the supervised
matrix runner for PCIe so board access remains serialized and health-checked.

## Numerical contract

Each case compiles and launches one kernel, then checks:

1. Output shape and dtype.
2. Finite output values where the mathematical result should be finite.
3. Elementwise error with the operator/dtype tolerances in
   `tpu_demo/common.py`.

Current example constraints are:

- Dimensions are positive compile-time integers.
- Tiled dimensions divide exactly; tail handling is not implemented.
- Native FP32 matmul uses `local.matrix` on both programming models.
- FP16/BF16/FP8 matmul uses same-dtype matrix inputs and FP32 accumulation.
- FlashAttention converts public FP32 Q/K/V to BF16 because its first matrix
  product needs the lower-precision NT form. Its softmax state remains FP32.
- Low-precision RMSNorm, RoPE, and SwiGLU use explicit FP32 intermediates.
- FlashAttention takes an FP32 additive mask and tests causal and non-causal
  execution. The `descending-max` and `weighted-keys` inputs expose online
  softmax state-update errors; they are validation variants, not new modes.

## Unit tests

After changing an example or registry, run:

```bash
python -m pytest -q \
  testing/python/jit/test_tpu_demo_contract.py \
  testing/python/jit/test_tpu_demo_ops_matrix.py
```

`test_tpu_demo_contract.py` checks builders, shapes, dtypes, comparison rules,
and invalid combinations. `test_tpu_demo_ops_matrix.py` checks matrix
selection, promotion order, and PCIe safety gates. The actual CModel/PCIe
runner is `tpu_demo_ops_matrix.py`.

## CModel and PCIe validation

Use a fresh output directory for every run. Validate BM1690 CModel first and
then SG2260E CModel:

```bash
export TILELANG_TPU_DEMO_RUN="research/artifacts/demo-run"

python testing/python/jit/tpu_demo_ops_matrix.py \
  --runtime-mode cmodel \
  --chip bm1690 \
  --programming-model tpukernel \
  --output-dir "${TILELANG_TPU_DEMO_RUN}/bm1690-cmodel"

python testing/python/jit/tpu_demo_ops_matrix.py \
  --runtime-mode cmodel \
  --chip sg2260e \
  --output-dir "${TILELANG_TPU_DEMO_RUN}/sg2260e-cmodel"
```

The SG2260E command runs TPU-Kernel and RV Tensor serially. Start PCIe only
after both CModel stages pass:

```bash
python testing/python/jit/tpu_demo_ops_matrix.py \
  --runtime-mode pcie \
  --chip sg2260e \
  --device-id 0 \
  --allow-pcie \
  --allow-pcie-profile \
  --all-pcie-cases \
  --bm-cmodel-summary "${TILELANG_TPU_DEMO_RUN}/bm1690-cmodel/summary.json" \
  --sg-cmodel-summary "${TILELANG_TPU_DEMO_RUN}/sg2260e-cmodel/summary.json" \
  --output-dir "${TILELANG_TPU_DEMO_RUN}/sg2260e-pcie"
```

The PCIe runner takes an exclusive device lock, checks board state before and
after launches, and stops on the first compile, runtime, numerical, profiling,
or health error. Raw instruction records do not require an optional decoder.
Decoded per-instruction timing can be made mandatory with:

```bash
--require-decoded-timing \
--pcie-decoder-python /path/to/decoder/python \
--pcie-decoder-pythonpath /path/to/decoder/package
```

One profiled execution is conformance evidence, not a performance benchmark.
