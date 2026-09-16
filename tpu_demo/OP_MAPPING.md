# TPU Operator-to-Instruction Mapping

This document defines the public `T.ppl_*` operation contract for the
TPU-Kernel and RV Tensor programming models. The support entries below are
based on numerical CModel validation on BM1690 and SG2260E, followed by
SG2260E PCIe validation where that programming model is available.

The floating-point names used here are E4M3 (`e4m3_float8`), E5M2
(`e5m2_float8`), FP16 (`float16`), BF16 (`bfloat16`), and FP32 (`float32`).

## Data movement and initialization

| TileLang op | TPU-Kernel instruction | RV Tensor instruction | Supported dtypes |
| --- | --- | --- | --- |
| `ppl_copy` | `tpu_gdma_cpy_S2L`, `tpu_gdma_cpy_L2S`, `tpu_gdma_cpy_S2S`, or `tpu_bdc_cpy` | `rvt_dma_ld`, `rvt_dma_st`, or `rvt_dma_cp` | Same-dtype E4M3, E5M2, FP16, BF16, FP32, INT8/16/32, and UINT8/16/32 |
| `ppl_copy` with local float conversion | `tpu_bdc_cast` | `rvt_cvt_f2f` | Pairwise FP16/BF16/FP32; E4M3 or E5M2 to/from FP16, BF16, or FP32 |
| `ppl_copy` with FP32 matrix layout | `tpu_gdma_matrix_S2L` or `tpu_gdma_matrix_L2S` | Matrix descriptors plus `rvt_dma_ld` or `rvt_dma_st` | FP32 |
| `ppl_fill` | `tpu_bdc_set_C` | `rvt_cr` plus `rvt_cp` | E4M3, E5M2, FP16, BF16, FP32 |

Copy conversion is local-to-local only. Global DMA preserves dtype. A direct
E4M3-to-E5M2 conversion is not part of the public contract. FP32
`local.matrix` transfers must cover a complete rank-2 matrix, have a width
divisible by 16, and cross the global/local boundary.

## Elementwise and scalar arithmetic

All operands and outputs use the same dtype. Binary operations accept equal
rank-2 shapes and the portable RHS broadcast form `(M, 1)` to `(M, W)`.

| TileLang op | TPU-Kernel instruction | RV Tensor instruction | TPU-Kernel dtypes | RV Tensor dtypes |
| --- | --- | --- | --- | --- |
| `ppl_add` | `tpu_bdc_fp_add` | `rvt_fadd` | E4M3, E5M2, FP16, BF16, FP32 | E4M3, E5M2, FP16, BF16, FP32 |
| `ppl_subtract` | `tpu_bdc_fp_sub` | `rvt_fsub` | E4M3, E5M2, FP16, BF16, FP32 | E4M3, E5M2, FP16, BF16, FP32 |
| `ppl_mul` | `tpu_bdc_fp_mul` | `rvt_fmul` | E4M3, E5M2, FP16, BF16, FP32 | E4M3, E5M2, FP16, BF16, FP32 |
| `ppl_div` | `tpu_bdc_fp_div` | `rvt_fdiv` | FP16, BF16, FP32 | FP16, BF16, FP32 |
| `ppl_max` | `tpu_bdc_max` | `rvt_fmax` | E4M3, E5M2, FP16, BF16, FP32 | E4M3, E5M2, FP16, BF16, FP32 |
| `ppl_add_C` | `tpu_bdc_fp_add_C` | Constant register plus `rvt_fadd` | E4M3, E5M2, FP16, BF16, FP32 | E4M3, E5M2, FP16, BF16, FP32 |
| `ppl_mul_C` | `tpu_bdc_fp_mul_C` | Constant register plus `rvt_fmul` | E4M3, E5M2, FP16, BF16, FP32 | E4M3, E5M2, FP16, BF16, FP32 |

FP8 division is deliberately rejected. Direct instruction probes either fail
or return incorrect values, so FP8 programs must convert to FP16, BF16, or
FP32 before division and convert back afterward.

## Matrix multiplication

`ppl_gemm` requires matching A/B dtypes and an explicit `accumulate` value.

| A/B dtype | C dtype | Form | TPU-Kernel | RV Tensor |
| --- | --- | --- | --- | --- |
| FP32 | FP32 | NN overwrite/accumulate | `tpu_bdc_fp32_mm` | `rvt_fmm_nn` / `rvt_fmma_nn` |
| FP32 | FP32 | TN overwrite/accumulate | `tpu_bdc_fp32_mm_L_trans` | `rvt_fmm_tn` / `rvt_fmma_tn` |
| FP16 or BF16 | Same dtype or FP32 | NN overwrite | `tpu_bdc_fp_mm` | `rvt_fmm2_nn` |
| FP16 or BF16 | FP32 | NN accumulate | `tpu_bdc_fp_mm` with `result_add` | `rvt_fmm2a_nn` |
| FP16 or BF16 | Same dtype or FP32 | NT overwrite | `tpu_bdc_fp_mm_R_trans` | `rvt_fmm2_nt` |
| FP16 or BF16 | FP32 | NT accumulate | Unsupported | `rvt_fmm2a_nt` |
| E4M3 or E5M2 | FP32 | NN overwrite/accumulate | `tpu_bdc_fp8_mm` | `rvt_fmm2_nn` / `rvt_fmm2a_nn` |
| E4M3 or E5M2 | FP32 | NT overwrite/accumulate | `tpu_bdc_fp8_mm_R_trans` | `rvt_fmm2_nt` / `rvt_fmm2a_nt` |

FP32 operands use `local.matrix`; lower-precision operands use the regular
local layout. FP32 right transpose and lower-precision left transpose are not
exposed by the corresponding instruction families. TPU-Kernel's FP16/BF16
right-transpose instruction has no accumulation flag, so that single form is
rejected instead of silently overwriting C. Native FP32 input and output are
supported directly; no automatic down-conversion is required.

## Extended math and reductions

| TileLang op | TPU-Kernel mapping | RV Tensor mapping | TPU-Kernel dtypes | RV Tensor dtypes |
| --- | --- | --- | --- | --- |
| `ppl_exp` | Coefficient load plus `tpu_bdc_fp_exp` | Range reduction and polynomial composed from RV arithmetic/conversion instructions | FP16, BF16, FP32 | FP32 |
| `ppl_rsqrt` | `tpu_bdc_fp_rsqrt` | `rvt_sfu_rsqrt` | FP16, BF16, FP32 | FP16, BF16, FP32 |
| `ppl_reduce_sum` | Zero padding plus staged `tpu_bdc_fp_avg_pool2d` | Sequential column slices plus `rvt_fadd` | FP16, BF16, FP32 | E4M3, E5M2, FP16, BF16, FP32 |
| `ppl_reduce_max` | Negative-maximum padding plus staged `tpu_bdc_fp_max_pool2d` | Sequential column slices plus `rvt_fmax` | E4M3, E5M2, FP16, BF16, FP32 | E4M3, E5M2, FP16, BF16, FP32 |

Both reductions currently accept rank-2 input and output and reduce only
`dim=1`. RV FP8 sum uses same-format additions, so every accumulation step is
rounded to the selected FP8 format. TPU-Kernel FP8 sum, FP8 exp, and FP8 rsqrt
are rejected because direct CModel probes do not produce a valid result.

## Row lookup and sorting

| TileLang op | TPU-Kernel mapping | RV Tensor mapping | Supported dtypes |
| --- | --- | --- | --- |
| `ppl_embedding` | `tpu_gdma_h_gather_S2S` | `rvt_dma_hgather` | E4M3, E5M2, FP16, BF16, FP32 payload; UINT32 index |
| `ppl_gather` | `tpu_gdma_h_gather_S2S` | Not exposed; use `ppl_embedding` | E4M3, E5M2, FP16, BF16, FP32 payload; UINT32 index |
| `ppl_topk` | `tpu_hau_sort_natural_index` on BM1690 | Not exposed | FP32, INT32, or UINT32 values; INT32 index output |

`ppl_gather` and `ppl_topk` are explicit TPU-Kernel operations.
`ppl_embedding` is the portable row-lookup semantic. The current SG2260E
runtime rejects `tpu_hau_sort_natural_index`, so `ppl_topk` is limited to
BM1690 and is diagnosed during target lowering.

## Llama operator composition

The examples intentionally express model operators using the independent
primitives above:

| Demo operator | Primitive composition |
| --- | --- |
| RMSNorm | copy/cast, multiply, reduce-sum, scalar multiply/add, rsqrt, multiply |
| RoPE | copy/cast, multiply, subtract, add |
| SwiGLU | scalar multiply, exp, fill, add, divide, multiply |
| FlashAttention | GEMM, scalar multiply, add/subtract/max, exp, reduce-max/sum, divide, copy/cast |

There is no public fused `ppl_sigmoid` or `ppl_rope_add` operation. This keeps
the frontend semantics independent of one backend's composite implementation
and makes each instruction mapping independently testable.
