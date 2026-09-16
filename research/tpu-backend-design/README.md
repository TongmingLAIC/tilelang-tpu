# TileLang-TPU Backend Design

This document summarizes the maintained compiler architecture for BM1690 and
SG2260E. Public operation and dtype details are defined in
[the operator mapping](../../tpu_demo/OP_MAPPING.md); runnable model operators
and validation commands are documented in
[the TPU demo guide](../../tpu_demo/README.md).

## Target model

TPU compilation makes three explicit choices:

| Axis | Values | Responsibility |
| --- | --- | --- |
| Chip | `bm1690`, `sg2260e` | Architecture, physical core count, and chip features |
| Programming model | `tpukernel`, `rv` | Device ABI, descriptors, instructions, and kernel lifecycle |
| Runtime mode | `cmodel`, `pcie` | Simulator or physical-board execution |

Supported target pairs are BM1690 with TPU-Kernel, SG2260E with TPU-Kernel,
and SG2260E with RV Tensor. Runtime mode does not change instruction
selection.

```python
kernel = tilelang.compile(
    program,
    target="tpu -mcpu=sg2260e -tpu-programming-model=rv",
    runtime_mode="cmodel",
)
```

## Semantic boundary

Portable public helpers emit a backend-neutral `tl.tpu.*` ABI:

- Copy, fill, and GEMM
- Tensor add, subtract, multiply, divide, and maximum
- Scalar add and multiply
- Exp and reciprocal square root
- Sum and maximum reductions
- Embedding row lookup

Target-bound code generation selects TPU-Kernel or RV Tensor instructions.
Backend-specific `tl.tpukernel.*` semantics are reserved for gather and top-k,
which have no portable RV contract. Raw `rvt_*` calls remain an expert
interface and cannot be mixed with compiler-managed semantics in one kernel.

RMSNorm, split-K RMSNorm, RoPE, SwiGLU, and FlashAttention are expressed in
`tpu_demo/` as compositions of the independent public primitives. There is no
backend semantic for a fused sigmoid or fused RoPE update.

## Compilation pipeline

```text
TileLang frontend
       |
       v
typed tl.tpu.* / tl.tpukernel.* semantics
       |
       v
target and residual-IR validation
       |
       v
frontend lowering and conservative TPU passes
       |
       v
TPUv7 local-memory address assignment
       |
       v
target.build.tilelang_tpu
       |
       +-- TPU-Kernel instruction selector
       |
       `-- RV Tensor instruction selector
       |
       v
CModel or PCIe runtime
```

The semantic registry is closed. Adding an operation requires a public helper,
typed region arguments, memory effects, target validation, both applicable
instruction selectors, and positive/negative tests. Unknown externs and
residual vector operations fail before native compilation.

## Tensor descriptors and memory effects

Global kernel parameters and compiler-owned local allocations receive
canonical descriptors containing shape, stride, address, dtype, and layout
information. Public operations accept ranks and layouts that their instruction
mapping can represent; shape-changing aliases, unbounded regions, and
unsupported partial local-channel views are rejected.

Address assignment uses explicit effects:

- Copy reads the source and writes the destination.
- Fill writes the destination.
- GEMM reads A/B and writes C, or reads and writes C when accumulating.
- Binary and scalar math write the output and read their inputs.
- Reduction input/scratch storage is conservative because TPU-Kernel may
  initialize physically padded elements.
- Exp workspaces are conservative across its multi-instruction sequence.
- Embedding/gather and top-k have explicit value/index roles.

Loop-carried allocations remain live across a loop back edge. The allocator
does not rely on operation names to guess effects.

## Instruction selection

The two selectors share frontend semantics but not descriptors or device
calls. TPU-Kernel uses GDMA/BDC/HAU APIs. RV Tensor builds CR/TR/GR descriptors,
configures instruction state, and emits RV commands. Neither selector falls
back to the other programming model.

The authoritative mapping tables cover:

- Floating and integer copies, local float conversion, and FP32 matrix copies
- FP8/base floating fill and arithmetic capability
- GEMM dtype, output, layout, transpose, and accumulation combinations
- Exp, rsqrt, reduction, embedding, gather, and top-k restrictions

See [OP_MAPPING.md](../../tpu_demo/OP_MAPPING.md) rather than duplicating those
tables here.

## Runtime and validation

CModel and PCIe use the same compile target. Validation proceeds serially:

1. BM1690 TPU-Kernel CModel.
2. SG2260E TPU-Kernel and RV Tensor CModel.
3. SG2260E TPU-Kernel and RV Tensor PCIe.

The PCIe runners require matching successful CModel summaries, take an
exclusive device lock, and check that the board returns to `Active` and idle.
Every numerical worker performs one compile and one launch in an isolated
process. Profiling records are conformance diagnostics, not performance
benchmarks.

## Current boundaries

- Demo tiles require positive static dimensions and exact divisibility.
- Public binary broadcast is limited to an equal RHS or `(M, 1)` W broadcast.
- Reductions are rank-2, overwrite their output, and support only `dim=1`.
- FP32 matrix operands require `local.matrix` and widths divisible by 16.
- Kernels currently use one logical core; multicore partitioning is separate
  future work.
- Asynchronous software-pipeline scheduling remains disabled until TPU command
  dependencies and hazards have a complete compiler model.

New capabilities should state their frontend semantics, supported targets and
dtypes, instruction selection, failure behavior, and CModel result before
physical-board validation.
