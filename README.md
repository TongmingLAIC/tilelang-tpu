# TileLang-TPU

TileLang-TPU is a TPU-oriented extension of
[TileLang](https://github.com/tile-ai/tilelang) for SOPHGO accelerators. It
preserves the TileLang Python DSL while adding TPU-specific lowering, source
generation, JIT compilation, profiling, and runtime integration.

The project turns TileLang programs into standalone TPU kernels. It uses the
compiler and runtime components distributed with SOPHGO PPL, but it is not a
wrapper around the PPL programming framework.

## Highlights

- Explicit chip and programming-model selection for BM1690 and SG2260E
- TPU-Kernel support on both chips and RV Tensor support on SG2260E
- A single PPL 1.7 toolchain layout for compilation, CModel, and PCIe execution
- TPU-specific TileLang intrinsics for data movement, matrix multiplication,
  elementwise operations, reductions, and common neural-network operations
- End-to-end JIT compilation with generated host and device code
- CModel and PCIe profiling based on the PPL `--profiling` compilation path
- Operator examples for elementwise arithmetic, matmul, RMSNorm, RoPE, SwiGLU,
  and FlashAttention

## Supported Targets

| Chip | Physical cores | Programming models | Runtime modes |
| --- | ---: | --- | --- |
| BM1690 | 8 | TPU-Kernel | CModel |
| SG2260E | 4 | TPU-Kernel, RV Tensor | CModel, PCIe |

The chip, programming model, and runtime mode are independent choices with
different responsibilities:

- `-mcpu` selects the hardware architecture, compiler definitions, and core
  count.
- `-tpu-programming-model` selects the device instruction interface.
- `runtime_mode` selects simulation through CModel or execution on a PCIe
  device.

The available target combinations are BM1690 with TPU-Kernel, SG2260E with
TPU-Kernel, and SG2260E with RV Tensor. A bare `target="tpu"`, an incomplete
target, or BM1690 with RV Tensor is rejected before code generation.

## Requirements

- Linux x86_64 and Python 3.8 or later
- CMake 3.26 or later and a C++17 compiler
- A complete SOPHGO PPL 1.7 SDK
- A TPUv7 driver and runtime installation for SG2260E PCIe execution

Only the PPL 1.7 `deps/` release layout is supported. Older PPL directory
layouts and environment scripts are not part of this toolchain.

## Quick Start

Initialize the bundled TVM dependency and create a Python environment:

```bash
git submodule update --init --recursive 3rdparty/tvm

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt "cmake>=3.26"
```

Point the project at a PPL 1.7 SDK and build the native libraries:

```bash
export PPL_PROJECT_ROOT=/absolute/path/to/ppl-1.7-sdk
./build_tpu.sh

export TILELANG_TPU_SOURCE="$(pwd)"
export PYTHONPATH="${TILELANG_TPU_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
```

Run an SG2260E RV Tensor kernel with CModel:

```bash
python -m tpu_demo.run \
  --case matmul.float16 \
  --chip sg2260e \
  --programming-model rv \
  --runtime-mode cmodel
```

See the [installation guide](./docs/get_started/Installation.md) for system
packages, PPL SDK checks, non-standard build directories, and PCIe setup.

## Programming Model

TileLang-TPU keeps the standard TileLang workflow and makes the TPU target
explicit at the compilation boundary:

```python
import tilelang

kernel = tilelang.compile(
    program,
    out_idx=-1,
    target="tpu -mcpu=sg2260e -tpu-programming-model=tpukernel",
    runtime_mode="cmodel",
)
```

The same TileLang program can select RV Tensor when every operation used by the
kernel has an RV mapping:

```python
kernel = tilelang.compile(
    program,
    out_idx=-1,
    target="tpu -mcpu=sg2260e -tpu-programming-model=rv",
    runtime_mode="cmodel",
)
```

The compiler keeps target-independent TPU semantics separate from instruction
selection:

```text
TileLang Python DSL
        |
        v
TPU semantic checks and lowering passes
        |
        v
codegen_tpu (shared TPU source generation and dispatch)
        |
        +---- codegen_tpukernel (TPU-Kernel instruction selection)
        |
        `---- codegen_rv        (RV Tensor instruction selection)
        |
        v
PPL 1.7 compiler and CModel or PCIe runtime
```

`codegen_tpu` owns the common source generator; `codegen_tpukernel` and
`codegen_rv` implement the two instruction interfaces. This keeps shared
lowering rules in one place while making hardware-specific mappings explicit.

## TPU Operations

The TileLang DSL exposes portable TPU operations through the `T.ppl_*`
namespace. The compiler maps the following core operations to either
TPU-Kernel or RV Tensor according to the selected target:

- `T.ppl_copy` and `T.ppl_fill`
- `T.ppl_gemm`
- `T.ppl_add`, `T.ppl_subtract`, `T.ppl_mul`, and `T.ppl_div`
- `T.ppl_max`

Both programming models provide scalar arithmetic, `exp`, `rsqrt`, sum/max
reductions, and portable embedding lookup, subject to the documented dtype
matrix. TPU-Kernel additionally exposes backend-specific gather and BM1690
top-k operations. The `T.rvt_*` namespace is a low-level SG2260E interface for
kernels that manage RV Tensor descriptors directly; it must not be mixed with
high-level `T.ppl_*` semantics inside one kernel. See the
[`tpu_demo` guide](./tpu_demo/README.md) for examples and the
[`T.ppl_*` mapping](./tpu_demo/OP_MAPPING.md) for exact instruction and dtype
support.

Operator shapes and data types are checked during lowering. Dimensions must be
positive compile-time integers, and tiled dimensions must divide evenly unless
an operator provides its own boundary handling.

## Examples

The examples under [`tpu_demo/`](./tpu_demo/) use reusable builder functions and
a common command-line runner. They cover:

| Example | Description | Programming models |
| --- | --- | --- |
| Elementwise | Add, subtract, multiply, and divide | TPU-Kernel, RV Tensor |
| Matmul | Tiled matrix multiplication | TPU-Kernel, RV Tensor |
| RMSNorm | Standard and split-K normalization | TPU-Kernel |
| RoPE | Rotary positional embedding | TPU-Kernel |
| SwiGLU | Gated activation | TPU-Kernel |
| FlashAttention | Tiled attention with online softmax | TPU-Kernel |

The examples accept `float16`, `bfloat16`, and `float32`. The TPU DSL and
TPU-Kernel instruction selector also provide operation-specific FP8 paths, with
their data-type constraints enforced during lowering. See the
[demo guide](./tpu_demo/README.md) for builder APIs, case selection, numerical
comparison rules, and the serial PCIe runner.

## Profiling

Profiling is enabled at compilation time through the PPL `--profiling` path.
TileLang-TPU carries the required compiler options and environment into each
isolated JIT build, then keeps the runtime records with the corresponding
kernel result.

- CModel profiling reports simulator instruction timing.
- PCIe profiling collects TPUDNN records and can invoke an optional decoder for
  per-instruction timing.
- Profiling output is intended for instruction mapping and performance
  diagnosis. A single run should not be treated as a stable benchmark.

## Repository Layout

- [`tilelang/`](./tilelang/): TileLang frontend and TPU-facing Python APIs
- [`tilelang/engine/`](./tilelang/engine/): target configuration and TPU pass
  pipeline
- [`tilelang/jit/adapter/`](./tilelang/jit/adapter/): PPL 1.7 toolchain, JIT,
  runtime, and profiling integration
- [`src/target/`](./src/target/): TPU source generation and runtime modules
- [`src/transform/`](./src/transform/): TPU-specific compiler transformations
- [`src/tl_templates/tpu/`](./src/tl_templates/tpu/): generated-code templates
- [`tpu_demo/`](./tpu_demo/): high-level operator examples
- [`testing/python/jit/`](./testing/python/jit/): compiler, operator, and runtime
  tests

## Development

Rebuild the native components after changing C++ code:

```bash
cmake --build build-tpu --parallel 10
```

Run the formatter before submitting changes:

```bash
./format.sh
```

The TPU test runners keep CModel and PCIe execution separate. PCIe cases must be
run serially through `testing/python/jit/tpu_demo_ops_matrix.py`; the runner
acquires the device lock, checks the CModel prerequisites, and stops at the
first device error.

## Acknowledgements

TileLang-TPU builds on open-source work from
[TileLang](https://github.com/tile-ai/tilelang) and uses compiler and runtime
components from [SOPHGO PPL](https://github.com/sophgo/PPL).
