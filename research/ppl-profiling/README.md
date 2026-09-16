# TPU Instruction Profiling

This document describes TileLang-TPU instruction profiling for CModel and PCIe
execution. It covers the PPL 1.7 integration, report format, process control,
and the limits of the collected timing data.

Host wall time, raw command records, and decoded device timing are different
measurements. Only decoded device timing represents the execution interval of
an individual TIU or GDMA command.

## Why TileLang has its own profiler

In PPL 1.7, `ppl_compile.py --profiling` enters the PPL autotuning workflow. It
expects a PPL `.pl` input and generates its own host code before compiling,
running, collecting traces, and processing the result.

TileLang already generates `kernel.c`, the host wrapper, and `main.so`.
Reusing the complete PPL workflow would generate a second host path and would
not attach the recorder to TileLang's `tpuRtKernelLaunch`. TileLang therefore
uses the PPL and TPUDNN recording protocols while managing compilation and
execution through `TPUInstructionProfiler`.

The generic `tilelang.profiler.Profiler` uses CUDA events and
`torch.cuda.synchronize()`, so it cannot provide TPU runtime or device timing.

## Data flow

```text
CModel worker
  -> FILE_DUMP_CMD
  -> BDC/GDMA/SDMA raw files and sidecar text
  -> optional PerfAI AutoRunner
  -> profile_data.js

PCIe worker
  -> tpudnnHandleFromStream
  -> tpudnnEnableProfile
  -> one kernel launch and synchronization
  -> tpudnnDisableProfile
  -> cdm_profile_data_dev*
  -> bigTpuProfile decoder
  -> stable TileLang JSON
```

## Components

| Component | Responsibility |
| --- | --- |
| `tilelang/jit/adapter/tpu_profiling.py` | Configuration, isolated workers, deadlines, trace collection, and normalized reports |
| `tilelang/jit/_tpu_profile_supervisor.py` | Parent-death handling, private process groups, and bounded termination |
| `tilelang/jit/_tpu_pcie_profile_decoder.py` | Conversion from `bigTpuProfile` output to stable JSON |
| `src/tl_templates/tpu/main_template.cpp` | TPUDNN profile-session lifecycle in the PCIe host wrapper |
| `tilelang/jit/adapter/ppl_layout.py` | PPL 1.7 SDK paths and separate CModel and PCIe runtimes |
| `tilelang/jit/adapter/libgen.py` | Runtime, RPATH, and TPUDNN selection during compilation |
| `testing/python/jit/tpu_*_ops_matrix.py` | Staged matrices, first-error stop, and report storage |

## Target and runtime identity

The TVM target resolves to `TPUTargetSpec(chip, programming_model)`. The host
runtime resolves separately to `TPURuntimeConfig(runtime_mode)`.

- BM1690 has eight physical cores and supports TPU-Kernel.
- SG2260E has four physical cores and supports TPU-Kernel and RV Tensor.
- `cmodel` and `pcie` select the host runtime; they do not change device
  instruction semantics.

Profiling records the chip, programming model, runtime mode, SDK path, runtime
path, and generated source identity. Directory names and environment defaults
are never used to infer a target.

## CModel profiling

Each CModel worker runs in a separate directory with:

- A relative `FILE_DUMP_CMD` path
- The chip-specific `TPU_RT_CORE_NUM`
- `TILELANG_TPU_BENCHMARK_RUNS=0`
- PCIe authorization variables removed from the worker environment

The raw text identifies the engine, core, command ID, and opcode. It does not
contain reliable begin and end timestamps, so TileLang does not substitute host
time for missing device timing.

When `perfai_root` or `PPL_PERFAI_ROOT` is set, the profiler runs
`AutoRunner.sh -d <session> -e <chip>`. Only BDC, TIU, GDMA, SDMA, VSDMA, CDMA,
and DMA channels are classified as device instructions. The full decoded
timeline remains available in the report.

## PCIe profiling

A PCIe profile session requires these worker variables:

```text
TILELANG_TPU_ALLOW_PCIE_LOAD=1
TILELANG_TPU_ALLOW_PCIE_PROFILE=1
TILELANG_TPU_DEVICE_ID=<non-negative device ID>
```

The generated host wrapper completes H2D transfer, creates a TPUDNN handle,
enables the recorder, launches one kernel, synchronizes, disables the recorder,
and then performs D2H transfer and cleanup. A profiled `main.so` is consumed by
one worker session.

CModel links against the runtime in
`$PPL_PROJECT_ROOT/deps/runtime/tpuv7-runtime/lib`. PCIe links against the
installed board runtime, normally `/opt/tpuv7/tpuv7-current/lib`. Kernel
headers and the chip-specific `libtpudnn.so` still come from the same PPL 1.7
SDK.

## Decoding PCIe traces

The decoder calls the structured
`bigTpuProfile.bmprofile_perfAI.ProfileParser.parse` API and writes
`tilelang_pcie_profile.json`. Each instruction contains:

- Engine and core
- Command ID and opcode
- Begin, end, duration, and unit

Strict timing mode requires finite timestamps, `end >= begin`,
`duration = end - begin`, and `unit = "ns"`.

`pcie_decoder_python` and `pcie_decoder_pythonpath` apply only to the decoder
process. They do not affect compilation or numerical execution. Before the
first device launch, `preflight_pcie_decoder()` checks the package version and
parser API in a process without device access.

## Python API

```python
from tilelang.jit import TPUInstructionProfiler, TPUProfilingConfig

config = TPUProfilingConfig(
    chip="sg2260e",
    programming_model="rv",
    runtime_mode="cmodel",
    output_dir="./profiles",
    label="rv-matmul",
    timeout_s=120,
)

report = TPUInstructionProfiler(config).run_cmodel(
    ["/path/to/python", "/path/to/fresh_worker.py"],
    environment={"PPL_PROJECT_ROOT": "/path/to/ppl-1.7"},
)
```

`run_pcie()` applies the session authorization, process supervision, and device
lock for one profiling worker. The matrix runners add CModel promotion checks,
board-health checks, and persistent quarantine state.

When `output_dir` is omitted, the Python API writes to
`./tilelang-tpu-profiles/<label>-*`. Matrix output belongs under the ignored
`research/artifacts/` directory, with a new directory for each run.

## Matrix command

The PCIe matrices require BM1690 and SG2260E CModel summaries generated from
the same clean commit:

```bash
python testing/python/jit/tpu_core_ops_matrix.py \
  --runtime-mode pcie \
  --output-dir research/artifacts/<date>/<run> \
  --chip sg2260e \
  --device-id 0 \
  --allow-pcie \
  --allow-pcie-profile \
  --all-pcie-cases \
  --require-decoded-timing \
  --bm-cmodel-summary <bm-summary.json> \
  --sg-cmodel-summary <sg-summary.json>
```

## Process and device safety

Every worker and decoder runs in a private process group under one monotonic
deadline. Parent exit, timeout, or an exception triggers bounded TERM and KILL
steps followed by process reaping and pipe cleanup. A worker that exits while
leaving descendants in its process group is treated as a failure.

The matrix runner holds one device lock across the preflight check, all
selected cases, and the final health check. After each launch, two consecutive
samples must report `Active` and 0% utilization. A device fault, invalid health
record, topology change, probe timeout, or failure to return to idle stops the
matrix and records a quarantine marker.

The runner removes only the temporary directories it created. Raw traces,
decoded reports, and failure diagnostics remain under `research/artifacts/`.

## Report fields

| Field | Meaning |
| --- | --- |
| `raw_trace_files` | CModel files or PCIe recorder directories |
| `raw_instructions` | Commands parsed from CModel sidecar files, without timing |
| `decoded_report_paths` | Stable JSON files written by the decoder |
| `decoder_identity` | Decoder package, version, and parser API |
| `timeline_events` | Complete valid timeline returned by the decoder |
| `instruction_timings` | Device engine, core, ID, opcode, and time interval |
| `parser_status` | `ready`, `unavailable`, `no-raw-trace`, or `invalid-report` |

## Limits

- CModel profiling can produce raw command records even when no compatible
  decoder is installed; such reports intentionally contain no device duration.
- SG2260E PCIe reports may include decoded nanosecond intervals when a
  compatible `bigTpuProfile` environment is configured.
- Each profile uses one recorded launch. It is suitable for checking
  instruction mapping and finding slow commands, not for latency or throughput
  benchmarks.
- Performance regression testing needs an unrecorded warm-up and repeated
  sampling with a fixed target, SDK, runtime, shape, and data type.
- The current software pipeline lacks a TPU dependency and hazard model, so
  correctness matrices use dependency-ordered serial execution.
