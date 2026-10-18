# Installing TileLang-TPU

This guide installs TileLang-TPU from source on an x86_64 Linux host. CModel
execution does not require a TPU card. SG2260E PCIe execution requires the
TPUv7 driver and runtime supplied with the card.

## Requirements

| Component | Requirement |
| --- | --- |
| Operating system | x86_64 Linux; the commands below use Ubuntu or Debian |
| Python | Python 3.8 or later; Python 3.10 is recommended |
| Build tools | CMake 3.26 or later and a C++17 compiler |
| TPU SDK | SOPHGO PPL 1.7.122 development package |
| PCIe runtime | TPUv7 driver and runtime for SG2260E |

## 1. Install system packages

On Ubuntu or Debian, run:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential ca-certificates curl git \
  python3 python3-dev python3-venv \
  libedit-dev libtinfo-dev libxml2-dev zlib1g-dev
```

CMake is installed in the Python environment in step 3.

## 2. Clone TileLang-TPU

```bash
git clone https://github.com/xwhzz/tilelang-tpu.git
cd tilelang-tpu
git submodule update --init --recursive 3rdparty/tvm
```

The TPU build uses the TVM revision included as a submodule. The remaining
commands assume that the current directory is the TileLang-TPU repository root.

## 3. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt "cmake>=3.26"
```

This installation uses the Python package directly from the source tree. An
editable pip installation is not required.

Install the test dependencies only when developing TileLang-TPU:

```bash
python -m pip install -r requirements-test.txt
```

## 4. Install the PPL 1.7 SDK

Download the full development package from the official
[SOPHGO PPL v1.7.122 release](https://github.com/sophgo/PPL/releases/tag/v1.7.122).
The required release asset is
[`ppl_v1.7.122-g05ebfb36-20260528.tar.gz`](https://github.com/sophgo/PPL/releases/download/v1.7.122/ppl_v1.7.122-g05ebfb36-20260528.tar.gz).
GitHub's automatically generated source archives do not contain the complete
SDK.

The following commands install the SDK under `$HOME/toolchains`:

```bash
mkdir -p "${HOME}/toolchains"
curl -fL \
  https://github.com/sophgo/PPL/releases/download/v1.7.122/ppl_v1.7.122-g05ebfb36-20260528.tar.gz \
  -o "${HOME}/toolchains/ppl_v1.7.122-g05ebfb36-20260528.tar.gz"
tar -xzf "${HOME}/toolchains/ppl_v1.7.122-g05ebfb36-20260528.tar.gz" \
  -C "${HOME}/toolchains"

export PPL_PROJECT_ROOT="${HOME}/toolchains/ppl_v1.7.122-g05ebfb36-20260528"
```

`PPL_PROJECT_ROOT` must point to the extracted directory that contains
`deps/`. Check the chip map before continuing:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

chip_map_path = Path(os.environ["PPL_PROJECT_ROOT"]) / "deps/chip/chip_map.json"
chip_map = json.loads(chip_map_path.read_text(encoding="utf-8"))
print(f"BM1690:  {chip_map['bm1690']}")
print(f"SG2260E: {chip_map['sg2260e']}")
PY
```

The output should be:

```text
BM1690:  tpub_7_1
SG2260E: tpub_7_1_e
```

TileLang-TPU reads headers, libraries, CModel files, firmware, and the PCIe
cross-compiler from this SDK. It does not require the PPL environment script.

## 5. Build TileLang-TPU

Run the build script from the repository root:

```bash
./build_tpu.sh
```

The script configures and builds TileLang and TVM in `build-tpu/`. It creates
the following native libraries:

```text
build-tpu/libtilelang_module.so
build-tpu/tvm/libtvm.so
```

Add the source tree to the current Python environment:

```bash
export TILELANG_TPU_ROOT="$(pwd)"
export PYTHONPATH="${TILELANG_TPU_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
```

TileLang finds the native libraries in `build-tpu/` automatically.

## 6. Check the installation

First, import the Python package and its native library:

```bash
python -c 'import tilelang; print(tilelang.__version__)'
```

Next, check the PPL SDK components used by both chips:

```bash
python - <<'PY'
import os

from tilelang.jit.adapter.ppl_layout import resolve_ppl_layout

root = os.environ["PPL_PROJECT_ROOT"]
for chip in ("bm1690", "sg2260e"):
    layout = resolve_ppl_layout(root, chip)
    layout.require_runtime("cmodel")
    if chip == "sg2260e":
        layout.require_rvt_api()
    print(f"{chip}: {layout.arch}, {layout.physical_core_count} cores")
PY
```

Expected output:

```text
bm1690: tpub_7_1, 8 cores
sg2260e: tpub_7_1_e, 4 cores
```

Run one TPU-Kernel example and one RV Tensor example with CModel:

```bash
python -m tilelang.jit.tpu_cmodel \
  --output-dir "$HOME/tmp/tilelang/install-tpukernel-$(date +%s%N)" -- \
  python -m tpu_demo.run \
  --case elementwise-add.float16 \
  --chip sg2260e \
  --programming-model tpukernel \
  --runtime-mode cmodel

python -m tilelang.jit.tpu_cmodel \
  --output-dir "$HOME/tmp/tilelang/install-rv-$(date +%s%N)" -- \
  python -m tpu_demo.run \
  --case matmul.float16 \
  --chip sg2260e \
  --programming-model rv \
  --runtime-mode cmodel
```

Each command returns a JSON result with `"status": "passed"` when the Python
environment, native libraries, PPL SDK, and CModel runtime are configured
correctly. Importing TileLang automatically creates and uses the private
`~/tmp/tilelang` scratch directory. CModel must run through the launcher:
the vendor's fixed `/tmp` communication path is remapped to a fresh directory
under `~/tmp/tilelang` for each invocation. The result directory must not
exist beforehand. This isolation requires Linux user namespaces and
`bubblewrap`.

## 7. Configure SG2260E PCIe execution

Install the TPUv7 driver and runtime package supplied with the SG2260E card,
then check the device and runtime library:

```bash
tpu-smi
test -f /opt/tpuv7/tpuv7-current/lib/libtpuv7_rt.so
```

`/opt/tpuv7/tpuv7-current/lib` is the default PCIe runtime directory. Set the
following variable only when the runtime is installed elsewhere:

```bash
export TILELANG_TPU_PCIE_RUNTIME_PATH=/absolute/path/to/tpuv7-runtime/lib
```

Check the PCIe compiler, firmware, and host runtime without launching a kernel:

```bash
python - <<'PY'
import os

from tilelang.jit.adapter.ppl_layout import resolve_ppl_layout

layout = resolve_ppl_layout(os.environ["PPL_PROJECT_ROOT"], "sg2260e")
layout.require_runtime("pcie")
print(f"Cross-compiler: {layout.pcie_cross_gcc()}")
print(f"Firmware:       {layout.firmware_archive}")
print(f"Runtime:        {layout.pcie_runtime_lib()}")
PY
```

PCIe profiling also needs the TPUDNN headers and library included in the PPL
SDK. Check them with:

```bash
python - <<'PY'
import os

from tilelang.jit.adapter.ppl_layout import resolve_ppl_layout

layout = resolve_ppl_layout(os.environ["PPL_PROJECT_ROOT"], "sg2260e")
layout.require_profiling("pcie")
print("PCIe profiling is ready")
PY
```

Run PCIe examples through the serial runner described in the
[TPU demo guide](../../tpu_demo/README.md). The runner manages device access
and keeps CModel and PCIe results separate.

## 8. Start a new shell

Restore the environment before using TileLang-TPU in a new shell:

```bash
cd /absolute/path/to/tilelang-tpu
source .venv/bin/activate
export PPL_PROJECT_ROOT="${HOME}/toolchains/ppl_v1.7.122-g05ebfb36-20260528"
export TILELANG_TPU_ROOT="$(pwd)"
export PYTHONPATH="${TILELANG_TPU_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
```

## 9. Update the installation

```bash
git pull --ff-only
git submodule update --init --recursive 3rdparty/tvm
./build_tpu.sh
```

After C++-only changes, an incremental build is sufficient:

```bash
cmake --build build-tpu --parallel 10
```

## Troubleshooting

### `PPL 1.7 SDK layout is required`

Set `PPL_PROJECT_ROOT` to the extracted SDK directory. The file
`$PPL_PROJECT_ROOT/deps/chip/chip_map.json` must exist.

### `libtilelang_module.so` or `libtvm.so` is missing

Run `./build_tpu.sh` from the repository root and keep the source tree on
`PYTHONPATH`.

### CMake is older than 3.26

Activate `.venv` and run:

```bash
python -m pip install --upgrade "cmake>=3.26"
```

### PCIe loads the CModel runtime

Set `TILELANG_TPU_PCIE_RUNTIME_PATH` to the installed TPUv7 board runtime. The
SDK path `deps/runtime/tpuv7-runtime/lib` contains the CModel runtime and is not
the PCIe runtime.
