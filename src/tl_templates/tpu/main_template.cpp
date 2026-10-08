// clang-format off
#include <tpuv7_rt.h>
#ifdef TILELANG_TPU_PCIE_PROFILING
#include <tpuDNN.h>
#endif
#include "kernel.h"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

tpuRtStream_t stream = nullptr;
tpuRtKernelModule_t tpu_module = nullptr;
// The vendor runtime and the loaded kernel module are process-global
// resources.  Rebuilding them for every dispatch makes a launch cost ~145 ms
// simply to reload this module, while the dispatch itself is ~0.2 ms.
static bool tilelang_tpu_runtime_ready = false;
static std::mutex tilelang_tpu_profile_mutex;
static int tilelang_tpu_expected_device_id = -1;
#ifdef TILELANG_TPU_PCIE_PROFILING
static bool tilelang_tpu_pcie_profile_consumed = false;
#endif

static bool tilelang_tpu_env_is_one(const char* name) {{
  const char* value = std::getenv(name);
  return value != nullptr && std::strcmp(value, "1") == 0;
}}

#ifdef TILELANG_TPU_PCIE_PROFILING
static bool tilelang_tpu_parse_profile_integer(
    const char* name, int minimum, int* result) {{
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {{
    return false;
  }}
  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed < minimum ||
      parsed > std::numeric_limits<int>::max()) {{
    return false;
  }}
  *result = static_cast<int>(parsed);
  return true;
}}

// PPL 1.7's PCIe profiler records commands through a TPUDNN handle wrapping
// the same runtime stream and module used by the kernel launch.  Keep this
// path dormant unless the isolated profiler supplies every explicit gate.
static int tilelang_tpu_begin_pcie_profile(tpudnnHandle_t* handle) {{
  if (!tilelang_tpu_env_is_one("TILELANG_TPU_PROFILE_SESSION")) {{
    return 0;
  }}
  const char* runtime_mode = std::getenv("TILELANG_TPU_PROFILE_RUNTIME_MODE");
  if (runtime_mode == nullptr || std::strcmp(runtime_mode, "pcie") != 0 ||
      !tilelang_tpu_env_is_one("TILELANG_TPU_ALLOW_PCIE_LOAD") ||
      !tilelang_tpu_env_is_one("TILELANG_TPU_ALLOW_PCIE_PROFILE") ||
      !tilelang_tpu_env_is_one("BMLIB_ENABLE_ALL_PROFILE")) {{
    std::cerr << "Incomplete TileLang PCIe profiling authorization.\n";
    return -11;
  }}
  {{
    std::lock_guard<std::mutex> lock(tilelang_tpu_profile_mutex);
    if (tilelang_tpu_pcie_profile_consumed) {{
      std::cerr << "A PCIe profiling artifact permits exactly one run.\n";
      return -16;
    }}
    // Consume the attempt before touching TPUDNN. A failed recorder setup is
    // not safe to retry in the same process/runtime instance.
    tilelang_tpu_pcie_profile_consumed = true;
  }}
  int record_size = 0;
  int book_keeping = 0;
  if (!tilelang_tpu_parse_profile_integer(
          "PROFILE_RECORD_SIZE", 1, &record_size) ||
      !tilelang_tpu_parse_profile_integer(
          "PROFILE_BOOK_KEEPING", 0, &book_keeping)) {{
    std::cerr << "Invalid TileLang PCIe profiling record configuration.\n";
    return -12;
  }}
  *handle = tpudnnHandleFromStream(
      tilelang_tpu_expected_device_id, stream, tpu_module);
  if (*handle == nullptr) {{
    return -13;
  }}
  if (tpudnnEnableProfile(*handle, record_size, book_keeping) !=
      TPUDNN_STATUS_SUCCESS) {{
    tpudnnDestroy(*handle);
    *handle = nullptr;
    return -14;
  }}
  return 1;
}}
#endif

// LibraryGenerator calls this immediately after dlopen, before tilelang_tpu_run
// can initialize the vendor runtime.  Keep the expected device inside main.so
// as well as Python's process-wide profile: a caller changing the environment
// between dlopen and dispatch must fail before tpuRtInit/tpuRtSetDevice.
extern "C" int tilelang_tpu_bind_device(int expected_device_id) {{
  if (expected_device_id < 0) {{
    return -1;
  }}
#ifdef USING_CMODEL
  if (expected_device_id != 0) {{
    return -2;
  }}
#endif
  std::lock_guard<std::mutex> lock(tilelang_tpu_profile_mutex);
  if (tilelang_tpu_expected_device_id >= 0 &&
      tilelang_tpu_expected_device_id != expected_device_id) {{
    return -3;
  }}
  tilelang_tpu_expected_device_id = expected_device_id;
  return 0;
}}

static int tilelang_tpu_device_id() {{
#ifdef USING_CMODEL
  return 0;
#else
  // Board execution is deliberately fail-closed.  The Python loader blocks
  // PCIe dlopen too; retain the same gate here so a manually loaded main.so
  // cannot reach tpuRtInit without an explicit acknowledgement.
  const char* allow_pcie = std::getenv("TILELANG_TPU_ALLOW_PCIE_LOAD");
  if (allow_pcie == nullptr || std::strcmp(allow_pcie, "1") != 0) {{
    std::cerr << "Set TILELANG_TPU_ALLOW_PCIE_LOAD=1 before a PCIe TPU dispatch.\n";
    return -1;
  }}
  // A caller must also name the intended PCIe device rather than inheriting
  // the historical hard-coded device ID 14.
  const char* value = std::getenv("TILELANG_TPU_DEVICE_ID");
  if (value == nullptr || *value == '\0') {{
    std::cerr << "Set TILELANG_TPU_DEVICE_ID before a PCIe TPU dispatch.\n";
    return -1;
  }}
  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed < 0 ||
      parsed > std::numeric_limits<int>::max()) {{
    std::cerr << "Invalid TILELANG_TPU_DEVICE_ID: " << value << "\n";
    return -1;
  }}
  return static_cast<int>(parsed);
#endif
}}

#ifndef TILELANG_PPL_KERNEL_PATH
#error "TPU host artifact must embed its private libkernel.so path"
#endif

static const char* tilelang_tpu_kernel_path() {{
  return TILELANG_PPL_KERNEL_PATH;
}}

int init() {{
  if (tilelang_tpu_runtime_ready) {{
    return 0;
  }}
#ifdef USING_CMODEL
#ifndef TILELANG_TPU_CMODEL_CORE_NUM
#error "CModel main.so must embed the target core count"
#endif
  // This must happen in the process that calls tpuRtInit.  Setting it only in
  // the compiler process breaks a fresh cached/from-database CModel process.
  if (setenv("TPU_RT_CORE_NUM", TILELANG_TPU_CMODEL_CORE_NUM, 1) != 0) {{
    return -9;
  }}
#endif
  const int device_id = tilelang_tpu_device_id();
  if (device_id < 0) {{
    return -2;
  }}
  {{
    std::lock_guard<std::mutex> lock(tilelang_tpu_profile_mutex);
    if (tilelang_tpu_expected_device_id < 0 ||
        tilelang_tpu_expected_device_id != device_id) {{
      // Do not allow an environment change after LibraryGenerator.load_lib()
      // to choose another board (or make a CModel library look like PCIe).
      return -10;
    }}
  }}
  tpuRtStatus_t ret = tpuRtInit();
  if (ret != tpuRtSuccess) {{
    return -1;
  }}
  ret = tpuRtSetDevice(device_id);
  if (ret != tpuRtSuccess) {{
    return -3;
  }}
  ret = tpuRtStreamCreate(&stream);
  if (ret != tpuRtSuccess) {{
    stream = nullptr;
    return -4;
  }}
  const char* kernel_dir = tilelang_tpu_kernel_path();
  if (kernel_dir == nullptr || *kernel_dir == '\0') {{
    tpuRtStreamDestroy(stream);
    stream = nullptr;
    return -5;
  }}
  tpu_module = tpuRtKernelLoadModuleFile(kernel_dir, stream);
  if (tpu_module == nullptr) {{
    tpuRtStreamDestroy(stream);
    stream = nullptr;
    return -6;
  }}
  tilelang_tpu_runtime_ready = true;
  return 0;
}}

void post() {{
  if (tpu_module != nullptr) {{
    tpuRtKernelUnloadModule(tpu_module, stream);
    tpu_module = nullptr;
  }}
  if (stream != nullptr) {{
    tpuRtStreamDestroy(stream);
    stream = nullptr;
  }}
}}

extern "C" int tilelang_tpu_run(void** args) {{
  if (args == nullptr) {{
    return -7;
  }}
{arg_declarations}

  int status = init();
  if (status != 0) {{
    return status;
  }}
  const bool profile_session =
      tilelang_tpu_env_is_one("TILELANG_TPU_PROFILE_SESSION");
#ifdef TILELANG_TPU_PCIE_PROFILING
  tpudnnHandle_t profile_handle = nullptr;
  bool pcie_profile_enabled = false;
#endif

  // Device pointers are initialized so cleanup remains safe after a partial
  // allocation or transfer failure.
{device_declarations}

  do {{
{malloc_statements}
    if (status != 0) {{
      break;
    }}
{memcpy_s2d_statements}
    if (status != 0) {{
      break;
    }}

#ifdef TILELANG_TPU_PCIE_PROFILING
    const int profile_state = tilelang_tpu_begin_pcie_profile(&profile_handle);
    if (profile_state < 0) {{
      status = profile_state;
      break;
    }}
    pcie_profile_enabled = profile_state > 0;
#endif

    auto start = std::chrono::high_resolution_clock::now();
{kernel_call}
    auto end = std::chrono::high_resolution_clock::now();
#ifdef TILELANG_TPU_PCIE_PROFILING
    if (pcie_profile_enabled) {{
      const tpudnnStatus_t sync_status = tpudnnSync(profile_handle);
      const tpudnnStatus_t disable_status = tpudnnDisableProfile(profile_handle);
      pcie_profile_enabled = false;
      if (sync_status != TPUDNN_STATUS_SUCCESS ||
          disable_status != TPUDNN_STATUS_SUCCESS) {{
        status = -15;
        break;
      }}
    }}
#endif
    if (rst != 0) {{
      std::cerr << "kernel_launch failed: " << rst << "\n";
      status = rst;
      break;
    }}
    std::cout << "kernel_launch success\n";

    const auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    const double elapsed_time_ms = duration.count() / 1000.0;
    std::cout << "Single kernel execution time: " << elapsed_time_ms << " ms ("
              << duration.count() << " us)\n";

    // Benchmarking is opt-in. Its default is zero extra launches for both
    // CModel and PCIe, so a first PCIe smoke remains exactly one dispatch.
    int measure_runs = 0;
    if (!profile_session) {{
      const char* value = std::getenv("TILELANG_TPU_BENCHMARK_RUNS");
      if (value != nullptr) {{
        measure_runs = std::max(0, std::atoi(value));
      }}
    }}
    if (measure_runs > 0) {{
      const int warmup_runs = std::min(5, measure_runs);
      std::cout << "\n=== Performance Benchmark (after " << warmup_runs
                << " warmup runs) ===\n";
      for (int i = 0; i < warmup_runs; ++i) {{
{pure_kernel_call}
        if (rst != 0) {{
          status = rst;
          break;
        }}
      }}
      if (status != 0) {{
        break;
      }}
      double total_time_us = 0.0;
      double min_time_us = std::numeric_limits<double>::max();
      double max_time_us = 0.0;
      for (int i = 0; i < measure_runs; ++i) {{
        const auto run_start = std::chrono::high_resolution_clock::now();
{pure_kernel_call}
        const auto run_end = std::chrono::high_resolution_clock::now();
        if (rst != 0) {{
          status = rst;
          break;
        }}
        const double run_time_us =
            std::chrono::duration_cast<std::chrono::microseconds>(run_end - run_start)
                .count();
        total_time_us += run_time_us;
        min_time_us = std::min(min_time_us, run_time_us);
        max_time_us = std::max(max_time_us, run_time_us);
      }}
      if (status != 0) {{
        break;
      }}
      std::cout << "Runs: " << measure_runs << ", average: "
                << total_time_us / measure_runs / 1000.0 << " ms, min: "
                << min_time_us / 1000.0 << " ms, max: "
                << max_time_us / 1000.0 << " ms\n";
    }}

    if (tpuRtStreamSynchronize(stream) != tpuRtSuccess) {{
      status = -8;
      break;
    }}
{memcpy_d2s_statements}
    if (status != 0) {{
      break;
    }}
  }} while (false);

{free_statements}
  // Teardown is the expensive half of a dispatch: tpuRtKernelUnloadModule plus
  // a fresh tpuRtKernelLoadModuleFile next call dominate the ~145 ms a launch
  // used to cost.  Keep the module resident by default and let a caller opt
  // back into per-call teardown for leak checking.
  if (tilelang_tpu_env_is_one("TILELANG_TPU_POST_EACH_CALL")) {{
    post();
    tilelang_tpu_runtime_ready = false;
  }}
#ifdef TILELANG_TPU_PCIE_PROFILING
  if (profile_handle != nullptr) {{
    tpudnnDestroy(profile_handle);
    profile_handle = nullptr;
  }}
#endif
  return status;
}}
// clang-format on
