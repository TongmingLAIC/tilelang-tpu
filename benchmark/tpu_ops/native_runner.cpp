// Common timing boundary for both compilers. Generated binding supplies
// prepare_api(), entry_name(), group_count(), and core_count().
#include <tpuv7_rt.h>
#include <tpuDNN.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
tpuRtStream_t stream = nullptr;
tpuRtKernelModule_t tpu_module = nullptr;
#include "binding.inc"

static void check(int code, const char *what) {
  if (code) throw std::runtime_error(std::string(what) + ": " + std::to_string(code));
}

// Bound one launch / progress gap, rather than a healthy 100-sample batch.
// Updates stay outside timing boundaries; cmodel has its own longer budget.
#ifndef USING_CMODEL
static std::atomic<int64_t> progress_deadline_ns{0};
static void heartbeat() {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
  progress_deadline_ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(
      deadline.time_since_epoch()).count(), std::memory_order_relaxed);
}
static void start_watchdog() {
  heartbeat();
  std::thread([] {
    while (true) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count();
      if (now > progress_deadline_ns.load(std::memory_order_relaxed)) {
        fprintf(stderr, "TPU_BENCH_ERROR=native launch/progress stalled for 120 seconds\n");
        std::_Exit(124);
      }
    }
  }).detach();
}
#else
static void heartbeat() {}
static void start_watchdog() {}
#endif
int main(int argc, char **argv) {
  try {
    if (argc < 5) throw std::runtime_error("runner module manifest warmup repeats");
    const int warmup = std::stoi(argv[3]), repeats = std::stoi(argv[4]);
    if (warmup < 0 || repeats < 0) throw std::runtime_error("negative repetition count");
    std::ifstream manifest(argv[2]);
    if (!manifest) throw std::runtime_error("missing buffer manifest");
    std::vector<std::vector<char>> host;
    std::vector<void *> device;
    std::vector<std::string> filenames;
    std::string filename; size_t bytes;
    while (manifest >> filename >> bytes) {
      filenames.push_back(filename);
      host.emplace_back(bytes);
      std::ifstream file(filename, std::ios::binary);
      if (!file.read(host.back().data(), bytes)) throw std::runtime_error("short input " + filename);
    }
    if (host.empty()) throw std::runtime_error("empty manifest");
    start_watchdog();
    check(tpuRtInit(), "init");
    check(tpuRtSetDevice(0), "device");
    check(tpuRtStreamCreate(&stream), "stream");
    fprintf(stderr, "TPU_BENCH_PHASE=module_load\n");
    tpu_module = tpuRtKernelLoadModuleFile(argv[1], stream);
    if (!tpu_module) throw std::runtime_error("module load");
    for (auto &buffer : host) {
      void *ptr = nullptr;
      check(tpuRtMalloc(&ptr, buffer.size(), 0), "malloc");
      check(tpuRtMemcpyS2D(ptr, buffer.data(), buffer.size()), "H2D");
      device.push_back(ptr);
    }
    auto api = prepare_api(device);
    auto launch = [&]() {
      check(tpuRtKernelLaunch(tpu_module, entry_name(), api.data(), api.size(),
                             group_count(), core_count(), stream), "launch");
      check(tpuRtStreamSynchronize(stream), "synchronize");
    };
    // A numerical run always precedes warmup and measurements.
    fprintf(stderr, "TPU_BENCH_PHASE=numerical_launch\n");
    heartbeat();
    launch();
    fprintf(stderr, "TPU_BENCH_PHASE=numerical_output\n");
    for (size_t i = 0; i < device.size(); ++i) {
      check(tpuRtMemcpyD2S(host[i].data(), device[i], host[i].size()), "D2H");
      std::ofstream file(filenames[i] + ".out", std::ios::binary);
      file.write(host[i].data(), host[i].size());
      if (!file) throw std::runtime_error("output write");
    }
    fprintf(stderr, "TPU_BENCH_PHASE=warmup\n");
    for (int i = 0; i < warmup; ++i) { heartbeat(); launch(); }
    fprintf(stderr, "TPU_BENCH_PHASE=measurement\n");
    std::vector<double> times;
    for (int i = 0; i < repeats; ++i) {
      heartbeat();
      auto start = std::chrono::steady_clock::now();
      launch();
      auto stop = std::chrono::steady_clock::now();
      times.push_back(std::chrono::duration<double, std::micro>(stop-start).count());
    }
    // Also check the last measured launch, to catch stateful repeated-run bugs.
    fprintf(stderr, "TPU_BENCH_PHASE=final_output\n");
    for (size_t i = 0; i < device.size(); ++i) {
      check(tpuRtMemcpyD2S(host[i].data(), device[i], host[i].size()), "final D2H");
      std::ofstream file(filenames[i] + ".final", std::ios::binary);
      file.write(host[i].data(), host[i].size());
    }
    printf("TPU_BENCH_SAMPLES_US=[");
    for (size_t i=0; i<times.size(); ++i) printf("%s%.9f", i ? "," : "", times[i]);
    printf("]\n");
    if (argc>5 && std::stoi(argv[5])==1) {
      auto handle=tpudnnHandleFromStream(0,stream,tpu_module);
      if (!handle) throw std::runtime_error("profile handle");
      check(tpudnnEnableProfile(handle,1000000,0),"profile enable");
      heartbeat();
      launch();
      check(tpudnnDisableProfile(handle),"profile disable");
      tpudnnDestroy(handle);
    }
    for (auto ptr : device) check(tpuRtFree(&ptr, 0), "free");
    check(tpuRtKernelUnloadModule(tpu_module, stream), "unload");
    check(tpuRtStreamDestroy(stream), "stream destroy");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "TPU_BENCH_ERROR=%s\n", error.what());
    return 1;
  }
}
