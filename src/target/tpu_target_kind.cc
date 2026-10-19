/*
 * Copyright (c) Tile-AI Corporation.
 * Licensed under the MIT License.
 */

/*!
 * \file target/tpu_target_kind.cc
 * \brief Register the TileLang TPU target without modifying TVM sources.
 */

#include <tvm/ir/expr.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

#include <algorithm>

namespace tvm {

// The target kind belongs to the TileLang TPU backend, not to a copied TVM
// source file. Register only when it is genuinely absent. An existing but
// incomplete registration is a build mismatch, not an API to repair at
// process startup.
// ``mcpu`` is the canonical physical-chip selector; ``model`` remains a
// generic TVM annotation and is supplied by the common target options below.
void RegisterTPUTargetKindIfAbsent() {
  Optional<TargetKind> existing_target_kind = TargetKind::Get("tpu");
  if (existing_target_kind.defined()) {
    Map<String, String> options =
        TargetKindRegEntry::ListTargetKindOptions(existing_target_kind.value());
    ICHECK(options.count("mcpu"))
        << "An existing TVM target kind named 'tpu' does not accept -mcpu; "
        << "TileLang TPU requires -mcpu=<bm1690|bm1690e>. Rebuild against a "
        << "compatible TVM target registration.";
    ICHECK_EQ(options["mcpu"], "runtime.String")
        << "An existing TVM target kind named 'tpu' has an incompatible "
        << "-mcpu option type; expected runtime.String, got "
        << options["mcpu"];
    ICHECK(options.count("tpu-programming-model"))
        << "An existing TVM target kind named 'tpu' lacks "
        << "-tpu-programming-model=<tpukernel|rv>. Rebuild TVM/TileLang "
        << "instead of extending an incompatible registration at runtime.";
    ICHECK_EQ(options["tpu-programming-model"], "runtime.String")
        << "An existing TVM target kind named 'tpu' has an incompatible "
        << "-tpu-programming-model option type; expected runtime.String, got "
        << options["tpu-programming-model"];
    ICHECK_EQ(existing_target_kind.value()->default_device_type, kDLCPU)
        << "An existing TVM target kind named 'tpu' uses an incompatible "
        << "default device type; TileLang TPU host wrappers require kDLCPU";
    const Array<String> &default_keys =
        existing_target_kind.value()->default_keys;
    ICHECK(std::find(default_keys.begin(), default_keys.end(), String("tpu")) !=
           default_keys.end())
        << "An existing TVM target kind named 'tpu' does not include the "
        << "required default key 'tpu'";
    return;
  }

  TargetKindRegEntry::RegisterOrGet("tpu")
      .set_name()
      // Generated TPU host wrappers use ordinary host pointers and are not a
      // TVM DeviceAPI. Keep kDLCPU as that ABI marker; scheduling/backend
      // dispatch uses the dedicated `tpu` key and target kind below.
      .set_default_device_type(kDLCPU)
      .add_attr_option<Array<String>>("keys")
      .add_attr_option<String>("tag")
      .add_attr_option<String>("device")
      .add_attr_option<String>("model")
      .add_attr_option<Array<String>>("libs")
      .add_attr_option<Target>("host")
      .add_attr_option<Integer>("from_device")
      .add_attr_option<Integer>("target_device_type")
      .add_attr_option<String>("mcpu")
      .add_attr_option<String>("tpu-programming-model")
      .add_attr_option<Integer>("workspace-byte-alignment")
      .add_attr_option<Integer>("constants-byte-alignment")
      .set_default_keys({"tpu"});
}

struct TPUTargetKindRegistrar {
  TPUTargetKindRegistrar() { RegisterTPUTargetKindIfAbsent(); }
};

static TPUTargetKindRegistrar tpu_target_kind_registrar;

} // namespace tvm
