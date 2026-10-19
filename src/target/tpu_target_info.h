/*
 * Copyright (c) Tile-AI Corporation.
 * Licensed under the MIT License.
 */

/*!
 * \file target/tpu_target_info.h
 * \brief Canonical native validation for TileLang TPU target identities.
 */
#ifndef TVM_TL_TARGET_TPU_TARGET_INFO_H_
#define TVM_TL_TARGET_TPU_TARGET_INFO_H_

#include <tvm/target/target.h>

#include <cctype>
#include <string>

namespace tvm {
namespace tl {
namespace tpu {

struct TargetSelection {
  std::string chip;
  std::string programming_model;
};

inline std::string LowerASCII(std::string value) {
  for (char &character : value) {
    character =
        static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
  }
  return value;
}

inline std::string TrimASCII(std::string value) {
  const auto is_space = [](unsigned char character) {
    return std::isspace(character) != 0;
  };
  size_t first = 0;
  while (first < value.size() &&
         is_space(static_cast<unsigned char>(value[first]))) {
    ++first;
  }
  size_t last = value.size();
  while (last > first &&
         is_space(static_cast<unsigned char>(value[last - 1]))) {
    --last;
  }
  return value.substr(first, last - first);
}

inline std::string TargetStringOrEmpty(const Optional<String> &value) {
  if (!value.defined()) {
    return "";
  }
  std::string normalized = TrimASCII(value.value());
  return normalized == "unknown" ? "" : normalized;
}

inline bool IsSupportedChip(const std::string &chip) {
  return chip == "bm1690" || chip == "bm1690e";
}

inline bool SupportsProgrammingModel(const std::string &chip,
                                     const std::string &programming_model) {
  return programming_model == "tpukernel" ||
         (chip == "bm1690e" && programming_model == "rv");
}

inline TargetSelection ResolveTarget(const Target &target,
                                     const char *boundary) {
  ICHECK_EQ(target->kind->name, "tpu") << boundary << " requires a TPU Target";
  const std::string chip =
      LowerASCII(TargetStringOrEmpty(target->GetAttr<String>("mcpu")));
  ICHECK(!chip.empty() && IsSupportedChip(chip))
      << boundary << " requires a supported target chip via "
      << "tpu -mcpu=<bm1690|bm1690e>; got "
      << (chip.empty() ? "no chip" : chip);

  const std::string programming_model =
      TargetStringOrEmpty(target->GetAttr<String>("tpu-programming-model"));
  ICHECK(programming_model == "tpukernel" || programming_model == "rv")
      << boundary << " requires a normalized target "
      << "tpu-programming-model=<tpukernel|rv>; got "
      << (programming_model.empty() ? "no programming model"
                                    : programming_model);
  ICHECK(SupportsProgrammingModel(chip, programming_model))
      << boundary << ": TPU target chip " << chip << " does not support "
      << "tpu-programming-model=" << programming_model;
  return {chip, programming_model};
}

inline void CheckSameSelection(const TargetSelection &function_selection,
                               const TargetSelection &build_selection,
                               const char *boundary) {
  ICHECK_EQ(function_selection.chip, build_selection.chip)
      << boundary << ": PrimFunc chip " << function_selection.chip
      << " disagrees with build target chip " << build_selection.chip;
  ICHECK_EQ(function_selection.programming_model,
            build_selection.programming_model)
      << boundary << ": PrimFunc programming model "
      << function_selection.programming_model
      << " disagrees with build target programming model "
      << build_selection.programming_model;
}

} // namespace tpu
} // namespace tl
} // namespace tvm

#endif // TVM_TL_TARGET_TPU_TARGET_INFO_H_
