/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef TVM_TL_TPUV7_LMEM_H_
#define TVM_TL_TPUV7_LMEM_H_

#include <tvm/tir/expr.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace tvm {
namespace tl {
// BM1690 and SG2260E use this common TPUv7 local-memory layout.  Chip-specific
// differences (PPL SDK target, ISA capability, and physical core count) live
// at the canonical Python/native target-capability boundaries, not in this
// shared allocator geometry.
namespace tpuv7 {

constexpr int64_t kLaneNum = 64;
constexpr int64_t kEuBytes = 64;
constexpr int64_t kBankNum = 16;
constexpr int64_t kBankSize = 16 * 1024;
constexpr int64_t kTensorAlignBytes = 64;
// PPL's TPUv7 tensor descriptors encode every N/C/H/W extent in the range
// [1, 65535].  Keep this hardware/API boundary next to the shared TPUv7
// geometry so address assignment and source emission cannot disagree.
constexpr int64_t kDescriptorDimMax =
    static_cast<int64_t>(std::numeric_limits<uint16_t>::max());

// PrimFunc attributes are string-keyed, so AddressAssign cannot attach an
// address directly to a Var object.  Keep generated LMEM metadata in a
// compiler-owned namespace: using the raw data-variable name could overwrite
// standard attributes such as ``target`` or ``global_symbol``.
constexpr const char *kAddressAttrPrefix = "tilelang.tpu.lmem.address.";

inline std::string AddressAttrKey(const std::string &data_var_name) {
  ICHECK(!data_var_name.empty())
      << "TPUv7 local allocation data variable must have a non-empty name";
  return std::string(kAddressAttrPrefix) + data_var_name;
}

inline bool IsLocalMemoryScope(const std::string &scope) {
  return scope == "shared" || scope == "shared.dyn" || scope == "local" ||
         scope == "local.fragment" || scope == "local.matrix";
}

inline int64_t DivUp(int64_t value, int64_t factor) {
  ICHECK_GT(factor, 0);
  return (value + factor - 1) / factor;
}

inline int64_t AlignUp(int64_t value, int64_t align) {
  ICHECK_GT(align, 0);
  return DivUp(value, align) * align;
}

inline int64_t ValidateDescriptorDim(int64_t value, const char *context) {
  ICHECK_GT(value, 0) << context << " expects a positive dimension";
  ICHECK_LE(value, kDescriptorDimMax)
      << context << " dimension " << value
      << " exceeds the TPUv7/PPL dim4 limit " << kDescriptorDimMax;
  return value;
}

inline int64_t GetIntImmValue(const PrimExpr &expr, const char *context) {
  auto *imm = expr.as<IntImmNode>();
  ICHECK(imm) << context << " expects compile-time integer dimensions";
  return ValidateDescriptorDim(imm->value, context);
}

template <typename Integer>
inline void ValidateDescriptorShape4(const std::vector<Integer> &shape4,
                                     const char *context) {
  ICHECK_EQ(shape4.size(), 4U) << context << " expects a normalized dim4";
  for (size_t axis = 0; axis < shape4.size(); ++axis) {
    ICHECK_GT(shape4[axis], 0)
        << context << " axis " << axis << " must be positive";
    ICHECK_LE(shape4[axis], kDescriptorDimMax)
        << context << " axis " << axis << " extent " << shape4[axis]
        << " exceeds the TPUv7/PPL dim4 limit " << kDescriptorDimMax;
  }
}

template <typename Integer>
inline int64_t DescriptorElementCount(const std::vector<Integer> &shape4,
                                      const char *context) {
  ValidateDescriptorShape4(shape4, context);
  int64_t count = 1;
  for (Integer extent : shape4) {
    ICHECK_LE(static_cast<int64_t>(extent),
              std::numeric_limits<int64_t>::max() / count)
        << context << " element count overflows int64";
    count *= static_cast<int64_t>(extent);
  }
  return count;
}

inline int64_t DTypeBytes(DataType dtype) {
  const int64_t bits = static_cast<int64_t>(dtype.bits()) * dtype.lanes();
  return std::max<int64_t>(1, DivUp(bits, 8));
}

inline std::vector<int64_t> NormalizeLocalShape(const Array<PrimExpr> &shape,
                                                const char *context) {
  if (shape.empty()) {
    LOG(FATAL) << context
               << " unsupported local tensor rank: 0; TPU descriptors require "
                  "rank 1 through 4";
  }
  if (shape.size() == 1) {
    return {1, 1, 1, GetIntImmValue(shape[0], context)};
  }
  if (shape.size() == 2) {
    return {1, GetIntImmValue(shape[0], context), 1,
            GetIntImmValue(shape[1], context)};
  }
  if (shape.size() == 3) {
    return {GetIntImmValue(shape[0], context),
            GetIntImmValue(shape[1], context), 1,
            GetIntImmValue(shape[2], context)};
  }
  if (shape.size() == 4) {
    return {
        GetIntImmValue(shape[0], context), GetIntImmValue(shape[1], context),
        GetIntImmValue(shape[2], context), GetIntImmValue(shape[3], context)};
  }
  LOG(FATAL) << context << " unsupported local tensor rank: " << shape.size();
  return {1, 1, 1, 1};
}

inline int64_t TpuAlignSizeBytesFromShape4(const std::vector<int64_t> &shape4,
                                           DataType dtype) {
  ValidateDescriptorShape4(shape4, "TPUv7 local tensor");
  int64_t dtype_bytes = DTypeBytes(dtype);
  int64_t eu_num = std::max<int64_t>(1, kEuBytes / dtype_bytes);
  int64_t stride_c = AlignUp(shape4[2] * shape4[3], eu_num);
  int64_t lane_groups = DivUp(shape4[1], kLaneNum);
  int64_t bytes = shape4[0] * lane_groups * stride_c * dtype_bytes;
  return AlignUp(std::max<int64_t>(bytes, kTensorAlignBytes),
                 kTensorAlignBytes);
}

inline int64_t TpuAlignSizeBytes(const Array<PrimExpr> &shape, DataType dtype,
                                 const char *context) {
  return TpuAlignSizeBytesFromShape4(NormalizeLocalShape(shape, context),
                                     dtype);
}

inline int64_t TpuMatrixSizeBytes(const Array<PrimExpr> &shape, DataType dtype,
                                  const char *context) {
  ICHECK_EQ(shape.size(), 2U) << context << " expects a rank-2 matrix";
  const int64_t rows = GetIntImmValue(shape[0], context);
  const int64_t cols = GetIntImmValue(shape[1], context);
  const int64_t elements_per_eu = kEuBytes / DTypeBytes(dtype);
  ICHECK_EQ(cols % elements_per_eu, 0)
      << context << " width must be a multiple of " << elements_per_eu;
  return TpuAlignSizeBytesFromShape4(
      {rows, DivUp(cols, elements_per_eu), 1, elements_per_eu}, dtype);
}

} // namespace tpuv7
} // namespace tl
} // namespace tvm

#endif // TVM_TL_TPUV7_LMEM_H_
