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

/*! \file target/codegen_rv.cc
 *  \brief SG2260E RV Tensor instruction selection for portable TPU ops.
 */

#include "codegen_tpu.h"

#include <tvm/runtime/logging.h>

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace tvm {
namespace codegen {

namespace {

std::string RVDTypeName(DataType dtype) {
  if (dtype == DataType::Float(32)) {
    return "DT_FP32";
  }
  if (dtype == DataType::Float(16)) {
    return "DT_FP16";
  }
  if (dtype == DataType::BFloat(16)) {
    return "DT_BFP16";
  }
  if (dtype.is_e4m3_float8()) {
    return "DT_FP8E4M3";
  }
  if (dtype.is_e5m2_float8()) {
    return "DT_FP8E5M2";
  }
  LOG(FATAL) << "RV Tensor floating-point lowering does not support dtype "
             << dtype;
  return "DT_FP32";
}

void CheckRVShape(const std::vector<int> &shape, const char *operation,
                  const char *operand) {
  ICHECK_EQ(shape.size(), 2U)
      << "RV Tensor " << operation << " expects a TileLang 2-D local tile for "
      << operand;
  for (int extent : shape) {
    ICHECK_GT(extent, 0) << "RV Tensor " << operation << " requires positive "
                         << operand << " extents";
    ICHECK_LT(extent, 1 << 16)
        << "RV Tensor " << operation << " descriptor extent exceeds 16 bits";
  }
}

} // namespace

void CodeGenTileLangTPU::EmitRVDescriptor(const std::string &tensor,
                                          int register_id, bool is_global,
                                          const std::string &dtype,
                                          bool hw_aligned) {
  const bool is_float = dtype == "DT_FP32" || dtype == "DT_FP16" ||
                        dtype == "DT_BFP16" || dtype == "DT_FP8E5M2" ||
                        dtype == "DT_FP8E4M3";
  PrintIndent();
  stream << (is_global ? "rvt_gr(" : "rvt_tr(") << register_id << ", PRECISION("
         << dtype << "), " << (is_float ? "FP8TYPE(" : "SIGN(") << dtype
         << "), " << tensor << ".addr, "
         << (hw_aligned ? "HW_ALIGN_LAYOUT" : "FREE_LAYOUT")
         << ", (array4_t){.n=" << tensor << ".shape.n, .c=" << tensor
         << ".shape.c, .h=" << tensor << ".shape.h, .w=" << tensor
         << ".shape.w}, "
         << (hw_aligned ? "(int *)NULL" : "(int *)&" + tensor + ".stride")
         << ");\n";
}

void CodeGenTileLangTPU::EmitRVCopy(const std::string &src, bool src_is_global,
                                    const std::string &src_dtype,
                                    const std::string &dst, bool dst_is_global,
                                    const std::string &dst_dtype) {
  // The DMA descriptors deliberately use FREE layout.  A TileLang region can
  // be a sub-view whose logical width is smaller than its parent's physical
  // HW-aligned row, so recomputing an aligned stride from the region shape is
  // not generally correct.
  const int src_register = src_is_global ? 32 : 8;
  // A single-global-operand transfer can reuse GR32.  GR33 is only needed
  // when both endpoints are global and GR32 is already occupied by src.
  const int dst_register = dst_is_global ? (src_is_global ? 33 : 32) : 9;
  EmitRVDescriptor(src, src_register, src_is_global, src_dtype, false);
  EmitRVDescriptor(dst, dst_register, dst_is_global, dst_dtype, false);

  if (src_dtype != dst_dtype) {
    ICHECK(!src_is_global && !dst_is_global)
        << "RV Tensor copy-and-convert requires two local tensors; DMA does "
           "not perform dtype conversion";
    auto is_supported_float = [](const std::string &dtype) {
      return dtype == "DT_FP16" || dtype == "DT_BFP16" ||
             dtype == "DT_FP32" || dtype == "DT_FP8E4M3" ||
             dtype == "DT_FP8E5M2";
    };
    ICHECK(is_supported_float(src_dtype) && is_supported_float(dst_dtype))
        << "RV Tensor rvt_cvt_f2f only accepts supported floating-point "
           "source/destination types; integer conversion requires a distinct "
           "rvt_cvt_i2i/i2f/f2i lowering";
    PrintIndent();
    stream << "rvt_cfg_satu(0, true);\n";
    PrintIndent();
    stream << "rvt_cfg_round_mode(0);\n";
    PrintIndent();
    stream << "rvt_cvt_f2f(" << dst_register << ", " << src_register << ");\n";
    return;
  }

  PrintIndent();
  if (src_is_global && !dst_is_global) {
    stream << "rvt_dma_ld(" << dst_register << ", " << src_register << ");\n";
  } else if (!src_is_global && dst_is_global) {
    stream << "rvt_dma_st(" << dst_register << ", " << src_register << ");\n";
  } else {
    stream << "rvt_dma_cp(" << dst_register << ", " << src_register << ");\n";
  }
}

void CodeGenTileLangTPU::EmitRVMatrixCopy(const std::string &src,
                                          bool src_is_global,
                                          const std::string &dst,
                                          bool dst_is_global, DataType dtype,
                                          int64_t rows, int64_t cols) {
  ICHECK_EQ(dtype, DataType::Float(32));
  constexpr int64_t kElementsPerEU = 16;
  ICHECK_EQ(cols % kElementsPerEU, 0)
      << "RV FP32 matrix width must be a multiple of 16";
  ICHECK_NE(src_is_global, dst_is_global)
      << "RV matrix copy requires exactly one global operand";
  const std::string matrix_shape =
      "(array4_t){.n=" + std::to_string(rows) +
      ", .c=" + std::to_string(cols / kElementsPerEU) +
      ", .h=1, .w=" + std::to_string(kElementsPerEU) + "}";
  auto emit_global = [&](const std::string &tensor, int register_id) {
    PrintIndent();
    stream << "rvt_gr(" << register_id
           << ", PRECISION(DT_FP32), FP8TYPE(DT_FP32), " << tensor
           << ".addr, FREE_LAYOUT, " << matrix_shape << ", (int[4]){" << tensor
           << ".stride.c, " << kElementsPerEU << ", " << kElementsPerEU
           << ", 1});\n";
  };
  auto emit_local = [&](const std::string &tensor, int register_id) {
    PrintIndent();
    stream << "rvt_tr(" << register_id
           << ", PRECISION(DT_FP32), FP8TYPE(DT_FP32), " << tensor
           << ".addr, HW_ALIGN_LAYOUT, " << matrix_shape << ", (int *)NULL);\n";
  };
  if (src_is_global) {
    emit_global(src, 32);
    emit_local(dst, 9);
    PrintIndent();
    stream << "rvt_dma_ld(9, 32);\n";
  } else {
    emit_local(src, 8);
    emit_global(dst, 32);
    PrintIndent();
    stream << "rvt_dma_st(32, 8);\n";
  }
}

void CodeGenTileLangTPU::EmitRVFill(const std::string &dst, DataType dtype,
                                    double value) {
  const auto type = RVDTypeName(dtype);
  EmitRVDescriptor(dst, 10, false, type, true);
  EmitRVConstant(value, type);
  stream << "rvt_cp(10, 1);\n";
}

void CodeGenTileLangTPU::EmitRVGemm(const std::string &a, const std::string &b,
                                    const std::string &c, DataType a_dtype,
                                    DataType b_dtype, DataType c_dtype,
                                    bool transpose_a, bool transpose_b,
                                    bool accumulate, int64_t m, int64_t n,
                                    int64_t k) {
  ICHECK(a_dtype == b_dtype)
      << "RV Tensor GEMM requires matching A/B dtypes, got " << a_dtype
      << " and " << b_dtype;
  const bool is_fp32 = a_dtype == DataType::Float(32);
  const bool is_fp8 = a_dtype.is_e4m3_float8() || a_dtype.is_e5m2_float8();
  ICHECK(is_fp32 || a_dtype == DataType::Float(16) ||
         a_dtype == DataType::BFloat(16) || is_fp8)
      << "RV Tensor GEMM accepts FP8, FP16, BF16, or FP32 TileLang inputs, got "
      << a_dtype;
  if (is_fp32) {
    ICHECK_EQ(c_dtype, DataType::Float(32))
        << "RV Tensor FP32 GEMM requires an FP32 output/accumulator";
    ICHECK(!transpose_b)
        << "RV Tensor FP32 GEMM uses rvt_fmm_nn and requires KxN weights; "
           "transpose_B is available only on the FP16/BF16 fmm2 path";
  } else {
    ICHECK(!transpose_a)
        << "RV Tensor FP16/BF16/FP8 fmm2 does not expose a standalone "
           "transpose-A form";
    ICHECK(
        (accumulate && c_dtype == DataType::Float(32)) ||
        (!accumulate && (c_dtype == DataType::Float(32) ||
                         (!is_fp8 && c_dtype == a_dtype))))
        << "RV Tensor accumulating and FP8 fmm2 forms require an FP32 C tile; "
           "FP16/BF16 overwrite mode also permits C to match A/B, got "
        << c_dtype;
  }
  ICHECK_GT(m, 0);
  ICHECK_GT(n, 0);
  ICHECK_GT(k, 0);
  ICHECK_LT(m, 1 << 16);
  ICHECK_LT(n, 1 << 16);
  ICHECK_LT(k, 1 << 16);

  if (is_fp32) {
    constexpr int64_t kElementsPerEU = 16;
    auto emit_matrix = [&](const std::string &tensor, int register_id,
                           int64_t rows, int64_t cols) {
      ICHECK_EQ(cols % kElementsPerEU, 0)
          << "RV FP32 matrix width must be a multiple of 16";
      PrintIndent();
      stream << "rvt_tr(" << register_id
             << ", PRECISION(DT_FP32), FP8TYPE(DT_FP32), " << tensor
             << ".addr, HW_ALIGN_LAYOUT, (array4_t){.n=" << rows
             << ", .c=" << cols / kElementsPerEU
             << ", .h=1, .w=" << kElementsPerEU << "}, (int *)NULL);\n";
    };
    emit_matrix(a, 8, transpose_a ? k : m, transpose_a ? m : k);
    emit_matrix(b, 9, k, n);
    emit_matrix(c, 10, m, n);
  } else {
    EmitRVDescriptor(a, 8, false, RVDTypeName(a_dtype), true);
    EmitRVDescriptor(b, 9, false, RVDTypeName(b_dtype), true);
    EmitRVDescriptor(c, 10, false, RVDTypeName(c_dtype), true);
  }
  PrintIndent();
  stream << "rvt_cfg_satu(0, false);\n";
  PrintIndent();
  if (is_fp32) {
    const char *instruction = transpose_a
                                  ? (accumulate ? "rvt_fmma_tn" : "rvt_fmm_tn")
                                  : (accumulate ? "rvt_fmma_nn" : "rvt_fmm_nn");
    stream << instruction
           << "(10, 8, 9, 0, 0);\n";
    return;
  }
  stream << "rvt_cfg_quant(0);\n";
  PrintIndent();
  // Accumulation is an explicit semantic bit; fmm2 has both variants.
  const char *instruction = nullptr;
  if (transpose_b) {
    instruction = accumulate ? "rvt_fmm2a_nt" : "rvt_fmm2_nt";
  } else {
    instruction = accumulate ? "rvt_fmm2a_nn" : "rvt_fmm2_nn";
  }
  stream << instruction << "(10, 8, 9, 0, 0, 0);\n";
}

void CodeGenTileLangTPU::EmitRVElementwise(
    const std::string &operation, const std::string &dst,
    const std::string &src0, const std::string &src1, DataType dst_dtype,
    DataType src0_dtype, DataType src1_dtype, const std::vector<int> &dst_shape,
    const std::vector<int> &src0_shape, const std::vector<int> &src1_shape) {
  CheckRVShape(dst_shape, operation.c_str(), "output");
  CheckRVShape(src0_shape, operation.c_str(), "lhs");
  CheckRVShape(src1_shape, operation.c_str(), "rhs");
  ICHECK(dst_dtype == src0_dtype && dst_dtype == src1_dtype)
      << "RV Tensor " << operation
      << " currently requires matching input/output dtypes";
  RVDTypeName(dst_dtype);
  if (operation == "div") {
    ICHECK(dst_dtype == DataType::Float(16) ||
           dst_dtype == DataType::BFloat(16) ||
           dst_dtype == DataType::Float(32))
        << "rvt_fdiv only supports FP16, BF16, and FP32";
  }
  ICHECK(dst_shape == src0_shape) << "RV Tensor " << operation
                                  << " requires output and lhs shapes to match";
  ICHECK(src1_shape == dst_shape ||
         (src1_shape[0] == dst_shape[0] && src1_shape[1] == 1))
      << "RV Tensor " << operation
      << " only supports an equal rhs shape or W-dimension broadcast for the "
         "current 2-D TileLang mapping";
  const bool rhs_w_broadcast = src1_shape != dst_shape;

  EmitRVDescriptor(src0, 8, false, RVDTypeName(src0_dtype), true);
  if (rhs_w_broadcast) {
    // The SG2260E CModel does not expand a shape-(M,1) TR merely because the
    // peer/output TR has width W. Describe the same storage as an (M,W) free
    // view with zero W stride. This is a descriptor-only broadcast: no LMEM
    // expansion, extra instruction, or out-of-bounds read is introduced.
    PrintIndent();
    stream << "rvt_tr(9, PRECISION(" << RVDTypeName(src1_dtype) << "), FP8TYPE("
           << RVDTypeName(src1_dtype) << "), " << src1
           << ".addr, FREE_LAYOUT, (array4_t){.n=" << dst
           << ".shape.n, .c=" << dst << ".shape.c, .h=" << dst
           << ".shape.h, .w=" << dst << ".shape.w}, (int[4]){" << src1
           << ".stride.n, " << src1 << ".stride.c, " << src1
           << ".stride.h, 0});\n";
  } else {
    EmitRVDescriptor(src1, 9, false, RVDTypeName(src1_dtype), true);
  }
  EmitRVDescriptor(dst, 10, false, RVDTypeName(dst_dtype), true);
  PrintIndent();
  stream << "rvt_cfg_satu(0, false);\n";
  PrintIndent();
  stream << "rvt_cfg_round_mode(0);\n";
  if (operation == "div") {
    PrintIndent();
    stream << "rvt_cfg_rsqrt_iter(3);\n";
  }
  PrintIndent();
  stream << "rvt_f" << operation << "(10, 8, 9);\n";
}

// Each extended operation owns CR1 and TR8..11 until its completion fence.
// Use SDK descriptor constructors (including SIGN for integer views), not
// the old backend's hand-encoded register words.
void CodeGenTileLangTPU::EmitRVConstant(double value,
                                        const std::string &dtype) {
  std::ostringstream literal;
  if (std::isinf(value)) {
    literal << (value > 0 ? "INFINITY" : "(-INFINITY)");
  } else if (std::isnan(value)) {
    literal << "NAN";
  } else {
    literal << std::scientific << std::setprecision(17) << value << "f";
  }
  stream << "{\nscalar_t rv_scalar = {.f32 = " << literal.str() << "};\n"
         << "rv_scalar = tpu_cast(rv_scalar, " << dtype
         << ", DT_FP32, RM_HALF_TO_EVEN);\n"
         << "uint64_t rv_bits = rv_scalar.u32;\n"
         << "rvt_cr(1, PRECISION(" << dtype << "), "
         << (dtype == "DT_INT32" ? "SIGN(" : "FP8TYPE(") << dtype
         << "), &rv_bits);\n}\n";
}

void CodeGenTileLangTPU::EmitRVScalar(const std::string &operation,
                                      const std::string &dst,
                                      const std::string &src, DataType dtype,
                                      double value) {
  const auto type = RVDTypeName(dtype);
  EmitRVDescriptor(src, 8, false, type, true);
  EmitRVDescriptor(dst, 10, false, type, true);
  EmitRVConstant(value, type);
  stream << "rvt_cfg_satu(0, false);\nrvt_cfg_round_mode(0);\n"
         << "rvt_f" << operation << "(10, 8, 1);\n";
}

void CodeGenTileLangTPU::EmitRVReduction(const std::string &operation,
                                         const std::string &src,
                                         const std::string &dst, DataType dtype,
                                         int width) {
  const auto type = RVDTypeName(dtype);
  EmitRVDescriptor(dst, 10, false, type, true);
  stream << "rvt_cfg_satu(0, false);\nrvt_cfg_round_mode(0);\n";
  // Both public reductions overwrite the destination. Starting max at the
  // first input also handles all-negative tiles without a finite sentinel.
  if (operation == "sum") {
    EmitRVConstant(0, type);
    stream << "rvt_cp(10, 1);\n";
  }
  stream << "{\nfor (int rv_column = 0; rv_column < " << width
         << "; ++rv_column) {\n"
         << "rvt_tr(8, PRECISION(" << type << "), FP8TYPE(" << type << "), "
         << src << ".addr + rv_column * " << dtype.bytes()
         << ", FREE_LAYOUT, (array4_t){.n=1, .c=" << src
         << ".shape.c, .h=1, .w=1}, (int[4]){" << src << ".stride.n, " << src
         << ".stride.c, " << src << ".stride.h, 1});\n";
  if (operation == "sum") {
    stream << "rvt_fadd(10, 10, 8);\n";
  } else {
    stream << "if (rv_column == 0) { rvt_cp(10, 8); } "
           << "else { rvt_fmax(10, 10, 8); }\n";
  }
  stream << "}\n}\n";
}

void CodeGenTileLangTPU::EmitRVExp(const std::string &dst,
                                   const std::string &work0,
                                   const std::string &work1, DataType dtype) {
  ICHECK(dtype == DataType::Float(32))
      << "RV exp currently requires FP32 local tensors";
  // Port the validated exp(x/2)^2 range reduction from the former RV backend.
  // dst=TR8, integer exponent/work0=TR9, polynomial/work1=TR10, input=TR11.
  // Workspaces and coefficient storage are validated by the shared semantic
  // parser. The RV polynomial does not read or initialize the coefficient tile.
  EmitRVDescriptor(dst, 8, false, "DT_FP32", true);
  EmitRVDescriptor(work0, 9, false, "DT_FP32", true);
  EmitRVDescriptor(work1, 10, false, "DT_FP32", true);
  stream << "rvt_cfg_satu(0, false);\nrvt_cfg_round_mode(0);\n";
  stream << "rvt_cp(10, 8);\n";
  EmitRVConstant(-104, "DT_FP32");
  stream << "rvt_fmax(8, 8, 1);\n";
  EmitRVConstant(89, "DT_FP32");
  stream << "rvt_fmin(8, 8, 1);\n";
  // Self-comparison on this SDK does not reliably classify unordered values.
  EmitRVDescriptor(work0, 9, false, "DT_INT32", true);
  stream << "{ uint64_t bits = 0x7fffffff;\n"
         << "rvt_cr(1, PRECISION(DT_INT32), SIGN(DT_INT32), &bits); }\n"
         << "rvt_and(9, 10, 1);\n"
         << "{ uint64_t bits = 0x7f800000;\n"
         << "rvt_cr(1, PRECISION(DT_INT32), SIGN(DT_INT32), &bits); }\n"
         << "rvt_cmpgt(8, 9, 1, 10, 8);\n";
  EmitRVConstant(0.5, "DT_FP32");
  stream << "rvt_fmul(8, 8, 1);\n";
  EmitRVConstant(1.4426950408889634, "DT_FP32");
  stream << "rvt_fmul(10, 8, 1);\n"
         << "rvt_cvt_f2i(9, 10);\nrvt_cvt_i2f(10, 9);\n";
  EmitRVConstant(0.6931471805599453, "DT_FP32");
  stream << "rvt_fmul(10, 10, 1);\nrvt_fsub(8, 8, 10);\n";
  const double coefficients[] = {1,        1,         0.5,       1.0 / 6,
                                 1.0 / 24, 1.0 / 120, 1.0 / 720, 1.0 / 5040};
  EmitRVConstant(coefficients[7], "DT_FP32");
  stream << "rvt_cp(10, 1);\n";
  for (int i = 6; i >= 0; --i) {
    stream << "rvt_fmul(10, 10, 8);\n";
    EmitRVConstant(coefficients[i], "DT_FP32");
    stream << "rvt_fadd(10, 10, 1);\n";
  }
  EmitRVConstant(127, "DT_INT32");
  stream << "rvt_add(9, 9, 1, 0, 0);\n";
  EmitRVConstant(8388608, "DT_INT32");
  stream << "rvt_mul(9, 9, 1, 0, 0);\n";
  EmitRVDescriptor(work0, 9, false, "DT_FP32", true);
  stream << "rvt_fmul(8, 10, 9);\nrvt_fmul(8, 8, 8);\n";
}

} // namespace codegen
} // namespace tvm
