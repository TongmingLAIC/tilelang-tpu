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

/*! \file target/codegen_tpukernel.cc
 *  \brief TPU-Kernel instruction selection for portable and backend-owned
 *  semantic TPU ops.
 */

#include "codegen_tpu.h"
#include "tpuv7_lmem.h"

#include <tvm/runtime/logging.h>

#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace tvm {
namespace codegen {

namespace {

void ValidateExpFamilyShape(const std::vector<int> &shape4,
                            const std::string &op_name) {
  tl::tpuv7::ValidateDescriptorShape4(shape4, op_name.c_str());
  const int64_t hw =
      static_cast<int64_t>(shape4[2]) * static_cast<int64_t>(shape4[3]);
  if (hw > tl::tpuv7::kDescriptorDimMax) {
    std::ostringstream message;
    message << op_name << " requires h*w <= " << tl::tpuv7::kDescriptorDimMax
            << " for tpu_bdc_fp_exp, got " << shape4[2] << "*" << shape4[3]
            << "=" << hw;
    // This is invalid user IR, not an internal compiler invariant.  Throw a
    // regular runtime error so callers get a deterministic diagnostic without
    // depending on the platform's native backtrace implementation.
    throw tvm::runtime::Error(message.str());
  }
}

void ValidateReductionAlignedWidth(int64_t aligned_width,
                                   const std::string &op_name) {
  if (aligned_width <= 0) {
    throw tvm::runtime::Error(op_name + " aligned width must be positive");
  }
  if (aligned_width > tl::tpuv7::kDescriptorDimMax) {
    std::ostringstream message;
    message << op_name << " aligned width " << aligned_width
            << " exceeds the TPUv7/PPL dim4 limit "
            << tl::tpuv7::kDescriptorDimMax
            << "; the reduction lowering materializes this width in padded "
               "dim4 descriptors";
    throw tvm::runtime::Error(message.str());
  }
}

const char *TPUKernelDTypeName(DataType dtype) {
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
  LOG(FATAL) << "TPU-Kernel floating-point op does not support dtype " << dtype;
  return "DT_FP32";
}

std::string FloatingLiteral(double value) {
  if (std::isnan(value)) {
    return "(0.0f / 0.0f)";
  }
  if (std::isinf(value)) {
    return value > 0 ? "(1.0f / 0.0f)" : "(-1.0f / 0.0f)";
  }
  std::ostringstream os;
  os << std::scientific << value << 'f';
  return os.str();
}

} // namespace

void CodeGenTileLangTPU::EmitTPUKernelCopy(
    const std::string &src, bool src_is_global, const std::string &src_dtype,
    const std::string &dst, bool dst_is_global, const std::string &dst_dtype) {
  PrintIndent();
  if (src_dtype != dst_dtype) {
    ICHECK(!src_is_global && !dst_is_global)
        << "TPU copy-and-convert currently requires two local tensors";
    auto is_supported_float = [](const std::string &dtype) {
      return dtype == "DT_FP16" || dtype == "DT_BFP16" ||
             dtype == "DT_FP32" || dtype == "DT_FP8E4M3" ||
             dtype == "DT_FP8E5M2";
    };
    ICHECK(is_supported_float(src_dtype) && is_supported_float(dst_dtype))
        << "TPU-Kernel copy-and-convert requires floating-point operands; got "
        << src_dtype << " -> " << dst_dtype;
    stream << "tpu_bdc_cast(" << dst << ".addr, " << src << ".addr, &" << dst
           << ".shape, (" << dst << ".default_stride ? NULL : &" << dst
           << ".stride), (" << src << ".default_stride ? NULL : &" << src
           << ".stride), " << dst_dtype << ", " << src_dtype
           << ", RM_HALF_TO_EVEN);\n";
    return;
  }

  const char *instruction = nullptr;
  if (src_is_global && !dst_is_global) {
    instruction = "tpu_gdma_cpy_S2L";
  } else if (!src_is_global && dst_is_global) {
    instruction = "tpu_gdma_cpy_L2S";
  } else if (src_is_global && dst_is_global) {
    instruction = "tpu_gdma_cpy_S2S";
  } else {
    instruction = "tpu_bdc_cpy";
  }
  stream << instruction << "(" << dst << ".addr, " << src << ".addr, &" << dst
         << ".shape, (" << dst << ".default_stride ? NULL : &" << dst
         << ".stride), (" << src << ".default_stride ? NULL : &" << src
         << ".stride), " << src_dtype << ");\n";
}

void CodeGenTileLangTPU::EmitTPUKernelMatrixCopy(
    const std::string &src, bool src_is_global, const std::string &dst,
    bool dst_is_global, DataType dtype, int64_t rows, int64_t cols) {
  ICHECK_EQ(dtype, DataType::Float(32));
  constexpr int64_t kElementsPerEU = 16;
  ICHECK_EQ(cols % kElementsPerEU, 0)
      << "TPU-Kernel FP32 matrix width must be a multiple of 16";
  ICHECK_NE(src_is_global, dst_is_global)
      << "TPU-Kernel matrix copy requires exactly one global operand";
  PrintIndent();
  if (src_is_global) {
    stream << "tpu_gdma_matrix_S2L(" << dst << ".addr, " << src << ".addr, "
           << rows << ", " << cols << ", " << kElementsPerEU << ", " << src
           << ".stride.c, DT_FP32);\n";
  } else {
    stream << "tpu_gdma_matrix_L2S(" << dst << ".addr, " << src << ".addr, "
           << rows << ", " << cols << ", " << kElementsPerEU << ", " << dst
           << ".stride.c, DT_FP32);\n";
  }
}

void CodeGenTileLangTPU::EmitTPUKernelFill(const std::string &dst,
                                           DataType dtype, double value) {
  const char *scalar_field = nullptr;
  const bool is_fp8 = dtype.is_e4m3_float8() || dtype.is_e5m2_float8();
  if (dtype == DataType::Float(16)) {
    scalar_field = "f16";
  } else if (dtype == DataType::Float(32)) {
    scalar_field = "f32";
  } else if (dtype == DataType::BFloat(16)) {
    scalar_field = "bf16";
  } else if (is_fp8) {
    scalar_field = "f32";
  } else {
    LOG(FATAL) << "TPU-Kernel fill does not support dtype " << dtype;
  }
  const char *dtype_name = TPUKernelDTypeName(dtype);

  PrintIndent();
  stream << "{\n";
  PrintIndent();
  stream << "scalar_t " << dst << "_scalar_" << scalar_field;
  stream << " = {.f32 = " << FloatingLiteral(value) << "};\n";
  if (dtype != DataType::Float(32)) {
    PrintIndent();
    stream << dst << "_scalar_" << scalar_field << " = tpu_cast(" << dst
           << "_scalar_" << scalar_field << ", " << dtype_name
           << ", DT_FP32, RM_HALF_TO_EVEN);\n";
  }
  PrintIndent();
  stream << "tpu_bdc_set_C(" << dst << ".addr, " << dst << "_scalar_"
         << scalar_field << ", &" << dst << ".shape, (" << dst
         << ".default_stride ? NULL : &" << dst << ".stride), " << dtype_name
         << ");\n";
  PrintIndent();
  stream << "}\n";
}

void CodeGenTileLangTPU::EmitTPUKernelGemm(
    const std::string &a, const std::string &b, const std::string &c,
    DataType a_dtype, DataType b_dtype, DataType c_dtype, bool transpose_a,
    bool transpose_b, bool accumulate, int64_t m, int64_t n, int64_t k) {
  ICHECK(a_dtype == b_dtype)
      << "TPU-Kernel GEMM requires matching input dtypes, got " << a_dtype
      << " and " << b_dtype;
  const bool is_fp32 = a_dtype == DataType::Float(32);
  const bool is_fp8 = a_dtype.is_e4m3_float8() || a_dtype.is_e5m2_float8();
  ICHECK(is_fp32 || a_dtype == DataType::Float(16) ||
         a_dtype == DataType::BFloat(16) || is_fp8)
      << "TPU-Kernel GEMM requires FP32, FP16, BF16, or matching FP8 inputs, "
         "got "
      << a_dtype;
  if (is_fp32) {
    ICHECK_EQ(c_dtype, DataType::Float(32))
        << "TPU-Kernel FP32 GEMM requires an FP32 output/accumulator";
    ICHECK(!transpose_b)
        << "TPU-Kernel FP32 GEMM currently requires KxN weights; the SDK "
           "does not expose an FP32 right-transpose form";
    PrintIndent();
    stream << (transpose_a ? "tpu_bdc_fp32_mm_L_trans(" : "tpu_bdc_fp32_mm(")
           << c << ".addr, " << a << ".addr, " << b << ".addr, 0, "
           << (transpose_a ? k : m) << ", " << (transpose_a ? m : k) << ", "
           << n << ", 16, 16, false, "
           << (accumulate ? "true" : "false") << ");\n";
    return;
  }
  ICHECK(!transpose_a)
      << "TPU-Kernel FP16/BF16/FP8 GEMM does not expose a standalone "
         "transpose-A form";
  if (is_fp8) {
    ICHECK_EQ(c_dtype, DataType::Float(32))
        << "TPU-Kernel FP8 GEMM requires an FP32 output/accumulator";
    const char *input_dtype = TPUKernelDTypeName(a_dtype);
    const char *output_dtype = TPUKernelDTypeName(c_dtype);
    PrintIndent();
    stream << (transpose_b ? "tpu_bdc_fp8_mm_R_trans(" : "tpu_bdc_fp8_mm(") << c
           << ".addr, " << a << ".addr, " << b << ".addr, " << m << ", " << k
           << ", " << n << ", " << output_dtype << ", " << input_dtype << ", "
           << input_dtype << ", " << (accumulate ? "true" : "false")
           << ", false, false, (var_context_t){0});\n";
    return;
  }
  ICHECK(c_dtype == DataType::Float(32) || (!accumulate && c_dtype == a_dtype))
      << "TPU-Kernel NN accumulation requires an FP32 C tile; "
         "non-accumulating forms additionally support C matching A/B";
  ICHECK(!transpose_b || !accumulate)
      << "TPU-Kernel has no accumulating right-transpose GEMM instruction; "
         "use accumulate=False or materialize the transpose";
  const char *input_dtype = TPUKernelDTypeName(a_dtype);
  const char *output_dtype = TPUKernelDTypeName(c_dtype);

  PrintIndent();
  if (!transpose_b) {
    stream << "tpu_bdc_fp_mm(" << c << ".addr, " << a << ".addr, " << b
           << ".addr, " << m << ", " << k << ", " << n << ", " << output_dtype
           << ", " << input_dtype << ", " << (accumulate ? "true" : "false")
           << ");\n";
  } else {
    stream << "tpu_bdc_fp_mm_R_trans(" << c << ".addr, " << a << ".addr, " << b
           << ".addr, " << m << ", " << k << ", " << n << ", " << output_dtype
           << ", " << input_dtype << ");\n";
  }
}

void CodeGenTileLangTPU::EmitTPUKernelElementwise(
    const std::string &operation, const std::string &dst,
    const std::string &src0, const std::string &src1, DataType dtype,
    const std::vector<int> &src0_shape, const std::vector<int> &src1_shape) {
  ICHECK(operation == "add" || operation == "sub" || operation == "mul" ||
         operation == "div" || operation == "max")
      << "Unsupported TPU-Kernel elementwise operation " << operation;
  const std::string dtype_name = TPUKernelDTypeName(dtype);
  std::string src1_stride;
  if (src1_shape[1] == 1 && src0_shape[1] != 1) {
    const std::string stride_var = name_supply_->FreshName(src1 + "_stride");
    PrintIndent();
    stream << "dim4 " << stride_var << ";\n";
    PrintIndent();
    stream << "tpu_aligned_stride(&" << stride_var << ", 0, &" << src1
           << ".shape, " << dtype_name << ");\n";
    PrintIndent();
    stream << stride_var << ".w = 0;\n";
    src1_stride = "&" + stride_var + ", ";
  } else {
    ICHECK_EQ(src1_shape[1], src0_shape[1]);
    src1_stride =
        "(" + src1 + ".default_stride ? NULL : &" + src1 + ".stride), ";
  }
  PrintIndent();
  stream << (operation == "max" ? "tpu_bdc_max" : "tpu_bdc_fp_" + operation)
         << "(" << dst << ".addr, " << src0 << ".addr, " << src1 << ".addr, &"
         << dst << ".shape, (" << dst << ".default_stride ? NULL : &" << dst
         << ".stride), (" << src0 << ".default_stride ? NULL : &" << src0
         << ".stride), " << src1_stride << dtype_name << ");\n";
}

void CodeGenTileLangTPU::EmitTPUKernelScalar(const std::string &operation,
                                             const std::string &dst,
                                             const std::string &src,
                                             DataType dtype, double value) {
  ICHECK(operation == "add" || operation == "mul")
      << "Unsupported TPU-Kernel scalar operation " << operation;
  ICHECK(dtype == DataType::Float(16) || dtype == DataType::BFloat(16) ||
         dtype == DataType::Float(32) || dtype.is_e4m3_float8() ||
         dtype.is_e5m2_float8())
      << "TPU-Kernel scalar " << operation
      << " supports FP8, FP16, BF16, and FP32, got " << dtype;
  ICHECK(std::isfinite(value))
      << "TPU-Kernel scalar " << operation << " requires a finite literal";
  const char *dtype_name = TPUKernelDTypeName(dtype);
  const std::string scalar = name_supply_->FreshName("tpukernel_scalar");

  PrintIndent();
  stream << "{\n";
  PrintIndent();
  stream << "scalar_t " << scalar << " = {.f32 = " << FloatingLiteral(value)
         << "};\n";
  if (dtype != DataType::Float(32)) {
    PrintIndent();
    stream << scalar << " = tpu_cast(" << scalar << ", " << dtype_name
           << ", DT_FP32, RM_HALF_TO_EVEN);\n";
  }
  PrintIndent();
  stream << "tpu_bdc_fp_" << operation << "_C(" << dst << ".addr, " << src
         << ".addr, " << scalar << ", &" << dst << ".shape, (" << dst
         << ".default_stride ? NULL : &" << dst << ".stride), (" << src
         << ".default_stride ? NULL : &" << src << ".stride), " << dtype_name
         << ");\n";
  PrintIndent();
  stream << "}\n";
}

// The former TPU-Kernel-only entry point was named TryEmitTPUKernelSemantic.
// Portable operations now dispatch through this shared selector.
bool CodeGenTileLangTPU::TryEmitTPUSemantic(const CallNode *op,
                                            const std::string &op_name) {
  auto handle_elementwise_const = [&, this](const std::string &semantic_name,
                                            const std::string &operation) {
    ICHECK_EQ(op->args.size(), 4U)
        << semantic_name << " expects dst, src, and one floating literal";
    auto dst_operand =
        ParseWholeBufferRegion(op->args[1], semantic_name + " dst", 2);
    auto src_operand =
        ParseWholeBufferRegion(op->args[2], semantic_name + " src", 1);
    ICHECK(dst_operand.is_local && src_operand.is_local)
        << semantic_name << " operands must reside in local memory";
    const std::string &dst = dst_operand.descriptor;
    const std::string &src = src_operand.descriptor;
    DataType dst_dtype = dst_operand.dtype;
    DataType src_dtype = src_operand.dtype;
    ICHECK_EQ(dst_dtype, src_dtype)
        << semantic_name << " requires matching dst/src dtypes";
    ICHECK(dst_dtype == DataType::Float(16) ||
           dst_dtype == DataType::BFloat(16) ||
           dst_dtype == DataType::Float(32) || dst_dtype.is_e4m3_float8() ||
           dst_dtype.is_e5m2_float8())
        << semantic_name << " supports only FP8, FP16, BF16, and FP32, got "
        << dst_dtype;
    ICHECK_EQ(dst_operand.rank, src_operand.rank)
        << semantic_name << " requires matching dst/src ranks";
    ICHECK(dst_operand.shape4 == src_operand.shape4)
        << semantic_name << " requires matching dst/src shapes";
    const auto *value_node = op->args[3].as<FloatImmNode>();
    ICHECK(value_node) << semantic_name
                       << " currently requires a floating literal";
    double value = value_node->value;
    ICHECK(std::isfinite(value))
        << semantic_name << " requires a finite scalar literal";
    if (target_programming_model_ == "rv") {
      EmitRVScalar(operation, dst, src, dst_dtype, value);
    } else {
      EmitTPUKernelScalar(operation, dst, src, dst_dtype, value);
    }
  };
  if (op_name == "tl.tpu.mul_scalar") {
    handle_elementwise_const(op_name, "mul");
  } else if (op_name == "tl.tpu.add_scalar") {
    handle_elementwise_const(op_name, "add");
  } else if (op_name == "tl.tpu.exp") {
    ICHECK_EQ(op->args.size(), 5U)
        << op_name << " expects out, work0, work1, and coeff";
    std::array<SemanticTensorOperand, 4> operands{};
    std::array<std::string, 4> tensors{};
    for (size_t i = 0; i < operands.size(); ++i) {
      operands[i] = ParseWholeBufferRegion(
          op->args[i + 1], op_name + " operand " + std::to_string(i), 3);
      ICHECK(operands[i].is_local)
          << op_name << " operands must all reside in local memory";
      tensors[i] = operands[i].descriptor;
    }
    DataType dtype = operands[0].dtype;
    ICHECK(dtype == DataType::Float(16) || dtype == DataType::BFloat(16) ||
           dtype == DataType::Float(32))
        << op_name << " supports only FP16, BF16, and FP32, got " << dtype;
    for (size_t i = 1; i < operands.size(); ++i) {
      ICHECK_EQ(operands[i].dtype, dtype)
          << op_name << " requires matching operand dtypes";
    }
    for (size_t i = 0; i < operands.size(); ++i) {
      for (size_t j = i + 1; j < operands.size(); ++j) {
        ICHECK(operands[i].data_var != operands[j].data_var)
            << op_name << " requires distinct storage for every operand";
      }
    }
    const size_t payload_rank = operands[0].rank;
    ICHECK_EQ(operands[1].rank, payload_rank);
    ICHECK_EQ(operands[2].rank, payload_rank);
    ICHECK_EQ(operands[3].rank, 2U)
        << op_name << " coefficient buffer must be rank 2";
    ICHECK(operands[0].shape4 == operands[1].shape4 &&
           operands[0].shape4 == operands[2].shape4)
        << op_name << " requires out/work0/work1 to have matching shapes";
    ValidateExpFamilyShape(operands[0].shape4, op_name);
    ICHECK(operands[3].shape4 == std::vector<int>({1, 64, 1, 32}))
        << op_name << " coefficient buffer must have shape (64, 32)";
    if (target_programming_model_ == "rv") {
      EmitRVExp(tensors[0], tensors[1], tensors[2], dtype);
      return true;
    }
    std::string dtype_name = TPUKernelDTypeName(dtype);
    this->PrintIndent();
    this->stream << "tpu_bdc_load_fp_exp_coeff(" << tensors[3] << ".addr, "
                 << dtype_name << ");\n";
    this->PrintIndent();
    this->stream << "tpu_bdc_fp_exp(" << tensors[0] << ".addr, " << tensors[0]
                 << ".addr, " << tensors[1] << ".addr, " << tensors[2]
                 << ".addr, " << tensors[3] << ".addr, &" << tensors[0]
                 << ".shape, " << dtype_name << ");\n";
  } else if (op_name == "tl.tpu.reduce_max") {
    ICHECK_EQ(op->args.size(), 7U)
        << op_name
        << " expects input, output, scratch, eu_num, align_w, and stride";
    std::array<SemanticTensorOperand, 3> operands{};
    constexpr std::array<int, 3> kAccessMasks = {3, 2, 3};
    for (size_t i = 0; i < operands.size(); ++i) {
      operands[i] = ParseWholeBufferRegion(
          op->args[i + 1], op_name + " operand " + std::to_string(i),
          kAccessMasks[i]);
      ICHECK(operands[i].is_local)
          << op_name << " operands must all reside in local memory";
    }
    ICHECK(operands[0].data_var != operands[1].data_var &&
           operands[0].data_var != operands[2].data_var &&
           operands[1].data_var != operands[2].data_var)
        << op_name << " requires distinct input, output, and scratch storage";
    for (const auto &operand : operands) {
      ICHECK_EQ(operand.rank, 2U)
          << op_name << " requires rank-2 input, output, and scratch tensors";
    }
    // Resolve the input, output, and scratch descriptors.
    const auto &input_tensor = operands[0].descriptor;
    const auto &output_tensor = operands[1].descriptor;
    const auto &tmp_tensor = operands[2].descriptor;
    const auto *eu_imm = op->args[4].as<IntImmNode>();
    const auto *align_imm = op->args[5].as<IntImmNode>();
    const auto *stride_imm = op->args[6].as<IntImmNode>();
    ICHECK(eu_imm && align_imm && stride_imm)
        << op_name << " shape parameters must be compile-time integers";
    int64_t eu_num = eu_imm->value;
    int64_t align_w = align_imm->value;
    int64_t stride_n = stride_imm->value;
    // Select the floating-point format used by the pool sequence.
    auto dtype_ = operands[0].dtype;
    const bool is_fp8 =
        dtype_.is_e4m3_float8() || dtype_.is_e5m2_float8();
    ICHECK(dtype_ == DataType::Float(16) ||
           dtype_ == DataType::BFloat(16) ||
           dtype_ == DataType::Float(32) || is_fp8)
        << op_name << " supports FP8, FP16, BF16, and FP32, got " << dtype_;
    std::string dtype;
    if (dtype_ == DataType::Float(16)) {
      dtype = "DT_FP16";
    } else if (dtype_ == DataType::Float(32)) {
      dtype = "DT_FP32";
    } else if (dtype_ == DataType::BFloat(16)) {
      dtype = "DT_BFP16";
    } else if (is_fp8) {
      dtype = TPUKernelDTypeName(dtype_);
    }
    for (size_t i = 1; i < operands.size(); ++i) {
      ICHECK_EQ(operands[i].dtype, dtype_)
          << op_name << " requires matching input/output/scratch dtypes";
    }
    const std::vector<int> input_shape = {operands[0].shape4[1],
                                          operands[0].shape4[3]};
    const std::vector<int> output_shape = {operands[1].shape4[1],
                                           operands[1].shape4[3]};
    const std::vector<int> tmp_shape = {operands[2].shape4[1],
                                        operands[2].shape4[3]};
    ICHECK_GT(input_shape[0], 0);
    ICHECK_GT(input_shape[1], 0);
    int64_t expected_eu = tl::tpuv7::kEuBytes / (dtype_.bits() / 8);
    int64_t expected_align_w =
        ((input_shape[1] + expected_eu - 1) / expected_eu) * expected_eu;
    int64_t expected_stride_n =
        ((input_shape[0] + tl::tpuv7::kLaneNum - 1) / tl::tpuv7::kLaneNum) *
        expected_align_w;
    ICHECK_EQ(eu_num, expected_eu)
        << op_name << " eu_num must be derived from the 64-byte TPUv7 EU";
    ICHECK_EQ(align_w, expected_align_w)
        << op_name << " align_w disagrees with the input dtype/width";
    ValidateReductionAlignedWidth(expected_align_w, op_name);
    ICHECK_EQ(stride_n, expected_stride_n)
        << op_name << " stride disagrees with the TPUv7 local layout";
    ICHECK_EQ(output_shape[0], input_shape[0]);
    ICHECK_EQ(output_shape[1], 1);
    ICHECK_EQ(tmp_shape[0], input_shape[0]);
    ICHECK_EQ(tmp_shape[1], expected_eu)
        << op_name << " scratch width must equal the dtype-specific EU size";
    if (target_programming_model_ == "rv") {
      EmitRVReduction("max", input_tensor, output_tensor, dtype_,
                      input_shape[1]);
      return true;
    }

    this->PrintIndent();
    int sid = this->BeginScope();
    this->stream << "{\n";

    // Check the EU width and derived padded layout supplied by the frontend.
    this->PrintIndent();
    this->stream << "int eu_num = " << eu_num << ";\n";
    this->PrintIndent();
    this->stream << "int align_w = " << align_w << ";\n";

    // Build pad_val. FP_NEG_MAX returns the integer bit pattern of -MAX,
    // so we must populate the `u32` field to reinterpret the bits,
    // not the float field which would cast the integer to float.
    this->PrintIndent();
    this->stream << "scalar_t pad_val = {.u32 = FP_NEG_MAX(" << dtype
                 << ")};\n";

    // Build the two descriptor views used by the reduction stages.
    this->PrintIndent();
    this->stream << "dim4 in_reduce_h = {" << input_tensor << ".shape.n, "
                 << input_tensor << ".shape.c, align_w / eu_num, eu_num};\n";
    this->PrintIndent();
    this->stream << "dim4 out_reduce_h = {" << input_tensor << ".shape.n, "
                 << input_tensor << ".shape.c, 1, eu_num};\n";
    this->PrintIndent();
    this->stream << "dim4 in_reduce_w = {" << input_tensor << ".shape.n, "
                 << input_tensor << ".shape.c, 1, eu_num};\n";
    this->PrintIndent();
    this->stream << "dim4 out_reduce_w = {" << input_tensor << ".shape.n, "
                 << input_tensor << ".shape.c, 1, 1};\n";

    // Configure the pool geometry.
    this->PrintIndent();
    this->stream << "dim2 kernel = {align_w / eu_num, 1};\n";
    this->PrintIndent();
    this->stream << "padding_t pad = {0, 0, 0, 0};\n";
    this->PrintIndent();
    this->stream << "dim2 stride = {1, 1};\n";
    this->PrintIndent();
    this->stream << "dim2 dilation = {1, 1};\n";

    this->PrintIndent();
    this->stream << "if (align_w > " << input_tensor
                 << ".shape.w && align_w == eu_num) {\n";
    this->PrintIndent();
    this->stream << "  dim4 padded_stride = {" << stride_n
                 << ", align_w, align_w, 1};\n";
    this->PrintIndent();
    this->stream << "  dim4 padded_shape = {" << input_tensor << ".shape.n, "
                 << input_tensor << ".shape.c, 1, align_w};\n";
    this->PrintIndent();
    this->stream << "  dim4 copy_shape = {" << input_tensor << ".shape.n, "
                 << input_tensor << ".shape.c, 1, " << input_tensor
                 << ".shape.w};\n";
    this->PrintIndent();
    this->stream << "  __tilelang_tpu_tensor_info padded_input = {.shape = "
                 << "padded_shape, .stride = padded_stride, .addr = "
                 << tmp_tensor << ".addr, .default_stride = false};\n";
    this->PrintIndent();
    this->stream
        << "  __tilelang_tpu_tensor_info input_copy = {.shape = copy_shape, "
        << ".stride = {0}, .addr = " << input_tensor
        << ".addr, .default_stride = true};\n";
    this->PrintIndent();
    this->stream
        << "  __tilelang_tpu_tensor_info padded_input_copy = {.shape = "
        << "copy_shape, .stride = padded_stride, .addr = " << tmp_tensor
        << ".addr, .default_stride = false};\n";
    this->PrintIndent();
    this->stream << "  __tilelang_tpu_tensor_info output_view = {.shape = "
                 << "out_reduce_w, .stride = {0}, .addr = " << output_tensor
                 << ".addr, .default_stride = true};\n";
    this->PrintIndent();
    this->stream << "  tpu_bdc_set_C(padded_input.addr, pad_val, "
                 << "&padded_shape, &padded_input.stride, " << dtype << ");\n";
    this->PrintIndent();
    this->stream << "  tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, "
                 << "&copy_shape, &padded_input_copy.stride, "
                 << "(input_copy.default_stride ? NULL : &input_copy.stride), "
                 << dtype << ");\n";
    this->PrintIndent();
    this->stream << "  dim2 kernel2 = {1, eu_num};\n";
    this->PrintIndent();
    this->stream << "  pad_val.u32 = FP_NEG_MAX(" << dtype << ");\n";
    this->PrintIndent();
    this->stream << "  tpu_bdc_fp_max_pool2d(output_view.addr, "
                 << "padded_input.addr, &padded_shape, &kernel2, &pad, "
                 << "&stride, &dilation, " << dtype << ", pad_val);\n";
    this->PrintIndent();
    this->stream << "} else {\n";

    this->PrintIndent();
    this->stream << "  if (align_w > " << input_tensor << ".shape.w) {\n";
    this->PrintIndent();
    this->stream << "    dim4 fill_shape = {" << input_tensor << ".shape.n, "
                 << input_tensor << ".shape.c, 1, align_w - " << input_tensor
                 << ".shape.w};\n";
    this->PrintIndent();
    int elem_size = dtype_.bytes();
    this->stream << "    int elem_size = " << elem_size << ";\n";
    this->PrintIndent();
    this->stream << "    int offset = " << input_tensor
                 << ".shape.w * elem_size;\n";
    this->PrintIndent();
    this->stream << "    dim4 fill_tensor_stride = {" << stride_n
                 << ", align_w, " << input_tensor << ".shape.w, 1};\n";
    this->PrintIndent();
    this->stream << "    __tilelang_tpu_tensor_info fill_tensor = {.shape = "
                    "fill_shape, .stride "
                    "= fill_tensor_stride, "
                 << ".addr = " << input_tensor
                 << ".addr + offset, .default_stride = false};\n";
    this->PrintIndent();
    this->stream
        << "    tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, "
        << "(fill_tensor.default_stride ? NULL : &fill_tensor.stride), "
        << dtype << ");\n";
    this->PrintIndent();
    this->stream << "  }\n";

    this->PrintIndent();
    this->stream
        << "  __tilelang_tpu_tensor_info input_view = {.shape = in_reduce_h, "
           ".stride = {0}, "
        << ".addr = " << input_tensor << ".addr, .default_stride = true};\n";

    this->PrintIndent();
    this->stream
        << "  __tilelang_tpu_tensor_info tmp_view = {.shape = out_reduce_h, "
           ".stride = {0}, "
        << ".addr = " << tmp_tensor << ".addr, .default_stride = true};\n";

    this->PrintIndent();
    this->stream << "  tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr, "
                    "&input_view.shape, "
                 << "&kernel, &pad, &stride, &dilation, " << dtype
                 << ", pad_val);\n";
    this->PrintIndent();
    this->stream << "  dim2 kernel2 = {1, eu_num};\n";
    this->PrintIndent();
    this->stream
        << "  __tilelang_tpu_tensor_info output_view = {.shape = out_reduce_w, "
           ".stride = {0}, "
        << ".addr = " << output_tensor << ".addr, .default_stride = true};\n";
    this->PrintIndent();
    this->stream
        << "  __tilelang_tpu_tensor_info tmp_view2 = {.shape = in_reduce_w, "
           ".stride = {0}, "
        << ".addr = " << tmp_tensor << ".addr, .default_stride = true};\n";
    this->PrintIndent();
    this->stream << "  pad_val.u32 = FP_NEG_MAX(" << dtype << ");\n";
    this->PrintIndent();
    this->stream << "  tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr, "
                    "&tmp_view2.shape, "
                 << "&kernel2, &pad, &stride, &dilation, " << dtype
                 << ", pad_val);\n";
    this->PrintIndent();
    this->stream << "}\n";

    // Close the composite operation scope.
    this->EndScope(sid);
    this->PrintIndent();
    this->stream << "}\n";
  } else if (op_name == "tl.tpu.reduce_sum") {
    ICHECK_EQ(op->args.size(), 7U)
        << op_name
        << " expects input, output, scratch, eu_num, align_w, and stride";
    std::array<SemanticTensorOperand, 3> operands{};
    constexpr std::array<int, 3> kAccessMasks = {3, 2, 3};
    for (size_t i = 0; i < operands.size(); ++i) {
      operands[i] = ParseWholeBufferRegion(
          op->args[i + 1], op_name + " operand " + std::to_string(i),
          kAccessMasks[i]);
      ICHECK(operands[i].is_local)
          << op_name << " operands must all reside in local memory";
    }
    ICHECK(operands[0].data_var != operands[1].data_var &&
           operands[0].data_var != operands[2].data_var &&
           operands[1].data_var != operands[2].data_var)
        << op_name << " requires distinct input, output, and scratch storage";
    for (const auto &operand : operands) {
      ICHECK_EQ(operand.rank, 2U)
          << op_name << " requires rank-2 input, output, and scratch tensors";
    }
    // Resolve the input, output, and scratch descriptors.
    const auto &input_tensor = operands[0].descriptor;
    const auto &output_tensor = operands[1].descriptor;
    const auto &tmp_tensor = operands[2].descriptor;
    const auto *eu_imm = op->args[4].as<IntImmNode>();
    const auto *align_imm = op->args[5].as<IntImmNode>();
    const auto *stride_imm = op->args[6].as<IntImmNode>();
    ICHECK(eu_imm && align_imm && stride_imm)
        << op_name << " shape parameters must be compile-time integers";
    int64_t eu_num = eu_imm->value;
    int64_t align_w = align_imm->value;
    int64_t stride_n = stride_imm->value;

    auto dtype_ = operands[0].dtype;
    const bool is_fp8 =
        dtype_.is_e4m3_float8() || dtype_.is_e5m2_float8();
    ICHECK(dtype_ == DataType::Float(16) ||
           dtype_ == DataType::BFloat(16) ||
           dtype_ == DataType::Float(32) || is_fp8)
        << op_name << " supports FP8, FP16, BF16, and FP32, got " << dtype_;
    if (is_fp8 && target_programming_model_ != "rv") {
      throw tvm::runtime::Error(
          op_name +
          " supports FP8 only with RV Tensor; TPU-Kernel supports FP16, "
          "BF16, and FP32");
    }
    std::string dtype, dtype_2;
    if (dtype_ == DataType::Float(16)) {
      dtype = "DT_FP16";
      dtype_2 = "f16";
    } else if (dtype_ == DataType::Float(32)) {
      dtype = "DT_FP32";
      dtype_2 = "f32";
    } else if (dtype_ == DataType::BFloat(16)) {
      dtype = "DT_BFP16";
      dtype_2 = "bf16";
    } else if (is_fp8) {
      dtype = TPUKernelDTypeName(dtype_);
      dtype_2 = "u32";
    }
    for (size_t i = 1; i < operands.size(); ++i) {
      ICHECK_EQ(operands[i].dtype, dtype_)
          << op_name << " requires matching input/output/scratch dtypes";
    }
    const std::vector<int> input_shape = {operands[0].shape4[1],
                                          operands[0].shape4[3]};
    const std::vector<int> output_shape = {operands[1].shape4[1],
                                           operands[1].shape4[3]};
    const std::vector<int> tmp_shape = {operands[2].shape4[1],
                                        operands[2].shape4[3]};
    ICHECK_GT(input_shape[0], 0);
    ICHECK_GT(input_shape[1], 0);
    int64_t expected_eu = tl::tpuv7::kEuBytes / (dtype_.bits() / 8);
    int64_t expected_align_w =
        ((input_shape[1] + expected_eu - 1) / expected_eu) * expected_eu;
    int64_t expected_stride_n =
        ((input_shape[0] + tl::tpuv7::kLaneNum - 1) / tl::tpuv7::kLaneNum) *
        expected_align_w;
    ICHECK_EQ(eu_num, expected_eu)
        << op_name << " eu_num must be derived from the 64-byte TPUv7 EU";
    ICHECK_EQ(align_w, expected_align_w)
        << op_name << " align_w disagrees with the input dtype/width";
    ValidateReductionAlignedWidth(expected_align_w, op_name);
    ICHECK_EQ(stride_n, expected_stride_n)
        << op_name << " stride disagrees with the TPUv7 local layout";
    ICHECK_EQ(output_shape[0], input_shape[0]);
    ICHECK_EQ(output_shape[1], 1);
    ICHECK_EQ(tmp_shape[0], input_shape[0]);
    ICHECK_EQ(tmp_shape[1], expected_eu)
        << op_name << " scratch width must equal the dtype-specific EU size";
    if (target_programming_model_ == "rv") {
      EmitRVReduction("sum", input_tensor, output_tensor, dtype_,
                      input_shape[1]);
      return true;
    }

    this->PrintIndent();
    int sid = this->BeginScope();
    this->stream << "{\n";
    // Check the EU width and derived padded layout supplied by the frontend.
    this->PrintIndent();
    this->stream << "int eu_num = " << eu_num << ";\n";
    this->PrintIndent();
    this->stream << "int align_w = " << align_w << ";\n";

    // Sum reduction pads inactive elements with zero.
    this->PrintIndent();
    this->stream << "scalar_t pad_val = {." << dtype_2 << " = 0};\n";

    // Build the two descriptor views used by the reduction stages.
    this->PrintIndent();
    this->stream << "dim4 in_reduce_h = {" << input_tensor << ".shape.n, "
                 << input_tensor << ".shape.c, align_w / eu_num, eu_num};\n";
    this->PrintIndent();
    this->stream << "dim4 out_reduce_h = {" << input_tensor << ".shape.n, "
                 << input_tensor << ".shape.c, 1, eu_num};\n";
    this->PrintIndent();
    this->stream << "dim4 in_reduce_w = {" << input_tensor << ".shape.n, "
                 << input_tensor << ".shape.c, 1, eu_num};\n";
    this->PrintIndent();
    this->stream << "dim4 out_reduce_w = {" << input_tensor << ".shape.n, "
                 << input_tensor << ".shape.c, 1, 1};\n";

    // Configure the pool geometry.
    this->PrintIndent();
    this->stream << "dim2 kernel = {align_w / eu_num, 1};\n";
    this->PrintIndent();
    this->stream << "padding_t pad = {0, 0, 0, 0};\n";
    this->PrintIndent();
    this->stream << "dim2 stride = {1, 1};\n";
    this->PrintIndent();
    this->stream << "dim2 dilation = {1, 1};\n";

    this->PrintIndent();
    this->stream << "scalar_t scale = {.f32 = (float)1.000000000e+00};\n";
    if (dtype_ == DataType::Float(16)) {
      this->PrintIndent();
      this->stream
          << "scale = tpu_cast(scale, DT_FP16, DT_FP32, RM_HALF_TO_EVEN);\n";
    } else if (dtype_ == DataType::BFloat(16)) {
      this->PrintIndent();
      this->stream
          << "scale = tpu_cast(scale, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);\n";
    }

    this->PrintIndent();
    this->stream << "if (align_w > " << input_tensor
                 << ".shape.w && align_w == eu_num) {\n";
    this->PrintIndent();
    this->stream << "  dim4 padded_stride = {" << stride_n
                 << ", align_w, align_w, 1};\n";
    this->PrintIndent();
    this->stream << "  dim4 padded_shape = {" << input_tensor << ".shape.n, "
                 << input_tensor << ".shape.c, 1, align_w};\n";
    this->PrintIndent();
    this->stream << "  dim4 copy_shape = {" << input_tensor << ".shape.n, "
                 << input_tensor << ".shape.c, 1, " << input_tensor
                 << ".shape.w};\n";
    this->PrintIndent();
    this->stream << "  __tilelang_tpu_tensor_info padded_input = {.shape = "
                 << "padded_shape, .stride = padded_stride, .addr = "
                 << tmp_tensor << ".addr, .default_stride = false};\n";
    this->PrintIndent();
    this->stream
        << "  __tilelang_tpu_tensor_info input_copy = {.shape = copy_shape, "
        << ".stride = {0}, .addr = " << input_tensor
        << ".addr, .default_stride = true};\n";
    this->PrintIndent();
    this->stream
        << "  __tilelang_tpu_tensor_info padded_input_copy = {.shape = "
        << "copy_shape, .stride = padded_stride, .addr = " << tmp_tensor
        << ".addr, .default_stride = false};\n";
    this->PrintIndent();
    this->stream << "  __tilelang_tpu_tensor_info output_view = {.shape = "
                 << "out_reduce_w, .stride = {0}, .addr = " << output_tensor
                 << ".addr, .default_stride = true};\n";
    this->PrintIndent();
    this->stream << "  tpu_bdc_set_C(padded_input.addr, pad_val, "
                 << "&padded_shape, &padded_input.stride, " << dtype << ");\n";
    this->PrintIndent();
    this->stream << "  tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, "
                 << "&copy_shape, &padded_input_copy.stride, "
                 << "(input_copy.default_stride ? NULL : &input_copy.stride), "
                 << dtype << ");\n";
    this->PrintIndent();
    this->stream << "  dim2 kernel2 = {1, eu_num};\n";
    this->PrintIndent();
    this->stream << "  tpu_bdc_fp_avg_pool2d(output_view.addr, "
                 << "padded_input.addr, &padded_shape, &kernel2, &pad, "
                 << "&stride, &dilation, " << dtype << ", scale);\n";
    this->PrintIndent();
    this->stream << "} else {\n";
    this->PrintIndent();
    this->stream << "  if (align_w > " << input_tensor << ".shape.w) {\n";
    this->PrintIndent();
    this->stream << "    dim4 fill_shape = {" << input_tensor << ".shape.n, "
                 << input_tensor << ".shape.c, 1, align_w - " << input_tensor
                 << ".shape.w};\n";
    this->PrintIndent();
    int elem_size = dtype_.bits() / 8;
    this->stream << "    int elem_size = " << elem_size << ";\n";
    this->PrintIndent();
    this->stream << "    int offset = " << input_tensor
                 << ".shape.w * elem_size;\n";
    this->PrintIndent();
    this->stream << "    dim4 fill_tensor_stride = {" << stride_n
                 << ", align_w, " << input_tensor << ".shape.w, 1};\n";
    this->PrintIndent();
    this->stream << "    __tilelang_tpu_tensor_info fill_tensor = {.shape = "
                    "fill_shape, .stride "
                    "= fill_tensor_stride, "
                 << ".addr = " << input_tensor
                 << ".addr + offset, .default_stride = false};\n";
    this->PrintIndent();
    this->stream
        << "    tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, "
        << "(fill_tensor.default_stride ? NULL : &fill_tensor.stride), "
        << dtype << ");\n";
    this->PrintIndent();
    this->stream << "  }\n";

    this->PrintIndent();
    this->stream
        << "  __tilelang_tpu_tensor_info input_view = {.shape = in_reduce_h, "
           ".stride = {0}, "
        << ".addr = " << input_tensor << ".addr, .default_stride = true};\n";
    this->PrintIndent();
    this->stream
        << "  __tilelang_tpu_tensor_info tmp_view = {.shape = out_reduce_h, "
           ".stride = {0}, "
        << ".addr = " << tmp_tensor << ".addr, .default_stride = true};\n";
    this->PrintIndent();
    this->stream << "  tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, "
                    "&input_view.shape, "
                 << "&kernel, &pad, &stride, &dilation, " << dtype
                 << ", scale);\n";
    this->PrintIndent();
    this->stream << "  dim2 kernel2 = {1, eu_num};\n";
    this->PrintIndent();
    this->stream
        << "  __tilelang_tpu_tensor_info output_view = {.shape = out_reduce_w, "
           ".stride = {0}, "
        << ".addr = " << output_tensor << ".addr, .default_stride = true};\n";
    this->PrintIndent();
    this->stream
        << "  __tilelang_tpu_tensor_info tmp_view2 = {.shape = in_reduce_w, "
           ".stride = {0}, "
        << ".addr = " << tmp_tensor << ".addr, .default_stride = true};\n";
    this->PrintIndent();
    this->stream << "  tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, "
                    "&tmp_view2.shape, "
                 << "&kernel2, &pad, &stride, &dilation, " << dtype
                 << ", scale);\n";
    this->PrintIndent();
    this->stream << "}\n";

    // Close the composite operation scope.
    this->EndScope(sid);
    this->PrintIndent();
    this->stream << "}\n";
  } else if (op_name == "tl.tpu.rsqrt") {
    ICHECK_EQ(op->args.size(), 3U) << op_name << " expects dst and src";
    auto dst_operand = ParseWholeBufferRegion(op->args[1], op_name + " dst", 2);
    auto src_operand = ParseWholeBufferRegion(op->args[2], op_name + " src", 1);
    ICHECK(dst_operand.is_local && src_operand.is_local)
        << op_name << " operands must reside in local memory";
    auto dst_dtype = dst_operand.dtype;
    auto src_dtype = src_operand.dtype;
    ICHECK_EQ(dst_dtype, src_dtype)
        << op_name << " requires matching dst/src dtypes";
    ICHECK(dst_dtype == DataType::Float(16) ||
           dst_dtype == DataType::BFloat(16) ||
           dst_dtype == DataType::Float(32))
        << op_name << " supports only FP16, BF16, and FP32, got " << dst_dtype;
    const auto &dst = dst_operand.descriptor;
    const auto &src0 = src_operand.descriptor;
    ICHECK_EQ(dst_operand.rank, src_operand.rank)
        << op_name << " requires matching dst/src ranks";
    ICHECK(dst_operand.shape4 == src_operand.shape4)
        << op_name << " requires matching dst/src shapes";
    if (target_programming_model_ == "rv") {
      EmitRVDescriptor(src0, 8, false, TPUKernelDTypeName(dst_dtype), true);
      EmitRVDescriptor(dst, 10, false, TPUKernelDTypeName(dst_dtype), true);
      stream << "rvt_cfg_satu(0, false);\nrvt_cfg_round_mode(0);\n"
             << "rvt_cfg_rsqrt_iter(3);\nrvt_sfu_rsqrt(10, 8);\n";
      return true;
    }
    this->PrintIndent();
    this->stream << "tpu_bdc_fp_rsqrt(" << dst << ".addr, " << src0
                 << ".addr, &" << src0 << ".shape, "
                 << TPUKernelDTypeName(dst_dtype) << ");\n";

  } else if (op_name == "tl.tpukernel.gather" ||
             op_name == "tl.tpu.embedding") {
    ICHECK_EQ(op->args.size(), 5U)
        << op_name << " expects output, param, index, and param_h";
    std::array<SemanticTensorOperand, 3> operands{};
    constexpr std::array<int, 3> kAccessMasks = {2, 1, 1};
    std::array<std::string, 3> tensors{};
    for (size_t i = 0; i < operands.size(); ++i) {
      operands[i] = ParseWholeBufferRegion(
          op->args[i + 1], op_name + " operand " + std::to_string(i),
          kAccessMasks[i]);
      ICHECK(!operands[i].is_local)
          << op_name << " uses the S2S API and requires global-memory operands";
      tensors[i] = operands[i].descriptor;
      ICHECK_EQ(operands[i].rank, 2U)
          << op_name << " requires rank-2 output, param, and index buffers";
    }
    ICHECK(operands[0].data_var != operands[1].data_var &&
           operands[0].data_var != operands[2].data_var &&
           operands[1].data_var != operands[2].data_var)
        << op_name << " requires distinct output, param, and index storage";
    const auto &dst_shape = operands[0].shape4;
    const auto &param_shape = operands[1].shape4;
    const auto &index_shape = operands[2].shape4;
    const auto *param_h_imm = op->args[4].as<IntImmNode>();
    ICHECK(param_h_imm && param_h_imm->value > 0)
        << op_name << " param_h must be a positive compile-time integer";
    auto param_h = param_h_imm->value;
    auto dtype_ = operands[0].dtype;
    ICHECK_EQ(operands[1].dtype, dtype_)
        << op_name << " output and param dtypes must match";
    ICHECK_EQ(operands[2].dtype, DataType::UInt(32))
        << op_name << " index dtype must be uint32";
    ICHECK(dtype_ == DataType::Float(16) || dtype_ == DataType::Float(32) ||
           dtype_ == DataType::BFloat(16) || dtype_.is_e4m3_float8() ||
           dtype_.is_e5m2_float8())
        << op_name << " supports FP8, FP16, BF16, or FP32 payloads, got "
        << dtype_;
    const std::string dtype = TPUKernelDTypeName(dtype_);
    ICHECK_EQ(dst_shape[0], 1);
    ICHECK_EQ(dst_shape[2], 1);
    ICHECK_EQ(param_shape[0], 1);
    ICHECK_EQ(param_shape[2], 1);
    ICHECK_EQ(index_shape[0], 1);
    ICHECK_EQ(index_shape[1], dst_shape[1])
        << op_name << " output row count must match index row count";
    ICHECK_EQ(index_shape[2], 1);
    ICHECK_EQ(index_shape[3], 1)
        << op_name << " index must have shape (count, 1)";
    ICHECK_EQ(param_shape[1], param_h)
        << op_name << " param_h must match param's leading dimension";
    ICHECK_EQ(dst_shape[3], param_shape[3])
        << op_name << " output and param row widths must match";
    const std::string &dst = tensors[0];
    const std::string &param = tensors[1];
    const std::string &index = tensors[2];
    if (target_programming_model_ == "rv") {
      // Row-major global tables: reinterpret logical C rows as DMA H rows.
      stream << "{\n";
      for (size_t i = 0; i < tensors.size(); ++i) {
        const auto &t = tensors[i];
        stream << "__tilelang_tpu_tensor_info emb" << i << " = " << t << ";\n";
        stream << "emb" << i << ".shape = (dim4){1,1," << t << ".shape.c," << t
               << ".shape.w};\n";
        // Whole global descriptors intentionally carry zero stride with
        // default_stride=true. FREE_LAYOUT needs explicit contiguous strides.
        stream << "emb" << i << ".stride = (dim4){" << t << ".shape.c*" << t
               << ".shape.w," << t << ".shape.c*" << t << ".shape.w," << t
               << ".shape.w,1};\n";
        EmitRVDescriptor("emb" + std::to_string(i), 32 + i, true,
                         i == 2 ? "DT_UINT32" : dtype, false);
      }
      stream << "rvt_dma_hgather(32, 33, 34, 0);\n}\n";
      return true;
    }

    this->PrintIndent();
    this->stream << "{\n";
    this->PrintIndent();
    this->stream << "dim4 __gather_shape = {1, 1, " << dst << ".shape.c, "
                 << dst << ".shape.w};\n";
    this->PrintIndent();
    this->stream << "tpu_gdma_h_gather_S2S(" << dst << ".addr, " << param
                 << ".addr, " << index << ".addr, "
                 << "false, (scalar_t){.u32 = 0}, &__gather_shape, " << param_h
                 << ", "
                 << "NULL, NULL, NULL, " << dtype << ");\n";
    this->PrintIndent();
    this->stream << "}\n";
  } else if (op_name == "tl.tpukernel.topk") {
    ICHECK_NE(target_chip_, "sg2260e")
        << op_name
        << " is unavailable on SG2260E: the PPL 1.7 "
           "tpub_7_1_e runtime rejects tpu_hau_sort_natural_index";
    ICHECK_EQ(op->args.size(), 7U)
        << op_name
        << " expects dst_data, dst_idx, src, K, descended, and length";
    std::array<SemanticTensorOperand, 3> operands{};
    constexpr std::array<int, 3> kAccessMasks = {2, 2, 1};
    std::array<std::string, 3> tensors{};
    for (size_t i = 0; i < operands.size(); ++i) {
      operands[i] = ParseWholeBufferRegion(
          op->args[i + 1], op_name + " operand " + std::to_string(i),
          kAccessMasks[i]);
      ICHECK(!operands[i].is_local)
          << op_name
          << " uses the HAU system-memory API and requires global operands";
      tensors[i] = operands[i].descriptor;
      ICHECK_EQ(operands[i].rank, 1U)
          << op_name << " requires rank-1 dst_data, dst_idx, and src buffers";
    }
    ICHECK(operands[0].data_var != operands[1].data_var &&
           operands[0].data_var != operands[2].data_var &&
           operands[1].data_var != operands[2].data_var)
        << op_name << " requires distinct dst_data, dst_idx, and src storage";
    const auto *k_imm = op->args[4].as<IntImmNode>();
    const auto *descended_imm = op->args[5].as<IntImmNode>();
    const auto *length_imm = op->args[6].as<IntImmNode>();
    ICHECK(k_imm && length_imm && k_imm->value > 0 && length_imm->value > 0 &&
           k_imm->value <= length_imm->value)
        << op_name << " requires compile-time integers 0 < K <= length";
    ICHECK(descended_imm && descended_imm->dtype.is_bool())
        << op_name << " descended must be a compile-time boolean";
    auto K_val = k_imm->value;
    auto descended_val = descended_imm->value != 0;
    auto length_val = length_imm->value;

    auto dtype_ = operands[0].dtype;
    ICHECK_EQ(operands[2].dtype, dtype_)
        << op_name << " dst_data and src dtypes must match";
    ICHECK_EQ(operands[1].dtype, DataType::Int(32))
        << op_name << " dst_idx dtype must be int32";
    std::string dtype;
    // tpu_hau_sort_natural_index supports FP32, INT32, and UINT32 only.
    if (dtype_ == DataType::Float(32)) {
      dtype = "DT_FP32";
    } else if (dtype_ == DataType::Int(32)) {
      dtype = "DT_INT32";
    } else if (dtype_ == DataType::UInt(32)) {
      dtype = "DT_UINT32";
    } else {
      ICHECK(false) << op_name << ": unsupported dtype " << dtype_
                    << "; HAU sort only supports fp32/int32/uint32";
    }

    const auto &dst_data_shape = operands[0].shape4;
    const auto &dst_idx_shape = operands[1].shape4;
    const auto &src_shape = operands[2].shape4;
    auto require_vector_shape = [&](const std::vector<int> &shape,
                                    int64_t expected, const char *operand) {
      ICHECK_EQ(shape.size(), 4U);
      ICHECK_EQ(shape[0], 1) << op_name << " " << operand << " must be rank 1";
      ICHECK_EQ(shape[1], 1) << op_name << " " << operand << " must be rank 1";
      ICHECK_EQ(shape[2], 1) << op_name << " " << operand << " must be rank 1";
      ICHECK_EQ(shape[3], expected)
          << op_name << " " << operand
          << " extent disagrees with its scalar contract";
    };
    require_vector_shape(src_shape, length_val, "src");
    // Sentinel experiments on BM1690 CModel confirm that HAU writes exactly K
    // values and indices.  Keep the output ABI honest instead of requiring an
    // unused, unspecified length-K tail.
    require_vector_shape(dst_data_shape, K_val, "dst_data");
    require_vector_shape(dst_idx_shape, K_val, "dst_idx");

    const std::string &dst_data = tensors[0];
    const std::string &dst_idx = tensors[1];
    const std::string &src = tensors[2];

    this->PrintIndent();
    this->stream << "tpu_hau_sort_natural_index(" << dst_data << ".addr, "
                 << dst_idx << ".addr, " << src << ".addr, " << length_val
                 << ", " << K_val << ", " << (descended_val ? "true" : "false")
                 << ", " << dtype << ");\n";
  } else {
    return false;
  }
  return true;
}

} // namespace codegen
} // namespace tvm
