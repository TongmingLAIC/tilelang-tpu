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

/*!
 * \file target/codegen_tpu.cc
 * \brief TileLang TPU source generation and programming-model dispatch.
 */

#include "codegen_tpu.h"
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../op/builtin.h"
#include "../op/bulk_copy.h"
#include "../op/gemm.h"
#include "tpuv7_lmem.h"

namespace tvm {
namespace codegen {

CodeGenTileLangTPU::CodeGenTileLangTPU(std::string target_chip,
                                       std::string target_programming_model)
    : target_chip_(std::move(target_chip)),
      target_programming_model_(std::move(target_programming_model)) {
  restrict_keyword_ = "global_addr_t";
}

void CodeGenTileLangTPU::PrintFuncPrefix(std::ostream &) {}

class TPUExternUsageExtractor : public tir::StmtExprVisitor {
private:
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::call_extern()) && !op->args.empty()) {
      if (const auto *name = op->args[0].as<StringImmNode>()) {
        const std::string &value = name->value;
        has_portable_tpu_op |= value.rfind("tl.tpu.", 0) == 0;
        has_raw_rvt_op |= value.rfind("rvt_", 0) == 0;
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

public:
  bool has_portable_tpu_op{false};
  bool has_raw_rvt_op{false};
};

bool IsValidRawRVTSymbol(const std::string &name) {
  if (name.rfind("rvt_", 0) != 0 || name.size() == 4U) {
    return false;
  }
  return std::all_of(name.begin() + 4, name.end(), [](unsigned char character) {
    return std::isalnum(character) != 0 || character == '_';
  });
}

void CodeGenTileLangTPU::PrintExtraAttrs(const PrimFunc &f, std::ostream &os) {}

std::string CodeGenTileLangTPU::Finish() {
  decl_stream << "/* TileLang TPU target: " << target_chip_
              << ", programming model: " << target_programming_model_
              << " */\n";
  decl_stream << "#include \"ppl_helper.h\"\n";
  if (uses_rvt_api_) {
    // RVT calls use the vendor ABI directly. Do not permit an accidental
    // implicit declaration in a TPU-Kernel build: only the RV programming
    // model sets TILELANG_TPU_RV and selects a PPL package with rvt_api.h.
    decl_stream << "#ifndef TILELANG_TPU_RV\n"
                << "#error \"RVT externs require "
                   "-tpu-programming-model=rv\"\n"
                << "#endif\n"
                << "#include \"atomic_def.h\"\n"
                << "#include \"rvt_api.h\"\n";
    if (uses_opaque_raw_rvt_) {
      decl_stream << "#define TILELANG_TPU_OPAQUE_RAW_RVT_ABI 1\n";
    }
  }
  if (uses_tpukernel_api_) {
    decl_stream << "#ifndef TILELANG_TPU_TPUKERNEL\n"
                << "#error \"TPU-Kernel externs require "
                   "-tpu-programming-model=tpukernel\"\n"
                << "#endif\n";
  }
  decl_stream << "typedef struct {\n"
              << "    dim4 shape;\n"
              << "    dim4 stride;\n"
              << "    global_addr_t addr;\n"
              << "    bool default_stride;\n"
              << "} __tilelang_tpu_tensor_info;\n\n";
  return CodeGenC::Finish();
}

void CodeGenTileLangTPU::VisitStmt_(const tir::ForNode *op) {
  ICHECK(op->kind == tir::ForKind::kSerial ||
         op->kind == tir::ForKind::kUnrolled)
      << "TPU source codegen supports only serial and unrolled loops; loop "
         "kind "
      << op->kind
      << " must be consumed by a target pass instead of being silently "
         "serialized";

  if (op->kind == tir::ForKind::kUnrolled) {
    PrintIndent();
    stream << "#pragma unroll\n";
  }
  std::string extent =
      PrintExpr(arith::Analyzer().Simplify(op->extent + op->min));
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  std::string start = PrintExpr(op->min);
  stream << "for (";
  PrintType(op->loop_var.dtype(), stream);
  stream << ' ' << vid << " = " << start << "; " << vid << " < " << extent
         << "; ++" << vid << ") {\n";
  int for_scope = BeginScope();
  loop_var_ranges_.push_back(
      {op->loop_var, Range::FromMinExtent(op->min, op->extent)});
  PrintStmt(op->body);
  loop_var_ranges_.pop_back();
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
}

void CodeGenTileLangTPU::BindThreadIndex(const IterVar &iv) {
  LOG(FATAL) << "GPU thread binding " << iv->thread_tag
             << " reached TPU codegen; lower it to an explicit TPU semantic "
                "operation before source emission";
}

void CodeGenTileLangTPU::PrintType(DataType t, std::ostream &os) { // NOLINT(*)
  ICHECK(t.is_scalar())
      << "Vector dtype " << t
      << " reached TPU codegen; the TPU residual-IR contract is scalar-only";
  if (t.is_handle()) {
    os << "void*";
    return;
  }

  if (t.is_void()) {
    os << "void";
    return;
  }

  if (t.is_float()) {
    switch (t.bits()) {
    case 16:
      os << "half_t";
      return;
    case 32:
      os << "float";
      return;
    case 64:
      os << "double";
      return;
    default:
      LOG(FATAL) << "Unsupported TPU floating-point type " << t;
    }
  } else if (t.is_bfloat16()) {
    os << "bfloat16_t";
    return;
  } else if (t.is_float8()) {
    os << "uint8_t";
    return;
  } else if (t == DataType::Bool()) {
    os << "bool";
    return;
  } else if (t.is_uint() || t.is_int()) {
    const bool is_unsigned = t.is_uint();
    switch (t.bits()) {
    case 1:
    case 4:
      os << (is_unsigned ? "unsigned int" : "int");
      return;
    case 8:
      os << (is_unsigned ? "uint8_t" : "int8_t");
      return;
    case 16:
      os << (is_unsigned ? "uint16_t" : "int16_t");
      return;
    case 32:
      os << (is_unsigned ? "uint32_t" : "int32_t");
      return;
    case 64:
      os << (is_unsigned ? "uint64_t" : "int64_t");
      return;
    default:
      LOG(FATAL) << "Unsupported TPU integer type " << t;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to a TPU C ABI type";
}

void CodeGenTileLangTPU::PrintVecBinaryOp(const std::string &op, DataType t,
                                          PrimExpr lhs, PrimExpr rhs,
                                          std::ostream &os) { // NOLINT(*)
  LOG(FATAL) << "Vector binary operation " << op << " with dtype " << t
             << " reached scalar-only TPU codegen";
}

void CodeGenTileLangTPU::PrintVecElemLoad(const std::string &vec, DataType t,
                                          int i,
                                          std::ostream &os) { // NOLINT(*)
  LOG(FATAL) << "Vector element load " << vec << "[" << i << "] of dtype " << t
             << " reached scalar-only TPU codegen";
}

void CodeGenTileLangTPU::PrintVecElemStore(const std::string &vec, DataType t,
                                           int i, const std::string &value) {
  LOG(FATAL) << "Vector element store " << vec << "[" << i << "] of dtype " << t
             << " reached scalar-only TPU codegen";
}

void CodeGenTileLangTPU::PrintStorageSync(const CallNode *op) {
  LOG(FATAL) << "GPU storage synchronization reached TPU codegen; it has no "
                "implicit TPU-Kernel or RV Tensor meaning";
}

void CodeGenTileLangTPU::PrintStorageScope(const std::string &scope,
                                           std::ostream &os) { // NOLINT(*)
  LOG(FATAL) << "Unlowered pointer storage scope " << scope
             << " reached TPU codegen; TPU local storage must use the "
                "compiler-owned tensor descriptor path";
}

std::string CodeGenTileLangTPU::CastFromTo(std::string value, DataType from,
                                           DataType target) {
  if (from == target)
    return value;
  std::ostringstream os;
  os << "((";
  this->PrintType(target, os);
  os << ")";
  if (from.is_float16() && (target.is_int() || target.is_uint()) &&
      target.bits() == 8) {
    os << "(";
    if (target.is_uint()) {
      os << "u";
    }
    os << "int)";
  }
  os << value << ")";
  return os.str();
}

void CodeGenTileLangTPU::VisitExpr_(const CastNode *op, std::ostream &os) {
  DataType from_ty = op->value.dtype();
  DataType target_ty = op->dtype;
  ICHECK_EQ(target_ty.lanes(), from_ty.lanes());
  ICHECK(from_ty.is_scalar())
      << "Vector cast from " << from_ty << " to " << target_ty
      << " reached scalar-only TPU codegen";
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenTileLangTPU::PrintCallExtern(Type ret_type, String global_symbol,
                                         const Array<PrimExpr> &args,
                                         bool skip_first_arg,
                                         std::ostream &os) { // NOLINT(*)
  DataType ret_dtype = GetRuntimeDataType(ret_type);
  ICHECK(!ret_dtype.is_vector()) << "Vector external return type " << ret_dtype
                                 << " reached scalar-only TPU codegen";
  CodeGenC::PrintCallExtern(ret_type, global_symbol, args, skip_first_arg, os);
}

// Print a reference expression to a buffer.
std::string CodeGenTileLangTPU::GetBufferRef(DataType t,
                                             const BufferNode *buffer,
                                             PrimExpr index) {
  const VarNode *buffer_var = buffer->data.get();
  ICHECK(!global_buffer_descriptor_.count(buffer_var) &&
         !buffer_addrs_.count(buffer_var))
      << "Direct scalar BufferLoad/BufferStore on TPU tensor descriptor "
      << buffer->name
      << " is unsupported; move data with tl.tpu.copy and compute through a "
         "typed tl.tpu.* or tl.tpukernel.* semantic operation";
  std::ostringstream os;
  std::string vid = GetVarID(buffer_var);
  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  auto ptr_cast = [this, scope](DataType pointed_to) {
    std::ostringstream ptr_os;
    ptr_os << "(";
    if (!scope.empty() && IsScopePartOfType()) {
      PrintStorageScope(scope, ptr_os);
    }
    PrintType(pointed_to, ptr_os);
    ptr_os << "*)";
    return ptr_os.str();
  };

  DataType buffer_element_dtype = buffer->dtype;

  std::string buffer_str = vid;
  if (!HandleTypeMatch(buffer_var, buffer_element_dtype)) {
    std::stringstream temp;
    temp << "(" << ptr_cast(buffer_element_dtype) << vid << ")";
    buffer_str = temp.str();
  }

  std::string index_str = PrintExpr(index);
  if (t.bits() == 4 || (t.bits() == 1 && t.is_int())) {
    // PrintType uses an int backing scalar for bool and 4-bit integers. In
    // most cases,
    // we divide by the number of lanes to determine the index.
    // However, the backing type for scalar int4 and scalar bool is
    // int32.  Therefore, we need to divide by the ratio of their
    // sizes in that case.
    int div_factor = (t.lanes() == 1) ? (32 / t.bits()) : t.lanes();

    os << "*("
       << "(" << ptr_cast(t) << vid << ")"
       << " + " << index_str << " / " << div_factor << ")";
  } else if (t == buffer_element_dtype) {
    os << buffer_str << "[" << index_str << "]";
  } else {
    os << "*" << ptr_cast(t) << "(" << buffer_str << " + " << index_str << ")";
  }

  return os.str();
}

CodeGenTileLangTPU::SemanticTensorOperand
CodeGenTileLangTPU::ParseWholeBufferRegion(const PrimExpr &expr,
                                           const std::string &context,
                                           int expected_access_mask,
                                           bool allow_matrix) const {
  const auto *region = expr.as<CallNode>();
  ICHECK(region && region->op.same_as(tl::RegionOp::Get()))
      << context << " must be a canonical whole-buffer tl.region operand";
  ICHECK(region->dtype.is_handle())
      << context << " tl.region must have handle dtype";
  ICHECK_GE(region->args.size(), 3U)
      << context
      << " unsupported tensor rank 0; TPU descriptors require rank 1 through 4";
  ICHECK_LE(region->args.size(), 6U)
      << context << " tl.region rank exceeds the TPU dim4 descriptor ABI";

  const auto *load = region->args[0].as<BufferLoadNode>();
  ICHECK(load) << context << " tl.region must start with a BufferLoad marker";
  const size_t rank = region->args.size() - 2;
  const Buffer &buffer = load->buffer;
  ICHECK_EQ(load->indices.size(), rank)
      << context << " tl.region rank must match its BufferLoad indices";
  ICHECK_EQ(buffer->shape.size(), rank)
      << context << " tl.region rank must match its logical Buffer rank";
  ICHECK_EQ(load->dtype, buffer->dtype)
      << context << " BufferLoad marker dtype disagrees with its Buffer";

  const int64_t *access_mask = as_const_int(region->args[1]);
  ICHECK(access_mask && *access_mask == expected_access_mask)
      << context << " tl.region requires access mask " << expected_access_mask
      << ", got "
      << (access_mask ? std::to_string(*access_mask) : "a dynamic value");
  arith::Analyzer analyzer;
  for (size_t axis = 0; axis < rank; ++axis) {
    PrimExpr min = analyzer.Simplify(load->indices[axis]);
    ICHECK(is_zero(min)) << context
                         << " must cover the whole Buffer from zero; axis "
                         << axis << " has minimum " << min;
    PrimExpr extent = analyzer.Simplify(region->args[axis + 2]);
    PrimExpr buffer_extent = analyzer.Simplify(buffer->shape[axis]);
    ICHECK(StructuralEqual()(extent, buffer_extent))
        << context << " must cover the whole Buffer; axis " << axis
        << " extent " << extent << " differs from Buffer extent "
        << buffer_extent;
  }

  ICHECK(is_zero(buffer->elem_offset) && buffer->strides.empty() &&
         buffer->axis_separators.empty() &&
         buffer->buffer_type == BufferType::kDefault)
      << context
      << " requires a canonical zero-offset contiguous Buffer descriptor";
  const std::string scope = buffer.scope();
  const bool is_local = tl::tpuv7::IsLocalMemoryScope(scope);
  ICHECK(is_local || scope == "global")
      << context << " has unsupported Buffer scope " << scope;
  ICHECK(allow_matrix || scope != "local.matrix")
      << context
      << " uses local.matrix storage, which is reserved for FP32 "
         "GEMM and its global transfers";

  const VarNode *data_var = buffer->data.get();
  ICHECK(data_var && compiler_descriptor_vars_.count(data_var))
      << context
      << " does not reference a compiler-owned TPU descriptor data Var";
  auto dtype_it = descriptor_dtype_.find(data_var);
  auto rank_it = descriptor_rank_.find(data_var);
  auto shape_it = descriptor_shape4_.find(data_var);
  auto element_count_it = descriptor_element_count_.find(data_var);
  ICHECK(dtype_it != descriptor_dtype_.end() &&
         rank_it != descriptor_rank_.end() &&
         shape_it != descriptor_shape4_.end() &&
         element_count_it != descriptor_element_count_.end())
      << context << " has incomplete compiler-owned descriptor metadata";
  ICHECK_EQ(buffer->dtype, dtype_it->second)
      << context << " Buffer dtype " << buffer->dtype
      << " disagrees with descriptor dtype " << dtype_it->second;
  ICHECK_EQ(rank, rank_it->second)
      << context << " logical Buffer rank disagrees with its descriptor owner";

  const auto logical_shape4 =
      tl::tpuv7::NormalizeLocalShape(buffer->shape, context.c_str());
  ICHECK_EQ(logical_shape4.size(), shape_it->second.size());
  ICHECK(std::equal(logical_shape4.begin(), logical_shape4.end(),
                    shape_it->second.begin()))
      << context
      << " logical Buffer shape disagrees with its descriptor owner; Buffer "
         "aliases/views must preserve the exact rank, shape, and dtype";
  ICHECK_EQ(tl::tpuv7::DescriptorElementCount(logical_shape4, context.c_str()),
            element_count_it->second)
      << context
      << " descriptor element count disagrees with its logical shape";

  std::string descriptor;
  if (is_local) {
    auto local_it = var_idmap_.find(data_var);
    ICHECK(buffer_addrs_.count(data_var) && local_it != var_idmap_.end())
        << context << " is not owned by a live local-memory allocation";
    auto allocation_scope = alloc_storage_scope_.find(data_var);
    ICHECK(allocation_scope != alloc_storage_scope_.end() &&
           allocation_scope->second == scope)
        << context << " logical Buffer scope " << scope
        << " disagrees with its allocation owner scope "
        << (allocation_scope == alloc_storage_scope_.end()
                ? std::string("<missing>")
                : allocation_scope->second);
    descriptor = local_it->second;
  } else {
    ICHECK(!buffer_addrs_.count(data_var))
        << context << " global Buffer unexpectedly owns an LMEM address";
    auto global_it = global_buffer_descriptor_.find(data_var);
    ICHECK(global_it != global_buffer_descriptor_.end())
        << context << " is not a global kernel Buffer descriptor";
    descriptor = global_it->second;
  }

  SemanticTensorOperand operand;
  operand.data_var = data_var;
  operand.descriptor = std::move(descriptor);
  operand.dtype = dtype_it->second;
  operand.shape4 = shape_it->second;
  operand.scope = scope;
  operand.rank = rank;
  operand.is_local = is_local;
  return operand;
}

const std::vector<int> &
CodeGenTileLangTPU::DescriptorShape4(const VarNode *data_var,
                                     const std::string &context) const {
  ICHECK(data_var) << context << " requires a buffer data Var";
  auto shape_it = descriptor_shape4_.find(data_var);
  ICHECK(shape_it != descriptor_shape4_.end())
      << context << " has no compiler-owned dim4 descriptor shape";
  return shape_it->second;
}

static inline std::string Dim4Initializer(const std::vector<int> &values) {
  ICHECK_EQ(values.size(), 4U);
  std::string ret = "{";
  for (int value : values) {
    ret += std::to_string(value) + ", ";
  }
  ret[ret.size() - 2] = '}';
  return ret;
}

static inline std::string TargetDTypeName(DataType dtype) {
  if (dtype == DataType::Float(32)) {
    return "DT_FP32";
  } else if (dtype == DataType::Float(16)) {
    return "DT_FP16";
  } else if (dtype == DataType::BFloat(16)) {
    return "DT_BFP16";
  } else if (dtype.is_e5m2_float8()) {
    return "DT_FP8E5M2";
  } else if (dtype.is_e4m3_float8()) {
    return "DT_FP8E4M3";
  } else if (dtype == DataType::UInt(32)) {
    return "DT_UINT32";
  } else if (dtype == DataType::Int(32)) {
    return "DT_INT32";
  } else if (dtype == DataType::UInt(16)) {
    return "DT_UINT16";
  } else if (dtype == DataType::Int(16)) {
    return "DT_INT16";
  } else if (dtype == DataType::UInt(8)) {
    return "DT_UINT8";
  } else if (dtype == DataType::Int(8)) {
    return "DT_INT8";
  }
  LOG(FATAL) << "Unsupported dtype " << dtype;
  return "DT_FP32";
}

static inline int TargetDTypeBytes(DataType dtype) {
  if (dtype == DataType::Float(32) || dtype == DataType::UInt(32) ||
      dtype == DataType::Int(32)) {
    return 4;
  } else if (dtype == DataType::Float(16) || dtype == DataType::BFloat(16) ||
             dtype == DataType::UInt(16) || dtype == DataType::Int(16)) {
    return 2;
  } else if (dtype.is_e5m2_float8() || dtype.is_e4m3_float8() ||
             dtype == DataType::UInt(8) || dtype == DataType::Int(8)) {
    return 1;
  }
  LOG(FATAL) << "Unsupported dtype " << dtype;
  return 0;
}

inline int GetIntImmValueForDim4(const PrimExpr &expr, const char *context) {
  auto *imm = expr.as<IntImmNode>();
  ICHECK(imm) << context << " expects a compile-time integer dimension";
  return static_cast<int>(
      tl::tpuv7::ValidateDescriptorDim(imm->value, context));
}

inline std::vector<int> LowerDimValuesToDim4(const std::vector<int> &dims) {
  int rank = dims.size();
  ICHECK(rank >= 1 && rank <= 4)
      << "Only support rank 1 to 4, but got " << rank;
  std::vector<int> dim4 = {1, 1, 1, 1};
  if (rank == 1) {
    dim4[3] = dims[0];
  } else if (rank == 2) {
    dim4[1] = dims[0];
    dim4[3] = dims[1];
  } else if (rank == 3) {
    dim4[0] = dims[0];
    dim4[1] = dims[1];
    dim4[3] = dims[2];
  } else {
    for (int i = 0; i < 4; i++) {
      dim4[i] = dims[i];
    }
  }
  return dim4;
}

inline std::vector<int> LowerRegionToDim4(const Array<Range> &ranges) {
  int rank = ranges.size();
  std::vector<int> dims;
  dims.reserve(rank);
  for (const auto &range : ranges) {
    dims.push_back(
        GetIntImmValueForDim4(range->extent, "TileLang TPU copy region"));
  }
  return LowerDimValuesToDim4(dims);
}

inline std::vector<int> LowerGlobalShapeToDim4(const Array<PrimExpr> &shape) {
  int rank = shape.size();
  std::vector<int> dims;
  dims.reserve(rank);
  for (const auto &dim : shape) {
    dims.push_back(GetIntImmValueForDim4(dim, "TileLang TPU global tensor"));
  }
  return LowerDimValuesToDim4(dims);
}

inline std::vector<int> StrideIndicesForRank(int rank) {
  if (rank == 4) {
    return {0, 1, 2, 3};
  } else if (rank == 3) {
    return {0, 1, 3};
  } else if (rank == 2) {
    return {1, 3};
  } else if (rank == 1) {
    return {3};
  }
  LOG(FATAL) << "Unsupported region dims: " << rank;
  return {};
}

void CodeGenTileLangTPU::VisitExpr_(const CallNode *op, std::ostream &os) {
  auto handle_elementwise = [&, this](const std::string &operation) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.tpu." << operation << " expects dst, lhs, and rhs";
    const std::string prefix = "tl.tpu." + operation;
    auto dst_operand =
        ParseWholeBufferRegion(op->args[1], prefix + " output", 2);
    auto src0_operand = ParseWholeBufferRegion(op->args[2], prefix + " lhs", 1);
    auto src1_operand = ParseWholeBufferRegion(op->args[3], prefix + " rhs", 1);
    ICHECK(dst_operand.is_local && src0_operand.is_local &&
           src1_operand.is_local)
        << "TileLang TPU " << operation
        << " operands must all have compiler-owned local descriptors";
    const auto &dst = dst_operand.descriptor;
    const auto &src0 = src0_operand.descriptor;
    const auto &src1 = src1_operand.descriptor;
    ICHECK_EQ(dst_operand.rank, 2U)
        << "TileLang TPU " << operation << " requires rank-2 output";
    ICHECK_EQ(src0_operand.rank, 2U)
        << "TileLang TPU " << operation << " requires rank-2 lhs";
    ICHECK_EQ(src1_operand.rank, 2U)
        << "TileLang TPU " << operation << " requires rank-2 rhs";
    const std::vector<int> dst_shape = {dst_operand.shape4[1],
                                        dst_operand.shape4[3]};
    const std::vector<int> src0_shape = {src0_operand.shape4[1],
                                         src0_operand.shape4[3]};
    const std::vector<int> src1_shape = {src1_operand.shape4[1],
                                         src1_operand.shape4[3]};
    DataType dst_dtype = dst_operand.dtype;
    DataType src0_dtype = src0_operand.dtype;
    DataType src1_dtype = src1_operand.dtype;
    ICHECK(dst_dtype == src0_dtype && dst_dtype == src1_dtype)
        << "TileLang TPU " << operation
        << " requires matching input/output dtypes";
    const bool is_fp8 =
        dst_dtype.is_e4m3_float8() || dst_dtype.is_e5m2_float8();
    if (is_fp8) {
      ICHECK(operation != "div")
          << "TileLang TPU " << operation
          << " has no validated FP8 instruction contract";
    } else {
      ICHECK(dst_dtype == DataType::Float(16) ||
             dst_dtype == DataType::BFloat(16) ||
             dst_dtype == DataType::Float(32))
          << "TileLang TPU floating-point " << operation
          << " only supports FP16, BF16, or FP32";
    }
    ICHECK(dst_shape == src0_shape)
        << "TileLang TPU " << operation
        << " requires output and lhs shapes to match";
    ICHECK(src1_shape == dst_shape ||
           (src1_shape[0] == dst_shape[0] && src1_shape[1] == 1))
        << "TileLang TPU " << operation
        << " only supports an equal rhs shape or W-dimension broadcast";
    if (target_programming_model_ == "tpukernel") {
      EmitTPUKernelElementwise(operation, dst, src0, src1, dst_dtype,
                               src0_shape, src1_shape);
    } else {
      EmitRVElementwise(operation, dst, src0, src1, dst_dtype, src0_dtype,
                        src1_dtype, dst_shape, src0_shape, src1_shape);
    }
  };
  std::vector<std::string> inst;
  if (op->op.same_as(builtin::call_pure_extern())) {
    LOG(FATAL) << "tir.call_pure_extern has no TPU semantic ABI; use a "
                  "side-effecting compiler-owned tl.tpu.* or tl.tpukernel.* "
                  "operation, or an isolated raw rvt_* call_extern on the RV "
                  "programming model";
  } else if (op->op.same_as(builtin::call_extern())) {
    ICHECK(!op->args.empty())
        << "TPU call_extern requires a compile-time function name";
    const auto *op_name_node = op->args[0].as<StringImmNode>();
    ICHECK(op_name_node) << "TPU call_extern function name must be a StringImm";
    std::string op_name = op_name_node->value;
    ICHECK(op_name.rfind("ppl.", 0) != 0)
        << "The internal ppl.* TIR ABI has been removed; use tl.tpu.* for "
           "portable core operations or tl.tpukernel.* for TPU-Kernel-only "
           "operations. Found "
        << op_name;
    ICHECK(op_name.rfind("tpu_", 0) != 0)
        << "Raw tpu_* call_extern is not a supported TileLang ABI. Use a "
           "typed tl.tpukernel.* semantic operation so ownership, memory "
           "effects, dtype constraints, and chip support can be validated. "
           "Found "
        << op_name;
    const bool has_rvt_prefix = op_name.rfind("rvt_", 0) == 0;
    ICHECK(!has_rvt_prefix || IsValidRawRVTSymbol(op_name))
        << "Raw RVT extern name must be a C identifier beginning with rvt_; "
           "got "
        << op_name;
    const bool is_tpukernel_extern = op_name.rfind("tl.tpukernel.", 0) == 0;
    const bool is_rvt_extern = has_rvt_prefix;
    const bool is_portable_tpu_op = op_name.rfind("tl.tpu.", 0) == 0;
    if (is_portable_tpu_op) {
      const bool is_supported_portable_op =
          op_name == "tl.tpu.copy" || op_name == "tl.tpu.fill" ||
          op_name == "tl.tpu.gemm" || op_name == "tl.tpu.add" ||
          op_name == "tl.tpu.sub" || op_name == "tl.tpu.mul" ||
          op_name == "tl.tpu.div" || op_name == "tl.tpu.max" ||
          op_name == "tl.tpu.add_scalar" || op_name == "tl.tpu.mul_scalar" ||
          op_name == "tl.tpu.rsqrt" || op_name == "tl.tpu.reduce_sum" ||
          op_name == "tl.tpu.reduce_max" || op_name == "tl.tpu.exp" ||
          op_name == "tl.tpu.embedding";
      ICHECK(is_supported_portable_op)
          << "Unknown backend-neutral TPU operation " << op_name;
      ICHECK_EQ(rvt_direct_call_count_, 0)
          << "Portable TPU operations cannot share a kernel with opaque raw "
             "RVT calls";
      ++canonical_tpu_op_count_;
      if (target_programming_model_ == "rv") {
        uses_rvt_api_ = true;
      } else {
        uses_tpukernel_api_ = true;
      }
    }
    if (is_tpukernel_extern) {
      ICHECK_EQ(target_programming_model_, "tpukernel")
          << "TPU-Kernel extern " << op_name
          << " requires target tpu-programming-model=tpukernel";
      ICHECK_EQ(rvt_direct_call_count_, 0)
          << "Raw RVT rvt_* calls cannot share a kernel with TPU-Kernel calls; "
          << "their descriptor and command-stream ownership models are "
             "distinct.";
      ++tpukernel_extern_count_;
      uses_tpukernel_api_ = true;
    }
    if (op_name == "tl.tpu.copy") {
      ICHECK_EQ(op->args.size(), 3U)
          << op_name
          << " expects exactly one source and one destination region";
      tl::BufferMap buffer_map;
      auto parse_copy_region = [&](const PrimExpr &expr,
                                   const char *operand_name,
                                   int expected_access_mask) {
        const auto *region = expr.as<CallNode>();
        ICHECK(region && region->op.same_as(tl::RegionOp::Get()))
            << op_name << " " << operand_name
            << " must be a canonical tl.region descriptor";
        ICHECK(region->dtype.is_handle())
            << op_name << " " << operand_name
            << " tl.region must have handle dtype";
        ICHECK_GE(region->args.size(), 3U)
            << op_name << " " << operand_name
            << " region must contain a BufferLoad, access mask, and extent";
        const auto *load = region->args[0].as<BufferLoadNode>();
        ICHECK(load) << op_name << " " << operand_name
                     << " region must start with a BufferLoad marker";
        const size_t rank = region->args.size() - 2;
        ICHECK_GE(rank, 1U);
        ICHECK_LE(rank, 4U)
            << op_name << " " << operand_name
            << " region rank exceeds the TPU dim4 descriptor ABI";
        ICHECK_EQ(load->indices.size(), rank)
            << op_name << " " << operand_name
            << " region extent rank must match its BufferLoad indices";
        for (size_t axis = 0; axis < rank; ++axis) {
          const auto *ramp = load->indices[axis].as<RampNode>();
          if (!ramp) {
            continue;
          }
          const int64_t *stride = as_const_int(ramp->stride);
          ICHECK(stride && *stride == 1)
              << op_name << " " << operand_name << " axis " << axis
              << " Ramp must have unit stride because TPU copy descriptors "
                 "represent contiguous regions";
          const int64_t *lanes = as_const_int(ramp->lanes);
          const int64_t *extent = as_const_int(region->args[axis + 2]);
          ICHECK(lanes && extent && *lanes == *extent)
              << op_name << " " << operand_name << " axis " << axis
              << " Ramp lane count must equal its explicit region extent";
        }
        const int64_t *access_mask = as_const_int(region->args[1]);
        ICHECK(access_mask && *access_mask == expected_access_mask)
            << op_name << " " << operand_name << " region requires access mask "
            << expected_access_mask;
        return tl::RegionOp(region->args, buffer_map);
      };
      auto check_copy_bounds = [&, this](const tir::Buffer &buffer,
                                         const Array<Range> &ranges,
                                         const char *operand_name) {
        ICHECK_EQ(ranges.size(), buffer->shape.size())
            << op_name << " " << operand_name << " rank mismatch: region rank "
            << ranges.size() << ", buffer rank " << buffer->shape.size();
        arith::Analyzer analyzer;
        for (const auto &loop_range : loop_var_ranges_) {
          analyzer.Bind(loop_range.first, loop_range.second);
        }
        for (size_t i = 0; i < ranges.size(); ++i) {
          const Range &range = ranges[i];
          PrimExpr raw_min = range->min;
          PrimExpr min =
              raw_min.as<RampNode>() ? raw_min.as<RampNode>()->base : raw_min;
          min = analyzer.Simplify(min);
          PrimExpr upper = analyzer.Simplify(min + range->extent);
          PrimExpr shape_dim = buffer->shape[i];
          bool lower_ok =
              analyzer.CanProve(min >= make_const(min.dtype(), 0),
                                arith::ProofStrength::kSymbolicBound);
          bool upper_ok = analyzer.CanProve(
              upper <= shape_dim, arith::ProofStrength::kSymbolicBound);
          ICHECK(lower_ok && upper_ok)
              << op_name << " " << operand_name
              << " region may be out of bounds "
              << "for buffer " << buffer->name << " at dim " << i
              << ": min=" << min << ", extent=" << range->extent
              << ", upper=" << upper << ", shape_dim=" << shape_dim
              << ". Portable TPU copy has no implicit tail masking; make "
              << "the tile divide the static shape or add explicit tail "
              << "handling in the frontend.";
        }
      };
      auto range_min_base = [](const Range &range) {
        if (const RampNode *ramp = range->min.as<RampNode>()) {
          return ramp->base;
        }
        return range->min;
      };
      auto process_copy = [&, this](const tl::RegionOp &src,
                                    const Array<Range> &src_ranges,
                                    const char *operand_name)
          -> std::tuple<std::string, std::string, std::string> {
        auto src_buffer = src.GetBuffer();
        const std::string scope = src_buffer.scope();
        const bool is_local = tl::tpuv7::IsLocalMemoryScope(scope);
        const bool is_global = scope == "global";
        ICHECK(is_local || is_global)
            << "Unsupported " << op_name << " buffer scope: " << scope
            << "; expected global or one of shared, shared.dyn, local, "
               "local.fragment, local.matrix";
        ICHECK(is_zero(src_buffer->elem_offset))
            << op_name << " " << operand_name
            << " Buffer elem_offset is unsupported; construct an explicit "
               "tl.tpu.copy region over the canonical tensor descriptor";
        ICHECK(src_buffer->strides.empty())
            << op_name << " " << operand_name
            << " Buffer has explicit strides that are not represented by the "
               "TPU tensor descriptor ABI";
        ICHECK(src_buffer->axis_separators.empty())
            << op_name << " " << operand_name
            << " Buffer axis separators are not supported by TPU descriptors";
        ICHECK_EQ(src_buffer->buffer_type, BufferType::kDefault)
            << op_name << " " << operand_name
            << " requires a default contiguous Buffer, not an auto-broadcast "
               "view";

        std::string src_id;
        if (is_global) {
          auto descriptor_it =
              global_buffer_descriptor_.find(src_buffer->data.get());
          ICHECK(descriptor_it != global_buffer_descriptor_.end())
              << op_name << " " << operand_name << " buffer "
              << src_buffer->name << " is not a global kernel parameter";
          src_id = descriptor_it->second;
        } else {
          const auto *data_var = src_buffer->data.get();
          auto local_it = var_idmap_.find(data_var);
          ICHECK(buffer_addrs_.count(data_var) && local_it != var_idmap_.end())
              << op_name << " " << operand_name << " buffer "
              << src_buffer->name
              << " has local scope but no compiler-owned LMEM descriptor";
          auto allocation_scope = alloc_storage_scope_.find(data_var);
          ICHECK(allocation_scope != alloc_storage_scope_.end() &&
                 allocation_scope->second == scope)
              << op_name << " " << operand_name << " logical Buffer scope "
              << scope << " disagrees with its allocation owner scope "
              << (allocation_scope == alloc_storage_scope_.end()
                      ? std::string("<missing>")
                      : allocation_scope->second);
          src_id = local_it->second;
        }
        auto dtype_it = descriptor_dtype_.find(src_buffer->data.get());
        auto rank_it = descriptor_rank_.find(src_buffer->data.get());
        ICHECK(dtype_it != descriptor_dtype_.end())
            << op_name << " " << operand_name
            << " has no compiler-owned descriptor dtype";
        ICHECK(rank_it != descriptor_rank_.end() &&
               rank_it->second == src_buffer->shape.size())
            << op_name << " " << operand_name
            << " Buffer view rank disagrees with its compiler-owned tensor "
               "descriptor; rank-changing aliases/views are not supported";
        ICHECK_EQ(src_buffer->dtype, dtype_it->second)
            << op_name << " " << operand_name << " buffer dtype "
            << src_buffer->dtype << " disagrees with its compiler-owned "
            << "descriptor dtype " << dtype_it->second;
        const auto declared_shape4 = tl::tpuv7::NormalizeLocalShape(
            src_buffer->shape, "TileLang TPU copy buffer");
        const auto &owned_shape = DescriptorShape4(src_buffer->data.get(),
                                                   "TileLang TPU copy buffer");
        ICHECK(std::equal(declared_shape4.begin(), declared_shape4.end(),
                          owned_shape.begin(), owned_shape.end()))
            << op_name << " " << operand_name
            << " Buffer view shape disagrees with its compiler-owned tensor "
               "descriptor; shape-changing aliases/views are not supported";
        if (is_local && src_ranges.size() >= 2U) {
          const size_t c_axis = src_ranges.size() == 2U ? 0U : 1U;
          PrimExpr c_min =
              arith::Analyzer().Simplify(range_min_base(src_ranges[c_axis]));
          ICHECK(is_zero(c_min))
              << op_name << " " << operand_name
              << " local C-axis minimum must be zero; TPUv7 channels map "
                 "across NPU lanes and cannot use a linear LMEM byte offset";
        }
        if (scope == "local.matrix") {
          ICHECK_EQ(src_ranges.size(), src_buffer->shape.size());
          arith::Analyzer matrix_analyzer;
          for (size_t axis = 0; axis < src_ranges.size(); ++axis) {
            ICHECK(is_zero(
                matrix_analyzer.Simplify(range_min_base(src_ranges[axis]))))
                << op_name
                << " local.matrix transfers must cover the whole "
                   "matrix from zero";
            ICHECK(StructuralEqual()(
                matrix_analyzer.Simplify(src_ranges[axis]->extent),
                matrix_analyzer.Simplify(src_buffer->shape[axis])))
                << op_name
                << " local.matrix transfers must cover the whole matrix";
          }
        }
        check_copy_bounds(src_buffer, src_ranges, operand_name);
        std::string new_src_var =
            name_supply_->FreshName(src_buffer->data->name_hint);
        std::string src_shape = Dim4Initializer(LowerRegionToDim4(src_ranges));

        std::string dtype = TargetDTypeName(src_buffer->dtype);
        int bytes_size = TargetDTypeBytes(src_buffer->dtype);
        if (is_global) {
          auto stride_it = buffer_stride.find(src_id);
          ICHECK(stride_it != buffer_stride.end() &&
                 stride_it->second.size() == 4U)
              << "buffer_stride not initialized for global buffer: "
              << src_buffer->name;
          const auto &strides = stride_it->second;
          std::string src_strides = Dim4Initializer(strides);

          std::string min_expr;
          std::vector<int> stride_idx = StrideIndicesForRank(src_ranges.size());

          for (size_t i = 0; i < src_ranges.size(); i++) {
            auto sr = src_ranges[i];
            const PrimExpr &e = sr->min;
            std::string idx_str;
            if (const RampNode *ramp = e.as<RampNode>()) {
              idx_str = PrintExpr(ramp->base);
            } else {
              idx_str = PrintExpr(e);
            }
            min_expr += "(" + idx_str + ") * " +
                        std::to_string(strides[stride_idx[i]]) + "+";
          }
          min_expr[min_expr.size() - 1] = ' ';
          min_expr = "(" + min_expr + ")" + " * " + std::to_string(bytes_size);
          inst.push_back("__tilelang_tpu_tensor_info " + new_src_var +
                         " = {.shape = " + src_shape + ", .stride = " +
                         src_strides + ", .addr = " + src_id + ".addr + " +
                         min_expr + ", .default_stride = false};\n");
        } else if (is_local) {
          const std::string &parent_var = src_id;
          std::string min_expr;
          std::vector<std::string> strides = {
              parent_var + ".stride.n", parent_var + ".stride.c",
              parent_var + ".stride.h", parent_var + ".stride.w"};
          std::vector<int> stride_idx = StrideIndicesForRank(src_ranges.size());
          for (size_t i = 0; i < src_ranges.size(); i++) {
            auto sr = src_ranges[i];
            const PrimExpr &e = sr->min;
            std::string idx_str;
            if (const RampNode *ramp = e.as<RampNode>()) {
              idx_str = PrintExpr(ramp->base);
            } else {
              idx_str = PrintExpr(e);
            }
            min_expr += "(" + idx_str + ") * " + strides[stride_idx[i]] + "+";
          }
          min_expr[min_expr.size() - 1] = ' ';
          min_expr = "(" + min_expr + ")" + " * " + std::to_string(bytes_size);
          inst.push_back(
              "__tilelang_tpu_tensor_info " + new_src_var +
              " = {.shape = " + src_shape + ", .stride = " + parent_var +
              ".stride, .addr = " + parent_var + ".addr + " + min_expr +
              ", .default_stride = " + parent_var + ".default_stride};\n");
        }
        return std::make_tuple(new_src_var, scope, dtype);
      };
      tl::RegionOp src = parse_copy_region(op->args[1], "src", 1);
      tl::RegionOp dst = parse_copy_region(op->args[2], "dst", 2);

      auto emit_copy_for_ranges = [&](const Array<Range> &src_ranges,
                                      const Array<Range> &dst_ranges) {
        const auto src_shape4 = LowerRegionToDim4(src_ranges);
        const auto dst_shape4 = LowerRegionToDim4(dst_ranges);
        ICHECK(src_shape4 == dst_shape4)
            << op_name
            << " requires source and destination regions to have "
               "the same normalized N/C/H/W extents; source="
            << Dim4Initializer(src_shape4)
            << ", destination=" << Dim4Initializer(dst_shape4);
        auto [src_var_id, src_scope, src_dtype] =
            process_copy(src, src_ranges, "src");
        auto [dst_var_id, dst_scope, dst_dtype] =
            process_copy(dst, dst_ranges, "dst");
        auto is_fp8 = [](const std::string &dtype) {
          return dtype == "DT_FP8E4M3" || dtype == "DT_FP8E5M2";
        };
        if (is_fp8(src_dtype) || is_fp8(dst_dtype)) {
          const bool same_format = src_dtype == dst_dtype;
          auto is_base_float = [](const std::string &dtype) {
            return dtype == "DT_FP16" || dtype == "DT_BFP16" ||
                   dtype == "DT_FP32";
          };
          const bool float_conversion =
              (is_fp8(src_dtype) && is_base_float(dst_dtype)) ||
              (is_base_float(src_dtype) && is_fp8(dst_dtype));
          ICHECK(same_format || float_conversion)
              << op_name
              << " FP8 supports same-format transport and conversion to or "
                 "from FP16, BF16, or FP32 only";
        }
        // Region descriptors must be declared before either backend emits an
        // instruction that references them.
        for (const auto &declaration : inst) {
          PrintIndent();
          stream << declaration;
        }
        inst.clear();
        const bool src_is_global = src_scope == "global";
        const bool dst_is_global = dst_scope == "global";
        const bool matrix_copy =
            src_scope == "local.matrix" || dst_scope == "local.matrix";
        if (matrix_copy) {
          ICHECK(src_dtype == "DT_FP32" && dst_dtype == "DT_FP32")
              << op_name << " matrix layout currently supports FP32 only";
          ICHECK(src_is_global != dst_is_global)
              << op_name
              << " matrix layout supports only global-to-local or "
                 "local-to-global transfers";
          ICHECK_EQ(src_ranges.size(), 2U)
              << op_name << " matrix layout requires rank-2 regions";
          const auto rows = src_shape4[1];
          const auto cols = src_shape4[3];
          if (target_programming_model_ == "rv") {
            EmitRVMatrixCopy(src_var_id, src_is_global, dst_var_id,
                             dst_is_global, DataType::Float(32), rows, cols);
          } else {
            EmitTPUKernelMatrixCopy(src_var_id, src_is_global, dst_var_id,
                                    dst_is_global, DataType::Float(32), rows,
                                    cols);
          }
          return;
        }
        if (target_programming_model_ == "rv") {
          EmitRVCopy(src_var_id, src_is_global, src_dtype, dst_var_id,
                     dst_is_global, dst_dtype);
        } else {
          EmitTPUKernelCopy(src_var_id, src_is_global, src_dtype, dst_var_id,
                            dst_is_global, dst_dtype);
        }
      };

      auto src_ranges = src.GetRanges();
      auto dst_ranges = dst.GetRanges();
      // A full [N,C,W] region is represented directly as [N,C,1,W].  The
      // backend DMA instruction understands the lane-distributed C axis; it
      // must not be split by adding a linear LMEM byte offset.  Partial local
      // C slices remain rejected in process_copy because their starting lane
      // is not encoded by the current descriptor ABI.
      emit_copy_for_ranges(src_ranges, dst_ranges);
    } else if (op_name == "tl.tpu.fill") {
      ICHECK_EQ(op->args.size(), 3U)
          << op_name << " expects a destination and floating literal";
      auto destination =
          ParseWholeBufferRegion(op->args[1], op_name + " destination", 2);
      ICHECK(destination.is_local)
          << op_name
          << " destination must have a compiler-owned local descriptor";
      const auto &data_ = destination.descriptor;
      auto dtype = destination.dtype;
      const auto *value_node = op->args[2].as<FloatImmNode>();
      ICHECK(value_node) << op_name << " currently requires a floating literal";
      double value = Downcast<FloatImm>(op->args[2])->value;
      if (target_programming_model_ == "rv") {
        EmitRVFill(data_, dtype, value);
      } else {
        EmitTPUKernelFill(data_, dtype, value);
      }
    } else if (op_name == "tl.tpu.gemm") {
      ICHECK_EQ(op->args.size(), 10U)
          << "tl.tpu.gemm requires the canonical 10-argument ABI, including "
             "an explicit accumulate flag";
      auto a_operand =
          ParseWholeBufferRegion(op->args[1], op_name + " A", 1, true);
      auto b_operand =
          ParseWholeBufferRegion(op->args[2], op_name + " B", 1, true);
      const auto *accumulate_imm = op->args[9].as<IntImmNode>();
      ICHECK(accumulate_imm && accumulate_imm->dtype.is_bool())
          << "tl.tpu.gemm accumulate must be a compile-time boolean";
      bool accumulate = accumulate_imm->value != 0;
      auto c_operand = ParseWholeBufferRegion(op->args[3], op_name + " C",
                                              accumulate ? 3 : 2, true);
      ICHECK(a_operand.is_local && b_operand.is_local && c_operand.is_local)
          << op_name
          << " operands must all have compiler-owned local descriptors";
      ICHECK(c_operand.data_var != a_operand.data_var &&
             c_operand.data_var != b_operand.data_var)
          << op_name
          << " output/accumulator C must use storage distinct from A and B";
      const auto &a_access_data = a_operand.descriptor;
      const auto &b_access_data = b_operand.descriptor;
      const auto &c_access_data = c_operand.descriptor;

      const auto *m_imm = op->args[6].as<IntImmNode>();
      const auto *n_imm = op->args[7].as<IntImmNode>();
      const auto *k_imm = op->args[8].as<IntImmNode>();
      ICHECK(m_imm && n_imm && k_imm)
          << op_name << " M/N/K must be compile-time integers";
      auto M = m_imm->value;
      auto N = n_imm->value;
      auto K = k_imm->value;
      constexpr int64_t kTPUGemmDimLimit =
          static_cast<int64_t>(std::numeric_limits<uint16_t>::max());
      ICHECK_GT(M, 0);
      ICHECK_GT(N, 0);
      ICHECK_GT(K, 0);
      ICHECK_LE(M, kTPUGemmDimLimit)
          << op_name << " M exceeds the TPU GEMM uint16_t limit";
      ICHECK_LE(N, kTPUGemmDimLimit)
          << op_name << " N exceeds the TPU GEMM uint16_t limit";
      ICHECK_LE(K, kTPUGemmDimLimit)
          << op_name << " K exceeds the TPU GEMM uint16_t limit";
      const auto *trans_a_imm = op->args[4].as<IntImmNode>();
      const auto *trans_b_imm = op->args[5].as<IntImmNode>();
      ICHECK(trans_a_imm && trans_a_imm->dtype.is_bool() && trans_b_imm &&
             trans_b_imm->dtype.is_bool())
          << op_name << " transpose flags must be compile-time booleans";
      auto trans_A = trans_a_imm->value != 0;
      auto trans_B = trans_b_imm->value != 0;
      ICHECK_EQ(a_operand.rank, 2U) << op_name << " requires rank-2 A";
      ICHECK_EQ(b_operand.rank, 2U) << op_name << " requires rank-2 B";
      ICHECK_EQ(c_operand.rank, 2U) << op_name << " requires rank-2 C";
      const std::vector<int> a_shape = {a_operand.shape4[1],
                                        a_operand.shape4[3]};
      const std::vector<int> b_shape = {b_operand.shape4[1],
                                        b_operand.shape4[3]};
      const std::vector<int> c_shape = {c_operand.shape4[1],
                                        c_operand.shape4[3]};
      ICHECK_EQ(a_shape[0], trans_A ? K : M)
          << op_name << " A shape disagrees with M/K";
      ICHECK_EQ(a_shape[1], trans_A ? M : K)
          << op_name << " A shape disagrees with M/K";
      ICHECK_EQ(b_shape[0], trans_B ? N : K)
          << op_name << " B shape disagrees with K/N";
      ICHECK_EQ(b_shape[1], trans_B ? K : N)
          << op_name << " B shape disagrees with K/N";
      ICHECK_EQ(c_shape[0], M) << op_name << " C shape disagrees with M/N";
      ICHECK_EQ(c_shape[1], N) << op_name << " C shape disagrees with M/N";

      auto a_dtype = a_operand.dtype;
      auto b_dtype = b_operand.dtype;
      auto c_dtype = c_operand.dtype;
      const bool fp32_gemm = a_dtype == DataType::Float(32);
      if (fp32_gemm) {
        ICHECK(a_operand.scope == "local.matrix" &&
               b_operand.scope == "local.matrix" &&
               c_operand.scope == "local.matrix")
            << op_name << " FP32 operands must use local.matrix storage";
      } else {
        ICHECK(a_operand.scope != "local.matrix" &&
               b_operand.scope != "local.matrix" &&
               c_operand.scope != "local.matrix")
            << op_name << " local.matrix storage is reserved for FP32 GEMM";
      }
      if (target_programming_model_ == "rv") {
        EmitRVGemm(a_access_data, b_access_data, c_access_data, a_dtype,
                   b_dtype, c_dtype, trans_A, trans_B, accumulate, M, N, K);
      } else {
        EmitTPUKernelGemm(a_access_data, b_access_data, c_access_data, a_dtype,
                          b_dtype, c_dtype, trans_A, trans_B, accumulate, M, N,
                          K);
      }
    } else if (op_name == "tl.tpu.sub") {
      handle_elementwise("sub");
    } else if (op_name == "tl.tpu.mul") {
      handle_elementwise("mul");
    } else if (op_name == "tl.tpu.add") {
      handle_elementwise("add");
    } else if (op_name == "tl.tpu.div") {
      handle_elementwise("div");
    } else if (op_name == "tl.tpu.max") {
      handle_elementwise("max");
    } else if (is_tpukernel_extern || is_portable_tpu_op) {
      ICHECK(TryEmitTPUSemantic(op, op_name))
          << "Unknown TPU semantic operation " << op_name;
      if (target_programming_model_ == "rv") {
        // Complete composite uses before the allocator may reuse their LMEM.
        stream << "rvt_sync_i(0xdeadbeef, 0);\n";
      }
    } else {
      ICHECK(is_rvt_extern)
          << "Unknown external call " << op_name
          << " reached TPU codegen; only compiler-owned tl.tpu.*, matching "
             "tl.tpukernel.*, or explicit rvt_* calls are supported";
      ICHECK_EQ(target_programming_model_, "rv")
          << "RVT extern " << op_name
          << " requires target tpu-programming-model=rv";
      ICHECK_EQ(tpukernel_extern_count_, 0)
          << "Raw RVT rvt_* calls cannot share a kernel with TPU-Kernel calls; "
          << "their descriptor and command-stream ownership models are "
             "distinct.";
      ICHECK_EQ(canonical_tpu_op_count_, 0)
          << "Raw RVT rvt_* calls cannot share a kernel with backend-neutral "
             "tl.tpu.* operations; the compiler owns descriptors and lifecycle "
             "for the latter.";
      for (size_t i = 1; i < op->args.size(); ++i) {
        const VarNode *descriptor_var = nullptr;
        PostOrderVisit(op->args[i], [&](const ObjectRef &object) {
          if (const auto *var = object.as<VarNode>()) {
            if (compiler_descriptor_vars_.count(var)) {
              descriptor_var = var;
            }
          }
        });
        ICHECK(!descriptor_var)
            << "Raw RVT extern " << op_name
            << " cannot consume TileLang tensor descriptor Var "
            << descriptor_var->name_hint
            << "; pass user-managed RVT register encodings, or use a "
               "compiler-owned tl.tpu.* operation";
      }
      ++rvt_direct_call_count_;
      uses_rvt_api_ = true;
      uses_opaque_raw_rvt_ = true;
      // Expert RVT calls already carry native C ABI arguments.
      CodeGenC::VisitExpr_(op, os);
    }

  } else if (op->op.same_as(builtin::if_then_else())) {
    // conditional that skips eval if cond evals to false
    const auto *true_var = op->args[1].as<VarNode>();
    const auto *false_var = op->args[2].as<VarNode>();
    const bool true_is_descriptor =
        true_var != nullptr && compiler_descriptor_vars_.count(true_var) != 0;
    const bool false_is_descriptor =
        false_var != nullptr && compiler_descriptor_vars_.count(false_var) != 0;
    ICHECK(!true_is_descriptor && !false_is_descriptor)
        << "A descriptor-valued if_then_else reached TPU codegen, but TPU "
           "residual IR has no contract for selecting tensor descriptors; "
           "select tensor values before lowering instead";
    std::string result = name_supply_->FreshName("condval");
    std::string cond = PrintExpr(op->args[0]);
    this->PrintIndent();
    PrintType(op->dtype, this->stream);
    this->stream << " " << result << ";\n";
    this->PrintIndent();
    this->stream << "if (" << cond << ") {\n";
    {
      int then_scope = this->BeginScope();
      std::string true_val = PrintExpr(op->args[1]);
      this->PrintIndent();
      this->stream << result << " = " << true_val << ";\n";
      this->EndScope(then_scope);
      this->PrintIndent();
      this->stream << "} else {\n";
    }
    {
      int else_scope = this->BeginScope();
      std::string false_val = PrintExpr(op->args[2]);
      this->PrintIndent();
      this->stream << result << " = " << false_val << ";\n";
      this->EndScope(else_scope);
      this->PrintIndent();
      this->stream << "}\n";
    }
    os << result;
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenTileLangTPU::VisitStmt_(const LetStmtNode *op) {
  const auto *value_var = op->value.as<VarNode>();
  ICHECK(value_var == nullptr ||
         compiler_descriptor_vars_.count(value_var) == 0)
      << "A tensor descriptor cannot be bound by LetStmt: descriptor "
         "aliases have no shape/address ownership contract in TPU residual IR";
  std::string value = PrintExpr(op->value);
  if (print_ssa_form_) {
    ICHECK(!var_idmap_.count(op->var.get()));
    var_idmap_[op->var.get()] = value;
  } else {
    PrintIndent();
    if (op->var.dtype() == DataType::Handle() &&
        handle_data_type_.count(op->var.get())) {
      PrintType(handle_data_type_.at(op->var.get()), stream);
      stream << "* " << AllocVarID(op->var.get()) << " = (";
      PrintType(handle_data_type_.at(op->var.get()), stream);
      stream << "*)" << value << ";\n";
    } else {
      // Let bindings are scalar/pointer values. Local tensors are represented
      // only by Allocate-owned descriptors, never inferred from name hints.
      PrintType(op->var.dtype(), this->stream);
      this->stream << ' ' << AllocVarID(op->var.get()) << " = " << value
                   << ";\n";
    }
  }
  PrintStmt(op->body);
}

void CodeGenTileLangTPU::VisitStmt_(const AttrStmtNode *op) {
  LOG(FATAL) << "Residual AttrStmt " << op->attr_key
             << " reached TPU source codegen; target passes must consume "
                "attributes instead of silently discarding their semantics";
}

std::string CodeGenTileLangTPU::AllocLocalVarID(const tir::VarNode *v) {
  std::string key = v->name_hint;
  std::string vid = name_supply_->FreshName(key);
  std::replace(vid.begin(), vid.end(), ':', '_');
  std::replace(vid.begin(), vid.end(), '-', '_');
  std::replace(vid.begin(), vid.end(), '.', '_');
  return vid;
}

static inline std::string
Shape4ToDim4Literal(const std::vector<int64_t> &shape) {
  tl::tpuv7::ValidateDescriptorShape4(shape, "TileLang TPU codegen");
  std::ostringstream os;
  os << "{ " << shape[0] << ", " << shape[1] << ", " << shape[2] << ", "
     << shape[3] << "}";
  return os.str();
}

void CodeGenTileLangTPU::VisitStmt_(const AllocateNode *op) {
  ICHECK(is_one(op->condition))
      << "TPU descriptor allocation is unconditional and requires a "
         "compile-time true condition; got "
      << op->condition;
  const tir::VarNode *buffer_var = op->buffer_var.get();
  const std::string storage_scope = tir::GetPtrStorageScope(op->buffer_var);
  ICHECK(tl::tpuv7::IsLocalMemoryScope(storage_scope))
      << "TileLang TPU allocation " << buffer_var->name_hint
      << " has unsupported scope " << storage_scope
      << "; TPU descriptor allocations require one of shared, shared.dyn, "
         "local, local.fragment";
  alloc_storage_scope_[buffer_var] = storage_scope;
  auto old_var_it = var_idmap_.find(buffer_var);
  bool had_old_var = old_var_it != var_idmap_.end();
  std::string old_var_id;
  if (had_old_var) {
    old_var_id = old_var_it->second;
  }
  std::string vid = AllocLocalVarID(buffer_var);
  var_idmap_[buffer_var] = vid;

  auto shape4 =
      tl::tpuv7::NormalizeLocalShape(op->extents, "TileLang TPU codegen");
  std::string bv_shape = Shape4ToDim4Literal(shape4);
  std::string op_dtype = TargetDTypeName(op->dtype);
  this->PrintIndent();
  const std::string address_attr =
      tl::tpuv7::AddressAttrKey(buffer_var->name_hint);
  Optional<PrimExpr> maybe_addr = f_attrs.GetAttr<PrimExpr>(address_attr);
  ICHECK(maybe_addr.defined())
      << "TileLang TPU codegen requires AddressAssign to attach an LMEM byte "
         "address for buffer "
      << buffer_var->name_hint << " as attribute " << address_attr;
  const auto *addr_imm = maybe_addr.value().as<IntImmNode>();
  ICHECK(addr_imm)
      << "TileLang TPU LMEM address must be a compile-time integer for buffer "
      << buffer_var->name_hint;
  int64_t addr = addr_imm->value;
  ICHECK_GE(addr, 0);
  ICHECK_LE(static_cast<uint64_t>(addr),
            static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
      << "RV TR/TPU-Kernel LMEM addresses are limited to 32 bits";
  buffer_addrs_[buffer_var] = addr;
  compiler_descriptor_vars_.insert(buffer_var);
  descriptor_dtype_[buffer_var] = op->dtype;
  descriptor_element_count_[buffer_var] =
      tl::tpuv7::DescriptorElementCount(shape4, "TileLang TPU allocation");
  descriptor_shape4_[buffer_var] = {
      static_cast<int>(shape4[0]), static_cast<int>(shape4[1]),
      static_cast<int>(shape4[2]), static_cast<int>(shape4[3])};
  descriptor_rank_[buffer_var] = op->extents.size();
  stream << "__tilelang_tpu_tensor_info " << vid << " = {.shape = " << bv_shape
         << ", .stride = {0}"
         << ", .addr = " << addr << ", .default_stride = false};\n";
  this->PrintIndent();
  stream << "tpu_aligned_stride(&" << vid << ".stride, 0, &" << vid
         << ".shape, " << op_dtype << ");\n";
  this->PrintStmt(op->body);

  if (had_old_var) {
    var_idmap_[buffer_var] = old_var_id;
  } else {
    var_idmap_.erase(buffer_var);
  }
}

void CodeGenTileLangTPU::VisitStmt_(const AllocateConstNode *op) {
  LOG(FATAL) << "AllocateConst has no TPU descriptor/load contract; introduce "
                "a typed constant-table operation before enabling constant "
                "array source emission";
}

void CodeGenTileLangTPU::VisitStmt_(const CustomizedCodeNode *op) {
  LOG(FATAL) << "CustomizedCode is forbidden at the TPU source boundary: "
                "verbatim source injection bypasses programming-model, "
                "descriptor, and command-lifecycle validation";
}

void CodeGenTileLangTPU::VisitStmt_(const DeclBufferNode *op) {
  const VarNode *data_var = op->buffer->data.get();
  auto dtype_it = descriptor_dtype_.find(data_var);
  auto descriptor_it = var_idmap_.find(data_var);
  ICHECK(dtype_it != descriptor_dtype_.end() &&
         descriptor_it != var_idmap_.end() && buffer_addrs_.count(data_var))
      << "Residual DeclBuffer " << op->buffer->name
      << " does not describe a compiler-owned TPU local allocation";
  ICHECK_EQ(op->buffer->dtype, dtype_it->second)
      << "DeclBuffer " << op->buffer->name << " dtype " << op->buffer->dtype
      << " disagrees with allocation dtype " << dtype_it->second;
  const std::string scope = op->buffer.scope();
  ICHECK(tl::tpuv7::IsLocalMemoryScope(scope))
      << "DeclBuffer " << op->buffer->name
      << " must use a TPU local-memory scope, got " << scope;
  auto allocation_scope = alloc_storage_scope_.find(data_var);
  ICHECK(allocation_scope != alloc_storage_scope_.end() &&
         allocation_scope->second == scope)
      << "DeclBuffer " << op->buffer->name << " scope " << scope
      << " disagrees with its Allocate scope "
      << (allocation_scope == alloc_storage_scope_.end()
              ? std::string("<missing>")
              : allocation_scope->second);
  ICHECK(is_zero(op->buffer->elem_offset) && op->buffer->strides.empty() &&
         op->buffer->axis_separators.empty() &&
         op->buffer->buffer_type == BufferType::kDefault)
      << "DeclBuffer " << op->buffer->name
      << " must be a canonical zero-offset contiguous TPU tensor descriptor";
  const auto declared_shape4 = tl::tpuv7::NormalizeLocalShape(
      op->buffer->shape, "TileLang TPU DeclBuffer");
  auto rank_it = descriptor_rank_.find(data_var);
  ICHECK(rank_it != descriptor_rank_.end() &&
         rank_it->second == op->buffer->shape.size())
      << "DeclBuffer " << op->buffer->name
      << " rank disagrees with its Allocate; rank-changing descriptor views "
         "are not supported";
  const auto &owned_shape =
      DescriptorShape4(data_var, "TileLang TPU DeclBuffer " + op->buffer->name);
  ICHECK(std::equal(declared_shape4.begin(), declared_shape4.end(),
                    owned_shape.begin(), owned_shape.end()))
      << "DeclBuffer " << op->buffer->name
      << " shape disagrees with its Allocate; shape-changing descriptor views "
         "are not supported";
  this->PrintStmt(op->body);
}

void CodeGenTileLangTPU::VisitExpr_(const RampNode *op, std::ostream &os) {
  LOG(FATAL) << "Ramp " << GetRef<PrimExpr>(op)
             << " reached scalar-only TPU codegen";
}

inline void PrintConst(const FloatImmNode *op, std::ostream &os,
                       CodeGenTileLangTPU *p) { // NOLINT(*)
  if (op->dtype.is_bfloat16()) {
    os << "bfloat16_t(";
    FloatImm const_f32 = FloatImm(DataType::Float(32), op->value);
    PrintConst(const_f32.get(), os, p);
    os << ')';
    return;
  }
  switch (op->dtype.bits()) {
  case 64:
  case 32: {
    std::ostringstream temp;
    if (std::isinf(op->value)) {
      if (op->value < 0) {
        temp << "-";
      }
      temp << ((op->dtype.bits() == 32) ? "__builtin_inff()"
                                        : "__builtin_inf()");
    } else if (std::isnan(op->value)) {
      temp << ((op->dtype.bits() == 32) ? "__builtin_nanf(\"\")"
                                        : "__builtin_nan(\"\")");
    } else {
      temp << std::scientific << op->value;
      if (op->dtype.bits() == 32)
        temp << 'f';
    }
    p->MarkConst(temp.str());
    os << temp.str();
    break;
  }
  case 16: {
    os << "half_t" << '(';
    FloatImm const_f32 = FloatImm(DataType::Float(32), op->value);
    PrintConst(const_f32.get(), os, p);
    os << ')';
    break;
  }
  default:
    LOG(FATAL) << "Bad bit-width for float: " << op->dtype << "\n";
  }
}

void CodeGenTileLangTPU::VisitExpr_(const FloatImmNode *op,
                                    std::ostream &os) { // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenTileLangTPU::VisitExpr_(const VarNode *op,
                                    std::ostream &os) { // NOLINT(*)
  ICHECK_EQ(compiler_descriptor_vars_.count(op), 0U)
      << "TPU tensor descriptor Var " << op->name_hint
      << " cannot be emitted as a scalar/pointer expression; consume it "
         "through a typed tl.tpu.* or tl.tpukernel.* operand";
  CodeGenC::VisitExpr_(op, os);
}

template <typename T>
inline void PrintBinaryExpr(const T *op, const char *opstr,
                            std::ostream &os, // NOLINT(*)
                            CodeGenC *p) {
  if (op->dtype.lanes() == 1) {
    if (isalpha(opstr[0])) {
      os << opstr << '(';
      p->PrintExpr(op->a, os);
      os << ", ";
      p->PrintExpr(op->b, os);
      os << ')';
    } else {
      os << '(';
      p->PrintExpr(op->a, os);
      os << ' ' << opstr << ' ';
      p->PrintExpr(op->b, os);
      os << ')';
    }
  } else {
    p->PrintVecBinaryOp(opstr, op->dtype, op->a, op->b, os);
  }
}

void CodeGenTileLangTPU::VisitExpr_(const FloorModNode *op,
                                    std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr(op, "%", os, this);
}

void CodeGenTileLangTPU::VisitExpr_(const FloorDivNode *op,
                                    std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr(op, "/", os, this);
}

void CodeGenTileLangTPU::HandleVolatileLoads(const std::string &value,
                                             const BufferLoadNode *op,
                                             std::ostream &os) {
  os << value;
}

void CodeGenTileLangTPU::PrintVecElemLoadExpr(DataType t, int i,
                                              const std::string &value,
                                              std::ostream &os) {
  LOG(FATAL) << "Vector element expression at lane " << i << " with dtype " << t
             << " reached scalar-only TPU codegen";
}

void CodeGenTileLangTPU::AddFunction(const PrimFunc &f) {
  this->InitFuncState(f);
  buffer_shape.clear();
  buffer_stride.clear();
  buffer_addrs_.clear();
  descriptor_dtype_.clear();
  descriptor_element_count_.clear();
  descriptor_shape4_.clear();
  descriptor_rank_.clear();
  global_buffer_descriptor_.clear();
  compiler_descriptor_vars_.clear();
  // Finish() emits one shared preamble for the entire IRModule, so this is a
  // module-level OR rather than per-function state.  Resetting it here makes
  // an RVT function silently lose rvt_api.h when a later ordinary PrimFunc is
  // emitted in the same module.
  rvt_direct_call_count_ = 0;
  tpukernel_extern_count_ = 0;
  canonical_tpu_op_count_ = 0;
  TPUExternUsageExtractor extern_usage;
  extern_usage(f->body);
  ICHECK(!(extern_usage.has_portable_tpu_op && extern_usage.has_raw_rvt_op))
      << "Backend-neutral tl.tpu.* operations cannot share a kernel with raw "
         "rvt_* calls: compiler-owned and user-owned RV descriptors/lifecycle "
         "would conflict";
  ReserveKeywordsAsUnique();
  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  f_attrs = f->attrs;
  ICHECK(global_symbol.defined())
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";
  auto buffer_map = f->buffer_map;
  this->PrintFuncPrefix(stream);
  CodeGenC::PrintType(f->ret_type, stream);
  this->PrintExtraAttrs(f, stream);
  std::string global_name = static_cast<std::string>(global_symbol.value());
  ICHECK(global_name != "main" && global_name != "main_kernel")
      << "TPU inner-kernel global_symbol " << global_name
      << " conflicts with a generated runtime entry; use main_kernel_inner or "
         "another non-reserved name";
  this->stream << " " << global_name << "(";
  std::vector<std::string> params_name;
  std::vector<std::string> global_descriptor_declarations;

  // The stable wrapper ABI uses v1..vN for raw addresses and v(N+1)..v(2N)
  // for their descriptors.  Reserve those identifiers before allocating any
  // local/temporary name so a legal TIR Var named, for example, ``v2`` cannot
  // generate two C declarations with the same identifier.
  const int param_len = static_cast<int>(f->params.size());
  for (int index = 1; index <= 2 * param_len; ++index) {
    name_supply_->ReserveName("v" + std::to_string(index));
  }

  // Compute the contiguous global tensor stride from its normalized dim4.
  auto default_stride = [this](const std::string &node) {
    auto shape_it = buffer_shape.find(node);
    ICHECK(shape_it != buffer_shape.end() && shape_it->second.size() == 4U)
        << "Cannot compute a TPU tensor stride without a normalized dim4 for "
        << node;
    const auto &buf_shape = shape_it->second;
    auto [stride_it, inserted] =
        buffer_stride.emplace(node, std::vector<int>{1, 1, 1, 1});
    ICHECK(inserted) << "Duplicate TPU stride metadata for " << node;
    auto &stride = stride_it->second;
    for (int i = 2; i >= 0; i--) {
      int64_t next_stride =
          static_cast<int64_t>(buf_shape[i + 1]) * stride[i + 1];
      ICHECK_LE(next_stride, std::numeric_limits<int>::max())
          << "TileLang TPU global tensor stride exceeds the descriptor ABI "
             "int range";
      stride[i] = static_cast<int>(next_stride);
    }
  };

  // Generated ABI identifiers are positional and independent of name_hint.
  auto allocate_name = [&, this](const Var &v, int index, int length) {
    auto v_node = v.get();
    std::string vid = "v" + std::to_string(index + 1);
    std::string rid = "v" + std::to_string(index + 1 + length);

    auto buffer_node = buffer_map[v];
    ICHECK_EQ(buffer_node.scope(), "global")
        << "TileLang TPU kernel parameter " << buffer_node->name
        << " must use global scope";
    ICHECK(is_zero(buffer_node->elem_offset) && buffer_node->strides.empty() &&
           buffer_node->axis_separators.empty() &&
           buffer_node->buffer_type == BufferType::kDefault)
        << "TileLang TPU kernel parameter " << buffer_node->name
        << " must be a canonical zero-offset contiguous Buffer";
    auto shape = buffer_node->shape;
    auto dim4_shape = LowerGlobalShapeToDim4(shape);
    auto [shape_it, shape_inserted] = buffer_shape.emplace(rid, dim4_shape);
    ICHECK(shape_inserted) << "Duplicate TPU descriptor metadata for " << rid;
    default_stride(rid);
    std::string shape_s = Dim4Initializer(dim4_shape);

    int64_t tensor_size = tl::tpuv7::DescriptorElementCount(
        dim4_shape, "TileLang TPU global tensor");
    descriptor_dtype_[buffer_node->data.get()] = buffer_node->dtype;
    descriptor_element_count_[buffer_node->data.get()] = tensor_size;
    descriptor_shape4_[buffer_node->data.get()] = dim4_shape;
    descriptor_rank_[buffer_node->data.get()] = buffer_node->shape.size();
    std::string inst =
        "__tilelang_tpu_tensor_info " + rid + " = {.shape = " + shape_s +
        ", .stride = {0}, .addr = " + vid + ", .default_stride = true};\n";
    global_descriptor_declarations.push_back(inst);
    this->var_idmap_[v_node] = rid;
    compiler_descriptor_vars_.insert(v_node);
    compiler_descriptor_vars_.insert(buffer_node->data.get());
    auto [descriptor_it, descriptor_inserted] =
        this->global_buffer_descriptor_.emplace(buffer_node->data.get(), rid);
    ICHECK(descriptor_inserted)
        << "Two TPU kernel parameters share one Buffer::data Var "
        << buffer_node->data->name_hint;
    return vid;
  };
  for (size_t i = 0; i < param_len; ++i) {
    tir::Var v = f->params[i];
    ICHECK(buffer_map.count(v))
        << "TileLang TPU source ABI supports only buffer parameters; scalar "
           "parameter "
        << v->name_hint << " has no host/device marshalling contract";
    std::string vid = allocate_name(v, i, param_len);
    params_name.push_back(vid);
    if (i != 0)
      stream << ", ";
    stream << restrict_keyword_ << ' ' << vid;
  }
  stream << ") {\n";

  this->PreFunctionBody(f);
  int func_scope = this->BeginScope();

  const bool use_compiler_owned_rv_lifecycle =
      target_programming_model_ == "rv" && extern_usage.has_portable_tpu_op;
  if (use_compiler_owned_rv_lifecycle) {
    this->PrintIndent();
    this->stream << "rvt_cfg_lanemask(gdma_get_lane_mask());\n";
  }

  for (const std::string &declaration : global_descriptor_declarations) {
    this->PrintIndent();
    this->stream << declaration;
  }
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";

  // Stable runtime wrapper shared by both device programming models.
  this->stream << "typedef struct {\n";
  for (auto &name : params_name) {
    this->stream << "  " << restrict_keyword_ << " " << name << ";\n";
  }
  std::string api_name = "tpu_kernel_api_main_inner_args_t";
  this->stream << "} " << api_name << ";\n";
  // The selected backend uniquely owns its command-stream lifecycle.
  this->stream << "int "
               << "main_kernel(const void * args) {\n"
               << "  " << api_name << " *api = (" << api_name << "*)args;\n";
  if (target_programming_model_ == "tpukernel") {
    this->stream << "  tpu_initialize();\n";
  } else if (use_compiler_owned_rv_lifecycle) {
    this->stream << "  rvt_kernel_start();\n";
  }
  this->stream << "  " << global_name << "(";
  int name_index = 0;
  int name_len = params_name.size();
  for (auto &name : params_name) {
    if (name_index != 0)
      this->stream << "    ";
    this->stream << "api->" << name;

    if (name_index == name_len - 1)
      this->stream << ");";
    else
      this->stream << ",";
    this->stream << "\n";
    name_index += 1;
  }
  if (params_name.empty()) {
    this->stream << ");\n";
  }
  if (target_programming_model_ == "tpukernel") {
    this->stream << "  tpu_poll();\n";
  } else if (use_compiler_owned_rv_lifecycle) {
    this->stream << "  rvt_sync_i(0xdeadbeef, 0);\n";
  }
  this->stream << "  return 0;\n}\n";
  this->stream << "TPUKERNEL_FUNC_REGISTER("
               << "main_kernel)\n";
}

} // namespace codegen
} // namespace tvm
