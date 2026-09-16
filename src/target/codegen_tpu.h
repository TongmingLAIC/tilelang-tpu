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
 * \file target/codegen_tpu.h
 * \brief TileLang TPU source code generator.
 */
#ifndef TVM_TL_TARGET_CODEGEN_TPU_H_
#define TVM_TL_TARGET_CODEGEN_TPU_H_

#include <tvm/target/codegen.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "target/source/codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenTileLangTPU final : public CodeGenC {
public:
  CodeGenTileLangTPU(std::string target_chip,
                     std::string target_programming_model);
  std::string Finish();
  // CodeGenC overrides required by the TPU source ABI.
  void PrintFuncPrefix(std::ostream &os) final;
  void PrintExtraAttrs(const PrimFunc &f, std::ostream &os) final;
  void VisitStmt_(const ForNode *op) final;
  void PrintStorageSync(const CallNode *op) final;
  void PrintStorageScope(const std::string &scope,
                         std::ostream &os) final; // NOLINT(*)
  void PrintVecBinaryOp(const std::string &op, DataType t, PrimExpr lhs,
                        PrimExpr rhs,
                        std::ostream &os) final;      // NOLINT(*)
  void PrintType(DataType t, std::ostream &os) final; // NOLINT(*)
  void PrintVecElemLoad(const std::string &vec, DataType t, int i,
                        std::ostream &os) final; // NOLINT(*)
  void PrintVecElemStore(const std::string &vec, DataType t, int i,
                         const std::string &value) final;
  void BindThreadIndex(const IterVar &iv) final; // NOLINT(*)
  void PrintVecElemLoadExpr(DataType t, int i, const std::string &value,
                            std::ostream &os) final;
  std::string CastFromTo(std::string value, DataType from,
                         DataType target) final;
  void VisitExpr_(const RampNode *op, std::ostream &os) final; // NOLINT(*)
  void VisitExpr_(const FloatImmNode *op, std::ostream &os) final;
  void VisitExpr_(const VarNode *op, std::ostream &os) final;
  void VisitExpr_(const CallNode *op, std::ostream &os) final;
  void VisitExpr_(const CastNode *op, std::ostream &os) final;
  void VisitStmt_(const AllocateNode *op) final;
  void VisitStmt_(const AllocateConstNode *op) final;
  void VisitStmt_(const CustomizedCodeNode *op) final;
  void VisitStmt_(const DeclBufferNode *op) final;
  void VisitStmt_(const AttrStmtNode *op) final;
  void VisitStmt_(const LetStmtNode *op) final;
  void VisitExpr_(const FloorDivNode *op, std::ostream &os) final;
  void VisitExpr_(const FloorModNode *op, std::ostream &os) final;

  // Emit the stable TPU entry wrapper and initialize descriptor state.
  void AddFunction(const PrimFunc &f);

protected:
  virtual std::string GetBufferRef(DataType t, const BufferNode *buffer,
                                   PrimExpr index) final;
  void PrintCallExtern(Type ret_type, String global_symbol,
                       const Array<PrimExpr> &args, bool skip_first_arg,
                       std::ostream &os) final; // NOLINT(*)

private:
  struct SemanticTensorOperand {
    const VarNode *data_var{nullptr};
    std::string descriptor;
    DataType dtype;
    std::vector<int> shape4;
    std::string scope;
    size_t rank{0};
    bool is_local{false};
  };

  // Backend-specific emission.  Parsing TIR operands and maintaining the
  // stable kernel ABI stay in this class; instruction selection lives in the
  // two sibling translation units named after their programming model.
  void EmitTPUKernelCopy(const std::string &src, bool src_is_global,
                         const std::string &src_dtype, const std::string &dst,
                         bool dst_is_global, const std::string &dst_dtype);
  void EmitRVCopy(const std::string &src, bool src_is_global,
                  const std::string &src_dtype, const std::string &dst,
                  bool dst_is_global, const std::string &dst_dtype);
  void EmitTPUKernelMatrixCopy(const std::string &src, bool src_is_global,
                               const std::string &dst, bool dst_is_global,
                               DataType dtype, int64_t rows, int64_t cols);
  void EmitRVMatrixCopy(const std::string &src, bool src_is_global,
                        const std::string &dst, bool dst_is_global,
                        DataType dtype, int64_t rows, int64_t cols);
  void EmitTPUKernelFill(const std::string &dst, DataType dtype, double value);
  void EmitRVFill(const std::string &dst, DataType dtype, double value);
  void EmitTPUKernelGemm(const std::string &a, const std::string &b,
                         const std::string &c, DataType a_dtype,
                         DataType b_dtype, DataType c_dtype, bool transpose_a,
                         bool transpose_b, bool accumulate, int64_t m,
                         int64_t n, int64_t k);
  void EmitRVGemm(const std::string &a, const std::string &b,
                  const std::string &c, DataType a_dtype, DataType b_dtype,
                  DataType c_dtype, bool transpose_a, bool transpose_b,
                  bool accumulate, int64_t m, int64_t n, int64_t k);
  void EmitTPUKernelElementwise(const std::string &operation,
                                const std::string &dst, const std::string &src0,
                                const std::string &src1, DataType dtype,
                                const std::vector<int> &src0_shape,
                                const std::vector<int> &src1_shape);
  void EmitTPUKernelScalar(const std::string &operation, const std::string &dst,
                           const std::string &src, DataType dtype,
                           double value);
  void EmitRVElementwise(const std::string &operation, const std::string &dst,
                         const std::string &src0, const std::string &src1,
                         DataType dst_dtype, DataType src0_dtype,
                         DataType src1_dtype, const std::vector<int> &dst_shape,
                         const std::vector<int> &src0_shape,
                         const std::vector<int> &src1_shape);
  void EmitRVConstant(double value, const std::string &dtype);
  void EmitRVScalar(const std::string &operation, const std::string &dst,
                    const std::string &src, DataType dtype, double value);
  void EmitRVReduction(const std::string &operation, const std::string &src,
                       const std::string &dst, DataType dtype, int width);
  void EmitRVExp(const std::string &dst, const std::string &work0,
                 const std::string &work1, DataType dtype);
  void EmitRVDescriptor(const std::string &tensor, int register_id,
                        bool is_global, const std::string &dtype,
                        bool hw_aligned);
  // Shared extended semantic validation preserves the existing TPU-Kernel ABI.
  // RV instruction sequences are implemented in codegen_rv.cc.
  bool TryEmitTPUSemantic(const CallNode *op, const std::string &op_name);

  // Handle volatile loads.
  void HandleVolatileLoads(const std::string &value, const BufferLoadNode *op,
                           std::ostream &os) final;

  // TPU address spaces are represented by tensor descriptors, not C types.
  bool IsScopePartOfType() const final { return false; }

  friend void PrintConst(const FloatImmNode *op, std::ostream &os,
                         CodeGenTileLangTPU *p);
  std::string AllocLocalVarID(const tir::VarNode *v);
  SemanticTensorOperand ParseWholeBufferRegion(const PrimExpr &expr,
                                               const std::string &context,
                                               int expected_access_mask,
                                               bool allow_matrix = false) const;
  const std::vector<int> &DescriptorShape4(const VarNode *data_var,
                                           const std::string &context) const;
  // Global semantic operands carry Buffer::data variables, while generated
  // function arguments are positional handles.  Bridge them by Var identity;
  // Buffer names are presentation metadata and need not be unique in TIR.
  std::unordered_map<const VarNode *, std::string> global_buffer_descriptor_;
  // Every handle/Var that the compiler maps to a tensor-info descriptor.
  // Raw RVT externs accept register encodings, never these C struct values.
  std::unordered_set<const VarNode *> compiler_descriptor_vars_;
  // Normalized N/C/H/W metadata for positional global descriptors. The
  // source-level shape maps below are keyed by Var and cover local tensors too.
  std::unordered_map<std::string, std::vector<int>> buffer_shape;
  std::unordered_map<std::string, std::vector<int>> buffer_stride;
  std::unordered_map<const VarNode *, int> buffer_addrs_;
  std::unordered_map<const VarNode *, DataType> descriptor_dtype_;
  std::unordered_map<const VarNode *, int64_t> descriptor_element_count_;
  std::unordered_map<const VarNode *, std::vector<int>> descriptor_shape4_;
  // Preserve source rank as well as normalized dim4.  Rank-changing Buffer
  // aliases may normalize to the same dim4 but are outside the closed tensor
  // descriptor/view contract.
  std::unordered_map<const VarNode *, size_t> descriptor_rank_;

  // Module-level include/ABI requirements plus per-function ownership fences.
  bool uses_rvt_api_{false};
  bool uses_tpukernel_api_{false};
  bool uses_opaque_raw_rvt_{false};
  int rvt_direct_call_count_{0};
  int tpukernel_extern_count_{0};
  int canonical_tpu_op_count_{0};
  std::string target_chip_;
  std::string target_programming_model_;

  DictAttrs f_attrs;
  std::vector<std::pair<tir::Var, Range>> loop_var_ranges_;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_CODEGEN_TPU_H_
