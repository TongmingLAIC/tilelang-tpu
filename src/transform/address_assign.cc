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
 * \file address_assign.cc
 * \brief TPUv7 local-memory lifetime, bank-conflict, and address assignment.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ir/type.h>
#include <tvm/relay/expr.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target_info.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../op/builtin.h"
#include "../op/bulk_copy.h"
#include "../op/gemm.h"
#include "../target/tpu_target_info.h"
#include "../target/tpuv7_lmem.h"

namespace tvm {
namespace tl {
using namespace tir;

namespace {

int64_t AlignUp(int64_t value, int64_t align) {
  return tpuv7::AlignUp(value, align);
}

std::string GetPointerStorageScope(const Var &var) {
  const auto *pointer_type = var->type_annotation.as<PointerTypeNode>();
  ICHECK(pointer_type)
      << "AddressAssign requires a pointer-typed allocation Var, got " << var
      << " with type " << var->type_annotation;
  return pointer_type->storage_scope;
}

} // namespace

class AddressAllocator : public StmtExprVisitor {
public:
  AddressAllocator() = default;

  std::vector<const BufferNode *> collectAllocOp(tir::Stmt body) {
    this->VisitStmt(body);
    ICHECK_EQ(allocation_data_vars_.size(), declared_data_vars_.size())
        << "AddressAssign requires every TPU local Allocate to own exactly one "
           "canonical DeclBuffer";
    return alloc_ops_;
  }

  void VisitStmt_(const AllocateNode *op) final {
    const VarNode *data_var = op->buffer_var.get();
    const std::string scope = GetPointerStorageScope(op->buffer_var);
    ICHECK(tpuv7::IsLocalMemoryScope(scope))
        << "AddressAssign only supports TPU descriptor allocations; buffer "
        << data_var->name_hint << " has scope " << scope;
    ICHECK(is_one(op->condition))
        << "AddressAssign requires an unconditional local allocation for "
        << data_var->name_hint;
    ICHECK(allocation_data_vars_.insert(data_var).second)
        << "AddressAssign found multiple Allocate nodes for data Var "
        << data_var->name_hint;
    ICHECK(active_allocations_.emplace(data_var, op).second)
        << "AddressAssign found a recursively shadowed allocation Var "
        << data_var->name_hint;
    this->VisitStmt(op->body);
    active_allocations_.erase(data_var);
  }

  void VisitStmt_(const DeclBufferNode *op) final {
    const VarNode *data_var = op->buffer->data.get();
    auto allocation = active_allocations_.find(data_var);
    ICHECK(allocation != active_allocations_.end())
        << "AddressAssign local DeclBuffer " << op->buffer->name
        << " is not lexically owned by a matching Allocate";
    ICHECK(declared_data_vars_.insert(data_var).second)
        << "AddressAssign found multiple local DeclBuffers for allocation "
        << data_var->name_hint;
    const AllocateNode *allocate = allocation->second;
    ICHECK_EQ(op->buffer->dtype, allocate->dtype)
        << "AddressAssign DeclBuffer " << op->buffer->name
        << " dtype disagrees with its Allocate";
    const std::string buffer_scope = op->buffer.scope();
    const std::string allocation_scope =
        GetPointerStorageScope(allocate->buffer_var);
    ICHECK_EQ(buffer_scope, allocation_scope)
        << "AddressAssign DeclBuffer " << op->buffer->name << " scope "
        << buffer_scope << " disagrees with its Allocate scope "
        << allocation_scope;
    ICHECK(is_zero(op->buffer->elem_offset) && op->buffer->strides.empty() &&
           op->buffer->axis_separators.empty() &&
           op->buffer->buffer_type == BufferType::kDefault)
        << "AddressAssign DeclBuffer " << op->buffer->name
        << " must be a canonical zero-offset contiguous TPU tensor";
    ICHECK_EQ(op->buffer->shape.size(), allocate->extents.size())
        << "AddressAssign DeclBuffer " << op->buffer->name
        << " rank disagrees with its Allocate";
    const auto buffer_shape = tpuv7::NormalizeLocalShape(
        op->buffer->shape, "AddressAssign DeclBuffer");
    const auto allocation_shape =
        tpuv7::NormalizeLocalShape(allocate->extents, "AddressAssign Allocate");
    ICHECK(buffer_shape == allocation_shape)
        << "AddressAssign DeclBuffer " << op->buffer->name
        << " shape disagrees with its Allocate";
    alloc_ops_.emplace_back(op->buffer.get());
    this->VisitStmt(op->body);
  }

private:
  std::vector<const BufferNode *> alloc_ops_;
  std::unordered_map<const VarNode *, const AllocateNode *> active_allocations_;
  std::unordered_set<const VarNode *> allocation_data_vars_;
  std::unordered_set<const VarNode *> declared_data_vars_;
};

struct TensorLive {
  uint32_t start = 0;
  uint32_t end = 0;
  int64_t tensor_size = 0;
};

struct OpAddr {
  const BufferNode *op;
  int64_t start = 0;
  int64_t end = 0;
  int64_t size = 0;
  uint32_t first_pos = 0;
  uint32_t end_pos = 0;

  OpAddr(const BufferNode *_op, int64_t _size, uint32_t _first_pos,
         uint32_t _end_pos) {
    op = _op;
    size = _size;
    first_pos = _first_pos;
    end_pos = _end_pos;
  }
};

bool LiveRangesOverlap(const TensorLive &lhs, const TensorLive &rhs) {
  return std::max(lhs.start, rhs.start) < std::min(lhs.end, rhs.end);
}

bool LiveRangesOverlap(const OpAddr &lhs, const OpAddr &rhs) {
  return std::max(lhs.first_pos, rhs.first_pos) <
         std::min(lhs.end_pos, rhs.end_pos);
}

class MemAllocBankConflictAware {
public:
  MemAllocBankConflictAware(int64_t bank_num, int64_t bank_size)
      : bank_num_(bank_num), bank_size_(bank_size) {
    mem_size_ = bank_num * bank_size;
    bank_ops.resize(bank_num);
  }

  bool assignAddr(
      std::vector<const BufferNode *> &ops,
      std::unordered_map<const BufferNode *, TensorLive> &liveRange,
      std::unordered_map<const BufferNode *,
                         std::unordered_set<const BufferNode *>> &conflictMap,
      std::unordered_map<const BufferNode *, int64_t> &addrMap) {
    std::list<const BufferNode *> op_list;
    std::copy(ops.begin(), ops.end(), std::back_inserter(op_list));

    op_list.sort(
        [&liveRange, &conflictMap](const BufferNode *a, const BufferNode *b) {
          auto &lhs = liveRange[a];
          auto &rhs = liveRange[b];
          size_t lhs_degree = conflictMap[a].size();
          size_t rhs_degree = conflictMap[b].size();
          if (lhs_degree != rhs_degree) {
            return lhs_degree > rhs_degree;
          }
          if (lhs.tensor_size != rhs.tensor_size) {
            return lhs.tensor_size > rhs.tensor_size;
          }
          return lhs.start < rhs.start;
        });
    for (auto &op : op_list) {
      std::shared_ptr<OpAddr> best_addr;
      int64_t min_conflict_count = std::numeric_limits<int64_t>::max();
      if (liveRange[op].tensor_size > mem_size_) {
        return false;
      }
      int64_t bytes = liveRange[op].tensor_size;
      int64_t mem_cross_bank_num =
          static_cast<int64_t>(tpuv7::DivUp(bytes, bank_size_));
      mem_cross_bank_num = std::max<int64_t>(mem_cross_bank_num, 1);
      for (int i = 0; i < bank_num_; ++i) {
        int64_t offset = i * bank_size_;

        if (i + mem_cross_bank_num > bank_num_) {
          break;
        }
        int64_t end_offset =
            std::min(offset + (mem_cross_bank_num + 1) * bank_size_, mem_size_);
        auto op_addr = searchAddr(op, liveRange, offset, end_offset);

        // op can insert
        if (op_addr->start + op_addr->size <= end_offset) {
          int64_t conf_count =
              getConflictCount(op_addr, liveRange, conflictMap);
          bool better_addr = !best_addr || conf_count < min_conflict_count ||
                             (conf_count == min_conflict_count &&
                              (op_addr->end < best_addr->end ||
                               (op_addr->end == best_addr->end &&
                                op_addr->start < best_addr->start)));
          if (better_addr) {
            min_conflict_count = conf_count;
            best_addr = op_addr;
          }
        }
      }
      if (!best_addr) {
        return false;
      }
      insertAddr(best_addr);
      // addrMap.Set(op, best_addr->start);
      addrMap[op] = best_addr->start;
    }
    return true;
  }

protected:
  void insertAddr(std::shared_ptr<OpAddr> &opAddr) {
    auto iter =
        std::find_if(allocated_op_list_.begin(), allocated_op_list_.end(),
                     [&opAddr](std::shared_ptr<OpAddr> &p) {
                       return p->start >= opAddr->start;
                     });
    allocated_op_list_.emplace(iter, opAddr);
    int64_t bank_start = opAddr->start / bank_size_;
    int64_t bank_end = (opAddr->end - 1) / bank_size_;
    for (int i = bank_start; i <= bank_end; i++) {
      bank_ops[i].push_back(opAddr->op);
    }
  }

  int64_t getConflictCount(
      std::shared_ptr<OpAddr> &opAddr,
      std::unordered_map<const BufferNode *, TensorLive> &liveRange,
      std::unordered_map<const BufferNode *,
                         std::unordered_set<const BufferNode *>> &conflictMap) {
    int64_t bank_start = opAddr->start / bank_size_;
    int64_t bank_end = (opAddr->end - 1) / bank_size_;
    int count = 0;
    for (int i = bank_start; i <= bank_end; i++) {
      for (auto op : bank_ops[i]) {
        if (conflictMap[opAddr->op].count(op) &&
            LiveRangesOverlap(liveRange[opAddr->op], liveRange[op])) {
          ++count;
        }
      }
    }
    return count;
  }

  std::shared_ptr<OpAddr>
  searchAddr(const BufferNode *op,
             std::unordered_map<const BufferNode *, TensorLive> &liveRange,
             int64_t offset, int64_t end_offset) {

    std::shared_ptr<OpAddr> op_addr = std::make_shared<OpAddr>(
        op, liveRange[op].tensor_size, liveRange[op].start, liveRange[op].end);
    int64_t prev_offset = AlignUp(offset, tpuv7::kTensorAlignBytes);
    int64_t best_offset = -1;
    int64_t smallest_gap = std::numeric_limits<int64_t>::max();

    for (auto &allocated_op_addr : allocated_op_list_) {
      if (allocated_op_addr->end <= offset) {
        continue;
      }
      if (allocated_op_addr->start >= end_offset) {
        break;
      }
      if (LiveRangesOverlap(*op_addr, *allocated_op_addr)) {
        int64_t candidate = AlignUp(prev_offset, tpuv7::kTensorAlignBytes);
        int64_t gap = allocated_op_addr->start - candidate;
        if (gap >= op_addr->size && gap < smallest_gap) {
          smallest_gap = gap;
          best_offset = candidate;
        }
        prev_offset = std::max(prev_offset, allocated_op_addr->end);
      }
    }
    int64_t trailing_candidate = AlignUp(prev_offset, tpuv7::kTensorAlignBytes);
    int64_t trailing_gap = end_offset - trailing_candidate;
    if (trailing_gap >= op_addr->size && trailing_gap < smallest_gap) {
      best_offset = trailing_candidate;
    } else if (best_offset == -1) {
      best_offset = trailing_candidate;
    }
    op_addr->start = best_offset;
    op_addr->end = op_addr->start + op_addr->size;
    return op_addr;
  }

protected:
  std::list<std::shared_ptr<OpAddr>> allocated_op_list_;
  std::vector<std::vector<const BufferNode *>> bank_ops;
  int64_t bank_num_;
  int64_t bank_size_;
  int64_t mem_size_;
};

enum class BufferAccessKind {
  kRead,
  kWrite,
  kReadWrite,
  kConservative,
};

class BufferUseCollector : public StmtExprVisitor {
public:
  explicit BufferUseCollector(
      const std::vector<const BufferNode *> &alloc_ops,
      std::unordered_map<const BufferNode *, TensorLive> *live_ranges,
      std::unordered_map<const BufferNode *,
                         std::unordered_set<const BufferNode *>> *conflict_map)
      : live_ranges_(live_ranges), conflict_map_(conflict_map) {
    for (const BufferNode *buffer : alloc_ops) {
      buffer_var_to_buffer_[buffer->data.get()] = buffer;
      (*conflict_map_)[buffer];
    }
  }

  void Analyze(const Stmt &body) {
    VisitStmt(body);
    for (auto &kv : *live_ranges_) {
      if (!seen_buffers_.count(kv.first)) {
        kv.second.start = 0;
        kv.second.end = 0;
      }
    }
  }

private:
  class OpScope {
  public:
    explicit OpScope(BufferUseCollector *collector) : collector_(collector) {
      if (!collector_->inside_op_) {
        started_ = true;
        collector_->inside_op_ = true;
        collector_->current_loc_ = collector_->NextLoc();
      }
    }

    ~OpScope() {
      if (started_) {
        collector_->FinishCurrentOp();
        collector_->inside_op_ = false;
      }
    }

  private:
    BufferUseCollector *collector_;
    bool started_ = false;
  };

  uint32_t NextLoc() { return ++loc_; }

  void FinishCurrentOp() {
    if (!conservative_buffers_.empty()) {
      for (auto buffer : read_buffers_) {
        conservative_buffers_.insert(buffer);
      }
      for (auto buffer : write_buffers_) {
        conservative_buffers_.insert(buffer);
      }
      for (auto lhs : conservative_buffers_) {
        for (auto rhs : conservative_buffers_) {
          if (lhs != rhs) {
            (*conflict_map_)[lhs].insert(rhs);
          }
        }
      }
    } else {
      for (auto lhs : read_buffers_) {
        for (auto rhs : read_buffers_) {
          if (lhs != rhs) {
            (*conflict_map_)[lhs].insert(rhs);
          }
        }
      }
    }
    read_buffers_.clear();
    write_buffers_.clear();
    conservative_buffers_.clear();
  }

  void MarkUse(const BufferNode *buffer, BufferAccessKind access_kind) {
    auto it = live_ranges_->find(buffer);
    if (it == live_ranges_->end()) {
      return;
    }
    if (!inside_op_) {
      current_loc_ = NextLoc();
    } else {
      switch (access_kind) {
      case BufferAccessKind::kRead:
        read_buffers_.insert(buffer);
        break;
      case BufferAccessKind::kWrite:
        write_buffers_.insert(buffer);
        break;
      case BufferAccessKind::kReadWrite:
        read_buffers_.insert(buffer);
        write_buffers_.insert(buffer);
        break;
      case BufferAccessKind::kConservative:
        conservative_buffers_.insert(buffer);
        break;
      }
    }
    auto &range = it->second;
    if (!seen_buffers_.count(buffer)) {
      range.start = current_loc_;
      range.end = current_loc_ + 1;
      seen_buffers_.insert(buffer);
    } else {
      range.start = std::min<uint32_t>(range.start, current_loc_);
      range.end = std::max<uint32_t>(range.end, current_loc_ + 1);
    }
    auto allocation = allocation_loop_depth_.find(buffer);
    if (allocation != allocation_loop_depth_.end()) {
      for (size_t depth = allocation->second; depth < loops_.size(); ++depth) {
        loops_[depth].external_buffers_used.insert(buffer);
      }
    }
  }

  void MarkExprAs(const PrimExpr &expr, BufferAccessKind access_kind) {
    bool previous_collecting = collecting_operand_;
    BufferAccessKind previous_access = operand_access_kind_;
    collecting_operand_ = true;
    operand_access_kind_ = access_kind;
    VisitExpr(expr);
    collecting_operand_ = previous_collecting;
    operand_access_kind_ = previous_access;
  }

  bool VisitExternEffects(const CallNode *op) {
    if (!op->op.same_as(builtin::call_extern()) || op->args.empty()) {
      return false;
    }
    auto *op_name_node = op->args[0].as<StringImmNode>();
    if (!op_name_node) {
      return false;
    }
    std::string op_name = op_name_node->value;
    auto mark_arg = [&](size_t index, BufferAccessKind access_kind) {
      if (index < op->args.size()) {
        MarkExprAs(op->args[index], access_kind);
      }
    };

    OpScope scope(this);
    if (op_name == "tl.tpu.copy") {
      mark_arg(1, BufferAccessKind::kRead);
      mark_arg(2, BufferAccessKind::kWrite);
    } else if (op_name == "tl.tpu.fill") {
      mark_arg(1, BufferAccessKind::kWrite);
    } else if (op_name == "tl.tpu.gemm") {
      ICHECK_EQ(op->args.size(), 10U)
          << "tl.tpu.gemm requires the canonical 10-argument ABI, including "
             "an explicit accumulate flag";
      mark_arg(1, BufferAccessKind::kRead);
      mark_arg(2, BufferAccessKind::kRead);
      const auto *accumulate_value = op->args[9].as<IntImmNode>();
      ICHECK(accumulate_value && accumulate_value->dtype.is_bool())
          << "tl.tpu.gemm accumulate must be a compile-time boolean";
      bool accumulate = accumulate_value->value != 0;
      mark_arg(3, accumulate ? BufferAccessKind::kReadWrite
                             : BufferAccessKind::kWrite);
    } else if (op_name == "tl.tpu.sub" || op_name == "tl.tpu.mul" ||
               op_name == "tl.tpu.add" || op_name == "tl.tpu.div" ||
               op_name == "tl.tpu.max") {
      mark_arg(1, BufferAccessKind::kWrite);
      mark_arg(2, BufferAccessKind::kRead);
      mark_arg(3, BufferAccessKind::kRead);
    } else if (op_name == "tl.tpu.mul_scalar" ||
               op_name == "tl.tpu.add_scalar" || op_name == "tl.tpu.rsqrt") {
      mark_arg(1, BufferAccessKind::kWrite);
      mark_arg(2, BufferAccessKind::kRead);
    } else if (op_name == "tl.tpu.reduce_sum" ||
               op_name == "tl.tpu.reduce_max") {
      // The current pool-based lowering initializes the physically padded
      // tail of the local input tile before reducing it.
      mark_arg(1, BufferAccessKind::kReadWrite);
      mark_arg(2, BufferAccessKind::kWrite);
      mark_arg(3, BufferAccessKind::kReadWrite);
    } else if (op_name == "tl.tpu.exp") {
      // exp lowers to a multi-instruction composite.  Its output, both
      // workspaces, and coefficient tensor are read and written at different
      // points inside that opaque sequence, so model them as one conservative
      // bank-conflict clique rather than trusting the outer access mask.
      for (size_t i = 1; i <= 4; ++i) {
        mark_arg(i, BufferAccessKind::kConservative);
      }
    } else if (op_name == "tl.tpukernel.gather" ||
               op_name == "tl.tpu.embedding") {
      mark_arg(1, BufferAccessKind::kWrite);
      mark_arg(2, BufferAccessKind::kRead);
      mark_arg(3, BufferAccessKind::kRead);
    } else if (op_name == "tl.tpukernel.topk") {
      mark_arg(1, BufferAccessKind::kWrite);
      mark_arg(2, BufferAccessKind::kWrite);
      mark_arg(3, BufferAccessKind::kRead);
    } else {
      // Unknown/non-TPU externs have no instruction contract in this pass.
      // Keep their buffer operands conservative; the residual-IR verifier or
      // native TPU codegen boundary is responsible for rejecting them.
      for (size_t i = 1; i < op->args.size(); ++i) {
        mark_arg(i, BufferAccessKind::kConservative);
      }
    }
    return true;
  }

  void VisitExpr_(const VarNode *op) {
    auto it = buffer_var_to_buffer_.find(op);
    if (it != buffer_var_to_buffer_.end()) {
      MarkUse(it->second, collecting_operand_
                              ? operand_access_kind_
                              : BufferAccessKind::kConservative);
    }
  }

  void VisitExpr_(const CallNode *op) {
    if (collecting_operand_) {
      StmtExprVisitor::VisitExpr_(op);
      return;
    }
    if (VisitExternEffects(op)) {
      return;
    }
    OpScope scope(this);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) {
    OpScope scope(this);
    auto canonical = buffer_var_to_buffer_.find(op->buffer->data.get());
    if (canonical != buffer_var_to_buffer_.end()) {
      MarkUse(canonical->second, collecting_operand_ ? operand_access_kind_
                                                     : BufferAccessKind::kRead);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) {
    OpScope scope(this);
    auto canonical = buffer_var_to_buffer_.find(op->buffer->data.get());
    if (canonical != buffer_var_to_buffer_.end()) {
      MarkUse(canonical->second, collecting_operand_
                                     ? operand_access_kind_
                                     : BufferAccessKind::kWrite);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocateNode *op) final {
    auto buffer = buffer_var_to_buffer_.find(op->buffer_var.get());
    ICHECK(buffer != buffer_var_to_buffer_.end());
    allocation_loop_depth_[buffer->second] = loops_.size();
    StmtExprVisitor::VisitStmt_(op);
    allocation_loop_depth_.erase(buffer->second);
  }

  void BeginLoop() { loops_.push_back({NextLoc(), {}}); }

  void FinishLoop() {
    const uint32_t end = NextLoc();
    // A single traversal misses the backedge: an externally allocated tensor
    // read early in the body may be read again after later scratch writes.
    // Without a definite-write analysis, keep every used external allocation
    // live over the whole loop. Allocations inside this loop remain eligible
    // for sequential reuse, but are protected across any nested loop.
    for (const BufferNode *buffer : loops_.back().external_buffers_used) {
      auto &range = live_ranges_->at(buffer);
      range.start = std::min(range.start, loops_.back().start);
      range.end = std::max(range.end, end);
    }
    loops_.pop_back();
  }

  void VisitStmt_(const ForNode *op) final {
    VisitExpr(op->min);
    VisitExpr(op->extent);
    const auto *extent = op->extent.as<IntImmNode>();
    // Static zero/one-trip loops have no backedge. Symbolic extents and
    // static extents above one must conservatively preserve loop state.
    const bool may_repeat = !extent || extent->value > 1;
    if (may_repeat) {
      BeginLoop();
    }
    VisitStmt(op->body);
    if (may_repeat) {
      FinishLoop();
    }
  }

  void VisitStmt_(const WhileNode *op) final {
    BeginLoop();
    // The condition is re-evaluated on the backedge as well.
    VisitExpr(op->condition);
    VisitStmt(op->body);
    FinishLoop();
  }

  struct LoopLiveScope {
    uint32_t start;
    std::unordered_set<const BufferNode *> external_buffers_used;
  };

  std::unordered_map<const VarNode *, const BufferNode *> buffer_var_to_buffer_;
  std::unordered_map<const BufferNode *, size_t> allocation_loop_depth_;
  std::vector<LoopLiveScope> loops_;
  std::unordered_set<const BufferNode *> seen_buffers_;
  std::unordered_set<const BufferNode *> read_buffers_;
  std::unordered_set<const BufferNode *> write_buffers_;
  std::unordered_set<const BufferNode *> conservative_buffers_;
  std::unordered_map<const BufferNode *, TensorLive> *live_ranges_;
  std::unordered_map<const BufferNode *, std::unordered_set<const BufferNode *>>
      *conflict_map_;
  uint32_t loc_ = 0;
  uint32_t current_loc_ = 0;
  bool inside_op_ = false;
  bool collecting_operand_ = false;
  BufferAccessKind operand_access_kind_ = BufferAccessKind::kConservative;
};

PrimFunc InferAddress(PrimFunc f) {
  int bank_num = tpuv7::kBankNum;
  int bank_size = tpuv7::kBankSize;
  std::unordered_map<const BufferNode *, std::unordered_set<const BufferNode *>>
      bank_conflict_map;
  std::unordered_map<const BufferNode *, TensorLive> live_ranges;
  std::vector<const BufferNode *> alloc_ops =
      AddressAllocator().collectAllocOp(f->body);

  // LMEM addresses are carried through namespaced PrimFunc attributes until
  // codegen.  String keys cannot distinguish two Vars with the same name, and
  // two DeclBuffers over one data Var would make allocation size/ownership
  // ambiguous.  Reject both cases instead of silently overwriting metadata.
  std::unordered_map<std::string, const VarNode *> allocation_names;
  std::unordered_set<const VarNode *> allocation_data_vars;
  for (auto &op : alloc_ops) {
    const VarNode *data_var = op->data.get();
    const std::string storage_scope = GetRef<Buffer>(op).scope();
    ICHECK(tpuv7::IsLocalMemoryScope(storage_scope))
        << "AddressAssign only supports TPU local-memory DeclBuffers; buffer "
        << op->name << " has scope " << storage_scope;
    ICHECK(allocation_data_vars.insert(data_var).second)
        << "AddressAssign found multiple local DeclBuffers for allocation "
        << data_var->name_hint
        << "; descriptor aliases do not have a unique size/address contract";
    bool inserted =
        allocation_names.emplace(data_var->name_hint, data_var).second;
    ICHECK(inserted) << "AddressAssign requires unique local allocation "
                        "data-variable names; "
                     << "duplicate name " << data_var->name_hint
                     << " would alias string-keyed LMEM metadata";
    TensorLive live;
    live.tensor_size =
        storage_scope == "local.matrix"
            ? tpuv7::TpuMatrixSizeBytes(op->shape, op->dtype,
                                        "AddressAssign matrix")
            : tpuv7::TpuAlignSizeBytes(op->shape, op->dtype, "AddressAssign");
    live_ranges[op] = live;
  }
  BufferUseCollector(alloc_ops, &live_ranges, &bank_conflict_map)
      .Analyze(f->body);

  std::unordered_map<const BufferNode *, int64_t> addrMapWithBC;
  MemAllocBankConflictAware allocatorBC(bank_num, bank_size);
  auto success = allocatorBC.assignAddr(alloc_ops, live_ranges,
                                        bank_conflict_map, addrMapWithBC);
  ICHECK(success) << "TPUv7 local memory allocation failed. buffers="
                  << alloc_ops.size() << ", lmem=" << bank_num * bank_size
                  << " bytes";

  auto fn = f.CopyOnWrite();
  auto fn_attr = fn->attrs.CopyOnWrite();
  for (auto op : alloc_ops) {
    int64_t address = addrMapWithBC[op];
    fn_attr->dict.Set(tpuv7::AddressAttrKey(op->data->name_hint),
                      IntImm(DataType::Int(64), address));
  }

  return f;
}

tvm::transform::Pass AddressAssign() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    Optional<Target> target = f->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined())
        << "AddressAssign requires a PrimFunc bound to a TPU target";
    tpu::ResolveTarget(target.value(), "AddressAssign");
    return InferAddress(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AddressAssign", {});
}

TVM_REGISTER_GLOBAL("tl.transform.AddressAssign").set_body_typed(AddressAssign);

} // namespace tl
} // namespace tvm
