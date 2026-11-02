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
#include <sstream>
#include <string>
#include <tuple>
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

// ``greedy`` remains the compatibility default.  ``compact`` and ``beam``
// are fail-safe ablations (they retain a better greedy incumbent), while
// ``portfolio`` runs both bounded search orders and verifies every completed
// layout before selection.  ``agent`` is a compatibility spelling for the
// same deterministic portfolio; no model service is used at compile time.
constexpr const char *kAddressAssignPolicy = "tl.AddressAssignPolicy";
constexpr const char *kAddressAssignReplayAddresses =
    "tl.AddressAssignReplayAddresses";
constexpr const char *kAddressAssignExternalAllocator =
    "tl.AddressAssignExternalAllocator";
constexpr const char *kAddressAssignEmitProblem = "tl.AddressAssignEmitProblem";
constexpr const char *kAddressAssignAnalysisOnly = "tl.AddressAssignAnalysisOnly";
TVM_REGISTER_PASS_CONFIG_OPTION(kAddressAssignPolicy, String);
TVM_REGISTER_PASS_CONFIG_OPTION(kAddressAssignReplayAddresses, String);
TVM_REGISTER_PASS_CONFIG_OPTION(kAddressAssignExternalAllocator, String);
TVM_REGISTER_PASS_CONFIG_OPTION(kAddressAssignEmitProblem, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kAddressAssignAnalysisOnly, Bool);

namespace {

int64_t AlignUp(int64_t value, int64_t align) {
  return tpuv7::AlignUp(value, align);
}

int64_t SaturatingAdd(int64_t lhs, int64_t rhs) {
  ICHECK_GE(lhs, 0);
  ICHECK_GE(rhs, 0);
  if (lhs > std::numeric_limits<int64_t>::max() - rhs) {
    return std::numeric_limits<int64_t>::max();
  }
  return lhs + rhs;
}

int64_t SaturatingMultiply(int64_t lhs, int64_t rhs) {
  ICHECK_GE(lhs, 0);
  ICHECK_GE(rhs, 0);
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs) {
    return std::numeric_limits<int64_t>::max();
  }
  return lhs * rhs;
}

std::string GetPointerStorageScope(const Var &var) {
  const auto *pointer_type = var->type_annotation.as<PointerTypeNode>();
  ICHECK(pointer_type)
      << "AddressAssign requires a pointer-typed allocation Var, got " << var
      << " with type " << var->type_annotation;
  return pointer_type->storage_scope;
}

std::vector<int64_t> ParseReplayAddresses(const std::string &encoded,
                                          size_t expected_count) {
  ICHECK(!encoded.empty())
      << "AddressAssign replay requires a comma-separated address list";
  std::stringstream stream(encoded);
  std::string token;
  std::vector<int64_t> addresses;
  while (std::getline(stream, token, ',')) {
    ICHECK(!token.empty()) << "AddressAssign replay contains an empty address";
    size_t parsed = 0;
    int64_t value = 0;
    try {
      value = std::stoll(token, &parsed);
    } catch (const std::exception &error) {
      LOG(FATAL) << "AddressAssign replay cannot parse address '" << token
                 << "': " << error.what();
    }
    ICHECK_EQ(parsed, token.size())
        << "AddressAssign replay address has trailing text: " << token;
    addresses.push_back(value);
  }
  ICHECK_EQ(addresses.size(), expected_count)
      << "AddressAssign replay expected " << expected_count << " addresses, got "
      << addresses.size();
  return addresses;
}

} // namespace

struct AllocationStatistics {
  int64_t bank_conflicts = 0;
  int64_t weighted_bank_conflicts = 0;
  int64_t peak_bytes = 0;
  int64_t explored_states = 0;
  int64_t live_bytes_lower_bound = 0;
  int64_t peak_slack_bytes = 0;
};

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

using ConflictMap =
    std::unordered_map<const BufferNode *,
                       std::unordered_map<const BufferNode *, int64_t>>;

struct ConflictCost {
  int64_t unweighted = 0;
  int64_t weighted = 0;
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

int64_t ComputeLiveBytesLowerBound(
    const std::vector<const BufferNode *> &ops,
    const std::unordered_map<const BufferNode *, TensorLive> &live_range) {
  std::vector<uint32_t> cuts;
  cuts.reserve(ops.size() * 2);
  for (const BufferNode *op : ops) {
    const TensorLive &live = live_range.at(op);
    if (live.start < live.end) {
      cuts.push_back(live.start);
      cuts.push_back(live.end - 1);
    }
  }
  int64_t lower_bound = 0;
  for (uint32_t cut : cuts) {
    int64_t live_bytes = 0;
    for (const BufferNode *op : ops) {
      const TensorLive &live = live_range.at(op);
      if (live.start <= cut && cut < live.end) {
        live_bytes = SaturatingAdd(live_bytes, live.tensor_size);
      }
    }
    lower_bound = std::max(lower_bound, live_bytes);
  }
  return lower_bound;
}

AllocationStatistics EvaluateAllocation(
    const std::vector<const BufferNode *> &ops,
    const std::unordered_map<const BufferNode *, TensorLive> &live_range,
    const ConflictMap &conflict_map,
    const std::unordered_map<const BufferNode *, int64_t> &addresses,
    int64_t bank_size) {
  AllocationStatistics statistics;
  statistics.live_bytes_lower_bound =
      ComputeLiveBytesLowerBound(ops, live_range);
  for (const BufferNode *op : ops) {
    statistics.peak_bytes =
        std::max(statistics.peak_bytes,
                 addresses.at(op) + live_range.at(op).tensor_size);
  }
  for (size_t lhs_index = 0; lhs_index < ops.size(); ++lhs_index) {
    const BufferNode *lhs = ops[lhs_index];
    for (size_t rhs_index = lhs_index + 1; rhs_index < ops.size(); ++rhs_index) {
      const BufferNode *rhs = ops[rhs_index];
      auto edge = conflict_map.at(lhs).find(rhs);
      if (edge == conflict_map.at(lhs).end() ||
          !LiveRangesOverlap(live_range.at(lhs), live_range.at(rhs))) {
        continue;
      }
      const int64_t lhs_first = addresses.at(lhs) / bank_size;
      const int64_t lhs_last =
          (addresses.at(lhs) + live_range.at(lhs).tensor_size - 1) /
          bank_size;
      const int64_t rhs_first = addresses.at(rhs) / bank_size;
      const int64_t rhs_last =
          (addresses.at(rhs) + live_range.at(rhs).tensor_size - 1) /
          bank_size;
      const int64_t shared =
          std::max<int64_t>(0, std::min(lhs_last, rhs_last) -
                                   std::max(lhs_first, rhs_first) + 1);
      statistics.bank_conflicts =
          SaturatingAdd(statistics.bank_conflicts, shared);
      statistics.weighted_bank_conflicts = SaturatingAdd(
          statistics.weighted_bank_conflicts,
          SaturatingMultiply(shared, edge->second));
    }
  }
  // This is a capacity lower-bound gap, not pure fragmentation: deliberate
  // bank separation can also contribute to it.
  statistics.peak_slack_bytes =
      std::max<int64_t>(0, statistics.peak_bytes -
                              statistics.live_bytes_lower_bound);
  return statistics;
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
      ConflictMap &conflictMap,
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
      int64_t selected_weighted_conflicts = 0;
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
        ++statistics_.explored_states;
        int64_t end_offset =
            std::min(offset + (mem_cross_bank_num + 1) * bank_size_, mem_size_);
        auto op_addr = searchAddr(op, liveRange, offset, end_offset);

        // op can insert
        if (op_addr->start + op_addr->size <= end_offset) {
          ConflictCost conflict_cost =
              getConflictCost(op_addr, liveRange, conflictMap);
          int64_t conf_count = conflict_cost.unweighted;
          bool better_addr = !best_addr || conf_count < min_conflict_count ||
                             (conf_count == min_conflict_count &&
                              (op_addr->end < best_addr->end ||
                               (op_addr->end == best_addr->end &&
                                op_addr->start < best_addr->start)));
          if (better_addr) {
            min_conflict_count = conf_count;
            selected_weighted_conflicts = conflict_cost.weighted;
            best_addr = op_addr;
          }
        }
      }
      if (!best_addr) {
        return false;
      }
      statistics_.bank_conflicts += min_conflict_count;
      statistics_.weighted_bank_conflicts += selected_weighted_conflicts;
      statistics_.peak_bytes =
          std::max(statistics_.peak_bytes, best_addr->end);
      insertAddr(best_addr);
      // addrMap.Set(op, best_addr->start);
      addrMap[op] = best_addr->start;
    }
    return true;
  }

  const AllocationStatistics &statistics() const { return statistics_; }

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

  ConflictCost getConflictCost(
      std::shared_ptr<OpAddr> &opAddr,
      std::unordered_map<const BufferNode *, TensorLive> &liveRange,
      ConflictMap &conflictMap) {
    int64_t bank_start = opAddr->start / bank_size_;
    int64_t bank_end = (opAddr->end - 1) / bank_size_;
    ConflictCost cost;
    for (int i = bank_start; i <= bank_end; i++) {
      for (auto op : bank_ops[i]) {
        auto conflict = conflictMap[opAddr->op].find(op);
        if (conflict != conflictMap[opAddr->op].end() &&
            LiveRangesOverlap(liveRange[opAddr->op], liveRange[op])) {
          ++cost.unweighted;
          cost.weighted = SaturatingAdd(cost.weighted, conflict->second);
        }
      }
    }
    return cost;
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
  AllocationStatistics statistics_;
};

// A low-cost left-edge allocator used as a capacity-recovery candidate.  It
// deliberately ignores the bank objective while constructing the layout: the
// verified portfolio compares its completed objective against the incumbent
// and may reject it.  Keeping this branch objective-independent is useful when
// a bank-aware greedy path fragments otherwise sufficient LMEM.
class MemAllocCompact {
public:
  MemAllocCompact(int64_t bank_num, int64_t bank_size)
      : bank_size_(bank_size), mem_size_(bank_num * bank_size) {}

  bool assignAddr(
      const std::vector<const BufferNode *> &ops,
      const std::unordered_map<const BufferNode *, TensorLive> &live_range,
      const ConflictMap &conflict_map,
      std::unordered_map<const BufferNode *, int64_t> &address_map) {
    std::vector<const BufferNode *> ordered = ops;
    std::stable_sort(
        ordered.begin(), ordered.end(),
        [&live_range, &conflict_map](const BufferNode *lhs,
                                     const BufferNode *rhs) {
          const TensorLive &lhs_live = live_range.at(lhs);
          const TensorLive &rhs_live = live_range.at(rhs);
          if (lhs_live.tensor_size != rhs_live.tensor_size) {
            return lhs_live.tensor_size > rhs_live.tensor_size;
          }
          if (conflict_map.at(lhs).size() != conflict_map.at(rhs).size()) {
            return conflict_map.at(lhs).size() > conflict_map.at(rhs).size();
          }
          return lhs_live.start < rhs_live.start;
        });

    std::vector<OpAddr> allocated;
    int64_t partial_peak = 0;
    for (const BufferNode *op : ordered) {
      const TensorLive &live = live_range.at(op);
      if (live.tensor_size > mem_size_) {
        return false;
      }
      std::vector<int64_t> starts{0};
      for (const OpAddr &other : allocated) {
        if (LiveRangesOverlap(live, live_range.at(other.op))) {
          starts.push_back(
              AlignUp(other.end, tpuv7::kTensorAlignBytes));
        }
      }
      std::sort(starts.begin(), starts.end());
      starts.erase(std::unique(starts.begin(), starts.end()), starts.end());

      bool found = false;
      int64_t selected_start = 0;
      int64_t selected_peak = std::numeric_limits<int64_t>::max();
      for (int64_t start : starts) {
        ++statistics_.explored_states;
        const int64_t end = start + live.tensor_size;
        if (start < 0 || end > mem_size_) {
          continue;
        }
        bool overlaps = false;
        for (const OpAddr &other : allocated) {
          if (LiveRangesOverlap(live, live_range.at(other.op)) &&
              std::max(start, other.start) < std::min(end, other.end)) {
            overlaps = true;
            break;
          }
        }
        if (overlaps) {
          continue;
        }
        const int64_t candidate_peak = std::max(partial_peak, end);
        if (!found || std::tie(candidate_peak, start) <
                          std::tie(selected_peak, selected_start)) {
          found = true;
          selected_start = start;
          selected_peak = candidate_peak;
        }
      }
      if (!found) {
        return false;
      }
      OpAddr placement(op, live.tensor_size, live.start, live.end);
      placement.start = selected_start;
      placement.end = selected_start + live.tensor_size;
      auto position = std::lower_bound(
          allocated.begin(), allocated.end(), placement.start,
          [](const OpAddr &other, int64_t start) {
            return other.start < start;
          });
      allocated.insert(position, placement);
      address_map[op] = selected_start;
      partial_peak = selected_peak;
    }

    const int64_t explored_states = statistics_.explored_states;
    statistics_ = EvaluateAllocation(ops, live_range, conflict_map,
                                     address_map, bank_size_);
    statistics_.explored_states = explored_states;
    return true;
  }

  const AllocationStatistics &statistics() const { return statistics_; }

private:
  int64_t bank_size_;
  int64_t mem_size_;
  AllocationStatistics statistics_;
};

enum class BeamOrder {
  kSize,
  kWeightedPressure,
};

struct BeamAllocationState {
  std::vector<OpAddr> allocated;
  std::unordered_map<const BufferNode *, int64_t> addresses;
  std::vector<int64_t> placement_trace;
  int64_t bank_conflicts = 0;
  int64_t weighted_bank_conflicts = 0;
  int64_t peak_bytes = 0;
};

class MemAllocBeamSearch {
public:
  MemAllocBeamSearch(int64_t bank_num, int64_t bank_size, int64_t beam_width,
                     int64_t candidate_limit, BeamOrder order)
      : bank_num_(bank_num), bank_size_(bank_size),
        mem_size_(bank_num * bank_size), beam_width_(beam_width),
        candidate_limit_(candidate_limit), order_(order) {
    ICHECK_GT(beam_width_, 0);
    ICHECK_GT(candidate_limit_, 0);
  }

  bool assignAddr(
      std::vector<const BufferNode *> &ops,
      std::unordered_map<const BufferNode *, TensorLive> &live_range,
      ConflictMap &conflict_map,
      std::unordered_map<const BufferNode *, int64_t> &address_map) {
    std::vector<const BufferNode *> ordered = ops;
    std::stable_sort(ordered.begin(), ordered.end(),
                     [this, &live_range,
                      &conflict_map](const BufferNode *lhs,
                                     const BufferNode *rhs) {
                       const auto &lhs_live = live_range.at(lhs);
                       const auto &rhs_live = live_range.at(rhs);
                       if (order_ == BeamOrder::kWeightedPressure) {
                         const int64_t lhs_pressure =
                             WeightedPressure(lhs, live_range, conflict_map);
                         const int64_t rhs_pressure =
                             WeightedPressure(rhs, live_range, conflict_map);
                         if (lhs_pressure != rhs_pressure) {
                           return lhs_pressure > rhs_pressure;
                         }
                       }
                       if (lhs_live.tensor_size != rhs_live.tensor_size) {
                         return lhs_live.tensor_size > rhs_live.tensor_size;
                       }
                       if (conflict_map.at(lhs).size() !=
                           conflict_map.at(rhs).size()) {
                         return conflict_map.at(lhs).size() >
                                conflict_map.at(rhs).size();
                       }
                       return lhs_live.start < rhs_live.start;
                     });

    std::vector<BeamAllocationState> beam(1);
    for (const BufferNode *op : ordered) {
      if (live_range.at(op).tensor_size > mem_size_) {
        return false;
      }
      std::vector<BeamAllocationState> expanded;
      for (const auto &state : beam) {
        struct Candidate {
          OpAddr address;
          int64_t conflicts;
          int64_t weighted_conflicts;
        };
        std::vector<Candidate> candidates;
        const int64_t bytes = live_range.at(op).tensor_size;
        std::vector<int64_t> starts;
        for (int64_t bank = 0; bank < bank_num_; ++bank) {
          starts.push_back(bank * bank_size_);
        }
        for (const OpAddr &allocated : state.allocated) {
          if (LiveRangesOverlap(live_range.at(op),
                                live_range.at(allocated.op))) {
            starts.push_back(
                AlignUp(allocated.end, tpuv7::kTensorAlignBytes));
          }
        }
        std::sort(starts.begin(), starts.end());
        starts.erase(std::unique(starts.begin(), starts.end()), starts.end());
        for (int64_t start : starts) {
          ++statistics_.explored_states;
          if (start < 0 || start + bytes > mem_size_) {
            continue;
          }
          const int64_t end = start + bytes;
          bool overlaps = false;
          for (const OpAddr &allocated : state.allocated) {
            if (LiveRangesOverlap(live_range.at(op),
                                  live_range.at(allocated.op)) &&
                std::max(start, allocated.start) <
                    std::min(end, allocated.end)) {
              overlaps = true;
              break;
            }
          }
          if (overlaps) {
            continue;
          }
          OpAddr candidate(op, bytes, live_range.at(op).start,
                           live_range.at(op).end);
          candidate.start = start;
          candidate.end = end;
          const ConflictCost cost = IncrementalConflict(
              candidate, state, live_range, conflict_map);
          candidates.push_back(
              {candidate, cost.unweighted, cost.weighted});
        }
        std::sort(candidates.begin(), candidates.end(),
                  [this](const Candidate &lhs, const Candidate &rhs) {
                    if (order_ == BeamOrder::kWeightedPressure) {
                      // Static frequency is an uncalibrated proxy.  Use it to
                      // break ties only after the vendor-compatible ordinary
                      // conflict count, so a hotness estimate cannot create a
                      // new structural conflict.
                      return std::tie(lhs.conflicts, lhs.weighted_conflicts,
                                      lhs.address.end, lhs.address.start) <
                             std::tie(rhs.conflicts, rhs.weighted_conflicts,
                                      rhs.address.end, rhs.address.start);
                    }
                    return std::tie(lhs.conflicts, lhs.address.end,
                                    lhs.address.start) <
                           std::tie(rhs.conflicts, rhs.address.end,
                                    rhs.address.start);
                  });
        if (candidates.size() > static_cast<size_t>(candidate_limit_)) {
          candidates.erase(candidates.begin() + candidate_limit_,
                           candidates.end());
        }
        for (const Candidate &candidate : candidates) {
          BeamAllocationState next = state;
          next.addresses[op] = candidate.address.start;
          next.placement_trace.push_back(candidate.address.start);
          next.bank_conflicts += candidate.conflicts;
          next.weighted_bank_conflicts = SaturatingAdd(
              next.weighted_bank_conflicts, candidate.weighted_conflicts);
          next.peak_bytes = std::max(next.peak_bytes, candidate.address.end);
          auto position = std::lower_bound(
              next.allocated.begin(), next.allocated.end(),
              candidate.address.start,
              [](const OpAddr &allocated, int64_t start) {
                return allocated.start < start;
              });
          next.allocated.insert(position, candidate.address);
          expanded.push_back(std::move(next));
        }
      }
      if (expanded.empty()) {
        return false;
      }
      std::sort(expanded.begin(), expanded.end(),
                [this](const BeamAllocationState &lhs,
                       const BeamAllocationState &rhs) {
                  return StateLess(lhs, rhs);
                });
      expanded.erase(
          std::unique(expanded.begin(), expanded.end(),
                      [](const BeamAllocationState &lhs,
                         const BeamAllocationState &rhs) {
                        return lhs.placement_trace == rhs.placement_trace;
                      }),
          expanded.end());
      if (expanded.size() > static_cast<size_t>(beam_width_)) {
        const size_t objective_quota =
            std::max<size_t>(1, static_cast<size_t>(beam_width_) / 2);
        std::vector<BeamAllocationState> selected(
            expanded.begin(), expanded.begin() + objective_quota);
        std::vector<BeamAllocationState> compact = expanded;
        std::sort(compact.begin(), compact.end(),
                  [this](const BeamAllocationState &lhs,
                         const BeamAllocationState &rhs) {
                    return CompactStateLess(lhs, rhs);
                  });
        for (const auto &candidate : compact) {
          if (selected.size() >= static_cast<size_t>(beam_width_)) {
            break;
          }
          const bool duplicate = std::any_of(
              selected.begin(), selected.end(),
              [&candidate](const BeamAllocationState &kept) {
                return kept.placement_trace == candidate.placement_trace;
              });
          if (!duplicate) {
            selected.push_back(candidate);
          }
          if (selected.size() >= static_cast<size_t>(beam_width_)) {
            break;
          }
        }
        std::sort(selected.begin(), selected.end(),
                  [this](const BeamAllocationState &lhs,
                         const BeamAllocationState &rhs) {
                    return StateLess(lhs, rhs);
                  });
        expanded = std::move(selected);
      }
      beam = std::move(expanded);
    }
    const BeamAllocationState &best = beam.front();
    address_map = best.addresses;
    statistics_.bank_conflicts = best.bank_conflicts;
    statistics_.weighted_bank_conflicts = best.weighted_bank_conflicts;
    statistics_.peak_bytes = best.peak_bytes;
    return true;
  }

  const AllocationStatistics &statistics() const { return statistics_; }

private:
  bool StateLess(const BeamAllocationState &lhs,
                 const BeamAllocationState &rhs) const {
    if (order_ == BeamOrder::kWeightedPressure) {
      return std::tie(lhs.bank_conflicts, lhs.weighted_bank_conflicts,
                      lhs.peak_bytes, lhs.placement_trace) <
             std::tie(rhs.bank_conflicts, rhs.weighted_bank_conflicts,
                      rhs.peak_bytes, rhs.placement_trace);
    }
    return std::tie(lhs.bank_conflicts, lhs.peak_bytes,
                    lhs.placement_trace) <
           std::tie(rhs.bank_conflicts, rhs.peak_bytes,
                    rhs.placement_trace);
  }

  bool CompactStateLess(const BeamAllocationState &lhs,
                        const BeamAllocationState &rhs) const {
    if (order_ == BeamOrder::kWeightedPressure) {
      return std::tie(lhs.peak_bytes, lhs.bank_conflicts,
                      lhs.weighted_bank_conflicts, lhs.placement_trace) <
             std::tie(rhs.peak_bytes, rhs.bank_conflicts,
                      rhs.weighted_bank_conflicts, rhs.placement_trace);
    }
    return std::tie(lhs.peak_bytes, lhs.bank_conflicts,
                    lhs.placement_trace) <
           std::tie(rhs.peak_bytes, rhs.bank_conflicts,
                    rhs.placement_trace);
  }

  int64_t WeightedPressure(
      const BufferNode *op,
      const std::unordered_map<const BufferNode *, TensorLive> &live_range,
      const ConflictMap &conflict_map) const {
    int64_t edge_weight = 0;
    for (const auto &edge : conflict_map.at(op)) {
      edge_weight = SaturatingAdd(edge_weight, edge.second);
    }
    const int64_t banks = std::max<int64_t>(
        1, tpuv7::DivUp(live_range.at(op).tensor_size, bank_size_));
    const int64_t lifetime = std::max<int64_t>(
        1, static_cast<int64_t>(live_range.at(op).end) -
               static_cast<int64_t>(live_range.at(op).start));
    return SaturatingMultiply(SaturatingMultiply(edge_weight, banks),
                              lifetime);
  }

  ConflictCost IncrementalConflict(
      const OpAddr &candidate, const BeamAllocationState &state,
      const std::unordered_map<const BufferNode *, TensorLive> &live_range,
      const ConflictMap &conflict_map) const {
    const int64_t candidate_first_bank = candidate.start / bank_size_;
    const int64_t candidate_last_bank = (candidate.end - 1) / bank_size_;
    ConflictCost cost;
    for (const OpAddr &allocated : state.allocated) {
      auto conflict = conflict_map.at(candidate.op).find(allocated.op);
      if (conflict == conflict_map.at(candidate.op).end() ||
          !LiveRangesOverlap(live_range.at(candidate.op),
                             live_range.at(allocated.op))) {
        continue;
      }
      const int64_t first =
          std::max(candidate_first_bank, allocated.start / bank_size_);
      const int64_t last =
          std::min(candidate_last_bank, (allocated.end - 1) / bank_size_);
      if (first <= last) {
        const int64_t shared_banks = last - first + 1;
        cost.unweighted += shared_banks;
        cost.weighted = SaturatingAdd(
            cost.weighted,
            SaturatingMultiply(shared_banks, conflict->second));
      }
    }
    return cost;
  }

  int64_t bank_num_;
  int64_t bank_size_;
  int64_t mem_size_;
  int64_t beam_width_;
  int64_t candidate_limit_;
  BeamOrder order_;
  AllocationStatistics statistics_;
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
      ConflictMap *conflict_map)
      : live_ranges_(live_ranges), conflict_map_(conflict_map) {
    for (const BufferNode *buffer : alloc_ops) {
      buffer_var_to_buffer_[buffer->data.get()] = buffer;
      (*conflict_map_)[buffer];
    }
  }

  void Analyze(const Stmt &body) {
    VisitStmt(body);
    ICHECK(!parallel_active_)
        << "TPU-Kernel parallel command group was not closed";
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
            int64_t &weight = (*conflict_map_)[lhs][rhs];
            weight = SaturatingAdd(weight, execution_weight_);
          }
        }
      }
    } else {
      for (auto lhs : read_buffers_) {
        for (auto rhs : read_buffers_) {
          if (lhs != rhs) {
            int64_t &weight = (*conflict_map_)[lhs][rhs];
            weight = SaturatingAdd(weight, execution_weight_);
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
    if (parallel_active_) {
      parallel_buffers_.insert(buffer);
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
    } else if (op_name == "tl.tpu.dq2") {
      mark_arg(1, BufferAccessKind::kWrite);
      mark_arg(2, BufferAccessKind::kRead);
      mark_arg(3, BufferAccessKind::kRead);
      mark_arg(5, BufferAccessKind::kReadWrite);
    } else if (op_name == "tl.tpu.scatter") {
      mark_arg(1, BufferAccessKind::kWrite);
      mark_arg(2, BufferAccessKind::kRead);
      mark_arg(3, BufferAccessKind::kRead);
    } else if (op_name == "tl.tpukernel.scatter_add") {
      // This opaque composite repeatedly loads, adds, and stores rows. Its
      // two internal local tiles are both read and written by that sequence.
      for (size_t i = 1; i <= 5; ++i) {
        mark_arg(i, BufferAccessKind::kConservative);
      }
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
    } else if (op_name == "tl.tpukernel.parallel_start" ||
               op_name == "tl.tpukernel.parallel_end") {
      ICHECK_EQ(op->args.size(), 1U)
          << op_name << " takes no buffer operands";
      if (op_name == "tl.tpukernel.parallel_start") {
        ICHECK(!parallel_active_)
            << "nested TPU-Kernel parallel command groups are unsupported";
        parallel_active_ = true;
        parallel_start_loc_ = current_loc_;
        parallel_buffers_.clear();
      } else {
        ICHECK(parallel_active_)
            << "TPU-Kernel parallel_end has no matching start";
        // The commands in one parallel group may execute concurrently even
        // when their calls are sequential in TIR.  Extend every participating
        // buffer's life to the complete command group before placing it.
        for (const BufferNode *buffer : parallel_buffers_) {
          auto &range = live_ranges_->at(buffer);
          range.start = std::min(range.start, parallel_start_loc_);
          range.end = std::max(range.end, current_loc_ + 1);
        }
        parallel_buffers_.clear();
        parallel_active_ = false;
      }
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
    const int64_t previous_execution_weight = execution_weight_;
    if (extent) {
      execution_weight_ = SaturatingMultiply(
          execution_weight_, std::max<int64_t>(extent->value, 0));
    }
    VisitStmt(op->body);
    execution_weight_ = previous_execution_weight;
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
  ConflictMap *conflict_map_;
  int64_t execution_weight_ = 1;
  uint32_t loc_ = 0;
  uint32_t current_loc_ = 0;
  bool inside_op_ = false;
  bool collecting_operand_ = false;
  bool parallel_active_ = false;
  uint32_t parallel_start_loc_ = 0;
  std::unordered_set<const BufferNode *> parallel_buffers_;
  BufferAccessKind operand_access_kind_ = BufferAccessKind::kConservative;
};

PrimFunc AttachProblemMetadata(
    PrimFunc f, const std::vector<const BufferNode *> &alloc_ops,
    const std::unordered_map<const BufferNode *, TensorLive> &live_ranges,
    const ConflictMap &bank_conflict_map) {
  auto fn = f.CopyOnWrite();
  auto fn_attr = fn->attrs.CopyOnWrite();
  const std::string prefix = "tilelang.tpu.lmem.problem.";
  fn_attr->dict.Set(prefix + "capacity",
                    IntImm(DataType::Int(64), tpuv7::kBankNum * tpuv7::kBankSize));
  fn_attr->dict.Set(prefix + "alignment",
                    IntImm(DataType::Int(64), tpuv7::kTensorAlignBytes));
  fn_attr->dict.Set(prefix + "bank_count",
                    IntImm(DataType::Int(64), tpuv7::kBankNum));
  fn_attr->dict.Set(prefix + "bank_size",
                    IntImm(DataType::Int(64), tpuv7::kBankSize));
  fn_attr->dict.Set(prefix + "buffer_count",
                    IntImm(DataType::Int(64), alloc_ops.size()));
  std::unordered_map<const BufferNode *, size_t> buffer_indices;
  for (size_t index = 0; index < alloc_ops.size(); ++index) {
    const BufferNode *op = alloc_ops[index];
    buffer_indices[op] = index;
    const std::string key = prefix + "buffer." + std::to_string(index) + ".";
    fn_attr->dict.Set(key + "name", String(op->data->name_hint));
    fn_attr->dict.Set(key + "size",
                      IntImm(DataType::Int(64), live_ranges.at(op).tensor_size));
    fn_attr->dict.Set(key + "start",
                      IntImm(DataType::Int(64), live_ranges.at(op).start));
    fn_attr->dict.Set(key + "end",
                      IntImm(DataType::Int(64), live_ranges.at(op).end));
  }
  size_t edge_index = 0;
  for (size_t lhs_index = 0; lhs_index < alloc_ops.size(); ++lhs_index) {
    const BufferNode *lhs = alloc_ops[lhs_index];
    for (size_t rhs_index = lhs_index + 1; rhs_index < alloc_ops.size();
         ++rhs_index) {
      const BufferNode *rhs = alloc_ops[rhs_index];
      auto edge = bank_conflict_map.at(lhs).find(rhs);
      if (edge == bank_conflict_map.at(lhs).end()) {
        continue;
      }
      const std::string key =
          prefix + "edge." + std::to_string(edge_index) + ".";
      fn_attr->dict.Set(key + "lhs",
                        IntImm(DataType::Int(64), buffer_indices.at(lhs)));
      fn_attr->dict.Set(key + "rhs",
                        IntImm(DataType::Int(64), buffer_indices.at(rhs)));
      fn_attr->dict.Set(key + "weight", IntImm(DataType::Int(64), edge->second));
      ++edge_index;
    }
  }
  fn_attr->dict.Set(prefix + "edge_count",
                    IntImm(DataType::Int(64), edge_index));
  return f;
}

PrimFunc InferAddress(PrimFunc f, const std::string &policy,
                      const std::string &replay_addresses,
                      const std::string &external_allocator, bool emit_problem,
                      bool analysis_only) {
  int bank_num = tpuv7::kBankNum;
  int bank_size = tpuv7::kBankSize;
  ConflictMap bank_conflict_map;
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

  ICHECK(!analysis_only || emit_problem)
      << "AddressAssign analysis-only mode requires problem export";
  if (emit_problem || policy == "external") {
    f = AttachProblemMetadata(f, alloc_ops, live_ranges, bank_conflict_map);
  }
  if (analysis_only) {
    return f;
  }

  ICHECK(policy == "greedy" || policy == "compact" || policy == "beam" ||
         policy == "portfolio" || policy == "agent" || policy == "replay" ||
         policy == "external")
      << "AddressAssign policy must be 'greedy', 'compact', 'beam', "
         "'portfolio', 'replay', 'external', or the compatibility alias 'agent', got "
      << policy;

  std::unordered_map<const BufferNode *, int64_t> greedy_addresses;
  MemAllocBankConflictAware greedy_allocator(bank_num, bank_size);
  const bool greedy_success = policy != "external" && greedy_allocator.assignAddr(
      alloc_ops, live_ranges, bank_conflict_map, greedy_addresses);

  std::unordered_map<const BufferNode *, int64_t> selected_addresses;
  AllocationStatistics selected_statistics;
  std::string selected_policy = "greedy";
  bool success = greedy_success;
  int64_t total_explored_states = greedy_allocator.statistics().explored_states;
  if (greedy_success) {
    selected_addresses = greedy_addresses;
    selected_statistics = EvaluateAllocation(
        alloc_ops, live_ranges, bank_conflict_map, greedy_addresses, bank_size);
    selected_statistics.explored_states =
        greedy_allocator.statistics().explored_states;
  }

  auto consider_candidate =
      [&](bool candidate_success,
          std::unordered_map<const BufferNode *, int64_t> candidate_addresses,
          AllocationStatistics candidate_statistics,
          const std::string &candidate_policy) {
        total_explored_states = SaturatingAdd(
            total_explored_states, candidate_statistics.explored_states);
        if (!candidate_success) {
          return;
        }
        const int64_t explored_states = candidate_statistics.explored_states;
        candidate_statistics = EvaluateAllocation(
            alloc_ops, live_ranges, bank_conflict_map, candidate_addresses,
            bank_size);
        candidate_statistics.explored_states = explored_states;
        const bool improves =
            !success ||
            std::tie(candidate_statistics.bank_conflicts,
                     candidate_statistics.weighted_bank_conflicts,
                     candidate_statistics.peak_bytes) <
                std::tie(selected_statistics.bank_conflicts,
                         selected_statistics.weighted_bank_conflicts,
                         selected_statistics.peak_bytes);
        if (improves) {
          success = true;
          selected_addresses = std::move(candidate_addresses);
          selected_statistics = candidate_statistics;
          selected_policy = candidate_policy;
        }
      };

  constexpr int64_t kBaselineBeamWidth = 32;
  constexpr int64_t kBaselineCandidateLimit = 32;
  constexpr int64_t kSizeBeamWidth = 32;
  constexpr int64_t kSizeCandidateLimit = 32;
  constexpr int64_t kPressureBeamWidth = 32;
  constexpr int64_t kPressureCandidateLimit = 16;
  if (policy == "replay" || policy == "external") {
    std::string chosen_replay = replay_addresses;
    if (policy == "external") {
      ICHECK(!external_allocator.empty())
          << "AddressAssign external policy requires a registered allocator name";
      const auto *callback = runtime::Registry::Get(external_allocator);
      ICHECK(callback != nullptr)
          << "AddressAssign external allocator is not registered: "
          << external_allocator;
      String callback_result = (*callback)(f);
      chosen_replay = callback_result;
    }
    const std::vector<int64_t> replay =
        ParseReplayAddresses(chosen_replay, alloc_ops.size());
    selected_addresses.clear();
    for (size_t index = 0; index < alloc_ops.size(); ++index) {
      const BufferNode *op = alloc_ops[index];
      const int64_t address = replay[index];
      const int64_t bytes = live_ranges.at(op).tensor_size;
      ICHECK_GE(address, 0)
          << "AddressAssign replay address must be non-negative for "
          << op->name;
      ICHECK_EQ(address % tpuv7::kTensorAlignBytes, 0)
          << "AddressAssign replay address must be "
          << tpuv7::kTensorAlignBytes << "-byte aligned for " << op->name;
      ICHECK_LE(bytes, bank_num * bank_size)
          << "AddressAssign replay buffer exceeds LMEM capacity: "
          << op->name;
      // Subtract before comparing so an untrusted near-INT64_MAX address
      // cannot overflow in ``address + bytes`` inside objective evaluation.
      ICHECK_LE(address, bank_num * bank_size - bytes)
          << "AddressAssign replay address exceeds LMEM capacity for "
          << op->name;
      selected_addresses[op] = address;
    }
    selected_statistics = EvaluateAllocation(
        alloc_ops, live_ranges, bank_conflict_map, selected_addresses,
        bank_size);
    total_explored_states = 0;
    success = true;
    selected_policy = policy == "external" ? "external:" + external_allocator
                                           : "replay";
  } else if (policy == "compact") {
    selected_policy = "compact-incumbent";
    std::unordered_map<const BufferNode *, int64_t> compact_addresses;
    MemAllocCompact compact_allocator(bank_num, bank_size);
    const bool compact_success = compact_allocator.assignAddr(
        alloc_ops, live_ranges, bank_conflict_map, compact_addresses);
    consider_candidate(compact_success, std::move(compact_addresses),
                       compact_allocator.statistics(), "compact-search");
  } else if (policy == "beam") {
    selected_policy = "beam-incumbent";
    std::unordered_map<const BufferNode *, int64_t> searched_addresses;
    MemAllocBeamSearch searched_allocator(
        bank_num, bank_size, kBaselineBeamWidth, kBaselineCandidateLimit,
        BeamOrder::kSize);
    const bool searched_success = searched_allocator.assignAddr(
        alloc_ops, live_ranges, bank_conflict_map, searched_addresses);
    consider_candidate(searched_success, std::move(searched_addresses),
                       searched_allocator.statistics(), "beam-search");
  } else if (policy == "portfolio" || policy == "agent") {
    const std::string prefix = policy == "agent" ? "agent" : "portfolio";
    selected_policy = prefix + "-incumbent";

    // Evaluate all bounded branches.  A previous diagnostic router skipped
    // apparently easy programs, but independent oracle evaluation showed that
    // it missed safe peak improvements.  Full enumeration retains the greedy
    // fail-safe while making the selected result dominate every constituent.
    std::unordered_map<const BufferNode *, int64_t> compact_addresses;
    MemAllocCompact compact_allocator(bank_num, bank_size);
    const bool compact_success = compact_allocator.assignAddr(
        alloc_ops, live_ranges, bank_conflict_map, compact_addresses);
    consider_candidate(compact_success, std::move(compact_addresses),
                       compact_allocator.statistics(), prefix + "-compact");

    std::unordered_map<const BufferNode *, int64_t> size_addresses;
    MemAllocBeamSearch size_allocator(
        bank_num, bank_size, kSizeBeamWidth, kSizeCandidateLimit,
        BeamOrder::kSize);
    const bool size_success = size_allocator.assignAddr(
        alloc_ops, live_ranges, bank_conflict_map, size_addresses);
    consider_candidate(size_success, std::move(size_addresses),
                       size_allocator.statistics(),
                       prefix + "-search-size");

    std::unordered_map<const BufferNode *, int64_t> pressure_addresses;
    MemAllocBeamSearch pressure_allocator(
        bank_num, bank_size, kPressureBeamWidth, kPressureCandidateLimit,
        BeamOrder::kWeightedPressure);
    const bool pressure_success = pressure_allocator.assignAddr(
        alloc_ops, live_ranges, bank_conflict_map, pressure_addresses);
    consider_candidate(pressure_success, std::move(pressure_addresses),
                       pressure_allocator.statistics(),
                       prefix + "-search-pressure");
  }
  selected_statistics.explored_states = total_explored_states;
  ICHECK(success) << "TPUv7 local memory allocation failed. buffers="
                  << alloc_ops.size() << ", lmem=" << bank_num * bank_size
                  << " bytes";

  for (const BufferNode *op : alloc_ops) {
    ICHECK(selected_addresses.count(op))
        << "AddressAssign selected plan omitted buffer " << op->name;
    const int64_t address = selected_addresses.at(op);
    ICHECK_GE(address, 0);
    ICHECK_EQ(address % tpuv7::kTensorAlignBytes, 0);
    ICHECK_LE(address + live_ranges.at(op).tensor_size,
              bank_num * bank_size);
  }
  for (size_t lhs_index = 0; lhs_index < alloc_ops.size(); ++lhs_index) {
    const BufferNode *lhs = alloc_ops[lhs_index];
    const int64_t lhs_start = selected_addresses.at(lhs);
    const int64_t lhs_end = lhs_start + live_ranges.at(lhs).tensor_size;
    for (size_t rhs_index = lhs_index + 1; rhs_index < alloc_ops.size();
         ++rhs_index) {
      const BufferNode *rhs = alloc_ops[rhs_index];
      if (!LiveRangesOverlap(live_ranges.at(lhs), live_ranges.at(rhs))) {
        continue;
      }
      const int64_t rhs_start = selected_addresses.at(rhs);
      const int64_t rhs_end = rhs_start + live_ranges.at(rhs).tensor_size;
      ICHECK(std::max(lhs_start, rhs_start) >= std::min(lhs_end, rhs_end))
          << "AddressAssign selected overlapping live buffers " << lhs->name
          << " and " << rhs->name;
    }
  }

  auto fn = f.CopyOnWrite();
  auto fn_attr = fn->attrs.CopyOnWrite();
  for (auto op : alloc_ops) {
    int64_t address = selected_addresses.at(op);
    fn_attr->dict.Set(tpuv7::AddressAttrKey(op->data->name_hint),
                      IntImm(DataType::Int(64), address));
  }
  fn_attr->dict.Set("tilelang.tpu.lmem.statistics.policy",
                    String(selected_policy));
  fn_attr->dict.Set("tilelang.tpu.lmem.statistics.bank_conflicts",
                    IntImm(DataType::Int(64),
                           selected_statistics.bank_conflicts));
  fn_attr->dict.Set("tilelang.tpu.lmem.statistics.weighted_bank_conflicts",
                    IntImm(DataType::Int(64),
                           selected_statistics.weighted_bank_conflicts));
  fn_attr->dict.Set("tilelang.tpu.lmem.statistics.peak_bytes",
                    IntImm(DataType::Int(64), selected_statistics.peak_bytes));
  fn_attr->dict.Set(
      "tilelang.tpu.lmem.statistics.live_bytes_lower_bound",
      IntImm(DataType::Int(64), selected_statistics.live_bytes_lower_bound));
  fn_attr->dict.Set(
      "tilelang.tpu.lmem.statistics.peak_slack_bytes",
      IntImm(DataType::Int(64), selected_statistics.peak_slack_bytes));
  fn_attr->dict.Set(
      "tilelang.tpu.lmem.statistics.explored_states",
      IntImm(DataType::Int(64), selected_statistics.explored_states));

  return f;
}

tvm::transform::Pass AddressAssign() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    Optional<Target> target = f->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined())
        << "AddressAssign requires a PrimFunc bound to a TPU target";
    tpu::ResolveTarget(target.value(), "AddressAssign");
    const std::string policy =
        ctx->GetConfig<String>(kAddressAssignPolicy, String("greedy"))
            .value();
    const std::string replay_addresses =
        ctx->GetConfig<String>(kAddressAssignReplayAddresses, String(""))
            .value();
    const std::string external_allocator =
        ctx->GetConfig<String>(kAddressAssignExternalAllocator, String(""))
            .value();
    const bool emit_problem =
        ctx->GetConfig<Bool>(kAddressAssignEmitProblem, Bool(false))
            .value();
    const bool analysis_only =
        ctx->GetConfig<Bool>(kAddressAssignAnalysisOnly, Bool(false))
            .value();
    return InferAddress(f, policy, replay_addresses, external_allocator, emit_problem,
                        analysis_only);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AddressAssign", {});
}

TVM_REGISTER_GLOBAL("tl.transform.AddressAssign").set_body_typed(AddressAssign);

} // namespace tl
} // namespace tvm
