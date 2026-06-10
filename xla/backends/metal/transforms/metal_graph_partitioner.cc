/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/backends/metal/transforms/metal_graph_partitioner.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xla/backends/metal/transforms/metal_graph_support.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace metal {
namespace {

constexpr int64_t kMaxRegionOps = 64;
constexpr int64_t kMaxRegionParams = 32;

using InstructionSet = absl::flat_hash_set<HloInstruction*>;

// Ops whose results are materialized today regardless of fusion decisions,
// so turning them into region parameters adds no boundary.
bool ResultMaterializesToday(const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kParameter:
    case HloOpcode::kConstant:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kDot:
    case HloOpcode::kConvolution:
    case HloOpcode::kCustomCall:
    case HloOpcode::kFusion:
    case HloOpcode::kCall:
    case HloOpcode::kWhile:
    case HloOpcode::kConditional:
    case HloOpcode::kInfeed:
    case HloOpcode::kRng:
    case HloOpcode::kRngGetAndUpdateState:
    case HloOpcode::kFft:
    case HloOpcode::kTriangularSolve:
    case HloOpcode::kCholesky:
    case HloOpcode::kSort:
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllGather:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectivePermute:
      return true;
    default:
      return false;
  }
}

// A value produced outside the region may feed it as a parameter without
// adding a boundary iff it is materialized today anyway.
bool ParamOk(const HloInstruction* instr) {
  return instr->user_count() > 1 || ResultMaterializesToday(instr);
}

class RegionBuilder {
 public:
  RegionBuilder(HloComputation* computation, const MetalGraphCapabilities& caps,
                absl::FunctionRef<bool(const HloInstruction*)> eligible)
      : computation_(computation), caps_(caps), eligible_(eligible) {}

  // Builds the maximal region around `anchor` and returns its members in
  // reverse topological order (root first), ready for
  // CreateFusionInstruction. Never returns an empty list.
  std::vector<HloInstruction*> Build(HloInstruction* anchor) {
    InstructionSet region;
    region.insert(anchor);
    // The anchor's operand edges are boundaries today (dots/convs never fuse
    // with their producers), so unabsorbable operands become parameters
    // freely; absorb what the strict rules allow.
    for (HloInstruction* operand : anchor->operands()) {
      if (!region.contains(operand)) {
        TryAbsorbValue(operand, region);
      }
    }
    GrowDownstream(region);
    SweepOperandDiamonds(region);
    return RootFirstOrder(region);
  }

 private:
  bool Translatable(const HloInstruction* instr) const {
    return IsMpsGraphTranslatable(*instr, caps_);
  }

  static bool AllUsersIn(const HloInstruction* instr,
                         const InstructionSet& region) {
    for (const HloInstruction* user : instr->users()) {
      if (!region.contains(const_cast<HloInstruction*>(user))) {
        return false;
      }
    }
    return true;
  }

  int64_t CountParams(const InstructionSet& region) const {
    InstructionSet params;
    for (HloInstruction* member : region) {
      for (HloInstruction* operand : member->operands()) {
        if (!region.contains(operand)) {
          params.insert(operand);
        }
      }
    }
    return params.size();
  }

  // Strictly absorbs `value` and its operand closure into `region`, or
  // leaves `region` unchanged. Scalar constants absorb unconditionally
  // (fusion clones them; duplication is free).
  bool TryAbsorbValue(HloInstruction* value, InstructionSet& region) {
    if (!eligible_(value)) {
      return false;
    }
    // The gate accepts kParameter for fusion-body validation, but a
    // computation parameter is a region input by definition.
    if (value->opcode() == HloOpcode::kParameter) {
      return false;
    }
    if (value->opcode() == HloOpcode::kConstant) {
      if (Translatable(value)) {
        region.insert(value);
        return true;
      }
      return false;
    }
    if (!Translatable(value) || !AllUsersIn(value, region)) {
      return false;
    }
    InstructionSet tentative = region;
    tentative.insert(value);
    if (!TryAbsorbOperandsOf(value, tentative)) {
      return false;
    }
    region = std::move(tentative);
    return true;
  }

  // Zero-stranding rule: every out-of-region operand must either absorb
  // (with its own closure) or be materialized today anyway. A single-use
  // producer that would have fused with `instr` may not be cut away from it.
  bool TryAbsorbOperandsOf(HloInstruction* instr, InstructionSet& region) {
    for (HloInstruction* operand : instr->operands()) {
      if (region.contains(operand)) {
        continue;
      }
      if (TryAbsorbValue(operand, region)) {
        continue;
      }
      if (!ParamOk(operand)) {
        return false;
      }
    }
    return region.size() <= kMaxRegionOps;
  }

  // Out-values: members with a consumer outside the region, plus the
  // computation root (its value escapes through the computation result).
  std::vector<HloInstruction*> OutValues(const InstructionSet& region) const {
    std::vector<HloInstruction*> outs;
    for (HloInstruction* member : region) {
      if (member == computation_->root_instruction() ||
          !AllUsersIn(member, region)) {
        outs.push_back(member);
      }
    }
    return outs;
  }

  bool TryAbsorbUser(HloInstruction* user, InstructionSet& region) {
    if (!eligible_(user)) {
      return false;
    }
    if (user->parent() != computation_ || user->HasSideEffect() ||
        !Translatable(user)) {
      return false;
    }
    InstructionSet tentative = region;
    tentative.insert(user);
    if (!TryAbsorbOperandsOf(user, tentative)) {
      return false;
    }
    region = std::move(tentative);
    return true;
  }

  // Grows past the current result while the region reconverges to a single
  // out-value: plain chains absorb one user at a time; reconvergent diamonds
  // (x and f(x) meeting at a binary op) absorb all users of the fan-out
  // value and chase until a single out-value remains. Rolls back to the last
  // single-out state on any failure.
  void GrowDownstream(InstructionSet& region) {
    while (true) {
      std::vector<HloInstruction*> outs = OutValues(region);
      CHECK_EQ(outs.size(), 1u);
      HloInstruction* result = outs.front();
      if (result == computation_->root_instruction() ||
          result->user_count() == 0) {
        return;
      }

      InstructionSet tentative = region;
      bool ok = true;
      for (int64_t iter = 0; iter <= kMaxRegionOps; ++iter) {
        std::vector<HloInstruction*> frontier = OutValues(tentative);
        if (frontier.size() == 1 && tentative.size() > region.size()) {
          break;
        }
        bool progressed = false;
        for (HloInstruction* value : frontier) {
          if (value == computation_->root_instruction()) {
            // The computation result escapes; reconvergence past it is
            // impossible when other out-values exist.
            ok = frontier.size() == 1;
            progressed = false;
            break;
          }
          for (HloInstruction* user : value->users()) {
            if (tentative.contains(user)) {
              continue;
            }
            if (!TryAbsorbUser(user, tentative)) {
              ok = false;
              break;
            }
            progressed = true;
          }
          if (!ok) {
            break;
          }
        }
        if (!ok || !progressed) {
          ok = ok && OutValues(tentative).size() == 1 &&
               tentative.size() > region.size();
          break;
        }
      }
      if (!ok || OutValues(tentative).size() != 1 ||
          tentative.size() <= region.size() ||
          tentative.size() > kMaxRegionOps ||
          CountParams(tentative) > kMaxRegionParams) {
        return;
      }
      region = std::move(tentative);
    }
  }

  // Operand-side diamonds: a producer whose users all landed in the region
  // only becomes absorbable once the last of them joined. Sweep to fixpoint.
  void SweepOperandDiamonds(InstructionSet& region) {
    bool changed = true;
    while (changed && region.size() < kMaxRegionOps) {
      changed = false;
      std::vector<HloInstruction*> candidates;
      for (HloInstruction* member : region) {
        for (HloInstruction* operand : member->operands()) {
          if (!region.contains(operand) &&
              operand->opcode() != HloOpcode::kConstant) {
            candidates.push_back(operand);
          }
        }
      }
      for (HloInstruction* candidate : candidates) {
        if (!region.contains(candidate) && TryAbsorbValue(candidate, region)) {
          changed = true;
        }
      }
    }
  }

  std::vector<HloInstruction*> RootFirstOrder(const InstructionSet& region) {
    std::vector<HloInstruction*> order;
    order.reserve(region.size());
    for (HloInstruction* instr : computation_->MakeInstructionPostOrder()) {
      if (region.contains(instr)) {
        order.push_back(instr);
      }
    }
    std::reverse(order.begin(), order.end());
    return order;
  }

  HloComputation* computation_;
  const MetalGraphCapabilities& caps_;
  absl::FunctionRef<bool(const HloInstruction*)> eligible_;
};

absl::StatusOr<bool> PartitionComputation(
    HloComputation* computation, const MetalGraphCapabilities& caps) {
  bool changed = false;
  InstructionSet refused;
  // Capture mutates the computation (fused originals are removed), so rescan
  // for the next live anchor after each capture instead of holding pointers.
  while (true) {
    HloInstruction* anchor = nullptr;
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if ((instr->opcode() == HloOpcode::kDot ||
           instr->opcode() == HloOpcode::kConvolution) &&
          !refused.contains(instr)) {
        anchor = instr;
        break;
      }
    }
    if (anchor == nullptr) {
      break;
    }
    if (!IsMpsGraphTranslatable(*anchor, caps)) {
      refused.insert(anchor);
      continue;
    }
    TF_RETURN_IF_ERROR(
        CaptureMetalGraphRegion(computation, anchor, caps,
                                [](const HloInstruction*) { return true; })
            .status());
    changed = true;
  }
  return changed;
}

}  // namespace

absl::StatusOr<HloInstruction*> CaptureMetalGraphRegion(
    HloComputation* computation, HloInstruction* anchor,
    const MetalGraphCapabilities& caps,
    absl::FunctionRef<bool(const HloInstruction*)> eligible) {
  RegionBuilder builder(computation, caps, eligible);
  std::vector<HloInstruction*> region = builder.Build(anchor);
  HloInstruction* fusion = computation->CreateFusionInstruction(
      region, HloInstruction::FusionKind::kCustom);
  gpu::GpuBackendConfig config;
  config.mutable_fusion_backend_config()->set_kind(
      std::string(gpu::kMetalGraphFusionKind));
  TF_RETURN_IF_ERROR(fusion->set_backend_config(config));
  return fusion;
}

absl::StatusOr<bool> MetalGraphPartitioner::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Capture in the entry computation and in control-flow bodies (their
  // fusions are thunk-emitted recursively); never inside combiners,
  // comparators, or other applied computations.
  absl::flat_hash_set<HloComputation*> eligible;
  eligible.insert(module->entry_computation());
  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* instr : computation->instructions()) {
      switch (instr->opcode()) {
        case HloOpcode::kWhile:
        case HloOpcode::kConditional:
        case HloOpcode::kCall:
          for (HloComputation* called : instr->called_computations()) {
            eligible.insert(called);
          }
          break;
        default:
          break;
      }
    }
  }

  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    if (!eligible.contains(computation)) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(bool computation_changed,
                        PartitionComputation(computation, caps_));
    changed |= computation_changed;
  }
  return changed;
}

}  // namespace metal
}  // namespace xla
