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

#include "xla/backends/gpu/autotuner/metal_graph_partition.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/metal/transforms/metal_graph_partitioner.h"
#include "xla/backends/metal/transforms/metal_graph_support.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

// Bounds the per-fusion compile/profile cost of plan racing.
constexpr int64_t kMaxPlans = 8;

bool IsAnchor(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kDot ||
         instr->opcode() == HloOpcode::kConvolution;
}

std::vector<const HloInstruction*> CollectAnchors(const HloComputation& body) {
  std::vector<const HloInstruction*> anchors;
  for (const HloInstruction* instr : body.MakeInstructionPostOrder()) {
    if (IsAnchor(instr)) {
      anchors.push_back(instr);
    }
  }
  return anchors;
}

}  // namespace

bool MetalGraphPartitionBackend::IsSupported(const HloInstruction& instr) {
  if (!metal::IsMetalGraphFusion(instr)) {
    return false;
  }
  const HloComputation& body =
      *Cast<HloFusionInstruction>(&instr)->fused_instructions_computation();
  int64_t anchors = 0;
  int64_t ops = 0;
  for (const HloInstruction* body_instr : body.instructions()) {
    if (body_instr->opcode() == HloOpcode::kParameter) {
      continue;
    }
    ++ops;
    anchors += IsAnchor(body_instr) ? 1 : 0;
  }
  // A split needs at least one anchor and something to split away from it.
  return anchors >= 1 && ops > 1;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
MetalGraphPartitionBackend::GetSupportedConfigs(const HloInstruction& instr) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  if (!IsSupported(instr)) {
    return configs;
  }
  const HloComputation& body =
      *Cast<HloFusionInstruction>(&instr)->fused_instructions_computation();
  std::vector<const HloInstruction*> anchors = CollectAnchors(body);
  const int64_t n = anchors.size();
  const uint64_t full_mask = (uint64_t{1} << n) - 1;
  for (uint64_t mask = 0; mask <= full_mask; ++mask) {
    // With a single anchor, growing it reproduces the fusion as-is, which
    // the MetalGraphBackend already offers.
    if (n == 1 && mask == full_mask) {
      continue;
    }
    if (configs.size() == kMaxPlans) {
      LOG(INFO) << "MetalGraphPartitionBackend: capping plan enumeration at "
                << kMaxPlans << " of " << (full_mask + 1) << " plans for "
                << instr.name();
      break;
    }
    auto config = std::make_unique<BackendConfig>();
    xla::gpu::MetalGraphPartitionConfig* plan =
        config->mutable_metal_graph_partition();
    for (int64_t i = 0; i < n; ++i) {
      if (mask & (uint64_t{1} << i)) {
        plan->add_grown_anchors(std::string(anchors[i]->name()));
      }
    }
    configs.push_back(std::move(config));
  }
  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>>
MetalGraphPartitionBackend::GetDefaultConfig(const HloInstruction& instr) {
  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<BackendConfig>> configs,
                      GetSupportedConfigs(instr));
  if (configs.empty()) {
    return absl::InvalidArgumentError(
        "MetalGraphPartitionBackend does not support this instruction.");
  }
  return std::move(configs.front());
}

absl::Status MetalGraphPartitionBackend::ApplyConfig(
    HloInstruction& instr, const BackendConfig& config) {
  if (!config.has_metal_graph_partition()) {
    return absl::InvalidArgumentError("Expected MetalGraphPartitionConfig.");
  }
  if (!metal::IsMetalGraphFusion(instr)) {
    return absl::InvalidArgumentError(
        "MetalGraphPartitionBackend requires a __metal_graph fusion.");
  }
  auto* fusion = Cast<HloFusionInstruction>(&instr);
  HloComputation* parent = fusion->parent();
  HloComputation* body = fusion->fused_instructions_computation();

  absl::flat_hash_set<std::string> grown;
  for (const std::string& name : config.metal_graph_partition().grown_anchors()) {
    const HloInstruction* body_instr = body->GetInstructionWithName(name);
    if (body_instr == nullptr || !IsAnchor(body_instr)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Plan anchor '", name, "' is not an anchor of fusion ",
          fusion->name()));
    }
    grown.insert(name);
  }

  // Inline the body into the parent with an explicit clone map; plan anchors
  // are resolved through it, since cloned instructions may be renamed.
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> clone_map;
  absl::flat_hash_set<const HloInstruction*> inlined;
  std::vector<HloInstruction*> anchor_clones;
  std::vector<bool> anchor_grown;
  absl::flat_hash_set<const HloInstruction*> anchor_clone_set;
  for (const HloInstruction* body_instr : body->MakeInstructionPostOrder()) {
    if (body_instr->opcode() == HloOpcode::kParameter) {
      clone_map[body_instr] =
          fusion->mutable_operand(body_instr->parameter_number());
      continue;
    }
    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(body_instr->operand_count());
    for (const HloInstruction* operand : body_instr->operands()) {
      new_operands.push_back(clone_map.at(operand));
    }
    HloInstruction* clone = parent->AddInstruction(
        body_instr->CloneWithNewOperands(body_instr->shape(), new_operands));
    clone_map[body_instr] = clone;
    inlined.insert(clone);
    if (IsAnchor(body_instr)) {
      anchor_clones.push_back(clone);
      anchor_grown.push_back(grown.contains(body_instr->name()));
      anchor_clone_set.insert(clone);
    }
  }
  HloInstruction* new_root = clone_map.at(body->root_instruction());
  TF_RETURN_IF_ERROR(parent->ReplaceInstruction(fusion, new_root));
  // `instr`/`fusion` are gone; only the clones below are touched.

  metal::MetalGraphCapabilities caps;
  for (size_t i = 0; i < anchor_clones.size(); ++i) {
    HloInstruction* anchor = anchor_clones[i];
    absl::StatusOr<HloInstruction*> region_fusion;
    if (anchor_grown[i]) {
      // Grow within the inlined body only, never across another anchor —
      // keeping the rewrite identical between the measured candidate module
      // and the real module regardless of surrounding producers.
      auto eligible = [&inlined, &anchor_clone_set,
                       anchor](const HloInstruction* candidate) {
        return inlined.contains(candidate) &&
               (candidate == anchor || !anchor_clone_set.contains(candidate));
      };
      region_fusion =
          metal::CaptureMetalGraphRegion(parent, anchor, caps, eligible);
    } else {
      auto eligible = [anchor](const HloInstruction* candidate) {
        return candidate == anchor;
      };
      region_fusion =
          metal::CaptureMetalGraphRegion(parent, anchor, caps, eligible);
    }
    TF_RETURN_IF_ERROR(region_fusion.status());
    // Mark the subregions configured so candidate compiles (which re-run the
    // autotuner pass) treat them as settled.
    TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                        (*region_fusion)->backend_config<GpuBackendConfig>());
    gpu_config.mutable_fusion_backend_config()
        ->mutable_metal_graph_fusion_config();
    TF_RETURN_IF_ERROR((*region_fusion)->set_backend_config(gpu_config));
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
