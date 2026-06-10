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

#include "xla/backends/gpu/autotuner/metal_graph.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/metal/transforms/metal_graph_support.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

bool MetalGraphBackend::IsSupported(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return false;
  }
  if (metal::IsMetalGraphFusion(instr)) {
    return true;
  }
  const auto* fusion = Cast<HloFusionInstruction>(&instr);
  if (fusion->fusion_kind() == HloInstruction::FusionKind::kCustom) {
    return false;
  }
  // Capabilities mirror ProbeMetalGraphCapabilities' current conservative
  // default; widen both together.
  return metal::RegionIsTranslatable(*fusion->fused_instructions_computation(),
                                     metal::MetalGraphCapabilities{})
      .ok();
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
MetalGraphBackend::GetSupportedConfigs(const HloInstruction& instr) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  if (!IsSupported(instr)) {
    return configs;
  }
  TF_ASSIGN_OR_RETURN(std::unique_ptr<BackendConfig> config,
                      GetDefaultConfig(instr));
  configs.push_back(std::move(config));
  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>>
MetalGraphBackend::GetDefaultConfig(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        "MetalGraphBackend does not support this instruction.");
  }
  auto config = std::make_unique<BackendConfig>();
  config->mutable_metal_graph();
  return config;
}

absl::Status MetalGraphBackend::ApplyConfig(HloInstruction& instr,
                                            const BackendConfig& config) {
  if (!config.has_metal_graph()) {
    return absl::InvalidArgumentError("Expected MetalGraphFusionConfig.");
  }
  auto* fusion = Cast<HloFusionInstruction>(&instr);
  fusion->set_fusion_kind(HloInstruction::FusionKind::kCustom);
  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                      instr.backend_config<GpuBackendConfig>());
  FusionBackendConfig& backend_config =
      *gpu_config.mutable_fusion_backend_config();
  backend_config.set_kind(std::string(kMetalGraphFusionKind));
  *backend_config.mutable_metal_graph_fusion_config() = config.metal_graph();
  TF_RETURN_IF_ERROR(fusion->set_backend_config(gpu_config));
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
