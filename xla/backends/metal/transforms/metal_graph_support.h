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

#ifndef XLA_BACKENDS_METAL_TRANSFORMS_METAL_GRAPH_SUPPORT_H_
#define XLA_BACKENDS_METAL_TRANSFORMS_METAL_GRAPH_SUPPORT_H_

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {
namespace metal {

// Device/OS capabilities that widen the translatable set. Probed from the
// MTLDevice by the codegen layer (ProbeMetalGraphCapabilities in
// metal_graph_emitter.h); plain data here so the transforms layer stays
// Obj-C-free.
struct MetalGraphCapabilities {
  bool supports_bf16 = false;
};

// True iff `instr` is a kCustom fusion with FusionBackendConfig.kind ==
// gpu::kMetalGraphFusionKind, i.e. an MPSGraph-bound region.
bool IsMetalGraphFusion(const HloInstruction& instr);

// True iff the MPSGraph emitter can translate `instr` as a node of a
// captured region. Single source of truth consulted by the partitioner and
// the autotuner backend. The emitter additionally accepts the
// float-normalization rewrites of gated ops (inserted converts and widened
// elementwise), which are themselves in the gated set.
bool IsMpsGraphTranslatable(const HloInstruction& instr,
                            const MetalGraphCapabilities& caps);

// OkStatus iff every instruction of `computation` passes the gate. Used to
// validate (would-be) fusion bodies offered to the autotuner.
absl::Status RegionIsTranslatable(const HloComputation& computation,
                                  const MetalGraphCapabilities& caps);

}  // namespace metal
}  // namespace xla

#endif  // XLA_BACKENDS_METAL_TRANSFORMS_METAL_GRAPH_SUPPORT_H_
