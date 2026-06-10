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

#ifndef XLA_BACKENDS_METAL_CODEGEN_METAL_GRAPH_EMITTER_H_
#define XLA_BACKENDS_METAL_CODEGEN_METAL_GRAPH_EMITTER_H_

#include <memory>

#include "absl/status/statusor.h"
#include "xla/backends/metal/runtime/metal_graph_artifact.h"
#include "xla/backends/metal/transforms/metal_graph_support.h"
#include "xla/hlo/ir/hlo_instructions.h"

namespace xla {
namespace metal {

// Probes MPSGraph-relevant capabilities of `mtl_device` (an id<MTLDevice>
// passed as an opaque void*; nullptr yields the conservative default).
MetalGraphCapabilities ProbeMetalGraphCapabilities(void* mtl_device);

// Translates the body of a __metal_graph fusion into an MPSGraph, compiles
// it for `mtl_device`, and returns the runtime artifact (executable plus
// feed/result binding descriptors, feeds in fusion operand order).
//
// Region boundaries bind in physical (slowest-to-fastest) dimension order;
// in-graph transposes at entry/exit recover logical order and are elided
// when the layout is descending. Interior translation is purely logical.
//
// The partitioner only forms regions the gate accepts, so any untranslatable
// instruction here is a compiler bug and returns InternalError (fail-loud).
absl::StatusOr<std::unique_ptr<MetalGraphArtifact>> BuildMetalGraphArtifact(
    void* mtl_device, const HloFusionInstruction& fusion);

}  // namespace metal
}  // namespace xla

#endif  // XLA_BACKENDS_METAL_CODEGEN_METAL_GRAPH_EMITTER_H_
