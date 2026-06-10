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

#ifndef XLA_BACKENDS_METAL_TRANSFORMS_METAL_GRAPH_PARTITIONER_H_
#define XLA_BACKENDS_METAL_TRANSFORMS_METAL_GRAPH_PARTITIONER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/metal/transforms/metal_graph_support.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace metal {

// Forms MPSGraph-bound regions: kCustom fusions with
// FusionBackendConfig.kind == "__metal_graph", anchored at translatable
// kDot/kConvolution instructions and grown over their translatable
// neighborhoods.
//
// Region invariant: every internally-produced value has all of its consumers
// inside the region except exactly one — the region result. Inward-crossing
// values become fusion operands. Growth never duplicates multi-user
// producers (scalar constants excepted) and never strands a single-use
// producer that would otherwise have fused with the absorbed op, so the
// number of materialized intermediates along any path is non-increasing.
//
// Runs after layout normalization and reduce canonicalization and before
// float normalization (see GpuCompiler::AddGraphCompilerFusionPasses), so
// captured regions may keep low-precision types the surrounding pipeline
// widens. Reduce-rooted regions are never formed here; a monoid reduce is
// only grown through from an anchor when its entire producer neighborhood is
// captured with it.
class MetalGraphPartitioner : public HloModulePass {
 public:
  explicit MetalGraphPartitioner(const MetalGraphCapabilities& caps)
      : caps_(caps) {}

  absl::string_view name() const override {
    return "metal-graph-partitioner";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads)
      override;

 private:
  MetalGraphCapabilities caps_;
};

}  // namespace metal
}  // namespace xla

#endif  // XLA_BACKENDS_METAL_TRANSFORMS_METAL_GRAPH_PARTITIONER_H_
