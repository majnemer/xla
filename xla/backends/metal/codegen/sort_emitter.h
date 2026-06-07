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

#ifndef XLA_BACKENDS_METAL_CODEGEN_SORT_EMITTER_H_
#define XLA_BACKENDS_METAL_CODEGEN_SORT_EMITTER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/device_description.h"

namespace xla::metal {

// Description of one bitonic-sort kernel invocation. The compiler plans a
// sequence of these for each sort and then emits MSL for each one through the
// same deferred pipeline used for fusions.
//
// Each stage compares element pairs across the sort dimension using the
// `xor_masks`. When `tile_size > 0`, the stage loads `tile_size` elements
// per threadgroup into a shared (threadgroup) tile, sweeps the masks in
// order, then writes the tile back; xor masks smaller than `tile_size`
// route through this path. Larger xor masks set `tile_size = 0` and stream
// pairs directly through global memory. `tile_size` is a power of two and
// decreases on PSO-retry.
struct SortStageDescription {
  // The sort HLO this stage belongs to. The comparator computation is re-
  // resolved from this pointer during Phase-2 MLIR emission.
  const HloSortInstruction* sort = nullptr;
  // xor_masks for this stage; one mask per kernel today, multi-mask
  // bundling is a future optimisation.
  std::vector<int64_t> xor_masks;
  // Tile width in elements per threadgroup when the stage runs in shared
  // memory; 0 means the stage runs in global memory.
  int64_t tile_size = 0;
  // Non-tiled stages: number of element-pair iterations along the sort
  // dimension. Tiled stages: number of tiles in the sort dimension.
  int64_t num_iterations_in_sort_dim = 0;
  // Launch dimensions chosen for this stage.
  gpu::LaunchDimensions launch_dimensions;
  // True only on the first stage that touches an iota operand. Subsequent
  // stages read the same operand from the output buffer it was written to.
  bool emit_iota_operands = false;
  // Kernel arguments (slice + alignment) for this stage's buffers.
  emitters::KernelArguments kernel_args;
  // Globally unique MSL entry-function name.
  std::string entry_name;
};

// Plans a sequence of bitonic-sort stages for `sort`. Picks `tile_size`,
// per-stage `launch_dimensions`, and the xor-mask groupings using the same
// algorithm the LLVM GPU backend's `EmitBitonicSortLLVMIR` uses, but with
// Metal-flavored device limits.
//
// `entry_name_prefix` is the MSL function-name prefix to use for each stage's
// emitted entry; the planner appends a stage suffix to make each name unique.
absl::StatusOr<std::vector<SortStageDescription>> PlanBitonicSort(
    const HloSortInstruction* sort, const BufferAssignment& buffer_assignment,
    const se::DeviceDescription& device,
    const emitters::KernelArguments::BufferAlignment& buffer_alignment,
    const std::string& entry_name_prefix);

// Re-plans `desc` with a halved tile_size (rounded down to the next power of
// two, never below threads_per_warp). Returns false if there's no smaller
// usable tile size — caller should surface ResourceExhausted in that case.
// Recomputes launch_dimensions accordingly.
bool ShrinkSortStageTile(SortStageDescription& desc,
                         const se::DeviceDescription& device);

// Builds an MLIR module for one sort stage. The module contains:
//   * a `func.func` per comparator subgraph (declared and bodies emitted)
//   * a `func.func @<entry_name>` with the `xla.entry` attribute carrying the
//     compare-and-swap kernel body
//
// `context` is mutated to load needed dialects. The returned module is then
// fed through Metal's standard MSL lowering + translation pipeline.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> EmitSortStageModule(
    mlir::MLIRContext* context, const SortStageDescription& desc);

}  // namespace xla::metal

#endif  // XLA_BACKENDS_METAL_CODEGEN_SORT_EMITTER_H_
