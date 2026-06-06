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

#include "xla/backends/metal/codegen/sort_emitter.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/layout_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::metal {
namespace {

// Match the LLVM emitter's unroll factor so the same tile_size formula
// applies. Each thread copies kBitonicSortUnrollFactor adjacent elements
// into the shared tile and performs kBitonicSortUnrollFactor/2 compares.
constexpr uint64_t kBitonicSortUnrollFactor = 4;

uint64_t Pow2Floor(uint64_t value) {
  CHECK_GT(value, 0u);
  return uint64_t{1} << Log2Floor(value);
}

}  // namespace

absl::StatusOr<std::vector<SortStageDescription>> PlanBitonicSort(
    const HloSortInstruction* sort, const BufferAssignment& buffer_assignment,
    const se::DeviceDescription& device, int64_t buffer_alignment,
    const std::string& entry_name_prefix) {
  // Layout invariant: all operands and results share the keys-shape layout
  // (the bitonic sort scans one logical dimension in-place across all of
  // them). Enforce it here before any planning math.
  const Shape& keys_shape = sort->operand(0)->shape();
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
    TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
        keys_shape, sort->operand(i)->shape(),
        Layout::Equal().IgnoreMemorySpace().IgnoreElementSize()));
    ShapeIndex shape_index =
        sort->operand_count() > 1 ? ShapeIndex({i}) : ShapeIndex({});
    TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
        keys_shape, ShapeUtil::GetSubshape(sort->shape(), shape_index),
        Layout::Equal().IgnoreMemorySpace().IgnoreElementSize()));
  }

  const int64_t dimension_to_sort = sort->sort_dimension();
  const uint64_t dimension_to_sort_bound =
      keys_shape.dimensions(dimension_to_sort);
  const int64_t num_stages = Log2Ceiling(dimension_to_sort_bound);
  CHECK_GE(uint64_t{1} << num_stages, dimension_to_sort_bound);

  uint64_t total_element_size = 0;
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
    total_element_size += ShapeUtil::ByteSizeOfPrimitiveType(
        sort->operand(i)->shape().element_type());
  }
  const uint64_t max_tile_in_shmem =
      device.shared_memory_per_block() / std::max<uint64_t>(total_element_size,
                                                            1);
  const uint64_t max_threads_per_block = device.threads_per_block_limit();

  uint64_t tile_size = std::min(
      {max_threads_per_block * kBitonicSortUnrollFactor, max_tile_in_shmem,
       uint64_t{1} << num_stages});
  tile_size = Pow2Floor(std::max<uint64_t>(tile_size, 1));

  // TODO(majnemer): build SortStageDescription entries by walking the
  // (stage, mask) tree the same way EmitBitonicSortLLVMIR does, choosing
  // tiled vs global per mask group, computing launch_dimensions per stage,
  // resolving kernel_args from buffer_assignment.
  (void)buffer_assignment;
  (void)buffer_alignment;
  (void)entry_name_prefix;
  (void)tile_size;
  return absl::UnimplementedError(
      "MetalCompiler::PlanBitonicSort: stage planning not yet implemented.");
}

bool ShrinkSortStageTile(SortStageDescription& desc,
                         const se::DeviceDescription& device) {
  // Single-mask global-memory stages have no tile_size knob to shrink; their
  // PSO-retry uses threads_per_block_limit instead, like fusion kernels.
  if (desc.tile_size == 0 || desc.xor_masks.size() <= 1) return false;
  const uint64_t warp = device.threads_per_warp();
  uint64_t next = static_cast<uint64_t>(desc.tile_size) / 2;
  if (next < warp) return false;
  desc.tile_size = static_cast<int64_t>(Pow2Floor(next));
  // TODO(majnemer): recompute launch_dimensions for the smaller tile.
  return true;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> EmitSortStageModule(
    mlir::MLIRContext* context, const SortStageDescription& desc) {
  (void)context;
  (void)desc;
  return absl::UnimplementedError(
      "MetalCompiler::EmitSortStageModule: MLIR emission not yet implemented.");
}

}  // namespace xla::metal
