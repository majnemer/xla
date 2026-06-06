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
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/emitters/type_util.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/llvm_ir/llvm_util.h"
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

namespace {

gpu::LaunchDimensions ComputeStandardLaunchDimensions(
    const Shape& keys_shape, int64_t dimension_to_sort,
    int64_t standard_num_iterations_in_sort_dim,
    const se::DeviceDescription& device) {
  Shape standard_iteration_shape = keys_shape;
  standard_iteration_shape.set_dimensions(
      dimension_to_sort,
      CeilOfRatio<int64_t>(standard_num_iterations_in_sort_dim,
                           kBitonicSortUnrollFactor));
  return gpu::CalculateLaunchDimensions(standard_iteration_shape, device);
}

gpu::LaunchDimensions ComputeTiledLaunchDimensions(
    const Shape& keys_shape, int64_t dimension_to_sort,
    int64_t dimension_to_sort_bound, int64_t tile_size,
    int64_t* num_iterations_in_sort_dim_out) {
  uint64_t rounded_bound = RoundUpTo<uint64_t>(dimension_to_sort_bound,
                                               tile_size);
  Shape iteration_shape = keys_shape;
  uint64_t num_iterations_in_sort_dim =
      CeilOfRatio<uint64_t>(rounded_bound, kBitonicSortUnrollFactor);
  iteration_shape.set_dimensions(dimension_to_sort,
                                 num_iterations_in_sort_dim);
  uint64_t num_iterations = ShapeUtil::ElementsIn(iteration_shape);
  uint64_t threads_per_block =
      std::max<uint64_t>(1, tile_size / kBitonicSortUnrollFactor);
  uint64_t num_blocks = CeilOfRatio<uint64_t>(num_iterations,
                                              threads_per_block);
  *num_iterations_in_sort_dim_out = num_iterations_in_sort_dim;
  return gpu::LaunchDimensions(num_blocks, threads_per_block);
}

}  // namespace

absl::StatusOr<std::vector<SortStageDescription>> PlanBitonicSort(
    const HloSortInstruction* sort, const BufferAssignment& buffer_assignment,
    const se::DeviceDescription& device,
    const emitters::KernelArguments::BufferAlignment& buffer_alignment,
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
      device.shared_memory_per_block() /
      std::max<uint64_t>(total_element_size, 1);
  const uint64_t max_threads_per_block = device.threads_per_block_limit();

  uint64_t tile_size = std::min(
      {max_threads_per_block * kBitonicSortUnrollFactor, max_tile_in_shmem,
       uint64_t{1} << num_stages});
  tile_size = Pow2Floor(std::max<uint64_t>(tile_size, 1));

  // Standard (non-tiled) launch covers ceil(2^(num_stages-1)/unroll) element
  // pairs per thread along the sort dimension; one element pair compared per
  // iteration. Tiled launch processes one tile_size-wide tile per block, so
  // threads_per_block = tile_size / unroll_factor.
  const uint64_t standard_num_iterations_in_sort_dim =
      uint64_t{1} << (num_stages - 1);
  gpu::LaunchDimensions standard_launch = ComputeStandardLaunchDimensions(
      keys_shape, dimension_to_sort, standard_num_iterations_in_sort_dim,
      device);
  int64_t tiled_num_iterations_in_sort_dim = 0;
  gpu::LaunchDimensions tiled_launch = ComputeTiledLaunchDimensions(
      keys_shape, dimension_to_sort, dimension_to_sort_bound, tile_size,
      &tiled_num_iterations_in_sort_dim);

  TF_ASSIGN_OR_RETURN(
      emitters::KernelArguments full_kernel_args,
      emitters::KernelArguments::Create(buffer_assignment, buffer_alignment,
                                        sort));
  // Sort runs in place on the output buffers; the input buffers alias the
  // outputs after the D2D copy emitted earlier, so the kernel only takes the
  // output buffers. Drop the operand half (the first operand_count args) so
  // we stay under Metal's 31-buffer-argument limit for wide many-input sorts.
  TF_RET_CHECK(full_kernel_args.args().size() == 2 * sort->operand_count());
  std::vector<emitters::KernelArgument> output_args(
      full_kernel_args.args().begin() + sort->operand_count(),
      full_kernel_args.args().end());
  emitters::KernelArguments kernel_args(std::move(output_args));

  std::vector<SortStageDescription> stages;
  bool emit_iota_operands = true;
  auto emit_stage = [&](std::vector<int64_t> xor_masks) {
    bool tiled = xor_masks.size() > 1;
    SortStageDescription stage{
        /*sort=*/sort,
        /*xor_masks=*/std::move(xor_masks),
        /*tile_size=*/tiled ? static_cast<int64_t>(tile_size) : 0,
        /*num_iterations_in_sort_dim=*/
        tiled ? tiled_num_iterations_in_sort_dim
              : static_cast<int64_t>(standard_num_iterations_in_sort_dim),
        /*launch_dimensions=*/tiled ? tiled_launch : standard_launch,
        /*emit_iota_operands=*/emit_iota_operands,
        /*kernel_args=*/kernel_args,
        /*entry_name=*/
        absl::StrCat(entry_name_prefix, "_stage", stages.size()),
    };
    stages.push_back(std::move(stage));
    emit_iota_operands = false;
  };

  // Mirror EmitBitonicSortLLVMIR's stage/mask loop. Masks smaller than
  // tile_size pile up into one tiled kernel; once we hit a mask >= tile_size
  // we flush the pile and emit the big mask standalone.
  std::vector<int64_t> pending;
  for (int64_t stage_idx = 0; stage_idx < num_stages; ++stage_idx) {
    for (int64_t mask = stage_idx; mask >= 0; --mask) {
      int64_t xor_mask = (mask == stage_idx) ? ((int64_t{1} << (stage_idx + 1)) - 1)
                                             : (int64_t{1} << mask);
      if (static_cast<uint64_t>(xor_mask) >= tile_size) {
        if (!pending.empty()) {
          emit_stage(std::move(pending));
          pending.clear();
        }
        emit_stage({xor_mask});
      } else {
        pending.push_back(xor_mask);
      }
    }
  }
  if (!pending.empty()) emit_stage(std::move(pending));
  return stages;
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

namespace {

constexpr absl::string_view kXlaEntryAttr = "xla.entry";
constexpr absl::string_view kXlaSliceIndexAttr = "xla.slice_index";
constexpr absl::string_view kXlaInvariantAttr = "xla.invariant";

}  // namespace

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> EmitSortStageModule(
    mlir::MLIRContext* context, const SortStageDescription& desc) {
  mlir::OpBuilder builder(context);
  auto loc = mlir::NameLoc::get(builder.getStringAttr(desc.entry_name));
  mlir::OwningOpRef<mlir::ModuleOp> module = llvm_ir::CreateMlirModuleOp(loc);

  mlir::ImplicitLocOpBuilder b(loc, *module);
  const auto& args = desc.kernel_args.args();
  const int64_t operand_count = desc.sort->operand_count();
  // PlanBitonicSort drops the operand-half of KernelArguments::Create's
  // output, so the kernel signature carries only the in-place output buffers.
  TF_RET_CHECK(args.size() == operand_count);

  llvm::SmallVector<mlir::Type> param_types;
  llvm::SmallVector<mlir::Attribute> arg_attrs;
  param_types.reserve(args.size());
  arg_attrs.reserve(args.size());
  for (const auto& arg : args) {
    param_types.push_back(emitters::TensorShapeToMlirType(arg.shape(), b));
    llvm::SmallVector<mlir::NamedAttribute> attrs;
    attrs.push_back(b.getNamedAttr(kXlaSliceIndexAttr,
                                   b.getIndexAttr(arg.slice_index())));
    attrs.push_back(
        b.getNamedAttr(mlir::LLVM::LLVMDialect::getAlignAttrName(),
                       b.getIndexAttr(arg.alignment())));
    attrs.push_back(
        b.getNamedAttr(mlir::LLVM::LLVMDialect::getDereferenceableAttrName(),
                       b.getIndexAttr(arg.slice().size())));
    if (!arg.written()) {
      attrs.push_back(b.getNamedAttr(kXlaInvariantAttr, b.getUnitAttr()));
    }
    arg_attrs.push_back(b.getDictionaryAttr(attrs));
  }

  // In-place sort: result tensors are the same buffers we received.
  llvm::SmallVector<mlir::Type> result_types(param_types);

  b.setInsertionPointToStart(module->getBody());
  auto entry_func = mlir::func::FuncOp::create(
      b, desc.entry_name,
      mlir::FunctionType::get(context, param_types, result_types),
      /*sym_visibility=*/mlir::StringAttr{},
      mlir::ArrayAttr::get(context, arg_attrs),
      /*res_attrs=*/mlir::ArrayAttr{});
  entry_func->setAttr(kXlaEntryAttr, mlir::UnitAttr::get(context));

  // Lower the comparator HLO into callable MLIR func(s) before the entry
  // body so the kernel can call_target it. PartitionedComputations handles
  // sub-graph extraction (and any nested called computations); the call
  // target provider maps each HLO root to its emitted FuncOp. Sort's
  // comparator is the canonical "called computation": (...scalar) -> i1.
  const HloComputation* comparator = desc.sort->to_apply();
  emitters::PartitionedComputations partitioned(comparator, context);
  TF_ASSIGN_OR_RETURN(
      emitters::CallTargetProvider comparator_call_targets,
      emitters::EmitPartitionedComputations(*module, partitioned));
  // Bound but not yet used: the kernel body below is still passthrough
  // pending the compare-and-swap lowering. Silence the unused warning.
  (void)comparator_call_targets;

  // Trivial passthrough body: return the args unchanged. The full
  // compare-and-swap body lands in a follow-up commit; this minimum-viable
  // body lets the Phase-2 plumbing exercise the end-to-end pipeline.
  mlir::Block* entry_block = entry_func.addEntryBlock();
  b.setInsertionPointToStart(entry_block);
  llvm::SmallVector<mlir::Value> returns(entry_block->getArguments());
  mlir::func::ReturnOp::create(b, returns);

  return module;
}

}  // namespace xla::metal
