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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
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

// Effective unroll factor for the MLIR sort kernel. The LLVM emitter uses 4
// (each thread copies 4 adjacent elements into shared memory and runs 2
// pair-compares). The first-pass MLIR kernel does one element pair per
// thread per stage; set the unroll factor to 1 so PlanBitonicSort's launch
// dimensions match.
//
// TODO(majnemer): bump to 4 to match CUDA. Requires emitting an inner
// `unroll_factor`-trip loop in EmitSortStageModule that adds `i` to the
// element_pair_index for i in [0, unroll), with optional bank-conflict-
// aware indexing when num_threads % kNumShmemBanks == 0.
constexpr uint64_t kBitonicSortUnrollFactor = 1;

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
    int64_t* num_tiles_in_sort_dim_out) {
  // One block per tile in the sort dim, per position in all other dims;
  // threads_per_block = tile_size / 2 so each thread handles one element
  // pair (load 2 / compare-swap 1 pair / store 2).
  uint64_t num_tiles =
      CeilOfRatio<uint64_t>(dimension_to_sort_bound, tile_size);
  Shape iteration_shape = keys_shape;
  iteration_shape.set_dimensions(dimension_to_sort, num_tiles);
  uint64_t num_blocks = ShapeUtil::ElementsIn(iteration_shape);
  uint64_t threads_per_block = std::max<uint64_t>(1, tile_size / 2);
  *num_tiles_in_sort_dim_out = num_tiles;
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

  // Tile size: power-of-two upper-bounded by (threads-per-block * 2),
  // available shmem, and 2^num_stages. The factor of 2 reflects that the
  // tiled body assigns one element pair per thread (so tile_size = 2 *
  // threads_per_block).
  uint64_t tile_size = std::min(
      {max_threads_per_block * 2, max_tile_in_shmem,
       uint64_t{1} << num_stages});
  tile_size = Pow2Floor(std::max<uint64_t>(tile_size, 2));

  // Standard (non-tiled) launch covers ceil(2^(num_stages-1)/unroll) element
  // pairs per thread along the sort dimension; one element pair compared per
  // iteration. Tiled launch processes one tile_size-wide tile per block, so
  // threads_per_block = tile_size / 2.
  const uint64_t standard_num_iterations_in_sort_dim =
      uint64_t{1} << (num_stages - 1);
  gpu::LaunchDimensions standard_launch = ComputeStandardLaunchDimensions(
      keys_shape, dimension_to_sort, standard_num_iterations_in_sort_dim,
      device);
  int64_t tiled_num_tiles_in_sort_dim = 0;
  gpu::LaunchDimensions tiled_launch = ComputeTiledLaunchDimensions(
      keys_shape, dimension_to_sort, dimension_to_sort_bound, tile_size,
      &tiled_num_tiles_in_sort_dim);

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
  auto emit_stage = [&](std::vector<int64_t> xor_masks, bool tiled) {
    SortStageDescription stage{
        /*sort=*/sort,
        /*xor_masks=*/std::move(xor_masks),
        /*tile_size=*/tiled ? static_cast<int64_t>(tile_size) : 0,
        /*num_iterations_in_sort_dim=*/
        tiled ? tiled_num_tiles_in_sort_dim
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

  // Adjacent xor_masks below tile_size accumulate into one tiled stage
  // (single tile load+store amortised across all bundled masks). When a
  // mask >= tile_size appears, flush the bundle and emit the big mask as
  // its own global-memory pass.
  std::vector<int64_t> pending;
  auto flush_pending = [&]() {
    if (pending.empty()) return;
    emit_stage(std::move(pending), /*tiled=*/true);
    pending.clear();
  };
  for (int64_t stage_idx = 0; stage_idx < num_stages; ++stage_idx) {
    for (int64_t mask = stage_idx; mask >= 0; --mask) {
      int64_t xor_mask =
          (mask == stage_idx) ? ((int64_t{1} << (stage_idx + 1)) - 1)
                              : (int64_t{1} << mask);
      if (static_cast<uint64_t>(xor_mask) < tile_size) {
        pending.push_back(xor_mask);
      } else {
        flush_pending();
        emit_stage({xor_mask}, /*tiled=*/false);
      }
    }
  }
  flush_pending();
  return stages;
}

bool ShrinkSortStageTile(SortStageDescription& desc,
                         const se::DeviceDescription& device) {
  // Non-tiled (global-memory) stages have no tile_size knob to shrink;
  // their PSO-retry uses threads_per_block_limit instead, like fusion
  // kernels.
  if (desc.tile_size == 0) return false;
  const uint64_t warp = device.threads_per_warp();
  uint64_t next = static_cast<uint64_t>(desc.tile_size) / 2;
  // Tiled body needs >= 2 elements per tile (one pair); refuse to shrink
  // below 2 * warp so threads_per_block = tile_size / 2 keeps at least one
  // full SIMD group.
  if (next < 2 * warp) return false;
  desc.tile_size = static_cast<int64_t>(Pow2Floor(next));
  // TODO(majnemer): recompute launch_dimensions for the smaller tile.
  return true;
}

namespace {

constexpr absl::string_view kXlaEntryAttr = "xla.entry";
constexpr absl::string_view kXlaSliceIndexAttr = "xla.slice_index";
constexpr absl::string_view kXlaInvariantAttr = "xla.invariant";

}  // namespace

namespace {

// Tile-local index of the "left" element of the pair handled by `iter`,
// given the bitonic-block size for this xor_mask. Mirrors EmitCompareLoopBody.
mlir::Value DeriveCurrentTileIndex(mlir::ImplicitLocOpBuilder& b,
                                   mlir::Value iter, int64_t block_size) {
  namespace ma = mlir::arith;
  auto const_idx = [&](int64_t v) {
    return ma::ConstantIndexOp::create(b, v).getResult();
  };
  if (block_size == 1) {
    return ma::MulIOp::create(b, iter, const_idx(2));
  }
  mlir::Value block_size_c = const_idx(block_size);
  mlir::Value blk = ma::DivUIOp::create(b, iter, block_size_c);
  mlir::Value idx_in_blk = ma::RemUIOp::create(b, iter, block_size_c);
  mlir::Value first_in_block =
      ma::MulIOp::create(b, blk, const_idx(2 * block_size));
  return ma::AddIOp::create(b, first_in_block, idx_in_blk);
}

absl::Status EmitTiledBitonicSortBody(mlir::ImplicitLocOpBuilder& b,
                                      mlir::func::FuncOp entry_func,
                                      mlir::func::FuncOp comparator_func,
                                      const SortStageDescription& desc) {
  namespace ma = mlir::arith;
  const Shape& keys_shape = desc.sort->operand(0)->shape();
  const int64_t sort_dim = desc.sort->sort_dimension();
  const int64_t dim_to_sort_bound = keys_shape.dimensions(sort_dim);
  const int64_t rank = keys_shape.dimensions().size();
  const int64_t tile_size = desc.tile_size;
  const int64_t num_tiles_in_sort = desc.num_iterations_in_sort_dim;
  const int64_t operand_count = desc.sort->operand_count();
  TF_RET_CHECK(!desc.xor_masks.empty())
      << "tiled stage requires at least one xor_mask";

  mlir::Block* entry_block = entry_func.addEntryBlock();
  b.setInsertionPointToStart(entry_block);
  auto const_idx = [&](int64_t v) {
    return ma::ConstantIndexOp::create(b, v).getResult();
  };

  // One threadgroup tile per operand, sized tile_size in the sort dimension.
  llvm::SmallVector<mlir::Value, 4> tiles;
  llvm::SmallVector<mlir::Type, 4> tile_types;
  tiles.reserve(operand_count);
  tile_types.reserve(operand_count);
  for (int64_t i = 0; i < operand_count; ++i) {
    mlir::Type elem = mlir::cast<mlir::RankedTensorType>(
                          entry_block->getArgument(i).getType())
                          .getElementType();
    auto tile_ty = mlir::RankedTensorType::get({tile_size}, elem);
    tiles.push_back(
        ::xla::gpu::AllocateSharedOp::create(b, tile_ty).getResult());
    tile_types.push_back(tile_ty);
  }

  mlir::Value tid = mlir::gpu::ThreadIdOp::create(b, mlir::gpu::Dimension::x);
  mlir::Value bid = mlir::gpu::BlockIdOp::create(b, mlir::gpu::Dimension::x);

  // Decompose bid into the kernel's iteration shape (keys_shape with sort_dim
  // replaced by num_tiles_in_sort) using physical layout order so the
  // innermost coord matches the stride-1 dimension.
  llvm::SmallVector<mlir::Value, 4> indices(rank);
  mlir::Value remaining = bid;
  for (int64_t d : keys_shape.layout().minor_to_major()) {
    int64_t size =
        (d == sort_dim) ? num_tiles_in_sort : keys_shape.dimensions(d);
    mlir::Value size_c = const_idx(size);
    indices[d] = ma::RemUIOp::create(b, remaining, size_c);
    remaining = ma::DivUIOp::create(b, remaining, size_c);
  }
  mlir::Value tile_in_sort = indices[sort_dim];
  mlir::Value tile_base =
      ma::MulIOp::create(b, tile_in_sort, const_idx(tile_size));
  mlir::Value bound = const_idx(dim_to_sort_bound);

  // Each thread owns positions (2*tid, 2*tid+1) in the tile and the
  // corresponding global positions in the sort dim. Bounds-check both.
  mlir::Value off0 = ma::MulIOp::create(b, tid, const_idx(2));
  mlir::Value off1 = ma::AddIOp::create(b, off0, const_idx(1));
  mlir::Value sort_idx_0 = ma::AddIOp::create(b, tile_base, off0);
  mlir::Value sort_idx_1 = ma::AddIOp::create(b, tile_base, off1);
  mlir::Value in_bound_0 =
      ma::CmpIOp::create(b, ma::CmpIPredicate::ult, sort_idx_0, bound);
  mlir::Value in_bound_1 =
      ma::CmpIOp::create(b, ma::CmpIPredicate::ult, sort_idx_1, bound);

  auto build_global_idx = [&](mlir::Value sort_idx) {
    llvm::SmallVector<mlir::Value, 4> result(indices.begin(), indices.end());
    result[sort_dim] = sort_idx;
    return result;
  };
  llvm::SmallVector<mlir::Value, 4> global_idx_0 = build_global_idx(sort_idx_0);
  llvm::SmallVector<mlir::Value, 4> global_idx_1 = build_global_idx(sort_idx_1);

  llvm::SmallVector<mlir::Value, 4> entry_tensors(
      entry_block->getArguments().begin(), entry_block->getArguments().end());
  llvm::SmallVector<mlir::Type, 4> entry_types;
  for (mlir::Value e : entry_tensors) entry_types.push_back(e.getType());

  // Helper to emit one in-bounds conditional load or store on every operand.
  auto emit_load = [&](mlir::Value in_bound, mlir::Value tile_off,
                       llvm::ArrayRef<mlir::Value> global_idx) {
    auto if_op = mlir::scf::IfOp::create(b, tile_types, in_bound,
                                         /*withElseRegion=*/true);
    {
      mlir::OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(if_op.thenBlock());
      llvm::SmallVector<mlir::Value, 4> new_tiles;
      llvm::SmallVector<mlir::Value, 1> tile_pos{tile_off};
      for (int64_t i = 0; i < operand_count; ++i) {
        mlir::Value v = mlir::tensor::ExtractOp::create(b, entry_tensors[i],
                                                       global_idx);
        new_tiles.push_back(
            mlir::tensor::InsertOp::create(b, v, tiles[i], tile_pos));
      }
      mlir::scf::YieldOp::create(b, new_tiles);
    }
    {
      mlir::OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(if_op.elseBlock());
      mlir::scf::YieldOp::create(b, tiles);
    }
    tiles.assign(if_op.getResults().begin(), if_op.getResults().end());
  };
  auto emit_store = [&](mlir::Value in_bound, mlir::Value tile_off,
                        llvm::ArrayRef<mlir::Value> global_idx) {
    auto if_op = mlir::scf::IfOp::create(b, entry_types, in_bound,
                                         /*withElseRegion=*/true);
    {
      mlir::OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(if_op.thenBlock());
      llvm::SmallVector<mlir::Value, 4> new_entries;
      llvm::SmallVector<mlir::Value, 1> tile_pos{tile_off};
      for (int64_t i = 0; i < operand_count; ++i) {
        mlir::Value v =
            mlir::tensor::ExtractOp::create(b, tiles[i], tile_pos);
        new_entries.push_back(
            mlir::tensor::InsertOp::create(b, v, entry_tensors[i], global_idx));
      }
      mlir::scf::YieldOp::create(b, new_entries);
    }
    {
      mlir::OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(if_op.elseBlock());
      mlir::scf::YieldOp::create(b, entry_tensors);
    }
    entry_tensors.assign(if_op.getResults().begin(),
                         if_op.getResults().end());
  };

  // Load: each thread copies its two adjacent elements from global into the
  // tile, then a barrier publishes the tile to the threadgroup.
  emit_load(in_bound_0, off0, global_idx_0);
  emit_load(in_bound_1, off1, global_idx_1);
  auto sync_after_load =
      ::xla::gpu::SyncThreadsOp::create(b, tile_types, tiles);
  tiles.assign(sync_after_load.getResults().begin(),
               sync_after_load.getResults().end());

  // Compare phase: one sweep per bundled xor_mask, with a barrier between
  // sweeps so the updated tile is visible to all threads before the next
  // mask reads it. The bounds check uses the global positions so partial
  // tiles at the tail of the sort dim don't shuffle phantom pairs.
  for (int64_t mask_idx = 0; mask_idx < desc.xor_masks.size(); ++mask_idx) {
    const int64_t xor_mask = desc.xor_masks[mask_idx];
    int64_t block_size = xor_mask;
    if (xor_mask > 1 && (xor_mask & (xor_mask + 1)) == 0) {
      block_size = (xor_mask + 1) / 2;
    }
    mlir::Value tile_current = DeriveCurrentTileIndex(b, tid, block_size);
    mlir::Value tile_compare =
        ma::XOrIOp::create(b, tile_current, const_idx(xor_mask));
    mlir::Value global_current =
        ma::AddIOp::create(b, tile_base, tile_current);
    mlir::Value global_compare =
        ma::AddIOp::create(b, tile_base, tile_compare);
    mlir::Value cur_in =
        ma::CmpIOp::create(b, ma::CmpIPredicate::ult, global_current, bound);
    mlir::Value cmp_in =
        ma::CmpIOp::create(b, ma::CmpIPredicate::ult, global_compare, bound);
    mlir::Value in_bound = ma::AndIOp::create(b, cur_in, cmp_in);

    auto cmp_if = mlir::scf::IfOp::create(b, tile_types, in_bound,
                                          /*withElseRegion=*/true);
    {
      mlir::OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(cmp_if.thenBlock());
      llvm::SmallVector<mlir::Value, 4> compare_args;
      llvm::SmallVector<mlir::Value, 4> v_compare(operand_count);
      llvm::SmallVector<mlir::Value, 4> v_current(operand_count);
      llvm::SmallVector<mlir::Value, 1> cur_pos{tile_current};
      llvm::SmallVector<mlir::Value, 1> cmp_pos{tile_compare};
      compare_args.reserve(2 * operand_count);
      for (int64_t i = 0; i < operand_count; ++i) {
        v_compare[i] = mlir::tensor::ExtractOp::create(b, tiles[i], cmp_pos);
        v_current[i] = mlir::tensor::ExtractOp::create(b, tiles[i], cur_pos);
        compare_args.push_back(v_compare[i]);
        compare_args.push_back(v_current[i]);
      }
      mlir::Value cmp_result =
          mlir::func::CallOp::create(b, comparator_func, compare_args)
              .getResult(0);
      mlir::Value cmp_i1 = ma::CmpIOp::create(
          b, ma::CmpIPredicate::ne, cmp_result,
          ma::ConstantOp::create(b, cmp_result.getType(),
                                 b.getIntegerAttr(cmp_result.getType(), 0))
              .getResult());
      auto swap_if = mlir::scf::IfOp::create(b, tile_types, cmp_i1,
                                             /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard sg(b);
        b.setInsertionPointToStart(swap_if.thenBlock());
        llvm::SmallVector<mlir::Value, 4> swapped;
        for (int64_t i = 0; i < operand_count; ++i) {
          mlir::Value t = mlir::tensor::InsertOp::create(b, v_current[i],
                                                         tiles[i], cmp_pos);
          t = mlir::tensor::InsertOp::create(b, v_compare[i], t, cur_pos);
          swapped.push_back(t);
        }
        mlir::scf::YieldOp::create(b, swapped);
      }
      {
        mlir::OpBuilder::InsertionGuard sg(b);
        b.setInsertionPointToStart(swap_if.elseBlock());
        mlir::scf::YieldOp::create(b, tiles);
      }
      mlir::scf::YieldOp::create(b, swap_if.getResults());
    }
    {
      mlir::OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(cmp_if.elseBlock());
      mlir::scf::YieldOp::create(b, tiles);
    }
    tiles.assign(cmp_if.getResults().begin(), cmp_if.getResults().end());

    auto sync_after_cmp =
        ::xla::gpu::SyncThreadsOp::create(b, tile_types, tiles);
    tiles.assign(sync_after_cmp.getResults().begin(),
                 sync_after_cmp.getResults().end());
  }

  // Store: each thread writes its two tile elements back to global.
  emit_store(in_bound_0, off0, global_idx_0);
  emit_store(in_bound_1, off1, global_idx_1);

  b.setInsertionPointToEnd(entry_block);
  mlir::func::ReturnOp::create(b, entry_tensors);
  return absl::OkStatus();
}

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
  mlir::func::FuncOp comparator_func =
      comparator_call_targets(comparator->root_instruction());

  TF_RET_CHECK(desc.tile_size != 0 || desc.xor_masks.size() == 1)
      << "non-tiled stages emit one mask per kernel";

  // TODO(majnemer): emit iota inline. EmitCompareLoopBody (sort_util.cc)
  // checks `emit_iota_operands && operand is kIota` and calls EmitIota
  // instead of reading from the buffer. The MLIR version would inline an
  // iota_op_from_index call on the first stage that touches each iota
  // operand and then read from the output buffer on subsequent stages.
  for (int64_t i = 0; i < operand_count; ++i) {
    if (HloPredicateIsOp<HloOpcode::kIota>(desc.sort->operand(i))) {
      return absl::UnimplementedError(absl::StrCat(
          "MetalCompiler::EmitSortStageModule: iota operand at index ", i,
          " is not yet supported by the MLIR sort kernel."));
    }
  }

  if (desc.tile_size != 0) {
    TF_RETURN_IF_ERROR(
        EmitTiledBitonicSortBody(b, entry_func, comparator_func, desc));
    return module;
  }

  const Shape& keys_shape = desc.sort->operand(0)->shape();
  const int64_t sort_dim = desc.sort->sort_dimension();
  const int64_t dim_to_sort_bound = keys_shape.dimensions(sort_dim);
  const int64_t rank = keys_shape.dimensions().size();
  // iteration_shape_sort_dim drives the thread→iteration decomposition; the
  // bitonic block-size logic and bounds checks use the *actual* array length
  // (dim_to_sort_bound). Match EmitCompareLoopBody, which is called with
  // dimension_to_sort_bound as its iteration_bound parameter.
  const int64_t iteration_shape_sort_dim = desc.num_iterations_in_sort_dim;
  const int64_t xor_mask = desc.xor_masks[0];
  int64_t block_size = xor_mask;
  if (xor_mask > 1 && (xor_mask & (xor_mask + 1)) == 0) {
    block_size = (xor_mask + 1) / 2;
  }
  if (block_size >= dim_to_sort_bound) {
    // No adjacent block to compare with; emit a no-op kernel body so the
    // PSO is still installed for thunk dispatch.
    mlir::Block* entry_block = entry_func.addEntryBlock();
    b.setInsertionPointToStart(entry_block);
    llvm::SmallVector<mlir::Value> returns(entry_block->getArguments());
    mlir::func::ReturnOp::create(b, returns);
    return module;
  }

  mlir::Block* entry_block = entry_func.addEntryBlock();
  b.setInsertionPointToStart(entry_block);
  namespace ma = mlir::arith;
  auto const_idx = [&](int64_t v) {
    return ma::ConstantIndexOp::create(b, v).getResult();
  };

  // linear = bid.x * threads_per_block + tid.x. We use only the x dim;
  // PlanBitonicSort issues a 1D launch by feeding a CeilOfRatio-shaped
  // standard iteration shape into CalculateLaunchDimensions.
  mlir::Value tid = mlir::gpu::ThreadIdOp::create(b, mlir::gpu::Dimension::x);
  mlir::Value bid = mlir::gpu::BlockIdOp::create(b, mlir::gpu::Dimension::x);
  mlir::Value linear = ma::AddIOp::create(
      b, ma::MulIOp::create(b, bid,
                            const_idx(desc.launch_dimensions
                                          .num_threads_per_block())),
      tid);

  // Decompose linear into iteration_shape (= keys_shape with sort_dim
  // replaced by iteration_shape_sort_dim) using minor-to-major order so
  // the innermost (most-frequent) coordinate corresponds to the physical
  // layout's stride-1 dimension.
  llvm::SmallVector<mlir::Value, 4> indices(rank);
  mlir::Value remaining = linear;
  for (int64_t d : keys_shape.layout().minor_to_major()) {
    int64_t size = (d == sort_dim) ? iteration_shape_sort_dim
                                    : keys_shape.dimensions(d);
    mlir::Value size_c = const_idx(size);
    indices[d] = ma::RemUIOp::create(b, remaining, size_c);
    remaining = ma::DivUIOp::create(b, remaining, size_c);
  }

  // Apply the xor-block derivation to convert iter_sort_idx (which ranges
  // [0, iteration_bound)) into the actual current_keys_index inside the
  // keys array. Three cases mirror EmitCompareLoopBody:
  mlir::Value iter_sort_idx = indices[sort_dim];
  mlir::Value current;
  mlir::Value block_size_c = const_idx(block_size);
  if (block_size == 1) {
    current = ma::MulIOp::create(b, iter_sort_idx, const_idx(2));
  } else if (block_size * 2 < dim_to_sort_bound) {
    mlir::Value blk = ma::DivUIOp::create(b, iter_sort_idx, block_size_c);
    mlir::Value idx_in_blk =
        ma::RemUIOp::create(b, iter_sort_idx, block_size_c);
    mlir::Value first_in_block =
        ma::MulIOp::create(b, blk, const_idx(2 * block_size));
    current = ma::AddIOp::create(b, first_in_block, idx_in_blk);
  } else {
    // Sentinel: a thread in the "right" block of the pair must skip; force
    // its current index to dim_to_sort_bound^xor_mask so the compare index
    // falls at dim_to_sort_bound and the bounds check below rejects it.
    mlir::Value sentinel = const_idx(dim_to_sort_bound ^ xor_mask);
    mlir::Value is_left = ma::CmpIOp::create(b, ma::CmpIPredicate::ult,
                                             iter_sort_idx, block_size_c);
    current = ma::SelectOp::create(b, is_left, iter_sort_idx, sentinel);
  }

  // compare = current XOR xor_mask. Bounds check both indices against the
  // *actual* sort dimension length (not iteration_bound) to skip phantom
  // pairs that the bitonic algorithm would otherwise touch on the next-
  // power-of-two padding.
  mlir::Value compare =
      ma::XOrIOp::create(b, current, const_idx(xor_mask));
  mlir::Value bound = const_idx(dim_to_sort_bound);
  mlir::Value in_bounds = ma::AndIOp::create(
      b,
      ma::CmpIOp::create(b, ma::CmpIPredicate::ult, current, bound),
      ma::CmpIOp::create(b, ma::CmpIPredicate::ult, compare, bound));

  llvm::SmallVector<mlir::Value, 4> current_indices(indices.begin(),
                                                     indices.end());
  llvm::SmallVector<mlir::Value, 4> compare_indices(indices.begin(),
                                                     indices.end());
  current_indices[sort_dim] = current;
  compare_indices[sort_dim] = compare;

  llvm::SmallVector<mlir::Type, 4> tensor_types;
  for (auto arg : entry_block->getArguments()) {
    tensor_types.push_back(arg.getType());
  }
  llvm::SmallVector<mlir::Value, 4> entry_tensors(
      entry_block->getArguments().begin(), entry_block->getArguments().end());

  // scf.if in_bounds: compare-and-conditionally-swap; else: passthrough.
  auto if_op = mlir::scf::IfOp::create(b, tensor_types, in_bounds,
                                       /*withElseRegion=*/true);
  {
    mlir::OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(if_op.thenBlock());
    llvm::SmallVector<mlir::Value, 4> compare_args;
    llvm::SmallVector<mlir::Value, 4> compare_vals;
    llvm::SmallVector<mlir::Value, 4> current_vals;
    compare_args.reserve(2 * operand_count);
    compare_vals.reserve(operand_count);
    current_vals.reserve(operand_count);
    for (int64_t i = 0; i < operand_count; ++i) {
      mlir::Value tensor = entry_tensors[i];
      mlir::Value v_compare =
          mlir::tensor::ExtractOp::create(b, tensor, compare_indices);
      mlir::Value v_current =
          mlir::tensor::ExtractOp::create(b, tensor, current_indices);
      compare_vals.push_back(v_compare);
      current_vals.push_back(v_current);
      compare_args.push_back(v_compare);
      compare_args.push_back(v_current);
    }
    mlir::Value cmp_result =
        mlir::func::CallOp::create(b, comparator_func, compare_args)
            .getResult(0);
    // PRED lowers to i8 in MLIR; truncate to i1 for scf.if.
    mlir::Value cmp_i1 = ma::CmpIOp::create(
        b, ma::CmpIPredicate::ne, cmp_result,
        ma::ConstantOp::create(b, cmp_result.getType(),
                               b.getIntegerAttr(cmp_result.getType(), 0))
            .getResult());
    auto swap_if = mlir::scf::IfOp::create(b, tensor_types, cmp_i1,
                                           /*withElseRegion=*/true);
    {
      mlir::OpBuilder::InsertionGuard sg(b);
      b.setInsertionPointToStart(swap_if.thenBlock());
      llvm::SmallVector<mlir::Value, 4> swapped;
      swapped.reserve(operand_count);
      for (int64_t i = 0; i < operand_count; ++i) {
        mlir::Value tensor = entry_tensors[i];
        // current position gets compare_value; compare position gets
        // current_value. This matches EmitCompareLoopBody's swap.
        tensor = mlir::tensor::InsertOp::create(b, compare_vals[i], tensor,
                                                current_indices);
        tensor = mlir::tensor::InsertOp::create(b, current_vals[i], tensor,
                                                compare_indices);
        swapped.push_back(tensor);
      }
      mlir::scf::YieldOp::create(b, swapped);
    }
    {
      mlir::OpBuilder::InsertionGuard sg(b);
      b.setInsertionPointToStart(swap_if.elseBlock());
      mlir::scf::YieldOp::create(b, entry_tensors);
    }
    mlir::scf::YieldOp::create(b, swap_if.getResults());
  }
  {
    mlir::OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(if_op.elseBlock());
    mlir::scf::YieldOp::create(b, entry_tensors);
  }

  b.setInsertionPointToEnd(entry_block);
  mlir::func::ReturnOp::create(b, if_op.getResults());

  return module;
}

}  // namespace xla::metal
