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

#include "xla/backends/metal/codegen/dynamic_shape_emitter.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
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
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/emitters/type_util.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::metal {
namespace {

namespace ma = mlir::arith;

constexpr absl::string_view kXlaEntryAttr = "xla.entry";
constexpr absl::string_view kXlaSliceIndexAttr = "xla.slice_index";
constexpr absl::string_view kXlaInvariantAttr = "xla.invariant";

bool IsPadToStatic(const HloCustomCallInstruction* custom_call) {
  return custom_call->custom_call_target() == "PadToStatic";
}

mlir::Value ConstIdx(mlir::ImplicitLocOpBuilder& b, int64_t value) {
  return ma::ConstantIndexOp::create(b, value).getResult();
}

mlir::Value ConstI32(mlir::ImplicitLocOpBuilder& b, int32_t value) {
  return ma::ConstantOp::create(b, b.getI32Type(), b.getI32IntegerAttr(value))
      .getResult();
}

// linear thread id over the full grid; launch dims may carry a y block
// dimension when num_blocks exceeds the device's x grid limit.
mlir::Value EmitGid(mlir::ImplicitLocOpBuilder& b,
                    const gpu::LaunchDimensions& launch) {
  mlir::Value tid = mlir::gpu::ThreadIdOp::create(b, mlir::gpu::Dimension::x);
  mlir::Value bid_x = mlir::gpu::BlockIdOp::create(b, mlir::gpu::Dimension::x);
  mlir::Value bid_y = mlir::gpu::BlockIdOp::create(b, mlir::gpu::Dimension::y);
  mlir::Value block_linear = ma::AddIOp::create(
      b, ma::MulIOp::create(b, bid_y, ConstIdx(b, launch.block_counts().x)),
      bid_x);
  return ma::AddIOp::create(
      b,
      ma::MulIOp::create(b, block_linear,
                         ConstIdx(b, launch.num_threads_per_block())),
      tid);
}

// Both delinearizations walk minor-to-major of `layout_ref`'s layout,
// mirroring llvm_ir::IrArray::Index::Delinearize so the compact <-> padded
// mapping matches the LLVM GPU kernels bit for bit.
llvm::SmallVector<mlir::Value> DelinearizeStatic(mlir::ImplicitLocOpBuilder& b,
                                                 mlir::Value linear,
                                                 const Shape& layout_ref) {
  const int64_t rank = layout_ref.dimensions().size();
  llvm::SmallVector<mlir::Value> coords(rank);
  int64_t divisor = 1;
  for (int64_t i = 0; i < rank; ++i) {
    const int64_t dim = layout_ref.layout().minor_to_major(i);
    mlir::Value quot = ma::DivUIOp::create(b, linear, ConstIdx(b, divisor));
    coords[dim] = i == rank - 1
                      ? quot
                      : ma::RemUIOp::create(
                            b, quot, ConstIdx(b, layout_ref.dimensions(dim)))
                            .getResult();
    divisor *= layout_ref.dimensions(dim);
  }
  return coords;
}

llvm::SmallVector<mlir::Value> DelinearizeDynamic(
    mlir::ImplicitLocOpBuilder& b, mlir::Value linear,
    absl::Span<const mlir::Value> dyn_sizes, const Shape& layout_ref) {
  const int64_t rank = layout_ref.dimensions().size();
  llvm::SmallVector<mlir::Value> coords(rank);
  mlir::Value divisor = ConstIdx(b, 1);
  for (int64_t i = 0; i < rank; ++i) {
    const int64_t dim = layout_ref.layout().minor_to_major(i);
    mlir::Value quot = ma::DivUIOp::create(b, linear, divisor);
    if (i == rank - 1) {
      coords[dim] = quot;
    } else {
      coords[dim] = ma::RemUIOp::create(b, quot, dyn_sizes[dim]);
      divisor = ma::MulIOp::create(b, divisor, dyn_sizes[dim]);
    }
  }
  return coords;
}

// The metadata sits at ByteSizeOf(static shape), unaligned in general (e.g.
// pred[<=5] puts it at byte 5), so it moves through the u8 view one byte at
// a time, little-endian.
mlir::Value AssembleS32(mlir::ImplicitLocOpBuilder& b, mlir::Value bytes,
                        int64_t offset) {
  mlir::Value value = ConstI32(b, 0);
  for (int64_t j = 0; j < 4; ++j) {
    mlir::Value byte = mlir::tensor::ExtractOp::create(
        b, bytes, mlir::ValueRange{ConstIdx(b, offset + j)});
    mlir::Value wide = ma::ExtUIOp::create(b, b.getI32Type(), byte);
    if (j > 0) {
      wide = ma::ShLIOp::create(b, wide, ConstI32(b, 8 * j));
    }
    value = ma::OrIOp::create(b, value, wide);
  }
  return value;
}

mlir::Value ScatterS32(mlir::ImplicitLocOpBuilder& b, mlir::Value bytes,
                       mlir::Value value, int64_t offset) {
  for (int64_t j = 0; j < 4; ++j) {
    mlir::Value piece =
        j == 0 ? value : ma::ShRUIOp::create(b, value, ConstI32(b, 8 * j));
    mlir::Value byte = ma::TruncIOp::create(b, b.getI8Type(), piece);
    bytes = mlir::tensor::InsertOp::create(
        b, byte, bytes, mlir::ValueRange{ConstIdx(b, offset + j)});
  }
  return bytes;
}

mlir::func::FuncOp CreateEntryFunc(mlir::ImplicitLocOpBuilder& b,
                                   mlir::MLIRContext* context,
                                   mlir::ModuleOp module,
                                   const DynamicShapeKernelDescription& desc) {
  const auto& args = desc.kernel_args.args();
  llvm::SmallVector<mlir::Type> param_types;
  llvm::SmallVector<mlir::Attribute> arg_attrs;
  llvm::SmallVector<mlir::Type> result_types;
  param_types.reserve(args.size());
  arg_attrs.reserve(args.size());
  for (const auto& arg : args) {
    mlir::Type type = emitters::TensorShapeToMlirType(arg.shape(), b);
    param_types.push_back(type);
    if (arg.written()) {
      result_types.push_back(type);
    }
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

  b.setInsertionPointToStart(module.getBody());
  auto entry_func = mlir::func::FuncOp::create(
      b, desc.entry_name,
      mlir::FunctionType::get(context, param_types, result_types),
      /*sym_visibility=*/mlir::StringAttr{},
      mlir::ArrayAttr::get(context, arg_attrs),
      /*res_attrs=*/mlir::ArrayAttr{});
  entry_func->setAttr(kXlaEntryAttr, mlir::UnitAttr::get(context));
  return entry_func;
}

// Args: [src dynamic array, dest static array, s32 dim outs..., src u8 view].
// Results: [dest', dim outs'...].
absl::Status EmitPadToStaticBody(mlir::ImplicitLocOpBuilder& b,
                                 mlir::func::FuncOp entry_func,
                                 const DynamicShapeKernelDescription& desc) {
  const Shape& static_shape = desc.data_static_shape;
  const int64_t rank = static_shape.dimensions().size();
  const int64_t metadata_offset = ShapeUtil::ByteSizeOf(static_shape);

  mlir::Block* block = entry_func.addEntryBlock();
  b.setInsertionPointToStart(block);
  mlir::Value src = block->getArgument(0);
  mlir::Value dest = block->getArgument(1);
  llvm::SmallVector<mlir::Value> dim_outs;
  for (int64_t k = 0; k < rank; ++k) {
    dim_outs.push_back(block->getArgument(2 + k));
  }
  mlir::Value src_bytes = block->getArgument(2 + rank);

  mlir::Value gid = EmitGid(b, desc.launch_dimensions);

  llvm::SmallVector<mlir::Value> dims_i32;
  llvm::SmallVector<mlir::Value> dims_idx;
  for (int64_t k = 0; k < rank; ++k) {
    dims_i32.push_back(AssembleS32(b, src_bytes, metadata_offset + 4 * k));
    dims_idx.push_back(
        ma::IndexCastOp::create(b, b.getIndexType(), dims_i32.back()));
  }

  mlir::Value is_thread0 =
      ma::CmpIOp::create(b, ma::CmpIPredicate::eq, gid, ConstIdx(b, 0));
  llvm::SmallVector<mlir::Type> dim_out_types;
  for (mlir::Value dim_out : dim_outs) {
    dim_out_types.push_back(dim_out.getType());
  }
  auto dims_if = mlir::scf::IfOp::create(b, dim_out_types, is_thread0,
                                         /*withElseRegion=*/true);
  {
    mlir::OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(dims_if.thenBlock());
    llvm::SmallVector<mlir::Value> updated;
    for (int64_t k = 0; k < rank; ++k) {
      updated.push_back(mlir::tensor::InsertOp::create(
          b, dims_i32[k], dim_outs[k], mlir::ValueRange{}));
    }
    mlir::scf::YieldOp::create(b, updated);
    b.setInsertionPointToStart(dims_if.elseBlock());
    mlir::scf::YieldOp::create(b, dim_outs);
  }

  mlir::Value dyn_total = ConstIdx(b, 1);
  for (mlir::Value dim : dims_idx) {
    dyn_total = ma::MulIOp::create(b, dyn_total, dim);
  }
  mlir::Value in_copy =
      ma::CmpIOp::create(b, ma::CmpIPredicate::ult, gid, dyn_total);
  auto copy_if = mlir::scf::IfOp::create(b, {dest.getType()}, in_copy,
                                         /*withElseRegion=*/true);
  {
    mlir::OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(copy_if.thenBlock());
    llvm::SmallVector<mlir::Value> src_coords =
        DelinearizeStatic(b, gid, static_shape);
    llvm::SmallVector<mlir::Value> dst_coords =
        DelinearizeDynamic(b, gid, dims_idx, static_shape);
    mlir::Value value = mlir::tensor::ExtractOp::create(b, src, src_coords);
    mlir::scf::YieldOp::create(
        b, mlir::Value(
               mlir::tensor::InsertOp::create(b, value, dest, dst_coords)));
    b.setInsertionPointToStart(copy_if.elseBlock());
    mlir::scf::YieldOp::create(b, dest);
  }

  llvm::SmallVector<mlir::Value> results{copy_if.getResult(0)};
  results.append(dims_if.getResults().begin(), dims_if.getResults().end());
  mlir::func::ReturnOp::create(b, results);
  return absl::OkStatus();
}

// Args: [src static array, s32 dim ins..., dest dynamic array, dest u8 view].
// Results: [dest', dest u8 view'].
absl::Status EmitSliceToDynamicBody(mlir::ImplicitLocOpBuilder& b,
                                    mlir::func::FuncOp entry_func,
                                    const DynamicShapeKernelDescription& desc) {
  const Shape& static_shape = desc.data_static_shape;
  const Shape& operand_shape = desc.custom_call->operand(0)->shape();
  const int64_t rank = static_shape.dimensions().size();
  const int64_t metadata_offset = ShapeUtil::ByteSizeOf(static_shape);

  mlir::Block* block = entry_func.addEntryBlock();
  b.setInsertionPointToStart(block);
  mlir::Value src = block->getArgument(0);
  llvm::SmallVector<mlir::Value> dim_ins;
  for (int64_t k = 0; k < rank; ++k) {
    dim_ins.push_back(block->getArgument(1 + k));
  }
  mlir::Value dest = block->getArgument(1 + rank);
  mlir::Value dest_bytes = block->getArgument(2 + rank);

  mlir::Value gid = EmitGid(b, desc.launch_dimensions);

  llvm::SmallVector<mlir::Value> dims_i32;
  llvm::SmallVector<mlir::Value> dims_idx;
  for (int64_t k = 0; k < rank; ++k) {
    dims_i32.push_back(mlir::tensor::ExtractOp::create(b, dim_ins[k],
                                                       mlir::ValueRange{}));
    dims_idx.push_back(
        ma::IndexCastOp::create(b, b.getIndexType(), dims_i32.back()));
  }

  mlir::Value is_thread0 =
      ma::CmpIOp::create(b, ma::CmpIPredicate::eq, gid, ConstIdx(b, 0));
  auto bytes_if = mlir::scf::IfOp::create(b, {dest_bytes.getType()},
                                          is_thread0,
                                          /*withElseRegion=*/true);
  {
    mlir::OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(bytes_if.thenBlock());
    mlir::Value updated = dest_bytes;
    for (int64_t k = 0; k < rank; ++k) {
      updated = ScatterS32(b, updated, dims_i32[k], metadata_offset + 4 * k);
    }
    mlir::scf::YieldOp::create(b, updated);
    b.setInsertionPointToStart(bytes_if.elseBlock());
    mlir::scf::YieldOp::create(b, dest_bytes);
  }

  mlir::Value dyn_total = ConstIdx(b, 1);
  for (mlir::Value dim : dims_idx) {
    dyn_total = ma::MulIOp::create(b, dyn_total, dim);
  }
  mlir::Value in_copy =
      ma::CmpIOp::create(b, ma::CmpIPredicate::ult, gid, dyn_total);
  auto copy_if = mlir::scf::IfOp::create(b, {dest.getType()}, in_copy,
                                         /*withElseRegion=*/true);
  {
    mlir::OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(copy_if.thenBlock());
    llvm::SmallVector<mlir::Value> src_coords =
        DelinearizeDynamic(b, gid, dims_idx, operand_shape);
    llvm::SmallVector<mlir::Value> dst_coords =
        DelinearizeStatic(b, gid, static_shape);
    mlir::Value value = mlir::tensor::ExtractOp::create(b, src, src_coords);
    mlir::scf::YieldOp::create(
        b, mlir::Value(
               mlir::tensor::InsertOp::create(b, value, dest, dst_coords)));
    b.setInsertionPointToStart(copy_if.elseBlock());
    mlir::scf::YieldOp::create(b, dest);
  }

  mlir::func::ReturnOp::create(
      b, mlir::ValueRange{copy_if.getResult(0), bytes_if.getResult(0)});
  return absl::OkStatus();
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> EmitDynamicShapeModule(
    mlir::MLIRContext* context, const DynamicShapeKernelDescription& desc) {
  mlir::OpBuilder builder(context);
  auto loc = mlir::NameLoc::get(builder.getStringAttr(desc.entry_name));
  mlir::OwningOpRef<mlir::ModuleOp> module = llvm_ir::CreateMlirModuleOp(loc);
  mlir::ImplicitLocOpBuilder b(loc, *module);
  mlir::func::FuncOp entry_func = CreateEntryFunc(b, context, *module, desc);
  if (IsPadToStatic(desc.custom_call)) {
    TF_RETURN_IF_ERROR(EmitPadToStaticBody(b, entry_func, desc));
  } else {
    TF_RETURN_IF_ERROR(EmitSliceToDynamicBody(b, entry_func, desc));
  }
  return module;
}

}  // namespace

absl::StatusOr<DynamicShapeKernelDescription> PlanDynamicShapeKernel(
    const HloCustomCallInstruction* custom_call,
    const BufferAssignment& buffer_assignment,
    const se::DeviceDescription& device,
    const emitters::KernelArguments::BufferAlignment& buffer_alignment,
    std::string entry_name) {
  const bool pad_to_static = IsPadToStatic(custom_call);
  TF_RET_CHECK(pad_to_static ||
               custom_call->custom_call_target() == "SliceToDynamic");

  const Shape scalar_s32 = ShapeUtil::MakeScalarShape(S32);
  Shape data_static_shape;
  if (pad_to_static) {
    const Shape& in = custom_call->operand(0)->shape();
    TF_RET_CHECK(custom_call->operand_count() == 1 && in.IsArray());
    data_static_shape = ShapeUtil::MakeStaticShape(in);
    const int64_t rank = data_static_shape.dimensions().size();
    TF_RET_CHECK(rank >= 1);
    TF_RET_CHECK(custom_call->shape().IsTuple() &&
                 custom_call->shape().tuple_shapes().size() == 1 + rank);
    for (int64_t k = 0; k < rank; ++k) {
      TF_RET_CHECK(
          Shape::Equal()(custom_call->shape().tuple_shapes(1 + k), scalar_s32));
    }
  } else {
    const Shape& out = custom_call->shape();
    TF_RET_CHECK(out.IsArray());
    data_static_shape = ShapeUtil::MakeStaticShape(out);
    const int64_t rank = data_static_shape.dimensions().size();
    TF_RET_CHECK(rank >= 1);
    TF_RET_CHECK(custom_call->operand_count() == 1 + rank);
    for (int64_t k = 0; k < rank; ++k) {
      TF_RET_CHECK(
          Shape::Equal()(custom_call->operand(1 + k)->shape(), scalar_s32));
    }
  }
  const int64_t rank = data_static_shape.dimensions().size();

  TF_ASSIGN_OR_RETURN(emitters::KernelArguments base_args,
                      emitters::KernelArguments::Create(
                          buffer_assignment, buffer_alignment, custom_call));
  // Positional [[buffer(i)]] binding: the MLIR func mirrors this vector
  // one-to-one, so a deduplicated (shorter) list would shift every index.
  TF_RET_CHECK(base_args.args().size() == 2 + rank)
      << custom_call->ToString();
  std::vector<emitters::KernelArgument> args(base_args.args());

  const int64_t operand_count = custom_call->operand_count();
  for (int64_t i = 0; i < static_cast<int64_t>(args.size()); ++i) {
    args[i].set_written(i >= operand_count);
  }
  // The kernel reads the data array and writes the other in the same launch;
  // they must not share a buffer.
  const BufferAllocation::Slice data_in_slice = args.front().slice();
  const BufferAllocation::Slice data_out_slice =
      pad_to_static ? args[1].slice() : args.back().slice();
  TF_RET_CHECK(data_in_slice != data_out_slice) << custom_call->ToString();

  // u8 view of the dynamic buffer for the byte-wise metadata access; binds
  // the same MTLBuffer at a second argument index.
  const emitters::KernelArgument& dynamic_arg =
      pad_to_static ? args.front() : args.back();
  emitters::KernelArgument byte_view(
      ShapeUtil::MakeShape(U8, {dynamic_arg.slice().size()}),
      dynamic_arg.slice());
  byte_view.set_written(!pad_to_static);
  byte_view.set_alignment(1);
  byte_view.set_slice_index(args.size());
  args.push_back(std::move(byte_view));

  DynamicShapeKernelDescription desc{
      custom_call,
      std::move(data_static_shape),
      emitters::KernelArguments(std::move(args)),
      gpu::LaunchDimensions(),
      std::move(entry_name),
  };
  RecomputeDynamicShapeLaunch(desc, device);
  return desc;
}

void RecomputeDynamicShapeLaunch(DynamicShapeKernelDescription& desc,
                                 const se::DeviceDescription& device) {
  desc.launch_dimensions =
      gpu::CalculateLaunchDimensions(desc.data_static_shape, device);
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> EmitPadToStaticMLIR(
    mlir::MLIRContext* context, const DynamicShapeKernelDescription& desc) {
  TF_RET_CHECK(IsPadToStatic(desc.custom_call));
  return EmitDynamicShapeModule(context, desc);
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> EmitSliceToDynamicMLIR(
    mlir::MLIRContext* context, const DynamicShapeKernelDescription& desc) {
  TF_RET_CHECK(!IsPadToStatic(desc.custom_call));
  return EmitDynamicShapeModule(context, desc);
}

}  // namespace xla::metal
