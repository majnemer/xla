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

#ifndef XLA_BACKENDS_METAL_CODEGEN_DYNAMIC_SHAPE_EMITTER_H_
#define XLA_BACKENDS_METAL_CODEGEN_DYNAMIC_SHAPE_EMITTER_H_

#include <string>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"

namespace xla::metal {

// Description of one PadToStatic / SliceToDynamic kernel invocation. These
// custom calls are DynamicPadder's runtime residue: a "dynamic" buffer is the
// static-bound array with one little-endian s32 per dimension appended at
// byte offset ByteSizeOf(static shape) (unaligned in general), holding its
// values dense in the dynamic shape. PadToStatic spreads compact -> padded
// and exports the dims as scalar outputs; SliceToDynamic is the inverse.
// MLIR port of the LLVM GPU backend's EmitPadToStaticLLVMIR /
// EmitSliceToDynamicLLVMIR.
struct DynamicShapeKernelDescription {
  const HloCustomCallInstruction* custom_call = nullptr;
  // The data array's static-bound shape (operand 0 for PadToStatic, the
  // result for SliceToDynamic). Drives the launch and the index remap.
  Shape data_static_shape;
  // Operands, then output leaves (KernelArguments::Create order), then one
  // appended u8 view of the whole dynamic buffer's slice through which the
  // kernel reads (PadToStatic) or writes (SliceToDynamic) the dim metadata
  // byte-wise.
  emitters::KernelArguments kernel_args;
  gpu::LaunchDimensions launch_dimensions;
  // Globally unique MSL entry-function name.
  std::string entry_name;
};

// Validates the custom call's shape contract, assembles the kernel arguments
// (including the metadata byte view), and picks launch dimensions covering
// the static bound (>= 1 thread so the metadata is written even for
// zero-element data).
absl::StatusOr<DynamicShapeKernelDescription> PlanDynamicShapeKernel(
    const HloCustomCallInstruction* custom_call,
    const BufferAssignment& buffer_assignment,
    const se::DeviceDescription& device,
    const emitters::KernelArguments::BufferAlignment& buffer_alignment,
    std::string entry_name);

// Recomputes `desc.launch_dimensions` for a (possibly capped) device, for
// PSO-retry like the fusion path.
void RecomputeDynamicShapeLaunch(DynamicShapeKernelDescription& desc,
                                 const se::DeviceDescription& device);

// Build the MLIR module for one kernel: a `func.func @<entry_name>` with the
// `xla.entry` attribute, fed through Metal's standard MSL lowering +
// translation pipeline. Per-thread, with gid covering the static bound:
//   dims    = the rank s32 dim sizes (PadToStatic: assembled from the source
//             tail bytes; SliceToDynamic: read from the scalar operands)
//   gid == 0 writes the dims out (scalar outputs / dest tail bytes)
//   gid < prod(dims) copies one element between compact slot `gid` and the
//             padded position at delinearize(gid, dims)
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> EmitPadToStaticMLIR(
    mlir::MLIRContext* context, const DynamicShapeKernelDescription& desc);

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> EmitSliceToDynamicMLIR(
    mlir::MLIRContext* context, const DynamicShapeKernelDescription& desc);

}  // namespace xla::metal

#endif  // XLA_BACKENDS_METAL_CODEGEN_DYNAMIC_SHAPE_EMITTER_H_
