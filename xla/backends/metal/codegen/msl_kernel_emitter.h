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

#ifndef XLA_BACKENDS_METAL_CODEGEN_MSL_KERNEL_EMITTER_H_
#define XLA_BACKENDS_METAL_CODEGEN_MSL_KERNEL_EMITTER_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/backends/metal/codegen/msl_kernel_source.h"
#include "xla/backends/metal/codegen/translate_to_msl.h"
#include "xla/stream_executor/device_description.h"

namespace xla {

class HloModule;
class NameUniquer;

namespace metal {

// Runs the full Metal MLIR lowering pipeline on `module`: GPU loop transforms,
// arith/affine simplification, complex lowering, narrow-float / sub-byte
// storage widening. Designed to be called once per kernel — not safe to run
// twice on the same module (the storage-lowering passes may not be no-ops on
// already-lowered IR). `hlo_module` is used only for IR-dump routing via
// `EnableIRPrintingIfRequested`; `dump_category` is the category string
// (e.g. "mlir-fusion", "mlir-sort").
absl::Status RunMetalLoweringPipeline(
    mlir::ModuleOp module, const stream_executor::DeviceDescription& device,
    int max_unroll_factor, const HloModule& hlo_module,
    absl::string_view entry_function_name, absl::string_view dump_category);

// Runs the Metal-specific MLIR pass pipeline on `module`, then translates
// the entry-point func.func to MSL via TranslateToMSL. The input module is
// mutated in place by the pipeline.
absl::StatusOr<MslKernelSource> EmitMslKernel(mlir::ModuleOp module);

// Same as above, but uses `function_name_uniquer` for emitted MSL function
// names. Share a uniquer across calls when concatenating the returned sources.
absl::StatusOr<MslKernelSource> EmitMslKernel(
    mlir::ModuleOp module, NameUniquer* function_name_uniquer);

// Same as above, but enables CUDA-style MLIR pass dumps when requested by
// XLA's dump flags.
absl::StatusOr<MslKernelSource> EmitMslKernel(
    mlir::ModuleOp module, NameUniquer* function_name_uniquer,
    const HloModule& hlo_module, absl::string_view entry_function_name);

}  // namespace metal
}  // namespace xla

#endif  // XLA_BACKENDS_METAL_CODEGEN_MSL_KERNEL_EMITTER_H_
