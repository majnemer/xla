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

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/backends/metal/codegen/msl_kernel_source.h"
#include "xla/backends/metal/codegen/translate_to_msl.h"

namespace xla {
namespace metal {

// Runs the Metal-specific MLIR pass pipeline on `module`, then translates
// the entry-point func.func to MSL via TranslateToMSL. The input module is
// mutated in place by the pipeline.
absl::StatusOr<MslKernelSource> EmitMslKernel(mlir::ModuleOp module);

}  // namespace metal
}  // namespace xla

#endif  // XLA_BACKENDS_METAL_CODEGEN_MSL_KERNEL_EMITTER_H_
