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

#ifndef XLA_BACKENDS_METAL_CODEGEN_TRANSLATE_TO_MSL_H_
#define XLA_BACKENDS_METAL_CODEGEN_TRANSLATE_TO_MSL_H_

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/backends/metal/codegen/msl_kernel_source.h"

namespace xla {

class NameUniquer;

namespace metal {

// Emits MSL source for the entry-point func.func in `module` (the one
// marked with the xla.entry unit attribute). Kernel argument types come
// from the tensor types; `[[buffer(N)]]` indices come from each argument's
// xla.slice_index attribute.
absl::StatusOr<MslKernelSource> TranslateToMSL(mlir::ModuleOp module);

// Emits MSL source, using `function_name_uniquer` to make emitted MSL function
// names unique. Callers that concatenate multiple translated modules into one
// MSL translation unit should share a single uniquer across those calls.
absl::StatusOr<MslKernelSource> TranslateToMSL(
    mlir::ModuleOp module, NameUniquer* function_name_uniquer);

}  // namespace metal
}  // namespace xla

#endif  // XLA_BACKENDS_METAL_CODEGEN_TRANSLATE_TO_MSL_H_
