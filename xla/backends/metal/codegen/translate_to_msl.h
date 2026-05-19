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

#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/backends/metal/codegen/msl_kernel_source.h"

namespace xla {
namespace metal {

// Cross-translation state held by MetalCompiler when it concatenates one MSL
// source per fusion into a single program. The translator consults this to
// dedup file-scope definitions (e.g. __xla_log1p_<ty> helpers) that would
// otherwise be defined more than once in the combined source.
struct MslTranslationState {
  absl::flat_hash_set<std::string> log1p_emitted;
};

// Emits MSL source for the entry-point func.func in `module` (the one
// marked with the xla.entry unit attribute). Kernel argument types come
// from the tensor types; `[[buffer(N)]]` indices come from each argument's
// xla.slice_index attribute.
//
// If `state` is non-null, file-scope helpers (e.g. log1p) tracked there are
// emitted at most once across calls sharing the same state.
absl::StatusOr<MslKernelSource> TranslateToMSL(
    mlir::ModuleOp module, MslTranslationState* state = nullptr);

}  // namespace metal
}  // namespace xla

#endif  // XLA_BACKENDS_METAL_CODEGEN_TRANSLATE_TO_MSL_H_
