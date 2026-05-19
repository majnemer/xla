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
#ifndef XLA_BACKENDS_METAL_CODEGEN_TRANSFORMS_PASSES_H_
#define XLA_BACKENDS_METAL_CODEGEN_TRANSFORMS_PASSES_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace xla {
namespace metal {

#define GEN_PASS_DECL
#include "xla/backends/metal/codegen/transforms/passes.h.inc"

std::unique_ptr<mlir::Pass> CreateLowerFloatStoragePass();
std::unique_ptr<mlir::Pass> CreateLowerSubByteStoragePass();
std::unique_ptr<mlir::Pass> CreateExpandFloatOpsPass();
std::unique_ptr<mlir::Pass> CreateConvertComplexToArithMathPass();

#define GEN_PASS_REGISTRATION
#include "xla/backends/metal/codegen/transforms/passes.h.inc"

}  // namespace metal
}  // namespace xla

#endif  // XLA_BACKENDS_METAL_CODEGEN_TRANSFORMS_PASSES_H_
